import torch
import torch.nn as nn
import torch.nn.functional as F

from argparse import Namespace
from src.base.model import BaseModel

def data_transformation_4_xformer(history_data: torch.Tensor, future_data: torch.Tensor, start_token_len: int):
    """Transfer the data into the XFormer format.

    Args:
        history_data (torch.Tensor): history data with shape: [B, L1, N, C].
        future_data (torch.Tensor): future data with shape: [B, L2, N, C].
                                    L1 and L2 are input sequence length and output sequence length, respectively.
        start_token_length (int): length of the decoder start token. Ref: Informer paper.

    Returns:
        torch.Tensor: x_enc, input data of encoder (without the time features). Shape: [B, L1, N]
        torch.Tensor: x_mark_enc, time features input of encoder w.r.t. x_enc. Shape: [B, L1, C-1]
        torch.Tensor: x_dec, input data of decoder. Shape: [B, start_token_length + L2, N]
        torch.Tensor: x_mark_dec, time features input to decoder w.r.t. x_dec. Shape: [B, start_token_length + L2, C-1]
    """

    # get the x_enc
    x_enc = history_data[..., 0]            # B, L1, N
    # get the corresponding x_mark_enc
    # following previous works, we re-scale the time features from [0, 1) to to [-0.5, 0.5).
    x_mark_enc = history_data[:, :, 0, 1:] - 0.5    # B, L1, C-1

    # get the x_dec
    if start_token_len == 0:
        x_dec = torch.zeros_like(future_data[..., 0])     # B, L2, N
        # get the corresponding x_mark_dec
        x_mark_dec = future_data[..., :, 0, 1:] - 0.5                 # B, L2, C-1
        return x_enc, x_mark_enc, x_dec, x_mark_dec
    else:
        x_dec_token = x_enc[:, -start_token_len:, :]            # B, start_token_length, N
        x_dec_zeros = torch.zeros_like(future_data[..., 0])     # B, L2, N
        x_dec = torch.cat([x_dec_token, x_dec_zeros], dim=1)    # B, (start_token_length+L2), N
        # get the corresponding x_mark_dec
        x_mark_dec_token = x_mark_enc[:, -start_token_len:, :]            # B, start_token_length, C-1
        x_mark_dec_future = future_data[..., :, 0, 1:] - 0.5          # B, L2, C-1
        x_mark_dec = torch.cat([x_mark_dec_token, x_mark_dec_future], dim=1)    # B, (start_token_length+L2), C-1

    return x_enc.float(), x_mark_enc.float(), x_dec.float(), x_mark_dec.float()


class STAR(nn.Module):
    def __init__(self, d_series, d_core):
        super(STAR, self).__init__()
        """
        STar Aggregate-Redistribute Module
        """

        self.gen1 = nn.Linear(d_series, d_series)
        self.gen2 = nn.Linear(d_series, d_core)
        self.gen3 = nn.Linear(d_series + d_core, d_series)
        self.gen4 = nn.Linear(d_series, d_series)

    def forward(self, input, *args, **kwargs):
        batch_size, channels, d_series = input.shape

        # set FFN
        combined_mean = F.gelu(self.gen1(input))
        combined_mean = self.gen2(combined_mean)

        # stochastic pooling
        if self.training:
            ratio = F.softmax(combined_mean, dim=1)
            ratio = ratio.permute(0, 2, 1)
            ratio = ratio.reshape(-1, channels)
            indices = torch.multinomial(ratio, 1)
            indices = indices.view(batch_size, -1, 1).permute(0, 2, 1)
            combined_mean = torch.gather(combined_mean, 1, indices)
            combined_mean = combined_mean.repeat(1, channels, 1)
        else:
            weight = F.softmax(combined_mean, dim=1)
            combined_mean = torch.sum(combined_mean * weight, dim=1, keepdim=True).repeat(1, channels, 1)

        # mlp fusion
        combined_mean_cat = torch.cat([input, combined_mean], -1)
        combined_mean_cat = F.gelu(self.gen3(combined_mean_cat))
        combined_mean_cat = self.gen4(combined_mean_cat)
        output = combined_mean_cat

        return output, None


class SOFTS(BaseModel):
    '''
    Paper: SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion
    Official Code: https://github.com/Secilia-Cxy/SOFTS
    Link: https://arxiv.org/pdf/2404.14197
    Venue: NeurIPS 2024
    Task: Long-term Time Series Forecasting
    '''
    def __init__(self, configs, **args):
        super(SOFTS, self).__init__(**args)
        # configs = Namespace(**args)
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(configs.seq_len, configs.d_model, configs.dropout)
        self.use_norm = configs.use_norm
        self.time_of_day_size = configs.time_of_day_size
        # Encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    STAR(configs.d_model, configs.d_core),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                ) for l in range(configs.e_layers)
            ],
        )

        # Decoder
        self.projection = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc):
        # Normalization from Non-stationary Transformer
        if self.use_norm:
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
            x_enc /= stdev

        _, _, N = x_enc.shape
        enc_out = self.enc_embedding(x_enc, x_mark_enc)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        dec_out = self.projection(enc_out).permute(0, 2, 1)[:, :, :N]

        # De-Normalization from Non-stationary Transformer
        if self.use_norm:
            dec_out = dec_out * (stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
            dec_out = dec_out + (means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        return dec_out

    def forward_xformer(self, x_enc, x_mark_enc):
        dec_out = self.forecast(x_enc, x_mark_enc)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


    def forward(self, history_data: torch.Tensor, future_data: torch.Tensor) -> torch.Tensor:
        """

        Args:
            history_data (Tensor): Input data with shape: [B, L1, N, C]
            future_data (Tensor): Future data with shape: [B, L2, N, C]

        Returns:
            torch.Tensor: outputs with shape [B, L2, N, 1]
        """

        # change TimeOfDay to MinuteOfHour
        history_data[..., 1] = history_data[..., 1] * self.time_of_day_size // (self.time_of_day_size / 24) / 23.0
        x_enc, x_mark_enc, x_dec, x_mark_dec = data_transformation_4_xformer(history_data=history_data,
                                                                             future_data=future_data,
                                                                             start_token_len=0)
        #print(x_mark_enc.shape, x_mark_dec.shape)
        prediction = self.forward_xformer(x_enc=x_enc, x_mark_enc=x_mark_enc)
        return prediction.unsqueeze(-1)
    



class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu", **kwargs):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None, tau=None, delta=None, **kwargs):
        new_x, attn = self.attention(
            x, x, x,
            attn_mask=attn_mask,
            tau=tau, delta=delta,
            **kwargs
        )
        x = x + self.dropout(new_x)

        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))

        return self.norm2(x + y), attn


class Encoder(nn.Module):
    def __init__(self, attn_layers, conv_layers=None, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.conv_layers = nn.ModuleList(conv_layers) if conv_layers is not None else None
        self.norm = norm_layer

    def forward(self, x, attn_mask=None, tau=None, delta=None, **kwargs):
        # x [B, L, D]
        attns = []
        if self.conv_layers is not None:
            for i, (attn_layer, conv_layer) in enumerate(zip(self.attn_layers, self.conv_layers)):
                delta = delta if i == 0 else None
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta)
                x = conv_layer(x)
                attns.append(attn)
            x, attn = self.attn_layers[-1](x, tau=tau, delta=None)
            attns.append(attn)
        else:
            for attn_layer in self.attn_layers:
                x, attn = attn_layer(x, attn_mask=attn_mask, tau=tau, delta=delta, **kwargs)
                attns.append(attn)

        if self.norm is not None:
            x = self.norm(x)

        return x, attns
    


class DataEmbedding_inverted(nn.Module):
    def __init__(self, c_in, d_model, dropout=0.1):
        super(DataEmbedding_inverted, self).__init__()
        self.value_embedding = nn.Linear(c_in, d_model)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark):
        x = x.permute(0, 2, 1)
        # x: [Batch Variate Time]
        if x_mark is None:
            x = self.value_embedding(x)
        else:
            # the potential to take covariates (e.g. timestamps) as tokens
            x = self.value_embedding(torch.cat([x, x_mark.permute(0, 2, 1)], 1))
        # x: [Batch Variate d_model]
        return self.dropout(x)