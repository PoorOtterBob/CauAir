import torch
from torch import nn, optim
import torch.nn.functional as F
from src.base.model import BaseModel
from src.utils.module.dsformer.TVA_block import TVA_block_att
from src.utils.module.dsformer.decoder_block import TVADE_block

class DSFormer(BaseModel):
    def __init__(self, 
                 # Input_len,
                 # out_len, 
                 # num_id, 
                 num_layer=1, 
                 dropout=0.2, 
                 muti_head=4, 
                 num_samp=3, 
                 IF_node=True,
                 **args):
        """
        Input_len: History length
        out_len: future length
        num_id: number of variables
        num_layer: number of layer. 1 or 2
        muti_head: number of muti_head attention. 1 to 4
        dropout: dropout. 0.15 to 0.3
        num_samp: muti_head subsequence. 2 or 3
        IF_node:Whether to use node embedding. True or False
        """
        super(DSFormer, self).__init__(**args)

        Input_len = self.seq_len
        out_len = self.horizon
        num_id = self.node_num

        if IF_node:
            self.inputlen = 2 * Input_len // num_samp
        else:
            self.inputlen = Input_len // num_samp

        ### embed and encoder
        self.RevIN = RevIN(num_id)
        self.embed_layer = embed(Input_len,num_id,num_samp,IF_node)
        self.encoder = TVA_block_att(self.inputlen,num_id,num_layer,dropout, muti_head,num_samp)
        self.laynorm = nn.LayerNorm([self.inputlen])

        ### decorder
        self.decoder = TVADE_block(self.inputlen, num_id, dropout, muti_head)
        self.output = nn.Conv1d(in_channels = self.inputlen, out_channels=out_len, kernel_size=1)

    def forward(self, x, label=None):
        # Input [B,H,N]: B is batch size. N is the number of variables. H is the history length
        # Output [B,L,N]: B is batch size. N is the number of variables. L is the future length
        x = x[..., 0]

        ### embed
        x = self.RevIN(x,'norm').transpose(-2,-1)
        x_1, x_2 = self.embed_layer(x)

        ### encoder
        x_1 = self.encoder(x_1)
        x_2 = self.encoder(x_2)
        x = x_1 + x_2
        x = self.laynorm(x)

        ### decorder
        x = self.decoder(x)
        x = self.output(x.transpose(-2,-1))
        x = self.RevIN(x, 'denorm')

        return x.unsqueeze(-1)
    
class embed(nn.Module):
    def __init__(self,Input_len, num_id,num_samp,IF_node):
        super(embed, self).__init__()
        self.IF_node = IF_node
        self.num_samp = num_samp
        self.embed_layer = nn.Linear(2*Input_len,Input_len)

        self.node_emb = nn.Parameter(torch.empty(num_id, Input_len))
        nn.init.xavier_uniform_(self.node_emb)

    def forward(self, x):

        x = x.unsqueeze(-1)
        batch_size, _, _ ,_ = x.shape
        node_emb1 = self.node_emb.unsqueeze(0).expand(batch_size, -1, -1).unsqueeze(-1)

        x_1 = embed.down_sampling(x, self.num_samp)
        if self.IF_node:
            x_1 = torch.cat([x_1, embed.down_sampling(node_emb1, self.num_samp)], dim=-1)

        x_2 = embed.Interval_sample(x, self.num_samp)
        if self.IF_node:
            x_2 = torch.cat([x_2, embed.Interval_sample(node_emb1, self.num_samp)], dim=-1)

        return x_1,x_2

    @staticmethod
    def down_sampling(data,n):
        result = 0.0
        for i in range(n):
            line = data[:,:,i::n,:]
            if i == 0:
                result = line
            else:
                result = torch.cat([result, line], dim=3)
        result = result.transpose(2, 3)
        return result

    @staticmethod
    def Interval_sample(data,n):
        result = 0.0
        data_len = data.shape[2] // n
        for i in range(n):
            line = data[:,:,data_len*i:data_len*(i+1),:]
            if i == 0:
                result = line
            else:
                result = torch.cat([result, line], dim=3)
        result = result.transpose(2, 3)
        return result
    
class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True, subtract_last=False):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        self.subtract_last = subtract_last
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim-1))
        if self.subtract_last:
            self.last = x[:,-1,:].unsqueeze(1)
        else:
            self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        if self.subtract_last:
            x = x - self.last
        else:
            x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        if self.subtract_last:
            x = x + self.last
        else:
            x = x + self.mean
        return x