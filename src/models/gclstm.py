import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from src.base.model import BaseModel


class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.Tensor(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(4 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(4 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(4 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters using Kaiming initialization for weights and zeros for biases."""
        nn.init.kaiming_uniform_(self.weight_ih, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_hh, a=math.sqrt(5))
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)

    def forward(self, x, hx, cx):
        """
        Forward pass of LSTM cell.

        Args:
        x:  [batch, input_size] - Input tensor.
        hx: [batch, hidden_size] - Previous hidden state.
        cx: [batch, hidden_size] - Previous cell state.

        Returns:
        hy: [batch, hidden_size] - New hidden state.
        cy: [batch, hidden_size] - New cell state.
        """
        gates = F.linear(x, self.weight_ih, self.bias_ih) + F.linear(
            hx, self.weight_hh, self.bias_hh
        )
        i, f, o, g = gates.chunk(4, dim=1)  # Chunk into four gate activations

        i = torch.sigmoid(i)  # Input gate
        f = torch.sigmoid(f)  # Forget gate
        o = torch.sigmoid(o)  # Output gate
        g = torch.tanh(g)  # Candidate cell state

        cy = f * cx + i * g  # Update cell state
        hy = o * torch.tanh(cy)  # Update hidden state

        return hy, cy


class GCLSTM(BaseModel):
    def __init__(self, gso, **args):
        super(GCLSTM, self).__init__(**args)
    
        self.hist_len = self.seq_len
        self.fut_len = self.horizon

        self.out_dim = self.output_dim
        self.in_dim = self.input_dim

        self.use_emis = False
        if self.use_emis:
            self.emis_vars = self.task["variables"]["emis"].split("+")
            self.in_dim += len(self.emis_vars)

        # Model configuration
        self.hidden_size = 128
        self.dropout_rate = 0.1

        # Modules
        self.mlp_in = nn.Sequential(
            nn.Linear(self.in_dim, self.hidden_size),
        )

        self.lstm_cell = LSTMCell(self.hidden_size, self.hidden_size)

        self.conv = ChebGraphConv(self.hidden_size, self.hidden_size, Ks=2, gso=gso)

        self.decoder = nn.Sequential(
            nn.Linear(self.hidden_size, self.fut_len * self.out_dim),
        )

    def forward(self, inputs, labels, adj=None):
        # aqi_hist = inputs["aqi_hist"]
        # mete_hist = inputs["mete_hist"]
        # mete_fut = inputs["mete_fut"]

        aqi_hist = inputs[..., :1]
        mete_hist = inputs[..., 1:]
        mete_fut = labels

        x_hist = mete_hist
        x_fut = mete_fut

        if self.use_emis:
            emis_hist = inputs["emis_hist"]
            x_hist = torch.cat([x_hist, emis_hist], dim=-1)

            emis_fut = inputs["emis_fut"]
            x_fut = torch.cat([x_fut, emis_fut], dim=-1)

        x_in = torch.cat([x_hist, x_fut], dim=1)  # [batch, time, nodes, in_dim]
        bs, ts, n, _ = x_in.shape
        bs, ts, n, _ = x_in.shape

        h_t = torch.zeros(
            bs * n, self.hidden_size, device=x_in.device, dtype=x_in.dtype
        )

        c_t = torch.zeros(
            bs * n, self.hidden_size, device=x_in.device, dtype=x_in.dtype
        )

        # 初始化 AQI 历史输入
        aqi_t = aqi_hist[:, 0]  # 使用第一个时间步的 AQI 初始化
        for t in range(self.hist_len):
            # 如果是历史阶段，直接使用真实值；如果是预测阶段，使用模型预测的值
            if t < self.hist_len:
                aqi_t = aqi_hist[:, t]

            # 将气象和 AQI 数据拼接
            x_t = torch.cat((x_in[:, t], aqi_t), dim=-1)  # [batch, nodes, features]
            x_t = self.mlp_in(x_t)

            x_t = self.conv(x_t)

            # 更新 GRU 隐藏状态
            h_t, c_t = self.lstm_cell(x_t.view(bs*n, -1), h_t, c_t)  # [batch, nodes, hidden_size]

        aqi_t = self.decoder(h_t)
        aqi_t = aqi_t.view(bs, n, self.fut_len, self.out_dim).permute(0, 2, 1, 3)

        return aqi_t


class ChebGraphConv(nn.Module):
    def __init__(self, c_in, c_out, Ks, gso):
        super(ChebGraphConv, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        self.Ks = Ks
        self.gso = gso
        self.weight = nn.Parameter(torch.FloatTensor(Ks, c_in, c_out))
        self.bias = nn.Parameter(torch.FloatTensor(c_out))
        self.reset_parameters()


    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        init.uniform_(self.bias, -bound, bound)


    def forward(self, x):

        if self.Ks - 1 < 0:
            raise ValueError(f'ERROR: the graph convolution kernel size Ks has to be a positive integer, but received {self.Ks}.')  
        elif self.Ks - 1 == 0:
            x_0 = x
            x_list = [x_0]
        elif self.Ks - 1 == 1:
            x_0 = x
            x_1 = torch.einsum('hi, bij->bhj', self.gso, x)
            x_list = [x_0, x_1]
        elif self.Ks - 1 >= 2:
            x_0 = x
            x_1 = torch.einsum('hi,bij->bhj', self.gso, x)
            x_list = [x_0, x_1]
            for k in range(2, self.Ks):
                x_list.append(torch.einsum('hi,bij->bhj', 2 * self.gso, x_list[k - 1]) - x_list[k - 2])
        
        x = torch.stack(x_list, dim=1)

        cheb_graph_conv = torch.einsum('bkhi,kij -> bhj', x, self.weight)
        cheb_graph_conv = torch.add(cheb_graph_conv, self.bias)
        return cheb_graph_conv