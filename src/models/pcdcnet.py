import os
import math
import torch
import torch.nn.functional as F
from torch import nn
from src.base.model import BaseModel
import math
import numbers
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Size
from torch.nn.parameter import Parameter


class GCNLayer(nn.Module):
    def __init__(
        self, in_features, out_features, gso, num_layers=2, dropout=0.1, drop_edge_p=0.1
    ):
        super(GCNLayer, self).__init__()
        self.num_layers = num_layers
        self.gcn_layers = nn.ModuleList(
            [nn.Linear(in_features, out_features) for _ in range(num_layers)]
        )
        self.dropout = nn.Dropout(dropout)
        self.drop_edge_p = drop_edge_p  # DropEdge 概率
        self.cached_adj = None  # 缓存邻接矩阵

        self.adj = gso
    def drop_edges(self, adj, drop_prob=0.1):
        """
        随机删除一定比例的边（DropEdge 机制）。
        :param adj: (nodes, nodes)  标准化后的邻接矩阵
        :param drop_prob: float, 要丢弃的边的比例
        :return: (nodes, nodes)  处理后的邻接矩阵
        """
        if drop_prob <= 0:
            return adj  # 不做 DropEdge

        mask = (torch.rand_like(adj) > drop_prob).float()  # 生成 0/1 掩码
        adj = adj * mask  # 丢弃部分边
        return adj

    def forward(self, x):
        """
        Applies graph convolution.
        - x: (batch, nodes, features)
        - adj: (nodes, nodes)  已经标准化的邻接矩阵
        """
        b, n, f = x.shape
        h = x

        # **缓存邻接矩阵，避免重复 expand**
        if self.cached_adj is None or self.training:  # 训练时允许 DropEdge
            adj = self.drop_edges(self.adj, self.drop_edge_p)
            self.cached_adj = adj.unsqueeze(0)  # 缓存

        adj = self.cached_adj.expand(b, -1, -1)  # (batch, nodes, nodes)

        for i in range(self.num_layers):
            # h_new = torch.bmm(adj.expand(b, -1, -1), h)  # 邻接矩阵传播
            h_new = torch.einsum("bmn,bnf->bmf", adj, h)  # **更快的矩阵传播**
            h_new = self.gcn_layers[i](h_new)  # 线性变换
            h_new = F.gelu(h_new)
            h_new = self.dropout(h_new)  # Dropout
            h = h + h_new  # 残差连接

        return h


class PCDCNet(BaseModel):
    def __init__(self, gso, **args):
        super(PCDCNet, self).__init__(**args)

        self.hist_len = self.seq_len
        self.fut_len = self.horizon

        self.out_dim = self.output_dim
        self.in_dim = self.input_dim

        self.use_emis = False
        self.use_adv = True
        if self.use_emis:
            self.emis_vars = self.task["variables"]["emis"].split("+")
            self.in_dim += len(self.emis_vars)

        self.hid_size = 64
        self.dropout_rate = 0.1
        self.eps = 1e-6
        self.fmix_size = 4 * self.hid_size

        self.use_fmix = True
        self.use_spatial = True
        self.use_temporal = True

        self.embed = nn.Linear(self.in_dim, self.hid_size)
        self.readout = nn.Linear(self.hid_size, self.out_dim)
        self.out_norm = nn.RMSNorm([self.hid_size], eps=self.eps)

        if self.use_temporal:
            self.gru_cell = GRUCell(self.hid_size, self.hid_size)
            self.gru_norm = nn.RMSNorm([self.hid_size], eps=self.eps)
            self.hid_norm = nn.RMSNorm([self.hid_size], eps=self.eps)

        if self.use_spatial:
            self.gcn = GCNLayer(self.hid_size, self.hid_size, gso, dropout=self.dropout_rate)
            self.gnn_norm = nn.RMSNorm([self.hid_size], eps=self.eps)
            if self.use_adv:
                self.adv_lambda = 0.0
                self.adv_method = "mae"
        if self.use_fmix:
            self.feat_mix = nn.Sequential(
                nn.Linear(self.hid_size, self.fmix_size),
                nn.SiLU(),
                nn.Linear(self.fmix_size, self.hid_size),
                nn.Dropout(self.dropout_rate),
            )
            self.fmix_norm = nn.RMSNorm([self.hid_size], eps=self.eps)

        
        self.adv_lambda = 10
        self.gso = gso


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

        if self.use_temporal:
            h_t = torch.zeros(
                bs * n, self.hid_size, device=x_in.device, dtype=x_in.dtype
            )

        all_preds = []

        if self.use_spatial and self.use_adv:
            gcn_preds = []

        adv_loss = 0.0

        for t in range(ts):
            if t < self.hist_len:
                aqi_t = aqi_hist[:, t]

            x_t = torch.cat((x_in[:, t], aqi_t), dim=-1)
            x_t = self.embed(x_t)

            if self.use_fmix:
                x_t = x_t + self.feat_mix(self.fmix_norm(x_t))

            if self.use_spatial:
                gcn_out = self.gcn(self.gnn_norm(x_t))
                x_t = x_t + gcn_out

                if self.use_adv:
                    # [batch, nodes, aqi]
                    gcn_preds.append(self.readout(self.out_norm(gcn_out)))

            if self.use_temporal:
                h_t = self.gru_cell(
                    self.gru_norm(x_t.view(bs * n, -1)), self.hid_norm(h_t)
                )
                x_t = x_t + h_t.view(bs, n, -1)

            x_delta = self.readout(self.out_norm(x_t))
            aqi_t = aqi_t + x_delta.view(bs, n, -1)

            all_preds.append(aqi_t)

        aqi_fut_hat = torch.stack(all_preds, dim=1)[:, self.hist_len :]

        if self.use_spatial and self.use_adv:
            # torch.Size([32, 72, 228])
            gcn_preds = torch.stack(gcn_preds, dim=1)[:, self.hist_len :]

            # Spatial Regularization
            # src, tgt = inputs["edge_index"]
            # diff_s = (
            #     gcn_preds[:, :, src, :] - gcn_preds[:, :, tgt, :]
            # )  # (batch, time, edge, aqi)


            diff_s = gcn_preds - torch.einsum('nm, btnf -> btmf', self.gso, gcn_preds)

            diff_s = diff_s.sum(2)
            if self.adv_method == "mae":
                adv_loss_s = torch.abs(diff_s.mean(dim=2)).mean()
            elif self.adv_method == "mse":
                adv_loss_s = torch.pow(diff_s.mean(dim=2), 2).mean()
            else:
                raise ValueError(f"Invalid adv_method: {self.adv_method}")

            # Temporal Regularization
            # torch.Size([32, 71, 228, 2])
            diff_t = gcn_preds[:, 1:] - gcn_preds[:, :-1]
            if self.adv_method == "mae":
                adv_loss_t = torch.abs(diff_t.mean(1)).mean()
            elif self.adv_method == "mse":
                adv_loss_t = torch.pow(diff_t, 2).mean()
            else:
                raise ValueError(f"Invalid adv_method: {self.adv_method}")

            if self.adv_lambda != 0:
                adv_loss = self.adv_lambda * (adv_loss_s + adv_loss_t)
            else:
                adv_loss = adv_loss_s + adv_loss_t

        return aqi_fut_hat, adv_loss



class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weight_ih = nn.Parameter(torch.Tensor(3 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.Tensor(3 * hidden_size, hidden_size))
        self.bias_ih = nn.Parameter(torch.Tensor(3 * hidden_size))
        self.bias_hh = nn.Parameter(torch.Tensor(3 * hidden_size))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight_ih, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight_hh, a=math.sqrt(5))
        nn.init.zeros_(self.bias_ih)
        nn.init.zeros_(self.bias_hh)

    def forward(self, x, hx):
        # x: [batch, input_size]
        # hx: [batch, hidden_size]
        gates = F.linear(x, self.weight_ih, self.bias_ih) + F.linear(
            hx, self.weight_hh, self.bias_hh
        )
        r, z, n = gates.chunk(3, dim=1)

        r = torch.sigmoid(r)
        z = torch.sigmoid(z)
        n = torch.tanh(n + r * hx)
        hy = (1 - z) * n + z * hx
        return hy



