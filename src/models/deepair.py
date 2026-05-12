import torch
import torch.nn as nn
from src.base.model import BaseModel
import numpy as np
import sys

class DeepAir(BaseModel):
    def __init__(self, d_hid, if_future_side_information, **args):
        super(DeepAir, self).__init__(**args)
        # self.input_dim = self.input_dim + (1 if if_future_side_information else 0)
        self.FusionNet = nn.ModuleList([
            FusionNet(d_hid*((self.input_dim + (1 if if_future_side_information else 0)) if i == 0 else 2))
        for i in range(self.input_dim + (1 if if_future_side_information else 0))])
        
        self.emb = nn.ModuleList([
            nn.Conv1d(self.seq_len, 
                      d_hid, 
                      kernel_size=1)
        for _ in range(self.input_dim)])

        self.merge = nn.ModuleList([
            nn.Conv1d(d_hid*((self.input_dim + (1 if if_future_side_information else 0)) if i == 0 else 2), 
                      self.output_dim*self.horizon,
                      kernel_size=1,
                      bias=False)
        for i in range(self.input_dim + (1 if if_future_side_information else 0))])

        if if_future_side_information:
            self.future_fusion = nn.Sequential(
                nn.Conv1d(self.horizon*(self.input_dim-1), d_hid, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(d_hid, d_hid, kernel_size=1),
            )

        self.if_future_side_information = if_future_side_information

    def forward(self, x, label=None, adj=None):
        h_x = self.emb[0](x[..., 0])
        feature = [h_x]
        output = 0
        for i in range(1, self.input_dim): 
            h = self.emb[i](x[..., i])
            feature.append(h)
            z = self.FusionNet[i](torch.cat([h_x, h], dim=1))
            output += self.merge[i](z)

        if self.if_future_side_information:
            h = label.transpose(-1, -2).reshape(x.shape[0], -1, self.node_num)
            h = self.future_fusion(h)
            feature.append(h)
            z = self.FusionNet[-1](torch.cat([h_x, h], dim=1))
            output += self.merge[-1](z)

        z = self.FusionNet[0](torch.cat(feature, dim=1))
        output += self.merge[0](z)
        
        return output.unsqueeze(-1)

class FusionNet(nn.Module):
    def __init__(self, d_hid):
        super().__init__()
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(d_hid, d_hid, kernel_size=1),
                nn.ReLU(),
                nn.Conv1d(d_hid, d_hid, kernel_size=1),
            )
        for _ in range(3)]
        )
        
    def forward(self, x):
        x = self.fc[0](x)
        x = self.fc[1](x) + x
        x = self.fc[2](x)
        return x