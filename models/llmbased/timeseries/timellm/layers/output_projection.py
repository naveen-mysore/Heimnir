import torch
import os
import json
import torch.nn as nn


class FlattenHead(nn.Module):
    def __init__(self, configs):
        super(FlattenHead, self).__init__()
        self.patch_nums = int((configs['model']['seq_len']['value'] - configs['llm']['stride']['value']) / configs['llm']['stride']['value'] + 2)
        self.head_nf = configs['model']['d_ff']['value'] * self.patch_nums

        self.n_vars = configs['model']['enc_in']['value']
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(self.head_nf, configs['model']['pred_len']['value'])
        self.dropout = nn.Dropout(configs['learning']['dropout']['value'])

    def forward(self, x):
        x = x[:, :, :, -self.patch_nums:]
        x = self.flatten(x)
        x = self.linear(x)
        x = self.dropout(x)
        x = x.permute(0, 2, 1).contiguous()
        return x