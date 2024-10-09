# Copyright 2024 LY Corporation

# LY Corporation licenses this file to you under the Apache License,
# version 2.0 (the "License"); you may not use this file except in compliance
# with the License. You may obtain a copy of the License at:

#   https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations
# under the License.

import torch
from torch import nn
from torch.nn import functional as F

from .embedding import PositionalEncoding, RelPositionalEncoding


class LayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.channels = channels
        self.eps = eps

        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))

    def forward(self, x):
        x = x.transpose(1, -1)
        x = F.layer_norm(x, (self.channels,), self.gamma, self.beta, self.eps)
        return x.transpose(1, -1)


class FramePriorNetwork(nn.Module):
    def __init__(
        self,
        out_channels,
        hidden_channels,
        n_layers,
        kernel_size,
        p_dropout,
        pos_enc_p_dropout=0.1,
        use_pos_enc=True,
        use_rel=False,
    ):
        super().__init__()
        self.out_channels = out_channels
        self.hidden_channels = hidden_channels

        _Conv1d = nn.Conv1d

        self.use_pos_enc = use_pos_enc
        if use_pos_enc:
            if use_rel:
                self.embed = RelPositionalEncoding(hidden_channels, pos_enc_p_dropout)
            else:
                self.embed = PositionalEncoding(hidden_channels, pos_enc_p_dropout)
            self.norm_emb = LayerNorm(hidden_channels)
        self.n_layers = n_layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.drop = nn.Dropout(p_dropout)
        self.act = nn.GELU()

        for _ in range(n_layers):
            self.convs.append(
                _Conv1d(
                    hidden_channels,
                    hidden_channels,
                    kernel_size,
                    padding=kernel_size // 2,
                )
            )
            self.norms.append(LayerNorm(hidden_channels))

    def forward(self, x, x_mask):
        if self.use_pos_enc:
            x = x * x_mask
            x = self.embed(x.transpose(1, 2)).transpose(1, 2)
            x = self.norm_emb(x)

        for i in range(self.n_layers):
            res = self.convs[i](x * x_mask)
            res = self.act(res)
            res = self.drop(res)
            x = self.norms[i](x + res)

        x = x * x_mask
        return x
