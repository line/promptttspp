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
import torch.nn as nn
import torch.nn.functional as F


class ConvNeXtLayer(nn.Module):
    def __init__(self, channels, h_channels, scale):
        super().__init__()
        self.dw_conv = nn.Conv1d(
            channels, channels, kernel_size=7, padding=3, groups=channels
        )
        self.norm = nn.LayerNorm(channels)
        self.pw_conv1 = nn.Linear(channels, h_channels)
        self.pw_conv2 = nn.Linear(h_channels, channels)
        self.scale = nn.Parameter(
            torch.full(size=(channels,), fill_value=scale), requires_grad=True
        )

    def forward(self, x, mask):
        res = x
        x = self.dw_conv(x)
        x = x.transpose(1, 2)
        x = self.norm(x)
        x = self.pw_conv1(x)
        x = F.gelu(x)
        x = self.pw_conv2(x)
        x = self.scale * x
        x = x.transpose(1, 2)
        x = res + x
        return x * mask


class ConvNeXt1d(nn.Module):
    def __init__(self, channels, h_channels, num_layers):
        super().__init__()
        scale = 1.0 / num_layers
        self.norm_pre = nn.LayerNorm(channels)
        self.layers = nn.ModuleList(
            [ConvNeXtLayer(channels, h_channels, scale) for _ in range(num_layers)]
        )
        self.norm_post = nn.LayerNorm(channels)

    def forward(self, x, mask):
        x = x.transpose(-1, -2)
        x = self.norm_pre(x)
        x = x.transpose(-1, -2)
        for layer in self.layers:
            x = layer(x, mask)
        x = x.transpose(-1, -2)
        x = self.norm_post(x)
        x = x.transpose(-1, -2)
        return x * mask
