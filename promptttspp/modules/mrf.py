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

import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import remove_weight_norm, weight_norm

LRELU_SLOPE = 0.1


class MRFLayer(nn.Module):
    def __init__(self, channels, kernel_size, dilation):
        super().__init__()
        self.conv1 = weight_norm(
            nn.Conv1d(
                channels,
                channels,
                kernel_size,
                padding=(kernel_size * dilation - dilation) // 2,
                dilation=dilation,
            )
        )
        self.conv2 = weight_norm(
            nn.Conv1d(
                channels, channels, kernel_size, padding=kernel_size // 2, dilation=1
            )
        )

    def forward(self, x):
        y = F.leaky_relu(x, LRELU_SLOPE)
        y = self.conv1(y)
        y = F.leaky_relu(y, LRELU_SLOPE)
        y = self.conv2(y)
        return x + y

    def remove_weight_norm(self):
        remove_weight_norm(self.conv1)
        remove_weight_norm(self.conv2)


class MRFBlock(nn.Module):
    def __init__(self, channels, kernel_size, dilations):
        super().__init__()
        self.layers = nn.ModuleList()
        for dilation in dilations:
            self.layers.append(MRFLayer(channels, kernel_size, dilation))

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x) * mask
        return x

    def remove_weight_norm(self):
        for layer in self.layers:
            layer.remove_weight_norm()


class MRFNet(nn.Module):
    def __init__(self, in_channels, channels, out_channels, kernel_sizes, dilations):
        super().__init__()
        self.in_conv = weight_norm(
            nn.Conv1d(in_channels, channels, kernel_size=3, padding=1)
        )
        self.blocks = nn.ModuleList()
        for kernel_size in kernel_sizes:
            self.blocks += [MRFBlock(channels, kernel_size, dilations)]
        self.out_conv = weight_norm(
            nn.Conv1d(channels, out_channels, kernel_size=3, padding=1)
        )

    def forward(self, x, mask, g=None):
        for block in self.blocks:
            if g is not None:
                x = x + g
            x = block(x, mask)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.in_conv)
        for block in self.blocks:
            block.remove_weight_norm()
        remove_weight_norm(self.out_conv)
