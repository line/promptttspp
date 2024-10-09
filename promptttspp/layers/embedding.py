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

import math

import torch
import torch.nn as nn


class PhonemeEmbedding(nn.Module):
    def __init__(self, num_vocab, channels, do_scale=True, init_normal=True):
        super().__init__()
        self.emb = nn.Embedding(num_vocab, channels, padding_idx=0)
        if init_normal:
            torch.nn.init.normal_(self.emb.weight, 0.0, channels ** -0.5)
        self.scale = math.sqrt(channels)
        self.do_scale = do_scale

    def forward(self, x, mask):
        x = self.emb(x)
        if self.do_scale:
            x = x * self.scale
        x = x.transpose(-1, -2)  # [B, C, T]
        x = x * mask
        return x


class PhonemeEmbedding2(nn.Module):
    def __init__(self, num_vocab, channels):
        super().__init__()
        self.emb = nn.Embedding(num_vocab, channels, padding_idx=0)

    def forward(self, x, mask):
        x = self.emb(x)
        x = x.transpose(-1, -2)  # [B, C, T]
        x = x * mask
        return x
