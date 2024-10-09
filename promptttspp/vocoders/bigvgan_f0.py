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

import numpy as np
import torch
import torch.nn as nn
from promptttspp.layers.activations import AntiAliasActivation
from torch.nn.utils import remove_weight_norm, weight_norm

from .bigvgan import AMPBlock
from .nsf import SourceModuleHnNSF


class F0AwareBigVGAN(nn.Module):
    def __init__(
        self,
        sampling_rate,
        harmonic_num,
        in_channel,
        upsample_initial_channel,
        upsample_rates,
        upsample_kernel_sizes,
        resblock_kernel_sizes,
        resblock_dilations,
    ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)

        self.f0_up = nn.Upsample(scale_factor=np.prod(upsample_rates))
        self.m_source = SourceModuleHnNSF(
            sampling_rate=sampling_rate,
            harmonic_num=harmonic_num,
        )
        self.noise_convs = nn.ModuleList()

        self.conv_pre = weight_norm(
            nn.Conv1d(
                in_channel, upsample_initial_channel, kernel_size=7, stride=1, padding=3
            )
        )
        self.upsamples = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.upsamples.append(
                weight_norm(
                    nn.ConvTranspose1d(
                        upsample_initial_channel // (2 ** i),
                        upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=k,
                        stride=u,
                        padding=upsample_rates[i] // 2 + upsample_rates[i] % 2,
                        output_padding=upsample_rates[i] % 2,
                    )
                )
            )
            if i + 1 < len(upsample_rates):
                stride_f0 = np.prod(upsample_rates[i + 1 :])
                self.noise_convs += [
                    nn.Conv1d(
                        1,
                        upsample_initial_channel // (2 ** (i + 1)),
                        kernel_size=stride_f0 * 2,
                        stride=stride_f0,
                        padding=stride_f0 // 2,
                    )
                ]
            else:
                self.noise_convs += [
                    nn.Conv1d(1, upsample_initial_channel // (2 ** (i + 1)), 1)
                ]

        self.mrfs = nn.ModuleList()
        for i in range(len(self.upsamples)):
            channel = upsample_initial_channel // (2 ** (i + 1))
            self.mrfs.append(
                nn.ModuleList(
                    [
                        AMPBlock(channel, kernel_size=k, dilations=d)
                        for k, d in zip(resblock_kernel_sizes, resblock_dilations)
                    ]
                )
            )
        self.act_post = AntiAliasActivation(channel)
        self.conv_post = weight_norm(
            nn.Conv1d(channel, 1, kernel_size=7, stride=1, padding=3)
        )

    def forward(self, x, f0):
        f0 = self.f0_up(f0).transpose(-1, -2)
        har_source, _, _ = self.m_source(f0)
        har_source = har_source.transpose(-1, -2)

        x = self.conv_pre(x)
        for up, noise_conv, mrf in zip(self.upsamples, self.noise_convs, self.mrfs):
            x = up(x)
            x_source = noise_conv(har_source)
            x = x + x_source
            xs = 0
            for layer in mrf:
                xs += layer(x)
            x = xs / self.num_kernels
        x = self.act_post(x)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x

    def remove_weight_norm(self):
        remove_weight_norm(self.conv_pre)
        for up in self.upsamples:
            remove_weight_norm(up)
        for mrf in self.resblocks:
            mrf.remove_weight_norm()
        remove_weight_norm(self.conv_post)
