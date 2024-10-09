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


class Glow(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        num_flows,
        n_blocks,
        gin_channels=0,
    ):
        super(Glow, self).__init__()

        self.flows = nn.ModuleList()
        for _ in range(num_flows):
            # self.flows.append(ActNorm(in_channels))
            self.flows.append(Invertible1x1Conv(in_channels))
            self.flows.append(
                AffineCoupling(
                    in_channels,
                    channels,
                    n_blocks,
                    gin_channels,
                )
            )

    def forward(self, z, g=None):
        log_df_dz = 0
        for flow in self.flows:
            z, log_df_dz = flow(z=z, log_df_dz=log_df_dz, g=g)
        return z, log_df_dz

    def reverse(self, y, g=None):
        log_df_dz = 0
        for flow in reversed(self.flows):
            y, log_df_dz = flow.reverse(y=y, log_df_dz=log_df_dz, g=g)
        return y, log_df_dz

    def remove_weight_norm(self):
        for layer in self.flows:
            if isinstance(layer, AffineCoupling):
                layer.remove_weight_norm()


class ActNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super(ActNorm, self).__init__()
        self.channels = channels
        self.eps = eps

        self.dimensions = [1, channels, 1]
        self.register_parameter("log_scale", nn.Parameter(torch.zeros(self.dimensions)))
        self.register_parameter("bias", nn.Parameter(torch.zeros(self.dimensions)))
        self.initialized = False

    def forward(self, z, log_df_dz, **kwargs):
        if not self.initialized:
            self.initialize(z)
            self.initialized = True

        z = z * torch.exp(self.log_scale) + self.bias

        log_df_dz += torch.sum(self.log_scale)
        return z, log_df_dz

    def reverse(self, y, log_df_dz, **kwargs):
        y = (y - self.bias) * torch.exp(-self.log_scale)
        log_df_dz -= torch.sum(self.log_scale)
        return y, log_df_dz

    @torch.no_grad()
    def initialize(self, x):
        # x: [B, C, 1]
        print("Initialized")
        m = x.mean(dim=[0, 2])
        logs = torch.log(torch.std(x, dim=[0, 2]) + self.eps)

        bias_init = (-m * torch.exp(-logs)).view(self.dimensions)
        logs_init = (-logs).view(self.dimensions)

        self.bias.data.copy_(bias_init)
        self.log_scale.data.copy_(logs_init)


class Invertible1x1Conv(nn.Module):
    def __init__(self, channels):
        super(Invertible1x1Conv, self).__init__()
        self.channels = channels

        w_init = torch.linalg.qr(
            torch.FloatTensor(self.channels, self.channels).normal_()
        )[0]
        self.weight = nn.Parameter(w_init)

    def forward(self, z, log_df_dz, **kwargs):
        weight = self.weight
        z = F.conv1d(z, weight.unsqueeze(-1))

        log_df_dz += torch.slogdet(weight)[1]
        return z, log_df_dz

    def reverse(self, y, log_df_dz, **kwargs):
        weight = self.weight.inverse()
        y = F.conv1d(y, weight.unsqueeze(-1))

        log_df_dz -= torch.slogdet(weight)[1]
        return y, log_df_dz


class ResBlockLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlockLinear, self).__init__()

        self.net = nn.Sequential(
            nn.ReLU(inplace=True),
            torch.nn.utils.weight_norm(nn.Conv1d(in_channels, out_channels, 1)),
            nn.ReLU(inplace=True),
            torch.nn.utils.weight_norm(nn.Conv1d(out_channels, out_channels, 1)),
        )

        if in_channels != out_channels:
            self.bridge = torch.nn.utils.weight_norm(
                nn.Conv1d(in_channels, out_channels, 1)
            )
        else:
            self.bridge = nn.Identity()

    def forward(self, x):
        y = self.net(x)
        x = self.bridge(x)
        return x + y


class MLP(nn.Module):
    def __init__(
        self, in_channels, out_channels, base_filters=256, n_blocks=2, gin_channels=0
    ):
        super(MLP, self).__init__()

        self.in_block = nn.Sequential(
            torch.nn.utils.weight_norm(nn.Conv1d(in_channels, base_filters, 1)),
        )

        self.mid_block = nn.ModuleList()
        for _ in range(n_blocks):
            self.mid_block.append(ResBlockLinear(base_filters, base_filters))

        self.out_block = nn.Sequential(
            nn.ReLU(inplace=True),
            torch.nn.utils.weight_norm(nn.Conv1d(base_filters, out_channels, 1)),
        )
        if gin_channels > 0:
            self.cond_layer = nn.Conv1d(gin_channels, base_filters, 1)

    def forward(self, x, g=None):
        x = self.in_block(x)
        if g is not None:
            x = x + self.cond_layer(g)
        for block in self.mid_block:
            x = block(x)
        return self.out_block(x)


class AffineCoupling(nn.Module):
    def __init__(
        self,
        in_channels,
        channels,
        n_blocks=2,
        gin_channels=0,
    ):
        super(AffineCoupling, self).__init__()

        self.split_channels = in_channels // 2

        self.start = torch.nn.utils.weight_norm(
            nn.Conv1d(in_channels // 2, channels, 1)
        )
        self.net = MLP(
            in_channels=channels,
            out_channels=channels,
            n_blocks=n_blocks,
            gin_channels=gin_channels,
        )
        self.end = nn.Conv1d(channels, in_channels, 1)
        self.end.weight.data.zero_()
        self.end.bias.data.zero_()

    def forward(self, z, log_df_dz, g=None):
        z0, z1 = self.squeeze(z)

        params = self.start(z1)
        params = self.net(params, g=g)
        params = self.end(params)
        t = params[:, : self.split_channels, :]
        logs = params[:, self.split_channels :, :]  # noqa

        z0 = z0 * torch.exp(logs) + t
        log_df_dz += torch.sum(logs)

        z = self.unsqueeze(z0, z1)
        return z, log_df_dz

    def reverse(self, y, log_df_dz, g=None):
        y0, y1 = self.squeeze(y)

        params = self.start(y1)
        params = self.net(params, g=g)
        params = self.end(params)
        t = params[:, : self.split_channels, :]
        logs = params[:, self.split_channels :, :]  # noqa

        y0 = (y0 - t) * torch.exp(-logs)
        log_df_dz -= torch.sum(logs)

        y = self.unsqueeze(y0, y1)
        return y, log_df_dz

    @staticmethod
    def squeeze(z, dim=1):
        C = z.size(dim)
        z0, z1 = torch.split(z, C // 2, dim=dim)
        return z0, z1

    @staticmethod
    def unsqueeze(z0, z1, dim=1):
        z = torch.cat([z0, z1], dim=dim)
        return z
