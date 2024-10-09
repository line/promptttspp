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
from torchdiffeq import odeint


# Continuous Normalizing Flow with flow matching
# https://arxiv.org/abs/2210.02747
class CNF(nn.Module):
    def __init__(self, net, out_channels, cfg=5):
        super().__init__()
        self.net = net
        self.out_channels = out_channels
        self.cfg = cfg

    def forward(self, x1, cond, mask=None):
        B = x1.shape[0]
        device = x1.device

        t = torch.rand(size=(B,), device=device)
        x0 = torch.randn_like(x1)
        # linear intrerpolation
        xt = t[:, None, None] * x1 + (1 - t[:, None, None]) * x0
        ut = x1 - x0
        vt = self.net(xt, t, cond, mask=mask)
        return ut, vt

    @torch.no_grad()
    def sample(self, cond, sample_step, method, do_cfg=False):
        B, _, T = cond.shape
        device = cond.device
        x0 = torch.randn(B, self.out_channels, T, device=device)
        zero_cond = torch.zeros_like(cond)

        def f(t, y):
            if do_cfg:
                vt = (1 + self.cfg) * self.net(y, t, cond) - self.cfg * self.net(
                    y, t, zero_cond
                )
            else:
                t = t.unsqueeze(0)
                vt = self.net(y, t, cond)
            return vt

        ts = torch.linspace(1, 1e-5, steps=sample_step, device=device)
        x1 = odeint(f, x0, ts, method=method)[-1]
        return x1
