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
from scipy.integrate import solve_ivp


class SDE:
    def __init__(self, beta_min=0.05, beta_max=20):
        self.beta_0 = beta_min
        self.beta_1 = beta_max

    def sde(self, x_t, mu, t):
        beta_t = self.beta_0 + (self.beta_1 - self.beta_0) * t
        drift = 0.5 * (mu - x_t) * beta_t[:, None, None, None]
        diffusion = torch.sqrt(beta_t)[:, None, None, None]
        return drift, diffusion

    def marginal_prob(self, x_0, mu, t):
        beta_int = self.beta_0 * t + 0.5 * (self.beta_1 - self.beta_0) * t ** 2
        c = torch.exp(-0.5 * beta_int)[:, None, None, None]
        mean = c * x_0 + (1 - c) * mu
        std = torch.sqrt(1.0 - torch.exp(-beta_int))[:, None, None, None]
        return mean, std

    def reverse_sde(self, score, x_t, mu, t):
        beta_t = self.beta_0 + (self.beta_1 - self.beta_0) * t
        drift = (0.5 * (mu - x_t) - score) * beta_t[:, None, None, None]
        diffusion = beta_t[:, None, None, None]
        return drift, diffusion

    def probability_flow(self, score, x_t, mu, t):
        beta_t = self.beta_0 + (self.beta_1 - self.beta_0) * t
        drift = 0.5 * (mu - x_t - score) * beta_t[:, None, None, None]
        diffusion = torch.zeros_like(drift)
        return drift, diffusion


class ScoreSDE(nn.Module):
    def __init__(self, mel_dim, denoise_fn, eps=1e-5, norm_scale=10):
        super().__init__()
        self.mel_dim = mel_dim
        self.eps = eps
        self.norm_scale = norm_scale

        self.denoise_fn = denoise_fn
        self.sde = SDE()

    @torch.no_grad()
    def forward(self, x, mu, mask, method="RK45"):
        device = x.device
        shape = x.shape
        b = x.shape[0]

        mu = mu / self.norm_scale

        def ode_func(t, x_t):
            x_t = torch.tensor(x_t, device=device, dtype=torch.float).reshape(shape)
            t = torch.full(size=(b,), fill_value=t, device=device, dtype=torch.float)
            score = self.denoise_fn(x_t, t, mu, mask)
            y, _ = self.sde.probability_flow(score, x_t, mu, t)
            y = y * mask
            return y.cpu().numpy().reshape((-1,)).astype(np.float64)

        res = solve_ivp(
            ode_func, (1.0, self.eps), x.reshape((-1,)).cpu().numpy(), method=method
        )
        x = torch.tensor(res.y[:, -1], device=device).reshape(shape)
        return x

    def compute_loss(self, x_0, mu, mask):
        t = torch.empty((x_0.shape[0],), device=x_0.device).uniform_(self.eps, 1)
        mean, std = self.sde.marginal_prob(x_0, mu, t)
        z = torch.randn_like(x_0) * mask
        x_t = (mean + std * z) * mask
        score = self.denoise_fn(x_t, t, mu, mask)
        loss = torch.sum((score * std + z) ** 2 * mask) / self.mel_dim / mask.sum()
        return loss
