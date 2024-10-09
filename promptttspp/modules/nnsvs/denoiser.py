import math
import random
from math import sqrt

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


class CondLayerNorm(nn.Module):
    """Conditional layer normalization

    Args:
        in_dim (int): Input dimension
        cond_dim (int): Conditional dimension
        eps (float): Epsilon value for numerical stability
    """

    def __init__(self, in_dim, cond_dim, eps=1e-5):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, in_dim)
        self.beta = nn.Linear(cond_dim, in_dim)
        self.eps = eps
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.constant_(self.gamma.weight, 0.0)
        torch.nn.init.constant_(self.gamma.bias, 0.0)
        torch.nn.init.constant_(self.beta.weight, 0.0)
        torch.nn.init.constant_(self.beta.bias, 0.0)

    def forward(self, x, g):
        """Forward

        Args:
            x (Tensor): Input features (B, C, T)
            g (Tensor): Conditional speaker embedding (B, g_dim) or (B, g_dim, T)

        Returns:
            Tensor: (B, C, T)
        """
        # Part 1. generate parameter-free normalized activations
        # (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        # NOTE: normalize hidden features for each time step
        mean = x.mean(dim=-1, keepdim=True)
        var = ((x - mean) ** 2).mean(dim=-1, keepdim=True)
        std = (var + self.eps).sqrt()
        normalized = (x - mean) / std
        normalized = normalized.transpose(1, 2)

        # Part 2. produce scaling and bias conditioned on auxiliay features
        gamma = self.gamma(g)
        beta = self.beta(g)

        is_time_varying_g = g.dim() == 3
        if is_time_varying_g:
            # (B, g_dim, T) -> (B, T, g_dim)
            gamma = gamma.transpose(1, 2)
            beta = beta.transpose(1, 2)
        else:
            # (B, g_dim) -> (B, g_dim, 1)
            gamma = gamma.unsqueeze(-1)
            beta = beta.unsqueeze(-1)

        # apply scale and bias
        out = normalized * (1 + gamma) + beta

        return out


class Mish(nn.Module):
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Conv1d(*args, **kwargs):
    layer = nn.Conv1d(*args, **kwargs)
    nn.init.kaiming_normal_(layer.weight)
    return layer


@torch.jit.script
def silu(x):
    return x * torch.sigmoid(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        encoder_hidden,
        residual_channels,
        dilation,
        gin_channels=0,
        g_proj_dim=128,
        cond_norm=False,
    ):
        super().__init__()
        self.dilated_conv = Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
        )
        self.diffusion_projection = nn.Linear(
            residual_channels + g_proj_dim
            if (gin_channels > 0 and not cond_norm)
            else residual_channels,
            residual_channels,
        )
        self.conditioner_projection = Conv1d(encoder_hidden, 2 * residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, 2 * residual_channels, 1)

        self.cond_norm = cond_norm
        if gin_channels > 0 and self.cond_norm:
            self.norm = CondLayerNorm(residual_channels, g_proj_dim)

    def forward(self, x, conditioner, diffusion_step, g=None):
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        conditioner = self.conditioner_projection(conditioner)

        if self.cond_norm:
            y = self.norm(x, g)
        else:
            y = x

        y = y + diffusion_step

        y = self.dilated_conv(y) + conditioner

        gate, filter = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter)

        y = self.output_projection(y)
        residual, skip = torch.chunk(y, 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class DiffNet(nn.Module):
    def __init__(
        self,
        in_dim=80,
        encoder_hidden_dim=256,
        residual_layers=20,
        residual_channels=256,
        dilation_cycle_length=4,
        scaled_tanh=False,
        gin_channels=0,
        g_proj_dim=128,
        g_dropout=0.0,
        cond_norm=False,
        time_varying_emb=False,
    ):
        super().__init__()
        self.in_dim = in_dim
        self.scaled_tanh = scaled_tanh
        self.time_varying_emb = time_varying_emb

        self.input_projection = Conv1d(in_dim, residual_channels, 1)
        self.diffusion_embedding = SinusoidalPosEmb(residual_channels)

        # Classifier-free conditioning
        self.gin_channels = gin_channels
        self.g_dropout = g_dropout
        self.cond_norm = cond_norm
        if self.gin_channels > 0:
            self.g_projection = nn.Linear(gin_channels, g_proj_dim)
            self.null_embedding = nn.Parameter(torch.randn(gin_channels))

        dim = residual_channels
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4), Mish(), nn.Linear(dim * 4, dim)
        )
        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    encoder_hidden_dim,
                    residual_channels,
                    2 ** (i % dilation_cycle_length),
                    gin_channels=gin_channels,
                    g_proj_dim=g_proj_dim,
                    cond_norm=cond_norm,
                )
                for i in range(residual_layers)
            ]
        )
        self.skip_projection = Conv1d(residual_channels, residual_channels, 1)
        self.output_projection = Conv1d(residual_channels, in_dim, 1)
        nn.init.zeros_(self.output_projection.weight)

    def requires_g(self):
        return self.gin_channels > 0

    def forward(self, spec, diffusion_step, cond, g=None):
        """

        :param spec: [B, 1, M, T]
        :param diffusion_step: [B, 1]
        :param cond: [B, M, T]
        :param g: [B, 1, C]
        :return:
        """
        x = spec[:, 0]
        x = self.input_projection(x)  # x [B, residual_channel, T]

        x = F.relu(x)
        diffusion_step = self.diffusion_embedding(diffusion_step)
        diffusion_step = self.mlp(diffusion_step)

        # Classifier-free conditioning
        if self.gin_channels > 0:
            always_use_null_embedding = self.g_dropout >= 1.0
            if self.time_varying_emb:
                assert self.cond_norm

                # (B, T, C)
                if always_use_null_embedding:
                    # add batch and time axis to null embedding
                    g_proj_inp = F.normalize(self.null_embedding, dim=-1).expand_as(g)
                elif self.training and (random.random() < self.g_dropout):
                    g_proj_inp = F.normalize(self.null_embedding, dim=-1).expand_as(g)
                else:
                    g_proj_inp = F.normalize(g, dim=-1)

                if not self.training:
                    null_indices = g.abs().sum(-1) == 0
                    if null_indices.any():
                        g_proj_inp = g_proj_inp.clone()
                        g_proj_inp[null_indices] = F.normalize(
                            self.null_embedding.to(g.dtype), dim=-1
                        ).expand_as(g[null_indices])

                g_proj = self.g_projection(g_proj_inp)
            else:
                # (B, 1, C) -> (B, C)
                g = g.squeeze(1) if g.dim() == 3 else g

                if always_use_null_embedding:
                    # add batch axis to null embedding
                    g_proj_inp = F.normalize(self.null_embedding, dim=-1).expand_as(g)
                elif self.training and (random.random() < self.g_dropout):
                    g_proj_inp = F.normalize(self.null_embedding, dim=-1).expand_as(g)
                else:
                    g_proj_inp = F.normalize(g, dim=-1)

                # Inference only: replace zero vector with null embedding
                if not self.training:
                    null_indices = g.abs().sum(-1) == 0
                    if null_indices.any():
                        g_proj_inp = g_proj_inp.clone()
                        g_proj_inp[null_indices] = F.normalize(
                            self.null_embedding.to(g.dtype), dim=-1
                        ).expand_as(g[null_indices])

                g_proj = self.g_projection(g_proj_inp)

                if not self.cond_norm:
                    assert diffusion_step.shape == g_proj.shape
                    # (B, C*2)
                    # Concat diffusion step and speaker embedding
                    # Ref: sec. 3.1 in https://arxiv.org/abs/2205.15370
                    # NOTE: according to OpenAI's code, we may consider simple addition
                    diffusion_step = torch.cat([diffusion_step, g_proj], dim=-1)
        else:
            g_proj = None

        skip = []
        for _, layer in enumerate(self.residual_layers):
            x, skip_connection = layer(x, cond, diffusion_step, g=g_proj)
            skip.append(skip_connection)

        x = torch.sum(torch.stack(skip), dim=0) / sqrt(len(self.residual_layers))
        x = self.skip_projection(x)
        x = F.relu(x)
        x = self.output_projection(x)  # [B, 80, T]

        # NOTE: this should be only enabled for residual F0 modeling
        # with objective=pred_x0
        if self.scaled_tanh:
            residual_f0_max_cent = 600
            max_lf0_ratio = residual_f0_max_cent * np.log(2) / 1200
            x = max_lf0_ratio * torch.tanh(x)

        return x[:, None, :, :]
