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
from inspect import isfunction

import torch
import torch.nn as nn
import torch.nn.functional as F


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),
        nn.Conv1d(dim, default(dim_out, dim), 3, padding=1),
    )


def Downsample(dim, dim_out=None):
    return nn.Conv1d(dim, default(dim_out, dim), kernel_size=3, stride=2, padding=1)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim, scale=1000):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = self.scale * time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class WeightStandardizedConv1d(nn.Conv1d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = weight.mean(dim=[1, 2], keepdim=True)
        std = weight.std(dim=[1, 2], keepdim=True)
        normalized_weight = (weight - mean) / (std + eps)

        return F.conv1d(
            x,
            normalized_weight,
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv1d(dim, dim_out, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, mask, scale_shift=None):
        x = self.proj(x * mask)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x * mask


class ResnetBlock(nn.Module):
    """https://arxiv.org/abs/1512.03385"""

    def __init__(self, dim, dim_out, *, time_emb_dim=None, groups=8):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2))
            if exists(time_emb_dim)
            else None
        )

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv1d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, mask, time_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(time_emb):
            time_emb = self.mlp(time_emb)
            time_emb = time_emb.unsqueeze(-1)
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, mask, scale_shift=scale_shift)
        h = self.block2(h, mask)
        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv1d(hidden_dim, dim, 1), nn.GroupNorm(1, dim))

    def forward(self, x):
        b, c, t = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = (x.view(b, self.heads, -1, t) for x in qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        # [B, H, C, T] x [B, H, T, C] => [B, H, C, C]
        context = k @ v.transpose(-1, -2)
        # [B, H, C, C] x [B, H, C, T] => [B, H, C, T]
        out = context.transpose(-1, -2) @ q
        out = out.view(b, -1, t)
        out = self.to_out(out)
        return out


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


class Unet1d(nn.Module):
    def __init__(
        self,
        in_channels,
        encoder_channels,
        out_channels,
        dim,
        dim_mults=(1, 2, 4),
        scale=1000,
    ):
        super().__init__()
        self.n_down = 2 ** (len(dim_mults) - 1)

        # determine dimensions
        self.in_channels = self.in_dim = in_channels

        self.init_conv = nn.Conv1d(in_channels, dim, 1)

        dims = [dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time embeddings
        time_dim = dim * 4

        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim, scale=scale),
            nn.Linear(dim, time_dim),
            nn.GELU(),
            nn.Linear(time_dim, time_dim),
        )

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            i, k, s, p = (encoder_channels, 3, 2 ** ind, 1)
            cond_layer = nn.Conv1d(i, dim_in, k, s, p)
            self.downs.append(
                nn.ModuleList(
                    [
                        cond_layer,
                        ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Downsample(dim_in, dim_out)
                        if not is_last
                        else nn.Conv1d(dim_in, dim_out, 3, padding=1),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        ResnetBlock(dim_out + dim_in, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Upsample(dim_out, dim_in)
                        if not is_last
                        else nn.Conv1d(dim_out, dim_in, 3, padding=1),
                    ]
                )
            )

        self.out_dim = out_channels

        self.final_res_block = ResnetBlock(dim * 2, dim, time_emb_dim=time_dim)
        self.final_conv = nn.Conv1d(dim, self.out_dim, 1)

    def forward(self, x, time, cond, mask=None):
        if mask is None:
            b, c, t = x.shape
            mask = torch.ones(b, 1, t, device=x.device)
        orig_t = x.shape[-1]
        pad_length = -x.shape[-1] % self.n_down
        x = F.pad(x, pad=[pad_length, 0], mode="reflect")
        cond = F.pad(cond, pad=[pad_length, 0], mode="reflect")
        mask = F.pad(mask, pad=[pad_length, 0], value=1.0)

        x = self.init_conv(x)
        residual = x

        t = self.time_mlp(time)

        h = []
        masks = [mask]
        for cond_layer, block1, block2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = x + cond_layer(cond) * mask_down
            x = block1(x, mask_down, t)
            h.append(x)

            x = block2(x, mask_down, t)
            x = attn(x)
            h.append(x)

            x = downsample(x * mask_down)
            masks.append(mask_down[..., ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for block1, block2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, mask_up, t)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, mask_up, t)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, residual), dim=1)

        x = self.final_res_block(x, mask, t)
        x = self.final_conv(x) * mask
        x = x[..., -orig_t:]
        return x
