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
from promptttspp.layers.norm import LayerNorm
from promptttspp.modules.mdn import MDNLayer, mdn_get_most_probable_sigma_and_mu
from promptttspp.utils.model import generate_path, sequence_mask
from torch.cuda.amp import autocast


class PredictorLayer(nn.Module):
    def __init__(self, channels, kernel_size, dropout):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2)
        self.act = nn.ReLU()
        self.norm = LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        x = self.conv(x)
        x = self.act(x)
        x = self.norm(x)
        x = self.dropout(x)
        return x * mask


class Predictor(nn.Module):
    def __init__(
        self, channels, out_channels, kernel_size, dropout, num_layers, detach=False
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [PredictorLayer(channels, kernel_size, dropout) for _ in range(num_layers)]
        )
        self.out_layer = nn.Conv1d(channels, out_channels, 1)
        self.detach = detach

    def forward(self, x, mask):
        if self.detach:
            x = x.detach()
        for layer in self.layers:
            x = layer(x, mask)
        x = self.out_layer(x) * mask
        return x

    def infer(self, x, mask):
        return self(x, mask)


class MDNPredictor(nn.Module):
    def __init__(
        self,
        channels,
        out_channels,
        kernel_size,
        dropout,
        num_layers,
        num_gaussians=4,
        dim_wise=True,
        detach=False,
        disable_amp=False,
    ):
        super().__init__()
        self.layers = nn.ModuleList(
            [PredictorLayer(channels, kernel_size, dropout) for _ in range(num_layers)]
        )
        self.out_layer = MDNLayer(channels, out_channels, num_gaussians, dim_wise)
        self.detach = detach
        self.disable_amp = disable_amp

    def forward(self, x, mask):
        if self.detach:
            x = x.detach()
        for layer in self.layers:
            x = layer(x, mask)
        # NOTE: To stabilize training, disable autocast for MDN
        if self.disable_amp:
            x = x.float()
            with autocast(enabled=False):
                out = self.out_layer(x.transpose(-1, -2))
        else:
            out = self.out_layer(x.transpose(-1, -2))
        return out

    def infer(self, x, mask):
        out = self(x, mask)
        sigma, mu = mdn_get_most_probable_sigma_and_mu(*out)
        sigma_sq = sigma.pow(2).clamp_min(1e-14)
        log_duration = mu + sigma_sq / 2
        return log_duration.transpose(-1, -2)  # [B, 1, T]


class VarianceAdaptor(nn.Module):
    def __init__(
        self,
        duration_predictor,
        pitch_predictor,
        pitch_emb,
        energy_predictor=None,
        energy_emb=None,
        frame_prior_network=None,
    ):
        super().__init__()
        self.duration_predictor = duration_predictor

        self.pitch_predictor = pitch_predictor
        self.pitch_emb = pitch_emb

        self.energy_predictor = energy_predictor
        self.energy_emb = energy_emb

        self.frame_prior_network = frame_prior_network

    def forward(self, x, phone_mask, frame_mask, duration, log_cf0, vuv, energy):
        log_duration_pred = self.duration_predictor(x, phone_mask)

        path_mask = phone_mask.unsqueeze(-1) * frame_mask.unsqueeze(2)
        attn_path = generate_path(duration.squeeze(1), path_mask.squeeze(1))
        x = x @ attn_path

        if self.frame_prior_network is not None:
            x = self.frame_prior_network(x, frame_mask)

        log_cf0_pred, vuv_pred = self.pitch_predictor(x, frame_mask).split(1, dim=1)
        pitch_emb = self.pitch_emb(log_cf0) * frame_mask

        if self.energy_predictor is not None:
            energy_pred = self.energy_predictor(x, frame_mask)
            energy_emb = self.energy_emb(energy) * frame_mask
        else:
            energy_pred = None
            energy_emb = 0

        x = x + pitch_emb + energy_emb

        return x, log_duration_pred, log_cf0_pred, vuv_pred, energy_pred

    def infer(self, x, phone_mask, return_f0=False):
        log_duration = self.duration_predictor.infer(x, phone_mask)
        duration = log_duration.exp().round().clamp_min(1).long()

        frame_mask = torch.ones([1, 1, duration.sum()], dtype=x.dtype, device=x.device)
        path_mask = phone_mask.unsqueeze(-1) * frame_mask.unsqueeze(2)
        attn_path = generate_path(duration.squeeze(1), path_mask.squeeze(1))
        x = x @ attn_path

        if self.frame_prior_network is not None:
            x = self.frame_prior_network(x, frame_mask)

        log_cf0, vuv = self.pitch_predictor.infer(x, frame_mask).split(1, dim=1)
        pitch_emb = self.pitch_emb(log_cf0)

        if self.energy_predictor is not None:
            energy = self.energy_predictor.infer(x, frame_mask)
            energy_emb = self.energy_emb(energy)
        else:
            energy_emb = 0

        x = x + pitch_emb + energy_emb

        if return_f0:
            return x, frame_mask, log_cf0, vuv
        else:
            return x, frame_mask

    def infer_batch(self, x, phone_mask, return_f0=False):
        log_duration = self.duration_predictor.infer(x, phone_mask)
        duration = log_duration.exp().round().clamp_min(1).long()
        duration = duration * phone_mask

        frame_lengths = duration.squeeze(1).sum(dim=-1)
        frame_mask = sequence_mask(frame_lengths).unsqueeze(1).to(x.dtype)
        path_mask = phone_mask.unsqueeze(-1) * frame_mask.unsqueeze(2)
        attn_path = generate_path(duration.squeeze(1), path_mask.squeeze(1))
        x = x @ attn_path

        if self.frame_prior_network is not None:
            x = self.frame_prior_network(x, frame_mask)

        log_cf0, vuv = self.pitch_predictor.infer(x, frame_mask).split(1, dim=1)
        pitch_emb = self.pitch_emb(log_cf0) * frame_mask

        if self.energy_predictor is not None:
            energy = self.energy_predictor.infer(x, frame_mask)
            energy_emb = self.energy_emb(energy) * frame_mask
        else:
            energy_emb = 0

        x = x + pitch_emb + energy_emb

        if return_f0:
            return x, frame_mask, log_cf0, vuv
        else:
            return x, frame_mask
