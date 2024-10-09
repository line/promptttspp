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
import torch.nn.functional as F
from promptttspp.layers.norm import LayerNorm


class MultiHeadAttention(nn.Module):
    def __init__(self, channels, n_heads, dropout):
        super().__init__()
        assert channels % n_heads == 0

        self.inter_channels = channels // n_heads
        self.n_heads = n_heads
        self.scale = 1 / math.sqrt(self.inter_channels)

        self.qkv = nn.Conv1d(channels, channels * 3, 1)
        self.out = nn.Conv1d(channels, channels, 1)

        self.drop = nn.Dropout(dropout)

    def forward(self, x, mask):
        B, C, T = x.shape
        H = self.n_heads

        qkv = self.qkv(x)
        # [B, 3 * C, T] => [B, 3, H, T, D]
        qkv = qkv.view(B, 3, H, self.inter_channels, T).transpose(-1, -2)
        # q, k, v \in [B, H, T, D]
        q, k, v = (x.squeeze(1) for x in qkv.split(1, dim=1))

        score = (q @ k.transpose(-1, -2)) * self.scale  # [B, H, T, T]
        if mask is not None:
            score = score.masked_fill(mask == 0, -1e4)
        score = F.softmax(score, dim=-1)
        score = self.drop(score)
        o = torch.matmul(score, v)  # [B, H, T, D]
        o = o.transpose(-1, -2).contiguous().view(B, C, T)
        o = self.out(o)
        return o


# Windowed Relative Positional Encoding is applied
class RelativeMultiHeadAttention(nn.Module):
    def __init__(self, channels, n_heads, dropout, window_size=4):
        super().__init__()
        assert channels % n_heads == 0

        self.inter_channels = channels // n_heads
        self.n_heads = n_heads
        self.window_size = window_size
        self.scale = math.sqrt(self.inter_channels)

        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.conv_o = nn.Conv1d(channels, channels, 1)
        self.drop = nn.Dropout(dropout)

        rel_stddev = self.inter_channels ** -0.5
        self.emb_rel_k = nn.Parameter(
            torch.randn(1, window_size * 2 + 1, self.inter_channels) * rel_stddev
        )
        self.emb_rel_v = nn.Parameter(
            torch.randn(1, window_size * 2 + 1, self.inter_channels) * rel_stddev
        )

        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        nn.init.xavier_uniform_(self.conv_v.weight)

    def forward(self, x, mask):
        q = self.conv_q(x)
        k = self.conv_k(x)
        v = self.conv_v(x)

        B, C, T = q.size()
        query = q.view(B, self.n_heads, self.inter_channels, T).transpose(2, 3)
        key = k.view(B, self.n_heads, self.inter_channels, T).transpose(2, 3)
        value = v.view(B, self.n_heads, self.inter_channels, T).transpose(2, 3)

        scores = torch.matmul(query / self.scale, key.transpose(-2, -1))

        pad_length = max(0, T - (self.window_size + 1))
        start = max(0, (self.window_size + 1) - T)
        end = start + 2 * T - 1

        pad_rel_emb = F.pad(self.emb_rel_k, [0, 0, pad_length, pad_length, 0, 0])
        k_emb = pad_rel_emb[:, start:end]

        rel_logits = torch.matmul(
            query / self.scale, k_emb.unsqueeze(0).transpose(-2, -1)
        )
        rel_logits = F.pad(rel_logits, [0, 1])
        rel_logits = rel_logits.view([B, self.n_heads, 2 * T * T])
        rel_logits = F.pad(rel_logits, [0, T - 1])
        scores_local = rel_logits.view([B, self.n_heads, T + 1, 2 * T - 1])[
            :, :, :T, T - 1 :
        ]

        scores = scores + scores_local
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = F.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)

        p_attn = F.pad(p_attn, [0, T - 1])
        p_attn = p_attn.view([B, self.n_heads, T * (2 * T - 1)])
        p_attn = F.pad(p_attn, [T, 0])
        relative_weights = p_attn.view([B, self.n_heads, T, 2 * T])[:, :, :, 1:]

        pad_rel_emb = F.pad(self.emb_rel_v, [0, 0, pad_length, pad_length, 0, 0])
        v_emb = pad_rel_emb[:, start:end]

        output = output + torch.matmul(relative_weights, v_emb.unsqueeze(0))

        x = output.transpose(2, 3).contiguous().view(B, C, T)

        x = self.conv_o(x)
        return x


class FFN(nn.Module):
    def __init__(self, channels, kernel_size, dropout, scale):
        super(FFN, self).__init__()
        self.conv1 = torch.nn.Conv1d(
            channels, channels * scale, kernel_size, padding=kernel_size // 2
        )
        self.conv2 = torch.nn.Conv1d(channels * scale, channels, 1)
        self.drop = torch.nn.Dropout(dropout)

    def forward(self, x, x_mask):
        x = self.conv1(x * x_mask)
        x = F.relu(x)
        x = self.drop(x)
        x = self.conv2(x * x_mask)
        return x * x_mask


class AttentionLayer(nn.Module):
    def __init__(self, channels, num_head, dropout):
        super().__init__()
        self.attention_layer = MultiHeadAttention(channels, num_head, dropout)
        self.norm = LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        y = self.attention_layer(x, attn_mask)
        y = self.dropout(y)
        x = self.norm(x + y)
        return x


class RelativeAttentionLayer(nn.Module):
    def __init__(self, channels, num_head, dropout, window_size):
        super().__init__()
        self.attention_layer = RelativeMultiHeadAttention(
            channels, num_head, dropout, window_size
        )
        self.norm = LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask):
        y = self.attention_layer(x, attn_mask)
        y = self.dropout(y)
        x = self.norm(x + y)
        return x


class FFNLayer(nn.Module):
    def __init__(self, channels, kernel_size, dropout, scale):
        super().__init__()
        self.ffn = FFN(channels, kernel_size, dropout, scale)
        self.norm = LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        y = self.ffn(x, mask)
        y = self.dropout(y)
        x = self.norm(x + y)
        return x * mask


class TransformerLayer(nn.Module):
    def __init__(
        self,
        channels,
        num_head,
        kernel_size,
        dropout,
        scale,
        window_size=None,
        use_rel=False,
    ):
        super().__init__()
        if use_rel:
            self.attention = RelativeAttentionLayer(
                channels, num_head, dropout, window_size
            )
        else:
            self.attention = AttentionLayer(channels, num_head, dropout)
        self.ffn = FFNLayer(channels, kernel_size, dropout, scale)

    def forward(self, x, mask, attn_mask):
        x = self.attention(x, attn_mask)
        x = self.ffn(x, mask)
        return x


class Transformer(nn.Module):
    def __init__(
        self,
        channels,
        num_head,
        num_layers,
        kernel_size,
        dropout,
        scale=4,
        window_size=None,
        use_rel=False,
    ):
        super().__init__()
        self.channels = channels

        self.layers = nn.ModuleList(
            [
                TransformerLayer(
                    channels,
                    num_head,
                    kernel_size,
                    dropout,
                    scale,
                    window_size,
                    use_rel,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, x, mask, g=None):
        # [B, 1, 1, T] x [B, 1, T, 1] => [B, 1, T, T]
        attn_mask = mask.unsqueeze(2) * mask.unsqueeze(-1)
        for layer in self.layers:
            if g is not None:
                x = x + g
            x = layer(x, mask, attn_mask)
        return x
