# Copyright 2020 Nagoya University (Tomoki Hayashi)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# Adapted from https://github.com/espnet/espnet

import math
from typing import Sequence

import torch
import torch.nn.functional as F
from promptttspp.modules.reference_encoder import ReferenceEncoder


class StyleEncoder(torch.nn.Module):
    """Style encoder.
    This module is style encoder introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.
    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017
    Args:
        idim (int, optional): Dimension of the input mel-spectrogram.
        gst_tokens (int, optional): The number of GST embeddings.
        gst_token_dim (int, optional): Dimension of each GST embedding.
        gst_heads (int, optional): The number of heads in GST multihead attention.
        conv_layers (int, optional): The number of conv layers in the reference encoder.
        conv_chans_list: (Sequence[int], optional):
            List of the number of channels of conv layers in the reference encoder.
        conv_kernel_size (int, optional):
            Kernel size of conv layers in the reference encoder.
        conv_stride (int, optional):
            Stride size of conv layers in the reference encoder.
        gru_layers (int, optional): The number of GRU layers in the reference encoder.
        gru_units (int, optional): The number of GRU units in the reference encoder.
    Todo:
        * Support manual weight specification in inference.
    """

    def __init__(
        self,
        idim: int = 80,
        gst_tokens: int = 10,
        gst_token_dim: int = 256,
        gst_heads: int = 4,
        conv_layers: int = 6,
        conv_chans_list: Sequence[int] = (32, 32, 64, 64, 128, 128),
        conv_kernel_size: int = 3,
        conv_stride: int = 2,
        gru_layers: int = 1,
        gru_units: int = 128,
    ):
        """Initialize global style encoder module."""
        super(StyleEncoder, self).__init__()

        self.ref_enc = ReferenceEncoder(
            idim=idim,
            conv_layers=conv_layers,
            conv_chans_list=conv_chans_list,
            conv_kernel_size=conv_kernel_size,
            conv_stride=conv_stride,
            gru_layers=gru_layers,
            gru_units=gru_units,
        )
        self.stl = StyleTokenLayer(
            ref_embed_dim=gru_units,
            gst_tokens=gst_tokens,
            gst_token_dim=gst_token_dim,
            gst_heads=gst_heads,
        )

    def forward(self, speech: torch.Tensor, in_lens=None) -> torch.Tensor:
        """Calculate forward propagation.
        Args:
            speech (Tensor): Batch of padded target features (B, odim. Lmax).
        Returns:
            Tensor: Style token embeddings (B, token_dim, 1).
        """
        ref_embs = self.ref_enc(speech, in_lens)  # [B, D, 1]
        style_embs = self.stl(ref_embs)

        return style_embs.unsqueeze(-1)


class StyleTokenLayer(torch.nn.Module):
    """Style token layer module.
    This module is style token layer introduced in `Style Tokens: Unsupervised Style
    Modeling, Control and Transfer in End-to-End Speech Synthesis`.
    .. _`Style Tokens: Unsupervised Style Modeling, Control and Transfer in End-to-End
        Speech Synthesis`: https://arxiv.org/abs/1803.09017
    Args:
        ref_embed_dim (int, optional): Dimension of the input reference embedding.
        gst_tokens (int, optional): The number of GST embeddings.
        gst_token_dim (int, optional): Dimension of each GST embedding.
        gst_heads (int, optional): The number of heads in GST multihead attention.
        dropout_rate (float, optional): Dropout rate in multi-head attention.
    """

    def __init__(
        self,
        ref_embed_dim: int = 128,
        gst_tokens: int = 10,
        gst_token_dim: int = 256,
        gst_heads: int = 4,
        dropout_rate: float = 0.0,
    ):
        """Initialize style token layer module."""
        super(StyleTokenLayer, self).__init__()

        gst_embs = torch.randn(gst_tokens, gst_token_dim // gst_heads)
        self.register_parameter("gst_embs", torch.nn.Parameter(gst_embs))
        self.mha = MultiHeadedAttention(
            q_dim=ref_embed_dim,
            k_dim=gst_token_dim // gst_heads,
            v_dim=gst_token_dim // gst_heads,
            n_head=gst_heads,
            n_feat=gst_token_dim,
            dropout_rate=dropout_rate,
        )

    def forward(self, ref_embs: torch.Tensor) -> torch.Tensor:
        """Calculate forward propagation.
        Args:
            ref_embs (Tensor): Reference embeddings (B, ref_embed_dim, 1).
        Returns:
            Tensor: Style token embeddings (B, gst_token_dim).
        """
        batch_size = ref_embs.size(0)
        # (num_tokens, token_dim) -> (batch_size, num_tokens, token_dim)
        gst_embs = torch.tanh(self.gst_embs).unsqueeze(0).expand(batch_size, -1, -1)
        # NOTE(kan-bayashi): Should we apply Tanh?
        ref_embs = ref_embs.transpose(-1, -2)  # (batch_size, 1, ref_embed_dim)
        style_embs = self.mha(ref_embs, gst_embs)

        return style_embs.squeeze(1)


class MultiHeadedAttention(torch.nn.Module):
    """Multi head attention module with different input dimension."""

    def __init__(self, q_dim, k_dim, v_dim, n_head, n_feat, dropout_rate=0.0):
        """Initialize multi head attention module."""
        # NOTE(kan-bayashi): Do not use super().__init__() here since we want to
        #   overwrite BaseMultiHeadedAttention.__init__() method.
        super().__init__()
        assert n_feat % n_head == 0
        # We assume d_v always equals d_k
        self.d_k = n_feat // n_head
        self.h = n_head
        self.linear_q = torch.nn.Linear(q_dim, n_feat)
        self.linear_k = torch.nn.Linear(k_dim, n_feat)
        self.linear_v = torch.nn.Linear(v_dim, n_feat)
        self.linear_out = torch.nn.Linear(n_feat, n_feat)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout_rate)

    def forward(self, ref_emb, gst_emb):
        B = ref_emb.shape[0]
        # [B, H, 1, D]
        q = self.linear_q(ref_emb).view(B, -1, self.h, self.d_k).transpose(1, 2)
        # [B, H, T, D]
        k = self.linear_k(gst_emb).view(B, -1, self.h, self.d_k).transpose(1, 2)
        v = self.linear_v(gst_emb).view(B, -1, self.h, self.d_k).transpose(1, 2)

        # [B, H, 1, T]
        score = (q @ k.transpose(-1, -2)) / math.sqrt(self.d_k * self.h)
        score = F.softmax(score, dim=-1)
        score = self.dropout(score)
        # [B, H, 1, T] x [B, H, T, D] => [B, H, 1, D]
        o = score @ v
        o = o.transpose(-1, -2).contiguous().view(B, 1, -1)
        o = self.linear_out(o)
        return o
