#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2019 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Positionwise feed forward layer definition."""

import torch


class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim, hidden_units, dropout_rate, activation=None):
        """Construct an PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        if activation is None:
            activation = torch.nn.ReLU()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.w_2 = torch.nn.Linear(hidden_units, idim)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.activation = activation

    def forward(self, x, mask=None):
        """Forward funciton.
        Args:
            x (torch.Tensor): Batch of input tensors (B, T, idim).
            mask (torch.Tensor): Batch of masks (B, T, 1).

        Returns:
            torch.Tensor: Batch of output tensors (B, T, idim).

        """
        if mask is None:
            mask = x.new_ones(x.size(0), x.size(1), 1)
        return self.w_2(self.dropout(self.activation(self.w_1(x) * mask))) * mask
