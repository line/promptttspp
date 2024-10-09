#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2020 Johns Hopkins University (Shinji Watanabe)
#                Northwestern Polytechnical University (Pengcheng Guo)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""ConvolutionModule definition."""

from torch import nn


class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model.

    Args:
        channels (int): The number of channels of conv layers.
        kernel_size (int): Kernel size of conv layers.

    """

    def __init__(self, channels, kernel_size, activation=None, bias=True):
        """Construct an ConvolutionModule object."""
        super(ConvolutionModule, self).__init__()
        # kernel_size should be a odd number for 'SAME' padding
        assert (kernel_size - 1) % 2 == 0

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=(kernel_size - 1) // 2,
            groups=channels,
            bias=bias,
        )
        self.norm = nn.BatchNorm1d(channels)
        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        if activation is None:
            activation = nn.ReLU()
        self.activation = activation

    def forward(self, x, mask=None):
        """Compute convolution module.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask (torch.Tensor): Mask tensor (#batch, time, 1).

        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).

        """
        if mask is None:
            mask = x.new_ones(x.size(0), x.size(1), 1)
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)
        mask = mask.transpose(1, 2)

        # GLU mechanism
        x = self.pointwise_conv1(x) * mask  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x) * mask
        x = self.activation(self.norm(x))

        x = self.pointwise_conv2(x) * mask

        return x.transpose(1, 2)
