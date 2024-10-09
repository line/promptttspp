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
import torch.nn.functional as F
from scipy import signal
from torch import nn


def remove_weight_norm_(m):
    try:
        nn.utils.remove_weight_norm(m)
    except ValueError:  # this module didn't have weight norm
        return


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1)


def generate_path(duration, mask):
    device = duration.device
    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, dim=1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)
    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, [0, 0, 1, 0, 0, 0])[:, :-1]
    path = path * mask
    return path


def positional_encoding(x, mask):
    _, C, T = x.size()
    pe = torch.zeros(T, C, device=x.device)
    position = torch.arange(0, T, dtype=torch.float).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, C, 2).float() * -(math.log(10000.0) / C))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0).transpose(1, 2)  # [1, C, T]
    pe = pe * mask
    return pe


def to_log_scale(x: torch.Tensor):
    x[x != 0] = torch.log(x[x != 0])
    return x


def pad_list(xs, pad_value, max_len=None):
    """Pad list of tensors

    Args:
        xs (list): List of input tensors
        pad_value (float): Value for padding
        max_len (int, optional): Maximum length. Defaults to None.

    Returns:
        torch.Tensor: Padded tensor
    """
    n_batch = len(xs)
    max_len = max(x.size(0) for x in xs) if max_len is None else max_len
    pad = xs[0].new(n_batch, max_len, *xs[0].size()[1:]).fill_(pad_value)

    for i in range(n_batch):
        pad[i, : xs[i].size(0)] = xs[i]

    return pad


def repeat_by_durations(xs, ds, input_lens=None, pad_value=0, max_len=None):
    """Repeat features by corresponding durations

    Args:
        x : T_in x C
        d : T_in

    Returns:
        torch.Tensor: T_out x C
    """
    ds = ds.squeeze(-1) if len(ds.shape) == 3 else ds
    xs = [torch.repeat_interleave(x, d, dim=0) for x, d in zip(xs, ds)]
    return pad_list(xs, pad_value, max_len)


def make_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """Make mask tensor containing indices of padded part.

    Args:
        lengths (LongTensor): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor. If set, masks will be the same shape
            as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor. See the example.

    Returns:
        Tensor: Mask tensor containing indices of padded part.
                dtype=torch.uint8 in PyTorch 1.2-
                dtype=torch.bool in PyTorch 1.2+ (including 1.2)
    """
    if length_dim == 0:
        raise ValueError("length_dim cannot be 0: {}".format(length_dim))

    bs = lengths.shape[0]
    if maxlen is None:
        if xs is None:
            maxlen = lengths.max()
        else:
            maxlen = xs.size(length_dim)

    seq_range = torch.arange(0, maxlen, dtype=torch.int64, device=lengths.device)
    seq_range_expand = seq_range.unsqueeze(0).expand(bs, maxlen)
    seq_length_expand = lengths
    if len(seq_length_expand.shape) == 1:
        seq_length_expand = seq_length_expand.unsqueeze(-1)
    mask = seq_range_expand >= seq_length_expand

    if xs is not None:
        assert xs.size(0) == bs, (xs.size(0), bs)

        if length_dim < 0:
            length_dim = xs.dim() + length_dim
        # ind = (:, None, ..., None, :, , None, ..., None)
        ind = tuple(
            slice(None) if i in (0, length_dim) else None for i in range(xs.dim())
        )
        mask = mask[ind].expand_as(xs).to(xs.device)
    return mask


def make_non_pad_mask(lengths, xs=None, length_dim=-1, maxlen=None):
    """Make mask tensor containing indices of non-padded part.

    Args:
        lengths (LongTensor or List): Batch of lengths (B,).
        xs (Tensor, optional): The reference tensor. If set, masks will be the same shape
            as this tensor.
        length_dim (int, optional): Dimension indicator of the above tensor. See the example.

    Returns:
        ByteTensor: mask tensor containing indices of padded part.
                    dtype=torch.uint8 in PyTorch 1.2-
                    dtype=torch.bool in PyTorch 1.2+ (including 1.2)
    """
    return ~make_pad_mask(lengths, xs, length_dim, maxlen)


def lowpass_filter(x, fs=100, cutoff=20, N=5):
    """Lowpass filter

    Args:
        x (array): input signal
        fs (int): sampling rate
        cutoff (int): cutoff frequency

    Returns:
        array: filtered signal
    """
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist
    Wn = [norm_cutoff]

    x_len = x.shape[-1]

    b, a = signal.butter(N, Wn, "lowpass")
    if x_len <= max(len(a), len(b)) * (N // 2 + 1):
        # NOTE: input signal is too short
        return x

    # NOTE: use zero-phase filter
    if isinstance(x, torch.Tensor):
        from torchaudio.functional import filtfilt

        a = torch.from_numpy(a).float().to(x.device)
        b = torch.from_numpy(b).float().to(x.device)
        y = filtfilt(x, a, b, clamp=False)
    else:
        y = signal.filtfilt(b, a, x)

    return y
