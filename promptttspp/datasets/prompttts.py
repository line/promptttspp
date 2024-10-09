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

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from tqdm import tqdm


class PromptTTSDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path,
        data_root,
        feats_dir,
        to_mel,
        split="|",
    ):
        self.data = self.read_data(file_path, split)
        self.data_root = Path(data_root)
        self.feats_dir = Path(feats_dir)
        self.to_mel = to_mel

        self.cache = dict()
        self.lengths = list()
        self.load_data()

    def read_data(self, file_path, split):
        with open(file_path) as f:
            lines = f.readlines()
        data = list()
        for line in lines:
            spk, utt_id, _, seq, duration, prompt = line.strip().split(split)
            seq = [int(x) for x in seq.split()]
            duration = [int(x) for x in duration.split()]
            data.append((spk, utt_id, seq, duration, prompt))
        return data

    def get_row(self, idx):
        spk, utt_id, seq, duration, prompt = self.data[idx]
        key = f"{spk}_{utt_id}"

        phonemes = torch.LongTensor(seq)
        duration = torch.FloatTensor(duration)

        wav, _ = torchaudio.load(self.data_root / f"{spk}/wav24k/{utt_id}.wav")
        mel = self.to_mel(wav).squeeze(0)

        log_cf0 = torch.FloatTensor(np.load(self.feats_dir / f"{spk}/cf0/{utt_id}.npy"))
        vuv = torch.FloatTensor(np.load(self.feats_dir / f"{spk}/vuv/{utt_id}.npy"))
        energy = mel.exp().pow(2).sum(dim=0).sqrt().view((-1,))
        assert mel.shape[-1] == log_cf0.shape[-1] == vuv.shape[-1] == energy.shape[-1]

        if mel.shape[-1] < duration.sum():
            diff = int(duration.sum()) - mel.shape[-1]
            mel, log_cf0, vuv, energy = (
                F.pad(x, [0, diff], mode="reflect").squeeze()
                for x in [
                    mel,
                    log_cf0.unsqueeze(0),
                    vuv.unsqueeze(0),
                    energy.unsqueeze(0),
                ]
            )
        assert mel.shape[-1] == duration.sum(), print(
            mel.shape[-1], log_cf0.shape[-1], duration.sum()
        )
        d = (spk, utt_id, phonemes, duration, mel, log_cf0, vuv, energy, prompt)
        length = mel.shape[-1]
        return key, d, length

    def load_data(self):
        bar = tqdm(
            range(len(self)),
            total=len(self),
            dynamic_ncols=True,
            desc="Loading Dataset...",
        )
        for idx in bar:
            key, row, length = self.get_row(idx)
            self.cache[key] = row
            self.lengths.append(length)

    def __getitem__(self, idx):
        spk, utt_id, *_ = self.data[idx]
        key = f"{spk}_{utt_id}"

        return self.cache[key]

    def __len__(self):
        return len(self.data)

    def num_tokens(self, index):
        return self.lengths[index]

    def ordered_indices(self):
        indices = np.arange(len(self))
        indices = indices[np.argsort(np.array(self.lengths)[indices], kind="mergesort")]
        return indices


class PromptTTSCollator:
    def __call__(self, batch):
        (
            spks,
            utt_ids,
            phonemes,
            durations,
            mels,
            log_cf0s,
            vuvs,
            energies,
            prompts,
        ) = tuple(zip(*batch))

        B = len(spks)
        phone_lengths = [x.size(-1) for x in phonemes]
        frame_lengths = [x.size(-1) for x in mels]

        phone_max_length = max(phone_lengths)
        frame_max_length = max(frame_lengths)
        mel_dim = mels[0].size(0)

        phone_pad = torch.zeros(size=(B, phone_max_length), dtype=torch.long)
        dur_pad = torch.zeros(size=(B, 1, phone_max_length), dtype=torch.float)
        mel_pad = torch.zeros(size=(B, mel_dim, frame_max_length), dtype=torch.float)
        log_cf0_pad = torch.zeros(size=(B, 1, frame_max_length), dtype=torch.float)
        vuv_pad = torch.zeros(size=(B, 1, frame_max_length), dtype=torch.float)
        energy_pad = torch.zeros(size=(B, 1, frame_max_length), dtype=torch.float)
        for i in range(B):
            p_l, f_l = phone_lengths[i], frame_lengths[i]
            phone_pad[i, :p_l] = phonemes[i]
            dur_pad[i, :, :p_l] = durations[i]
            mel_pad[i, :, :f_l] = mels[i]
            log_cf0_pad[i, :, :f_l] = log_cf0s[i]
            vuv_pad[i, :, :f_l] = vuvs[i]
            energy_pad[i, :, :f_l] = energies[i]

        phone_lengths = torch.LongTensor(phone_lengths)
        frame_lengths = torch.LongTensor(frame_lengths)

        return (
            spks,
            utt_ids,
            phone_pad,
            dur_pad,
            phone_lengths,
            mel_pad,
            log_cf0_pad,
            vuv_pad,
            energy_pad,
            frame_lengths,
            prompts,
        )
