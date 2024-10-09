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

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


class AllWithSpkPromptNormDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        file_path,
        data_root,
        feats_dir,
        mel_dir,
        to_mel,
        prompt_candidate_file,
        spk_prompt_candidate_file,
        use_spk_prompt=True,
        p_augment=0.0,
    ):
        self.data, self.lengths = self.read_data(file_path)
        self.data_root = Path(data_root)
        self.feats_dir = Path(feats_dir)
        self.mel_dir = Path(mel_dir)
        self.to_mel = to_mel
        self.prompt_candidate = self.read_prompt_candidate(prompt_candidate_file)
        self.spk_prompt_candidate = self.read_spk_prompt_candidate(
            spk_prompt_candidate_file
        )
        self.use_spk_prompt = use_spk_prompt
        self.p_augment = p_augment

        self.stats = OmegaConf.load(self.mel_dir / "stats.yaml")

        self.cache = dict()

    def read_data(self, file_path):
        use_col = [
            "spk_id",
            "item_name",
            "gender",
            "pitch",
            "speaking_speed",
            "energy",
            "style_prompt_key",
            "seq",
            "durations",
        ]
        df = pd.read_csv(file_path, usecols=use_col)
        data = df[use_col].values.tolist()
        lengths = list()
        for idx in tqdm(range(len(data)), total=len(df), desc="Loading durations..."):
            row = data[idx]
            durations = [int(d) for d in row[-1].split()]
            lengths.append(sum(durations))
        return data, lengths

    def read_prompt_candidate(self, filepath):
        df_style_prompt = pd.read_csv(
            filepath, header=None, sep="|", names=["style_key", "prompt"]
        )
        style_prompt_dict = {}
        for _, row in df_style_prompt.iterrows():
            style_key, style_prompt = row.iloc[0], row.iloc[1]
            assert isinstance(style_prompt, str)
            style_prompt_dict[style_key] = list(
                map(lambda s: s.lower().strip(), style_prompt.split(";"))
            )
        return style_prompt_dict

    def read_spk_prompt_candidate(self, filepath):
        df = pd.read_csv(filepath, sep="|", header=None, names=["spk", "words"])
        df["words"] = df["words"].map(lambda x: x.split(","))
        # dict(key: spk_id, value: words)
        spk_prompt_cand_dict = df.set_index("spk")["words"].to_dict()
        return spk_prompt_cand_dict

    def augment_style_prompt(self, style_prompt, pitch, speaking_speed, energy):
        adverbs = ["very", "extremely", "highly", "really", "particularly"]
        if random.random() > self.p_augment:
            return style_prompt
        if "very" in pitch:
            adverb = random.choice(adverbs)
            style_prompt = (
                style_prompt.replace(" high pitch ", f" {adverb} high pitch ")
                .replace(" high-pitched ", f" {adverb} high-pitched ")
                .replace(" high-pitched,", f" {adverb} high-pitched,")
            )
            style_prompt = (
                style_prompt.replace(" low pitch ", f" {adverb} low pitch ")
                .replace(" low-pitched ", f" {adverb} low-pitched ")
                .replace(" low-pitched,", f" {adverb} low-pitched,")
            )
        if "very" in speaking_speed:
            adverb = random.choice(adverbs)
            style_prompt = (
                style_prompt.replace(" fast ", f" {adverb} fast ")
                .replace(" quick ", f" {adverb} quick ")
                .replace(" quickly ", f" {adverb} quickly ")
                .replace(" quickly,", f" {adverb} quickly,")
            )
            style_prompt = (
                style_prompt.replace(" slow ", f" {adverb} slow ")
                .replace(" slowly ", f" {adverb} slowly ")
                .replace(" slowly,", f" {adverb} slowly,")
            )
            style_prompt = style_prompt.replace(
                " rapidly ", f" {adverb} rapidly "
            ).replace(" rapidly,", f" {adverb} rapidly,")
        if "very" in energy:
            adverb = random.choice(adverbs)
            style_prompt = (
                style_prompt.replace(" loud ", f" {adverb} loud ")
                .replace(" loudly ", f" {adverb} loudly ")
                .replace(" loudly,", f" {adverb} loudly,")
            )
            style_prompt = (
                style_prompt.replace(" quiet ", f" {adverb} quiet ")
                .replace(" quietly ", f" {adverb} quietly ")
                .replace(" quietly,", f" {adverb} quietly,")
            )
        return style_prompt

    def words2prompt(self, words, min_words=5):
        # random order
        # note: inplace operation
        random.shuffle(words)

        # use random ${n_words} words
        n_words = random.randint(min_words, len(words))
        words = words[:n_words]

        # select template
        templates = [
            "The speaker identity can be described as {words}.",
            "The voice characteristics can be described as {words}.",
            "The speaker's voice can be described as {words}.",
        ]
        template = random.choice(templates)

        prompt = template.format(words=", ".join(words))
        return prompt

    def add_spk_prompt(self, style_prompt, spk_id):
        if int(spk_id) in self.spk_prompt_candidate:
            spk_prompt_words = self.spk_prompt_candidate[int(spk_id)]
            spk_prompt = self.words2prompt(spk_prompt_words)
            style_prompt = random.choice(
                [
                    f"{style_prompt} {spk_prompt}",
                    f"{spk_prompt} {style_prompt}",
                    f"{spk_prompt}",
                    f"{style_prompt}",
                ]
            )
        return style_prompt

    def get_data(self, spk, utt_id, seq, durations):
        phonemes = torch.LongTensor([int(s) for s in seq.split()])
        durations = torch.FloatTensor([int(d) for d in durations.split()])

        mel = torch.FloatTensor(np.load(self.mel_dir / f"{spk}/{utt_id}.npy"))
        mel_norm = (mel - self.stats["mean"]) / self.stats["std"]
        log_cf0 = torch.FloatTensor(np.load(self.feats_dir / f"{spk}/cf0/{utt_id}.npy"))
        vuv = torch.FloatTensor(np.load(self.feats_dir / f"{spk}/vuv/{utt_id}.npy"))
        energy = mel.exp().pow(2).sum(dim=0).sqrt().view((-1,))
        assert mel.shape[-1] == log_cf0.shape[-1] == vuv.shape[-1] == energy.shape[-1]
        if mel.shape[-1] < durations.sum():
            durations[-1] = durations[-1] - 1
        assert mel.shape[-1] == durations.sum(), print(
            mel.shape[-1], log_cf0.shape[-1], durations.sum()
        )
        d = (spk, utt_id, phonemes, durations, mel_norm, log_cf0, vuv, energy)
        return d

    def __getitem__(self, idx):
        (
            spk_id,
            utt_id,
            gender,
            pitch,
            speaking_speed,
            energy,
            style_prompt_key,
            seq,
            durations,
        ) = self.data[idx]
        cache_key = utt_id
        style_prompt = random.choice(self.prompt_candidate[style_prompt_key])
        style_prompt = self.augment_style_prompt(
            style_prompt, pitch, speaking_speed, energy
        )
        style_prompt = f"{style_prompt}."  # add "."
        if self.use_spk_prompt:
            style_prompt = self.add_spk_prompt(style_prompt, spk_id)
        if cache_key in self.cache:
            (
                spk_id,
                utt_id,
                phonemes,
                duration,
                mel,
                log_cf0,
                vuv,
                energy,
            ) = self.cache[cache_key]
        else:
            (
                spk_id,
                utt_id,
                phonemes,
                duration,
                mel,
                log_cf0,
                vuv,
                energy,
            ) = self.get_data(spk_id, utt_id, seq, durations)
            # self.cache[cache_key] = d
        return (
            spk_id,
            utt_id,
            phonemes,
            duration,
            mel,
            log_cf0,
            vuv,
            energy,
            style_prompt,
        )

    def __len__(self):
        return len(self.data)

    def num_tokens(self, index):
        return self.lengths[index]

    def ordered_indices(self):
        indices = np.arange(len(self))
        indices = indices[np.argsort(np.array(self.lengths)[indices], kind="mergesort")]
        return indices
