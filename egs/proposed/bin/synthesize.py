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

import hydra
import pandas as pd
import torch
import torchaudio
from hydra.utils import instantiate
from omegaconf import OmegaConf
from promptttspp.utils import seed_everything
from promptttspp.utils.model import remove_weight_norm_
from promptttspp.vocoders import F0AwareBigVGAN
from scipy import signal
from tqdm import tqdm


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


def read_prompt_candidate(filepath):
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


def read_spk_prompt_candidate(filepath):
    df = pd.read_csv(filepath, sep="|", header=None, names=["spk", "words"])
    df["words"] = df["words"].map(lambda x: x.split(","))
    # dict(key: spk_id, value: words)
    spk_prompt_cand_dict = df.set_index("spk")["words"].to_dict()
    return spk_prompt_cand_dict


def add_spk_prompt(style_prompt, words):
    spk_prompt = f"The speaker identity can be described as {words}."
    prompt = f"{style_prompt}. {spk_prompt}"
    return prompt


@hydra.main(version_base=None, config_path="conf/", config_name="synthesize")
def main(cfg):
    data_root = Path(cfg.path.data_root)
    output_dir = Path(cfg.output_dir)

    seed_everything(cfg.train.seed)

    prompt_candidate = read_prompt_candidate(cfg.path.prompt_candidate_file)
    spk_prompt_candidate = read_spk_prompt_candidate(cfg.path.spk_prompt_candidate_file)
    mel_stats = OmegaConf.load(f"{cfg.path.mel_dir}/stats.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instantiate(cfg.model)
    model.load_state_dict(torch.load(cfg.ckpt_path, map_location="cpu")["model"])
    model = model.to(device).eval()
    model.apply(remove_weight_norm_)
    to_mel = instantiate(cfg.transforms).to(device).eval()

    vocoder = instantiate(cfg.vocoder)
    vocoder.load_state_dict(
        torch.load(cfg.vocoder_ckpt_path, map_location="cpu")["generator"]
    )
    vocoder = vocoder.to(device).eval()
    vocoder.apply(remove_weight_norm_)

    use_col = [
        "spk_id",
        "item_name",
        "gender",
        "pitch",
        "speaking_speed",
        "energy",
        "style_prompt",
        "style_prompt_key",
        "seq",
    ]
    df = pd.read_csv(cfg.label_file, usecols=use_col)
    data = df[use_col].values.tolist()

    for row in tqdm(data, total=len(data)):
        spk = row[0]
        utt_id = row[1]
        seq = row[-1]
        style_prompt_key = row[-2]
        style_prompt = prompt_candidate[style_prompt_key][0]
        if spk in spk_prompt_candidate:
            spk_prompt = spk_prompt_candidate[spk]
            words = ", ".join(spk_prompt)
            if cfg.use_spk_prompt:
                prompt = add_spk_prompt(style_prompt, words)
            else:
                prompt = style_prompt
        else:
            prompt = style_prompt

        spk_dir = output_dir / str(spk)

        ref_dir = spk_dir / "ref"
        ref_mel_dir = ref_dir / "mel"
        ref_plot_dir = ref_dir / "plot"
        ref_wav_dir = ref_dir / "wav"

        prompt_dir = spk_dir / "prompt"
        prompt_mel_dir = prompt_dir / "mel"
        prompt_plot_dir = prompt_dir / "plot"
        prompt_wav_dir = prompt_dir / "wav"

        dirs = [
            ref_mel_dir,
            ref_plot_dir,
            ref_wav_dir,
            prompt_mel_dir,
            prompt_plot_dir,
            prompt_wav_dir,
        ]
        [d.mkdir(parents=True, exist_ok=True) for d in dirs]

        label = torch.LongTensor([int(s) for s in seq.split()])[None, :]
        label = label.to(device)
        wav, _ = torchaudio.load(data_root / f"{spk}/wav24k/{utt_id}.wav")
        wav = wav.to(device)
        mel = to_mel(wav)
        mel = (mel - mel_stats["mean"]) / mel_stats["std"]

        is_f0_aware_vocoder = isinstance(vocoder, F0AwareBigVGAN)
        with torch.no_grad():
            if is_f0_aware_vocoder:
                dec, log_cf0, vuv = model.infer(
                    label, reference_mel=mel, return_f0=True
                )
                # NOTE: hard code for 10ms frame shift
                modfs = int(1.0 / (10 * 0.001))
                log_cf0 = lowpass_filter(log_cf0, modfs, cutoff=20)
                f0 = log_cf0.exp()
                f0[vuv < 0.5] = 0
                dec = dec * mel_stats["std"] + mel_stats["mean"]
                o_ref = vocoder(dec, f0).squeeze(1).cpu()
            else:
                dec = model.infer(label, reference_mel=mel)
                dec = dec * mel_stats["std"] + mel_stats["mean"]
                o_ref = vocoder(dec).squeeze(1).cpu()

        torchaudio.save(ref_wav_dir / f"{utt_id}.wav", o_ref, to_mel.sample_rate)

        with torch.no_grad():
            style_prompt = [prompt]
            if is_f0_aware_vocoder:
                dec, log_cf0, vuv = model.infer(
                    label, style_prompt=style_prompt, return_f0=True
                )
                # NOTE: hard code for 10ms frame shift
                modfs = int(1.0 / (10 * 0.001))
                log_cf0 = lowpass_filter(log_cf0, modfs, cutoff=20)
                f0 = log_cf0.exp()
                f0[vuv < 0.5] = 0
                dec = dec * mel_stats["std"] + mel_stats["mean"]
                o_prompt = vocoder(dec, f0).squeeze(1).cpu()
            else:
                dec = model.infer(label, style_prompt=style_prompt)
                dec = dec * mel_stats["std"] + mel_stats["mean"]
                o_prompt = vocoder(dec).squeeze(1).cpu()
        torchaudio.save(prompt_wav_dir / f"{utt_id}.wav", o_prompt, to_mel.sample_rate)

    with open(output_dir / "finish", "w") as f:
        f.write("finish")


if __name__ == "__main__":
    main()
