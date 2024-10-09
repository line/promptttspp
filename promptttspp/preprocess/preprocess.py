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
import pandas as pd
import soundfile as sf
from joblib import Parallel, delayed
from omegaconf import OmegaConf
from promptttspp.utils.joblib import tqdm_joblib

from .duration import process_textgrid
from .pitch import extract_pitch


def process_duration(spk, utt_id, wav, data_root, sample_rate, n_fft, hop_length):
    textgrid_path = data_root / str(spk) / "textgrid" / f"{utt_id}.TextGrid"
    result = process_textgrid(
        spk=spk,
        utt_id=utt_id,
        wav=wav,
        textgrid_path=textgrid_path,
        sample_rate=sample_rate,
        n_fft=n_fft,
        hop_length=hop_length,
    )
    return result


def process_pitch(wav, sample_rate, hop_length, f0_floor, f0_ceil):
    f0, cf0, vuv = extract_pitch(
        wav=wav,
        sample_rate=sample_rate,
        hop_length=hop_length,
        f0_floor=f0_floor,
        f0_ceil=f0_ceil,
    )
    return f0, cf0, vuv


def process_row(
    spk, utt_id, data_root, feats_dir, sample_rate, n_fft, hop_length, f0_floor, f0_ceil
):
    wav_path = data_root / str(spk) / "wav24k" / f"{utt_id}.wav"
    wav, sr = sf.read(wav_path)
    assert sr == 24000

    result = process_duration(
        spk,
        utt_id,
        wav,
        data_root,
        sample_rate,
        n_fft,
        hop_length,
    )
    if result is not None:
        seq, durations = result
    else:
        return None

    seq_s = " ".join([str(x) for x in seq])
    durations_s = " ".join([str(x) for x in durations.reshape(-1).tolist()])

    _, cf0, vuv = process_pitch(
        wav,
        sample_rate,
        hop_length,
        f0_floor,
        f0_ceil,
    )

    spk_out_dir = feats_dir / f"{spk}"
    cf0_dir = spk_out_dir / "cf0"
    vuv_dir = spk_out_dir / "vuv"
    [d.mkdir(parents=True, exist_ok=True) for d in [cf0_dir, vuv_dir]]

    np.save(cf0_dir / f"{utt_id}.npy", cf0)
    np.save(vuv_dir / f"{utt_id}.npy", vuv)

    return utt_id, seq_s, durations_s


def preprocess(cfg, debug=False):
    out_dir = Path(cfg.path.data_dir)

    if Path(out_dir / "finish").exists():
        return

    data_root = Path(cfg.path.data_root)
    feats_dir = Path(cfg.path.feats_dir)
    df_dir = Path(cfg.path.df_dir)

    f0_stats = OmegaConf.load(cfg.path.f0_stats_file)

    mel_cfg = cfg.transforms
    df = pd.read_csv(cfg.path.data_csv_file)
    df = df[df["invalid"] == 0]

    eval_ids = [int(s) for s in cfg.eval_ids]
    if debug:
        df = df.sample(n=100)

    spks = df["spk_id"].tolist()
    utt_ids = df["item_name"].tolist()

    with tqdm_joblib(len(df)):
        result = Parallel(n_jobs=cfg.n_jobs)(
            delayed(process_row)(
                spk=spk,
                utt_id=utt_id,
                data_root=data_root,
                feats_dir=feats_dir,
                sample_rate=mel_cfg.sample_rate,
                n_fft=mel_cfg.n_fft,
                hop_length=mel_cfg.hop_length,
                f0_floor=f0_stats[f"{spk}"]["f0_floor"],
                f0_ceil=f0_stats[f"{spk}"]["f0_ceil"],
            )
            for spk, utt_id in zip(spks, utt_ids)
        )
    result = [r for r in result if r is not None]
    result = list(sorted(result))

    df_dir.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame(result, columns=["item_name", "seq", "durations"])
    df = df[df["item_name"].isin(new_df["item_name"])]
    df = df.merge(new_df, on="item_name", how="left")

    train_df = df[~df["spk_id"].isin(eval_ids)]
    eval_df = df[df["spk_id"].isin(eval_ids)]

    df.to_csv(df_dir / "data.csv", index=False)
    train_df.to_csv(df_dir / "train.csv", index=False)
    eval_df.to_csv(df_dir / "eval.csv", index=False)

    with open(out_dir / "finish", "w") as f:
        f.write("finish\n")
