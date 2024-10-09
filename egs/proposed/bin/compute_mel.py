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
import numpy as np
import pandas as pd
import torch
import torchaudio
from hydra.utils import instantiate
from joblib import Parallel, delayed
from omegaconf import DictConfig, OmegaConf
from promptttspp.utils.joblib import tqdm_joblib


@hydra.main(version_base=None, config_path="conf/", config_name="preprocess")
def main(cfg: DictConfig):
    data_root = Path(cfg.path.data_root)
    mel_dir = Path(cfg.path.mel_dir)

    if (mel_dir / "finish").exists():
        print("Already finished")
        return

    df = pd.read_csv(cfg.path.data_file)
    # NOTE: use cpu for multi-processing
    device = torch.device("cpu")
    to_mel = instantiate(cfg.transforms).to(device)

    def process(row):
        spk_id, utt_id = row["spk_id"], row["item_name"]
        wav_path = data_root / f"{spk_id}/wav24k/{utt_id}.wav"
        wav, _ = torchaudio.load(wav_path)
        wav = wav.to(device)
        mel = to_mel(wav).squeeze().cpu()

        spk_dir = mel_dir / f"{spk_id}"
        spk_dir.mkdir(parents=True, exist_ok=True)
        np.save(spk_dir / f"{utt_id}.npy", mel.numpy())
        return mel

    with tqdm_joblib(len(df)):
        mels = Parallel(n_jobs=cfg.n_jobs)(
            delayed(process)(df.iloc[idx]) for idx in range(len(df))
        )

    mels = torch.cat(mels, dim=1)
    stats = {
        "min": float(mels.min()),
        "max": float(mels.max()),
        "mean": float(mels.mean()),
        "std": float(mels.std()),
        "var": float(mels.var()),
    }
    conf = OmegaConf.create(stats)
    OmegaConf.save(conf, mel_dir / "stats.yaml")
    with open(mel_dir / "finish", "w") as f:
        f.write("finish")


if __name__ == "__main__":
    main()
