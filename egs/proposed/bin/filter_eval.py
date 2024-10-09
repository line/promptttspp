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
import torchaudio
from omegaconf import DictConfig
from tqdm import tqdm


@hydra.main(version_base=None, config_path="conf/", config_name="preprocess")
def main(cfg: DictConfig):
    data_root = Path(cfg.path.data_root)
    output_file = Path(cfg.path.filtered_eval_file)

    eval_df = pd.read_csv(cfg.path.eval_file)
    use_ids = []
    for i in tqdm(range(len(eval_df))):
        row = eval_df.iloc[i]
        spk_id = row["spk_id"]
        utt_id = row["item_name"]

        wav_file = data_root / f"{spk_id}/wav24k/{utt_id}.wav"
        assert wav_file.exists()
        wav, sr = torchaudio.load(wav_file)
        assert sr == 24000
        wav_sec = wav.shape[-1] / 24000
        if wav_sec < cfg.wav_min_sec:
            print(f"too short wav : {utt_id}")
        elif wav_sec > cfg.wav_max_sec:
            print(f"too long wav : {utt_id}")
        else:
            use_ids.append(utt_id)
    filtered_eval_df = eval_df[eval_df["item_name"].isin(use_ids)]
    print(f"Filtered : {len(eval_df)} => {len(filtered_eval_df)}")
    print("num files per speaker id")
    print(filtered_eval_df.groupby("spk_id").size())
    filtered_eval_df.to_csv(output_file, index=False)


if __name__ == "__main__":
    main()
