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
from omegaconf import DictConfig


def split(df):
    unique_spk_ids = df["spk_id"].unique()

    split_idx = int(len(unique_spk_ids) * 0.98)
    trn_spk_ids = unique_spk_ids[:split_idx]
    val_spk_ids = unique_spk_ids[split_idx:]

    trn_df = df[df["spk_id"].isin(trn_spk_ids)]
    val_df = df[df["spk_id"].isin(val_spk_ids)]

    trn_df = trn_df.sort_values(by=["item_name"])
    val_df = val_df.sort_values(by=["item_name"])

    return trn_df, val_df


@hydra.main(version_base=None, config_path="conf/", config_name="preprocess")
def main(cfg: DictConfig):
    df_dir = Path(cfg.path.df_dir)
    filtered_df_dir = Path(cfg.path.filtered_df_dir)
    filtered_df_dir.mkdir(exist_ok=True)
    df = pd.read_csv(df_dir / "train.csv")
    data_df = pd.read_csv(cfg.path.data_csv_file)
    data_df = data_df[data_df["invalid"] == 0]
    print(df.shape, data_df.shape)
    df = df[df["item_name"].isin(data_df["item_name"])]
    print(df.shape)
    merged_df = pd.merge(
        df, data_df[["item_name", "style_prompt_key"]], on="item_name", how="left"
    )
    merged_df = merged_df.drop(columns="style_prompt_key_x")
    df = merged_df.rename(columns={"style_prompt_key_y": "style_prompt_key"})

    trn_df, val_df = split(df)
    trn_df.to_csv(filtered_df_dir / "trn.csv", index=False)
    val_df.to_csv(filtered_df_dir / "val.csv", index=False)
    print(trn_df.shape, val_df.shape)


if __name__ == "__main__":
    main()
