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

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from common import getLogger, load_libritts_spk_metadata
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Add style prompt tags and save the results to a CSV file",
    )
    parser.add_argument("in_dir", type=str, help="LibriTTS original data root")
    parser.add_argument("utt_stats", type=str, help="Utterance stats file")
    parser.add_argument(
        "style_prompt_candidates", type=str, help="Style prompt candidates"
    )
    parser.add_argument("--debug", action="store_true", help="Debug")

    parser.add_argument(
        "--out_filename",
        type=str,
        default="metadata_w_style_prompt_key.csv",
        help="Output filename",
    )

    return parser


def norm2label(val, level=3, labels=None):
    """Map a N(0, 1) normalized value to a discrete label

    Args:
        val (float): Normalized value
        level (int, optional): Number of levels. Defaults to 3.
        labels (List[str], optional): Labels. Defaults to None.

    Returns:
        str: Label
    """
    if labels is None:
        labels = ["low", "normal", "high"]
    if level == 3:
        # (-∞, -0.7]: low
        # (-0.7, 0.7]: normal
        # (0.7, ∞]: high

        if val < -0.7:
            return labels[0]
        elif val > 0.7:
            return labels[2]
        else:
            return labels[1]
    elif level == 5:
        # (-∞, -1.3): very low
        # [-1.3, -0.5): low
        # [-0.5, 0.5): normal
        # [0.5, 1.3): high
        # [1.3, ∞]: very high
        if val < -1.3:
            return f"very {labels[0]}"
        elif val >= -1.3 and val < -0.5:
            return f"{labels[0]}"
        elif val >= -0.5 and val < 0.5:
            return labels[1]
        elif val >= 0.5 and val < 1.3:
            return labels[2]
        else:
            return f"very {labels[2]}"


def speeking_speed_pseudo_label(spk2meta, scalers, val, spk, level=3):
    gender = spk2meta[spk]["gender"]
    scaler = scalers[gender]

    val_scaled = (val - scaler.mean_[0]) / scaler.scale_[0]

    return norm2label(val_scaled, level=level, labels=["slow", "normal", "fast"])


def pitch_pseudo_label(spk2meta, scalers, val, spk, level=3):
    gender = spk2meta[spk]["gender"]
    scaler = scalers[gender]

    val_scaled = (val - scaler.mean_[0]) / scaler.scale_[0]

    return norm2label(val_scaled, level=level, labels=["low", "normal", "high"])


def energy_pseudo_label(spk2meta, scalers, val, spk, level=3):
    gender = spk2meta[spk]["gender"]
    scaler = scalers[gender]

    val_scaled = (val - scaler.mean_[0]) / scaler.scale_[0]

    return norm2label(val_scaled, level=level, labels=["low", "normal", "high"])


def uttid2path(utt_id, data_root, spk2meta):
    (
        spk,
        subset2,
        _,
    ) = utt_id.split("_", 2)

    subset = spk2meta[spk]["subset"]

    path = data_root / subset / spk / subset2 / f"{utt_id}.wav"

    return path


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    in_dir = Path(args.in_dir)

    spk2meta = load_libritts_spk_metadata(debug=args.debug)
    logger = getLogger(
        verbose=100, filename="log/add_style_prompt.log", name="add_style_prompt"
    )

    logger.info("Loading utterance stats...")
    with open(args.utt_stats) as f:
        libritts_r_per_utt_metadata = yaml.load(f, Loader=yaml.SafeLoader)
    logger.info("Done")

    failed_utt_ids = []
    for path in Path("./external/libritts_r_failed_speech_restoration_examples/").glob(
        "*_list.txt"
    ):
        with open(path) as f:
            for line in f:
                if len(line.strip()) > 0:
                    utt_id = Path(line.strip()).stem
                    failed_utt_ids.append(utt_id)

    df_style_prompt = pd.read_csv(
        args.style_prompt_candidates,
        header=None,
        sep="|",
        names=["style_key", "prompt"],
    )
    style_prompt_dict = {}
    for _, row in df_style_prompt.iterrows():
        style_key, style_prompt = row.iloc[0], row.iloc[1]
        assert isinstance(style_prompt, str)
        style_prompt_dict[style_key] = list(
            map(lambda s: s.lower().strip(), style_prompt.split(";"))
        )

    # Loudness
    logger.info("Computing loudness stats...")
    all_loudness_mean = {"F": [], "M": []}
    for utt_id, meta in tqdm(libritts_r_per_utt_metadata.items()):
        if meta["invalid"] == 1:
            continue
        spk = utt_id.split("_")[0]
        gender = spk2meta[spk]["gender"]
        all_loudness_mean[gender].append(meta["raw_loudness_mean"])
    all_loudness_mean_norm = {"F": [], "M": []}
    scalers_loudness_mean = {"F": StandardScaler(), "M": StandardScaler()}
    for k, scaler in scalers_loudness_mean.items():
        arr = np.array(all_loudness_mean[k]).reshape(-1, 1)
        scaler.fit(arr)
        all_loudness_mean_norm[k] = scaler.transform(arr).reshape(-1)
    logger.info("Done")

    # log-F0
    logger.info("Computing log-F0 stats...")
    all_lf0_mean = {"F": [], "M": []}
    for utt_id, meta in tqdm(libritts_r_per_utt_metadata.items()):
        if meta["invalid"] == 1:
            continue

        spk = utt_id.split("_")[0]
        gender = spk2meta[spk]["gender"]
        all_lf0_mean[gender].append(meta["raw_lf0_mean"])
    all_lf0_mean_norm = {"F": [], "M": []}
    scalers_lf0_mean = {"F": StandardScaler(), "M": StandardScaler()}
    for k, scaler in scalers_lf0_mean.items():
        if meta["invalid"] == 1:
            continue

        arr = np.array(all_lf0_mean[k]).reshape(-1, 1)
        scaler.fit(arr)
        all_lf0_mean_norm[k] = scaler.transform(arr).reshape(-1)
    logger.info("Done")

    # speaking speed
    logger.info("Computing speaking speed stats...")
    all_speaking_rate = {"F": [], "M": []}
    for utt_id, meta in tqdm(libritts_r_per_utt_metadata.items()):
        if meta["invalid"] == 1:
            continue

        spk = utt_id.split("_")[0]
        gender = spk2meta[spk]["gender"]
        all_speaking_rate[gender].append(meta["raw_speaking_rate"])
    all_speaking_rate_norm = {"F": [], "M": []}
    scalers_speaking_rate = {"F": StandardScaler(), "M": StandardScaler()}
    for k, scaler in scalers_speaking_rate.items():
        if meta["invalid"] == 1:
            continue

        arr = np.array(all_speaking_rate[k]).reshape(-1, 1)
        scaler.fit(arr)
        all_speaking_rate_norm[k] = scaler.transform(arr).reshape(-1)
    logger.info("Done")

    rows = []
    for idx, (k, v) in tqdm(enumerate(libritts_r_per_utt_metadata.items())):
        if args.debug and idx > 100:
            break

        spk_id = k.split("_")[0]
        gender = spk2meta[spk_id]["gender"]

        # content_prompt
        text_path = Path(
            uttid2path(k, in_dir, spk2meta)
            .as_posix()
            .replace(".wav", ".normalized.txt")
        )
        assert text_path.exists()
        content_prompt = open(text_path).read().strip()

        # Peseudo labeling
        level = 5
        pitch = pitch_pseudo_label(
            spk2meta, scalers_lf0_mean, v["raw_lf0_mean"], spk_id, level=level
        )
        speaking_speed = speeking_speed_pseudo_label(
            spk2meta, scalers_speaking_rate, v["raw_speaking_rate"], spk_id, level=level
        )
        energy = energy_pseudo_label(
            spk2meta, scalers_loudness_mean, v["raw_loudness_mean"], spk_id, level=level
        )

        pitch3 = pitch.replace("very", "").strip()
        speaking_speed3 = speaking_speed.replace("very", "").strip()
        energy3 = energy.replace("very", "").strip()

        style_key = f"{gender}_p-{pitch3}_s-{speaking_speed3}_e-{energy3}"
        style_prompts = style_prompt_dict[style_key]

        rows.append(
            {
                "item_name": k,
                "spk_id": spk_id,
                "gender": gender,
                "pitch": pitch,
                "speaking_speed": speaking_speed,
                "energy": energy,
                "content_prompt": content_prompt,
                "style_prompt_key": style_key,
                # speaker name is unused but for compatibility with older scripts
                "raw_f0_mean": v["raw_f0_mean"],
                "raw_f0_scale": v["raw_f0_scale"],
                "raw_lf0_mean": v["raw_lf0_mean"],
                "raw_lf0_scale": v["raw_lf0_scale"],
                "raw_speaking_rate": v["raw_speaking_rate"],
                "raw_loudness_lufs": v["raw_loudness_lufs"],
                "raw_loudness_mean": v["raw_loudness_mean"],
                "raw_loudness_scale": v["raw_loudness_scale"],
                "invalid": v["invalid"],
            }
        )

    df_new = pd.DataFrame(rows)

    df_new.loc[df_new.content_prompt.str.startswith("-"), "invalid"] = 1
    df_new.loc[df_new.item_name.isin(failed_utt_ids), "invalid"] = 1

    df_new.to_csv(args.out_filename, index=False, header=True)
    logger.info(f"Saved to {args.out_filename}")
