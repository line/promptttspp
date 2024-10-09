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
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import librosa
import numpy as np
import pyloudnorm as pyln
import pyworld
import soundfile as sf
import syllables
import yaml
from common import getLogger, load_libritts_spk_metadata
from promptttspp.utils.textgrid import read_textgrid
from tqdm.auto import tqdm


def compute_speaking_rate(textgrid_path):
    labels = read_textgrid(textgrid_path.as_posix(), "words")
    if len(labels) < 2:
        return -1
    assert len(labels) >= 2

    start_time = None
    end_time = 0
    num_syllables = 0

    sil_dur = 0
    for label in labels:
        if start_time is None and len(label.name) > 0:
            start_time = label.start
        if len(label.name) > 0:
            num_syllables += syllables.estimate(label.name)
        else:
            sil_dur += label.stop - label.start
    end_time = labels[-1].stop

    try:
        rate = num_syllables / (end_time - start_time - sil_dur)
    except ZeroDivisionError:
        print(f"warn: {textgrid_path}. {end_time}, {start_time}, {sil_dur}")
        rate = -1
    if rate < 0:
        print(f"warn: {textgrid_path}. {end_time}, {start_time}, {sil_dur}")
        rate = -1

    return round(rate, 2)


def loudness_extract(audio, sampling_rate, n_fft=1024, hop_length=240):
    stft = librosa.stft(audio, n_fft=n_fft, hop_length=hop_length) + 1e-7
    power_spectrum = np.abs(stft) ** 2
    bins = librosa.fft_frequencies(sr=sampling_rate, n_fft=n_fft)
    bins[0] += 1e-5  # To prevent zero division
    loudness = librosa.perceptual_weighting(power_spectrum, bins)
    loudness = librosa.db_to_power(loudness)
    loudness = np.log(np.mean(loudness, axis=0) + 1e-5)
    return loudness


def get_parser():
    parser = argparse.ArgumentParser(
        description="Compute per-utterance statistics",
    )
    parser.add_argument(
        "in_dir", type=str, help="LibriTTS per-speaker restructured data root"
    )
    parser.add_argument("f0_stats", type=str, help="f0 stats")
    parser.add_argument(
        "--out_filename",
        type=str,
        default="libritts_r_metadata.yaml",
        help="Output filename",
    )
    parser.add_argument("--num_jobs", type=int, default=8, help="Number of jobs")
    parser.add_argument("--debug", action="store_true", help="Debug")

    return parser


def process_utterance(logger, wav_file, textgrid_file, f0_stats):
    utt_id = wav_file.stem
    spk = utt_id.split("_")[0]
    x, sr = sf.read(wav_file)
    hop_length = int(sr * 0.010)

    invalid = 0

    # Loudness in LUFS
    block_size = min(0.4, len(x) / sr - 0.01)
    meter = pyln.Meter(sr, block_size=block_size)
    loudness_lufs = round(meter.integrated_loudness(x), 2)

    # Per-frame loudness
    frame_loudness = loudness_extract(x, sr, n_fft=1024, hop_length=hop_length)

    # F0
    if spk in f0_stats:
        f0_floor = f0_stats[spk]["f0_floor"]
        f0_ceil = f0_stats[spk]["f0_ceil"]
    else:
        f0_floor = 70
        f0_ceil = 800
        logger.warning(f"Using default f0_floor={f0_floor}, f0_ceil={f0_ceil}")

    f0, timeaxis = pyworld.dio(
        x, sr, frame_period=5, f0_floor=f0_floor, f0_ceil=f0_ceil
    )
    f0 = pyworld.stonemask(x, f0, timeaxis, sr)
    f0_v = f0[f0 > 0]
    lf0_v = np.log(f0_v)

    # e.g. 14_212_000011_000004, 14_212_000011_000009, 14_212_000018_000001
    if len(f0_v) == 0:
        logger.warning(f"{utt_id} has no f0")
        f0_mean = 0
        f0_scale = 1.0
        invalid = 1
        lf0_mean = 0
        lf0_scale = 1.0
    else:
        lf0_mean = np.mean(lf0_v)
        lf0_scale = np.std(lf0_v)
        f0_mean = np.mean(f0_v)
        f0_scale = np.std(f0_v)

    try:
        speaking_rate = compute_speaking_rate(textgrid_file)
        if speaking_rate < 0:
            invalid = 1
    except RuntimeError:
        logger.warning(f"{utt_id} has no valid speaking rate")
        speaking_rate = 0
        invalid = 1

    out = {
        "raw_loudness_lufs": round(float(loudness_lufs), 2),
        "raw_loudness_mean": round(float(frame_loudness.mean()), 2),
        "raw_loudness_scale": round(float(frame_loudness.std()), 2),
        "raw_f0_mean": round(float(f0_mean), 2),
        "raw_f0_scale": round(float(f0_scale), 2),
        "raw_lf0_mean": round(float(lf0_mean), 2),
        "raw_lf0_scale": round(float(lf0_scale), 2),
        "raw_speaking_rate": round(float(speaking_rate), 2),
        "invalid": invalid,
    }

    return utt_id, out


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    num_jobs = int(args.num_jobs)

    spk2meta = load_libritts_spk_metadata(debug=args.debug)
    in_dir = Path(args.in_dir)

    logger = getLogger(
        verbose=100, filename="log/compute_utt_stats.log", name="compute_utt_stats"
    )

    executor = ProcessPoolExecutor(max_workers=num_jobs)
    futures = []

    with open(args.f0_stats) as f:
        f0_stats = yaml.load(f, Loader=yaml.SafeLoader)

    for spk, _ in tqdm(spk2meta.items()):
        spk_in_dir = in_dir / spk
        spk_mfa_dir = spk_in_dir / "textgrid"

        if not spk_in_dir.exists():
            continue

        textgrid_files = sorted(list(spk_mfa_dir.glob("*.TextGrid")))
        # valid utt_ids
        utt_ids = [f.stem for f in textgrid_files]
        wav_files = [spk_in_dir / "wav24k" / f"{utt_id}.wav" for utt_id in utt_ids]

        for wav_file, textgrid_file in zip(wav_files, textgrid_files):
            futures.append(
                executor.submit(
                    process_utterance,
                    logger,
                    wav_file,
                    textgrid_file,
                    f0_stats,
                )
            )

    metadata = {}
    for future in tqdm(futures):
        utt_id, meta = future.result()
        metadata[utt_id] = meta

    with open(args.out_filename, "w") as of:
        yaml.dump(metadata, of)
