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
from shutil import copy2

import numpy as np
import soundfile as sf
from common import getLogger, load_libritts_spk_metadata
from tqdm.auto import tqdm

format = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"


def get_parser():
    parser = argparse.ArgumentParser(
        description="Finalize MFA and LibriTTS-R data",
    )
    parser.add_argument(
        "in_dir", type=str, help="LibriTTS per-speaker restructured data root"
    )
    parser.add_argument("mfa_dir", type=str, help="MFA output directory")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument("--debug", action="store_true", help="Debug")

    return parser


def read_and_save(in_file, out_file):
    # let's make sure to have int16 dtype for saved files
    x, sr = sf.read(in_file)
    assert sr == 24000
    if x.dtype == np.float32 or x.dtype == np.float64:
        assert np.abs(x).max() <= 1.0
        x = (x * 32767).astype(np.int16)
    assert x.dtype == np.int16
    sf.write(out_file, x, sr)


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    spk2meta = load_libritts_spk_metadata(debug=args.debug)
    in_dir = Path(args.in_dir)
    mfa_dir = Path(args.mfa_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    logger = getLogger(
        verbose=100, filename="log/finalize_mfa.log", name="finalize_mfa"
    )

    total_duration = 0
    missing_utt_ids = []
    for spk, _ in tqdm(spk2meta.items()):
        spk_in_dir = in_dir / spk
        spk_mfa_dir = mfa_dir / spk

        if not spk_in_dir.exists():
            logger.warning(f"No input dir for {spk}")
            continue

        out_tgr_dir = out_dir / spk / "textgrid"
        out_wav_dir = out_dir / spk / "wav24k"
        out_txt_dir = out_dir / spk
        for d in [out_tgr_dir, out_wav_dir, out_txt_dir]:
            d.mkdir(exist_ok=True, parents=True)

        org_wav_files = sorted(list(spk_in_dir.glob("*.wav")))
        org_utt_ids = [f.stem for f in org_wav_files]

        textgrid_files = sorted(list(spk_mfa_dir.glob("*.TextGrid")))
        # valid utt_ids
        utt_ids = [f.stem for f in textgrid_files]
        wav_files = [spk_in_dir / f"{utt_id}.wav" for utt_id in utt_ids]

        if len(org_utt_ids) != len(utt_ids):
            spk_missing_utt_ids = list(set(org_utt_ids) - set(utt_ids))
            logger.warning(f"Missing {len(spk_missing_utt_ids)} utt_ids for {spk}")
            missing_utt_ids.extend(spk_missing_utt_ids)

        phones = {}
        for utt_id in utt_ids:
            # wav
            in_wav_file = spk_in_dir / f"{utt_id}.wav"
            assert in_wav_file.exists()
            out_wav_file = out_wav_dir / f"{utt_id}.wav"
            read_and_save(in_wav_file, out_wav_file)

            # textgrid
            in_textgrid_file = spk_mfa_dir / f"{utt_id}.TextGrid"
            assert in_textgrid_file.exists()
            out_textgrid_file = out_tgr_dir / f"{utt_id}.TextGrid"
            copy2(in_textgrid_file, out_textgrid_file)

    logger.info(f"Total duration: {total_duration/3600:.2f} hours")
    logger.info(f"Numbere of missing utterance IDs: {len(missing_utt_ids)}")

    # Write missing_utt_ids.txt
    with open(out_dir / "missing_utt_ids.txt", "w") as f:
        for utt_id in missing_utt_ids:
            f.write(f"{utt_id}\n")
