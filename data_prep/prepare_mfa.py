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
import shutil
import sys
from pathlib import Path

from common import load_libritts_spk_metadata
from joblib import Parallel, delayed
from promptttspp.utils.joblib import tqdm_joblib
from tqdm.auto import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Restructure the LibriTTS-R dataset for convenience",
    )
    parser.add_argument("in_dir", type=str, help="LibriTTS original data root")
    parser.add_argument("out_dir", type=str, help="Output directory")
    parser.add_argument("--n_jobs", type=int, default=8, help="Number of jobs")
    parser.add_argument("--debug", action="store_true", help="Debug")

    return parser


def process_spk(spk, meta, in_dir, out_dir):
    subset = meta["subset"]
    wav_files = sorted((in_dir / subset).glob(f"*/*/{spk}_*.wav"))

    if len(wav_files) == 0:
        print(f"No wav files found for {spk}", meta)
        return

    spk_out_dir = out_dir / spk
    spk_out_dir.mkdir(exist_ok=True, parents=True)
    # copy to spk_out_dir/filename
    for wav_file in tqdm(wav_files, leave=False):
        utt_id = wav_file.name.replace(".wav", "")
        text_file = wav_file.parent / f"{utt_id}.normalized.txt"

        # Sadly, some text transcriptions are missing
        # train-clean-360/1382/130492/1382_130492_000049_000000.normalized.txt
        if not text_file.exists():
            print(f"Text file not found for {wav_file}")
            continue

        out_wav_file = spk_out_dir / wav_file.name
        out_lab_file = spk_out_dir / f"{utt_id}.lab"
        shutil.copy2(wav_file, out_wav_file)
        shutil.copy2(text_file, out_lab_file)


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(exist_ok=True, parents=True)

    spk2meta = load_libritts_spk_metadata(debug=args.debug)
    with tqdm_joblib(len(spk2meta)):
        Parallel(n_jobs=args.n_jobs)(
            delayed(process_spk)(spk, meta, in_dir, out_dir)
            for spk, meta in spk2meta.items()
        )
