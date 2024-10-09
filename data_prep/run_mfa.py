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
from subprocess import PIPE, Popen

from common import getLogger, load_libritts_spk_metadata
from tqdm.auto import tqdm


def get_parser():
    parser = argparse.ArgumentParser(
        description="Run MFA on LibriTTS-R",
    )
    parser.add_argument(
        "in_dir", type=str, help="LibriTTS per-speaker restructured data root"
    )
    parser.add_argument("mfa_out_dir", type=str, help="Output directory")
    parser.add_argument("--n_jobs", type=int, default=1, help="Number of jobs")
    parser.add_argument("--debug", action="store_true", help="Debug")

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args(sys.argv[1:])

    n_jobs = int(args.n_jobs)

    spk2meta = load_libritts_spk_metadata(debug=args.debug)

    in_dir = Path(args.in_dir)
    mfa_out_dir = Path(args.mfa_out_dir)
    mfa_out_dir.mkdir(exist_ok=True, parents=True)

    logger = getLogger(verbose=100, filename="log/run_mfa.log", name="run_mfa")

    for spk, _ in tqdm(spk2meta.items()):
        spk_in_dir = in_dir / spk
        spk_mfa_dir = mfa_out_dir / spk

        if not spk_in_dir.exists():
            logger.warning(f"No input dir for {spk}; skipping")
            continue

        cmd = f"mfa align {spk_in_dir} english_us_arpa english_us_arpa {spk_mfa_dir}"
        cmd = cmd + f" --num_jobs {n_jobs} --clean --quiet --use_mp"
        logger.info(cmd)
        p = Popen(cmd, shell=True, stdout=PIPE)
        r = p.wait()
        if r != 0:
            logger.error(f"Error in MFA for {spk}")
            continue
