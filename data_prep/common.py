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

import logging
import os
from os.path import dirname

format = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"


def getLogger(verbose=0, filename=None, name="ttwave"):
    logger = logging.getLogger(name)
    if verbose >= 100:
        logger.setLevel(logging.DEBUG)
    elif verbose > 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.WARN)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(logging.Formatter(format))
    logger.addHandler(stream_handler)

    if filename is not None:
        os.makedirs(dirname(filename), exist_ok=True)
        file_handler = logging.FileHandler(filename=filename)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(format))
        logger.addHandler(file_handler)

    return logger


def load_libritts_spk_metadata(path="external/speakers.tsv", debug=False):
    spk2meta = {}
    with open(path) as f:
        lines = f.readlines()
        for line in lines[1:]:  # skip header
            vals = line.strip().split("\t")
            if len(vals) == 4:
                spk, gender, subset, name = vals
            elif len(vals) == 3:
                spk, gender, subset = vals
                name = "Unknown"
            spk2meta[spk] = {"gender": gender, "subset": subset, "name": name}

    if debug:
        eval_spks = [1188, 1995, 260]
        val_spks = [89, 90, 91]
        train_spks = [100, 101, 102, 1001]
        spks = eval_spks + val_spks + train_spks
        spk2meta = {k: v for k, v in spk2meta.items() if int(k) in spks}

    return spk2meta
