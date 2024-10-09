#!/bin/bash -eu

python bin/preprocess.py
python bin/split_df.py
python bin/compute_mel.py
python bin/split_df.py
