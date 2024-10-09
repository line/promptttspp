#!/bin/bash

libritts_r_dir=./LibriTTS_R
metadata_dir=../metadata
output_dir=./out

n_jobs=64
merged_dir=${output_dir}/libritts_r_per_spk_cleaned
f0_stats=${metadata_dir}/libritts_r_f0_stats.yaml
style_prompt_candidates=${metadata_dir}/style_prompt_candidates_v230922.csv

if [ ! -d ${merged_dir} ]; then
    echo "You must need to create dataset with MFA alignments"
    exit 1
fi

utt_metadata=${output_dir}/libritts_r_per_utt_metadata.yaml
out_csv_file=${output_dir}/metadata_w_style_prompt_key.csv

if [ ! -e ${utt_metadata} ]; then
    python compute_utt_stats.py ${merged_dir} ${f0_stats} --out_filename ${utt_metadata} \
        --num_jobs ${n_jobs}
fi

python add_style_prompt_tags.py ${libritts_r_dir} ${utt_metadata} ${style_prompt_candidates} \
    --out_filename ${out_csv_file}
