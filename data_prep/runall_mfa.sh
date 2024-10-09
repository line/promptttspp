#!/bin/bash

# Set to true to run MFA for a small subset of speakers for debugging purpose.
debug=false

libritts_r_dir=./LibriTTS_R
output_dir=./out/
mkdir -p ${output_dir}

if [ ${debug} == "true" ]; then
    extra_arg="--debug"
    out_wav_dir=${output_dir}/debug_libritts_r_per_spk
    out_textgrid_dir=${output_dir}/debug_libritts_r_per_spk_mfa
    out_merged_dir=${output_dir}/debug_libritts_r_per_spk_cleaned
else
    extra_arg=""
    out_wav_dir=${output_dir}/libritts_r_per_spk
    out_textgrid_dir=${output_dir}/libritts_r_per_spk_mfa
    out_merged_dir=${output_dir}/libritts_r_per_spk_cleaned
fi

python prepare_mfa.py ${libritts_r_dir} ${out_wav_dir} --n_jobs 64 ${extra_arg}

# NOTE: this will take a few days
if [ ! -d ${out_textgrid_dir} ]; then
    python run_mfa.py ${out_wav_dir} ${out_textgrid_dir} ${extra_arg}
fi

python finalize_mfa.py ${out_wav_dir} ${out_textgrid_dir} ${out_merged_dir}
