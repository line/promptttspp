# data_prep

This directory contains the following data preparation scripts:

1. MFA data preparation: Code for extracting phone alignments by Montréal Forced Aligner (MFA)
2. Style prompt data preparation: Code for preparing synthetic annotations of style prompts.

## 0. Download LibriTTS_R

Before running any scripts, be sure to put the [LibriTTS-R](https://www.openslr.org/141/) dataset to `./LibriTTS_R`. You must have the following directory structure:

```
LibriTTS_R/
├── BOOKS.txt
├── CHAPTERS.txt
├── LICENSE.txt
├── NOTE.txt
├── README_librispeech.txt
├── README_libritts.txt
├── README_libritts_r.txt
├── SPEAKERS.txt
├── dev-clean
├── dev-other
├── reader_book.tsv
├── speakers.tsv
├── test-clean
├── test-other
├── train-clean-100
├── train-clean-360
└── train-other-500
```

## 1. MFA data preparation

### Setup for MFA

```
conda install -c conda-forge montreal-forced-aligner
```

```
mfa model download dictionary english_us_arpa
mfa model download acoustic english_us_arpa
```

### Usage

Please check `runall_mfa.sh` for the usage.

Note that running MFA for all the utterances in LibriTTS-R takes a long time (likely a few days).


### Directory structure

After all the data preparation steps, the following directories will be created:

- `libritts_r_per_spk_cleaned`
  - `${spk}`
    - `textgrid`: text grid files
    - `wav24k`: 24kHz wav files

```
├── 100
│   ├── textgrid
│   └── wav24k
├── 1001
│   ├── textgrid
│   └── wav24k
├── 1006
│   ├── textgrid
│   └── wav24k
...
```


## 2. Style prompt data preparation

Code for estimating per-utterance style tags (e.g., low pitch, normal pitch and high pitch) from the data statistics.

### Usage

Please check `runall_style_prompt_tags.sh` for the usage.
