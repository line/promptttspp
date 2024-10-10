# PromptTTS++: Controlling Speaker Identity in Prompt-Based Text-to-Speech Using Natural Language Descriptions

You can check the [paper](https://arxiv.org/abs/2309.08140) and [huggingface demo](https://huggingface.co/spaces/line-corporation/promptttspp).

## Installation

### Conda

```
conda create -n py38_prompt python=3.8 numpy scipy scikit-learn numba cython pandas tqdm
```

```
conda activate py38_prompt
```

### Pytorch

```
pip install "torch==1.11.0+cu113" "torchvision==0.12.0+cu113" "torchaudio==0.11.0" --extra-index-url https://download.pytorch.org/whl/cu113
```

Once the conda env and pytorch is ready, you can install the rest of prompttts depenendices by

```
pip install -e .
```

## Directory structure

- `data_prep`: Data preparation scripts
- `metadata`: Metadata
- `egs`: Code for experiments
- `promptttspp`: PromptTTS++ Python packages

## Demo

We provide pre-trained models and demo on [Hugging Face (WIP)](). If you want to run the models locally, please follow the instructions below.
Note that a pretrained BigVGAN must be used as a vocoder to synthesize speech.

Please modify the following path in `egs/proposed/bin/conf/demo.yaml`:
```
model_ckpt_path:
vocoder_ckpt_path:
mel_stats_file:
```

And then, run the demo as follows:
```
python app.py
```

## Training a model from scratch

To train a model from scratch, you need to prepare the dataset. Please follow the steps in `data_prep/README.md` to make a complete dataset for training.

Note that the default configs in `egs/proposed/bin/conf` for preprocessing and training assume that the dataset is stored in `data_prep/out/libritts_r_per_spk_cleaned`.

```
tree -L 2 data_prep/out/libritts_r_per_spk_cleaned | head -10
```

```
data_prep/out/libritts_r_per_spk_cleaned
├── 100
│   ├── textgrid
│   └── wav24k
├── 1001
│   ├── textgrid
│   └── wav24k
├── 1006
│   ├── textgrid
│   └── wav24k
```

### Preprocess

Before preprocessing, please check the settings in `egs/proposed/bin/conf`. In particular, you should set the `root` parameter in `egs/proposed/bin/conf/path/default.yaml` to the path of this cloned repository.
Other parameters should be okay with the default settings unless you want to change them.

Once you set the parameters, go to the `egs/proposed` directory and run the following command:

```
sh preprocess.sh
```

You can also directly run the individual scripts used in the `preprocess.sh`. Please check the `preprocess.sh` for the details.


### Train
```
CUDA_VISIBLE_DEVICES=0 python bin/train.py model=prompttts_mdn_v2_wo_erg_final output_dir=./out/proposed train=noam path=default dataset.max_tokens=30000 train.fp16=false dataset=mel
```
### Note
This repository contains the code used for experiments during internship. It includes code that was not used in the experiments for the [PromptTTS++ paper](https://arxiv.org/abs/2309.08140).

## License
[Apache License 2.0](LICENSE)

## Citation
```
@inproceedings{promptttspp,
    authors={Reo Shimizu, Ryuichi Yamamoto, Masaya Kawamura, Yuma Shirahata, Hironori Doi, Tatsuya Komatsu, Kentaro Tachibana},
    title={PromptTTS++: Controlling Speaker Identity in Prompt-Based Text-To-Speech Using Natural Language Descriptions},
    booktitle={Proc. ICASSP 2024},
    pages={12672-12676},
    year={2024},
}
```

## Acknowledgements

This project was done with an internship student: Reo Shimizu from Tohoku University.

Blog post: https://engineering.linecorp.com/ja/blog/natural-language-control-speakability.
