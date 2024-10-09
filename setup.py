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

from os.path import exists

from setuptools import find_packages, setup

if exists("README.md"):
    with open("README.md", "r") as fh:
        LONG_DESC = fh.read()
else:
    LONG_DESC = ""

setup(
    name="promptttspp",
    version="0.0.1",
    description="PromptTTS++",
    author="LY Corp.",
    author_email="ryuichi.yamamoto@lycorp.co.jp",
    long_description=LONG_DESC,
    long_description_content_type="text/markdown",
    packages=find_packages(include=["promptttspp*"]),
    include_package_data=True,
    install_requires=[
        "transformers",
        "hydra-core",
        "pysptk",
        "pyworld",
        "tensorboard",
        "pandas",
        "matplotlib",
        "soundfile",
        "faster_whisper",
        "nnmnkwii",
        "g2p_en",
        "gradio",
        "filelock",
        "pillow<=9.5.0",  # to avoid error in PIL
        # NOTE: the following packages are used in data preparation
        "librosa",
        "pyloudnorm",
        "syllables",
    ],
    extras_require={
        "test": [
            "pysen",
            "mypy<=0.910",
            "black>=19.19b0,<=20.8",
            "flake8>=3.7,<4",
            "flake8-bugbear",
            "isort>=4.3,<5.2.0",
            "click<8.1.0",  # black<22.3 is incompatible with click>=8.1.0
        ],
    },
)
