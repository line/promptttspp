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

from typing import List

import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer


class BertWrapper(nn.Module):
    def __init__(self, class_name="bert-base-uncased"):
        super().__init__()
        self.model = BertModel.from_pretrained(class_name)
        self.tokenizer = BertTokenizer.from_pretrained(class_name)

        for p in self.model.parameters():
            p.requires_grad = False
        for p in self.model.encoder.layer[-1].attention.parameters():
            p.requires_grad = True

    def forward(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        inputs = self.tokenizer(prompts, padding=True, return_tensors="pt").to(device)
        outputs = self.model(**inputs)
        last_hidden_state = outputs.last_hidden_state
        # NOTE: To use CLS token is best method ?
        return last_hidden_state[:, 0, :]


class PromptEncoder(nn.Module):
    def __init__(self, model_name, in_channels, mid_channels, out_channels):
        super().__init__()
        self.bert = BertWrapper(model_name)
        self.adaptor = nn.Sequential(
            nn.Linear(in_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, mid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(mid_channels, out_channels),
        )

    def forward(self, prompts: List[str], device: torch.device) -> torch.Tensor:
        x = self.bert(prompts, device)
        emb = self.adaptor(x)
        return emb.unsqueeze(-1)


class SepPromptEncoder(nn.Module):
    def __init__(self, model_name, in_channels, mid_channels, out_channels):
        super().__init__()
        self.style_enc = PromptEncoder(
            model_name, in_channels, mid_channels, out_channels
        )
        self.spk_enc = PromptEncoder(
            model_name, in_channels, mid_channels, out_channels
        )

    def forward(self, prompts, device: torch.device) -> torch.Tensor:
        prompts = [p.split("|") for p in prompts]
        style_prompts = [p[0] for p in prompts]
        spk_prompts = [p[1] for p in prompts]

        x1 = self.style_enc(style_prompts, device)
        x2 = self.spk_enc(spk_prompts, device)
        emb = x1 + x2
        return emb

    def infer(self, prompts, device):
        prompts = [p.split("|") for p in prompts]
        style_prompts = [p[0] for p in prompts]
        spk_prompts = [p[1] for p in prompts]

        x1 = self.style_enc(style_prompts, device)
        x2 = self.spk_enc(spk_prompts, device)
        emb = x1 + x2
        return emb, x1, x2
