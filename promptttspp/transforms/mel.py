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

from torchaudio.transforms import MelSpectrogram


class MelSpectrogramTransform(MelSpectrogram):
    def to_spec(self, wav):
        spec = self.spectrogram(wav)
        return spec

    def spec_to_mel(self, spec):
        mel = self.mel_scale(spec)
        mel = mel.clamp_min(1e-5).log()
        return mel

    def to_mel(self, wav):
        spec = self.to_spec(wav)
        mel = self.spec_to_mel(spec)
        return mel

    def forward(self, wav):
        return self.to_mel(wav)
