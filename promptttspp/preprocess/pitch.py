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

import numpy as np
import pyworld as pw
from nnmnkwii.preprocessing import interp1d


def extract_pitch(wav, sample_rate, hop_length, f0_floor, f0_ceil):
    _f0, t = pw.dio(
        wav,
        sample_rate,
        frame_period=hop_length / sample_rate * 1e3,
        f0_floor=f0_floor,
        f0_ceil=f0_ceil,
    )
    f0 = pw.stonemask(wav, _f0, t, sample_rate)
    vuv = (f0 != 0).astype(np.float32)

    cf0 = interp1d(f0)
    nonzero_idx = np.nonzero(cf0)
    cf0[nonzero_idx] = np.log(cf0[nonzero_idx])

    return f0, cf0, vuv
