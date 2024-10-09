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
from promptttspp.text.eng import text_to_sequence
from promptttspp.utils.textgrid import Entry, read_textgrid


def adjust_textgrid(labels):
    """Adjust textgrid labels to have BOS and EOS segments

    This process may not be necessary but for compatibility with older scripts.
    """
    # BOS
    if labels[0].name in ["sil", "sp", ""]:
        # Use the first sil segment as BOS
        lbl = labels[0]
        labels[0] = Entry(start=lbl.start, stop=lbl.stop, name="^", tier=lbl.tier)
    else:
        assert len(labels) >= 2
        # Add a new BOS segment by consuming the first segment
        if labels[0].stop - labels[0].start > 0.01:
            bos = Entry(start=0.0, stop=0.01, name="^", tier="phone")
            lbl = labels[0]
            labels[0] = Entry(
                start=bos.stop, stop=lbl.stop, name=lbl.name, tier=lbl.tier
            )
            labels = [bos] + labels

    # EOS
    # add a new EOS segment by consuming the last segment
    assert len(labels) >= 2
    lbl = labels[-1]
    eos = Entry(start=lbl.stop - 0.01, stop=lbl.stop, name="$", tier="phone")
    labels[-1] = Entry(start=lbl.start, stop=eos.start, name=lbl.name, tier=lbl.tier)
    labels = labels + [eos]

    return labels


def _round_by_hop_length(sec, sr=24000, hop_length=240):
    new_sec = round(sec * sr / hop_length) * hop_length / sr
    return new_sec


def textgrid2phonedur(labels, sr=24000, hop_length=240, feats_len=None):
    ph_seq = []
    durations = []

    for idx in range(len(labels)):
        ph = labels[idx].name

        if ph == "":
            ph = "sil"
        ph_seq.append(ph)
        d = _round_by_hop_length(
            labels[idx].stop, sr=sr, hop_length=hop_length
        ) - _round_by_hop_length(labels[idx].start, sr=sr, hop_length=hop_length)
        if d <= 0:
            raise RuntimeError(f"Too short segment is detected: {labels[idx]}")

        d = round(sr / hop_length * d)
        durations.append(d)

    # Adjust EOS durations if feats_len is given
    if feats_len is not None:
        assert ph_seq[-1] == "$"
        eos_dur = feats_len - sum(durations[:-1])
        assert eos_dur >= 0
        durations[-1] = eos_dur

    return ph_seq, np.array(durations)


def process_textgrid(
    spk, utt_id, wav, textgrid_path, sample_rate=24000, n_fft=512, hop_length=240
):
    assert textgrid_path.exists()

    labels = read_textgrid(textgrid_path.as_posix())
    if len(labels) == 1:
        print(f"{utt_id} is ignored: only one phone is detected")
        return None
    feats_len = (wav.shape[-1] + n_fft // 2) // hop_length
    labels = adjust_textgrid(labels)

    try:
        ph_seq, durations = textgrid2phonedur(
            labels, sr=sample_rate, hop_length=hop_length, feats_len=feats_len
        )
    except RuntimeError as e:
        print(f"{utt_id} is ignored: {e}")
        return None

    text = " ".join(ph_seq)
    seq = text_to_sequence(text, add_special_token=False)

    assert len(durations) == len(
        seq
    ), f"number of durations must be equal to that of phones: {len(durations)} != {len(seq)}"
    s = durations.sum()
    assert (
        s == feats_len
    ), f"sum of durations must be equal to the number of frames: {s} != {feats_len}"

    return seq, durations
