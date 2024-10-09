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

import gradio as gr
import hydra
import matplotlib.pyplot as plt
import torch
import torchaudio
from g2p_en import G2p
from hydra.utils import instantiate
from omegaconf import OmegaConf
from promptttspp.text.eng import symbols, text_to_sequence
from promptttspp.utils.model import lowpass_filter
import nltk


def load_model(model_cfg, model_ckpt_path, vocoder_cfg, vocoder_ckpt_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instantiate(model_cfg)
    model.load_state_dict(torch.load(model_ckpt_path, map_location="cpu")["model"])
    model = model.to(device).eval()

    vocoder = instantiate(vocoder_cfg)
    vocoder.load_state_dict(
        torch.load(vocoder_ckpt_path, map_location="cpu")["generator"]
    )
    vocoder = vocoder.to(device).eval()
    return model, vocoder


def build_ui(g2p, model, vocoder, to_mel, mel_stats):
    content_placeholder = (
        "This is text to speech demo, which allows you to control the speaker identity "
        "in natural language as follows."
    )
    style_placeholder = "A man speaks slowly in a low tone."

    @torch.no_grad()
    def onclick_synthesis(content_prompt, style_prompt=None, reference_mel=None):
        assert style_prompt is not None or reference_mel is not None
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        phonemes = g2p(content_prompt)
        phonemes = [p if p not in [",", "."] else "sil" for p in phonemes]
        phonemes = [p for p in phonemes if p in symbols]
        phoneme_ids = text_to_sequence(" ".join(phonemes))
        phoneme_ids = torch.LongTensor(phoneme_ids)[None, :].to(device)
        if style_prompt is not None:
            dec, log_cf0, vuv = model.infer(
                phoneme_ids,
                style_prompt=style_prompt,
                use_max=True,
                noise_scale=0.5,
                return_f0=True,
            )
        else:
            reference_mel = (reference_mel - mel_stats["mean"]) / mel_stats["std"]
            reference_mel = reference_mel.to(device)
            dec, log_cf0, vuv = model.infer(
                phoneme_ids,
                reference_mel=reference_mel,
                use_max=True,
                noise_scale=0.5,
                return_f0=True,
            )
        modfs = int(1.0 / (10 * 0.001))
        log_cf0 = lowpass_filter(log_cf0, modfs, cutoff=20)
        f0 = log_cf0.exp()
        f0[vuv < 0.5] = 0
        dec = dec * mel_stats["std"] + mel_stats["mean"]
        wav = vocoder(dec, f0).squeeze(1).cpu()
        return wav

    def onclick_with_style_prompt(content_prompt, style_prompt):
        wav = onclick_synthesis(
            content_prompt=content_prompt, style_prompt=style_prompt
        )
        mel = to_mel(wav)
        fig = plt.figure(figsize=(12, 8))
        plt.imshow(mel.squeeze().numpy(), aspect="auto", origin="lower")
        return (to_mel.sample_rate, wav.squeeze().numpy()), fig

    def onclick_with_reference_mel(content_prompt, reference_wav_path):
        wav, _ = torchaudio.load(reference_wav_path)
        ref_mel = to_mel(wav)
        wav = onclick_synthesis(content_prompt=content_prompt, reference_mel=ref_mel)
        mel = to_mel(wav)
        fig = plt.figure(figsize=(12, 8))
        plt.imshow(mel.squeeze().numpy(), aspect="auto", origin="lower")
        return (to_mel.sample_rate, wav.squeeze().numpy()), fig

    with gr.Blocks() as demo:
        gr.Markdown("# PromptTTS++")
        gr.Markdown("### NOTE: Please do not enter personal information.")
        content_prompt = gr.Textbox(
            content_placeholder, lines=3, label="Content prompt"
        )
        with gr.Tabs():
            with gr.TabItem("Style prompt"):
                style_prompt = gr.Textbox(
                    style_placeholder, lines=3, label="Style prompt"
                )
                syn_button1 = gr.Button("Synthesize")
                wav1 = gr.Audio(label="Output wav", elem_id="prompt")
                plot1 = gr.Plot(label="Output mel", elem_id="prompt")
            with gr.TabItem("Reference wav"):
                ref_wav_path = gr.Audio(
                    type="filepath", label="Reference wav", elem_id="ref"
                )
                syn_button2 = gr.Button("Synthesize")
                wav2 = gr.Audio(label="Output wav", elem_id="ref")
                plot2 = gr.Plot(label="Output mel", elem_id="ref")
        syn_button1.click(
            onclick_with_style_prompt,
            inputs=[content_prompt, style_prompt],
            outputs=[wav1, plot1],
        )
        syn_button2.click(
            onclick_with_reference_mel,
            inputs=[content_prompt, ref_wav_path],
            outputs=[wav2, plot2],
        )
    demo.launch()


@hydra.main(version_base=None, config_path="egs/proposed/bin/conf", config_name="demo")
def main(cfg):
    model, vocoder = load_model(
        cfg.model, cfg.model_ckpt_path, cfg.vocoder, cfg.vocoder_ckpt_path
    )
    to_mel = instantiate(cfg.transforms)
    # If the NLTK version is 3.9.1, this download code might be necessary.
    nltk.download('averaged_perceptron_tagger_eng') 
    g2p = G2p()
    mel_stats = OmegaConf.load(cfg.mel_stats_file)
    build_ui(g2p, model, vocoder, to_mel, mel_stats)


if __name__ == "__main__":
    main()
