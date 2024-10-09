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

import torch
import torch.nn as nn
import torch.nn.functional as F
from promptttspp.modules.esp import ConformerEncoder
from promptttspp.modules.mdn import (
    mdn_get_most_probable_sigma_and_mu,
    mdn_loss,
    mdn_sample_sigma_and_mu,
)
from promptttspp.utils.model import sequence_mask, to_log_scale
from torch.cuda.amp import autocast


class PromptTTSMDNDurCFG(nn.Module):
    def __init__(
        self,
        phoneme_embedding,
        encoder,
        variance_adaptor,
        reference_encoder,
        prompt_encoder,
        decoder,
        out_conv=None,
        style_mdn=None,
        norm_style_emb=False,
        mdn_disable_amp=False,
        loss_dec_scale=8.0,
    ) -> None:
        """PromptTTSMDNDurCFG

        PromptTTS with the following features:
            - MDN-based duration model
            - MDN-based style embedding modeling
            - (Optional) classifier-free guidance (CFG) for diffusion-based decoder

        Args:
            norm_style_emb (bool, optional): Normalize style embedding to unit norm.
        """
        super().__init__()
        self.phoneme_emb = phoneme_embedding
        self.encoder = encoder
        self.variance_adaptor = variance_adaptor
        self.reference_encoder = reference_encoder
        self.prompt_encoder = prompt_encoder
        self.style_mdn = style_mdn
        self.decoder = decoder
        self.out_conv = out_conv
        self.norm_style_emb = norm_style_emb
        self.mdn_disable_amp = mdn_disable_amp
        self.loss_dec_scale = loss_dec_scale

        if isinstance(decoder, ConformerEncoder):
            assert out_conv is not None

        # NOTE: variance adaptor must have frame prior network
        assert self.variance_adaptor.frame_prior_network is not None

    def forward(self, batch):
        (
            phoneme,
            duration,
            phone_lengths,
            mel,
            log_cf0,
            vuv,
            energy,
            frame_lengths,
            prompt,
        ) = batch

        phone_mask = (
            sequence_mask(phone_lengths, phoneme.shape[-1])
            .unsqueeze(1)
            .to(phoneme.dtype)
        )
        x = self.phoneme_emb(phoneme, phone_mask)

        if isinstance(self.encoder, ConformerEncoder):
            x = self.encoder(x.transpose(1, 2), phone_lengths).transpose(1, 2)
        else:
            x = self.encoder(x, phone_mask)

        frame_mask = (
            sequence_mask(frame_lengths, mel.shape[-1]).unsqueeze(1).to(mel.dtype)
        )
        style_emb = self.reference_encoder(mel, frame_lengths)
        prompt_emb = self.prompt_encoder(prompt, x.device)
        if self.norm_style_emb:
            # (B, C, 1)
            assert style_emb.shape[-1] == 1
            style_emb = F.normalize(style_emb, dim=1)
            prompt_emb = F.normalize(prompt_emb, dim=1)

        if self.style_mdn is not None:
            with autocast(enabled=not self.mdn_disable_amp):
                style_mdn_out = self.style_mdn(prompt_emb.transpose(-1, -2))
        x = x + style_emb

        (
            x,
            log_duration_pred,
            log_cf0_pred,
            vuv_pred,
            energy_pred,
        ) = self.variance_adaptor(
            x, phone_mask, frame_mask, duration, log_cf0, vuv, energy
        )

        if isinstance(self.decoder, ConformerEncoder):
            x = self.decoder(x.transpose(1, 2), frame_lengths).transpose(1, 2)
            x = self.out_conv(x) * frame_mask
            loss_dec = (x - mel).abs().sum() / frame_mask.sum() / self.loss_dec_scale
        else:
            noise, x_recon = self.decoder(
                cond=x.transpose(-1, -2),
                y=mel.transpose(-1, -2),
                mask=frame_mask,
                g=style_emb.transpose(-1, -2),
            )
            noise = noise.transpose(-1, -2) * frame_mask
            x_recon = x_recon.transpose(-1, -2) * frame_mask

            # 10 times bigger than usual
            loss_dec = (
                (noise - x_recon).abs().sum() / frame_mask.sum() / self.loss_dec_scale
            )

        # (B, T, 1)
        log_duration = to_log_scale(duration)
        phone_mask_btc = phone_mask.transpose(1, 2) == 1
        assert phone_mask_btc.shape[-1] == 1

        with autocast(enabled=not self.mdn_disable_amp):
            loss_dur = mdn_loss(
                *log_duration_pred,
                log_duration.transpose(-1, -2),
                reduce=False,
                mask=phone_mask_btc,
            )
            loss_dur = loss_dur.masked_select(phone_mask_btc).mean()

        loss_cf0 = (log_cf0_pred - log_cf0).abs().sum() / frame_mask.sum()
        loss_vuv = (vuv_pred - vuv).abs().sum() / frame_mask.sum()

        if self.style_mdn is not None:
            with autocast(enabled=not self.mdn_disable_amp):
                loss_style = mdn_loss(
                    *style_mdn_out, style_emb.detach().transpose(-1, -2)
                ).mean()
        else:
            loss_style = (style_emb.detach() - prompt_emb).pow(2).mean()

        loss = loss_dec + loss_dur + loss_cf0 + loss_vuv + loss_style
        if energy_pred is not None:
            loss_energy = (energy_pred - energy).abs().sum() / frame_mask.sum()
            loss += loss_energy

        loss_dict = dict(
            loss=loss,
            dec=loss_dec,
            dur=loss_dur,
            cf0=loss_cf0,
            vuv=loss_vuv,
            style=loss_style,
        )
        if energy_pred is not None:
            loss_dict["energy"] = loss_energy

        return loss_dict

    def sample_style_emb(self, log_pi, log_sigma, mu, noise_scale, use_max):
        if use_max:
            sigma, mu = mdn_get_most_probable_sigma_and_mu(log_pi, log_sigma, mu)
        else:
            sigma, mu = mdn_sample_sigma_and_mu(log_pi, log_sigma, mu)

        style_emb = mu + sigma * torch.randn_like(sigma) * noise_scale

        if self.norm_style_emb:
            style_emb = F.normalize(style_emb, dim=-1)

        return style_emb.transpose(-1, -2)

    def infer(
        self,
        x,
        style_prompt=None,
        reference_mel=None,
        use_max=True,
        noise_scale=1.0,
        return_f0=False,
    ):
        # x : [1, L]

        assert (style_prompt is not None) ^ (
            reference_mel is not None
        ), "One of style inputs must not be None."

        phone_mask = torch.ones_like(x[:, None, :])
        x = self.phoneme_emb(x, phone_mask)

        if isinstance(self.encoder, ConformerEncoder):
            phone_lengths = torch.LongTensor([x.shape[-1]]).to(x.device)
            x = self.encoder(x.transpose(1, 2), phone_lengths).transpose(1, 2)
        else:
            x = self.encoder(x, phone_mask)

        if style_prompt is not None:
            style_emb = self.prompt_encoder(style_prompt, x.device)
            if self.norm_style_emb:
                style_emb = F.normalize(style_emb, dim=1)
            if self.style_mdn is not None:
                log_pi, log_sigma, mu = self.style_mdn(style_emb.transpose(-1, -2))
                style_emb = self.sample_style_emb(
                    log_pi, log_sigma, mu, noise_scale=noise_scale, use_max=use_max
                )
        else:
            ref_lengths = torch.LongTensor([reference_mel.shape[-1]])
            style_emb = self.reference_encoder(reference_mel, ref_lengths)
            if self.norm_style_emb:
                style_emb = F.normalize(style_emb, dim=1)

        x = x + style_emb
        if return_f0:
            x, frame_mask, log_cf0, vuv = self.variance_adaptor.infer(
                x, phone_mask, return_f0=True
            )
        else:
            x, frame_mask = self.variance_adaptor.infer(x, phone_mask, return_f0=False)

        frame_lengths = frame_mask.sum(dim=(1, 2))
        if isinstance(self.decoder, ConformerEncoder):
            x = self.decoder(x.transpose(1, 2), frame_lengths).transpose(1, 2)
            x = self.out_conv(x) * frame_mask
        else:
            x = self.decoder.inference(
                x.transpose(-1, -2), frame_lengths, g=style_emb.transpose(-1, -2)
            )
            x = x.transpose(-1, -2)
            x = x * frame_mask

        if return_f0:
            return x, log_cf0, vuv
        else:
            return x

    def infer_batch(
        self,
        phoneme,
        phone_lengths,
        style_prompt=None,
        reference_mel=None,
        ref_lengths=None,
        use_max=True,
        noise_scale=1.0,
        return_f0=False,
    ):
        # x : [1, L]

        assert (style_prompt is not None) ^ (
            reference_mel is not None
        ), "One of style inputs must not be None."

        phone_mask = sequence_mask(phone_lengths).unsqueeze(1).to(phoneme.dtype)
        x = self.phoneme_emb(phoneme, phone_mask)

        if isinstance(self.encoder, ConformerEncoder):
            x = self.encoder(x.transpose(1, 2), phone_lengths).transpose(1, 2)
        else:
            x = self.encoder(x, phone_mask)

        if style_prompt is not None:
            style_emb = self.prompt_encoder(style_prompt, x.device)
            if self.norm_style_emb:
                style_emb = F.normalize(style_emb, dim=1)
            if self.style_mdn is not None:
                log_pi, log_sigma, mu = self.style_mdn(style_emb.transpose(-1, -2))
                style_emb = self.sample_style_emb(
                    log_pi, log_sigma, mu, noise_scale=noise_scale, use_max=use_max
                )
        else:
            assert ref_lengths is not None
            style_emb = self.reference_encoder(reference_mel, ref_lengths)
            if self.norm_style_emb:
                style_emb = F.normalize(style_emb, dim=1)

        x = x + style_emb
        if return_f0:
            x, frame_mask, log_cf0, vuv = self.variance_adaptor.infer_batch(
                x, phone_mask, return_f0=True
            )
        else:
            x, frame_mask = self.variance_adaptor.infer_batch(
                x, phone_mask, return_f0=False
            )

        frame_lengths = frame_mask.sum(dim=(1, 2))
        if isinstance(self.decoder, ConformerEncoder):
            x = self.decoder(x.transpose(1, 2), frame_lengths).transpose(1, 2)
            x = self.out_conv(x) * frame_mask
        else:
            x = self.decoder.inference(
                x.transpose(-1, -2), frame_lengths, g=style_emb.transpose(-1, -2)
            )
            x = x.transpose(-1, -2)
            x = x * frame_mask

        if return_f0:
            return x, log_cf0, vuv, frame_lengths
        else:
            return x, frame_lengths

    def generate_style_emb(
        self, style_prompt, reference_mel, use_max=True, noise_scale=1.0
    ):
        prompt_emb = self.prompt_encoder(style_prompt, reference_mel.device)
        if self.norm_style_emb:
            prompt_emb = F.normalize(prompt_emb, dim=1)
        if self.style_mdn is not None:
            log_pi, log_sigma, mu = self.style_mdn(prompt_emb.transpose(-1, -2))
            prompt_emb = self.sample_style_emb(
                log_pi, log_sigma, mu, noise_scale=noise_scale, use_max=use_max
            )
        if self.norm_style_emb:
            prompt_emb = F.normalize(prompt_emb, dim=1)
        ref_lengths = torch.LongTensor([reference_mel.shape[-1]])
        ref_emb = self.reference_encoder(reference_mel, ref_lengths)
        if self.norm_style_emb:
            ref_emb = F.normalize(ref_emb, dim=1)
        return prompt_emb, ref_emb
