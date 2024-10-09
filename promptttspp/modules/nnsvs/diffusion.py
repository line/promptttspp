from collections import deque
from functools import partial

import numpy as np
import torch
from torch import nn
from tqdm.auto import tqdm


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def noise_like(shape, noise_fn, device, repeat=False):
    if repeat:
        resid = [1] * (len(shape) - 1)
        shape_one = (1, *shape[1:])

        return noise_fn(*shape_one, device=device).repeat(shape[0], *resid)

    else:
        return noise_fn(*shape, device=device)


def linear_beta_schedule(timesteps, min_beta=1e-4, max_beta=0.06):
    """
    linear schedule
    """
    betas = np.linspace(min_beta, max_beta, timesteps)
    return betas


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return np.clip(betas, a_min=0, a_max=0.999)


beta_schedule = {
    "cosine": cosine_beta_schedule,
    "linear": linear_beta_schedule,
}


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        denoise_fn,
        encoder=None,
        K_step=100,
        betas=None,
        schedule_type="linear",
        scheduler_params=None,
        norm_scale=1.0,
        pndm_speedup=None,
        objective="pred_noise",
        do_classifier_free_guidance=False,
        guidance_scale=1.0,
        clip_denoised=True,
        a_min=None,
        a_max=None,
    ):
        """Gaussian Diffusion model from DiffSinger

        https://arxiv.org/abs/2105.02446

        .. note::

            Target data is scaled by norm_scale by default. If a_min and a_max are
            specified, data is scaled to by min and max values.

        Args:
            in_dim (int): Input dimension.
            out_dim (int): Output dimension.
            denoise_fn (nn.Module): Denoising function.
            encoder (nn.Module, optional): Encoder function. Defaults to None.
            K_step (int, optional): Number of diffusion steps. Defaults to 100.
            betas (list, optional): List of beta values. Defaults to None.
            schedule_type (str, optional): Type of beta schedule. Defaults to ``linear``.
            scheduler_params (dict, optional): Parameters for beta schedule. Defaults to None.
            norm_scale (int, optional): Scale for the noise. Defaults to 1.
            pndm_speedup (int, optional): Speedup for PNDM. Defaults to None.
            objective (str, optional): ``pred_noise`` or ``pred_x0``.
                Defaults to ``pred_noise``.
            do_classifier_free_guidance (bool, optional): Whether to use
                classifier-free guidance. Defaults to False.
            guidance_scale (float, optional): Scale for classifier-free guidance.
                Defaults to 1.0.
            clip_denoised (bool, optional): Whether to clip denoised output.
            a_min (float, optional): Minimum value of the data. Defaults to None.
            a_max (float, optional): Maximum value of the data. Defaults to None.
        """
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.denoise_fn = denoise_fn
        self.K_step = K_step
        self.pndm_speedup = pndm_speedup
        self.encoder = encoder
        self.norm_scale = norm_scale
        self.objective = objective
        self.do_classifier_free_guidance = do_classifier_free_guidance
        self.guidance_scale = guidance_scale
        self.clip_denoised = clip_denoised
        self.a_min = a_min
        self.a_max = a_max

        if do_classifier_free_guidance and objective != "pred_noise":
            raise ValueError(
                "do_classifier_free_guidance is only valid for pred_noise objective"
            )

        if scheduler_params is None:
            if schedule_type == "linear":
                scheduler_params = {"max_beta": 0.06}
            elif schedule_type == "cosine":
                scheduler_params = {"s": 0.008}

        if encoder is not None:
            assert encoder.in_dim == in_dim, "encoder input dim must match in_dim"
        assert out_dim == denoise_fn.in_dim, "denoise_fn input dim must match out_dim"

        if betas is not None:
            betas = (
                betas.detach().cpu().numpy()
                if isinstance(betas, torch.Tensor)
                else betas
            )
        else:
            betas = beta_schedule[schedule_type](K_step, **scheduler_params)

        alphas = 1.0 - betas
        alphas_cumprod = np.cumprod(alphas, axis=0)
        alphas_cumprod_prev = np.append(1.0, alphas_cumprod[:-1])

        self.noise_list = deque(maxlen=4)

        to_torch = partial(torch.tensor, dtype=torch.float32)
        self.register_buffer("betas", to_torch(betas))
        self.register_buffer("alphas_cumprod", to_torch(alphas_cumprod))
        self.register_buffer("alphas_cumprod_prev", to_torch(alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer("sqrt_alphas_cumprod", to_torch(np.sqrt(alphas_cumprod)))
        self.register_buffer(
            "sqrt_one_minus_alphas_cumprod", to_torch(np.sqrt(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "log_one_minus_alphas_cumprod", to_torch(np.log(1.0 - alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recip_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod))
        )
        self.register_buffer(
            "sqrt_recipm1_alphas_cumprod", to_torch(np.sqrt(1.0 / alphas_cumprod - 1))
        )

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)
        )
        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)
        self.register_buffer("posterior_variance", to_torch(posterior_variance))
        # below: log calculation clipped because the posterior variance is 0
        # at the beginning of the diffusion chain
        self.register_buffer(
            "posterior_log_variance_clipped",
            to_torch(np.log(np.maximum(posterior_variance, 1e-20))),
        )
        self.register_buffer(
            "posterior_mean_coef1",
            to_torch(betas * np.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod)),
        )
        self.register_buffer(
            "posterior_mean_coef2",
            to_torch(
                (1.0 - alphas_cumprod_prev) * np.sqrt(alphas) / (1.0 - alphas_cumprod)
            ),
        )

    def _norm(self, x, a_max=10):
        if self.a_min is not None:
            # min-max normalization following DiffSinger
            # https://github.com/MoonInTheRiver/DiffSinger
            assert self.a_max is not None
            return (x - self.a_min) / (self.a_max - self.a_min) * 2 - 1
        else:
            # NOTE: since the original impl. assume the data is distributed in [-1, 1]
            # let us (roughly) convert N(0,1) noramlized to data to [-1, 1]
            return x / a_max

    def _denorm(self, x, a_max=10):
        if self.a_min is not None:
            assert self.a_max is not None
            return (x + 1) / 2 * (self.a_max - self.a_min) + self.a_min
        else:
            return x * a_max

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0
        ) / extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape
        )
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, cond, clip_denoised: bool, g=None):
        if self.objective == "pred_noise":
            if self.do_classifier_free_guidance:
                x = torch.cat([x] * 2)
                cond = torch.cat([cond] * 2)
                # NOTE: Use zero vectors to indicate null speaker embedding
                g = torch.cat([torch.zeros_like(g), g])
                t = torch.cat([t] * 2)

            pred_noise = self.denoise_fn(x, t, cond=cond, g=g)

            if self.do_classifier_free_guidance:
                pred_noise_uncond, pred_noise = pred_noise.chunk(2, dim=0)
                pred_noise = pred_noise_uncond + self.guidance_scale * (
                    pred_noise - pred_noise_uncond
                )
                x = x.chunk(2, dim=0)[0]
                t = t.chunk(2, dim=0)[0]

            pred_x_start = self.predict_start_from_noise(x, t=t, noise=pred_noise)
            if clip_denoised:
                pred_x_start.clamp_(-1.0, 1.0)
        elif self.objective == "pred_x0":
            pred_x_start = self.denoise_fn(x, t, cond=cond, g=g)
            if clip_denoised:
                pred_x_start.clamp_(-1.0, 1.0)
            pred_noise = self.predict_noise_from_start(x, t, x0=pred_x_start)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=pred_x_start, x_t=x, t=t
        )
        return model_mean, posterior_variance, posterior_log_variance

    @torch.no_grad()
    def p_sample(
        self,
        x,
        t,
        cond,
        noise_fn=torch.randn,
        clip_denoised=True,
        repeat_noise=False,
        g=None,
    ):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x,
            t=t,
            cond=cond,
            clip_denoised=clip_denoised,
            g=g,
        )
        noise = noise_like(x.shape, noise_fn, device, repeat_noise)
        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.no_grad()
    def p_sample_plms(self, x, t, interval, cond, g=None):
        """
        Use the PLMS method from Pseudo Numerical Methods for Diffusion Models on Manifolds
        https://arxiv.org/abs/2202.09778.
        """

        def get_x_pred(x, noise_t, t):
            a_t = extract(self.alphas_cumprod, t, x.shape)
            a_prev = extract(
                self.alphas_cumprod,
                torch.max(t - interval, torch.zeros_like(t)),
                x.shape,
            )
            a_t_sq, a_prev_sq = a_t.sqrt(), a_prev.sqrt()

            x_delta = (a_prev - a_t) * (
                (1 / (a_t_sq * (a_t_sq + a_prev_sq))) * x
                - 1
                / (a_t_sq * (((1 - a_prev) * a_t).sqrt() + ((1 - a_t) * a_prev).sqrt()))
                * noise_t
            )
            x_pred = x + x_delta

            return x_pred

        noise_list = self.noise_list
        noise_pred = self.denoise_fn(x, t, cond=cond, g=g)

        if len(noise_list) == 0:
            x_pred = get_x_pred(x, noise_pred, t)
            noise_pred_prev = self.denoise_fn(
                x_pred,
                torch.max(t - interval, torch.zeros_like(t)),
                cond=cond,
                g=g,
            )
            noise_pred_prime = (noise_pred + noise_pred_prev) / 2
        elif len(noise_list) == 1:
            noise_pred_prime = (3 * noise_pred - noise_list[-1]) / 2
        elif len(noise_list) == 2:
            noise_pred_prime = (
                23 * noise_pred - 16 * noise_list[-1] + 5 * noise_list[-2]
            ) / 12
        elif len(noise_list) >= 3:
            noise_pred_prime = (
                55 * noise_pred
                - 59 * noise_list[-1]
                + 37 * noise_list[-2]
                - 9 * noise_list[-3]
            ) / 24

        x_prev = get_x_pred(x, noise_pred_prime, t)
        noise_list.append(noise_pred)

        return x_prev

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def forward(self, cond, lengths=None, y=None, g=None, mask=None):
        """Forward step

        Args:
            cond (torch.Tensor): conditioning features of shaep (B, T, encoder_hidden_dim)
            lengths (torch.Tensor): lengths of each sequence in the batch
            y (torch.Tensor): ground truth of shape (B, T, C)
            g (torch.Tensor): speaker embedding of shape (B, 1, g_dim)

        Returns:
            tuple of tensors (B, T, in_dim), (B, T, in_dim)
        """
        B = cond.shape[0]
        device = cond.device

        if self.encoder is not None:
            cond = self.encoder(cond, lengths)

        # (B, M, T)
        cond = cond.transpose(1, 2)

        t = torch.randint(0, self.K_step, (B,), device=device).long()
        x_start = self._norm(y, self.norm_scale)
        x_start = x_start.transpose(1, 2)[:, None, :, :]  # [B, 1, M, T]

        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_out = self.denoise_fn(x_noisy, t, cond, g=g)

        if self.objective == "pred_noise":
            target = noise
        elif self.objective == "pred_x0":
            target = x_start

        target = target.squeeze(1).transpose(1, 2)
        model_out = model_out.squeeze(1).transpose(1, 2)

        return target, model_out

    def inference(self, cond, lengths=None, g=None):
        B = cond.shape[0]
        device = cond.device

        if self.encoder is not None:
            cond = self.encoder(cond, lengths)

        # (B, M, T)
        cond = cond.transpose(1, 2)

        t = self.K_step
        shape = (cond.shape[0], 1, self.out_dim, cond.shape[2])
        x = torch.randn(shape, device=device)

        if self.pndm_speedup:
            self.noise_list = deque(maxlen=4)
            iteration_interval = int(self.pndm_speedup)
            for i in tqdm(
                reversed(range(0, t, iteration_interval)),
                desc="sample time step",
                total=t // iteration_interval,
            ):
                x = self.p_sample_plms(
                    x,
                    torch.full((B,), i, device=device, dtype=torch.long),
                    iteration_interval,
                    cond,
                    g=g,
                )
        else:
            for i in tqdm(reversed(range(0, t)), desc="sample time step", total=t):
                x = self.p_sample(
                    x,
                    torch.full((B,), i, device=device, dtype=torch.long),
                    cond,
                    g=g,
                    clip_denoised=self.clip_denoised,
                )
        x = self._denorm(x[:, 0].transpose(1, 2), self.norm_scale)
        return x
