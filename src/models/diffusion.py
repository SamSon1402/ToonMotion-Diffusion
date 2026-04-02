"""
DDPM/DDIM Diffusion Schedule
=============================
Forward process: q(x_t | x_0) = N(sqrt(alpha_bar_t) * x_0, (1-alpha_bar_t) * I)
Reverse process: p_theta(x_{t-1} | x_t) using predicted noise
DDIM: Accelerated deterministic sampling (50 steps vs 1000)
"""

import torch
import torch.nn.functional as F
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class DiffusionConfig:
    num_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    inference_steps: int = 50


class DiffusionSchedule:
    """Manages noise scheduling for DDPM training and DDIM inference."""

    def __init__(self, config: DiffusionConfig, device: str = "cpu"):
        self.num_steps = config.num_steps
        self.device = device

        self.betas = torch.linspace(
            config.beta_start, config.beta_end, self.num_steps, device=device
        )
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.alpha_bars_prev = F.pad(self.alpha_bars[:-1], (1, 0), value=1.0)

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1.0 - self.alpha_bars)

        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )
        self.posterior_mean_coef1 = (
            self.betas * torch.sqrt(self.alpha_bars_prev) / (1.0 - self.alpha_bars)
        )
        self.posterior_mean_coef2 = (
            (1.0 - self.alpha_bars_prev) * torch.sqrt(self.alphas) / (1.0 - self.alpha_bars)
        )

    def q_sample(
        self, x_0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward diffusion: add noise to clean motion at timestep t."""
        if noise is None:
            noise = torch.randn_like(x_0)

        sqrt_ab = self.sqrt_alpha_bars[t][:, None, None]
        sqrt_1m_ab = self.sqrt_one_minus_alpha_bars[t][:, None, None]

        x_t = sqrt_ab * x_0 + sqrt_1m_ab * noise
        return x_t, noise

    def p_sample(
        self, noise_pred: torch.Tensor, x_t: torch.Tensor, t: torch.Tensor
    ) -> torch.Tensor:
        """DDPM reverse step: denoise x_t to x_{t-1}."""
        beta = self.betas[t][:, None, None]
        sqrt_1m_ab = self.sqrt_one_minus_alpha_bars[t][:, None, None]
        sqrt_recip_a = (1.0 / torch.sqrt(self.alphas[t]))[:, None, None]

        mean = sqrt_recip_a * (x_t - beta * noise_pred / sqrt_1m_ab)

        if t[0] > 0:
            noise = torch.randn_like(x_t)
            var = torch.sqrt(self.posterior_variance[t])[:, None, None]
            return mean + var * noise
        return mean

    def ddim_sample(
        self, noise_pred: torch.Tensor, x_t: torch.Tensor,
        t: torch.Tensor, t_prev: torch.Tensor, eta: float = 0.0
    ) -> torch.Tensor:
        """DDIM accelerated sampling (deterministic when eta=0)."""
        ab_t = self.alpha_bars[t][:, None, None]
        ab_prev = self.alpha_bars[t_prev][:, None, None] if t_prev[0] >= 0 else torch.ones_like(ab_t)

        x_0_pred = (x_t - torch.sqrt(1 - ab_t) * noise_pred) / torch.sqrt(ab_t)

        sigma = eta * torch.sqrt((1 - ab_prev) / (1 - ab_t) * (1 - ab_t / ab_prev))
        dir_xt = torch.sqrt(1 - ab_prev - sigma ** 2) * noise_pred
        x_prev = torch.sqrt(ab_prev) * x_0_pred + dir_xt

        if eta > 0:
            x_prev += sigma * torch.randn_like(x_t)
        return x_prev

    def get_ddim_timesteps(self, num_inference_steps: int) -> torch.Tensor:
        """Create evenly-spaced DDIM timestep schedule."""
        step_size = self.num_steps // num_inference_steps
        timesteps = torch.arange(0, self.num_steps, step_size)
        return torch.flip(timesteps, [0])