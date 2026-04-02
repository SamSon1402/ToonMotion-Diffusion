"""
ToonMotionDiffusion: Complete Model
=====================================
Combines: TextEncoder + ToonAdapter + MotionTransformer + DiffusionSchedule
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List
from dataclasses import dataclass

from .diffusion import DiffusionSchedule, DiffusionConfig
from .text_encoder import TextEncoder
from .toon_adapter import ToonAdapter
from .motion_transformer import MotionTransformer


@dataclass
class ToonMotionConfig:
    num_joints: int = 18
    joint_dim: int = 3
    seq_len: int = 120
    num_diffusion_steps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02
    inference_steps: int = 50
    d_model: int = 512
    nhead: int = 8
    num_layers: int = 8
    dim_feedforward: int = 2048
    dropout: float = 0.1
    clip_dim: int = 512
    num_characters: int = 4
    adapter_dim: int = 256
    batch_size: int = 64
    learning_rate: float = 1e-4
    num_epochs: int = 500
    ema_decay: float = 0.999

    @property
    def motion_dim(self) -> int:
        return self.num_joints * self.joint_dim


class ToonMotionDiffusion(nn.Module):
    """
    Text-to-Motion diffusion model for cartoon characters.

    Pipeline:
        text -> CLIP -> text_emb
        character_id -> ToonAdapter -> char_condition, style, joint_limits
        [noise, text_emb, char_condition] -> MotionTransformer -> noise_pred
        DDPM/DDIM reverse -> clean motion
        clamp(joint_limits) + topology_bias -> valid rig controllers
    """

    def __init__(self, config: ToonMotionConfig, device: str = "cpu"):
        super().__init__()
        self.config = config

        self.text_encoder = TextEncoder(clip_dim=config.clip_dim, use_clip=False)
        self.toon_adapter = ToonAdapter(
            num_characters=config.num_characters,
            adapter_dim=config.adapter_dim,
            num_joints=config.num_joints,
            joint_dim=config.joint_dim,
            d_model=config.d_model,
        )
        self.denoiser = MotionTransformer(
            motion_dim=config.motion_dim,
            seq_len=config.seq_len,
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_ff=config.dim_feedforward,
            clip_dim=config.clip_dim,
            dropout=config.dropout,
        )
        diff_cfg = DiffusionConfig(
            config.num_diffusion_steps, config.beta_start,
            config.beta_end, config.inference_steps,
        )
        self.diffusion = DiffusionSchedule(diff_cfg, device)

    def compute_loss(
        self, motion: torch.Tensor, text: List[str], character_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Training loss: noise MSE + joint limit penalty + smoothness."""
        B = motion.shape[0]
        device = motion.device
        t = torch.randint(0, self.config.num_diffusion_steps, (B,), device=device)

        x_t, noise = self.diffusion.q_sample(motion, t)
        text_emb = self.text_encoder(text)
        char_out = self.toon_adapter(character_ids)

        noise_pred = self.denoiser(
            x_t, t, text_emb, char_out["condition"], char_out["style"]
        )

        noise_loss = F.mse_loss(noise_pred, noise)

        sqrt_ab = self.diffusion.sqrt_alpha_bars[t][:, None, None]
        sqrt_1m = self.diffusion.sqrt_one_minus_alpha_bars[t][:, None, None]
        x0_pred = (x_t - sqrt_1m * noise_pred) / sqrt_ab

        violations = (
            F.relu(char_out["joint_limits_min"].unsqueeze(1) - x0_pred) +
            F.relu(x0_pred - char_out["joint_limits_max"].unsqueeze(1))
        )
        limit_loss = violations.mean()

        if x0_pred.shape[1] > 2:
            vel = x0_pred[:, 1:] - x0_pred[:, :-1]
            acc = vel[:, 1:] - vel[:, :-1]
            smooth_loss = acc.pow(2).mean()
        else:
            smooth_loss = torch.tensor(0.0, device=device)

        total = noise_loss + 0.01 * limit_loss + 0.001 * smooth_loss

        return {
            "total_loss": total,
            "noise_loss": noise_loss,
            "limit_loss": limit_loss,
            "smooth_loss": smooth_loss,
        }

    @torch.no_grad()
    def generate(
        self, text: List[str], character_ids: torch.Tensor,
        num_steps: int = 50, guidance_scale: float = 7.5, use_ddim: bool = True,
    ) -> torch.Tensor:
        """Generate motion from text via reverse diffusion."""
        B = len(text)
        device = character_ids.device

        text_emb = self.text_encoder(text)
        char_out = self.toon_adapter(character_ids)
        x_t = torch.randn(B, self.config.seq_len, self.config.motion_dim, device=device)

        timesteps = self.diffusion.get_ddim_timesteps(num_steps) if use_ddim else \
            torch.arange(self.config.num_diffusion_steps - 1, -1, -1)

        for i, t_val in enumerate(timesteps):
            t = torch.full((B,), t_val.item(), device=device, dtype=torch.long)

            noise_pred = self.denoiser(
                x_t, t, text_emb, char_out["condition"], char_out["style"]
            )

            if guidance_scale > 1.0:
                null_emb = torch.zeros_like(text_emb)
                noise_uncond = self.denoiser(
                    x_t, t, null_emb, char_out["condition"], char_out["style"]
                )
                noise_pred = noise_uncond + guidance_scale * (noise_pred - noise_uncond)

            if use_ddim and i < len(timesteps) - 1:
                t_prev = torch.full((B,), timesteps[i + 1].item(), device=device, dtype=torch.long)
                x_t = self.diffusion.ddim_sample(noise_pred, x_t, t, t_prev)
            else:
                x_t = self.diffusion.p_sample(noise_pred, x_t, t)

        motion = torch.clamp(
            x_t,
            char_out["joint_limits_min"].unsqueeze(1),
            char_out["joint_limits_max"].unsqueeze(1),
        )
        motion = motion + char_out["topology"].unsqueeze(1)

        return motion