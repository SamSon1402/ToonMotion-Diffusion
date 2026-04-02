"""
Motion Transformer: Denoising Backbone
========================================
Predicts noise given noisy motion, timestep, text embedding, and character.
  - Self-attention over temporal motion sequence
  - Cross-attention with text conditioning
  - Adaptive LayerNorm (conditioned on timestep + character)
"""

import torch
import torch.nn as nn
import math


class SinusoidalTimestepEmbedding(nn.Module):
    """Sinusoidal encoding for diffusion timesteps."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
        args = t[:, None].float() * freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class MotionTransformerBlock(nn.Module):
    """Single block: self-attn -> cross-attn (text) -> FFN with Adaptive LayerNorm."""

    def __init__(self, d_model: int, nhead: int, dim_ff: int, dropout: float = 0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)

        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_ff), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(dim_ff, d_model), nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

        self.adaLN = nn.Sequential(nn.SiLU(), nn.Linear(d_model, d_model * 6))

    def forward(self, x: torch.Tensor, text_cond: torch.Tensor, time_cond: torch.Tensor) -> torch.Tensor:
        s1, sh1, s2, sh2, s3, sh3 = self.adaLN(time_cond).chunk(6, dim=-1)

        h = self.norm1(x) * (1 + s1[:, None, :]) + sh1[:, None, :]
        x = x + self.self_attn(h, h, h)[0]

        h = self.norm2(x) * (1 + s2[:, None, :]) + sh2[:, None, :]
        x = x + self.cross_attn(h, text_cond, text_cond)[0]

        h = self.norm3(x) * (1 + s3[:, None, :]) + sh3[:, None, :]
        x = x + self.ffn(h)

        return x


class MotionTransformer(nn.Module):
    """Full transformer denoiser: predicts noise epsilon_theta(x_t, t, text, char)."""

    def __init__(
        self, motion_dim: int = 54, seq_len: int = 120, d_model: int = 512,
        nhead: int = 8, num_layers: int = 8, dim_ff: int = 2048,
        clip_dim: int = 512, dropout: float = 0.1,
    ):
        super().__init__()
        self.input_proj = nn.Linear(motion_dim, d_model)
        self.temporal_pos = nn.Parameter(torch.randn(1, seq_len, d_model) * 0.02)

        self.time_embed = nn.Sequential(
            SinusoidalTimestepEmbedding(d_model),
            nn.Linear(d_model, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.text_proj = nn.Sequential(
            nn.Linear(clip_dim, d_model), nn.GELU(), nn.Linear(d_model, d_model),
        )

        self.blocks = nn.ModuleList([
            MotionTransformerBlock(d_model, nhead, dim_ff, dropout)
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(d_model)
        self.output_proj = nn.Linear(d_model, motion_dim)

    def forward(
        self, x_t: torch.Tensor, t: torch.Tensor, text_emb: torch.Tensor,
        char_cond: torch.Tensor, char_style: torch.Tensor,
    ) -> torch.Tensor:
        B, T, _ = x_t.shape
        x = self.input_proj(x_t) + self.temporal_pos[:, :T, :]

        time_cond = self.time_embed(t) + char_cond + char_style
        text_cond = self.text_proj(text_emb).unsqueeze(1)

        for block in self.blocks:
            x = block(x, text_cond, time_cond)

        return self.output_proj(self.output_norm(x))