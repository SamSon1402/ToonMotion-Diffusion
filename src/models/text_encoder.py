"""
Text Encoder (CLIP Wrapper)
============================
Encodes text prompts into dense embeddings for conditioning the diffusion model.
Uses pretrained CLIP ViT-B/32 when available, with a learned fallback encoder.
"""

import torch
import torch.nn as nn
import logging
from typing import List

logger = logging.getLogger("ToonMotion")


class TextEncoder(nn.Module):
    """CLIP-based text encoder with lightweight fallback."""

    def __init__(self, clip_dim: int = 512, use_clip: bool = True):
        super().__init__()
        self.clip_dim = clip_dim
        self.use_clip = use_clip
        self._clip_model = None

        if use_clip:
            try:
                import clip
                self._clip_model, _ = clip.load("ViT-B/32", device="cpu")
                self._clip_model.eval()
                for p in self._clip_model.parameters():
                    p.requires_grad = False
                logger.info("CLIP ViT-B/32 loaded")
            except ImportError:
                logger.warning("CLIP not available, using learned fallback")
                self.use_clip = False

        if not self.use_clip:
            self.token_embedding = nn.Embedding(5000, clip_dim)
            self.pos_encoding = nn.Parameter(torch.randn(1, 77, clip_dim) * 0.02)
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=clip_dim, nhead=8, dim_feedforward=1024,
                    dropout=0.1, batch_first=True,
                ),
                num_layers=4,
            )
            self.proj = nn.Linear(clip_dim, clip_dim)

    def forward(self, text: List[str]) -> torch.Tensor:
        """Encode text to [B, clip_dim]."""
        if self.use_clip and self._clip_model is not None:
            import clip
            tokens = clip.tokenize(text, truncate=True)
            tokens = tokens.to(next(self._clip_model.parameters()).device)
            with torch.no_grad():
                return self._clip_model.encode_text(tokens).float()

        batch = []
        for t in text:
            ids = [hash(w) % 5000 for w in t.lower().split()[:77]]
            ids += [0] * (77 - len(ids))
            batch.append(ids)

        tokens = torch.tensor(batch, dtype=torch.long)
        tokens = tokens.to(self.token_embedding.weight.device)

        x = self.token_embedding(tokens) + self.pos_encoding
        x = self.encoder(x)
        x = x.mean(dim=1)
        return self.proj(x)