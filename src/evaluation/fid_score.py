"""FID score adapted for motion sequences."""

import numpy as np
import torch
import torch.nn as nn
from typing import List
from .metrics import MotionMetrics


class MotionFeatureExtractor(nn.Module):
    def __init__(self, motion_dim=54, feature_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(motion_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, feature_dim),
        )

    def forward(self, motion):
        return self.encoder(motion.mean(dim=1))


def compute_fid_score(real_motions, generated_motions, extractor=None):
    if extractor is None:
        D = real_motions[0].reshape(real_motions[0].shape[0], -1).shape[-1]
        extractor = MotionFeatureExtractor(motion_dim=D)
        extractor.eval()

    def extract(motions):
        feats = []
        for m in motions:
            flat = m.reshape(m.shape[0], -1) if m.ndim == 3 else m
            t = torch.tensor(flat, dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                feats.append(extractor(t).squeeze(0).numpy())
        return np.array(feats)

    return MotionMetrics.compute_fid(extract(real_motions), extract(generated_motions))