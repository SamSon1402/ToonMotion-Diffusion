"""
Motion Data Preprocessing
==========================
Clean, normalize, and prepare raw motion data from Maya files.
"""

import numpy as np
import logging
from typing import Optional

logger = logging.getLogger("ToonMotion")


class MotionPreprocessor:
    """Preprocesses raw rig controller values into training-ready format."""

    def __init__(self, seq_len: int = 120, motion_dim: int = 54):
        self.seq_len = seq_len
        self.motion_dim = motion_dim
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None

    def clean(self, motion: np.ndarray) -> np.ndarray:
        """Remove NaN, Inf, and clip extreme outliers."""
        motion = np.nan_to_num(motion, nan=0.0, posinf=0.0, neginf=0.0)
        if self.mean is not None and self.std is not None:
            lower = self.mean - 5 * self.std
            upper = self.mean + 5 * self.std
            motion = np.clip(motion, lower, upper)
        else:
            motion = np.clip(motion, -10.0, 10.0)
        return motion

    def fit(self, motions: list) -> None:
        """Compute normalization statistics from training set."""
        all_frames = np.concatenate(motions, axis=0)
        self.mean = all_frames.mean(axis=0)
        self.std = all_frames.std(axis=0) + 1e-8
        logger.info(f"Fit on {len(all_frames)} frames")

    def normalize(self, motion: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise ValueError("Call fit() first")
        return (motion - self.mean) / self.std

    def denormalize(self, motion: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise ValueError("Call fit() first")
        return motion * self.std + self.mean

    def pad_or_trim(self, motion: np.ndarray) -> np.ndarray:
        T = motion.shape[0]
        if T > self.seq_len:
            start = np.random.randint(0, T - self.seq_len)
            return motion[start:start + self.seq_len]
        elif T < self.seq_len:
            pad = np.zeros((self.seq_len - T, self.motion_dim))
            return np.concatenate([motion, pad], axis=0)
        return motion

    def process(self, motion: np.ndarray) -> np.ndarray:
        """Full pipeline: clean -> pad/trim -> normalize."""
        motion = self.clean(motion)
        motion = self.pad_or_trim(motion)
        if self.mean is not None:
            motion = self.normalize(motion)
        return motion.astype(np.float32)

    def save_stats(self, path: str) -> None:
        np.savez(path, mean=self.mean, std=self.std)

    def load_stats(self, path: str) -> None:
        data = np.load(path)
        self.mean = data["mean"]
        self.std = data["std"]