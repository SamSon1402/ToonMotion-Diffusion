"""
Motion Data Augmentation
=========================
Augmentation strategies preserving physical plausibility.
"""

import numpy as np
from typing import Optional


class MotionAugmentor:
    """Applies augmentations to motion sequences during training."""

    def __init__(self, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)

    def mirror(self, motion: np.ndarray, joint_pairs: list) -> np.ndarray:
        """Mirror motion left-right by swapping joint pairs."""
        mirrored = motion.copy()
        for left, right in joint_pairs:
            l_s, r_s = left * 3, right * 3
            mirrored[:, l_s:l_s+3], mirrored[:, r_s:r_s+3] = \
                motion[:, r_s:r_s+3].copy(), motion[:, l_s:l_s+3].copy()
            mirrored[:, l_s] *= -1
            mirrored[:, r_s] *= -1
        return mirrored

    def time_warp(self, motion: np.ndarray, factor_range: tuple = (0.8, 1.2)) -> np.ndarray:
        """Speed up or slow down by resampling temporal axis."""
        T, D = motion.shape
        factor = self.rng.uniform(*factor_range)
        new_T = int(T * factor)
        indices = np.linspace(0, T - 1, new_T)
        warped = np.zeros((new_T, D))
        for d in range(D):
            warped[:, d] = np.interp(indices, np.arange(T), motion[:, d])
        return warped

    def add_noise(self, motion: np.ndarray, sigma: float = 0.01) -> np.ndarray:
        return motion + self.rng.randn(*motion.shape).astype(np.float32) * sigma

    def random_crop(self, motion: np.ndarray, crop_len: int) -> np.ndarray:
        T = motion.shape[0]
        if T <= crop_len:
            return motion
        start = self.rng.randint(0, T - crop_len)
        return motion[start:start + crop_len]

    def joint_dropout(self, motion: np.ndarray, drop_prob: float = 0.05) -> np.ndarray:
        augmented = motion.copy()
        num_joints = motion.shape[1] // 3
        for j in range(num_joints):
            if self.rng.random() < drop_prob:
                augmented[:, j*3:(j+1)*3] = 0.0
        return augmented

    def augment(self, motion: np.ndarray, joint_pairs: list = None) -> np.ndarray:
        """Apply random subset of augmentations."""
        if self.rng.random() < 0.3 and joint_pairs:
            motion = self.mirror(motion, joint_pairs)
        if self.rng.random() < 0.3:
            motion = self.time_warp(motion)
        if self.rng.random() < 0.5:
            motion = self.add_noise(motion, sigma=0.005)
        if self.rng.random() < 0.2:
            motion = self.joint_dropout(motion)
        return motion