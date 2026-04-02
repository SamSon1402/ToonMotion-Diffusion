"""Smoothness, self-penetration, physical plausibility checks."""

import numpy as np
from typing import Dict


class MotionQualityEvaluator:
    def __init__(self, fps=24):
        self.fps = fps
        self.dt = 1.0 / fps

    def smoothness(self, motion):
        if motion.shape[0] < 4:
            return 0.0
        vel = np.diff(motion, axis=0) / self.dt
        acc = np.diff(vel, axis=0) / self.dt
        jerk = np.diff(acc, axis=0) / self.dt
        return float(np.mean(np.abs(jerk)))

    def self_penetration_score(self, motion, min_dist=0.05):
        if motion.ndim == 2:
            J = motion.shape[1] // 3
            motion = motion.reshape(-1, J, 3)
        T, J, _ = motion.shape
        pen, checks = 0, 0
        for i, j in [(4, 10), (7, 13), (5, 11), (8, 14)]:
            if i < J and j < J:
                dist = np.linalg.norm(motion[:, i] - motion[:, j], axis=-1)
                pen += (dist < min_dist).sum()
                checks += T
        return float(pen) / max(checks, 1)

    def evaluate(self, motion) -> Dict:
        flat = motion.reshape(motion.shape[0], -1) if motion.ndim == 3 else motion
        return {
            "smoothness_jerk": self.smoothness(flat),
            "avg_velocity": float(np.mean(np.abs(np.diff(flat, axis=0)))) if flat.shape[0] > 1 else 0.0,
            "self_penetration_rate": self.self_penetration_score(motion),
            "num_frames": motion.shape[0],
            "duration_s": motion.shape[0] / self.fps,
        }