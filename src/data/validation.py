"""
Dataset Validation
===================
Quality checks: corrupted files, wrong dims, NaN, outliers, temporal discontinuities.
"""

import numpy as np
import logging
from typing import List, Dict

logger = logging.getLogger("ToonMotion")


class DatasetValidator:
    """Validates motion dataset quality before training."""

    def __init__(self, motion_dim: int = 54, seq_len: int = 120, fps: int = 24):
        self.motion_dim = motion_dim
        self.seq_len = seq_len
        self.fps = fps
        self.max_velocity = 10.0
        self.max_value = 180.0

    def validate_sample(self, motion: np.ndarray, sample_id: str = "") -> Dict:
        issues = []

        if motion.ndim != 2 or motion.shape[1] != self.motion_dim:
            issues.append(f"Wrong shape: {motion.shape}")

        if np.any(np.isnan(motion)):
            issues.append(f"Contains {np.isnan(motion).sum()} NaN values")

        if np.any(np.isinf(motion)):
            issues.append(f"Contains {np.isinf(motion).sum()} Inf values")

        max_val = np.abs(motion).max()
        if max_val > self.max_value:
            issues.append(f"Values exceed range: max={max_val:.1f}")

        if motion.shape[0] > 1:
            velocity = np.diff(motion, axis=0)
            max_vel = np.abs(velocity).max()
            if max_vel > self.max_velocity:
                issues.append(f"Velocity spike: {max_vel:.2f}")

        if np.std(motion) < 1e-6:
            issues.append("Static motion (near-zero variance)")

        duration = motion.shape[0] / self.fps
        if duration < 0.5:
            issues.append(f"Too short: {duration:.1f}s")

        return {
            "sample_id": sample_id,
            "valid": len(issues) == 0,
            "issues": issues,
            "stats": {
                "frames": motion.shape[0],
                "duration_s": duration,
                "mean": float(motion.mean()),
                "std": float(motion.std()),
                "max_abs": float(max_val) if not np.isnan(max_val) else 0.0,
            },
        }

    def validate_dataset(self, motions: List[np.ndarray], sample_ids: List[str] = None) -> Dict:
        if sample_ids is None:
            sample_ids = [f"sample_{i}" for i in range(len(motions))]

        results = [self.validate_sample(m, sid) for m, sid in zip(motions, sample_ids)]
        valid = [r for r in results if r["valid"]]
        invalid = [r for r in results if not r["valid"]]

        report = {
            "total": len(results),
            "valid": len(valid),
            "invalid": len(invalid),
            "pass_rate": len(valid) / max(len(results), 1) * 100,
            "invalid_samples": [{"id": r["sample_id"], "issues": r["issues"]} for r in invalid],
        }

        logger.info(f"Validation: {report['valid']}/{report['total']} passed ({report['pass_rate']:.1f}%)")
        return report