import pytest
import sys
sys.path.insert(0, ".")

from src.data.dataset import ToonMotionDataset
from src.data.preprocessing import MotionPreprocessor
from src.data.validation import DatasetValidator
import numpy as np


def test_synthetic_dataset():
    ds = ToonMotionDataset(seq_len=24, motion_dim=54, num_synthetic_samples=100)
    assert len(ds) == 100
    sample = ds[0]
    assert sample["motion"].shape == (24, 54)
    assert sample["character_id"].item() in [0, 1, 2, 3]


def test_preprocessor():
    proc = MotionPreprocessor(seq_len=24, motion_dim=54)
    motion = np.random.randn(50, 54).astype(np.float32)
    result = proc.pad_or_trim(motion)
    assert result.shape == (24, 54)


def test_validator():
    validator = DatasetValidator(motion_dim=54)
    good = np.random.randn(60, 54).astype(np.float32) * 0.5
    result = validator.validate_sample(good, "test")
    assert result["valid"] is True

    bad = np.full((60, 54), np.nan)
    result = validator.validate_sample(bad, "bad")
    assert result["valid"] is False