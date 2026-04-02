import torch
import numpy as np
import pytest
import sys
sys.path.insert(0, ".")

from src.models.toonmotion import ToonMotionDiffusion, ToonMotionConfig
from src.inference.generate import generate_motion
from src.inference.rig_export import motion_to_rig_controllers


@pytest.fixture
def model():
    config = ToonMotionConfig(
        seq_len=24, d_model=128, nhead=4, num_layers=2,
        dim_feedforward=256, clip_dim=128, adapter_dim=64,
        num_diffusion_steps=20, inference_steps=5,
    )
    return ToonMotionDiffusion(config)


def test_generate_motion(model):
    result = generate_motion(model, "Pocoyo jumps", "Pocoyo", num_steps=5, guidance_scale=3.0)
    assert result["motion"].shape[1] == 18
    assert result["motion"].shape[2] == 3
    assert result["metadata"]["character"] == "Pocoyo"


def test_rig_export():
    motion = np.random.randn(24, 18, 3)
    controllers = motion_to_rig_controllers(motion, "Pocoyo")
    assert len(controllers) == 54
    assert "Pocoyo:Head_ctrl.rotateX" in controllers
    assert len(controllers["Pocoyo:Head_ctrl.rotateX"]) == 24