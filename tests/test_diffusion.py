import torch
import pytest
import sys
sys.path.insert(0, ".")

from src.models.diffusion import DiffusionSchedule, DiffusionConfig


@pytest.fixture
def schedule():
    return DiffusionSchedule(DiffusionConfig(num_steps=100))


def test_q_sample(schedule):
    x_0 = torch.randn(2, 24, 54)
    t = torch.tensor([10, 50])
    x_t, noise = schedule.q_sample(x_0, t)
    assert x_t.shape == x_0.shape
    assert noise.shape == x_0.shape


def test_p_sample(schedule):
    noise_pred = torch.randn(2, 24, 54)
    x_t = torch.randn(2, 24, 54)
    t = torch.tensor([10, 10])
    result = schedule.p_sample(noise_pred, x_t, t)
    assert result.shape == x_t.shape


def test_ddim_timesteps(schedule):
    steps = schedule.get_ddim_timesteps(10)
    assert len(steps) == 10
    assert steps[0] > steps[-1]