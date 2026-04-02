"""Single-prompt motion generation."""

import torch
import numpy as np
import time
import logging
from typing import Dict
from ..models.toonmotion import ToonMotionDiffusion

logger = logging.getLogger("ToonMotion")
CHARACTER_MAP = {"Pocoyo": 0, "Elly": 1, "Pato": 2, "Maya": 3}


@torch.no_grad()
def generate_motion(model, prompt, character="Pocoyo", num_steps=50, guidance_scale=7.5, device="cpu"):
    model.eval()
    char_ids = torch.tensor([CHARACTER_MAP.get(character, 0)], device=device)

    t0 = time.time()
    motion = model.generate(text=[prompt], character_ids=char_ids, num_steps=num_steps, guidance_scale=guidance_scale)
    elapsed_ms = (time.time() - t0) * 1000

    motion_np = motion[0].cpu().numpy()
    T, D = motion_np.shape
    J = D // 3

    return {
        "motion": motion_np.reshape(T, J, 3),
        "motion_flat": motion_np,
        "metadata": {
            "prompt": prompt, "character": character, "num_frames": T,
            "num_joints": J, "fps": 24, "duration_seconds": T / 24,
            "inference_steps": num_steps, "guidance_scale": guidance_scale,
            "inference_time_ms": elapsed_ms,
        },
    }