"""Batch generation for multiple prompts."""

import os
import json
import numpy as np
import logging
from typing import List, Dict
from .generate import generate_motion
from .rig_export import export_fbx_keyframes

logger = logging.getLogger("ToonMotion")


def batch_generate(model, prompts, output_dir, num_steps=50, guidance_scale=7.5, device="cpu"):
    os.makedirs(output_dir, exist_ok=True)
    results = []

    for i, item in enumerate(prompts):
        result = generate_motion(model, item["prompt"], item.get("character", "Pocoyo"), num_steps, guidance_scale, device)
        npy_path = os.path.join(output_dir, f"motion_{i:04d}.npy")
        np.save(npy_path, result["motion"])
        fbx_path = os.path.join(output_dir, f"motion_{i:04d}_fbx.json")
        export_fbx_keyframes(result["motion"], item.get("character", "Pocoyo"), fbx_path)
        result["metadata"]["files"] = {"npy": npy_path, "fbx": fbx_path}
        results.append(result["metadata"])
        logger.info(f"[{i+1}/{len(prompts)}] Done")

    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(results, f, indent=2)
    return results