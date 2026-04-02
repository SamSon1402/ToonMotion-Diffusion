"""Ablation study runner - systematically test model components."""

import json
import os
import logging
from typing import Dict, List
from dataclasses import dataclass

logger = logging.getLogger("ToonMotion")


@dataclass
class AblationConfig:
    name: str
    description: str
    params: Dict


class AblationRunner:
    STANDARD_ABLATIONS = [
        AblationConfig("no_adapter", "Without Toon-Adapter (baseline)", {"use_adapter": False}),
        AblationConfig("no_cfg", "Without classifier-free guidance", {"guidance_scale": 1.0}),
        AblationConfig("low_cfg", "Low guidance", {"guidance_scale": 3.0}),
        AblationConfig("high_cfg", "High guidance", {"guidance_scale": 15.0}),
        AblationConfig("steps_10", "10 DDIM steps", {"inference_steps": 10}),
        AblationConfig("steps_25", "25 DDIM steps", {"inference_steps": 25}),
        AblationConfig("no_limit_loss", "Without joint limit loss", {"w_limit": 0.0}),
        AblationConfig("no_smooth_loss", "Without smoothness loss", {"w_smooth": 0.0}),
        AblationConfig("4_layers", "4 transformer layers", {"num_layers": 4}),
    ]

    def __init__(self, output_dir="ablation_results"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.results = []

    def run(self, ablation, metrics):
        self.results.append({"name": ablation.name, "description": ablation.description, "params": ablation.params, "metrics": metrics})
        logger.info(f"Ablation '{ablation.name}': {metrics}")

    def compare(self):
        return sorted(self.results, key=lambda r: r["metrics"].get("fid", float("inf")))

    def save_report(self):
        path = os.path.join(self.output_dir, "ablation_report.json")
        with open(path, "w") as f:
            json.dump({"experiments": self.compare(), "best": self.compare()[0]["name"] if self.results else None}, f, indent=2)
        return path