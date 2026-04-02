"""
ToonMotion Dataset
===================
Paired (text_description, motion_sequence, character_id) dataset.
Production: motion extracted from Maya animation files (770K+ frames).
"""

import torch
from torch.utils.data import Dataset
import numpy as np
import json
import os
import logging
from typing import List, Dict, Optional

logger = logging.getLogger("ToonMotion")

CHARACTER_NAMES = ["Pocoyo", "Elly", "Pato", "Maya"]

MOTION_TEMPLATES = [
    "{char} jumps excitedly and waves both arms",
    "{char} walks forward slowly and stops",
    "{char} does a happy victory dance",
    "{char} looks around curiously then points up",
    "{char} sits down and crosses arms",
    "{char} runs in a circle and falls dizzy",
    "{char} hugs with both arms open wide",
    "{char} claps hands rhythmically",
    "{char} bends down to pick something up",
    "{char} spins around then strikes a pose",
    "{char} tiptoes carefully and peeks around corner",
    "{char} shakes head no and stamps foot",
    "{char} nods enthusiastically and bounces",
    "{char} stretches arms up and yawns",
    "{char} waves goodbye with one hand",
]


class ToonMotionDataset(Dataset):
    """
    Two modes:
    1. Real data: Load from processed Maya exports (JSON/NPY)
    2. Synthetic: Generate paired data for prototyping
    """

    def __init__(
        self,
        data_dir: Optional[str] = None,
        seq_len: int = 120,
        motion_dim: int = 54,
        num_characters: int = 4,
        num_synthetic_samples: int = 10000,
    ):
        self.seq_len = seq_len
        self.motion_dim = motion_dim
        self.num_characters = num_characters

        if data_dir and os.path.exists(data_dir):
            self.data = self._load_real_data(data_dir)
            logger.info(f"Loaded {len(self.data)} real samples from {data_dir}")
        else:
            self.data = self._generate_synthetic(num_synthetic_samples)
            logger.info(f"Generated {len(self.data)} synthetic samples")

    def _load_real_data(self, data_dir: str) -> List[Dict]:
        data = []
        manifest_path = os.path.join(data_dir, "manifest.json")

        if os.path.exists(manifest_path):
            with open(manifest_path) as f:
                manifest = json.load(f)

            for item in manifest:
                motion_path = os.path.join(data_dir, item["motion_file"])
                motion = np.load(motion_path)

                if motion.shape[0] > self.seq_len:
                    start = np.random.randint(0, motion.shape[0] - self.seq_len)
                    motion = motion[start:start + self.seq_len]
                elif motion.shape[0] < self.seq_len:
                    pad = np.zeros((self.seq_len - motion.shape[0], self.motion_dim))
                    motion = np.concatenate([motion, pad], axis=0)

                data.append({
                    "text": item["text"],
                    "motion": motion.astype(np.float32),
                    "character_id": item["character_id"],
                })
        return data

    def _generate_synthetic(self, n: int) -> List[Dict]:
        data = []
        char_freq = [1.0, 0.8, 1.2, 0.9]
        char_amp = [1.0, 1.3, 0.8, 1.1]

        for i in range(n):
            cid = i % self.num_characters
            text = MOTION_TEMPLATES[i % len(MOTION_TEMPLATES)].format(
                char=CHARACTER_NAMES[cid]
            )

            t = np.linspace(0, 2 * np.pi, self.seq_len)
            motion = np.zeros((self.seq_len, self.motion_dim), dtype=np.float32)
            for j in range(self.motion_dim):
                phase = np.random.uniform(0, 2 * np.pi)
                freq = np.random.uniform(0.5, 3.0) * char_freq[cid]
                amp = np.random.uniform(0.3, 1.0) * char_amp[cid]
                motion[:, j] = amp * np.sin(freq * t + phase)

            data.append({"text": text, "motion": motion, "character_id": cid})
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "text": item["text"],
            "motion": torch.tensor(item["motion"], dtype=torch.float32),
            "character_id": torch.tensor(item["character_id"], dtype=torch.long),
        }