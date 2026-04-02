"""
Toon-Adapter: Character-Specific Rig Adaptation
=================================================
The key innovation for non-human cartoon characters.
Standard motion models assume SMPL human skeleton.
Cartoon characters have non-standard proportions:
  - Pocoyo: giant head, tiny body (head:body ~1:1.5 vs human ~1:7)
  - Pato: duck body, wide stance
  - Elly: elephant proportions
  - Maya: insect body, wing-like arms

The adapter learns per-character:
  1. Joint topology embedding (rig structure)
  2. Joint limit constraints (valid range of motion)
  3. Style embedding (snappy vs smooth, exaggerated vs subtle)
"""

import torch
import torch.nn as nn
from typing import Dict


class ToonAdapter(nn.Module):
    """Character-specific conditioning for the diffusion model."""

    def __init__(
        self,
        num_characters: int = 4,
        adapter_dim: int = 256,
        num_joints: int = 18,
        joint_dim: int = 3,
        d_model: int = 512,
    ):
        super().__init__()
        self.num_joints = num_joints
        self.joint_dim = joint_dim
        motion_dim = num_joints * joint_dim

        self.char_embedding = nn.Embedding(num_characters, adapter_dim)

        self.topology_encoder = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, motion_dim),
        )

        self.joint_limits = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, motion_dim * 2),
        )

        self.style_encoder = nn.Sequential(
            nn.Linear(adapter_dim, adapter_dim),
            nn.GELU(),
            nn.Linear(adapter_dim, d_model),
        )

        self.projection = nn.Linear(adapter_dim, d_model)

    def forward(self, character_ids: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            character_ids: [B] indices (0=Pocoyo, 1=Elly, 2=Pato, 3=Maya)
        Returns:
            dict with condition, style, joint_limits_min/max, topology
        """
        emb = self.char_embedding(character_ids)

        topology = self.topology_encoder(emb)
        limits = self.joint_limits(emb)
        limits_min, limits_max = limits.chunk(2, dim=-1)
        style = self.style_encoder(emb)
        condition = self.projection(emb)

        return {
            "condition": condition,
            "style": style,
            "joint_limits_min": limits_min,
            "joint_limits_max": limits_max,
            "topology": topology,
        }