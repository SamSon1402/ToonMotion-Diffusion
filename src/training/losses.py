"""
Loss Functions
===============
  1. Noise MSE (primary)
  2. Joint limit violation penalty
  3. Velocity smoothness
  4. Self-penetration loss
  5. Foot contact loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class ToonMotionLoss(nn.Module):
    def __init__(
        self, w_noise=1.0, w_limit=0.01, w_smooth=0.001, w_penetration=0.005, w_foot=0.002,
    ):
        super().__init__()
        self.w_noise = w_noise
        self.w_limit = w_limit
        self.w_smooth = w_smooth
        self.w_penetration = w_penetration
        self.w_foot = w_foot
        self.foot_joints = [12, 15]

    def noise_loss(self, pred, target):
        return F.mse_loss(pred, target)

    def joint_limit_loss(self, x0_pred, limits_min, limits_max):
        violations = F.relu(limits_min.unsqueeze(1) - x0_pred) + F.relu(x0_pred - limits_max.unsqueeze(1))
        return violations.mean()

    def smoothness_loss(self, x0_pred):
        if x0_pred.shape[1] < 3:
            return torch.tensor(0.0, device=x0_pred.device)
        vel = x0_pred[:, 1:] - x0_pred[:, :-1]
        acc = vel[:, 1:] - vel[:, :-1]
        return acc.pow(2).mean()

    def penetration_loss(self, x0_pred):
        B, T, D = x0_pred.shape
        J = D // 3
        poses = x0_pred.view(B, T, J, 3)
        min_dist = 0.05
        total = torch.tensor(0.0, device=x0_pred.device)
        for i, j in [(4, 10), (7, 13), (5, 11), (8, 14)]:
            if i < J and j < J:
                dist = (poses[:, :, i] - poses[:, :, j]).norm(dim=-1)
                total = total + F.relu(min_dist - dist).mean()
        return total

    def foot_contact_loss(self, x0_pred):
        B, T, D = x0_pred.shape
        J = D // 3
        poses = x0_pred.view(B, T, J, 3)
        loss = torch.tensor(0.0, device=x0_pred.device)
        for fi in self.foot_joints:
            if fi < J and T > 1:
                foot_vel = (poses[:, 1:, fi] - poses[:, :-1, fi]).norm(dim=-1)
                contact = (poses[:, 1:, fi, 1] < 0.1).float()
                loss = loss + (foot_vel * contact).mean()
        return loss

    def forward(self, noise_pred, noise_target, x0_pred, limits_min, limits_max):
        l_n = self.noise_loss(noise_pred, noise_target)
        l_l = self.joint_limit_loss(x0_pred, limits_min, limits_max)
        l_s = self.smoothness_loss(x0_pred)
        l_p = self.penetration_loss(x0_pred)
        l_f = self.foot_contact_loss(x0_pred)

        total = self.w_noise*l_n + self.w_limit*l_l + self.w_smooth*l_s + self.w_penetration*l_p + self.w_foot*l_f

        return {
            "total_loss": total, "noise_loss": l_n, "limit_loss": l_l,
            "smooth_loss": l_s, "penetration_loss": l_p, "foot_loss": l_f,
        }