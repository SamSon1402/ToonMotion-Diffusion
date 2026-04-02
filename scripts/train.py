"""
Training Script
================
Usage: python scripts/train.py --config configs/default.yaml --device cuda
"""

import argparse
import torch
import yaml
import sys
sys.path.insert(0, ".")

from src.models.toonmotion import ToonMotionDiffusion, ToonMotionConfig
from src.data.dataset import ToonMotionDataset
from src.training.trainer import Trainer


def main():
    parser = argparse.ArgumentParser(description="Train ToonMotion-Diffusion")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--data_dir", default=None, help="Path to processed data")
    parser.add_argument("--output_dir", default="checkpoints")
    parser.add_argument("--num_synthetic", type=int, default=10000)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg_dict = yaml.safe_load(f)

    config = ToonMotionConfig(
        num_epochs=cfg_dict["training"]["num_epochs"],
        batch_size=cfg_dict["training"]["batch_size"],
        learning_rate=cfg_dict["training"]["learning_rate"],
        d_model=cfg_dict["model"]["d_model"],
        nhead=cfg_dict["model"]["nhead"],
        num_layers=cfg_dict["model"]["num_layers"],
        seq_len=cfg_dict["model"]["seq_len"],
        num_diffusion_steps=cfg_dict["diffusion"]["num_steps"],
        inference_steps=cfg_dict["diffusion"]["inference_steps"],
    )

    print(f"Device: {args.device}")
    print(f"Config: {config}")

    model = ToonMotionDiffusion(config, args.device)
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Parameters: {params:,}")

    dataset = ToonMotionDataset(
        data_dir=args.data_dir,
        seq_len=config.seq_len,
        motion_dim=config.motion_dim,
        num_synthetic_samples=args.num_synthetic,
    )

    trainer = Trainer(model, config, args.device, args.output_dir)
    trainer.train(dataset)


if __name__ == "__main__":
    main()