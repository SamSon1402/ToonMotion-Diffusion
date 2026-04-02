"""
Evaluation Script
==================
Usage: python scripts/evaluate.py --checkpoint checkpoints/best.pt --data_dir data/processed
"""

import argparse
import torch
import numpy as np
import json
import sys
sys.path.insert(0, ".")

from src.training.trainer import Trainer
from src.inference.generate import generate_motion
from src.evaluation.metrics import MotionMetrics
from src.evaluation.motion_quality import MotionQualityEvaluator
from src.data.dataset import ToonMotionDataset


def main():
    parser = argparse.ArgumentParser(description="Evaluate ToonMotion model")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--data_dir", default=None)
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--output", default="outputs/eval_report.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, config = Trainer.load_checkpoint(args.checkpoint, args.device)
    model.to(args.device)

    # Load test data
    dataset = ToonMotionDataset(
        data_dir=args.data_dir,
        seq_len=config.seq_len,
        motion_dim=config.motion_dim,
        num_synthetic_samples=args.num_samples,
    )

    quality_eval = MotionQualityEvaluator(fps=24)

    # Generate motions and evaluate
    real_motions = []
    gen_motions = []
    quality_scores = []

    for i in range(min(args.num_samples, len(dataset))):
        sample = dataset[i]
        real_motions.append(sample["motion"].numpy())

        result = generate_motion(
            model, sample["text"],
            ["Pocoyo", "Elly", "Pato", "Maya"][sample["character_id"].item()],
            num_steps=50, guidance_scale=7.5, device=args.device,
        )
        gen_motions.append(result["motion_flat"])
        quality_scores.append(quality_eval.evaluate(result["motion"]))

        if (i + 1) % 20 == 0:
            print(f"  Evaluated {i+1}/{args.num_samples}")

    # Compute metrics
    diversity = MotionMetrics.diversity(gen_motions)
    avg_smoothness = np.mean([q["smoothness_jerk"] for q in quality_scores])
    avg_penetration = np.mean([q["self_penetration_rate"] for q in quality_scores])

    report = {
        "num_samples": len(gen_motions),
        "diversity": float(diversity),
        "avg_smoothness_jerk": float(avg_smoothness),
        "avg_self_penetration_rate": float(avg_penetration),
    }

    with open(args.output, "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n--- Evaluation Report ---")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()