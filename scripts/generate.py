"""
Generation Script
==================
Usage: python scripts/generate.py --checkpoint checkpoints/best.pt --prompt "Pocoyo jumps and waves"
"""

import argparse
import torch
import sys
sys.path.insert(0, ".")

from src.training.trainer import Trainer
from src.inference.generate import generate_motion
from src.inference.rig_export import export_fbx_keyframes, export_csv


def main():
    parser = argparse.ArgumentParser(description="Generate motion from text")
    parser.add_argument("--checkpoint", default="checkpoints/best.pt")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--character", default="Pocoyo")
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--output", default="outputs/generated")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model, config = Trainer.load_checkpoint(args.checkpoint, args.device)
    model.to(args.device)

    result = generate_motion(model, args.prompt, args.character, args.steps, args.guidance, args.device)

    export_fbx_keyframes(result["motion"], args.character, f"{args.output}_fbx.json")
    export_csv(result["motion"], args.character, f"{args.output}.csv")

    print(f"\nGenerated: {result['metadata']['num_frames']} frames")
    print(f"Duration: {result['metadata']['duration_seconds']:.1f}s")
    print(f"Inference: {result['metadata']['inference_time_ms']:.0f}ms")
    print(f"Saved: {args.output}_fbx.json, {args.output}.csv")


if __name__ == "__main__":
    main()