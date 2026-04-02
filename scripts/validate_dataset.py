"""
Dataset Validation Script
===========================
Usage: python scripts/validate_dataset.py --data_dir data/processed
"""

import argparse
import numpy as np
import json
import os
import sys
sys.path.insert(0, ".")

from src.data.validation import DatasetValidator


def main():
    parser = argparse.ArgumentParser(description="Validate motion dataset quality")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--motion_dim", type=int, default=54)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    validator = DatasetValidator(motion_dim=args.motion_dim)

    manifest_path = os.path.join(args.data_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"No manifest.json found in {args.data_dir}")
        return

    with open(manifest_path) as f:
        manifest = json.load(f)

    motions = []
    ids = []
    for item in manifest:
        npy_path = os.path.join(args.data_dir, item["motion_file"])
        if os.path.exists(npy_path):
            motions.append(np.load(npy_path))
            ids.append(item["motion_file"])

    report = validator.validate_dataset(motions, ids)

    print(f"\n--- Validation Report ---")
    print(f"  Total: {report['total']}")
    print(f"  Valid: {report['valid']}")
    print(f"  Invalid: {report['invalid']}")
    print(f"  Pass rate: {report['pass_rate']:.1f}%")

    if report["invalid_samples"]:
        print(f"\n  Failed samples:")
        for s in report["invalid_samples"][:10]:
            print(f"    {s['id']}: {', '.join(s['issues'])}")

    if args.output:
        with open(args.output, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()