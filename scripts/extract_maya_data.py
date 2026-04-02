"""
Maya Data Extraction Script
==============================
Usage (inside Maya Python): python scripts/extract_maya_data.py --scene_dir /path/to/scenes --output_dir data/processed
"""

import argparse
import sys
sys.path.insert(0, ".")

from src.data.maya_extractor import MayaExtractor, POCOYO_RIG


def main():
    parser = argparse.ArgumentParser(description="Extract motion data from Maya scenes")
    parser.add_argument("--scene_dir", required=True, help="Directory of .ma/.mb files")
    parser.add_argument("--output_dir", default="data/processed")
    parser.add_argument("--character", default="Pocoyo", choices=["Pocoyo", "Elly", "Pato", "Maya"])
    parser.add_argument("--fps", type=int, default=24)
    args = parser.parse_args()

    extractor = MayaExtractor(rig=POCOYO_RIG, fps=args.fps)
    manifest = extractor.extract_batch(args.scene_dir, args.output_dir)
    print(f"\nExtracted {len(manifest)} scenes to {args.output_dir}")
    print("NOTE: Text annotations must be added manually to manifest.json")


if __name__ == "__main__":
    main()