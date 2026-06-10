#!/usr/bin/env python3
"""Build EP14 4x Loss Atlas: TCForge 4x training demo + loss figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from build_ep14_tcforge_demo import build_tcforge_training_demo, save_training_demo_bundle
from thermal_core.ep14_loss_atlas import build_loss_atlas_figures_4x


def main() -> None:
    parser = argparse.ArgumentParser(description="Build EP14 4x loss atlas cache.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "ep14_4x_loss_atlas")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    manifest_path = args.output_dir / "loss_breakdown.json"
    if manifest_path.exists() and not args.force:
        print(f"EP14 cache already exists: {manifest_path}")
        print("Use --force to rebuild.")
        return

    demo = build_tcforge_training_demo(PROJECT_ROOT)
    npz_path = save_training_demo_bundle(args.output_dir, demo)
    print(f"Saved TCForge 4x demo bundle: {npz_path}")
    print(f"rotation_deg={demo['rotation_deg']:.2f}, demo_frames={demo['lr_burst'].shape[0]}")

    manifest = build_loss_atlas_figures_4x(args.output_dir)
    print(f"Built {len(manifest['figures'])} figures under {args.output_dir}")
    print(f"Total demo loss = {manifest['loss_breakdown']['total']:.6f}")


if __name__ == "__main__":
    main()
