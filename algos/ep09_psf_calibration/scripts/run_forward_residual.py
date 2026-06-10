#!/usr/bin/env python3
"""Run EP09 route A: forward-model residual PSF sigma sweep."""

from __future__ import annotations

from thermal_core.alignment_paths import default_contour_alignment_csv
import argparse
import sys
from pathlib import Path


def bootstrap() -> Path:
    path = Path(__file__).resolve()
    root = path
    while not (root / "AGENTS.md").exists() and root != root.parent:
        root = root.parent
    for add in [path.parents[1] / "src", root / "algos" / "ep06_sr_poc" / "src", root / "core" / "src"]:
        text = str(add)
        if text not in sys.path:
            sys.path.insert(0, text)
    return root


PROJECT_ROOT = bootstrap()

from psf_calibration.forward_sweep import run_forward_sweep  # noqa: E402
from psf_calibration.utils import OUTPUT_DIR, default_workers, relative  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "data_raw" / "infrared_avi")
    parser.add_argument("--frame-audit-csv", type=Path, default=PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--alignment-csv", type=Path, default=default_contour_alignment_csv(project_root_path=PROJECT_ROOT))
    parser.add_argument("--alignment-method", default="contour_refined")
    parser.add_argument("--hr-path", type=Path, default=None)
    parser.add_argument("--highpass-sigma", type=float, default=5.0)
    parser.add_argument("--coarse-min", type=float, default=0.10)
    parser.add_argument("--coarse-max", type=float, default=0.60)
    parser.add_argument("--coarse-step", type=float, default=0.02)
    parser.add_argument("--fine-half-width", type=float, default=0.05)
    parser.add_argument("--fine-step", type=float, default=0.005)
    parser.add_argument("--val-stride", type=int, default=5)
    parser.add_argument("--crop-margin", type=int, default=8)
    parser.add_argument("--bootstrap", type=int, default=300)
    parser.add_argument("--seed", type=int, default=909)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_forward_sweep(**vars(args))
    print(f"Route A sigma: {result.summary['sigma_forward_lr_px']:.4f} LR px")
    print(f"Saved: {relative(args.output_dir / 'sigma_forward.json')}")


if __name__ == "__main__":
    main()
