#!/usr/bin/env python3
"""Run EP09 route C: short-budget joint MAP-TV hold-out sweep."""

from __future__ import annotations

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

import numpy as np  # noqa: E402

from psf_calibration.joint_sweep import run_joint_sweep  # noqa: E402
from psf_calibration.utils import OUTPUT_DIR, default_workers, relative  # noqa: E402


def parse_sigmas(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("sigma list must not be empty")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "data_raw" / "infrared_avi")
    parser.add_argument("--frame-audit-csv", type=Path, default=PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--alignment-csv", type=Path, default=PROJECT_ROOT / "output" / "ep05_contour_alignment" / "contour_alignment_results.csv")
    parser.add_argument("--alignment-method", default="contour_refined")
    parser.add_argument("--highpass-sigma", type=float, default=5.0)
    parser.add_argument("--sigmas", type=parse_sigmas, default=None)
    parser.add_argument("--val-stride", type=int, default=5)
    parser.add_argument("--max-train", type=int, default=48)
    parser.add_argument("--max-val", type=int, default=32)
    parser.add_argument("--max-iter", type=int, default=20)
    parser.add_argument("--lambda-tv", type=float, default=0.01)
    parser.add_argument("--step-size", type=float, default=0.25)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--tv-inner-iter", type=int, default=15)
    parser.add_argument("--crop-margin", type=int, default=8)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--seed", type=int, default=2909)
    parser.add_argument("--no-save-best-hr", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    if args.sigmas is None:
        args.sigmas = np.arange(0.10, 0.6001, 0.05).round(6).tolist()
    args.save_best_hr = not args.no_save_best_hr
    delattr(args, "no_save_best_hr")
    return args


def main() -> None:
    args = parse_args()
    result = run_joint_sweep(**vars(args))
    print(f"Route C sigma: {result.summary['sigma_joint_lr_px']:.4f} LR px")
    print(f"Saved: {relative(args.output_dir / 'sigma_joint.json')}")


if __name__ == "__main__":
    main()
