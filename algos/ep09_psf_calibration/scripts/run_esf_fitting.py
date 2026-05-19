#!/usr/bin/env python3
"""Run EP09 route B: ESF fitting on EP04 contour anchors."""

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

from psf_calibration.esf_fitting import run_esf_fitting  # noqa: E402
from psf_calibration.utils import OUTPUT_DIR, default_workers, relative  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "data_raw" / "infrared_avi")
    parser.add_argument("--frame-audit-csv", type=Path, default=PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--alignment-csv", type=Path, default=PROJECT_ROOT / "output" / "ep05_contour_alignment" / "contour_alignment_results.csv")
    parser.add_argument("--alignment-method", default="contour_refined")
    parser.add_argument("--contour-segments-csv", type=Path, default=PROJECT_ROOT / "output" / "ep04_global_validation" / "inputs" / "contour_segments.csv")
    parser.add_argument("--reference-file", default=None)
    parser.add_argument("--min-contrast-c", type=float, default=2.0)
    parser.add_argument("--min-normal-projection", type=float, default=0.5)
    parser.add_argument("--min-r2", type=float, default=0.95)
    parser.add_argument("--half-width", type=float, default=10.0)
    parser.add_argument("--profile-step", type=float, default=0.25)
    parser.add_argument("--tangent-half-width", type=int, default=2)
    parser.add_argument("--max-fit-plots", type=int, default=16)
    parser.add_argument("--bootstrap", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=1909)
    parser.add_argument("--workers", type=int, default=default_workers())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_esf_fitting(**vars(args))
    print(f"Route B valid segments: {result.summary['n_valid']}/{result.summary['n_candidates']}")
    print(f"Route B sigma median: {result.summary['sigma_esf_median_lr_px']:.4f} LR px")
    print(f"Saved: {relative(args.output_dir / 'sigma_esf.json')}")


if __name__ == "__main__":
    main()
