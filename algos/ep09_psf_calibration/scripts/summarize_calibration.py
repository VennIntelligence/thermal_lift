#!/usr/bin/env python3
"""Summarize EP09 route outputs and update the global PSF config."""

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

from psf_calibration.summary import summarize_calibration  # noqa: E402
from psf_calibration.utils import CONFIG_PATH, OUTPUT_DIR, REPORT_DIR, relative  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--report-dir", type=Path, default=REPORT_DIR)
    parser.add_argument("--config-path", type=Path, default=CONFIG_PATH)
    parser.add_argument("--consistency-tolerance", type=float, default=0.05)
    parser.add_argument("--ci-width-gate", type=float, default=0.10)
    parser.add_argument("--residual-depth-gate", type=float, default=1e-3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = summarize_calibration(**vars(args))
    print(f"Final sigma: {summary['final_sigma_lr_px']:.4f} LR px")
    print(f"4x verdict: {summary['four_x_verdict']}")
    print(f"Saved: {relative(args.output_dir / 'calibration_summary.json')}")
    print(f"Updated: {relative(args.config_path)}")


if __name__ == "__main__":
    main()
