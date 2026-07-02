#!/usr/bin/env python3
"""Run Solver V2 Stage 0a MVP real-frame operator calibration diagnostics."""

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

from psf_calibration.stage0a_mvp import (  # noqa: E402
    DEFAULT_STAGE0A_OUTPUT_DIR,
    DEFAULT_STAGE_CONFIG,
    parse_float_list,
    run_stage0a_mvp,
)
from psf_calibration.utils import default_workers, relative  # noqa: E402
from thermal_core.alignment_paths import default_contour_alignment_csv  # noqa: E402


def _csv_floats(text: str) -> list[float]:
    try:
        return parse_float_list(text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_STAGE0A_OUTPUT_DIR)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "data_raw" / "infrared_avi")
    parser.add_argument("--frame-audit-csv", type=Path, default=PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--alignment-csv", type=Path, default=default_contour_alignment_csv(project_root_path=PROJECT_ROOT))
    parser.add_argument("--alignment-method", default="contour_refined")
    parser.add_argument("--alignment-source", choices=("auto", "contour", "stage_prior"), default="auto")
    parser.add_argument("--stage-config-path", type=Path, default=DEFAULT_STAGE_CONFIG)
    parser.add_argument("--highpass-sigma", type=float, default=5.0)
    parser.add_argument("--band-sigma-lr-px", type=float, default=5.0)
    parser.add_argument("--sigmas", type=_csv_floats, default="0.2,0.3,0.4,0.5")
    parser.add_argument("--anisotropy-ratios", type=_csv_floats, default="1.0,1.5")
    parser.add_argument("--angles", type=_csv_floats, default="0,45,90,135")
    parser.add_argument("--shift-refine-radius", type=float, default=0.10)
    parser.add_argument("--shift-refine-step", type=float, default=0.10)
    parser.add_argument("--val-stride", type=int, default=5)
    parser.add_argument("--max-train-score", type=int, default=48)
    parser.add_argument("--max-val-score", type=int, default=32)
    parser.add_argument(
        "--use-all-score",
        action="store_true",
        help="Score every deterministic train/validation frame instead of the default MVP subset.",
    )
    parser.add_argument("--crop-margin", type=int, default=8)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--huber-delta", type=float, default=0.0)
    parser.add_argument("--save-xhat", action="store_true")
    args = parser.parse_args()
    if args.use_all_score:
        args.max_train_score = None
        args.max_val_score = None
    delattr(args, "use_all_score")
    return args


def main() -> None:
    args = parse_args()
    result = run_stage0a_mvp(**vars(args))
    probe = result.summary["dc_resid_floor_probe"]
    print(f"Loaded frames: {result.summary['n_loaded_frames']}")
    print(f"Shift source: {result.summary['shift_source']}")
    print(
        "Val band RMS floor probe: "
        f"{probe['baseline_val_band_rms_from_mean_mse']:.6g} -> "
        f"{probe['best_val_band_rms_from_mean_mse']:.6g} "
        f"({probe['val_band_mse_improvement_pct']:.3f}% MSE change)"
    )
    print(f"Best candidate: {result.summary['best_train_selected']['candidate']}")
    print(f"Saved: {relative(args.output_dir / 'stage0a_summary.json')}")


if __name__ == "__main__":
    main()
