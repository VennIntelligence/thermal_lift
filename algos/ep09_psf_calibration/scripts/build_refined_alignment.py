#!/usr/bin/env python3
"""Build a refined alignment CSV from Stage 0a per-frame shift refinements.

Stage 0f found the per-frame shift error against ``contour_refined`` to be a
zero-mean annulus of |delta| ~ 0.25-0.40 LR px (ACL-046/047).  This tool feeds
the Stage 0a refinements back as a drop-in alignment artifact: it merges the
base contour alignment CSV with ``stage0a_best_shift_refinements.csv`` and
emits a CSV with the same ``refined_align_dx_px/refined_align_dy_px`` schema,
so every consumer that accepts ``--alignment-csv`` (stage0a, run_real_split_frc_v2,
run_m2_frc, SAA/drizzle scripts) can use the refined shifts unchanged via
``--alignment-method contour_refined``.

Interpretation discipline: the refinements were fit against a shared SAA x̂,
so downstream split-half FRC gains must be validated against the M2 negative
controls (shift-shuffle / drift), not read as standalone proof.
"""

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
import pandas as pd  # noqa: E402

from psf_calibration.refined_alignment import build_refined_alignment  # noqa: E402
from thermal_core.alignment_paths import default_contour_alignment_csv  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--refinements-csv",
        type=Path,
        required=True,
        help="stage0a_best_shift_refinements.csv from a --use-all-score centered run",
    )
    parser.add_argument(
        "--base-alignment-csv",
        type=Path,
        default=default_contour_alignment_csv(project_root_path=PROJECT_ROOT),
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--expected-frames", type=int, default=248, help="0 disables the frame-count check")
    parser.add_argument("--max-initial-mismatch-px", type=float, default=1e-3)
    args = parser.parse_args()

    out = build_refined_alignment(
        refinements_csv=args.refinements_csv,
        base_alignment_csv=args.base_alignment_csv,
        expected_frames=args.expected_frames if args.expected_frames > 0 else None,
        max_initial_mismatch_px=args.max_initial_mismatch_px,
    )
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    delta = np.hypot(out["stage0a_delta_dx_px"], out["stage0a_delta_dy_px"])
    print(f"rows: {len(out)}")
    print(
        f"applied delta: mean={float(delta.mean()):.4f} px, p95={float(np.percentile(delta, 95)):.4f} px, "
        f"axis means=({float(out['stage0a_delta_dx_px'].mean()):+.4f}, {float(out['stage0a_delta_dy_px'].mean()):+.4f})"
    )
    print(f"wrote {args.output_csv}")
    print("use downstream via: --alignment-csv <output> --alignment-method contour_refined")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
