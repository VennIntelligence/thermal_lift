#!/usr/bin/env python3
"""Diagnose filename-affine outliers and robust refit impact for EP05.

用法: uv run python scripts/diagnose_affine_outliers.py
      [--alignment-csv CSV] [--frame-audit-csv CSV] [--output-dir DIR]
      [--outlier-threshold 3.0] [--top 20]
输入: EP05 contour alignment CSV（默认 thermal_core.alignment_paths.default_contour_alignment_csv()）、
      output/ep01_data_processing/frame_audit.csv（仅存在性检查）
输出: output/ep05_overlay_alignment/affine_outlier_diagnosis.csv + 终端统计与 Top-N 离群报告
关联: EP05
"""

from __future__ import annotations

from thermal_core.alignment_paths import default_contour_alignment_csv
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from thermal_core.ep05 import filename_affine_diagnostics, fit_filename_affine


def project_root() -> Path:
    root = Path.cwd()
    while root != root.parent and not (root / "AGENTS.md").exists():
        root = root.parent
    return root


def quantile_stats(values: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"count": 0}
    return {
        "count": int(clean.size),
        "median": float(clean.median()),
        "mean": float(clean.mean()),
        "p90": float(clean.quantile(0.90)),
        "p95": float(clean.quantile(0.95)),
        "p99": float(clean.quantile(0.99)),
        "max": float(clean.max()),
    }


def format_beta(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):+.6f}" for v in values) + "]"


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--alignment-csv",
        type=Path,
        default=default_contour_alignment_csv(project_root_path=root),
    )
    parser.add_argument(
        "--frame-audit-csv",
        type=Path,
        default=root / "output" / "ep01_data_processing" / "frame_audit.csv",
        help="Loaded as an existence check; alignment CSV already carries the needed metadata.",
    )
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "ep05_overlay_alignment")
    parser.add_argument("--outlier-threshold", type=float, default=3.0)
    parser.add_argument("--top", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.alignment_csv.exists():
        raise FileNotFoundError(args.alignment_csv)
    if not args.frame_audit_csv.exists():
        raise FileNotFoundError(args.frame_audit_csv)

    alignment = pd.read_csv(args.alignment_csv)
    fit = fit_filename_affine(alignment, robust=True, outlier_threshold=args.outlier_threshold)
    diagnosis = filename_affine_diagnostics(alignment, fit)

    output_csv = args.output_dir / "affine_outlier_diagnosis.csv"
    preferred_cols = [
        "file",
        "acquisition_order",
        "X",
        "Y",
        "R",
        "refined_align_dx_px",
        "refined_align_dy_px",
        "pred_dx_px",
        "pred_dy_px",
        "res_dx_px",
        "res_dy_px",
        "res_norm_px",
        "robust_pred_dx_px",
        "robust_pred_dy_px",
        "robust_res_norm_px",
        "refined_holdout_chamfer_px",
        "gradient_corr_refined",
        "refined_shift_norm_px",
        "outlier_score",
        "used_for_affine_fit",
        "residual_gate_outlier",
        "recommended_exclude_from_fit",
        "recommended_exclude_from_affine_application",
        "excluded_from_robust_fit",
        "suspicious_all_frames",
    ]
    diagnosis[[col for col in preferred_cols if col in diagnosis]].to_csv(output_csv, index=False)

    stats = {
        "res_norm_px": quantile_stats(diagnosis["res_norm_px"]),
        "robust_res_norm_px": quantile_stats(diagnosis["robust_res_norm_px"]),
        "refined_holdout_chamfer_px": quantile_stats(diagnosis["refined_holdout_chamfer_px"]),
        "gradient_corr_refined": quantile_stats(diagnosis["gradient_corr_refined"]),
        "refined_shift_norm_px": quantile_stats(diagnosis["refined_shift_norm_px"]),
        "outlier_score": quantile_stats(diagnosis["outlier_score"]),
    }

    top_cols = [
        "file",
        "acquisition_order",
        "X",
        "Y",
        "R",
        "res_norm_px",
        "refined_holdout_chamfer_px",
        "gradient_corr_refined",
        "refined_shift_norm_px",
        "outlier_score",
        "recommended_exclude_from_fit",
        "recommended_exclude_from_affine_application",
    ]
    top = diagnosis.sort_values("outlier_score", ascending=False)[top_cols].head(args.top)

    fit_rows = diagnosis[diagnosis["used_for_affine_fit"]].copy()
    baseline_fit_stats = quantile_stats(fit_rows["res_norm_px"])
    robust_clean_stats = quantile_stats(fit_rows.loc[~fit_rows["excluded_from_robust_fit"], "robust_res_norm_px"])

    print("Filename affine outlier diagnosis")
    print(f"alignment_csv: {args.alignment_csv}")
    print(f"output_csv: {output_csv}")
    print(
        "baseline fit: "
        f"n={fit.fit_count}, median_res={fit.median_residual_px:.6f}px, "
        f"threshold={fit.outlier_threshold_px:.6f}px"
    )
    print(f"robust refit: clean_n={fit.clean_count}, excluded_n={len(fit.excluded_files)}")
    print(f"excluded_files: {list(fit.excluded_files)}")
    application_excluded = diagnosis.loc[
        diagnosis["recommended_exclude_from_affine_application"],
        "file",
    ].astype(str).tolist()
    print(f"recommended_exclude_from_affine_application: {application_excluded}")
    print("\nCoefficient change")
    print(f"dx baseline {format_beta(fit.baseline_beta_dx)}")
    print(f"dx robust   {format_beta(fit.beta_dx)}")
    print(f"dx delta    {format_beta(fit.beta_dx - fit.baseline_beta_dx)}")
    print(f"dy baseline {format_beta(fit.baseline_beta_dy)}")
    print(f"dy robust   {format_beta(fit.beta_dy)}")
    print(f"dy delta    {format_beta(fit.beta_dy - fit.baseline_beta_dy)}")

    print("\nOverall statistics")
    for name, item in stats.items():
        print(
            f"{name}: count={item['count']}, median={item['median']:.6f}, "
            f"mean={item['mean']:.6f}, p90={item['p90']:.6f}, p95={item['p95']:.6f}, "
            f"p99={item['p99']:.6f}, max={item['max']:.6f}"
        )
    print("\nR=0 fit residual change")
    print(f"baseline all-fit rows: {baseline_fit_stats}")
    print(f"robust clean rows: {robust_clean_stats}")

    print(f"\nTop {len(top)} outlier candidates")
    print(top.to_string(index=False, float_format=lambda x: f"{x:.6f}"))


if __name__ == "__main__":
    main()
