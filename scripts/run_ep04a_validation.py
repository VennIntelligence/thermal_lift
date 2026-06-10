"""Run EP04-A data-driven multi-frame ESF global validation."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import pandas as pd

from thermal_core.ep04 import (
    build_ep06_gate_recommendations,
    create_ep04_anchor_gate_figures,
    create_ep04_figures,
    ep04_data_contract_summary,
    prepare_ep04_segment_inputs,
    run_all_segments,
    run_all_inner_segments,
    save_ep06_gate_outputs,
    save_validation_outputs,
    select_clean_sr_reference_row,
)
from thermal_core.io import load_frame
from thermal_core.plotting import setup_academic_style


def project_root() -> Path:
    root = Path.cwd()
    while not (root / "AGENTS.md").exists() and root != root.parent:
        root = root.parent
    return root


def default_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, 16))


def display_path(path: Path, root: Path) -> Path:
    """Prefer project-relative paths, but allow smoke-test outputs outside the repo."""
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--segments-csv",
        type=Path,
        default=root / "output" / "ep03_theoretical_limits" / "contour_segments.csv",
    )
    parser.add_argument(
        "--inner-segments-csv",
        type=Path,
        default=root / "output" / "ep03_theoretical_limits" / "inner_contour_segments.csv",
    )
    parser.add_argument(
        "--frame-audit-csv",
        type=Path,
        default=root / "output" / "ep01_data_processing" / "frame_audit.csv",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=root / "data" / "data_raw" / "infrared_avi",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "output" / "ep04_global_validation",
    )
    parser.add_argument("--n-jobs", type=int, default=default_workers())
    parser.add_argument("--min-snr", type=float, default=8.0)
    parser.add_argument("--min-delta-t", type=float, default=0.5)
    parser.add_argument("--max-split-half", type=float, default=0.06)
    parser.add_argument("--limit-segments", type=int, default=None)
    parser.add_argument("--limit-scanlines", type=int, default=None)
    parser.add_argument("--outer-only", action="store_true")
    parser.add_argument("--force-segment-inputs", action="store_true")
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    root = project_root()
    args = parse_args()
    setup_academic_style()

    with open(root / "configs" / "stage_calibration.json", encoding="utf-8") as f:
        stage_config = json.load(f)
    with open(root / "configs" / "noise_floor.json", encoding="utf-8") as f:
        noise_config = json.load(f)

    theta_deg = float(stage_config["theta_deg"])
    pixel_size_um = float(stage_config["pixel_size_um"])
    noise_floor_c = float(noise_config["noise_floor_celsius"])

    print("EP04-A data-driven global validation")
    segment_inputs = prepare_ep04_segment_inputs(
        args.frame_audit_csv,
        args.data_dir,
        args.output_dir,
        outer_segments_csv=args.segments_csv,
        inner_segments_csv=args.inner_segments_csv,
        theta_deg=theta_deg,
        noise_floor_c=noise_floor_c,
        force=bool(args.force_segment_inputs),
    )
    outer_segments_csv = segment_inputs["outer_segments_csv"]
    inner_segments_csv = segment_inputs["inner_segments_csv"]

    print(f"outer segments: {outer_segments_csv}")
    print(f"inner segments: {inner_segments_csv}")
    print(f"audit:    {args.frame_audit_csv}")
    print(f"data:     {args.data_dir}")
    print(f"output:   {args.output_dir}")
    print(f"n_jobs={args.n_jobs}, theta={theta_deg:.1f} deg, pixel={pixel_size_um:.1f} um, noise={noise_floor_c:.4f} C")
    audit = pd.read_csv(args.frame_audit_csv)
    data_contract = ep04_data_contract_summary(audit)
    print(
        "data contract: "
        f"raw session=2 {data_contract['raw_main_session_frame_count']} frames, "
        f"clean SR input {data_contract['clean_sr_input_frame_count']} frames, "
        f"EP04 complete X scanlines {data_contract['ep04_scanline_count']} "
        f"({data_contract['ep04_unique_frame_count']} unique frames)"
    )

    started = time.perf_counter()
    results = run_all_segments(
        outer_segments_csv,
        args.frame_audit_csv,
        args.data_dir,
        args.output_dir,
        min_snr=args.min_snr,
        min_delta_t=args.min_delta_t,
        max_split_half=args.max_split_half,
        n_jobs=args.n_jobs,
        theta_deg=theta_deg,
        pixel_size_um=pixel_size_um,
        noise_floor_c=noise_floor_c,
        limit_segments=args.limit_segments,
        limit_scanlines=args.limit_scanlines,
        show_progress=True,
        save_outputs=False,
    )
    runtime_seconds = time.perf_counter() - started
    segment_summary, summary = save_validation_outputs(
        results,
        args.output_dir,
        extra_summary={
            "runtime_seconds": float(runtime_seconds),
            "n_jobs": int(args.n_jobs),
            "theta_deg": theta_deg,
            "pixel_size_um": pixel_size_um,
            "noise_floor_c": noise_floor_c,
            "segments_csv": str(outer_segments_csv),
            "frame_audit_csv": str(args.frame_audit_csv),
            "data_dir": str(args.data_dir),
            "output_dir": str(args.output_dir),
            "limit_segments": args.limit_segments,
            "limit_scanlines": args.limit_scanlines,
            **data_contract,
        },
    )

    inner_results = pd.DataFrame()
    inner_segment_summary = pd.DataFrame()
    inner_summary = {}
    if not args.outer_only:
        inner_output_dir = args.output_dir / "inner"
        inner_started = time.perf_counter()
        inner_results = run_all_inner_segments(
            inner_segments_csv,
            args.frame_audit_csv,
            args.data_dir,
            inner_output_dir,
            min_snr=args.min_snr,
            min_delta_t=args.min_delta_t,
            max_split_half=args.max_split_half,
            n_jobs=args.n_jobs,
            theta_deg=theta_deg,
            pixel_size_um=pixel_size_um,
            noise_floor_c=noise_floor_c,
            limit_segments=args.limit_segments,
            limit_scanlines=args.limit_scanlines,
            show_progress=True,
            save_outputs=False,
        )
        inner_runtime_seconds = time.perf_counter() - inner_started
        inner_segment_summary, inner_summary = save_validation_outputs(
            inner_results,
            inner_output_dir,
            output_prefix="inner_",
            extra_summary={
                "runtime_seconds": float(inner_runtime_seconds),
                "n_jobs": int(args.n_jobs),
                "theta_deg": theta_deg,
                "pixel_size_um": pixel_size_um,
                "noise_floor_c": noise_floor_c,
                "segments_csv": str(inner_segments_csv),
                "frame_audit_csv": str(args.frame_audit_csv),
                "data_dir": str(args.data_dir),
                "output_dir": str(inner_output_dir),
                "limit_segments": args.limit_segments,
                "limit_scanlines": args.limit_scanlines,
                **data_contract,
            },
        )

    if not args.skip_figures:
        ref_row = select_clean_sr_reference_row(audit)
        reference_frame = load_frame(args.data_dir / str(ref_row["file"]))
        create_ep04_figures(results, segment_summary, reference_frame, args.output_dir)
        if not args.outer_only:
            recommendations = build_ep06_gate_recommendations(segment_summary, inner_segment_summary)
            save_ep06_gate_outputs(recommendations, args.output_dir)
            create_ep04_anchor_gate_figures(
                reference_frame,
                results,
                segment_summary,
                inner_results,
                inner_segment_summary,
                recommendations,
                args.output_dir,
            )

    print("\nGlobal summary")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    if inner_summary:
        print("\nInner summary")
        for key, value in inner_summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    print(f"  validation runtime: {runtime_seconds:.1f} s")
    print("\nSaved outputs")
    for name in [
        "segment_validation_results.csv",
        "segment_summary.csv",
        "global_summary.json",
        "split_half_distribution.png",
        "crb_ratio_scatter.png",
        "pass_fail_contour_map.png",
        "phase_coverage_vs_precision.png",
        "failure_taxonomy.png",
        "cross_scanline_consistency.png",
        "global_segment_quality_distribution.png",
        "anchor_coverage_map.png",
        "anchor_scanline_support.png",
        "inner_failure_reasons.png",
        "ep06_gate_recommendations.png",
        "ep06_gate_recommendations.csv",
        "ep06_gate_recommendation_summary.csv",
    ]:
        path = args.output_dir / name
        if path.exists():
            print(f"  {display_path(path, root)}")
    if not args.outer_only:
        for name in [
            "inner_segment_validation_results.csv",
            "inner_segment_summary.csv",
            "inner_global_summary.json",
        ]:
            path = args.output_dir / "inner" / name
            if path.exists():
                print(f"  {display_path(path, root)}")


if __name__ == "__main__":
    main()
