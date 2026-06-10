"""EP04 cache builder and loader — notebook reads artifacts; script rebuilds them."""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from thermal_core.ep03 import select_main_scan, select_reference_frame_row
from thermal_core.ep04 import (
    build_ep06_gate_recommendations,
    create_ep04_anchor_gate_figures,
    create_ep04_figures,
    prepare_ep04_segment_inputs,
    run_all_inner_segments,
    run_all_segments,
    save_ep06_gate_outputs,
    save_validation_outputs,
)
from thermal_core.io import load_frame
from thermal_core.notebook_cache import (
    cache_is_complete,
    project_root,
    require_artifacts,
    write_manifest,
)
from thermal_core.plotting import setup_academic_style

EP04_CACHE_VERSION = 1
REBUILD_COMMAND = "uv run python scripts/build_ep04_cache.py"

EP04_TABLE_ARTIFACTS = (
    "segment_validation_results.csv",
    "segment_summary.csv",
    "global_summary.json",
    "ep06_gate_recommendations.csv",
    "ep06_gate_recommendation_summary.csv",
)

EP04_INNER_TABLE_ARTIFACTS = (
    "inner/inner_segment_validation_results.csv",
    "inner/inner_segment_summary.csv",
    "inner/inner_global_summary.json",
)

EP04_FIGURE_ARTIFACTS = (
    "split_half_distribution.png",
    "crb_ratio_scatter.png",
    "phase_coverage_vs_precision.png",
    "failure_taxonomy.png",
    "cross_scanline_consistency.png",
    "segment_scanline_pass_heatmap.png",
    "normal_angle_coverage.png",
    "anchor_coverage_map.png",
    "anchor_scanline_support.png",
    "ep06_gate_recommendations.png",
    "inner_failure_reasons.png",
    "global_segment_quality_distribution.png",
)

EP04_ARTIFACTS = (
    *EP04_TABLE_ARTIFACTS,
    *EP04_INNER_TABLE_ARTIFACTS,
    *EP04_FIGURE_ARTIFACTS,
    "cache_manifest.json",
)


@dataclass(frozen=True)
class Ep04Cache:
    """Loaded EP04 artifacts for notebook display."""

    output_dir: Path
    inner_output_dir: Path
    outer_results: pd.DataFrame
    outer_segment_summary: pd.DataFrame
    outer_global_summary: dict
    inner_results: pd.DataFrame
    inner_segment_summary: pd.DataFrame
    inner_global_summary: dict
    ep06_recommendations: pd.DataFrame
    ep06_recommendation_summary: pd.DataFrame
    outer_segments_csv: Path
    inner_segments_csv: Path
    reference_file: str
    reference_order: int
    theta_deg: float
    pixel_size_um: float
    noise_floor_c: float
    manifest: dict

    def figure_path(self, name: str) -> Path:
        return self.output_dir / Path(name).name


def _default_workers() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, 16))


def _load_configs(root: Path) -> tuple[float, float, float]:
    with open(root / "configs" / "stage_calibration.json", encoding="utf-8") as f:
        stage_config = json.load(f)
    with open(root / "configs" / "noise_floor.json", encoding="utf-8") as f:
        noise_config = json.load(f)
    return (
        float(stage_config["theta_deg"]),
        float(stage_config["pixel_size_um"]),
        float(noise_config["noise_floor_celsius"]),
    )


def _run_validation(
    *,
    outer_segments_csv: Path,
    inner_segments_csv: Path,
    frame_audit_csv: Path,
    data_dir: Path,
    output_dir: Path,
    inner_output_dir: Path,
    theta_deg: float,
    pixel_size_um: float,
    noise_floor_c: float,
    n_jobs: int,
    force: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, dict]:
    outer_results_path = output_dir / "segment_validation_results.csv"
    outer_summary_path = output_dir / "segment_summary.csv"
    outer_global_path = output_dir / "global_summary.json"
    inner_results_path = inner_output_dir / "inner_segment_validation_results.csv"
    inner_summary_path = inner_output_dir / "inner_segment_summary.csv"
    inner_global_path = inner_output_dir / "inner_global_summary.json"

    need_outer = force or not (
        outer_results_path.exists() and outer_summary_path.exists() and outer_global_path.exists()
    )
    need_inner = force or not (
        inner_results_path.exists() and inner_summary_path.exists() and inner_global_path.exists()
    )

    if need_outer:
        started = time.perf_counter()
        outer_results = run_all_segments(
            outer_segments_csv,
            frame_audit_csv,
            data_dir,
            output_dir,
            n_jobs=n_jobs,
            theta_deg=theta_deg,
            pixel_size_um=pixel_size_um,
            noise_floor_c=noise_floor_c,
            show_progress=True,
            save_outputs=False,
        )
        outer_runtime_seconds = time.perf_counter() - started
        outer_segment_summary, outer_global_summary = save_validation_outputs(
            outer_results,
            output_dir,
            extra_summary={
                "runtime_seconds": float(outer_runtime_seconds),
                "n_jobs": int(n_jobs),
                "theta_deg": theta_deg,
                "pixel_size_um": pixel_size_um,
                "noise_floor_c": noise_floor_c,
                "segments_csv": str(outer_segments_csv),
                "frame_audit_csv": str(frame_audit_csv),
                "data_dir": str(data_dir),
                "output_dir": str(output_dir),
            },
        )
    else:
        outer_results = pd.read_csv(outer_results_path)
        outer_segment_summary = pd.read_csv(outer_summary_path)
        with open(outer_global_path, encoding="utf-8") as f:
            outer_global_summary = json.load(f)

    if need_inner:
        inner_started = time.perf_counter()
        inner_results = run_all_inner_segments(
            inner_segments_csv,
            frame_audit_csv,
            data_dir,
            inner_output_dir,
            n_jobs=n_jobs,
            theta_deg=theta_deg,
            pixel_size_um=pixel_size_um,
            noise_floor_c=noise_floor_c,
            show_progress=True,
            save_outputs=False,
        )
        inner_runtime_seconds = time.perf_counter() - inner_started
        inner_segment_summary, inner_global_summary = save_validation_outputs(
            inner_results,
            inner_output_dir,
            output_prefix="inner_",
            extra_summary={
                "runtime_seconds": float(inner_runtime_seconds),
                "n_jobs": int(n_jobs),
                "theta_deg": theta_deg,
                "pixel_size_um": pixel_size_um,
                "noise_floor_c": noise_floor_c,
                "segments_csv": str(inner_segments_csv),
                "frame_audit_csv": str(frame_audit_csv),
                "data_dir": str(data_dir),
                "output_dir": str(inner_output_dir),
            },
        )
    else:
        inner_results = pd.read_csv(inner_results_path)
        inner_segment_summary = pd.read_csv(inner_summary_path)
        with open(inner_global_path, encoding="utf-8") as f:
            inner_global_summary = json.load(f)

    return (
        outer_results,
        outer_segment_summary,
        outer_global_summary,
        inner_results,
        inner_segment_summary,
        inner_global_summary,
    )


def _build_figures(
    *,
    reference_frame,
    outer_results: pd.DataFrame,
    outer_segment_summary: pd.DataFrame,
    inner_results: pd.DataFrame,
    inner_segment_summary: pd.DataFrame,
    recommendations: pd.DataFrame,
    output_dir: Path,
) -> None:
    create_ep04_figures(outer_results, outer_segment_summary, reference_frame, output_dir)
    create_ep04_anchor_gate_figures(
        reference_frame,
        outer_results,
        outer_segment_summary,
        inner_results,
        inner_segment_summary,
        recommendations,
        output_dir,
    )


def build_ep04_cache(
    *,
    project_root_path: Path | None = None,
    data_dir: Path | None = None,
    ep01_output_dir: Path | None = None,
    ep03_output_dir: Path | None = None,
    output_dir: Path | None = None,
    n_jobs: int | None = None,
    force: bool = False,
    force_segment_inputs: bool = False,
) -> Ep04Cache:
    """Run EP04 validation and write CSV/PNG artifacts for the notebook."""
    root = project_root(project_root_path)
    data_dir = (data_dir or root / "data" / "data_raw" / "infrared_avi").resolve()
    ep01_output_dir = (ep01_output_dir or root / "output" / "ep01_data_processing").resolve()
    ep03_output_dir = (ep03_output_dir or root / "output" / "ep03_theoretical_limits").resolve()
    output_dir = (output_dir or root / "output" / "ep04_global_validation").resolve()
    inner_output_dir = output_dir / "inner"
    output_dir.mkdir(parents=True, exist_ok=True)
    inner_output_dir.mkdir(parents=True, exist_ok=True)

    if not force and cache_is_complete(output_dir, EP04_ARTIFACTS):
        return load_ep04_cache(output_dir=output_dir, project_root_path=root)

    theta_deg, pixel_size_um, noise_floor_c = _load_configs(root)
    n_jobs = n_jobs if n_jobs is not None else _default_workers()
    setup_academic_style()

    segment_inputs = prepare_ep04_segment_inputs(
        ep01_output_dir / "frame_audit.csv",
        data_dir,
        output_dir,
        outer_segments_csv=ep03_output_dir / "contour_segments.csv",
        inner_segments_csv=ep03_output_dir / "inner_contour_segments.csv",
        theta_deg=theta_deg,
        noise_floor_c=noise_floor_c,
        force=force_segment_inputs,
    )
    outer_segments_csv = segment_inputs["outer_segments_csv"]
    inner_segments_csv = segment_inputs["inner_segments_csv"]

    (
        outer_results,
        outer_segment_summary,
        outer_global_summary,
        inner_results,
        inner_segment_summary,
        inner_global_summary,
    ) = _run_validation(
        outer_segments_csv=outer_segments_csv,
        inner_segments_csv=inner_segments_csv,
        frame_audit_csv=ep01_output_dir / "frame_audit.csv",
        data_dir=data_dir,
        output_dir=output_dir,
        inner_output_dir=inner_output_dir,
        theta_deg=theta_deg,
        pixel_size_um=pixel_size_um,
        noise_floor_c=noise_floor_c,
        n_jobs=n_jobs,
        force=force,
    )

    recommendations = build_ep06_gate_recommendations(outer_segment_summary, inner_segment_summary)
    save_ep06_gate_outputs(recommendations, output_dir)

    audit = pd.read_csv(ep01_output_dir / "frame_audit.csv")
    main_df = select_main_scan(audit)
    ref_row = select_reference_frame_row(main_df)
    reference_frame = load_frame(data_dir / str(ref_row["file"]))

    _build_figures(
        reference_frame=reference_frame,
        outer_results=outer_results,
        outer_segment_summary=outer_segment_summary,
        inner_results=inner_results,
        inner_segment_summary=inner_segment_summary,
        recommendations=recommendations,
        output_dir=output_dir,
    )

    manifest = write_manifest(
        output_dir,
        version=EP04_CACHE_VERSION,
        artifacts=EP04_ARTIFACTS,
        rebuild_command=REBUILD_COMMAND,
        extra={
            "data_dir": str(data_dir),
            "frame_audit_csv": str(ep01_output_dir / "frame_audit.csv"),
            "outer_segments_csv": str(outer_segments_csv),
            "inner_segments_csv": str(inner_segments_csv),
            "n_outer_rows": int(len(outer_results)),
            "n_inner_rows": int(len(inner_results)),
            "reference_file": str(ref_row["file"]),
            "reference_order": int(ref_row["acquisition_order"]),
            "theta_deg": theta_deg,
            "pixel_size_um": pixel_size_um,
            "noise_floor_c": noise_floor_c,
        },
    )

    ep06_recommendation_summary = pd.read_csv(output_dir / "ep06_gate_recommendation_summary.csv")
    return Ep04Cache(
        output_dir=output_dir,
        inner_output_dir=inner_output_dir,
        outer_results=outer_results,
        outer_segment_summary=outer_segment_summary,
        outer_global_summary=outer_global_summary,
        inner_results=inner_results,
        inner_segment_summary=inner_segment_summary,
        inner_global_summary=inner_global_summary,
        ep06_recommendations=recommendations,
        ep06_recommendation_summary=ep06_recommendation_summary,
        outer_segments_csv=outer_segments_csv,
        inner_segments_csv=inner_segments_csv,
        reference_file=str(ref_row["file"]),
        reference_order=int(ref_row["acquisition_order"]),
        theta_deg=theta_deg,
        pixel_size_um=pixel_size_um,
        noise_floor_c=noise_floor_c,
        manifest=manifest,
    )


def load_ep04_cache(
    *,
    output_dir: Path | None = None,
    project_root_path: Path | None = None,
) -> Ep04Cache:
    """Load EP04 CSV/JSON artifacts without re-running validation."""
    root = project_root(project_root_path)
    output_dir = (output_dir or root / "output" / "ep04_global_validation").resolve()
    inner_output_dir = output_dir / "inner"
    require_artifacts(output_dir, EP04_ARTIFACTS, rebuild_command=REBUILD_COMMAND)

    with open(output_dir / "cache_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(output_dir / "global_summary.json", encoding="utf-8") as f:
        outer_global_summary = json.load(f)
    with open(inner_output_dir / "inner_global_summary.json", encoding="utf-8") as f:
        inner_global_summary = json.load(f)

    theta_deg, pixel_size_um, noise_floor_c = _load_configs(root)

    return Ep04Cache(
        output_dir=output_dir,
        inner_output_dir=inner_output_dir,
        outer_results=pd.read_csv(output_dir / "segment_validation_results.csv"),
        outer_segment_summary=pd.read_csv(output_dir / "segment_summary.csv"),
        outer_global_summary=outer_global_summary,
        inner_results=pd.read_csv(inner_output_dir / "inner_segment_validation_results.csv"),
        inner_segment_summary=pd.read_csv(inner_output_dir / "inner_segment_summary.csv"),
        inner_global_summary=inner_global_summary,
        ep06_recommendations=pd.read_csv(output_dir / "ep06_gate_recommendations.csv"),
        ep06_recommendation_summary=pd.read_csv(output_dir / "ep06_gate_recommendation_summary.csv"),
        outer_segments_csv=Path(manifest.get("outer_segments_csv", output_dir / "inputs" / "contour_segments.csv")),
        inner_segments_csv=Path(
            manifest.get("inner_segments_csv", output_dir / "inputs" / "inner_contour_segments.csv")
        ),
        reference_file=str(manifest.get("reference_file", "")),
        reference_order=int(manifest.get("reference_order", 0)),
        theta_deg=float(manifest.get("theta_deg", theta_deg)),
        pixel_size_um=float(manifest.get("pixel_size_um", pixel_size_um)),
        noise_floor_c=float(manifest.get("noise_floor_c", noise_floor_c)),
        manifest=manifest,
    )


def require_ep04_cache(**kwargs) -> Ep04Cache:
    """Load cache or raise with rebuild instructions."""
    return load_ep04_cache(**kwargs)
