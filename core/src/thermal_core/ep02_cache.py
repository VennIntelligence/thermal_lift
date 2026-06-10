"""EP02 cache builder and loader — notebook reads artifacts; script rebuilds them."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from thermal_core.ep02 import (
    load_frame_audit,
    load_stage_config,
    plot_alignment_comparison,
    plot_raster_acquisition_path,
    plot_small_step_diagnostics,
    plot_stage_prior_coverage,
)
from thermal_core.notebook_cache import cache_is_complete, project_root, write_manifest
from thermal_core.plotting import setup_academic_style

EP02_CACHE_VERSION = 1
EP02_DIRNAME = "ep02_displacement_calibration"
REBUILD_COMMAND = "uv run python scripts/build_ep02_cache.py"

EP02_TABLE_ARTIFACTS = (
    "time_adjacent_registration_pairs.csv",
    "time_adjacent_pairs_r0.csv",
    "y_coordinate_pairs.csv",
    "coordinate_pair_time_gap_audit.csv",
    "time_adjacent_method_measurements.csv",
    "time_adjacent_method_summary.csv",
    "time_adjacent_x_step_fit.csv",
    "y_coordinate_method_measurements.csv",
    "y_coordinate_method_summary.csv",
    "y_coordinate_monotonic_summary.csv",
    "ep02_data_driven_alignment_comparison.csv",
)

EP02_FIGURE_ARTIFACTS = (
    "ep02_raster_acquisition_path.png",
    "ep02_stage_prior_coverage.png",
    "ep02_small_step_smoke_tests.png",
    "ep02_data_driven_alignment_comparison.png",
)

EP02_OPTIONAL_ARTIFACTS = (
    "avi_theta_estimates.csv",
    "avi_theta_summary.csv",
    "avi_theta_bracket_plot.png",
    "avi_theta_forest_plot.png",
    "avi_theta_result.json",
)

EP02_ARTIFACTS = (*EP02_TABLE_ARTIFACTS, *EP02_FIGURE_ARTIFACTS, "cache_manifest.json")


@dataclass(frozen=True)
class Ep02Cache:
    """Loaded EP02 artifacts for notebook display."""

    output_dir: Path
    frame_audit: pd.DataFrame
    stage_config: dict
    manifest: dict
    theta_deg: float
    pixel_size_um: float

    def figure_path(self, name: str) -> Path:
        return self.output_dir / Path(name).name


def _recompute_tables(project_root: Path) -> None:
    script = project_root / "scripts" / "recompute_ep02_displacement_tables.py"
    if not script.exists():
        raise FileNotFoundError(f"Missing recompute script: {script}")
    ep01_audit = project_root / "output" / "ep01_data_processing" / "frame_audit.csv"
    if not ep01_audit.exists():
        raise FileNotFoundError(
            f"Missing {ep01_audit}. Run: uv run python scripts/build_ep01_cache.py"
        )
    subprocess.run(
        [sys.executable, str(script)],
        cwd=project_root,
        check=True,
    )


def _build_avi_theta(project_root: Path, output_dir: Path) -> list[str]:
    """Run AVI theta estimation when AVI data exists; return built optional artifact names."""
    avi_dir = project_root / "data" / "data_raw" / "infrared_avi"
    if not avi_dir.exists():
        return []
    avi_files = list(avi_dir.glob("*.avi"))
    if not avi_files:
        return []

    script = project_root / "scripts" / "avi_theta_estimation.py"
    if not script.exists():
        return []

    try:
        subprocess.run(
            [sys.executable, str(script)],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError:
        return []

    built = [name for name in EP02_OPTIONAL_ARTIFACTS if (output_dir / name).exists()]
    return built


def build_ep02_cache(
    *,
    project_root_arg: Path | None = None,
    output_dir: Path | None = None,
    force: bool = False,
) -> Ep02Cache:
    """Recompute displacement tables and write EP02 CSV/PNG artifacts."""
    root = project_root(project_root_arg)
    output_dir = (output_dir or root / "output" / EP02_DIRNAME).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not force and cache_is_complete(output_dir, EP02_ARTIFACTS):
        return load_ep02_cache(output_dir=output_dir, project_root_arg=root)

    setup_academic_style()
    _recompute_tables(root)

    frame_audit = load_frame_audit(root)
    stage_config = load_stage_config(root)
    theta_deg = float(stage_config["theta_deg"])
    pixel_size_um = float(stage_config["pixel_size_um"])

    plot_raster_acquisition_path(
        frame_audit,
        output_dir / "ep02_raster_acquisition_path.png",
    )
    plot_stage_prior_coverage(
        frame_audit,
        theta_deg=theta_deg,
        pixel_size_um=pixel_size_um,
        output_path=output_dir / "ep02_stage_prior_coverage.png",
    )
    plot_small_step_diagnostics(
        output_dir,
        output_dir / "ep02_small_step_smoke_tests.png",
    )
    plot_alignment_comparison(
        root,
        output_dir / "ep02_data_driven_alignment_comparison.png",
    )

    optional_built = _build_avi_theta(root, output_dir)
    all_artifacts = list(EP02_ARTIFACTS) + [n for n in optional_built if n not in EP02_ARTIFACTS]

    manifest = write_manifest(
        output_dir,
        version=EP02_CACHE_VERSION,
        artifacts=all_artifacts,
        rebuild_command=REBUILD_COMMAND,
        extra={
            "theta_deg": theta_deg,
            "pixel_size_um": pixel_size_um,
            "n_frames": int(len(frame_audit)),
            "optional_artifacts": optional_built,
        },
    )

    return Ep02Cache(
        output_dir=output_dir,
        frame_audit=frame_audit,
        stage_config=stage_config,
        manifest=manifest,
        theta_deg=theta_deg,
        pixel_size_um=pixel_size_um,
    )


def load_ep02_cache(
    *,
    output_dir: Path | None = None,
    project_root_arg: Path | None = None,
) -> Ep02Cache:
    """Load EP02 CSV artifacts without recomputing displacement measurements."""
    root = project_root(project_root_arg)
    output_dir = (output_dir or root / "output" / EP02_DIRNAME).resolve()

    missing = [name for name in EP02_ARTIFACTS if not (output_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            "EP02 cache incomplete. Missing: "
            + ", ".join(missing)
            + f"\nRun: {REBUILD_COMMAND}"
        )

    manifest_path = output_dir / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    stage_config = load_stage_config(root)
    frame_audit = load_frame_audit(root)

    return Ep02Cache(
        output_dir=output_dir,
        frame_audit=frame_audit,
        stage_config=stage_config,
        manifest=manifest,
        theta_deg=float(stage_config["theta_deg"]),
        pixel_size_um=float(stage_config["pixel_size_um"]),
    )


def require_ep02_cache(**kwargs) -> Ep02Cache:
    """Load cache or raise with rebuild instructions."""
    return load_ep02_cache(**kwargs)
