"""EP06 cache validation, 4x ROI figure build, and loader for notebook fragments."""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from thermal_core.notebook_cache import (
    cache_is_complete,
    load_manifest,
    missing_artifacts,
    project_root,
    require_artifacts,
    write_manifest,
)

EP06_CACHE_VERSION = 1

EP06_REQUIRED_OUTPUTS = (
    "saa_uniform_highpass.npy",
    "saa_weighted_highpass.npy",
    "saa_uniform_raw.npy",
    "saa_weighted_raw.npy",
    "bicubic_reference.npy",
    "lr_reference.npy",
    "lr_raw_reference.npy",
    "bicubic_raw_reference.npy",
    "saa_synthetic_validation.json",
    "ibp_highpass.npy",
    "ibp_raw.npy",
    "ibp_convergence.csv",
    "ibp_synthetic_validation.json",
    "map_tv_highpass.npy",
    "map_tv_raw.npy",
    "map_tv_lambda_selection.csv",
    "map_tv_convergence.csv",
    "map_tv_synthetic_validation.json",
    "evaluation_summary.csv",
    "comparison_fullview.png",
    "comparison_roi_1.png",
    "comparison_roi_2.png",
    "comparison_roi_3.png",
    "comparison_control_track.png",
    "comparison_center_raw_temperature.png",
    "gradient_magnitude_comparison.png",
    "split_half_consistency.png",
    "artifact_audit.png",
)

EP06_4X_FIGURE = "comparison_center_raw_temperature.png"
EP06_4X_DIRNAME = "ep06_sr_poc_4x"
EP06_CACHE_ARTIFACTS = (*EP06_REQUIRED_OUTPUTS, "cache_manifest.json")

EP06_ALGO_BUILD_HINT = (
    "uv run python algos/ep06_sr_poc/scripts/run_saa.py --psf-sigma 0.5 && "
    "uv run python algos/ep06_sr_poc/scripts/run_ibp.py --max-iter 8 --psf-sigma 0.5 && "
    "uv run python algos/ep06_sr_poc/scripts/run_map_tv.py --max-iter 8 --step-size 0.25 "
    "--psf-sigma 0.5 --lambda-grid 0.0003,0.001,0.003,0.01 --no-fista && "
    "uv run python algos/ep06_sr_poc/scripts/run_evaluation.py --center-roi-sizes 160,112,80"
)

REBUILD_COMMAND = "uv run python scripts/build_ep06_cache.py"


@dataclass(frozen=True)
class Ep06Cache:
    """Loaded EP06 artifact paths for notebook display."""

    output_dir: Path
    output_dir_4x: Path
    ablation_output_dir: Path
    sweep_summary_dir: Path
    manifest: dict
    missing_required: tuple[str, ...]

    def figure_path(self, name: str, *, subdir: str = "main") -> Path:
        if subdir == "4x":
            return self.output_dir_4x / name
        if subdir == "sweep":
            return self.sweep_summary_dir / name
        if subdir == "ablation":
            return self.ablation_output_dir / name
        return self.output_dir / name


def validate_ep06_outputs(output_dir: Path) -> list[str]:
    return missing_artifacts(output_dir, EP06_REQUIRED_OUTPUTS)


def build_ep06_4x_roi(
    *,
    project_root_path: Path | None = None,
    output_dir_4x: Path | None = None,
    force: bool = False,
) -> Path:
    """Run EP06 algo scripts at scale=4 and produce center raw-temperature PNG."""
    root = project_root(project_root_path)
    output_dir_4x = (output_dir_4x or root / "output" / EP06_4X_DIRNAME).resolve()
    target_png = output_dir_4x / EP06_4X_FIGURE
    if not force and target_png.exists():
        return target_png

    output_dir_4x.mkdir(parents=True, exist_ok=True)
    algo_dir = root / "algos" / "ep06_sr_poc" / "scripts"
    out = str(output_dir_4x)
    steps = [
        [
            sys.executable,
            str(algo_dir / "run_saa.py"),
            "--scale",
            "4",
            "--splat-sigma",
            "1.5",
            "--output-dir",
            out,
        ],
        [
            sys.executable,
            str(algo_dir / "run_ibp.py"),
            "--scale",
            "4",
            "--max-iter",
            "4",
            "--splat-sigma",
            "1.5",
            "--output-dir",
            out,
        ],
        [
            sys.executable,
            str(algo_dir / "run_map_tv.py"),
            "--scale",
            "4",
            "--max-iter",
            "4",
            "--lambda-grid",
            "0.001",
            "--splat-sigma",
            "1.5",
            "--output-dir",
            out,
        ],
        [sys.executable, str(algo_dir / "run_evaluation.py"), "--scale", "4", "--output-dir", out],
    ]
    for cmd in steps:
        result = subprocess.run(cmd, cwd=root, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(
                f"EP06 4x build failed: {' '.join(cmd)}\n"
                f"{result.stderr or result.stdout}"
            )
    if not target_png.exists():
        raise FileNotFoundError(f"4x build finished but missing: {target_png}")
    return target_png


def build_ep06_cache(
    *,
    project_root_path: Path | None = None,
    output_dir: Path | None = None,
    force: bool = False,
    skip_4x: bool = False,
) -> Ep06Cache:
    """Validate EP06 SR outputs, optionally build 4x ROI figure, write manifest."""
    root = project_root(project_root_path)
    output_dir = (output_dir or root / "output" / "ep06_sr_poc").resolve()
    output_dir_4x = root / "output" / EP06_4X_DIRNAME
    manifest_path = output_dir / "cache_manifest.json"

    missing = validate_ep06_outputs(output_dir)
    if missing:
        raise FileNotFoundError(
            f"EP06 SR outputs incomplete in {output_dir}. Missing: {', '.join(missing)}\n"
            f"Run EP06 algo scripts first, e.g.:\n  {EP06_ALGO_BUILD_HINT}"
        )

    if not skip_4x:
        build_ep06_4x_roi(
            project_root_path=root,
            output_dir_4x=output_dir_4x,
            force=force,
        )

    four_x_built = (output_dir_4x / EP06_4X_FIGURE).exists()
    if not force and manifest_path.exists() and cache_is_complete(output_dir, EP06_CACHE_ARTIFACTS):
        manifest = load_manifest(output_dir)
        if manifest.get("four_x_built") == four_x_built:
            return load_ep06_cache(output_dir=output_dir, project_root_path=root)

    manifest = write_manifest(
        output_dir,
        version=EP06_CACHE_VERSION,
        artifacts=EP06_CACHE_ARTIFACTS,
        rebuild_command=REBUILD_COMMAND,
        extra={
            "required_outputs_validated": True,
            "four_x_figure": EP06_4X_FIGURE,
            "four_x_output_dir": str(output_dir_4x.relative_to(root)),
            "four_x_built": four_x_built,
            "algo_build_hint": EP06_ALGO_BUILD_HINT,
        },
    )
    return Ep06Cache(
        output_dir=output_dir,
        output_dir_4x=output_dir_4x,
        ablation_output_dir=root / "output" / "ep06_alignment_ablation",
        sweep_summary_dir=root / "output" / "ep06_sr_poc_data_driven_align_sweep" / "summary",
        manifest=manifest,
        missing_required=(),
    )


def load_ep06_cache(
    *,
    output_dir: Path | None = None,
    project_root_path: Path | None = None,
    require_complete: bool = True,
) -> Ep06Cache:
    """Load EP06 cache metadata; optionally require all validated artifacts."""
    root = project_root(project_root_path)
    output_dir = (output_dir or root / "output" / "ep06_sr_poc").resolve()
    missing = validate_ep06_outputs(output_dir)
    if require_complete and missing:
        raise FileNotFoundError(
            f"EP06 SR outputs incomplete. Missing: {', '.join(missing)}\n"
            f"Run EP06 algo scripts first, e.g.:\n  {EP06_ALGO_BUILD_HINT}"
        )
    if require_complete:
        require_artifacts(output_dir, ("cache_manifest.json",), rebuild_command=REBUILD_COMMAND)

    manifest = load_manifest(output_dir) if (output_dir / "cache_manifest.json").exists() else {}
    return Ep06Cache(
        output_dir=output_dir,
        output_dir_4x=root / "output" / EP06_4X_DIRNAME,
        ablation_output_dir=root / "output" / "ep06_alignment_ablation",
        sweep_summary_dir=root / "output" / "ep06_sr_poc_data_driven_align_sweep" / "summary",
        manifest=manifest,
        missing_required=tuple(missing),
    )
