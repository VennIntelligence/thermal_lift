"""EP09 notebook cache — validate algo outputs and write manifest."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from thermal_core.notebook_cache import (
    cache_is_complete,
    project_root,
    require_artifacts,
    write_manifest,
)

EP09_CACHE_VERSION = 1
REBUILD_COMMAND = "uv run python scripts/build_ep09_cache.py"

EP09_JSON_ARTIFACTS = (
    "calibration_summary.json",
    "sigma_forward.json",
    "sigma_esf.json",
    "sigma_joint.json",
)

EP09_CSV_ARTIFACTS = (
    "route_sigma_summary.csv",
    "forward_residual_fine_sweep.csv",
    "esf_sigma_distribution.csv",
    "joint_sigma_sweep.csv",
)

EP09_FIGURE_ARTIFACTS = (
    "forward_residual_curve.png",
    "esf_sigma_histogram.png",
    "joint_sigma_curve.png",
)

EP09_ARTIFACTS = (
    *EP09_JSON_ARTIFACTS,
    *EP09_CSV_ARTIFACTS,
    *EP09_FIGURE_ARTIFACTS,
    "cache_manifest.json",
)


@dataclass(frozen=True)
class Ep09Cache:
    output_dir: Path
    report_dir: Path
    config_path: Path
    summary: dict
    forward: dict
    esf: dict
    joint: dict
    route_table: pd.DataFrame
    manifest: dict

    def figure_path(self, name: str) -> Path:
        return self.output_dir / name

    def read_json(self, name: str) -> dict:
        path = self.output_dir / name
        if not path.exists():
            return {}
        return json.loads(path.read_text(encoding="utf-8"))

    def read_csv(self, name: str) -> pd.DataFrame:
        path = self.output_dir / name
        if not path.exists():
            return pd.DataFrame()
        return pd.read_csv(path)


def build_ep09_cache(
    *,
    project_root_path: Path | None = None,
    output_dir: Path | None = None,
    force: bool = False,
) -> Ep09Cache:
    """Validate EP09 algo outputs and write cache_manifest.json."""
    root = project_root(project_root_path)
    output_dir = (output_dir or root / "output" / "ep09_psf_calibration").resolve()
    report_dir = root / "reports" / "ep09_psf_calibration"
    config_path = root / "configs" / "psf_calibration.json"

    if not force and cache_is_complete(output_dir, EP09_ARTIFACTS):
        return load_ep09_cache(project_root_path=root, output_dir=output_dir)

    require_artifacts(output_dir, EP09_ARTIFACTS[:-1], rebuild_command=_algo_rebuild_hint())

    manifest = write_manifest(
        output_dir,
        version=EP09_CACHE_VERSION,
        artifacts=list(EP09_ARTIFACTS),
        rebuild_command=REBUILD_COMMAND,
        extra={
            "algo_pipeline": _algo_rebuild_hint(),
            "validated": True,
        },
    )
    return load_ep09_cache(project_root_path=root, output_dir=output_dir, manifest=manifest)


def _algo_rebuild_hint() -> str:
    return (
        "uv run python algos/ep09_psf_calibration/scripts/run_forward_residual.py && "
        "uv run python algos/ep09_psf_calibration/scripts/run_esf_fitting.py && "
        "uv run python algos/ep09_psf_calibration/scripts/run_joint_estimation.py && "
        "uv run python algos/ep09_psf_calibration/scripts/summarize_calibration.py"
    )


def load_ep09_cache(
    *,
    project_root_path: Path | None = None,
    output_dir: Path | None = None,
    manifest: dict | None = None,
    require_complete: bool = False,
) -> Ep09Cache:
    root = project_root(project_root_path)
    output_dir = (output_dir or root / "output" / "ep09_psf_calibration").resolve()
    if require_complete:
        require_artifacts(output_dir, EP09_ARTIFACTS[:-1], rebuild_command=_algo_rebuild_hint())

    manifest_path = output_dir / "cache_manifest.json"
    if manifest is None:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    summary = json.loads((output_dir / "calibration_summary.json").read_text(encoding="utf-8"))
    forward = json.loads((output_dir / "sigma_forward.json").read_text(encoding="utf-8"))
    esf = json.loads((output_dir / "sigma_esf.json").read_text(encoding="utf-8"))
    joint = json.loads((output_dir / "sigma_joint.json").read_text(encoding="utf-8"))
    route_table = pd.read_csv(output_dir / "route_sigma_summary.csv")

    return Ep09Cache(
        output_dir=output_dir,
        report_dir=root / "reports" / "ep09_psf_calibration",
        config_path=root / "configs" / "psf_calibration.json",
        summary=summary,
        forward=forward,
        esf=esf,
        joint=joint,
        route_table=route_table,
        manifest=manifest,
    )
