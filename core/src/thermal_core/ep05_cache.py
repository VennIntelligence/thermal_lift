"""EP05 cache validator and loader — notebook reads artifacts; script writes manifests."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from thermal_core.ep05 import (
    load_alignment_tuning_outputs,
    load_capacity_outputs,
    load_contour_alignment_outputs,
    load_displacement_outputs,
    load_overlay_alignment_outputs,
)
from thermal_core.notebook_cache import (
    cache_is_complete,
    project_root,
    require_artifacts,
    write_manifest,
)

EP05_CACHE_VERSION = 1
REBUILD_COMMAND = "uv run python scripts/build_ep05_cache.py"

EP05_DISPLACEMENT_ARTIFACTS = (
    "displacement_reassessment_summary.json",
    "displacement_measurements.csv",
    "displacement_summary_by_class.csv",
    "main_session_cumulative_trajectory.csv",
    "main_session_cumulative_trajectory.png",
    "visible_shift_by_pair_class.png",
    "endpoint_displacement_vectors.png",
)

EP05_CAPACITY_ARTIFACTS = (
    "alignment_sr_capacity_summary.json",
    "alignment_method_summary.csv",
    "alignment_method_holdout_scores.csv",
    "phase_bin_summary_2x.csv",
    "phase_bin_counts_2x.csv",
    "alignment_method_comparison.png",
    "phase_bin_coverage_2x.png",
    "alignment_overlay_evidence.png",
    "alignment_overlay_density_metrics.csv",
)

EP05_CONTOUR_ARTIFACTS = (
    "contour_alignment_results.csv",
    "contour_alignment_summary.json",
)

EP05_OVERLAY_ARTIFACTS = (
    "overlay_alignment_summary.csv",
    "all_main_4x4_txt_bmp_overlay.png",
    "all_main_4x4_edge_line_overlay.png",
)

EP05_TUNING_STUDY_ARTIFACTS = (
    "tuning_summary.csv",
    "candidate_comparison_summary.csv",
    "candidate_phase_coverage.csv",
    "tuning_heatmap_heldout_chamfer.png",
    "candidate_alignment_comparison.png",
)

EP05_DIR_SPECS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("displacement", "ep05_sr_reassessment", EP05_DISPLACEMENT_ARTIFACTS),
    ("capacity", "ep05_alignment_sr_capacity", EP05_CAPACITY_ARTIFACTS),
    ("contour", "ep05_contour_alignment", EP05_CONTOUR_ARTIFACTS),
    ("overlay", "ep05_overlay_alignment", EP05_OVERLAY_ARTIFACTS),
)


@dataclass(frozen=True)
class Ep05Cache:
    """Loaded EP05 artifacts for notebook display."""

    project_root: Path
    displacement_dir: Path
    capacity_dir: Path
    contour_dir: Path
    overlay_dir: Path
    tuning_study_dir: Path
    tuning_dir_candidates: tuple[Path, ...]
    displacement_outputs: dict
    capacity_outputs: dict
    contour_outputs: dict
    overlay_outputs: dict
    tuning_outputs: dict
    manifests: dict[str, dict]

    @property
    def summary_json(self) -> dict:
        return self.capacity_outputs["summary_json"]

    def figure_path(self, directory: Path, name: str) -> Path:
        return directory / Path(name).name


def _output_dirs(root: Path) -> dict[str, Path]:
    return {
        "displacement": (root / "output" / "ep05_sr_reassessment").resolve(),
        "capacity": (root / "output" / "ep05_alignment_sr_capacity").resolve(),
        "contour": (root / "output" / "ep05_contour_alignment").resolve(),
        "overlay": (root / "output" / "ep05_overlay_alignment").resolve(),
        "tuning_study": (root / "output" / "ep05_alignment_tuning_study").resolve(),
    }


def _validate_and_write_manifests(
    root: Path,
    *,
    force: bool,
) -> dict[str, dict]:
    dirs = _output_dirs(root)
    manifests: dict[str, dict] = {}

    for key, rel_name, artifacts in EP05_DIR_SPECS:
        output_dir = dirs[key]
        manifest_path = output_dir / "cache_manifest.json"
        if not force and manifest_path.exists() and cache_is_complete(output_dir, artifacts):
            manifests[key] = json.loads(manifest_path.read_text(encoding="utf-8"))
            continue
        require_artifacts(
            output_dir,
            artifacts,
            rebuild_command=(
                "Run EP05 scripts first, then: uv run python scripts/build_ep05_cache.py"
            ),
        )
        manifests[key] = write_manifest(
            output_dir,
            version=EP05_CACHE_VERSION,
            artifacts=artifacts,
            rebuild_command=REBUILD_COMMAND,
            extra={"episode": key, "output_subdir": rel_name},
        )

    tuning_dir = dirs["tuning_study"]
    tuning_manifest_path = tuning_dir / "cache_manifest.json"
    if tuning_dir.exists() and cache_is_complete(tuning_dir, EP05_TUNING_STUDY_ARTIFACTS):
        if force or not tuning_manifest_path.exists():
            manifests["tuning_study"] = write_manifest(
                tuning_dir,
                version=EP05_CACHE_VERSION,
                artifacts=EP05_TUNING_STUDY_ARTIFACTS,
                rebuild_command="uv run python scripts/run_ep05_alignment_tuning_study.py --mode quick --limit-frames 96 --n-jobs 8",
                extra={"episode": "tuning_study", "optional": True},
            )
        else:
            manifests["tuning_study"] = json.loads(tuning_manifest_path.read_text(encoding="utf-8"))
    else:
        manifests["tuning_study"] = {}

    return manifests


def build_ep05_cache(
    *,
    project_root_path: Path | None = None,
    force: bool = False,
) -> Ep05Cache:
    """Validate EP05 output directories and write cache manifests."""
    root = project_root(project_root_path)
    manifests = _validate_and_write_manifests(root, force=force)
    return load_ep05_cache(project_root_path=root, manifests=manifests)


def load_ep05_cache(
    *,
    project_root_path: Path | None = None,
    manifests: dict[str, dict] | None = None,
) -> Ep05Cache:
    """Load EP05 CSV/JSON/PNG artifacts without re-running EP05 scripts."""
    root = project_root(project_root_path)
    dirs = _output_dirs(root)
    tuning_dir_candidates = (
        root / "output" / "ep05_alignment_tuning",
        root / "output" / "ep05_alignment_tuning_study",
    )

    if manifests is None:
        for key, _, artifacts in EP05_DIR_SPECS:
            require_artifacts(dirs[key], artifacts, rebuild_command=REBUILD_COMMAND)
        loaded_manifests: dict[str, dict] = {}
        for key, _, _ in EP05_DIR_SPECS:
            manifest_path = dirs[key] / "cache_manifest.json"
            if manifest_path.exists():
                loaded_manifests[key] = json.loads(manifest_path.read_text(encoding="utf-8"))
        tuning_manifest_path = dirs["tuning_study"] / "cache_manifest.json"
        if tuning_manifest_path.exists():
            loaded_manifests["tuning_study"] = json.loads(tuning_manifest_path.read_text(encoding="utf-8"))
        manifests = loaded_manifests

    return Ep05Cache(
        project_root=root,
        displacement_dir=dirs["displacement"],
        capacity_dir=dirs["capacity"],
        contour_dir=dirs["contour"],
        overlay_dir=dirs["overlay"],
        tuning_study_dir=dirs["tuning_study"],
        tuning_dir_candidates=tuning_dir_candidates,
        displacement_outputs=load_displacement_outputs(dirs["displacement"]),
        capacity_outputs=load_capacity_outputs(dirs["capacity"]),
        contour_outputs=load_contour_alignment_outputs(dirs["contour"]),
        overlay_outputs=load_overlay_alignment_outputs(dirs["overlay"]),
        tuning_outputs=load_alignment_tuning_outputs(list(tuning_dir_candidates)),
        manifests=manifests,
    )


def require_ep05_cache(**kwargs) -> Ep05Cache:
    """Load cache or raise with rebuild instructions."""
    return load_ep05_cache(**kwargs)
