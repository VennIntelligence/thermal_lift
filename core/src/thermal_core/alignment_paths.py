"""Resolve repo-tracked alignment artifact paths from configs/alignment/paths.json."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

PATHS_CONFIG_REL = Path("configs") / "alignment" / "paths.json"
CONTOUR_ALIGNMENT_RESULTS_KEY = "contour_alignment_results_csv"


def project_root(start: Path | None = None) -> Path:
    root = (start or Path.cwd()).resolve()
    while not (root / "AGENTS.md").exists() and root != root.parent:
        root = root.parent
    if not (root / "AGENTS.md").exists():
        raise FileNotFoundError("Could not locate project root (AGENTS.md missing).")
    return root


@lru_cache(maxsize=8)
def _load_paths_json(root_str: str) -> dict:
    path = Path(root_str) / PATHS_CONFIG_REL
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{PATHS_CONFIG_REL} must contain a JSON object")
    return payload


def load_alignment_paths(*, project_root_path: Path | None = None) -> dict:
    root = project_root(project_root_path)
    return _load_paths_json(str(root.resolve()))


def resolve_repo_path(relative: str | Path, *, project_root_path: Path | None = None) -> Path:
    root = project_root(project_root_path)
    path = Path(relative).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def default_contour_alignment_csv(*, project_root_path: Path | None = None) -> Path:
    """Return the default EP05 contour-refined shift CSV path.

    ``TCFORGE_REAL_SHIFT_CSV`` still overrides this for ad-hoc experiments.
    """

    env_path = os.environ.get("TCFORGE_REAL_SHIFT_CSV")
    if env_path:
        return Path(env_path).expanduser().resolve()
    cfg = load_alignment_paths(project_root_path=project_root_path)
    if CONTOUR_ALIGNMENT_RESULTS_KEY not in cfg:
        raise KeyError(f"{PATHS_CONFIG_REL} missing key: {CONTOUR_ALIGNMENT_RESULTS_KEY}")
    return resolve_repo_path(str(cfg[CONTOUR_ALIGNMENT_RESULTS_KEY]), project_root_path=project_root_path)
