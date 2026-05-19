"""Shared EP09 path, IO, and numerical helpers."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np


def find_project_root(start: Path | None = None) -> Path:
    """Return the repository root by walking upward to ``AGENTS.md``."""

    path = (start or Path(__file__)).resolve()
    if path.is_file():
        path = path.parent
    while path != path.parent:
        if (path / "AGENTS.md").exists():
            return path
        path = path.parent
    raise FileNotFoundError("Could not find project root containing AGENTS.md")


PROJECT_ROOT = find_project_root()
EP09_ROOT = PROJECT_ROOT / "algos" / "ep09_psf_calibration"
OUTPUT_DIR = PROJECT_ROOT / "output" / "ep09_psf_calibration"
REPORT_DIR = PROJECT_ROOT / "reports" / "ep09_psf_calibration"
CONFIG_PATH = PROJECT_ROOT / "configs" / "psf_calibration.json"


def bootstrap_project_paths() -> None:
    """Make EP09, EP06, and core modules importable from standalone scripts."""

    paths = [
        EP09_ROOT / "src",
        PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src",
        PROJECT_ROOT / "core" / "src",
    ]
    for path in paths:
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)


bootstrap_project_paths()


def default_workers(max_workers: int = 8) -> int:
    """Conservative worker count for CPU-only EP09 jobs."""

    return max(1, min(int(max_workers), (os.cpu_count() or 2) // 2))


def ensure_dir(path: str | Path) -> Path:
    """Create and return a directory path."""

    out = Path(path)
    out.mkdir(parents=True, exist_ok=True)
    return out


def relative(path: str | Path, root: Path = PROJECT_ROOT) -> str:
    """Return a repo-relative path string when possible."""

    path = Path(path)
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def to_jsonable(value: Any) -> Any:
    """Convert NumPy and Path values into JSON-serializable objects."""

    if isinstance(value, Path):
        return relative(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    """Write pretty JSON with NumPy scalar support."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(to_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out


def read_json(path: str | Path) -> dict[str, Any]:
    """Read a JSON object."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def deterministic_split(
    n_items: int,
    *,
    val_stride: int = 5,
    max_train: int | None = None,
    max_val: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically split acquisition-ordered frames into train/val indices."""

    if n_items <= 1:
        raise ValueError("Need at least two items for a train/val split")
    stride = max(2, int(val_stride))
    all_idx = np.arange(n_items, dtype=int)
    val = all_idx[::stride]
    train = np.setdiff1d(all_idx, val, assume_unique=True)
    if max_train is not None and len(train) > int(max_train):
        take = np.linspace(0, len(train) - 1, int(max_train), dtype=int)
        train = train[take]
    if max_val is not None and len(val) > int(max_val):
        take = np.linspace(0, len(val) - 1, int(max_val), dtype=int)
        val = val[take]
    if len(train) == 0 or len(val) == 0:
        raise ValueError("Split produced an empty train or validation set")
    return train.astype(int), val.astype(int)


def parabolic_minimum(sigmas: np.ndarray, values: np.ndarray, *, window: int = 2) -> tuple[float, bool]:
    """Estimate a curve minimum by fitting a local quadratic around the best grid point."""

    x = np.asarray(sigmas, dtype=float)
    y = np.asarray(values, dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if finite.sum() < 3:
        return float(x[finite][np.nanargmin(y[finite])]), False
    x = x[finite]
    y = y[finite]
    idx = int(np.argmin(y))
    lo = max(0, idx - int(window))
    hi = min(len(x), idx + int(window) + 1)
    if hi - lo < 3:
        lo = max(0, min(idx - 1, len(x) - 3))
        hi = min(len(x), lo + 3)
    coef = np.polyfit(x[lo:hi], y[lo:hi], deg=2)
    a, b, _ = map(float, coef)
    if a <= 0:
        return float(x[idx]), False
    optimum = -b / (2.0 * a)
    if optimum < float(x[0]) or optimum > float(x[-1]):
        return float(x[idx]), False
    return float(optimum), True


def bootstrap_curve_minima(
    sigmas: np.ndarray,
    per_frame_scores: np.ndarray,
    *,
    n_bootstrap: int,
    seed: int,
) -> np.ndarray:
    """Bootstrap frame-resampled minima from a ``(n_sigma, n_frame)`` score matrix."""

    scores = np.asarray(per_frame_scores, dtype=float)
    if scores.ndim != 2:
        raise ValueError("per_frame_scores must have shape (n_sigma, n_frame)")
    rng = np.random.default_rng(seed)
    n_frames = scores.shape[1]
    if n_frames < 2:
        return np.asarray([], dtype=float)
    minima = np.empty(int(n_bootstrap), dtype=float)
    for idx in range(int(n_bootstrap)):
        sample = rng.integers(0, n_frames, size=n_frames)
        curve = np.nanmean(scores[:, sample], axis=1)
        minima[idx], _ = parabolic_minimum(np.asarray(sigmas, dtype=float), curve)
    return minima[np.isfinite(minima)]


def crop_slices(shape: tuple[int, int], margin: int) -> tuple[slice, slice]:
    """Return 2D crop slices used for residual scoring."""

    margin = max(0, int(margin))
    if margin == 0:
        return slice(None), slice(None)
    h, w = map(int, shape)
    if 2 * margin >= h or 2 * margin >= w:
        raise ValueError(f"crop margin {margin} is too large for shape {shape}")
    return slice(margin, h - margin), slice(margin, w - margin)
