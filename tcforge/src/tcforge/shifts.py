"""Shift profile loading/generation for TCForge synthetic bursts."""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Literal

import numpy as np

SHIFT_CONVENTION = "LR-to-reference alignment shift"
PATHS_CONFIG_REL = Path("configs") / "alignment" / "paths.json"
CONTOUR_ALIGNMENT_RESULTS_KEY = "contour_alignment_results_csv"
ShiftProfileName = Literal["real_default_contour_refined", "ideal_phase_grid"]


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _load_alignment_paths_config(*, project_root: Path | None = None) -> dict[str, Any]:
    root = project_root or _project_root()
    with (root / PATHS_CONFIG_REL).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"{PATHS_CONFIG_REL} must contain a JSON object")
    return payload


def _resolve_repo_path(relative: str | Path, *, project_root: Path | None = None) -> Path:
    root = project_root or _project_root()
    path = Path(relative).expanduser()
    if path.is_absolute():
        return path.resolve()
    return (root / path).resolve()


def default_contour_alignment_csv(*, project_root: Path | None = None) -> Path:
    env_path = os.environ.get("TCFORGE_REAL_SHIFT_CSV")
    if env_path:
        return Path(env_path).expanduser().resolve()
    cfg = _load_alignment_paths_config(project_root=project_root)
    if CONTOUR_ALIGNMENT_RESULTS_KEY not in cfg:
        raise KeyError(f"{PATHS_CONFIG_REL} missing key: {CONTOUR_ALIGNMENT_RESULTS_KEY}")
    return _resolve_repo_path(str(cfg[CONTOUR_ALIGNMENT_RESULTS_KEY]), project_root=project_root)


def _default_real_shift_csv() -> Path:
    return default_contour_alignment_csv()


def _validate_shifts(shifts: np.ndarray, *, n_frames: int | None = None) -> np.ndarray:
    arr = np.asarray(shifts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("shifts must have shape (N, 2) with columns [dx, dy]")
    if n_frames is not None and arr.shape[0] != int(n_frames):
        raise ValueError(f"expected {n_frames} shifts, got {arr.shape[0]}")
    if not np.isfinite(arr).all():
        raise ValueError("shifts contain NaN or Inf")
    return arr


def _repeat_or_trim(shifts: np.ndarray, n_frames: int) -> np.ndarray:
    arr = _validate_shifts(shifts)
    n = int(n_frames)
    if n <= 0:
        raise ValueError("n_frames must be > 0")
    repeats = int(np.ceil(n / len(arr)))
    return np.tile(arr, (repeats, 1))[:n].astype(np.float32, copy=False)


def ideal_phase_grid(
    n_frames: int = 255,
    *,
    scale: int = 2,
    phase_steps: int = 4,
    jitter_std_px: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a repeated sub-pixel phase grid in LR pixels."""

    n = int(n_frames)
    scale = int(scale)
    phase_steps = int(phase_steps)
    if n <= 0:
        raise ValueError("n_frames must be > 0")
    if scale <= 0:
        raise ValueError("scale must be > 0")
    if phase_steps <= 0:
        raise ValueError("phase_steps must be > 0")
    phases = np.array(
        [(x / phase_steps, y / phase_steps) for y in range(phase_steps) for x in range(phase_steps)],
        dtype=np.float32,
    )
    shifts = phases[np.arange(n) % len(phases)].copy()
    jitter = float(jitter_std_px)
    if jitter < 0:
        raise ValueError("jitter_std_px must be >= 0")
    if jitter > 0:
        shifts += np.random.default_rng(seed).normal(0.0, jitter, size=shifts.shape).astype(np.float32)
    return _validate_shifts(shifts)


def load_real_default_contour_refined(
    path: str | Path | None = None,
    *,
    n_frames: int | None = None,
    strict_success: bool = True,
) -> np.ndarray:
    """Load EP05 refined contour shifts from a CSV file.

    ``n_frames`` is optional because the real clean input frame count can change
    after repeat/quality gating. Synthetic profiles that need a fixed burst
    length should call ``load_shift_profile()``, which repeats/trims explicitly.
    """

    csv_path = Path(path).expanduser() if path is not None else _default_real_shift_csv()
    if not csv_path.exists():
        raise FileNotFoundError(
            "real_default_contour_refined CSV not found: "
            f"{csv_path}. Pass path=..., set TCFORGE_REAL_SHIFT_CSV, or update {PATHS_CONFIG_REL}."
        )
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    if not rows:
        raise ValueError(f"shift CSV is empty: {csv_path}")
    required = {"refined_align_dx_px", "refined_align_dy_px"}
    missing = required - set(rows[0].keys())
    if missing:
        raise ValueError(f"shift CSV missing required columns: {sorted(missing)}")
    if "acquisition_order" in rows[0]:
        rows.sort(key=lambda row: float(row["acquisition_order"]))
    elif "frame_index" in rows[0]:
        rows.sort(key=lambda row: int(float(row["frame_index"])))
    if strict_success and "success" in rows[0]:
        bad = [row for row in rows if str(row.get("success", "")).strip().lower() not in {"true", "1", "yes"}]
        if bad:
            raise ValueError(f"{len(bad)} rows in {csv_path} have success != True")
    shifts = np.array(
        [[float(row["refined_align_dx_px"]), float(row["refined_align_dy_px"])] for row in rows],
        dtype=np.float32,
    )
    return _validate_shifts(shifts, n_frames=n_frames)


def load_shift_profile(
    profile: ShiftProfileName | str,
    *,
    n_frames: int = 255,
    path: str | Path | None = None,
    scale: int = 2,
    phase_steps: int = 4,
    seed: int | None = None,
    jitter_std_px: float = 0.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load or generate a shift profile for a synthetic burst.

    The default 255 is the requested synthetic observation count used by P0
    smoke configs; it is not a claim about the current real clean SR input size.
    """

    if profile == "real_default_contour_refined":
        shifts = _repeat_or_trim(load_real_default_contour_refined(path, n_frames=None), n_frames)
    elif profile == "ideal_phase_grid":
        shifts = ideal_phase_grid(
            n_frames=n_frames,
            scale=scale,
            phase_steps=phase_steps,
            jitter_std_px=jitter_std_px,
            seed=seed,
        )
    else:
        raise ValueError("profile must be 'real_default_contour_refined' or 'ideal_phase_grid'")
    metadata = {
        "profile": str(profile),
        "convention": SHIFT_CONVENTION,
        "n_frames": int(shifts.shape[0]),
        "columns": ["dx_px", "dy_px"],
    }
    if path is not None:
        metadata["path"] = str(path)
    if profile == "ideal_phase_grid":
        metadata.update(
            {
                "scale": int(scale),
                "phase_steps": int(phase_steps),
                "jitter_std_px": float(jitter_std_px),
                "seed": seed,
            }
        )
    return shifts, metadata


def save_shift_profile_json(shifts: np.ndarray, path: str | Path, metadata: dict[str, Any] | None = None) -> None:
    """Write a small JSON shift profile with explicit convention metadata."""

    arr = _validate_shifts(shifts)
    payload = {
        "convention": SHIFT_CONVENTION,
        "columns": ["dx_px", "dy_px"],
        "shifts": arr.astype(float).tolist(),
    }
    if metadata:
        payload.update(metadata)
        payload["convention"] = SHIFT_CONVENTION
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
