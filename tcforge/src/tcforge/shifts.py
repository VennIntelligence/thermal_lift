"""Shift profile loading/generation for TCForge synthetic bursts."""

from __future__ import annotations

import csv
import functools
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


def random_constellation(
    n_frames: int,
    *,
    scale: int = 2,
    phase_steps: int = 4,
    coverage: str = "good",
    jitter_std_px: float = 0.0,
    seed: int | None = None,
) -> np.ndarray:
    """Generate a random per-scene sub-pixel shift constellation in LR pixels.

    Base sub-pixel phases live on a ``phase_steps × phase_steps`` grid with
    cells at ``(k/phase_steps, m/phase_steps)``. Frames are assigned to cells
    according to ``coverage``:

    * ``"good"``   — N frames spread as evenly as possible over ALL cells.
    * ``"medium"`` — randomly drop ~1/3 of the cells; frames use the rest.
    * ``"poor"``   — cluster frames into ~half the cells (deficient phase
      coverage).

    Gaussian jitter ``N(0, jitter_std_px)`` is added to every shift, plus a
    small random integer-pixel offset per frame so frames are not all near the
    origin. Returns ``(n_frames, 2)`` float32 ``[dx, dy]`` shifts.
    """

    n = int(n_frames)
    scale = int(scale)
    phase_steps = int(phase_steps)
    if n <= 0:
        raise ValueError("n_frames must be > 0")
    if scale <= 0:
        raise ValueError("scale must be > 0")
    if phase_steps <= 0:
        raise ValueError("phase_steps must be > 0")
    jitter = float(jitter_std_px)
    if jitter < 0:
        raise ValueError("jitter_std_px must be >= 0")
    coverage = str(coverage)
    if coverage not in {"good", "medium", "poor"}:
        raise ValueError("coverage must be 'good', 'medium', or 'poor'")

    rng = np.random.default_rng(seed)

    # All sub-pixel phase cells on the phase_steps × phase_steps grid.
    cells = np.array(
        [(x / phase_steps, y / phase_steps) for y in range(phase_steps) for x in range(phase_steps)],
        dtype=np.float32,
    )
    n_cells = len(cells)

    if coverage == "good":
        active_idx = np.arange(n_cells)
    elif coverage == "medium":
        # Drop ~1/3 of the cells (keep at least one).
        n_keep = max(1, n_cells - int(round(n_cells / 3.0)))
        active_idx = np.sort(rng.choice(n_cells, size=n_keep, replace=False))
    else:  # poor
        # Cluster into ~half the cells (keep at least one).
        n_keep = max(1, n_cells // 2)
        active_idx = np.sort(rng.choice(n_cells, size=n_keep, replace=False))

    # Round-robin assignment of the N frames across the active cells, in a
    # shuffled frame order so cells fill as evenly as possible.
    order = rng.permutation(n)
    assigned = np.empty(n, dtype=np.int64)
    assigned[order] = active_idx[np.arange(n) % len(active_idx)]
    shifts = cells[assigned].copy()

    # Small random integer-pixel offset per frame (realistic, keeps frames off
    # the origin) — does not change sub-pixel phase.
    int_offset = rng.integers(-2, 3, size=(n, 2)).astype(np.float32)
    shifts += int_offset

    if jitter > 0:
        shifts = shifts + rng.normal(0.0, jitter, size=shifts.shape).astype(np.float32)
    return _validate_shifts(shifts.astype(np.float32, copy=False), n_frames=n)


def build_scene_shifts(
    seed: int,
    n_frames: int,
    constellation_cfg: dict[str, Any],
    *,
    scale: int = 2,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build per-scene shifts from a constellation config (deterministic from seed).

    With probability ``constellation_cfg["include_real_like_fraction"]`` a
    "real_like" constellation is returned (``n_frames`` subsampled with
    replacement from the real contour-refined profile plus small jitter).
    Otherwise ``random_constellation`` is called with ``phase_steps`` drawn from
    ``phase_steps_choices``, ``coverage`` drawn from ``coverage_quality_weights``,
    and ``jitter_std_px`` drawn uniformly from the configured range.

    Returns ``(shifts, metadata)`` where metadata describes the choice.
    """

    n = int(n_frames)
    if n <= 0:
        raise ValueError("n_frames must be > 0")
    cfg = dict(constellation_cfg or {})
    rng = np.random.default_rng(seed)

    real_like_fraction = float(cfg.get("include_real_like_fraction", 0.0))
    use_real_like = bool(rng.random() < real_like_fraction)

    jitter_low, jitter_high = _range_pair(cfg.get("jitter_std_px", [0.0, 0.0]), "jitter_std_px")
    jitter = float(rng.uniform(jitter_low, jitter_high))

    if use_real_like:
        try:
            base = load_real_default_contour_refined(n_frames=None)
        except (FileNotFoundError, ValueError) as exc:
            # Fall back to a random constellation if the real profile is absent.
            phase_steps_choices = list(cfg.get("phase_steps_choices", [4]))
            phase_steps = int(rng.choice(phase_steps_choices))
            coverage = _weighted_coverage(rng, cfg.get("coverage_quality_weights", {"good": 1.0}))
            shifts = random_constellation(
                n, scale=scale, phase_steps=phase_steps, coverage=coverage,
                jitter_std_px=jitter, seed=int(rng.integers(1, np.iinfo(np.int32).max)),
            )
            metadata = {
                "constellation_mode": "random_phase",
                "convention": SHIFT_CONVENTION,
                "n_frames": int(shifts.shape[0]),
                "columns": ["dx_px", "dy_px"],
                "phase_steps": phase_steps,
                "coverage": coverage,
                "jitter_std_px": jitter,
                "real_like_requested": True,
                "real_like_fallback_reason": str(exc),
            }
            return shifts, metadata
        idx = rng.integers(0, base.shape[0], size=n)
        shifts = base[idx].astype(np.float32, copy=True)
        if jitter > 0:
            shifts = shifts + rng.normal(0.0, jitter, size=shifts.shape).astype(np.float32)
        shifts = _validate_shifts(shifts.astype(np.float32, copy=False), n_frames=n)
        metadata = {
            "constellation_mode": "real_like",
            "convention": SHIFT_CONVENTION,
            "n_frames": int(shifts.shape[0]),
            "columns": ["dx_px", "dy_px"],
            "jitter_std_px": jitter,
            "real_like_source": "real_default_contour_refined",
        }
        return shifts, metadata

    phase_steps_choices = list(cfg.get("phase_steps_choices", [4]))
    phase_steps = int(rng.choice(phase_steps_choices))
    coverage = _weighted_coverage(rng, cfg.get("coverage_quality_weights", {"good": 1.0}))
    shifts = random_constellation(
        n,
        scale=scale,
        phase_steps=phase_steps,
        coverage=coverage,
        jitter_std_px=jitter,
        seed=int(rng.integers(1, np.iinfo(np.int32).max)),
    )
    metadata = {
        "constellation_mode": "random_phase",
        "convention": SHIFT_CONVENTION,
        "n_frames": int(shifts.shape[0]),
        "columns": ["dx_px", "dy_px"],
        "phase_steps": phase_steps,
        "coverage": coverage,
        "jitter_std_px": jitter,
        "real_like_requested": False,
    }
    return shifts, metadata


def _range_pair(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        return float(value), float(value)
    if isinstance(value, (list, tuple)) and len(value) == 2:
        low, high = float(value[0]), float(value[1])
        if high < low:
            raise ValueError(f"{name} range must satisfy high >= low")
        return low, high
    raise ValueError(f"{name} must be a scalar or [low, high] range")


def _weighted_coverage(rng: np.random.Generator, weights: dict[str, float]) -> str:
    names = list(weights)
    values = np.asarray([float(weights[name]) for name in names], dtype=np.float64)
    if len(names) == 0 or np.any(values < 0) or float(values.sum()) <= 0:
        raise ValueError("coverage_quality_weights must be non-negative with positive sum")
    return str(rng.choice(names, p=values / values.sum()))


@functools.lru_cache(maxsize=4)
def _cached_load_contour_csv(csv_path_str: str, strict_success: bool) -> np.ndarray:
    """Parse and cache contour alignment CSV. Keyed on resolved path string."""
    csv_path = Path(csv_path_str)
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
    return shifts


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
    shifts = _cached_load_contour_csv(str(csv_path.resolve()), strict_success)
    return _validate_shifts(shifts.copy(), n_frames=n_frames)


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
