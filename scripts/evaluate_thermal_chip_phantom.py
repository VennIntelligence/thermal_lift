#!/usr/bin/env python3
"""Write a lightweight synthetic ThermalChipPhantom evaluation summary."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TCFORGE_SRC = PROJECT_ROOT / "tcforge" / "src"
if TCFORGE_SRC.exists() and str(TCFORGE_SRC) not in sys.path:
    sys.path.insert(0, str(TCFORGE_SRC))

DEFAULT_DATASET_ROOT = PROJECT_ROOT / "data" / "synthetic" / "thermal_chip_phantom"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "thermal_chip_phantom"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _maybe_tcforge_evaluate(scene_dir: Path) -> dict[str, Any]:
    try:
        from tcforge import evaluate as tc_eval  # type: ignore
    except Exception:
        return {}
    for name in ("evaluate_scene", "summarize_scene"):
        fn = getattr(tc_eval, name, None)
        if callable(fn):
            result = fn(scene_dir)
            return dict(result) if result is not None else {}
    finite_summary = getattr(tc_eval, "finite_summary", None)
    if callable(finite_summary):
        hr = np.load(scene_dir / "hr_temperature_2x.npy")
        raw = np.load(scene_dir / "lr_burst_raw.npy", mmap_mode="r")
        hr_summary = finite_summary(hr)
        raw_summary = finite_summary(raw)
        return {
            "tcforge_hr_finite_fraction": hr_summary["finite_fraction"],
            "tcforge_raw_finite_fraction": raw_summary["finite_fraction"],
        }
    return {}


def _sample_frame_indices(n_frames: int, limit: int | None) -> np.ndarray:
    n = int(n_frames)
    if n <= 0:
        return np.asarray([], dtype=int)
    if limit is None or int(limit) <= 0 or int(limit) >= n:
        return np.arange(n, dtype=int)
    return np.unique(np.linspace(0, n - 1, int(limit), dtype=int))


def _reference_highpass(frames: np.ndarray, *, sigma_bg: float, mode: str = "nearest") -> np.ndarray:
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 2:
        bg = ndimage.gaussian_filter(arr, sigma=float(sigma_bg), mode=mode)
    elif arr.ndim == 3:
        bg = ndimage.gaussian_filter(arr, sigma=(0.0, float(sigma_bg), float(sigma_bg)), mode=mode)
    else:
        raise ValueError("frames must be 2D or 3D")
    return (arr - bg).astype(np.float32, copy=False)


def _load_metadata(scene_dir: Path) -> dict[str, Any]:
    with (scene_dir / "metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _highpass_reference_summary(scene_dir: Path, raw: np.ndarray, hp: np.ndarray, *, limit: int | None) -> dict[str, Any]:
    metadata = _load_metadata(scene_dir)
    physics = metadata.get("physics", {})
    indices = _sample_frame_indices(raw.shape[0], limit)
    reference = _reference_highpass(
        raw[indices],
        sigma_bg=float(physics.get("highpass_sigma_lr_px", 5.0)),
        mode=str(physics.get("highpass_mode", "nearest")),
    )
    diff = np.asarray(hp[indices], dtype=np.float32) - reference
    return {
        "highpass_reference_check_frames": int(len(indices)),
        "highpass_reference_max_abs_diff_c": float(np.max(np.abs(diff))) if diff.size else 0.0,
        "highpass_reference_allclose": bool(np.allclose(hp[indices], reference, rtol=1e-5, atol=1e-5)),
    }


def _local_scene_summary(row: dict[str, str], *, highpass_check_frames: int | None = 16) -> dict[str, Any]:
    scene_dir = Path(row["scene_dir"])
    hr = np.load(scene_dir / "hr_temperature_2x.npy")
    mask = np.load(scene_dir / "hr_mask_2x.npy")
    edge = np.load(scene_dir / "hr_edge_map_2x.npy")
    raw = np.load(scene_dir / "lr_burst_raw.npy", mmap_mode="r")
    hp = np.load(scene_dir / "lr_burst_highpass.npy", mmap_mode="r")
    shifts = np.load(scene_dir / "shifts.npy")
    shift_norms = np.linalg.norm(shifts, axis=1)
    summary: dict[str, Any] = {
        "scene_id": row.get("scene_id", scene_dir.name),
        "split": row.get("split", ""),
        "difficulty": row.get("difficulty", ""),
        "scale": int(float(row.get("scale", 2))),
        "seed": int(float(row.get("seed", 0))),
        "forward_mode": row.get("forward_mode", ""),
        "drift_model": row.get("drift_model", ""),
        "n_frames": int(raw.shape[0]),
        "lr_rows": int(raw.shape[1]),
        "lr_cols": int(raw.shape[2]),
        "hr_rows": int(hr.shape[0]),
        "hr_cols": int(hr.shape[1]),
        "hr_temperature_min_c": float(np.min(hr)),
        "hr_temperature_max_c": float(np.max(hr)),
        "mask_coverage": float(np.mean(mask > 0)),
        "edge_density": float(np.mean(edge > 0)),
        "lr_raw_mean_c": float(np.mean(raw)),
        "lr_raw_std_c": float(np.std(raw)),
        "lr_highpass_abs_p95_c": float(np.percentile(np.abs(hp), 95)),
        "shift_norm_mean_px": float(np.mean(shift_norms)),
        "shift_norm_max_px": float(np.max(shift_norms)),
    }
    summary.update(_highpass_reference_summary(scene_dir, raw, hp, limit=highpass_check_frames))
    summary.update(_maybe_tcforge_evaluate(scene_dir))
    return summary


def evaluate_dataset(dataset_root: Path, output_dir: Path, *, highpass_check_frames: int | None = 16) -> list[dict[str, Any]]:
    manifest_path = dataset_root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    rows = _read_csv(manifest_path)
    summaries = [_local_scene_summary(row, highpass_check_frames=highpass_check_frames) for row in rows]
    _write_csv(output_dir / "evaluation_summary.csv", summaries)
    _write_json(output_dir / "evaluation_summary.json", {"scenes": summaries})
    return summaries


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--highpass-check-frames",
        type=int,
        default=16,
        help="number of evenly spaced frames to recompute with an independent highpass reference; <=0 checks all frames",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    hp_limit = None if int(args.highpass_check_frames) <= 0 else int(args.highpass_check_frames)
    summaries = evaluate_dataset(args.dataset_root, args.output_dir, highpass_check_frames=hp_limit)
    print(f"wrote {args.output_dir / 'evaluation_summary.csv'} ({len(summaries)} scenes)")
    print(f"wrote {args.output_dir / 'evaluation_summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
