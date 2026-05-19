"""Numeric and scene-level evaluation helpers for TCForge outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .physics import edge_map

REQUIRED_SCENE_FILES: tuple[str, ...] = (
    "hr_temperature_2x.npy",
    "hr_mask_2x.npy",
    "hr_edge_map_2x.npy",
    "lr_burst_raw.npy",
    "lr_burst_highpass.npy",
    "shifts.npy",
    "metadata.json",
)
DEFAULT_RECONSTRUCTION_NAMES: tuple[str, ...] = (
    "sr_temperature_2x.npy",
    "reconstruction_2x.npy",
    "reconstruction.npy",
)
DEFAULT_MASK_NAMES: tuple[str, ...] = ("sr_mask_2x.npy", "mask_prediction_2x.npy")
DEFAULT_EDGE_NAMES: tuple[str, ...] = ("sr_edge_map_2x.npy", "edge_prediction_2x.npy")


def finite_summary(array: np.ndarray) -> dict[str, float | int | bool]:
    arr = np.asarray(array)
    finite = np.isfinite(arr)
    if not finite.any():
        return {"shape_ndim": arr.ndim, "size": int(arr.size), "finite_all": False, "finite_fraction": 0.0}
    vals = arr[finite].astype(float)
    return {
        "shape_ndim": arr.ndim,
        "size": int(arr.size),
        "finite_all": bool(finite.all()),
        "finite_fraction": float(finite.mean()),
        "min": float(np.min(vals)),
        "max": float(np.max(vals)),
        "mean": float(np.mean(vals)),
        "std": float(np.std(vals)),
    }


def mae(prediction: np.ndarray, target: np.ndarray) -> float:
    pred, truth = _paired(prediction, target)
    return float(np.mean(np.abs(pred - truth)))


def rmse(prediction: np.ndarray, target: np.ndarray) -> float:
    pred, truth = _paired(prediction, target)
    return float(np.sqrt(np.mean((pred - truth) ** 2)))


def nrmse(prediction: np.ndarray, target: np.ndarray, *, data_range: float | None = None) -> float:
    pred, truth = _paired(prediction, target)
    denom = float(data_range) if data_range is not None else float(np.max(truth) - np.min(truth))
    if denom <= 0:
        raise ValueError("data_range must be > 0 for NRMSE")
    return float(np.sqrt(np.mean((pred - truth) ** 2)) / denom)


def psnr(prediction: np.ndarray, target: np.ndarray, *, data_range: float | None = None) -> float:
    pred, truth = _paired(prediction, target)
    err = float(np.mean((pred - truth) ** 2))
    if err == 0:
        return float("inf")
    peak = float(data_range) if data_range is not None else float(np.max(truth) - np.min(truth))
    if peak <= 0:
        raise ValueError("data_range must be > 0 for PSNR")
    return float(20.0 * np.log10(peak) - 10.0 * np.log10(err))


def binary_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(prediction).astype(bool)
    truth = np.asarray(target).astype(bool)
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch: {pred.shape} != {truth.shape}")
    union = np.logical_or(pred, truth).sum()
    if union == 0:
        return 1.0
    return float(np.logical_and(pred, truth).sum() / union)


def boundary_f1(prediction: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(prediction).astype(bool)
    truth = np.asarray(target).astype(bool)
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch: {pred.shape} != {truth.shape}")
    tp = float(np.logical_and(pred, truth).sum())
    fp = float(np.logical_and(pred, ~truth).sum())
    fn = float(np.logical_and(~pred, truth).sum())
    precision = tp / max(tp + fp, 1e-12)
    recall = tp / max(tp + fn, 1e-12)
    return float(2.0 * precision * recall / max(precision + recall, 1e-12))


def summarize_scene(scene_dir: str | Path) -> dict[str, Any]:
    """Summarize one generated scene without requiring an SR reconstruction."""

    root = _validate_scene_dir(scene_dir)
    metadata = _read_metadata(root)
    hr = np.load(root / "hr_temperature_2x.npy", mmap_mode="r")
    mask = np.load(root / "hr_mask_2x.npy", mmap_mode="r")
    edge = np.load(root / "hr_edge_map_2x.npy", mmap_mode="r")
    raw = np.load(root / "lr_burst_raw.npy", mmap_mode="r")
    hp = np.load(root / "lr_burst_highpass.npy", mmap_mode="r")
    shifts = np.load(root / "shifts.npy", mmap_mode="r")
    shift_norms = np.linalg.norm(np.asarray(shifts), axis=1)
    hr_summary = finite_summary(hr)
    raw_summary = finite_summary(raw)
    hp_summary = finite_summary(hp)

    return {
        "scene_id": metadata.get("scene_id", root.name),
        "scene_dir": str(root),
        "split": metadata.get("split", ""),
        "difficulty": metadata.get("difficulty", ""),
        "scale": int(metadata.get("scale", _infer_scale(hr.shape, raw.shape[1:]))),
        "n_frames": int(raw.shape[0]),
        "lr_rows": int(raw.shape[1]),
        "lr_cols": int(raw.shape[2]),
        "hr_rows": int(hr.shape[0]),
        "hr_cols": int(hr.shape[1]),
        "hr_temperature_min_c": float(hr_summary["min"]),
        "hr_temperature_max_c": float(hr_summary["max"]),
        "hr_temperature_mean_c": float(hr_summary["mean"]),
        "mask_coverage": float(np.mean(np.asarray(mask) > 0)),
        "edge_density": float(np.mean(np.asarray(edge) > 0)),
        "lr_raw_mean_c": float(raw_summary["mean"]),
        "lr_raw_std_c": float(raw_summary["std"]),
        "lr_highpass_abs_p95_c": float(np.percentile(np.abs(hp), 95)),
        "shift_norm_mean_px": float(np.mean(shift_norms)),
        "shift_norm_max_px": float(np.max(shift_norms)),
        "tcforge_hr_finite_fraction": hr_summary["finite_fraction"],
        "tcforge_raw_finite_fraction": raw_summary["finite_fraction"],
        "tcforge_highpass_finite_fraction": hp_summary["finite_fraction"],
    }


def evaluate_scene(
    scene_dir: str | Path,
    *,
    reconstruction_path: str | Path | None = None,
    mask_prediction_path: str | Path | None = None,
    edge_prediction_path: str | Path | None = None,
    threshold_c: float | None = None,
) -> dict[str, Any]:
    """Evaluate one scene and optional 2x SR outputs against TCForge ground truth.

    If explicit prediction paths are not supplied, common filenames inside
    ``scene_dir`` are auto-detected. The returned dictionary always contains
    scene summaries; SR metric keys are added only when matching prediction
    files are present.
    """

    root = _validate_scene_dir(scene_dir)
    summary = summarize_scene(root)
    hr = np.load(root / "hr_temperature_2x.npy")
    mask = np.load(root / "hr_mask_2x.npy")
    edge = np.load(root / "hr_edge_map_2x.npy")

    recon_path = _resolve_optional_path(root, reconstruction_path, DEFAULT_RECONSTRUCTION_NAMES)
    if recon_path is not None:
        pred = np.load(recon_path)
        data_range = float(np.max(hr) - np.min(hr))
        summary.update(
            {
                "sr_temperature_path": str(recon_path),
                "sr_temperature_mae_c": mae(pred, hr),
                "sr_temperature_rmse_c": rmse(pred, hr),
                "sr_temperature_nrmse": nrmse(pred, hr, data_range=data_range),
                "sr_temperature_psnr_db": psnr(pred, hr, data_range=data_range),
            }
        )
        if threshold_c is not None:
            pred_mask = np.asarray(pred) >= float(threshold_c)
            truth_mask = np.asarray(hr) >= float(threshold_c)
            summary["sr_threshold_mask_iou"] = binary_iou(pred_mask, truth_mask)
            summary["sr_threshold_boundary_f1"] = boundary_f1(edge_map(pred_mask), edge_map(truth_mask))

    mask_path = _resolve_optional_path(root, mask_prediction_path, DEFAULT_MASK_NAMES)
    if mask_path is not None:
        pred_mask = np.load(mask_path)
        summary.update(
            {
                "sr_mask_path": str(mask_path),
                "sr_mask_iou": binary_iou(pred_mask, mask),
                "sr_mask_boundary_f1": boundary_f1(edge_map(pred_mask), edge_map(mask)),
            }
        )

    edge_path = _resolve_optional_path(root, edge_prediction_path, DEFAULT_EDGE_NAMES)
    if edge_path is not None:
        pred_edge = np.load(edge_path)
        summary.update(
            {
                "sr_edge_path": str(edge_path),
                "sr_edge_iou": binary_iou(pred_edge, edge),
                "sr_edge_boundary_f1": boundary_f1(pred_edge, edge),
            }
        )
    return summary


def evaluate_dataset(
    dataset_root: str | Path,
    *,
    reconstruction_root: str | Path | None = None,
    reconstruction_filename: str = "sr_temperature_2x.npy",
) -> dict[str, Any]:
    """Evaluate every scene listed in a dataset manifest and aggregate metrics."""

    root = Path(dataset_root)
    manifest_path = root / "manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    summaries: list[dict[str, Any]] = []
    for row in rows:
        scene_dir = Path(row["scene_dir"])
        recon_path = None
        if reconstruction_root is not None:
            candidate = Path(reconstruction_root) / row.get("scene_id", scene_dir.name) / reconstruction_filename
            recon_path = candidate if candidate.exists() else None
        summaries.append(evaluate_scene(scene_dir, reconstruction_path=recon_path))
    return {"scenes": summaries, "aggregate": aggregate_scene_metrics(summaries)}


def aggregate_scene_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Aggregate numeric scene metrics with finite mean/std/count summaries."""

    aggregate: dict[str, float | int] = {"scene_count": len(rows)}
    if not rows:
        return aggregate
    keys = sorted({key for row in rows for key, value in row.items() if isinstance(value, (int, float, np.number))})
    for key in keys:
        vals = np.asarray([float(row[key]) for row in rows if key in row], dtype=np.float64)
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        aggregate[f"{key}_mean"] = float(np.mean(vals))
        aggregate[f"{key}_std"] = float(np.std(vals))
        aggregate[f"{key}_count"] = int(vals.size)
    return aggregate


def _paired(prediction: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pred = np.asarray(prediction, dtype=np.float64)
    truth = np.asarray(target, dtype=np.float64)
    if pred.shape != truth.shape:
        raise ValueError(f"shape mismatch: {pred.shape} != {truth.shape}")
    finite = np.isfinite(pred) & np.isfinite(truth)
    if not finite.all():
        raise ValueError("inputs contain NaN or Inf")
    return pred, truth


def _validate_scene_dir(scene_dir: str | Path) -> Path:
    root = Path(scene_dir)
    if not root.exists():
        raise FileNotFoundError(f"scene_dir not found: {root}")
    missing = [name for name in REQUIRED_SCENE_FILES if not (root / name).exists()]
    if missing:
        raise FileNotFoundError(f"{root} missing required scene files: {missing}")
    return root


def _read_metadata(scene_dir: Path) -> dict[str, Any]:
    with (scene_dir / "metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_scale(hr_shape: tuple[int, ...], lr_shape: tuple[int, ...]) -> int:
    if len(hr_shape) != 2 or len(lr_shape) != 2 or lr_shape[0] == 0:
        return 2
    return int(round(hr_shape[0] / lr_shape[0]))


def _resolve_optional_path(root: Path, explicit: str | Path | None, names: tuple[str, ...]) -> Path | None:
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"prediction file not found: {path}")
        return path
    for name in names:
        candidate = root / name
        if candidate.exists():
            return candidate
    return None
