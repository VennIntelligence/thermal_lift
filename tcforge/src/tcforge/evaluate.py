"""Numeric and scene-level evaluation helpers for TCForge outputs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from .classical_sr import DRIZZLE_CH_COVERAGE, DRIZZLE_CH_MEAN, drizzle_features
from .fusion import OBS_CH_COVERAGE, OBS_CH_HIGHPASS
from .manifest import COMPACT_4X_REQUIRED_FILES, FULL_2X_REQUIRED_FILES
from .physics import edge_map
from .reconstruct import reconstruct_hr_temperature
from .storage import load_scene_compact

REQUIRED_SCENE_FILES: tuple[str, ...] = FULL_2X_REQUIRED_FILES
COMPACT_SCENE_FILES: tuple[str, ...] = COMPACT_4X_REQUIRED_FILES
DEFAULT_RECONSTRUCTION_NAMES: tuple[str, ...] = (
    "sr_temperature_2x.npy",
    "sr_temperature_4x.npy",
    "reconstruction_2x.npy",
    "reconstruction_4x.npy",
    "reconstruction.npy",
)
DEFAULT_MASK_NAMES: tuple[str, ...] = ("sr_mask_2x.npy", "sr_mask_4x.npy", "mask_prediction_2x.npy", "mask_prediction_4x.npy")
DEFAULT_EDGE_NAMES: tuple[str, ...] = (
    "sr_edge_map_2x.npy",
    "sr_edge_map_4x.npy",
    "edge_prediction_2x.npy",
    "edge_prediction_4x.npy",
)


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


def masked_mae(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    pred, truth, valid = _paired_masked(prediction, target, mask)
    if not np.any(valid):
        return float("nan")
    return float(np.mean(np.abs(pred[valid] - truth[valid])))


def masked_rmse(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
    pred, truth, valid = _paired_masked(prediction, target, mask)
    if not np.any(valid):
        return float("nan")
    return float(np.sqrt(np.mean((pred[valid] - truth[valid]) ** 2)))


def split_half_drizzle_consistency(
    lr_burst: np.ndarray,
    shifts: np.ndarray,
    *,
    scale: int = 4,
    output_shape: tuple[int, int] | None = None,
    kernel: str = "bilinear",
    min_even_coverage: float = 0.0,
) -> dict[str, float | int]:
    """Measure odd/even drizzle consistency for one LR burst.

    Odd frames form the prediction-side direct observation; even frames form an
    independent target-side direct observation. Metrics are evaluated only
    where the even half has coverage above ``min_even_coverage``.
    """

    frames = np.asarray(lr_burst)
    shift_arr = np.asarray(shifts, dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError("lr_burst must have shape (N, H, W)")
    if shift_arr.shape != (frames.shape[0], 2):
        raise ValueError("shifts must have shape (N, 2) matching lr_burst")
    if frames.shape[0] < 2:
        raise ValueError("split-half consistency requires at least two frames")

    odd_idx = np.arange(0, frames.shape[0], 2)
    even_idx = np.arange(1, frames.shape[0], 2)
    if even_idx.size == 0:
        raise ValueError("split-half consistency requires at least one even frame")

    odd = drizzle_features(
        frames[odd_idx],
        shift_arr[odd_idx],
        scale=scale,
        output_shape=output_shape,
        kernel=kernel,
    )
    even = drizzle_features(
        frames[even_idx],
        shift_arr[even_idx],
        scale=scale,
        output_shape=tuple(map(int, odd.shape[1:])),
        kernel=kernel,
    )
    mask = even[DRIZZLE_CH_COVERAGE] > float(min_even_coverage)
    return {
        "split_half_odd_frames": int(odd_idx.size),
        "split_half_even_frames": int(even_idx.size),
        "split_half_eval_pixels": int(mask.sum()),
        "split_half_eval_fraction": float(mask.mean()),
        "split_half_odd_coverage_mean": float(np.mean(odd[DRIZZLE_CH_COVERAGE])),
        "split_half_even_coverage_mean": float(np.mean(even[DRIZZLE_CH_COVERAGE])),
        "split_half_mae_c": masked_mae(odd[DRIZZLE_CH_MEAN], even[DRIZZLE_CH_MEAN], mask),
        "split_half_rmse_c": masked_rmse(odd[DRIZZLE_CH_MEAN], even[DRIZZLE_CH_MEAN], mask),
    }


def split_half_drizzle_consistency_scene(
    scene_dir: str | Path,
    *,
    scale: int | None = None,
    kernel: str = "bilinear",
    min_even_coverage: float = 0.0,
) -> dict[str, Any]:
    """Load ``lr_burst.npy`` from a compact scene and run split-half drizzle consistency."""

    root = Path(scene_dir)
    scene = load_scene_compact(root)
    if "lr_burst" not in scene:
        raise FileNotFoundError(f"{root} has no lr_burst.npy; split-half consistency needs raw burst storage")
    metadata = scene["metadata"]
    eval_scale = int(scale if scale is not None else metadata.get("scale", 4))
    metrics = split_half_drizzle_consistency(
        np.asarray(scene["lr_burst"]),
        np.asarray(scene["shifts"]),
        scale=eval_scale,
        output_shape=tuple(metadata.get("hr_shape", scene["hr_mask"].shape)),
        kernel=kernel,
        min_even_coverage=min_even_coverage,
    )
    metrics.update(
        {
            "scene_id": metadata.get("scene_id", root.name),
            "scene_dir": str(root),
            "scale": eval_scale,
        }
    )
    return metrics


def summarize_scene(scene_dir: str | Path) -> dict[str, Any]:
    """Summarize one generated scene without requiring an SR reconstruction."""

    root, scene_format = _validate_scene_dir(scene_dir)
    if scene_format == "compact_4x":
        return _summarize_compact_scene(root)
    return _summarize_full_scene(root)


def _summarize_full_scene(root: Path) -> dict[str, Any]:
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


def _summarize_compact_scene(root: Path) -> dict[str, Any]:
    scene = load_scene_compact(root)
    metadata = scene["metadata"]
    mask = scene["hr_mask"]
    edge = scene["hr_edge"]
    obs = np.asarray(scene["obs_features"])
    shifts = np.asarray(scene["shifts"])
    shift_norms = np.linalg.norm(shifts, axis=1)
    obs_summary = finite_summary(obs)
    coverage = obs[OBS_CH_COVERAGE] if obs.ndim == 3 and obs.shape[0] > OBS_CH_COVERAGE else np.ones(obs.shape[-2:], dtype=np.float32)
    highpass = obs[OBS_CH_HIGHPASS] if obs.ndim == 3 and obs.shape[0] > OBS_CH_HIGHPASS else np.zeros(obs.shape[-2:], dtype=np.float32)
    lr_shape = tuple(metadata.get("lr_shape", obs.shape[-2:]))
    hr_shape = tuple(metadata.get("hr_shape", mask.shape))
    return {
        "scene_id": metadata.get("scene_id", root.name),
        "scene_dir": str(root),
        "scene_format": "compact_4x",
        "split": metadata.get("split", ""),
        "difficulty": metadata.get("difficulty", ""),
        "scale": int(metadata.get("scale", _infer_scale(mask.shape, obs.shape[-2:]))),
        "n_frames": int(metadata.get("n_frames", shifts.shape[0])),
        "lr_rows": int(lr_shape[0]),
        "lr_cols": int(lr_shape[1]),
        "hr_rows": int(hr_shape[0]),
        "hr_cols": int(hr_shape[1]),
        "mask_coverage": float(np.mean(mask > 0)),
        "edge_density": float(np.mean(edge > 0)),
        "obs_features_channels": int(obs.shape[0]) if obs.ndim == 3 else 0,
        "obs_features_finite_fraction": obs_summary["finite_fraction"],
        "coverage_mean": float(np.mean(coverage)),
        "coverage_min": float(np.min(coverage)),
        "highpass_abs_p95_c": float(np.percentile(np.abs(highpass), 95)),
        "shift_norm_mean_px": float(np.mean(shift_norms)) if shift_norms.size else 0.0,
        "shift_norm_max_px": float(np.max(shift_norms)) if shift_norms.size else 0.0,
    }


def evaluate_scene(
    scene_dir: str | Path,
    *,
    reconstruction_path: str | Path | None = None,
    mask_prediction_path: str | Path | None = None,
    edge_prediction_path: str | Path | None = None,
    threshold_c: float | None = None,
) -> dict[str, Any]:
    """Evaluate one scene and optional SR outputs against TCForge synthetic ground truth.

    If explicit prediction paths are not supplied, common filenames inside
    ``scene_dir`` are auto-detected. The returned dictionary always contains
    scene summaries; SR metric keys are added only when matching prediction
    files are present.
    """

    root, scene_format = _validate_scene_dir(scene_dir)
    summary = summarize_scene(root)
    if scene_format == "compact_4x":
        scene = load_scene_compact(root)
        mask = scene["hr_mask"]
        edge = scene["hr_edge"]
        hr = _reconstruct_compact_target(mask, scene["metadata"])
    else:
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


def _paired_masked(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    pred, truth = _paired(prediction, target)
    valid = np.asarray(mask).astype(bool)
    if valid.shape != pred.shape:
        raise ValueError(f"mask shape mismatch: {valid.shape} != {pred.shape}")
    return pred, truth, valid


def _validate_scene_dir(scene_dir: str | Path) -> tuple[Path, str]:
    root = Path(scene_dir)
    if not root.exists():
        raise FileNotFoundError(f"scene_dir not found: {root}")
    scene_format = _detect_scene_format(root)
    if scene_format is None:
        full_missing = [name for name in REQUIRED_SCENE_FILES if not (root / name).exists()]
        compact_missing = [name for name in COMPACT_SCENE_FILES if not (root / name).exists()]
        raise FileNotFoundError(
            f"{root} missing required scene files; full_2x missing: {full_missing}; "
            f"compact_4x missing: {compact_missing}"
        )
    return root, scene_format


def _detect_scene_format(root: Path) -> str | None:
    if all((root / name).exists() for name in REQUIRED_SCENE_FILES):
        return "full_2x"
    if all((root / name).exists() for name in COMPACT_SCENE_FILES):
        return "compact_4x"
    return None


def _read_metadata(scene_dir: Path) -> dict[str, Any]:
    with (scene_dir / "metadata.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def _infer_scale(hr_shape: tuple[int, ...], lr_shape: tuple[int, ...]) -> int:
    if len(hr_shape) != 2 or len(lr_shape) != 2 or lr_shape[0] == 0:
        return 2
    return int(round(hr_shape[0] / lr_shape[0]))


def _reconstruct_compact_target(mask: np.ndarray, metadata: dict[str, Any]) -> np.ndarray:
    return reconstruct_hr_temperature(
        mask,
        T_bg_c=float(metadata.get("T_bg_c", metadata.get("t_bg_c", 21.0))),
        delta_T_c=float(metadata.get("delta_T_c", metadata.get("delta_t_c", 2.0))),
        low_freq_amplitude_c=float(metadata.get("low_freq_amplitude_c", 0.2)),
        low_freq_sigma_px=float(metadata.get("low_freq_sigma_px", 96.0)),
        seed=metadata.get("low_freq_seed", metadata.get("seed")),
    )


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
