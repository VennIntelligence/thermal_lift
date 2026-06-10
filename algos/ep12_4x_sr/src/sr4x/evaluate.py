"""EP12 split-half evaluation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from tcforge.classical_sr import DRIZZLE_CH_COVERAGE, DRIZZLE_CH_MEAN, drizzle_features
from tcforge.evaluate import masked_mae, masked_rmse, split_half_drizzle_consistency_scene
from tcforge.storage import load_scene_compact

Predictor = Callable[[np.ndarray], np.ndarray]


def split_half_consistency(
    lr_burst: np.ndarray,
    shifts: np.ndarray,
    *,
    predictor: Predictor | None = None,
    scale: int = 4,
    output_shape: tuple[int, int] | None = None,
    kernel: str = "bilinear",
    min_even_coverage: float = 0.0,
) -> dict[str, float | int]:
    """Evaluate odd-frame prediction against even-frame drizzle observations.

    If ``predictor`` is omitted, odd-frame drizzle mean is used as the
    prediction. A model wrapper can be passed later as a callable that accepts
    odd drizzle features ``(3,H,W)`` and returns a single ``(H,W)`` prediction.
    """

    frames = np.asarray(lr_burst)
    shift_arr = np.asarray(shifts, dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError("lr_burst must have shape (N,H,W)")
    if shift_arr.shape != (frames.shape[0], 2):
        raise ValueError("shifts must have shape (N,2) matching lr_burst")
    if frames.shape[0] < 2:
        raise ValueError("split-half consistency requires at least two frames")

    odd_idx = np.arange(0, frames.shape[0], 2)
    even_idx = np.arange(1, frames.shape[0], 2)
    odd_features = drizzle_features(
        frames[odd_idx],
        shift_arr[odd_idx],
        scale=scale,
        output_shape=output_shape,
        kernel=kernel,
    )
    even_features = drizzle_features(
        frames[even_idx],
        shift_arr[even_idx],
        scale=scale,
        output_shape=tuple(map(int, odd_features.shape[1:])),
        kernel=kernel,
    )
    prediction = odd_features[DRIZZLE_CH_MEAN] if predictor is None else np.asarray(predictor(odd_features), dtype=np.float32)
    if prediction.shape != even_features.shape[1:]:
        raise ValueError(f"predictor output shape {prediction.shape} != {even_features.shape[1:]}")
    mask = even_features[DRIZZLE_CH_COVERAGE] > float(min_even_coverage)
    return {
        "split_half_odd_frames": int(odd_idx.size),
        "split_half_even_frames": int(even_idx.size),
        "split_half_eval_pixels": int(mask.sum()),
        "split_half_eval_fraction": float(mask.mean()),
        "split_half_odd_coverage_mean": float(np.mean(odd_features[DRIZZLE_CH_COVERAGE])),
        "split_half_even_coverage_mean": float(np.mean(even_features[DRIZZLE_CH_COVERAGE])),
        "split_half_mae_c": masked_mae(prediction, even_features[DRIZZLE_CH_MEAN], mask),
        "split_half_rmse_c": masked_rmse(prediction, even_features[DRIZZLE_CH_MEAN], mask),
    }


def split_half_consistency_scene(
    scene_dir: str | Path,
    *,
    predictor: Predictor | None = None,
    scale: int | None = None,
    kernel: str = "bilinear",
    min_even_coverage: float = 0.0,
) -> dict[str, Any]:
    """Run EP12 split-half consistency for one compact scene."""

    if predictor is None:
        return split_half_drizzle_consistency_scene(
            scene_dir,
            scale=scale,
            kernel=kernel,
            min_even_coverage=min_even_coverage,
        )

    root = Path(scene_dir)
    scene = load_scene_compact(root)
    if "lr_burst" not in scene:
        raise FileNotFoundError(f"{root} has no lr_burst.npy; split-half consistency needs raw burst storage")
    metadata = scene["metadata"]
    eval_scale = int(scale if scale is not None else metadata.get("scale", 4))
    metrics = split_half_consistency(
        np.asarray(scene["lr_burst"]),
        np.asarray(scene["shifts"]),
        predictor=predictor,
        scale=eval_scale,
        output_shape=tuple(metadata.get("hr_shape", scene["hr_mask"].shape)),
        kernel=kernel,
        min_even_coverage=min_even_coverage,
    )
    metrics.update({"scene_id": metadata.get("scene_id", root.name), "scene_dir": str(root), "scale": eval_scale})
    return metrics
