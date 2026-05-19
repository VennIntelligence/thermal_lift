"""Shared EP08 evaluation metrics.

The metrics in this module are intentionally lightweight and data-free: they
operate on reconstructed images, observations, and an injected forward operator.
They are proxies for contour-level SR validation, not optical ground truth.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import torch


def _first_tensor_device_dtype(*values: Any) -> tuple[torch.device, torch.dtype]:
    for value in values:
        if torch.is_tensor(value):
            return value.device, value.dtype
    return torch.device("cpu"), torch.float32


def _operator_device_dtype(forward_operator: Any) -> tuple[torch.device | None, torch.dtype | None]:
    if isinstance(forward_operator, torch.nn.Module):
        for tensor in list(forward_operator.parameters()) + list(forward_operator.buffers()):
            return tensor.device, tensor.dtype
    return None, None


def _as_torch(value: Any, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    if torch.is_tensor(value):
        return value.to(device=device, dtype=dtype)
    return torch.as_tensor(value, device=device, dtype=dtype)


def _as_numpy_float(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float64)


def _finite_pair(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]


def _normalized_indices(indices: Any, count: int) -> list[int]:
    if indices is None:
        return list(range(count))
    if torch.is_tensor(indices):
        values = indices.detach().cpu().numpy().tolist()
    elif isinstance(indices, np.ndarray):
        values = indices.tolist()
    elif isinstance(indices, Iterable) and not isinstance(indices, (str, bytes)):
        values = list(indices)
    else:
        raise TypeError("indices must be None or an iterable of frame indices")
    return [int(idx) for idx in values]


def holdout_residual(
    x_hr: torch.Tensor | np.ndarray,
    observations: torch.Tensor | np.ndarray,
    forward_operator: Any,
    indices: Sequence[int] | np.ndarray | torch.Tensor | None = None,
    noise_sigma: float = 0.0724,
) -> float:
    """Mean hold-out MSE normalized by detector noise variance.

    ``forward_operator`` is called through its single-frame interface as
    ``forward_operator(x_hr, index)``. Non-finite pixels are ignored per frame.
    """

    if noise_sigma <= 0:
        raise ValueError("noise_sigma must be positive")

    op_device, op_dtype = _operator_device_dtype(forward_operator)
    fallback_device, fallback_dtype = _first_tensor_device_dtype(x_hr, observations)
    device = op_device or fallback_device
    dtype = op_dtype or fallback_dtype
    if not torch.is_floating_point(torch.empty((), dtype=dtype)):
        dtype = torch.float32

    x = _as_torch(x_hr, device=device, dtype=dtype)
    obs = _as_torch(observations, device=device, dtype=dtype)
    if obs.ndim != 3:
        raise ValueError("observations must have shape (N, H, W)")
    selected = _normalized_indices(indices, int(obs.shape[0]))
    if not selected:
        return float("nan")

    frame_scores: list[torch.Tensor] = []
    with torch.no_grad():
        for index in selected:
            pred = forward_operator(x, int(index))
            target = obs[int(index)]
            if pred.shape != target.shape:
                raise ValueError(f"prediction shape {tuple(pred.shape)} does not match observation {tuple(target.shape)}")
            finite = torch.isfinite(pred) & torch.isfinite(target)
            if bool(finite.any()):
                mse = torch.mean((pred[finite] - target[finite]).square())
                frame_scores.append(mse / float(noise_sigma * noise_sigma))
    if not frame_scores:
        return float("nan")
    return float(torch.mean(torch.stack(frame_scores)).detach().cpu())


def split_half_nrmse(sr_a: torch.Tensor | np.ndarray, sr_b: torch.Tensor | np.ndarray) -> float:
    """Pixel-level NRMSE between two split-half reconstructions.

    The RMSE is normalized by the pooled finite-pixel standard deviation. This
    makes the metric insensitive to a common DC offset and keeps lower values
    better. If both images are flat and identical, the score is 0.
    """

    a, b = _finite_pair(_as_numpy_float(sr_a), _as_numpy_float(sr_b))
    if a.size == 0:
        return float("nan")
    rmse = float(np.sqrt(np.mean((a - b) ** 2)))
    pooled = np.concatenate([a, b])
    denom = float(np.std(pooled))
    if denom <= np.finfo(np.float64).eps:
        return 0.0 if rmse <= np.finfo(np.float64).eps else float("inf")
    return rmse / denom


def _laplacian_numpy(image: np.ndarray) -> np.ndarray:
    center = -4.0 * image
    out = center.copy()
    out[1:, :] += image[:-1, :]
    out[:-1, :] += image[1:, :]
    out[:, 1:] += image[:, :-1]
    out[:, :-1] += image[:, 1:]
    return out


def artifact_score(x_hr: torch.Tensor | np.ndarray, pin_mask: torch.Tensor | np.ndarray | None = None) -> float:
    """Laplacian energy in the pin/flat region.

    This follows the EP08 plan definition: ``mean(laplace(x_hr)[mask] ** 2)``.
    It is not EP06's heuristic ringing score. When ``pin_mask`` is ``None``,
    all finite image pixels are used.
    """

    image = _as_numpy_float(x_hr)
    if image.ndim != 2:
        raise ValueError("x_hr must be 2D")
    finite = np.isfinite(image)
    if pin_mask is None:
        mask = finite
    else:
        mask = np.asarray(_as_numpy_float(pin_mask), dtype=bool)
        if mask.shape != image.shape:
            raise ValueError(f"pin_mask shape {mask.shape} does not match image {image.shape}")
        mask = mask & finite
    if not np.any(mask):
        return float("nan")

    fill_value = float(np.nanmean(image[finite])) if np.any(finite) else 0.0
    clean = np.where(finite, image, fill_value)
    lap = _laplacian_numpy(clean)
    return float(np.mean(lap[mask] ** 2))


def raw_control_agreement(a: torch.Tensor | np.ndarray, b: torch.Tensor | np.ndarray) -> float:
    """Global SSIM-like correlation proxy without scikit-image.

    The score combines global luminance, contrast, and Pearson covariance terms.
    It is useful for a quick raw-control consistency check, but it is not a
    windowed SSIM implementation and should not be treated as optical ground
    truth. Higher is better; identical finite arrays return 1.
    """

    x, y = _finite_pair(_as_numpy_float(a), _as_numpy_float(b))
    if x.size == 0:
        return float("nan")
    if np.array_equal(x, y):
        return 1.0

    mux = float(np.mean(x))
    muy = float(np.mean(y))
    vx = float(np.mean((x - mux) ** 2))
    vy = float(np.mean((y - muy) ** 2))
    cov = float(np.mean((x - mux) * (y - muy)))
    data_range = float(np.max(np.concatenate([x, y])) - np.min(np.concatenate([x, y])))
    c1 = (0.01 * data_range) ** 2 + np.finfo(np.float64).eps
    c2 = (0.03 * data_range) ** 2 + np.finfo(np.float64).eps
    score = ((2.0 * mux * muy + c1) * (2.0 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2))
    return float(score)


def p95_gradient(image: torch.Tensor | np.ndarray) -> float:
    """95th percentile gradient magnitude over finite pixels."""

    arr = _as_numpy_float(image)
    if arr.ndim != 2:
        raise ValueError("image must be 2D")
    finite = np.isfinite(arr)
    if not np.any(finite):
        return float("nan")
    fill_value = float(np.nanmean(arr[finite]))
    clean = np.where(finite, arr, fill_value)
    gy, gx = np.gradient(clean)
    mag = np.sqrt(gx * gx + gy * gy)
    values = mag[finite & np.isfinite(mag)]
    if values.size == 0:
        return float("nan")
    return float(np.percentile(values, 95.0))


def summarize_metrics(
    *,
    x_hr: torch.Tensor | np.ndarray | None = None,
    observations: torch.Tensor | np.ndarray | None = None,
    forward_operator: Any | None = None,
    indices: Sequence[int] | np.ndarray | torch.Tensor | None = None,
    noise_sigma: float = 0.0724,
    split_a: torch.Tensor | np.ndarray | None = None,
    split_b: torch.Tensor | np.ndarray | None = None,
    artifact_image: torch.Tensor | np.ndarray | None = None,
    pin_mask: torch.Tensor | np.ndarray | None = None,
    raw_a: torch.Tensor | np.ndarray | None = None,
    raw_b: torch.Tensor | np.ndarray | None = None,
    gradient_image: torch.Tensor | np.ndarray | None = None,
) -> dict[str, float]:
    """Compute whichever EP08 metric inputs are supplied."""

    summary: dict[str, float] = {}
    if x_hr is not None and observations is not None and forward_operator is not None:
        summary["holdout_residual"] = holdout_residual(
            x_hr,
            observations,
            forward_operator,
            indices=indices,
            noise_sigma=noise_sigma,
        )
    if split_a is not None and split_b is not None:
        summary["split_half_nrmse"] = split_half_nrmse(split_a, split_b)
    if artifact_image is None:
        artifact_image = x_hr
    if artifact_image is not None:
        summary["artifact_score"] = artifact_score(artifact_image, pin_mask=pin_mask)
    if raw_a is not None and raw_b is not None:
        summary["raw_control_agreement"] = raw_control_agreement(raw_a, raw_b)
    if gradient_image is None:
        gradient_image = x_hr
    if gradient_image is not None:
        summary["p95_gradient"] = p95_gradient(gradient_image)
    return summary


__all__ = [
    "artifact_score",
    "holdout_residual",
    "p95_gradient",
    "raw_control_agreement",
    "split_half_nrmse",
    "summarize_metrics",
]
