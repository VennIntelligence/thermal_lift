"""Drizzle-based 2x/4x reconstruction for EP10.

The shift convention matches EP05/EP06:
``shift=(dx, dy)`` is in LR pixels and moves an observed frame into the
reference coordinate system. Therefore an LR pixel at ``(row, col)`` lands at
HR coordinate ``scale * (row + dy, col + dx)``.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from drizzle.resample import Drizzle
from scipy import ndimage


def _validate_scale(scale: int) -> int:
    scale = int(scale)
    if scale not in (2, 4):
        raise ValueError("EP10 Drizzle is scoped to 2x or exploratory 4x reconstruction")
    return scale


def _as_frames(frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError("frames must have shape (N, H, W) or (H, W)")
    if arr.shape[0] == 0:
        raise ValueError("at least one frame is required")
    return arr


def _as_shifts(shifts: np.ndarray, n_frames: int) -> np.ndarray:
    arr = np.asarray(shifts, dtype=np.float64)
    if arr.ndim == 1:
        if arr.shape != (2,):
            raise ValueError("a single shift must have shape (2,)")
        arr = np.repeat(arr[np.newaxis, :], n_frames, axis=0)
    if arr.shape != (n_frames, 2):
        raise ValueError("shifts must have shape (N, 2) with columns [dx, dy]")
    return arr


def _as_weights(weights: np.ndarray | None, n_frames: int) -> np.ndarray:
    if weights is None:
        return np.ones(n_frames, dtype=np.float32)
    arr = np.asarray(weights, dtype=np.float32)
    if arr.shape != (n_frames,):
        raise ValueError("weights must have shape (N,)")
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.maximum(arr, 0.0)
    if not np.any(arr > 0):
        raise ValueError("at least one frame weight must be positive")
    return arr


def build_pixmap(
    shift: tuple[float, float] | np.ndarray,
    *,
    lr_shape: tuple[int, int] = (480, 640),
    scale: int = 2,
) -> np.ndarray:
    """Build a Drizzle pixmap for one EP05/EP06 alignment shift.

    Drizzle expects ``pixmap[..., 0]`` to be output X/column coordinates and
    ``pixmap[..., 1]`` to be output Y/row coordinates.
    """

    scale = _validate_scale(scale)
    rows, cols = map(int, lr_shape)
    dx, dy = np.asarray(shift, dtype=np.float64)
    rr, cc = np.mgrid[0:rows, 0:cols].astype(np.float64)
    pixmap = np.empty((rows, cols, 2), dtype=np.float64)
    pixmap[..., 0] = scale * (cc + dx)
    pixmap[..., 1] = scale * (rr + dy)
    return pixmap


def drizzle_reconstruct(
    frames: np.ndarray,
    shifts: np.ndarray,
    *,
    scale: int = 2,
    pixfrac: float = 0.7,
    weights: np.ndarray | None = None,
    kernel: str = "square",
    coverage_threshold: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruct a 2x or exploratory 4x HR image with STScI Drizzle.

    Returns
    -------
    hr_image, coverage_map:
        Both have shape ``(H * scale, W * scale)``. ``coverage_map`` is a copy
        of Drizzle's ``out_wht`` accumulated weight/count image.
        HR pixels with ``coverage_map < coverage_threshold`` are marked NaN in
        ``hr_image``.
    """

    scale = _validate_scale(scale)
    if not (0.0 < float(pixfrac) <= 1.0):
        raise ValueError("pixfrac must be in (0, 1]")

    frame_arr = _as_frames(frames)
    shift_arr = _as_shifts(shifts, frame_arr.shape[0])
    frame_weights = _as_weights(weights, frame_arr.shape[0])

    n_frames, rows, cols = frame_arr.shape
    hr_shape = (rows * scale, cols * scale)
    driz = Drizzle(kernel=kernel, out_shape=hr_shape, disable_ctx=True)

    base_x = scale * np.arange(cols, dtype=np.float64)[np.newaxis, :]
    base_y = scale * np.arange(rows, dtype=np.float64)[:, np.newaxis]
    pixmap = np.empty((rows, cols, 2), dtype=np.float64)

    for frame, (dx, dy), frame_weight in zip(frame_arr, shift_arr, frame_weights, strict=True):
        if frame_weight <= 0:
            continue
        finite = np.isfinite(frame)
        clean = np.where(finite, frame, 0.0).astype(np.float32, copy=False)
        weight_map = finite.astype(np.float32, copy=False)

        pixmap[..., 0] = base_x + scale * float(dx)
        pixmap[..., 1] = base_y + scale * float(dy)
        driz.add_image(
            data=clean,
            exptime=1.0,
            pixmap=pixmap,
            weight_map=weight_map,
            wht_scale=float(frame_weight),
            pixfrac=float(pixfrac),
            pixel_scale_ratio=1.0 / float(scale),
            in_units="cps",
        )

    hr = np.asarray(driz.out_img, dtype=np.float32).copy()
    coverage = np.asarray(driz.out_wht, dtype=np.float32).copy()
    invalid = (~np.isfinite(hr)) | (~np.isfinite(coverage)) | (coverage < float(coverage_threshold))
    hr[invalid] = np.nan
    return hr, coverage


def gaussian_unsharp(
    image: np.ndarray,
    *,
    sigma: float = 1.0,
    amount: float = 0.3,
    mode: str = "nearest",
) -> np.ndarray:
    """Apply a small Gaussian unsharp mask while preserving NaN support."""

    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full(arr.shape, np.nan, dtype=np.float32)
    fill = float(np.nanmedian(arr[finite]))
    clean = np.where(finite, arr, fill)
    blurred = ndimage.gaussian_filter(clean, sigma=float(sigma), mode=mode)
    out = clean + float(amount) * (clean - blurred)
    out[~finite] = np.nan
    return out.astype(np.float32, copy=False)


def highpass_image(
    image: np.ndarray,
    *,
    sigma: float,
    mode: str = "nearest",
) -> np.ndarray:
    """Subtract a Gaussian background from a 2D image with NaN preservation."""

    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if not finite.any():
        return np.full(arr.shape, np.nan, dtype=np.float32)
    fill = float(np.nanmedian(arr[finite]))
    clean = np.where(finite, arr, fill)
    bg = ndimage.gaussian_filter(clean, sigma=float(sigma), mode=mode)
    out = clean - bg
    out[~finite] = np.nan
    return out.astype(np.float32, copy=False)


def coverage_statistics(coverage: np.ndarray, *, threshold: float = 1.0) -> dict[str, float]:
    """Summarize Drizzle coverage/count support."""

    arr = np.asarray(coverage, dtype=np.float64)
    finite = np.isfinite(arr)
    if not finite.any():
        return {
            "min_coverage": float("nan"),
            "coverage_lt1_fraction": float("nan"),
            "coverage_p05": float("nan"),
            "coverage_median": float("nan"),
            "coverage_p95": float("nan"),
        }
    values = arr[finite]
    return {
        "min_coverage": float(np.min(values)),
        "coverage_lt1_fraction": float(np.mean(values < float(threshold))),
        "coverage_p05": float(np.percentile(values, 5)),
        "coverage_median": float(np.percentile(values, 50)),
        "coverage_p95": float(np.percentile(values, 95)),
    }


def _pearson_finite(a: np.ndarray, b: np.ndarray) -> float:
    lhs = np.asarray(a, dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(lhs) & np.isfinite(rhs)
    if int(valid.sum()) < 2:
        return float("nan")
    return float(np.corrcoef(lhs[valid].ravel(), rhs[valid].ravel())[0, 1])


def raw_control_agreement(
    hr_highpass: np.ndarray,
    raw_frames: np.ndarray,
    shifts: np.ndarray,
    *,
    scale: int = 2,
    pixfrac: float = 0.7,
    highpass_sigma_lr: float = 5.0,
    coverage_threshold: float = 1.0,
) -> tuple[float, np.ndarray]:
    """Compare highpass-Drizzle with raw-Drizzle followed by HR highpass."""

    scale = _validate_scale(scale)
    raw_hr, _ = drizzle_reconstruct(
        raw_frames,
        shifts,
        scale=scale,
        pixfrac=pixfrac,
        coverage_threshold=coverage_threshold,
    )
    raw_control_hp = highpass_image(raw_hr, sigma=float(highpass_sigma_lr) * scale)
    return _pearson_finite(hr_highpass, raw_control_hp), raw_control_hp


def holdout_residual_mse(
    frames: np.ndarray,
    shifts: np.ndarray,
    *,
    scale: int = 2,
    pixfrac: float = 0.7,
    psf_sigma: float = 0.5,
    holdout_fn: Callable[[np.ndarray], np.ndarray] | None = None,
    coverage_threshold: float = 1.0,
    forward_fn: Callable[..., np.ndarray] | None = None,
) -> float:
    """Train on 80% of frames and measure forward residual MSE on holdout.

    The default holdout is ``frame_index % 5 == 0``. NaN HR pixels are filled
    with zero before applying the EP06 highpass-domain forward model.
    """

    frame_arr = _as_frames(frames)
    shift_arr = _as_shifts(shifts, frame_arr.shape[0])
    indices = np.arange(frame_arr.shape[0], dtype=int)
    holdout = (indices % 5) == 0 if holdout_fn is None else np.asarray(holdout_fn(indices), dtype=bool)
    if holdout.shape != (frame_arr.shape[0],):
        raise ValueError("holdout_fn must return a boolean mask with shape (N,)")
    if not np.any(holdout) or not np.any(~holdout):
        raise ValueError("holdout split must contain both train and holdout frames")

    hr, _ = drizzle_reconstruct(
        frame_arr[~holdout],
        shift_arr[~holdout],
        scale=scale,
        pixfrac=pixfrac,
        coverage_threshold=coverage_threshold,
    )
    hr_for_forward = np.nan_to_num(hr, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32, copy=False)

    if forward_fn is None:
        from common.forward_model import forward as forward_fn

    mses: list[float] = []
    for frame, shift in zip(frame_arr[holdout], shift_arr[holdout], strict=True):
        pred = forward_fn(hr_for_forward, shift, psf_sigma=psf_sigma, scale=scale)
        valid = np.isfinite(pred) & np.isfinite(frame)
        if np.any(valid):
            diff = np.asarray(pred[valid], dtype=np.float64) - np.asarray(frame[valid], dtype=np.float64)
            mses.append(float(np.mean(diff * diff)))
    return float(np.mean(mses)) if mses else float("nan")


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    """Finite-pixel PSNR helper for synthetic checks."""

    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    valid = np.isfinite(ref) & np.isfinite(est)
    if not np.any(valid):
        return float("nan")
    mse = float(np.mean((ref[valid] - est[valid]) ** 2))
    if mse <= 0:
        return float("inf")
    data_range = float(np.nanmax(ref[valid]) - np.nanmin(ref[valid]))
    return float(20.0 * np.log10(max(data_range, 1e-12) / np.sqrt(mse)))
