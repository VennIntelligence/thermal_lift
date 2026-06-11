"""LR burst fusion into compact observation features."""

from __future__ import annotations

import warnings

import numpy as np
from scipy import ndimage

# Obs-features channel layout (keep in sync with fuse_burst_to_features output)
OBS_CH_MEAN: int = 0
OBS_CH_MEDIAN: int = 1
OBS_CH_COVERAGE: int = 2
OBS_CH_VARIANCE: int = 3
OBS_CH_HIGHPASS: int = 4
OBS_N_CHANNELS: int = 5


def _validate_burst_and_shifts(lr_burst: np.ndarray, shifts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frames = np.asarray(lr_burst, dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError("lr_burst must have shape (N, H, W)")
    if frames.shape[0] <= 0:
        raise ValueError("lr_burst must contain at least one frame")
    if not np.isfinite(frames).all():
        raise ValueError("lr_burst contains NaN or Inf")
    shift_arr = np.asarray(shifts, dtype=np.float32)
    if shift_arr.ndim != 2 or shift_arr.shape != (frames.shape[0], 2):
        raise ValueError("shifts must have shape (N, 2) matching lr_burst")
    if not np.isfinite(shift_arr).all():
        raise ValueError("shifts contain NaN or Inf")
    return frames, shift_arr


def _aligned_stack(
    frames: np.ndarray,
    shifts: np.ndarray,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = map(int, output_shape)
    if rows <= 0 or cols <= 0:
        raise ValueError("output_shape entries must be positive")
    yy, xx = np.meshgrid(np.arange(rows, dtype=np.float32), np.arange(cols, dtype=np.float32), indexing="ij")
    aligned = np.empty((frames.shape[0], rows, cols), dtype=np.float32)
    valid = np.empty((frames.shape[0], rows, cols), dtype=bool)
    for idx, (frame, shift) in enumerate(zip(frames, shifts, strict=True)):
        dx, dy = float(shift[0]), float(shift[1])
        src_y = yy - dy
        src_x = xx - dx
        valid[idx] = (src_y >= 0.0) & (src_y <= frame.shape[0] - 1) & (src_x >= 0.0) & (src_x <= frame.shape[1] - 1)
        aligned[idx] = ndimage.map_coordinates(
            frame,
            (src_y, src_x),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
    return aligned, valid


def _shift_and_accumulate(
    frames: np.ndarray,
    shifts: np.ndarray,
    output_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Align frames to the reference grid and return sum plus coverage count."""

    frame_arr, shift_arr = _validate_burst_and_shifts(frames, shifts)
    aligned, valid = _aligned_stack(frame_arr, shift_arr, output_shape)
    summed = np.where(valid, aligned, 0.0).sum(axis=0, dtype=np.float32)
    coverage_count = valid.sum(axis=0, dtype=np.float32)
    return summed.astype(np.float32, copy=False), coverage_count.astype(np.float32, copy=False)


def fuse_burst_to_features(
    lr_burst: np.ndarray,
    shifts: np.ndarray,
    *,
    output_shape: tuple[int, int] | None = None,
    sigma_bg: float = 5.0,
) -> np.ndarray:
    """Fuse an N-frame LR burst into five 1x observation feature channels.

    Channels are aligned mean, aligned median, coverage fraction, variance, and
    highpass fused mean. Shifts are ``[dx, dy]`` LR-pixel alignment shifts that
    move each observed frame into reference-frame coordinates.

    Optimized: uses a single ``_aligned_stack`` call, then derives highpass from
    the aligned stack (align → blur → subtract) instead of the old approach
    (blur → subtract → re-align).
    """

    frames, shift_arr = _validate_burst_and_shifts(lr_burst, shifts)
    out_shape = tuple(map(int, output_shape)) if output_shape is not None else tuple(map(int, frames.shape[1:]))
    sigma = float(sigma_bg)
    if sigma < 0:
        raise ValueError("sigma_bg must be >= 0")

    # Single alignment pass for all downstream statistics
    aligned, valid = _aligned_stack(frames, shift_arr, out_shape)
    valid_f = valid.astype(np.float32)
    counts = valid_f.sum(axis=0)
    coverage = counts / float(frames.shape[0])
    safe_counts = np.maximum(counts, 1.0)

    aligned_valid = np.where(valid, aligned, 0.0)
    aligned_mean = aligned_valid.sum(axis=0, dtype=np.float32) / safe_counts
    median_stack = np.where(valid, aligned, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        aligned_median = np.nanmedian(median_stack, axis=0).astype(np.float32)
    aligned_median = np.where(np.isfinite(aligned_median), aligned_median, 0.0).astype(np.float32, copy=False)
    variance = (np.where(valid, (aligned - aligned_mean[None, :, :]) ** 2, 0.0).sum(axis=0) / safe_counts).astype(
        np.float32,
        copy=False,
    )

    # Highpass: compute from aligned stack (no second warp needed)
    # Blur each aligned frame, then subtract to get highpass, then fuse.
    # Fill invalid pixels with aligned_mean before blurring so the cval=0
    # boundary doesn't leak into the highpass through the Gaussian kernel.
    if sigma > 0:
        # Fill invalid regions with the local mean to neutralize boundary effects
        fill_value = aligned_mean[None, :, :]  # (1, H, W) broadcast
        aligned_filled = np.where(valid, aligned, fill_value)
        # Vectorized: blur all aligned frames at once using 3D gaussian_filter
        # with sigma=(0, sigma, sigma) — no temporal smoothing, spatial only
        aligned_blurred = ndimage.gaussian_filter(
            aligned_filled.astype(np.float32, copy=False),
            sigma=(0.0, sigma, sigma),
            mode="nearest",
        ).astype(np.float32, copy=False)
        hp_aligned = aligned_filled - aligned_blurred
    else:
        hp_aligned = np.zeros_like(aligned)
    hp_counts = np.maximum(valid_f.sum(axis=0), 1.0)
    highpass_fused = np.where(valid, hp_aligned, 0.0).sum(axis=0, dtype=np.float32) / hp_counts

    features = np.stack(
        [
            aligned_mean,
            aligned_median,
            coverage.astype(np.float32, copy=False),
            np.maximum(variance, 0.0),
            highpass_fused.astype(np.float32, copy=False),
        ],
        axis=0,
    )
    return np.where(np.isfinite(features), features, 0.0).astype(np.float32, copy=False)
