"""Shift-and-add reconstruction for EP06.

The EP06 convention is intentionally narrow:

* the output grid is 2x the LR grid;
* shifts are ``(dx, dy)`` in LR pixels;
* a positive shift is the displacement that moves the LR frame into the
  reference coordinate system.

SAA therefore scatters each LR pixel into the reference HR grid at
``scale * (pixel_coordinate + shift)``. This is suitable for highpass
structural maps as well as offset-corrected raw-temperature tracks, but EP06
does not claim 4x SR or absolute-temperature recovery.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import numpy as np
from scipy import ndimage


def _as_frames(frames: np.ndarray) -> np.ndarray:
    arr = np.asarray(frames, dtype=np.float64)
    if arr.ndim == 2:
        arr = arr[np.newaxis, ...]
    if arr.ndim != 3:
        raise ValueError("frames must have shape (n_frames, height, width)")
    if arr.shape[0] == 0:
        raise ValueError("at least one frame is required")
    return arr


def _as_shifts(shifts: np.ndarray, n_frames: int) -> np.ndarray:
    arr = np.asarray(shifts, dtype=np.float64)
    if arr.ndim == 1:
        if arr.size != 2:
            raise ValueError("a single shift must have two values: (dx, dy)")
        arr = np.repeat(arr[np.newaxis, :], n_frames, axis=0)
    if arr.shape != (n_frames, 2):
        raise ValueError("shifts must have shape (n_frames, 2) with columns (dx, dy)")
    return arr


def _as_weights(weights: np.ndarray | None, n_frames: int) -> np.ndarray:
    if weights is None:
        return np.ones(n_frames, dtype=np.float64)
    arr = np.asarray(weights, dtype=np.float64)
    if arr.shape != (n_frames,):
        raise ValueError("weights must have shape (n_frames,)")
    arr = np.where(np.isfinite(arr), arr, 0.0)
    arr = np.maximum(arr, 0.0)
    if not np.any(arr > 0):
        raise ValueError("at least one frame weight must be positive")
    return arr


def _resolve_workers(workers: int | None = None, n_jobs: int | None = None) -> int:
    value = n_jobs if n_jobs is not None else workers
    if value is None:
        return 1
    return max(1, int(value))


def _weighted_lr_average(frames: np.ndarray, weights: np.ndarray) -> np.ndarray:
    numer = np.zeros(frames.shape[1:], dtype=np.float64)
    denom = np.zeros(frames.shape[1:], dtype=np.float64)
    for frame, weight in zip(frames, weights, strict=True):
        if weight <= 0:
            continue
        finite = np.isfinite(frame)
        numer += np.where(finite, frame, 0.0) * weight
        denom += finite.astype(np.float64) * weight
    return np.divide(numer, denom, out=np.zeros_like(numer), where=denom > 0)


def _upsample_lr(image: np.ndarray, scale: int, order: int = 3) -> np.ndarray:
    return ndimage.zoom(image, zoom=(scale, scale), order=order, mode="nearest")


def _scatter_chunk(
    frames: np.ndarray,
    shifts: np.ndarray,
    weights: np.ndarray,
    *,
    scale: int,
    hr_shape: tuple[int, int],
    splat_sigma: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    h_lr, w_lr = frames.shape[1:]
    h_hr, w_hr = hr_shape
    accum = np.zeros(hr_shape, dtype=np.float64)
    norm = np.zeros(hr_shape, dtype=np.float64)
    accum_flat = accum.ravel()
    norm_flat = norm.ravel()

    base_y = scale * np.arange(h_lr, dtype=np.float64)[:, np.newaxis]
    base_x = scale * np.arange(w_lr, dtype=np.float64)[np.newaxis, :]
    use_gaussian = splat_sigma is not None and float(splat_sigma) > 0.0
    sigma = float(splat_sigma) if use_gaussian else 0.0
    radius = int(np.ceil(3.0 * sigma)) if use_gaussian else 0

    for frame, (dx, dy), frame_weight in zip(frames, shifts, weights, strict=True):
        if frame_weight <= 0:
            continue

        finite = np.isfinite(frame)
        if not np.any(finite):
            continue
        clean = np.where(finite, frame, 0.0)

        y_center = base_y + scale * dy
        x_center = base_x + scale * dx
        y0 = np.floor(y_center).astype(np.int64)
        x0 = np.floor(x_center).astype(np.int64)

        if use_gaussian:
            for oy in range(-radius, radius + 1):
                yy = (y0 + oy)[:, 0]
                valid_y = (yy >= 0) & (yy < h_hr)
                if not np.any(valid_y):
                    continue
                dy_frac = yy[:, np.newaxis] - y_center
                wy = np.exp(-0.5 * (dy_frac[:, 0] * dy_frac[:, 0]) / (sigma * sigma))
                for ox in range(-radius, radius + 1):
                    xx = (x0 + ox)[0, :]
                    valid_x = (xx >= 0) & (xx < w_hr)
                    if not np.any(valid_x):
                        continue
                    dx_frac = xx[np.newaxis, :] - x_center
                    wx = np.exp(-0.5 * (dx_frac[0, :] * dx_frac[0, :]) / (sigma * sigma))

                    rows = yy[valid_y]
                    cols = xx[valid_x]
                    finite_block = finite[np.ix_(valid_y, valid_x)]
                    if not np.any(finite_block):
                        continue
                    clean_block = clean[np.ix_(valid_y, valid_x)]
                    contribution_weight = frame_weight * wy[valid_y, np.newaxis] * wx[np.newaxis, valid_x]
                    accum[np.ix_(rows, cols)] += np.where(
                        finite_block,
                        clean_block * contribution_weight,
                        0.0,
                    )
                    norm[np.ix_(rows, cols)] += np.where(finite_block, contribution_weight, 0.0)
            continue

        fy = y_center - y0
        fx = x_center - x0

        for oy in (0, 1):
            yy = y0 + oy
            wy = (1.0 - fy) if oy == 0 else fy
            valid_y = (yy >= 0) & (yy < h_hr)
            for ox in (0, 1):
                xx = x0 + ox
                wx = (1.0 - fx) if ox == 0 else fx
                valid_x = (xx >= 0) & (xx < w_hr)

                bilinear = wy * wx
                mask = valid_y & valid_x & finite
                if not np.any(mask):
                    continue

                indices = yy * w_hr + xx
                contribution_weight = frame_weight * bilinear
                np.add.at(
                    accum_flat,
                    indices[mask],
                    (clean * contribution_weight)[mask],
                )
                np.add.at(norm_flat, indices[mask], contribution_weight[mask])

    return accum, norm


def _ranges(n_items: int, n_chunks: int) -> Iterable[tuple[int, int]]:
    edges = np.linspace(0, n_items, num=n_chunks + 1, dtype=int)
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        if start < stop:
            yield int(start), int(stop)


def reconstruct_saa(
    frames: np.ndarray,
    shifts: np.ndarray,
    weights: np.ndarray | None = None,
    *,
    scale: int = 2,
    workers: int | None = None,
    n_jobs: int | None = None,
    fill_missing: bool = True,
    splat_sigma: float | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    """Reconstruct a 2x HR reference-grid image with bilinear shift-and-add.

    Parameters
    ----------
    frames:
        LR observations with shape ``(n_frames, height, width)``.
    shifts:
        ``(dx, dy)`` alignment shifts in LR pixels. Applying this shift moves
        each LR frame into the reference coordinate system.
    weights:
        Optional non-negative per-frame quality weights. ``None`` gives the
        uniform SAA baseline.
    scale:
        Output scale. EP06 uses ``scale=2``; other values are rejected to avoid
        accidental 4x claims.
    workers / n_jobs:
        Number of per-frame accumulation workers. The default is conservative
        single-thread execution.
    fill_missing:
        Fill uncovered HR pixels with a bicubic upsample of the weighted LR
        average. This only affects pixels with zero accumulated support.
    splat_sigma:
        Optional Gaussian splat width in HR pixels. ``None`` preserves the
        original 2x2 bilinear splat exactly; positive values use a wider
        Gaussian footprint to avoid 4x LR-periodic weight troughs.
    """

    if scale not in (2, 4):
        raise ValueError("EP06 SAA is defined for scale=2 or 4 only")

    frames_arr = _as_frames(frames)
    shifts_arr = _as_shifts(shifts, frames_arr.shape[0])
    weights_arr = _as_weights(weights, frames_arr.shape[0])
    n_workers = min(_resolve_workers(workers, n_jobs), frames_arr.shape[0])

    h_lr, w_lr = frames_arr.shape[1:]
    hr_shape = (h_lr * scale, w_lr * scale)

    if n_workers == 1:
        accum, norm = _scatter_chunk(
            frames_arr,
            shifts_arr,
            weights_arr,
            scale=scale,
            hr_shape=hr_shape,
            splat_sigma=splat_sigma,
        )
    else:
        accum = np.zeros(hr_shape, dtype=np.float64)
        norm = np.zeros(hr_shape, dtype=np.float64)
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(
                    _scatter_chunk,
                    frames_arr[start:stop],
                    shifts_arr[start:stop],
                    weights_arr[start:stop],
                    scale=scale,
                    hr_shape=hr_shape,
                    splat_sigma=splat_sigma,
                )
                for start, stop in _ranges(frames_arr.shape[0], n_workers)
            ]
            for future in futures:
                chunk_accum, chunk_norm = future.result()
                accum += chunk_accum
                norm += chunk_norm

    hr = np.divide(accum, norm, out=np.zeros_like(accum), where=norm > eps)

    if fill_missing and np.any(norm <= eps):
        lr_fill = _weighted_lr_average(frames_arr, weights_arr)
        fill = _upsample_lr(lr_fill, scale=scale, order=3)
        hr = np.where(norm > eps, hr, fill)

    return hr


def saa_uniform(
    frames: np.ndarray,
    shifts: np.ndarray,
    *,
    scale: int = 2,
    workers: int | None = None,
    n_jobs: int | None = None,
    splat_sigma: float | None = None,
) -> np.ndarray:
    """Uniform-weight SAA reconstruction."""

    return reconstruct_saa(
        frames,
        shifts,
        weights=None,
        scale=scale,
        workers=workers,
        n_jobs=n_jobs,
        splat_sigma=splat_sigma,
    )


def saa_weighted(
    frames: np.ndarray,
    shifts: np.ndarray,
    weights: np.ndarray,
    *,
    scale: int = 2,
    workers: int | None = None,
    n_jobs: int | None = None,
    splat_sigma: float | None = None,
) -> np.ndarray:
    """Quality-weighted SAA reconstruction."""

    return reconstruct_saa(
        frames,
        shifts,
        weights=weights,
        scale=scale,
        workers=workers,
        n_jobs=n_jobs,
        splat_sigma=splat_sigma,
    )


shift_and_add = reconstruct_saa
reconstruct = reconstruct_saa
