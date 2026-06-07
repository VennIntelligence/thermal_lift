"""Iterative back-projection for EP06.

Shift convention
----------------
``shift=(dx, dy)`` is the displacement, in LR pixels, that moves the raw LR
frame into the reference coordinate system. Therefore:

* SAA/adjoint use ``+shift`` to back-project residuals into the reference HR
  grid;
* ``forward(x_hr, shift)`` predicts the original raw observation by internally
  applying the reverse scene displacement, equivalent to sampling the reference
  HR image at ``pixel + shift`` after blur/downsample.

The implementation is matrix-free and limited to 2x EP06 structural maps.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import ndimage

from saa import reconstruct_saa

try:  # Optional shared implementation supplied by the EP06 common package.
    from common.forward_model import adjoint as _common_adjoint
    from common.forward_model import forward as _common_forward
except Exception:  # pragma: no cover - exercised when common is absent.
    _common_adjoint = None
    _common_forward = None


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


def _resolve_workers(workers: int | None = None, n_jobs: int | None = None) -> int:
    value = n_jobs if n_jobs is not None else workers
    if value is None:
        return 1
    return max(1, int(value))


def _psf_sigma_hr(psf_sigma: float, scale: int) -> float:
    sigma = float(psf_sigma)
    if sigma <= 0:
        return 0.0
    return sigma * scale


def _sample_reference_to_lr(
    image_hr: np.ndarray,
    shift: tuple[float, float] | np.ndarray,
    *,
    scale: int,
) -> np.ndarray:
    h_hr, w_hr = image_hr.shape
    if h_hr % scale != 0 or w_hr % scale != 0:
        raise ValueError("HR shape must be divisible by scale")
    h_lr, w_lr = h_hr // scale, w_hr // scale
    dx, dy = np.asarray(shift, dtype=np.float64)

    yy = scale * (np.arange(h_lr, dtype=np.float64) + dy)
    xx = scale * (np.arange(w_lr, dtype=np.float64) + dx)
    coords = np.meshgrid(yy, xx, indexing="ij")
    return ndimage.map_coordinates(
        image_hr,
        coords,
        order=1,
        mode="constant",
        cval=0.0,
        prefilter=False,
    )


def _scatter_lr_to_reference(
    image_lr: np.ndarray,
    shift: tuple[float, float] | np.ndarray,
    *,
    hr_shape: tuple[int, int],
    scale: int,
    splat_sigma: float | None = None,
) -> np.ndarray:
    lr = np.asarray(image_lr, dtype=np.float64)
    if lr.ndim != 2:
        raise ValueError("image_lr must be a 2D array")

    h_lr, w_lr = lr.shape
    h_hr, w_hr = hr_shape
    dx, dy = np.asarray(shift, dtype=np.float64)

    y = scale * (np.arange(h_lr, dtype=np.float64)[:, np.newaxis] + dy)
    x = scale * (np.arange(w_lr, dtype=np.float64)[np.newaxis, :] + dx)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)

    out = np.zeros(hr_shape, dtype=np.float64)
    out_flat = out.ravel()
    finite = np.isfinite(lr)
    clean = np.where(finite, lr, 0.0)
    use_gaussian = splat_sigma is not None and float(splat_sigma) > 0.0
    if use_gaussian:
        sigma = float(splat_sigma)
        radius = int(np.ceil(3.0 * sigma))
        for oy in range(-radius, radius + 1):
            yy = (y0 + oy)[:, 0]
            valid_y = (yy >= 0) & (yy < h_hr)
            if not np.any(valid_y):
                continue
            dy_frac = yy[:, np.newaxis] - y
            wy = np.exp(-0.5 * (dy_frac[:, 0] * dy_frac[:, 0]) / (sigma * sigma))
            for ox in range(-radius, radius + 1):
                xx = (x0 + ox)[0, :]
                valid_x = (xx >= 0) & (xx < w_hr)
                if not np.any(valid_x):
                    continue
                dx_frac = xx[np.newaxis, :] - x
                wx = np.exp(-0.5 * (dx_frac[0, :] * dx_frac[0, :]) / (sigma * sigma))

                rows = yy[valid_y]
                cols = xx[valid_x]
                finite_block = finite[np.ix_(valid_y, valid_x)]
                if not np.any(finite_block):
                    continue
                clean_block = clean[np.ix_(valid_y, valid_x)]
                weight = wy[valid_y, np.newaxis] * wx[np.newaxis, valid_x]
                out[np.ix_(rows, cols)] += np.where(finite_block, clean_block * weight, 0.0)
        return out

    fy = y - y0
    fx = x - x0

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
            np.add.at(out_flat, indices[mask], (clean * bilinear)[mask])

    return out


def forward(
    x_hr: np.ndarray,
    shift: tuple[float, float] | np.ndarray,
    psf_sigma: float = 1.0,
    *,
    scale: int = 2,
) -> np.ndarray:
    """Predict one raw LR observation from a reference HR image.

    ``shift`` is the LR-to-reference alignment displacement. Internally this is
    the same as moving the HR scene by ``-shift * scale`` before sampling the LR
    detector grid.
    """

    if _common_forward is not None:
        return np.asarray(_common_forward(x_hr, shift, psf_sigma=psf_sigma, scale=scale), dtype=np.float64)

    if scale not in (2, 4):
        raise ValueError("EP06 forward model is defined for scale=2 or 4 only")
    x = np.asarray(x_hr, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x_hr must be a 2D array")

    sigma_hr = _psf_sigma_hr(psf_sigma, scale)
    observed = (
        ndimage.gaussian_filter(x, sigma=sigma_hr, mode="constant", cval=0.0)
        if sigma_hr > 0
        else x
    )
    return _sample_reference_to_lr(observed, shift, scale=scale)


def adjoint(
    y_residual: np.ndarray,
    shift: tuple[float, float] | np.ndarray,
    psf_sigma: float = 1.0,
    *,
    hr_shape: tuple[int, int] | None = None,
    scale: int = 2,
    splat_sigma: float | None = None,
) -> np.ndarray:
    """Back-project one LR residual into the reference HR grid.

    ``splat_sigma`` optionally uses a wider Gaussian splat in HR pixels;
    ``None`` preserves the original bilinear adjoint.
    """

    if _common_adjoint is not None:
        return np.asarray(
            _common_adjoint(
                y_residual,
                shift,
                psf_sigma=psf_sigma,
                hr_shape=hr_shape,
                scale=scale,
                splat_sigma=splat_sigma,
            ),
            dtype=np.float64,
        )

    if scale not in (2, 4):
        raise ValueError("EP06 adjoint model is defined for scale=2 or 4 only")
    residual = np.asarray(y_residual, dtype=np.float64)
    if residual.ndim != 2:
        raise ValueError("y_residual must be a 2D array")
    if hr_shape is None:
        hr_shape = (residual.shape[0] * scale, residual.shape[1] * scale)

    scattered = _scatter_lr_to_reference(
        residual,
        shift,
        hr_shape=hr_shape,
        scale=scale,
        splat_sigma=splat_sigma,
    )
    sigma_hr = _psf_sigma_hr(psf_sigma, scale)
    if sigma_hr <= 0:
        return scattered
    return ndimage.gaussian_filter(scattered, sigma=sigma_hr, mode="constant", cval=0.0)


def _bicubic_initial(frames: np.ndarray, scale: int) -> np.ndarray:
    mean_lr = np.nanmean(frames, axis=0)
    return ndimage.zoom(mean_lr, zoom=(scale, scale), order=3, mode="nearest")


def _initial_image(
    frames: np.ndarray,
    shifts: np.ndarray,
    initial: str | np.ndarray,
    *,
    scale: int,
    workers: int,
    splat_sigma: float | None = None,
) -> np.ndarray:
    if isinstance(initial, str):
        key = initial.lower()
        if key == "saa":
            return reconstruct_saa(frames, shifts, scale=scale, workers=workers, splat_sigma=splat_sigma)
        if key in {"bicubic", "cubic"}:
            return _bicubic_initial(frames, scale)
        raise ValueError("initial must be 'saa', 'bicubic', or an HR ndarray")

    arr = np.asarray(initial, dtype=np.float64)
    expected = (frames.shape[1] * scale, frames.shape[2] * scale)
    if arr.shape != expected:
        raise ValueError(f"initial HR image must have shape {expected}")
    return arr.copy()


def _ranges(n_items: int, n_chunks: int) -> Iterable[tuple[int, int]]:
    edges = np.linspace(0, n_items, num=n_chunks + 1, dtype=int)
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        if start < stop:
            yield int(start), int(stop)


def _ibp_chunk(
    x_hr: np.ndarray,
    frames: np.ndarray,
    shifts: np.ndarray,
    *,
    psf_sigma: float,
    scale: int,
    splat_sigma: float | None = None,
) -> tuple[np.ndarray, float]:
    correction = np.zeros_like(x_hr, dtype=np.float64)
    sse = 0.0
    for frame, shift in zip(frames, shifts, strict=True):
        pred = forward(x_hr, shift, psf_sigma=psf_sigma, scale=scale)
        residual = np.where(np.isfinite(frame), frame - pred, 0.0)
        correction += adjoint(
            residual,
            shift,
            psf_sigma=psf_sigma,
            hr_shape=x_hr.shape,
            scale=scale,
            splat_sigma=splat_sigma,
        )
        sse += float(np.sum(residual * residual))
    return correction, sse


def _residual_backprojection(
    x_hr: np.ndarray,
    frames: np.ndarray,
    shifts: np.ndarray,
    *,
    psf_sigma: float,
    scale: int,
    workers: int,
    splat_sigma: float | None = None,
) -> tuple[np.ndarray, float]:
    workers = min(max(1, workers), frames.shape[0])
    if workers == 1:
        correction, sse = _ibp_chunk(
            x_hr,
            frames,
            shifts,
            psf_sigma=psf_sigma,
            scale=scale,
            splat_sigma=splat_sigma,
        )
    else:
        correction = np.zeros_like(x_hr, dtype=np.float64)
        sse = 0.0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _ibp_chunk,
                    x_hr,
                    frames[start:stop],
                    shifts[start:stop],
                    psf_sigma=psf_sigma,
                    scale=scale,
                    splat_sigma=splat_sigma,
                )
                for start, stop in _ranges(frames.shape[0], workers)
            ]
            for future in futures:
                chunk_correction, chunk_sse = future.result()
                correction += chunk_correction
                sse += chunk_sse

    correction /= frames.shape[0]
    residual_mse = sse / float(frames.size)
    return correction, residual_mse


def reconstruct_ibp(
    frames: np.ndarray,
    shifts: np.ndarray,
    *,
    initial: str | np.ndarray = "saa",
    max_iter: int = 50,
    beta: float = 0.5,
    tol: float = 1e-4,
    psf_sigma: float = 1.0,
    scale: int = 2,
    splat_sigma: float | None = None,
    workers: int | None = None,
    n_jobs: int | None = None,
    return_records: bool = False,
) -> tuple[np.ndarray, pd.DataFrame | list[dict[str, float | int | bool]]]:
    """Run iterative back-projection.

    Returns ``(image, convergence)``. By default convergence is a pandas
    ``DataFrame``; set ``return_records=True`` for a list of dictionaries.
    ``splat_sigma`` optionally widens the SAA initialization and adjoint splat
    in HR pixels; ``None`` preserves the original bilinear path.
    """

    if scale not in (2, 4):
        raise ValueError("EP06 IBP is defined for scale=2 or 4 only")
    frames_arr = _as_frames(frames)
    shifts_arr = _as_shifts(shifts, frames_arr.shape[0])
    n_workers = _resolve_workers(workers, n_jobs)
    x = _initial_image(
        frames_arr,
        shifts_arr,
        initial,
        scale=scale,
        workers=n_workers,
        splat_sigma=splat_sigma,
    )

    records: list[dict[str, float | int | bool]] = []
    max_iter = max(0, int(max_iter))
    beta = float(beta)

    for iteration in range(1, max_iter + 1):
        correction, residual_mse = _residual_backprojection(
            x,
            frames_arr,
            shifts_arr,
            psf_sigma=psf_sigma,
            scale=scale,
            workers=n_workers,
            splat_sigma=splat_sigma,
        )
        update = beta * correction
        denom = float(np.linalg.norm(x))
        rel_update = float(np.linalg.norm(update) / max(denom, 1e-12))
        x = x + update
        stopped = rel_update < tol

        records.append(
            {
                "iteration": iteration,
                "residual_mse": float(residual_mse),
                "relative_update": rel_update,
                "correction_norm": float(np.linalg.norm(correction)),
                "beta": beta,
                "psf_sigma": float(psf_sigma),
                "stopped": bool(stopped),
            }
        )
        if stopped:
            break

    convergence = records if return_records else pd.DataFrame.from_records(records)
    return x, convergence


ibp_reconstruct = reconstruct_ibp
reconstruct = reconstruct_ibp
