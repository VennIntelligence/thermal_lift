"""TCForge LR burst generation forward models."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
from scipy import ndimage

from ._ep06_reference.forward import forward as ep06_point_forward
from ._utils import resolve_workers
from .physics import apply_psf_blur

ForwardMode = Literal["exact_ep06_point", "physical_block_average"]
FORWARD_MODES: tuple[str, ...] = ("exact_ep06_point", "physical_block_average")


def _validate_hr(image_hr: np.ndarray, scale: int) -> np.ndarray:
    arr = np.asarray(image_hr, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("image_hr must be 2D")
    if int(scale) <= 0:
        raise ValueError("scale must be > 0")
    if arr.shape[0] % int(scale) or arr.shape[1] % int(scale):
        raise ValueError("image_hr shape must be divisible by scale")
    if not np.isfinite(arr).all():
        raise ValueError("image_hr contains NaN or Inf")
    return arr


def _validate_shifts(shifts: np.ndarray) -> np.ndarray:
    arr = np.asarray(shifts, dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError("shifts must have shape (N, 2) with columns [dx, dy]")
    if not np.isfinite(arr).all():
        raise ValueError("shifts contain NaN or Inf")
    return arr


def _block_average_from_blurred(
    blurred: np.ndarray,
    shift: tuple[float, float] | np.ndarray,
    *,
    scale: int = 2,
) -> np.ndarray:
    """Block-average a *pre-blurred* HR image for one LR frame.

    Internal helper — callers are responsible for PSF blurring beforehand.
    This avoids redundant gaussian_filter calls when generating multi-frame
    bursts from the same HR scene.

    Uses pure-numpy separable bilinear interpolation: the sampling coordinates
    are separable (row coords independent of column, and vice versa), so 2D
    bilinear interpolation decomposes into two 1D lookups.  This is much faster
    than calling ``map_coordinates`` 16 times (scale=4).
    """

    scale = int(scale)
    h_hr, w_hr = blurred.shape
    h_lr, w_lr = h_hr // scale, w_hr // scale
    dx, dy = np.asarray(shift, dtype=np.float64)

    # All row/col sample positions for every sub-pixel offset within the block
    # shape: (scale * h_lr,)
    offsets = np.arange(scale, dtype=np.float64)
    yy_all = (scale * (np.arange(h_lr, dtype=np.float64) + dy))[:, None] + offsets[None, :]  # (h_lr, scale)
    xx_all = (scale * (np.arange(w_lr, dtype=np.float64) + dx))[:, None] + offsets[None, :]  # (w_lr, scale)
    yy_flat = yy_all.ravel()  # (scale*h_lr,)
    xx_flat = xx_all.ravel()  # (scale*w_lr,)

    # Bilinear interpolation — separable decomposition
    # Step 1: clamp and compute integer indices + fractional weights for rows
    y0 = np.floor(yy_flat).astype(np.intp)
    fy = (yy_flat - y0).astype(np.float32)
    y0 = np.clip(y0, 0, h_hr - 1)
    y1 = np.clip(y0 + 1, 0, h_hr - 1)

    # Step 2: same for columns
    x0 = np.floor(xx_flat).astype(np.intp)
    fx = (xx_flat - x0).astype(np.float32)
    x0 = np.clip(x0, 0, w_hr - 1)
    x1 = np.clip(x0 + 1, 0, w_hr - 1)

    # Step 3: boundary masking (set out-of-bounds to cval=0)
    valid_y = (yy_flat >= 0.0) & (yy_flat <= h_hr - 1)
    valid_x = (xx_flat >= 0.0) & (xx_flat <= w_hr - 1)

    # Step 4: separable bilinear — interpolate along rows first, then columns
    # Intermediate: for each (y_sample, x_col), compute row-interpolated value
    #   interp_rows[yi, xj] = blurred[y0[yi], xj] * (1-fy[yi]) + blurred[y1[yi], xj] * fy[yi]
    # Then average over block offsets.
    # Since coords are separable, we can use advanced indexing efficiently.
    #
    # row_interp has shape (scale*h_lr, scale*w_lr)
    # blurred[y0, :]  shape: (scale*h_lr, w_hr)  — gather rows
    wy0 = (1.0 - fy)[:, None]  # (scale*h_lr, 1)
    wy1 = fy[:, None]
    rows_interp = blurred[y0, :] * wy0 + blurred[y1, :] * wy1  # (scale*h_lr, w_hr)

    # Apply y validity mask (out-of-range rows → 0)
    rows_interp *= valid_y[:, None]

    # Now interpolate columns: gather from rows_interp at x0, x1
    wx0 = 1.0 - fx  # (scale*w_lr,)
    wx1 = fx
    # result[yi, xj] = rows_interp[yi, x0[xj]] * wx0[xj] + rows_interp[yi, x1[xj]] * wx1[xj]
    result = rows_interp[:, x0] * wx0[None, :] + rows_interp[:, x1] * wx1[None, :]
    # Apply x validity mask
    result *= valid_x[None, :]

    # Reshape to (h_lr, scale, w_lr, scale) and average over the two scale dims
    result = result.reshape(h_lr, scale, w_lr, scale).mean(axis=(1, 3))
    return result.astype(np.float32, copy=False)


def physical_block_average_forward(
    image_hr: np.ndarray,
    shift: tuple[float, float] | np.ndarray,
    *,
    psf_sigma_lr_px: float = 0.5,
    psf_shape: str = "gaussian",
    psf_sigma_y_lr_px: float | None = None,
    psf_angle_deg: float = 0.0,
    psf_kernel: np.ndarray | None = None,
    scale: int = 2,
    mode: str = "constant",
) -> np.ndarray:
    """Predict one LR frame by shifted detector block averaging.

    Public single-frame API; blurs HR internally.  For multi-frame burst
    generation prefer :func:`generate_lr_burst` which pre-computes the blur.
    """

    scale = int(scale)
    x = _validate_hr(image_hr, scale).astype(np.float64, copy=False)
    blurred = apply_psf_blur(
        x,
        psf_sigma_lr_px=psf_sigma_lr_px,
        scale=scale,
        psf_shape=psf_shape,  # type: ignore[arg-type]
        psf_sigma_y_lr_px=psf_sigma_y_lr_px,
        psf_angle_deg=psf_angle_deg,
        psf_kernel=psf_kernel,
        mode=mode,
        cval=0.0,
    )
    return _block_average_from_blurred(blurred, shift, scale=scale)


def generate_lr_burst(
    image_hr: np.ndarray,
    shifts: np.ndarray,
    *,
    forward_mode: ForwardMode = "exact_ep06_point",
    psf_sigma_lr_px: float = 0.5,
    psf_shape: str = "gaussian",
    psf_sigma_y_lr_px: float | None = None,
    psf_angle_deg: float = 0.0,
    psf_kernel: np.ndarray | None = None,
    scale: int = 2,
    workers: int | None = None,
    n_jobs: int | None = None,
) -> np.ndarray:
    """Generate an LR burst using exactly one named forward mode."""

    hr = _validate_hr(image_hr, scale)
    shift_arr = _validate_shifts(shifts)
    if forward_mode not in FORWARD_MODES:
        raise ValueError(f"forward_mode must be one of {FORWARD_MODES}")
    n_workers = min(resolve_workers(workers, n_jobs), max(1, len(shift_arr)))

    if forward_mode == "exact_ep06_point":
        if psf_shape != "gaussian" or psf_sigma_y_lr_px is not None or psf_kernel is not None:
            raise ValueError("exact_ep06_point supports only isotropic Gaussian PSF")
        make_one = lambda shift: ep06_point_forward(hr, shift, psf_sigma=float(psf_sigma_lr_px), scale=scale).astype(
            np.float32,
            copy=False,
        )
    elif forward_mode == "physical_block_average":
        # Pre-compute PSF blur once — PSF is shift-independent, so blurring
        # 248 times was pure waste (~28s saved on 1920×2560 HR canvas).
        hr64 = hr.astype(np.float64, copy=False)
        blurred = apply_psf_blur(
            hr64,
            psf_sigma_lr_px=psf_sigma_lr_px,
            scale=scale,
            psf_shape=psf_shape,  # type: ignore[arg-type]
            psf_sigma_y_lr_px=psf_sigma_y_lr_px,
            psf_angle_deg=psf_angle_deg,
            psf_kernel=psf_kernel,
            mode="constant",
            cval=0.0,
        )
        make_one = lambda shift: _block_average_from_blurred(blurred, shift, scale=scale)
    else:
        raise AssertionError("unreachable forward_mode branch")

    if n_workers == 1:
        frames = [make_one(shift) for shift in shift_arr]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            frames = list(executor.map(make_one, shift_arr))
    return np.stack(frames, axis=0).astype(np.float32, copy=False)
