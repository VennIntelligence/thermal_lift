"""TCForge LR burst generation forward models."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Literal

import numpy as np
from scipy import ndimage

from ._ep06_reference.forward import forward as ep06_point_forward
from ._utils import resolve_workers

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


def physical_block_average_forward(
    image_hr: np.ndarray,
    shift: tuple[float, float] | np.ndarray,
    *,
    psf_sigma_lr_px: float = 0.5,
    scale: int = 2,
    mode: str = "constant",
) -> np.ndarray:
    """Predict one LR frame by shifted detector block averaging."""

    scale = int(scale)
    x = _validate_hr(image_hr, scale).astype(np.float64, copy=False)
    sigma_hr = max(0.0, float(psf_sigma_lr_px) * scale)
    blurred = ndimage.gaussian_filter(x, sigma=sigma_hr, mode=mode, cval=0.0) if sigma_hr > 0 else x
    h_lr, w_lr = blurred.shape[0] // scale, blurred.shape[1] // scale
    dx, dy = np.asarray(shift, dtype=np.float64)
    yy0 = scale * (np.arange(h_lr, dtype=np.float64) + dy)
    xx0 = scale * (np.arange(w_lr, dtype=np.float64) + dx)
    acc = np.zeros((h_lr, w_lr), dtype=np.float64)
    for oy in range(scale):
        for ox in range(scale):
            coords = np.meshgrid(yy0 + oy, xx0 + ox, indexing="ij")
            acc += ndimage.map_coordinates(
                blurred,
                coords,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            )
    return (acc / float(scale * scale)).astype(np.float32, copy=False)


def generate_lr_burst(
    image_hr: np.ndarray,
    shifts: np.ndarray,
    *,
    forward_mode: ForwardMode = "exact_ep06_point",
    psf_sigma_lr_px: float = 0.5,
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
        make_one = lambda shift: ep06_point_forward(hr, shift, psf_sigma=float(psf_sigma_lr_px), scale=scale).astype(
            np.float32,
            copy=False,
        )
    elif forward_mode == "physical_block_average":
        make_one = lambda shift: physical_block_average_forward(
            hr,
            shift,
            psf_sigma_lr_px=psf_sigma_lr_px,
            scale=scale,
        )
    else:
        raise AssertionError("unreachable forward_mode branch")

    if n_workers == 1:
        frames = [make_one(shift) for shift in shift_arr]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            frames = list(executor.map(make_one, shift_arr))
    return np.stack(frames, axis=0).astype(np.float32, copy=False)
