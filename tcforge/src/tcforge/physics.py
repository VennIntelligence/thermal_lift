"""Temperature rendering, noise, edge maps, and reproducible drift models."""

from __future__ import annotations

from typing import Literal

import numpy as np
from scipy import ndimage

DriftModel = Literal["none", "scalar_offset", "lowfreq", "gain_offset"]


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _as_float_array(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim not in (2, 3):
        raise ValueError(f"{name} must be 2D or 3D")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def render_temperature_field(
    mask: np.ndarray,
    *,
    t_bg_c: float = 21.0,
    delta_t_c: float = 2.0,
    low_freq_amplitude_c: float = 0.2,
    low_freq_sigma_px: float = 96.0,
    seed: int | None = None,
) -> np.ndarray:
    """Render a smooth HR temperature field from a binary structure mask."""

    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError("mask must be 2D")
    if not np.isin(m, [0, 1]).all():
        raise ValueError("mask must be binary with values 0/1")
    field = np.full(m.shape, float(t_bg_c), dtype=np.float32)
    field += m.astype(np.float32) * float(delta_t_c)

    amp = float(low_freq_amplitude_c)
    if amp > 0:
        noise = _rng(seed).normal(size=m.shape).astype(np.float32)
        smooth = ndimage.gaussian_filter(noise, sigma=float(low_freq_sigma_px), mode="nearest")
        smooth -= float(np.mean(smooth))
        denom = float(np.max(np.abs(smooth)))
        if denom > 0:
            field += (amp * smooth / denom).astype(np.float32)
    return field.astype(np.float32, copy=False)


def add_noise(
    image: np.ndarray,
    *,
    noise_sigma_c: float = 0.0724,
    seed: int | None = None,
) -> np.ndarray:
    """Add independent Gaussian detector noise in Celsius."""

    arr = _as_float_array(image, "image")
    sigma = float(noise_sigma_c)
    if sigma < 0:
        raise ValueError("noise_sigma_c must be >= 0")
    if sigma == 0:
        return arr.copy()
    return (arr + _rng(seed).normal(0.0, sigma, size=arr.shape).astype(np.float32)).astype(np.float32, copy=False)


def edge_map(mask: np.ndarray, *, edge_width_px: int = 1) -> np.ndarray:
    """Return a binary contour map for a mask using morphology only."""

    arr = np.asarray(mask)
    if arr.ndim != 2:
        raise ValueError("mask must be 2D")
    binary = arr.astype(bool)
    width = int(edge_width_px)
    if width < 1:
        raise ValueError("edge_width_px must be >= 1")
    structure = np.ones((3, 3), dtype=bool)
    dilated = ndimage.binary_dilation(binary, structure=structure, iterations=width)
    eroded = ndimage.binary_erosion(binary, structure=structure, iterations=width, border_value=0)
    return np.logical_xor(dilated, eroded).astype(np.uint8)


def scalar_offset_drift(
    frames: np.ndarray,
    *,
    amplitude_c: float = 0.2,
    seed: int | None = None,
) -> np.ndarray:
    """Add one reproducible scalar offset per frame."""

    arr = _as_float_array(frames, "frames")
    if arr.ndim != 3:
        raise ValueError("frames must be 3D for drift models")
    offsets = _rng(seed).normal(0.0, float(amplitude_c), size=(arr.shape[0], 1, 1)).astype(np.float32)
    return (arr + offsets).astype(np.float32, copy=False)


def lowfreq_drift(
    frames: np.ndarray,
    *,
    amplitude_c: float = 0.2,
    sigma_px: float = 96.0,
    seed: int | None = None,
) -> np.ndarray:
    """Add a per-frame smooth spatial drift field."""

    arr = _as_float_array(frames, "frames")
    if arr.ndim != 3:
        raise ValueError("frames must be 3D for drift models")
    rng = _rng(seed)
    out = arr.copy()
    amp = float(amplitude_c)
    for idx in range(out.shape[0]):
        field = rng.normal(size=out.shape[1:]).astype(np.float32)
        field = ndimage.gaussian_filter(field, sigma=float(sigma_px), mode="nearest")
        field -= float(np.mean(field))
        denom = float(np.max(np.abs(field)))
        if denom > 0:
            out[idx] += (amp * field / denom).astype(np.float32)
    return out.astype(np.float32, copy=False)


def gain_offset_drift(
    frames: np.ndarray,
    *,
    gain_sigma: float = 0.01,
    offset_sigma_c: float = 0.1,
    seed: int | None = None,
) -> np.ndarray:
    """Apply a reproducible per-frame multiplicative gain and scalar offset."""

    arr = _as_float_array(frames, "frames")
    if arr.ndim != 3:
        raise ValueError("frames must be 3D for drift models")
    rng = _rng(seed)
    gain = rng.normal(1.0, float(gain_sigma), size=(arr.shape[0], 1, 1)).astype(np.float32)
    offset = rng.normal(0.0, float(offset_sigma_c), size=(arr.shape[0], 1, 1)).astype(np.float32)
    return (arr * gain + offset).astype(np.float32, copy=False)


def apply_drift(
    frames: np.ndarray,
    *,
    model: DriftModel = "none",
    seed: int | None = None,
    amplitude_c: float = 0.2,
    lowfreq_sigma_px: float = 96.0,
    gain_sigma: float = 0.01,
    offset_sigma_c: float = 0.1,
) -> np.ndarray:
    """Apply one of the supported P1 drift models to an LR burst."""

    if model == "none":
        return _as_float_array(frames, "frames").copy()
    if model == "scalar_offset":
        return scalar_offset_drift(frames, amplitude_c=amplitude_c, seed=seed)
    if model == "lowfreq":
        return lowfreq_drift(frames, amplitude_c=amplitude_c, sigma_px=lowfreq_sigma_px, seed=seed)
    if model == "gain_offset":
        return gain_offset_drift(frames, gain_sigma=gain_sigma, offset_sigma_c=offset_sigma_c, seed=seed)
    raise ValueError("model must be one of: none, scalar_offset, lowfreq, gain_offset")
