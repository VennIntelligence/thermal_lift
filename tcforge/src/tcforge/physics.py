"""Temperature rendering, noise, edge maps, and reproducible drift models."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import numpy as np
from scipy import ndimage, special

DriftModel = Literal["none", "scalar_offset", "lowfreq", "gain_offset", "temporal_trend"]
NoiseModel = Literal[
    "iid_gaussian",
    "fpn_lowfreq",
    "column_stripe",
    "spatial_correlated",
    "mixed",
    "detector_realistic",
]
PsfShape = Literal["gaussian", "elliptical_gaussian", "airy_disk"]
DefectMode = Literal["offset", "stuck"]

DEFAULT_PHYSICS_RANDOMIZATION: dict[str, object] = {
    "delta_T_c": {"dist": "uniform", "low": 0.5, "high": 5.0},
    "psf_sigma_lr_px": {"dist": "uniform", "low": 0.15, "high": 0.55},
    "noise_sigma_c": {"dist": "lognormal", "mean": 0.0724, "sigma_factor": 0.25},
    "low_freq_amplitude_c": {"dist": "uniform", "low": 0.02, "high": 0.5},
    "drift_amplitude_c": {"dist": "uniform", "low": 0.0, "high": 0.3},
}


def _rng(seed: int | None) -> np.random.Generator:
    return np.random.default_rng(seed)


def _as_float_array(array: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(array, dtype=np.float32)
    if arr.ndim not in (2, 3):
        raise ValueError(f"{name} must be 2D or 3D")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or Inf")
    return arr


def _sample_spec(rng: np.random.Generator, spec: object, name: str) -> float:
    if isinstance(spec, (int, float)):
        return float(spec)
    if isinstance(spec, Sequence) and not isinstance(spec, (str, bytes)):
        values = list(spec)
        if len(values) != 2:
            raise ValueError(f"{name} range must have exactly two values")
        low, high = float(values[0]), float(values[1])
        if high < low:
            raise ValueError(f"{name} range must satisfy high >= low")
        return float(rng.uniform(low, high))
    if not isinstance(spec, Mapping):
        raise ValueError(f"{name} must be a scalar, [low, high], or distribution mapping")

    dist = str(spec.get("dist", "uniform"))
    if dist == "constant":
        if "value" not in spec:
            raise ValueError(f"{name} constant distribution requires value")
        return float(spec["value"])
    if dist == "uniform":
        low, high = float(spec["low"]), float(spec["high"])
        if high < low:
            raise ValueError(f"{name} uniform distribution requires high >= low")
        return float(rng.uniform(low, high))
    if dist == "lognormal":
        mean = float(spec["mean"])
        sigma_factor = float(spec.get("sigma_factor", spec.get("sigma", 0.25)))
        if mean <= 0:
            raise ValueError(f"{name} lognormal mean must be > 0")
        if sigma_factor < 0:
            raise ValueError(f"{name} lognormal sigma_factor must be >= 0")
        return float(mean * np.exp(rng.normal(0.0, sigma_factor)))
    if dist == "choice":
        choices = list(spec["values"])
        if not choices:
            raise ValueError(f"{name} choice distribution requires non-empty values")
        weights = spec.get("weights")
        if weights is None:
            return float(rng.choice(choices))
        probs = np.asarray(list(weights), dtype=np.float64)
        if probs.size != len(choices) or np.any(probs < 0) or float(probs.sum()) <= 0:
            raise ValueError(f"{name} choice weights must match values and have positive sum")
        return float(rng.choice(choices, p=probs / probs.sum()))
    raise ValueError(f"unsupported distribution for {name}: {dist}")


def sample_physics_parameters(
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    config: Mapping[str, object] | None = None,
    difficulty: str | None = None,
) -> dict[str, float]:
    """Sample wide-range TCForge physics parameters from distribution specs.

    Specs may be scalars, ``[low, high]`` uniform ranges, or dictionaries with
    ``dist`` in ``constant``, ``uniform``, ``lognormal``, or ``choice``. Nested
    ``*_by_difficulty`` mappings are resolved when ``difficulty`` is supplied.
    """

    if rng is not None and seed is not None:
        raise ValueError("pass either rng or seed, not both")
    local_rng = rng if rng is not None else _rng(seed)
    specs = dict(DEFAULT_PHYSICS_RANDOMIZATION if config is None else config)
    out: dict[str, float] = {}
    for key, spec in specs.items():
        if key.endswith("_by_difficulty"):
            if difficulty is None:
                continue
            nested = dict(spec)  # type: ignore[arg-type]
            if difficulty not in nested:
                raise ValueError(f"{key} missing difficulty {difficulty!r}")
            out[key[: -len("_by_difficulty")]] = _sample_spec(local_rng, nested[difficulty], key)
        else:
            out[str(key)] = _sample_spec(local_rng, spec, str(key))
    return out


def render_temperature_field(
    mask: np.ndarray,
    *,
    t_bg_c: float = 21.0,
    delta_t_c: float = 2.0,
    temperature_offsets_c: Sequence[float] | None = None,
    low_freq_amplitude_c: float = 0.2,
    low_freq_sigma_px: float = 96.0,
    seed: int | None = None,
) -> np.ndarray:
    """Render a smooth HR temperature field from a coverage or label mask.

    By default this accepts a binary mask or soft coverage mask in ``[0, 1]``:
    background is ``t_bg_c`` and coverage 1 is ``t_bg_c + delta_t_c``.
    Pass ``temperature_offsets_c`` to render integer multi-temperature labels,
    where each label indexes an offset from ``t_bg_c`` and label 0 is usually
    the background offset.
    """

    m = np.asarray(mask)
    if m.ndim != 2:
        raise ValueError("mask must be 2D")
    field = np.full(m.shape, float(t_bg_c), dtype=np.float32)
    if temperature_offsets_c is None:
        coverage = m.astype(np.float32, copy=False)
        if not np.isfinite(coverage).all():
            raise ValueError("mask contains NaN or Inf")
        if float(coverage.min()) < 0.0 or float(coverage.max()) > 1.0:
            raise ValueError("mask must be binary or soft coverage in [0, 1] unless temperature_offsets_c is provided")
        field += coverage * float(delta_t_c)
    else:
        offsets = np.asarray(list(temperature_offsets_c), dtype=np.float32)
        if offsets.ndim != 1 or offsets.size == 0:
            raise ValueError("temperature_offsets_c must be a non-empty 1D sequence")
        if not np.allclose(m, np.round(m)):
            raise ValueError("multi-temperature masks must contain integer labels")
        labels = np.asarray(m, dtype=np.int64)
        if int(labels.min()) < 0 or int(labels.max()) >= offsets.size:
            raise ValueError("mask labels must index temperature_offsets_c")
        field += offsets[labels].astype(np.float32, copy=False)

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
    noise_model: NoiseModel = "iid_gaussian",
    fpn_sigma_px: float = 5.0,
    stripe_sigma_c: float | None = None,
    mix_weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Add reproducible LR detector noise in Celsius.

    ``noise_sigma_c`` is the total residual RMS anchor. Correlated models
    normalize their generated residual field back to this target so adding
    spatial texture does not silently change the noise budget. ``mix_weights``
    (default None => historical blend) is forwarded to :func:`make_noise`.
    """

    arr = _as_float_array(image, "image")
    sigma = float(noise_sigma_c)
    if sigma < 0:
        raise ValueError("noise_sigma_c must be >= 0")
    if sigma == 0:
        return arr.copy()
    residual = make_noise(
        arr.shape,
        noise_sigma_c=sigma,
        seed=seed,
        noise_model=noise_model,
        fpn_sigma_px=fpn_sigma_px,
        stripe_sigma_c=stripe_sigma_c,
        mix_weights=mix_weights,
    )
    return (arr + residual).astype(np.float32, copy=False)


def _normalize_noise(residual: np.ndarray, target_sigma: float) -> np.ndarray:
    out = np.asarray(residual, dtype=np.float32)
    out -= float(np.mean(out))
    current = float(np.std(out))
    if current <= 0:
        return np.zeros_like(out, dtype=np.float32)
    out *= float(target_sigma) / current
    out -= float(np.mean(out))
    return out.astype(np.float32, copy=False)


def _spatial_base_shape(shape: tuple[int, ...]) -> tuple[int, int]:
    if len(shape) == 2:
        return int(shape[0]), int(shape[1])
    if len(shape) == 3:
        return int(shape[1]), int(shape[2])
    raise ValueError("shape must be 2D or 3D")


def _broadcast_spatial(field: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    if len(shape) == 2:
        return field.astype(np.float32, copy=False)
    return np.broadcast_to(field.astype(np.float32, copy=False), shape).copy()


def _lowfreq_fpn(shape: tuple[int, ...], rng: np.random.Generator, sigma_px: float) -> np.ndarray:
    spatial_shape = _spatial_base_shape(shape)
    field = rng.normal(size=spatial_shape).astype(np.float32)
    field = ndimage.gaussian_filter(field, sigma=max(0.0, float(sigma_px)), mode="nearest")
    return _broadcast_spatial(_normalize_noise(field, 1.0), shape)


def _column_stripes(shape: tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    rows, cols = _spatial_base_shape(shape)
    stripe = rng.normal(size=(1, cols)).astype(np.float32)
    field = np.broadcast_to(stripe, (rows, cols)).copy()
    return _broadcast_spatial(_normalize_noise(field, 1.0), shape)


def _spatial_correlated_noise(shape: tuple[int, ...], rng: np.random.Generator, sigma_px: float) -> np.ndarray:
    sigma = max(0.0, float(sigma_px))
    if len(shape) == 2:
        field = rng.normal(size=shape).astype(np.float32)
        if sigma > 0:
            field = ndimage.gaussian_filter(field, sigma=sigma, mode="nearest")
    else:
        field = rng.normal(size=shape).astype(np.float32)
        if sigma > 0:
            field = ndimage.gaussian_filter(field, sigma=(0.0, sigma, sigma), mode="nearest")
    return _normalize_noise(field, 1.0)


def powerlaw_field(
    shape: tuple[int, int],
    alpha: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Unit-std zero-mean 2D field whose radial power spectrum follows P(f) ~ f^-alpha.

    FFT synthesis: one white-noise draw -> FFT -> multiply amplitude by f^(-alpha/2) (so
    power ~ f^-alpha), zero the DC (f=0) bin -> inverse FFT real part -> _normalize_noise(.,1).
    Exactly ONE rng.normal(size=shape) draw (deterministic order), returns float32. Used by
    realism.field_noise_burst for the true 1/f^alpha static low-frequency detector field."""
    h, w = int(shape[0]), int(shape[1])
    white = rng.normal(size=(h, w)).astype(np.float64)
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.fftfreq(w)[None, :]
    f = np.sqrt(fy * fy + fx * fx)
    amp = np.zeros_like(f)
    nz = f > 0
    amp[nz] = f[nz] ** (-float(alpha) / 2.0)   # amplitude^2 ~ f^-alpha => PSD ~ f^-alpha
    spec = np.fft.fft2(white) * amp
    field = np.fft.ifft2(spec).real.astype(np.float32)
    return _normalize_noise(field, 1.0)


def make_noise(
    shape: tuple[int, ...],
    *,
    noise_sigma_c: float = 0.0724,
    seed: int | None = None,
    noise_model: NoiseModel = "iid_gaussian",
    fpn_sigma_px: float = 5.0,
    stripe_sigma_c: float | None = None,
    mix_weights: Mapping[str, float] | None = None,
) -> np.ndarray:
    """Generate a zero-mean detector-noise residual field.

    ``mix_weights`` (default None => the historical hard-coded blend, bit-identical) overrides the
    relative family mixing coefficients per model. Recognised keys / defaults:
      - fpn_lowfreq:        fpn_w=0.75, iid_w=0.25
      - column_stripe:      stripe_default_scale=0.5, stripe_scale_cap=0.95
      - spatial_correlated: corr_w=0.70, iid_w=0.30
      - mixed:              fpn_w=0.55, stripe_default_scale=0.25, stripe_scale_cap=0.75, iid_w=0.35
      - detector_realistic: fpn_w=0.45, corr_w=0.30, stripe_default_scale=0.20, stripe_scale_cap=0.50,
                            iid_w=0.25, corr_sigma_factor=0.5, corr_sigma_min=1.0
    These are RELATIVE weights: the summed residual is always renormalized by _normalize_noise to
    the ``noise_sigma_c`` anchor, so mix_weights change the spatial *character*, not the RMS budget.
    They alter neither the number nor the order of RNG draws (stream-invariant)."""

    shape = tuple(int(v) for v in shape)
    if len(shape) not in (2, 3) or any(v <= 0 for v in shape):
        raise ValueError("shape must be a positive 2D or 3D shape")
    sigma = float(noise_sigma_c)
    if sigma < 0:
        raise ValueError("noise_sigma_c must be >= 0")
    if sigma == 0:
        return np.zeros(shape, dtype=np.float32)
    if fpn_sigma_px < 0:
        raise ValueError("fpn_sigma_px must be >= 0")
    if stripe_sigma_c is not None and float(stripe_sigma_c) < 0:
        raise ValueError("stripe_sigma_c must be >= 0")

    rng = _rng(seed)
    if noise_model == "iid_gaussian":
        return rng.normal(0.0, sigma, size=shape).astype(np.float32)
    if noise_model == "fpn_lowfreq":
        mw = {"fpn_w": 0.75, "iid_w": 0.25}
        mw.update(mix_weights or {})
        fpn = _lowfreq_fpn(shape, rng, fpn_sigma_px)
        iid = rng.normal(size=shape).astype(np.float32)
        residual = mw["fpn_w"] * fpn + mw["iid_w"] * _normalize_noise(iid, 1.0)
        return _normalize_noise(residual, sigma)
    if noise_model == "column_stripe":
        mw = {"stripe_default_scale": 0.5, "stripe_scale_cap": 0.95}
        mw.update(mix_weights or {})
        stripe_scale = float(stripe_sigma_c) / sigma if stripe_sigma_c is not None else mw["stripe_default_scale"]
        iid_scale = max(0.0, 1.0 - min(stripe_scale, mw["stripe_scale_cap"]))
        stripes = _column_stripes(shape, rng)
        iid = _normalize_noise(rng.normal(size=shape).astype(np.float32), 1.0)
        residual = stripe_scale * stripes + iid_scale * iid
        return _normalize_noise(residual, sigma)
    if noise_model == "spatial_correlated":
        mw = {"corr_w": 0.70, "iid_w": 0.30}
        mw.update(mix_weights or {})
        corr = _spatial_correlated_noise(shape, rng, fpn_sigma_px)
        iid = _normalize_noise(rng.normal(size=shape).astype(np.float32), 1.0)
        residual = mw["corr_w"] * corr + mw["iid_w"] * iid
        return _normalize_noise(residual, sigma)
    if noise_model == "mixed":
        mw = {"fpn_w": 0.55, "stripe_default_scale": 0.25, "stripe_scale_cap": 0.75, "iid_w": 0.35}
        mw.update(mix_weights or {})
        stripe_scale = float(stripe_sigma_c) / sigma if stripe_sigma_c is not None else mw["stripe_default_scale"]
        fpn = _lowfreq_fpn(shape, rng, fpn_sigma_px)
        stripes = _column_stripes(shape, rng)
        iid = _normalize_noise(rng.normal(size=shape).astype(np.float32), 1.0)
        residual = mw["fpn_w"] * fpn + min(stripe_scale, mw["stripe_scale_cap"]) * stripes + mw["iid_w"] * iid
        return _normalize_noise(residual, sigma)
    if noise_model == "detector_realistic":
        mw = {"fpn_w": 0.45, "corr_w": 0.30, "stripe_default_scale": 0.20, "stripe_scale_cap": 0.50,
              "iid_w": 0.25, "corr_sigma_factor": 0.5, "corr_sigma_min": 1.0}
        mw.update(mix_weights or {})
        stripe_scale = float(stripe_sigma_c) / sigma if stripe_sigma_c is not None else mw["stripe_default_scale"]
        fpn = _lowfreq_fpn(shape, rng, fpn_sigma_px)
        corr = _spatial_correlated_noise(shape, rng, max(mw["corr_sigma_min"], fpn_sigma_px * mw["corr_sigma_factor"]))
        stripes = _column_stripes(shape, rng)
        iid = _normalize_noise(rng.normal(size=shape).astype(np.float32), 1.0)
        residual = (mw["fpn_w"] * fpn + mw["corr_w"] * corr
                    + min(stripe_scale, mw["stripe_scale_cap"]) * stripes + mw["iid_w"] * iid)
        return _normalize_noise(residual, sigma)
    raise ValueError(
        "noise_model must be one of: iid_gaussian, fpn_lowfreq, column_stripe, "
        "spatial_correlated, mixed, detector_realistic"
    )


def add_detector_defects(
    image: np.ndarray,
    *,
    defect_rate: float = 0.001,
    seed: int | None = None,
    mode: DefectMode = "offset",
    hot_delta_c: float = 0.5,
    cold_delta_c: float = -0.5,
    stuck_values_c: tuple[float, float] | None = None,
) -> np.ndarray:
    """Inject reproducible fixed hot/cold detector defects into 2D or 3D data."""

    arr = _as_float_array(image, "image")
    rate = float(defect_rate)
    if rate < 0:
        raise ValueError("defect_rate must be >= 0")
    if rate == 0:
        return arr.copy()
    rows, cols = _spatial_base_shape(arr.shape)
    total = rows * cols
    n_defects = min(total, int(round(total * rate)))
    if n_defects == 0:
        return arr.copy()
    rng = _rng(seed)
    flat_idx = rng.choice(total, size=n_defects, replace=False)
    yy = flat_idx // cols
    xx = flat_idx % cols
    out = arr.copy()
    if mode == "offset":
        offsets = rng.choice([float(hot_delta_c), float(cold_delta_c)], size=n_defects).astype(np.float32)
        if out.ndim == 2:
            out[yy, xx] += offsets
        else:
            out[:, yy, xx] += offsets[None, :]
        return out.astype(np.float32, copy=False)
    if mode == "stuck":
        if stuck_values_c is None:
            low = float(np.percentile(arr, 1.0))
            high = float(np.percentile(arr, 99.0))
        else:
            low, high = float(stuck_values_c[0]), float(stuck_values_c[1])
        values = rng.choice([low, high], size=n_defects).astype(np.float32)
        if out.ndim == 2:
            out[yy, xx] = values
        else:
            out[:, yy, xx] = values[None, :]
        return out.astype(np.float32, copy=False)
    raise ValueError("mode must be one of: offset, stuck")


def make_psf_kernel(
    *,
    psf_sigma_lr_px: float = 0.5,
    scale: int = 2,
    psf_shape: PsfShape = "gaussian",
    psf_sigma_y_lr_px: float | None = None,
    psf_angle_deg: float = 0.0,
    kernel_radius_sigma: float = 4.0,
) -> np.ndarray:
    """Create a normalized HR-grid PSF kernel for synthetic forward models."""

    scale = int(scale)
    if scale <= 0:
        raise ValueError("scale must be > 0")
    sigma_x = float(psf_sigma_lr_px) * scale
    sigma_y = (float(psf_sigma_y_lr_px) if psf_sigma_y_lr_px is not None else float(psf_sigma_lr_px)) * scale
    if sigma_x < 0 or sigma_y < 0:
        raise ValueError("PSF sigmas must be >= 0")
    if sigma_x == 0 and sigma_y == 0:
        return np.ones((1, 1), dtype=np.float32)
    radius = int(np.ceil(max(sigma_x, sigma_y, 1.0) * float(kernel_radius_sigma)))
    radius = max(radius, 1)
    yy, xx = np.mgrid[-radius : radius + 1, -radius : radius + 1].astype(np.float64)
    theta = np.radians(float(psf_angle_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    xr = xx * cos_t + yy * sin_t
    yr = -xx * sin_t + yy * cos_t
    sx = max(sigma_x, 1e-6)
    sy = max(sigma_y, 1e-6)

    if psf_shape == "gaussian":
        kernel = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    elif psf_shape == "elliptical_gaussian":
        kernel = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    elif psf_shape == "airy_disk":
        r = np.sqrt((xr / sx) ** 2 + (yr / sy) ** 2)
        z = np.pi * np.maximum(r, 1e-8)
        kernel = (2.0 * special.j1(z) / z) ** 2
        kernel[r < 1e-8] = 1.0
    else:
        raise ValueError("psf_shape must be one of: gaussian, elliptical_gaussian, airy_disk")
    kernel = np.maximum(kernel, 0.0)
    total = float(kernel.sum())
    if total <= 0:
        raise ValueError("PSF kernel has zero mass")
    return (kernel / total).astype(np.float32)


def apply_psf_blur(
    image: np.ndarray,
    *,
    psf_sigma_lr_px: float = 0.5,
    scale: int = 2,
    psf_shape: PsfShape = "gaussian",
    psf_sigma_y_lr_px: float | None = None,
    psf_angle_deg: float = 0.0,
    psf_kernel: np.ndarray | None = None,
    mode: str = "constant",
    cval: float = 0.0,
) -> np.ndarray:
    """Blur a 2D HR image with an isotropic or randomized PSF shape."""

    arr = _as_float_array(image, "image")
    if arr.ndim != 2:
        raise ValueError("image must be 2D")
    if (
        psf_kernel is None
        and psf_shape == "gaussian"
        and psf_sigma_y_lr_px is None
        and float(psf_angle_deg) == 0.0
    ):
        sigma_hr = max(0.0, float(psf_sigma_lr_px) * int(scale))
        return (
            ndimage.gaussian_filter(arr.astype(np.float64, copy=False), sigma=sigma_hr, mode=mode, cval=float(cval))
            if sigma_hr > 0
            else arr.copy()
        ).astype(np.float32, copy=False)
    kernel = (
        np.asarray(psf_kernel, dtype=np.float32)
        if psf_kernel is not None
        else make_psf_kernel(
            psf_sigma_lr_px=psf_sigma_lr_px,
            scale=scale,
            psf_shape=psf_shape,
            psf_sigma_y_lr_px=psf_sigma_y_lr_px,
            psf_angle_deg=psf_angle_deg,
        )
    )
    if kernel.ndim != 2 or not np.isfinite(kernel).all() or float(kernel.sum()) <= 0:
        raise ValueError("psf_kernel must be a finite 2D array with positive sum")
    kernel = (kernel / float(kernel.sum())).astype(np.float32, copy=False)
    return ndimage.convolve(arr, kernel, mode=mode, cval=float(cval)).astype(np.float32, copy=False)


def sample_psf_parameters(
    *,
    seed: int | None = None,
    rng: np.random.Generator | None = None,
    sigma_range: tuple[float, float] = (0.15, 0.55),
    elliptical_probability: float = 0.30,
    airy_probability: float = 0.10,
    elliptical_ratio_range: tuple[float, float] = (0.80, 1.20),
    airy_ratio_range: tuple[float, float] = (0.85, 1.20),
) -> dict[str, float | str | None]:
    """Sample a PSF shape and LR-grid sigma parameters for domain randomization.

    ``elliptical_ratio_range`` scales each of the two elliptical-Gaussian axes off the base sigma
    (two independent draws); ``airy_ratio_range`` scales the airy y-axis. Both default to the
    historical hard-coded ranges, so the number and order of RNG draws — and hence a fixed-seed
    parameter sequence — are unchanged (golden-pinned)."""

    if rng is not None and seed is not None:
        raise ValueError("pass either rng or seed, not both")
    local_rng = rng if rng is not None else _rng(seed)
    low, high = float(sigma_range[0]), float(sigma_range[1])
    if low < 0 or high < low:
        raise ValueError("sigma_range must satisfy 0 <= low <= high")
    p_elliptical = float(elliptical_probability)
    p_airy = float(airy_probability)
    if p_elliptical < 0 or p_airy < 0 or p_elliptical + p_airy > 1:
        raise ValueError("PSF probabilities must be non-negative and sum to <= 1")
    sigma = float(local_rng.uniform(low, high))
    draw = float(local_rng.random())
    if draw < p_airy:
        return {
            "psf_shape": "airy_disk",
            "psf_sigma_lr_px": sigma,
            "psf_sigma_y_lr_px": sigma * float(local_rng.uniform(*airy_ratio_range)),
            "psf_angle_deg": float(local_rng.uniform(0.0, 180.0)),
        }
    if draw < p_airy + p_elliptical:
        return {
            "psf_shape": "elliptical_gaussian",
            "psf_sigma_lr_px": sigma * float(local_rng.uniform(*elliptical_ratio_range)),
            "psf_sigma_y_lr_px": sigma * float(local_rng.uniform(*elliptical_ratio_range)),
            "psf_angle_deg": float(local_rng.uniform(0.0, 180.0)),
        }
    return {
        "psf_shape": "gaussian",
        "psf_sigma_lr_px": sigma,
        "psf_sigma_y_lr_px": None,
        "psf_angle_deg": 0.0,
    }


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


def temporal_trend_drift(
    frames: np.ndarray,
    *,
    amplitude_c: float = 0.3,
    seed: int | None = None,
    spatial_sigma_px: float = 0.0,
) -> np.ndarray:
    """Add a slow linear thermal trend across the burst."""

    arr = _as_float_array(frames, "frames")
    if arr.ndim != 3:
        raise ValueError("frames must be 3D for drift models")
    amp = float(amplitude_c)
    if amp < 0:
        raise ValueError("amplitude_c must be >= 0")
    rng = _rng(seed)
    end_offset = float(rng.uniform(-amp, amp))
    trend = np.linspace(0.0, end_offset, arr.shape[0], dtype=np.float32)
    if float(spatial_sigma_px) <= 0:
        return (arr + trend[:, None, None]).astype(np.float32, copy=False)
    field = rng.normal(size=arr.shape[1:]).astype(np.float32)
    field = ndimage.gaussian_filter(field, sigma=float(spatial_sigma_px), mode="nearest")
    field = _normalize_noise(field, 1.0)
    modulation = 1.0 + 0.25 * field
    return (arr + trend[:, None, None] * modulation[None, :, :]).astype(np.float32, copy=False)


def apply_drift(
    frames: np.ndarray,
    *,
    model: DriftModel = "none",
    seed: int | None = None,
    amplitude_c: float = 0.2,
    lowfreq_sigma_px: float = 96.0,
    gain_sigma: float = 0.01,
    offset_sigma_c: float = 0.1,
    temporal_spatial_sigma_px: float = 0.0,
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
    if model == "temporal_trend":
        return temporal_trend_drift(
            frames,
            amplitude_c=amplitude_c,
            spatial_sigma_px=temporal_spatial_sigma_px,
            seed=seed,
        )
    raise ValueError("model must be one of: none, scalar_offset, lowfreq, gain_offset, temporal_trend")
