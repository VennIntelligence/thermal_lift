"""Pure flat-region noise statistics estimators, shared by the real-noise audit
(scripts/audit_real_noise.py) and the synthetic-noise pilot audit (scripts/audit_synth_noise.py)
and the v7 realism statistical tests (tcforge/tests/test_realism.py §5.2).

`autocorr_1e_length` and `radial_psd_slope` were LIFTED VERBATIM from audit_real_noise.py
(§5.0 sink) so the synthetic pool is measured with byte-for-byte the same estimator as the real
detector. `stripe_profiles` and `lag1_autocorr_median` replicate the audit main() column/row
stripe and lag-1 temporal-autocorrelation measurement so the same code path is reusable.

Units: everything is in the input array's own units (the real .txt frames are calibrated deg C).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage


def autocorr_1e_length(profile: np.ndarray, valid: np.ndarray) -> int | None:
    """1D autocorrelation of `profile` (NaN-masked by ~valid, mean-subtracted over valid entries,
    invalid entries zero-filled), first lag where it drops below 1/e. None if never / all-invalid."""
    if valid.sum() == 0:
        return None
    mu = float(np.mean(profile[valid]))
    full = np.where(valid, profile - mu, 0.0)
    n = len(full)
    ac = np.correlate(full, full, mode="full")[n - 1:]
    if ac[0] <= 0:
        return None
    ac = ac / ac[0]
    below = np.where(ac < 1.0 / np.e)[0]
    return int(below[0]) if len(below) else None


def radial_psd_slope(crop: np.ndarray, crop_size: int) -> dict:
    """Quadratic-detrend + Hann window + 2D FFT power spectrum, radially averaged (excluding the
    exact kx=0/ky=0 axes to avoid column/row-stripe contamination of the isotropic estimate),
    log-log slope fit -> alpha such that P(f) ~ f^-alpha."""
    yy, xx = np.mgrid[0:crop_size, 0:crop_size].astype(np.float64)
    xr, yr = xx.ravel(), yy.ravel()
    A = np.stack([np.ones_like(xr), xr, yr, xr * xr, yr * yr, xr * yr], axis=1)
    coef, *_ = np.linalg.lstsq(A, crop.ravel(), rcond=None)
    trend = (A @ coef).reshape(crop.shape)
    dcrop = crop - trend

    win = np.outer(np.hanning(crop_size), np.hanning(crop_size))
    windowed = dcrop * win
    F = np.fft.fftshift(np.fft.fft2(windowed))
    P = (np.abs(F) ** 2).astype(np.float64)

    c = crop_size // 2
    ky, kx = np.mgrid[-c:crop_size - c, -c:crop_size - c]
    kr = np.hypot(kx, ky)
    axis_mask = (kx == 0) | (ky == 0)
    valid_spec = ~axis_mask

    kmax = c
    bins = np.arange(1, kmax)
    radial = np.full(len(bins), np.nan)
    for i, kbin in enumerate(bins):
        sel = valid_spec & (kr >= kbin - 0.5) & (kr < kbin + 0.5)
        if sel.any():
            radial[i] = P[sel].mean()

    freq = bins / crop_size  # cycles / px
    fit_lo, fit_hi = 2, int(kmax * 0.6)
    fit_sel = (bins >= fit_lo) & (bins <= fit_hi) & np.isfinite(radial) & (radial > 0)
    logf = np.log10(freq[fit_sel])
    logp = np.log10(radial[fit_sel])
    slope, intercept = np.polyfit(logf, logp, 1)
    alpha = -float(slope)
    pred = slope * logf + intercept
    ss_res = float(np.sum((logp - pred) ** 2))
    ss_tot = float(np.sum((logp - logp.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return {
        "alpha": alpha, "intercept": float(intercept), "r2_loglog_fit": r2,
        "fit_freq_min_cyc_px": float(freq[fit_sel].min()),
        "fit_freq_max_cyc_px": float(freq[fit_sel].max()),
        "n_fit_bins": int(fit_sel.sum()),
        "freq": freq, "radial_psd": radial, "fit_sel": fit_sel,
    }


def stripe_profiles(mean_img: np.ndarray, bg_sigma_px: float = 25.0,
                    flat: np.ndarray | None = None) -> dict:
    """Column/row stripe FPN measurement, mirroring audit_real_noise.py items 1/2/6.

    profile = per-column / per-row mean (over flat pixels) of the residual
    `mean_img - gaussian_filter(mean_img, bg_sigma_px)`; amplitude = std of that profile over the
    valid entries; corr-length = autocorr 1/e length of the profile. `flat` (bool mask, same
    shape) restricts to structure-free pixels; None => all pixels are flat (synthetic flat scene)."""
    mean_img = np.asarray(mean_img, dtype=np.float64)
    if flat is None:
        flat = np.ones(mean_img.shape, dtype=bool)
    else:
        flat = np.asarray(flat, dtype=bool)
    smooth_bg = ndimage.gaussian_filter(mean_img, float(bg_sigma_px))
    bg_resid = mean_img - smooth_bg
    masked = np.where(flat, bg_resid, np.nan)
    with np.errstate(invalid="ignore"):
        col_profile = np.nanmean(masked, axis=0)
        row_profile = np.nanmean(masked, axis=1)
    col_valid = np.sum(flat, axis=0) > 0
    row_valid = np.sum(flat, axis=1) > 0
    col_amp = float(np.nanstd(col_profile[col_valid])) if col_valid.any() else float("nan")
    row_amp = float(np.nanstd(row_profile[row_valid])) if row_valid.any() else float("nan")
    return {
        "col_profile": col_profile, "row_profile": row_profile,
        "col_valid": col_valid, "row_valid": row_valid,
        "col_amp": col_amp, "row_amp": row_amp,
        "col_1e_length_px": autocorr_1e_length(col_profile, col_valid),
        "row_1e_length_px": autocorr_1e_length(row_profile, row_valid),
    }


def lag1_autocorr_median(burst: np.ndarray, flat: np.ndarray | None = None) -> float:
    """Median lag-1 temporal autocorrelation of the per-pixel residual (burst - temporal mean),
    mirroring audit_real_noise.py item 4. `burst` is (M, H, W) in acquisition order; `flat`
    (bool mask, same H,W) restricts to structure-free pixels; None => all pixels."""
    burst = np.asarray(burst, dtype=np.float64)
    if burst.ndim != 3:
        raise ValueError("burst must be (M, H, W)")
    mean_img = burst.mean(axis=0)
    resid = burst - mean_img[None]
    r0 = resid[:-1]
    r1 = resid[1:]
    r0m = r0 - r0.mean(axis=0, keepdims=True)
    r1m = r1 - r1.mean(axis=0, keepdims=True)
    num = (r0m * r1m).sum(axis=0)
    den = np.sqrt((r0m ** 2).sum(axis=0) * (r1m ** 2).sum(axis=0)) + 1e-12
    lag1 = num / den
    if flat is None:
        return float(np.median(lag1))
    return float(np.median(lag1[np.asarray(flat, dtype=bool)]))
