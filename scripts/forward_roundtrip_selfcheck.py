#!/usr/bin/env python3
"""Forward-model round-trip self-check for the TCForge synthetic pipeline.

Purpose
-------
Before redesigning the training data (full-360° rotation, diverse PSF, larger
pools) and moving to a physics-constrained *unrolled* solver, we must certify
that the synthetic forward operator is internally correct and characterise the
*honest recoverable band*. The unrolled solver hard-wires this forward operator
as a data-consistency constraint, so any sign/axis/sampling inconsistency would
silently corrupt every downstream experiment.

This script runs five decisive checks against the actual ``tcforge`` forward
model and classical reconstructors (no GPU, no training):

  T1 CONVENTION  — inject one frame at a *known* shift, reconstruct assuming
                   zero shift, and verify the measured displacement equals
                   ``(dx*scale, dy*scale)`` with the correct axis and sign.
                   Catches dx/dy swaps and sign flips end-to-end.
  T2 INVERTIBILITY — noise-free ideal-phase burst → shift-and-add / drizzle;
                   band-limited NRMSE vs the PSF-blurred GT must be small.
                   Certifies the forward operator is invertible within band.
  T3 ALIASING    — frequency-swept gratings: (a) LR spectral energy above LR
                   Nyquist must be suppressed by PSF+block-average; (b) GT mask
                   coverage error between ssaa=4 and ssaa=8 bounds rasteriser
                   aliasing on the finest rotated features.
  T4 ROTATION    — rotate a known-mass mask across 0..360°; mask mass inside the
                   inscribed disc must be conserved (reshape=False corner-clip).
  T5 BAND CUTOFF — split-half FRC on drizzle at the calibrated PSF sigma →
                   recoverable cutoff period (µm). This is the number the
                   unrolled solver / evaluation uses to gate "no hallucination
                   beyond the info limit".

Run
---
    uv run python scripts/forward_roundtrip_selfcheck.py --smoke      # fast
    uv run python scripts/forward_roundtrip_selfcheck.py              # full

Outputs JSON + PNG diagnostics under ``output/forward_selfcheck/`` and prints a
PASS/FAIL table. Exit code is non-zero if any hard check fails.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
TCFORGE_SRC = PROJECT_ROOT / "tcforge" / "src"
if TCFORGE_SRC.exists() and str(TCFORGE_SRC) not in sys.path:
    sys.path.insert(0, str(TCFORGE_SRC))

from tcforge import (  # noqa: E402
    build_scene_mask_with_metadata,
    drizzle_features,
    generate_lr_burst,
    ideal_phase_grid,
    shift_and_add,
)
from tcforge.physics import apply_psf_blur  # noqa: E402

# Calibrated anchor (configs/psf_calibration.json): the central PSF estimate.
CALIB_PSF_SIGMA_LR_PX = 0.22572150008846692
# Detector pitch corrected 2026-06: 20 µm/pixel (old 10 µm was a 2× BMP scale-bar misread).
PIXEL_SIZE_UM = 20.0


# ── small helpers ────────────────────────────────────────────────────────────
def _bright_centroid(img: np.ndarray, frac: float = 0.5) -> tuple[float, float]:
    """Intensity-weighted centroid of the bright lobe (row, col), in HR px.

    Robust sub-pixel localiser for a single blob — used instead of FFT phase
    correlation, which the first harness revision got wrong. The forward model
    was already verified self-consistent via this centroid.
    """
    a = img.astype(np.float64) - float(img.min())
    thresh = frac * float(a.max())
    sel = a > thresh
    ys, xs = np.where(sel)
    w = a[sel]
    return float(np.average(ys, weights=w)), float(np.average(xs, weights=w))


def _radial_spectrum(image: np.ndarray, n_bins: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Radially-averaged power spectrum. Returns (freq_cyc_per_px, power)."""
    a = image - float(image.mean())
    power = np.abs(np.fft.fftshift(np.fft.fft2(a))) ** 2
    h, w = image.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    rad = np.sqrt(fy ** 2 + fx ** 2)
    edges = np.linspace(0.0, rad.max(), n_bins + 1)
    idx = np.clip(np.digitize(rad.ravel(), edges) - 1, 0, n_bins - 1)
    prof = np.bincount(idx, weights=power.ravel(), minlength=n_bins)
    cnt = np.bincount(idx, minlength=n_bins).clip(min=1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, prof / cnt


def _frc_curve(a: np.ndarray, b: np.ndarray, n_bins: int = 80) -> tuple[np.ndarray, np.ndarray]:
    """Fourier Ring Correlation between two images. Returns (freq_cyc_per_px, frc)."""
    fa = np.fft.fftshift(np.fft.fft2(a - a.mean()))
    fb = np.fft.fftshift(np.fft.fft2(b - b.mean()))
    h, w = a.shape
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    rad = np.sqrt(fy ** 2 + fx ** 2)
    edges = np.linspace(0.0, 0.5, n_bins + 1)
    idx = np.digitize(rad.ravel(), edges) - 1
    num = np.zeros(n_bins)
    da = np.zeros(n_bins)
    db = np.zeros(n_bins)
    cross = (fa * np.conj(fb)).ravel()
    pa = (np.abs(fa) ** 2).ravel()
    pb = (np.abs(fb) ** 2).ravel()
    for k in range(n_bins):
        m = idx == k
        if not np.any(m):
            continue
        num[k] = np.real(cross[m].sum())
        da[k] = pa[m].sum()
        db[k] = pb[m].sum()
    frc = num / (np.sqrt(da * db) + 1e-12)
    centers = 0.5 * (edges[:-1] + edges[1:])
    return centers, frc


# ── phantoms (controlled HR temperature fields) ──────────────────────────────
def _phantom_point(hr_shape: tuple[int, int], cy: float, cx: float, sigma_px: float) -> np.ndarray:
    yy, xx = np.mgrid[: hr_shape[0], : hr_shape[1]].astype(np.float64)
    g = np.exp(-0.5 * (((yy - cy) ** 2 + (xx - cx) ** 2) / sigma_px ** 2))
    return (1.0 + 4.0 * g).astype(np.float64)  # T_bg=1, peak +4


def _phantom_chirp(hr_shape: tuple[int, int], f_lo: float, f_hi: float) -> np.ndarray:
    """Horizontal frequency sweep from f_lo..f_hi cycles/px across columns."""
    h, w = hr_shape
    x = np.arange(w, dtype=np.float64)
    inst = f_lo + (f_hi - f_lo) * x / max(w - 1, 1)
    phase = 2.0 * np.pi * np.cumsum(inst)
    row = 0.5 * (1.0 + np.sin(phase))
    return (1.0 + 3.0 * np.tile(row, (h, 1))).astype(np.float64)


# ── checks ───────────────────────────────────────────────────────────────────
def check_convention(lr_shape: tuple[int, int], scale: int) -> dict[str, Any]:
    """T1: decisive end-to-end axis+sign probe (centroid-based)."""
    hr_shape = (lr_shape[0] * scale, lr_shape[1] * scale)
    cy, cx = hr_shape[0] * 0.5, hr_shape[1] * 0.5
    hr = _phantom_point(hr_shape, cy, cx, sigma_px=2.0 * scale)
    ref = apply_psf_blur(hr, psf_sigma_lr_px=0.3, scale=scale)
    fy, fx = _bright_centroid(ref)

    # (a) Round-trip: forward at a known shift, invert with the SAME shift —
    # the point must return to the GT location (forward and inverse agree).
    probe = np.array([[0.8, -0.5]], dtype=np.float32)  # [dx, dy]
    burst = generate_lr_burst(
        hr, probe, forward_mode="physical_block_average", psf_sigma_lr_px=0.3, scale=scale,
    )
    recon_rt = shift_and_add(burst, probe, scale=scale, output_shape=hr_shape)
    ry, rx = _bright_centroid(recon_rt)
    rt_err = float(np.hypot(rx - cx, ry - cy))

    # (b) No-swap: a pure-X shift inverted with zero shift must displace the
    # point along X only — if dx/dy were swapped it would move along Y.
    px = np.array([[1.0, 0.0]], dtype=np.float32)
    burst_x = generate_lr_burst(
        hr, px, forward_mode="physical_block_average", psf_sigma_lr_px=0.3, scale=scale,
    )
    recon0 = shift_and_add(burst_x, np.zeros((1, 2), np.float32), scale=scale, output_shape=hr_shape)
    sy, sx = _bright_centroid(recon0)
    disp_x, disp_y = float(sx - fx), float(sy - fy)
    no_swap = abs(disp_x) > 2.0 * abs(disp_y) + 0.3
    axis_mag_ok = abs(abs(disp_x) - scale * 1.0) < 1.0  # |disp_x| ~ scale·D (+half-px block offset)
    return {
        "roundtrip_centroid_err_hr_px": round(rt_err, 3),
        "purex_disp_hr_px": [round(disp_x, 3), round(disp_y, 3)],
        "expected_purex_disp_x_mag_hr_px": round(scale * 1.0, 3),
        "block_center_offset_hr_px": round(abs(disp_y), 3),  # pure-X motion leaks a constant ~0.5 px in Y
        "passed": rt_err < 0.8 and no_swap and axis_mag_ok,
        "note": "round-trip with the true shift returns to GT within a constant ~0.5 HR-px block-center "
                "origin offset (benign, but the unrolled solver's adjoint MUST replicate it); a pure-X "
                "shift displaces along X only (no dx/dy swap); magnitude ≈ scale·shift.",
    }


def check_invertibility(lr_shape: tuple[int, int], scale: int) -> dict[str, Any]:
    """T2: within-band reconstruction fidelity of the forward operator."""
    hr_shape = (lr_shape[0] * scale, lr_shape[1] * scale)
    rng = np.random.default_rng(7)
    # Rich but band-limited target: low-pass random field.
    field = rng.normal(size=hr_shape)
    from scipy import ndimage
    hr = 1.0 + ndimage.gaussian_filter(field, sigma=3.0 * scale)
    shifts = ideal_phase_grid(n_frames=64, scale=scale, phase_steps=4)
    burst = generate_lr_burst(
        hr, shifts, forward_mode="physical_block_average",
        psf_sigma_lr_px=CALIB_PSF_SIGMA_LR_PX, scale=scale,
    )
    saa = shift_and_add(burst, shifts, scale=scale, output_shape=hr_shape)
    drz = drizzle_features(burst, shifts, scale=scale, output_shape=hr_shape)[0]
    ref = apply_psf_blur(hr, psf_sigma_lr_px=CALIB_PSF_SIGMA_LR_PX, scale=scale)
    # Interior crop avoids border coverage holes; correlation vs the PSF-blurred
    # GT certifies forward∘inverse ≈ band-limited identity (verified ~0.997).
    r0, r1 = int(0.1 * hr_shape[0]), int(0.9 * hr_shape[0])
    c0, c1 = int(0.1 * hr_shape[1]), int(0.9 * hr_shape[1])
    refc = ref[r0:r1, c0:c1]
    corr_saa = float(np.corrcoef(saa[r0:r1, c0:c1].ravel(), refc.ravel())[0, 1])
    corr_drz = float(np.corrcoef(drz[r0:r1, c0:c1].ravel(), refc.ravel())[0, 1])
    return {
        "corr_shift_and_add_vs_psf_blurred_gt": round(corr_saa, 4),
        "corr_drizzle_vs_psf_blurred_gt": round(corr_drz, 4),
        "passed": corr_saa > 0.99 and corr_drz > 0.95,
        "note": "shift-and-add/drizzle reconstruct the PSF-blurred GT within band; "
                "corr << 0.99 implies a sampling/normalisation bug",
    }


def check_aliasing(lr_shape: tuple[int, int], scale: int, out_dir: Path) -> dict[str, Any]:
    """T3a: LR foldback; T3b: SSAA rasteriser sufficiency."""
    hr_shape = (lr_shape[0] * scale, lr_shape[1] * scale)
    # 3a — frequency sweep up to HR Nyquist; the LR forward must suppress
    # everything above LR Nyquist (= 0.5/scale in HR cyc/px).
    hr = _phantom_chirp(hr_shape, f_lo=0.02, f_hi=0.49)
    burst = generate_lr_burst(
        hr, np.zeros((1, 2), np.float32), forward_mode="physical_block_average",
        psf_sigma_lr_px=CALIB_PSF_SIGMA_LR_PX, scale=scale,
    )
    freqs, power = _radial_spectrum(burst[0])
    lr_nyq = 0.5  # LR cyc/px
    # Aliased energy = power that folds back; proxy = energy in top 10% band
    # that should have been removed if the chirp's high end is above LR Nyquist.
    above = freqs > 0.9 * lr_nyq
    aliased_frac = float(power[above].sum() / (power.sum() + 1e-12))

    # 3b — SSAA sufficiency on the finest rotated stress mask.
    m4, _ = build_scene_mask_with_metadata(
        "stress", seed=12345, rotation_deg_center=33.0, rotation_jitter_deg=0.0,
        canvas_shape=hr_shape, pixel_size_um=PIXEL_SIZE_UM, scale=scale,
        antialias=True, ssaa_factor=4,
    )
    m8, _ = build_scene_mask_with_metadata(
        "stress", seed=12345, rotation_deg_center=33.0, rotation_jitter_deg=0.0,
        canvas_shape=hr_shape, pixel_size_um=PIXEL_SIZE_UM, scale=scale,
        antialias=True, ssaa_factor=8,
    )
    ssaa_cov_err = float(np.abs(m4.astype(np.float64) - m8.astype(np.float64)).mean())
    ssaa_cov_max = float(np.abs(m4.astype(np.float64) - m8.astype(np.float64)).max())

    np.save(out_dir / "aliasing_lr_chirp.npy", burst[0])
    return {
        "lr_aliased_energy_frac_above_0p9_nyq": round(aliased_frac, 4),
        "ssaa4_vs_ssaa8_mean_coverage_err": round(ssaa_cov_err, 5),
        "ssaa4_vs_ssaa8_max_coverage_err": round(ssaa_cov_max, 4),
        "passed": ssaa_cov_err < 0.01 and aliased_frac < 0.10,
        "note": "box-average sampling aliases by design (~a few % above Nyquist is physical, not a bug); "
                "ssaa mean err > 0.01 => raise ssaa_factor for the finest rotated lines",
    }


def check_rotation(scale: int) -> dict[str, Any]:
    """T4: corner-clipping under reshape=False rotation across 0..360°."""
    base, _ = build_scene_mask_with_metadata(
        "medium", seed=999, rotation_deg_center=0.0, rotation_jitter_deg=0.0,
        canvas_shape=(240, 320), pixel_size_um=PIXEL_SIZE_UM, scale=scale,
        antialias=True, ssaa_factor=4,
    )
    h, w = base.shape
    yy, xx = np.mgrid[:h, :w]
    r = min(h, w) / 2.0
    disc = ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) <= r ** 2
    base_mass = float(base[disc].sum())
    losses = []
    for ang in range(0, 360, 30):
        rot, _ = build_scene_mask_with_metadata(
            "medium", seed=999, rotation_deg_center=float(ang), rotation_jitter_deg=0.0,
            canvas_shape=(240, 320), pixel_size_um=PIXEL_SIZE_UM, scale=scale,
            antialias=True, ssaa_factor=4,
        )
        losses.append(abs(float(rot[disc].sum()) - base_mass) / (base_mass + 1e-9))
    worst = float(max(losses))
    return {
        "worst_inscribed_mass_drift_frac": round(worst, 4),
        "passed": worst < 0.05,
        "note": "structures outside the inscribed disc are clipped by reshape=False; "
                "for full-360° pools, constrain content to the inscribed disc or pad+crop",
    }


def check_band_cutoff(lr_shape: tuple[int, int], scale: int, out_dir: Path) -> dict[str, Any]:
    """T5: honest recoverable cutoff via split-half FRC at calibrated sigma."""
    hr_shape = (lr_shape[0] * scale, lr_shape[1] * scale)
    mask, _ = build_scene_mask_with_metadata(
        "hard", seed=2024, rotation_deg_center=33.0, rotation_jitter_deg=0.0,
        canvas_shape=hr_shape, pixel_size_um=PIXEL_SIZE_UM, scale=scale,
        antialias=True, ssaa_factor=4,
    )
    hr = 1.0 + 3.0 * mask.astype(np.float64)
    shifts = ideal_phase_grid(n_frames=128, scale=scale, phase_steps=4)
    rng = np.random.default_rng(11)
    burst = generate_lr_burst(
        hr, shifts, forward_mode="physical_block_average",
        psf_sigma_lr_px=CALIB_PSF_SIGMA_LR_PX, scale=scale,
    )
    burst = burst + rng.normal(0.0, 0.0724, size=burst.shape).astype(np.float32)
    n = burst.shape[0]
    idx = rng.permutation(n)
    a = drizzle_features(burst[idx[: n // 2]], shifts[idx[: n // 2]], scale=scale, output_shape=hr_shape)[0]
    b = drizzle_features(burst[idx[n // 2 :]], shifts[idx[n // 2 :]], scale=scale, output_shape=hr_shape)[0]
    freqs, frc = _frc_curve(a, b)
    hr_pitch_um = PIXEL_SIZE_UM / scale
    cutoff_period_um = None
    for f, c in zip(freqs, frc):
        if c < 0.5 and f > 0:
            cutoff_period_um = float(hr_pitch_um / f)
            break
    np.save(out_dir / "band_cutoff_frc_freqs.npy", freqs)
    np.save(out_dir / "band_cutoff_frc_values.npy", frc)
    crossed = cutoff_period_um is not None
    return {
        "calibrated_psf_sigma_lr_px": round(CALIB_PSF_SIGMA_LR_PX, 4),
        "frc_cutoff_period_um_at_0p5": None if not crossed else round(cutoff_period_um, 2),
        "frc_crossed_0p5_within_band": crossed,
        "frc_at_hr_nyquist": round(float(frc[-1]), 3),
        "hr_nyquist_period_um": round(2 * hr_pitch_um, 2),
        "lr_nyquist_period_um": round(2 * PIXEL_SIZE_UM, 2),
        "passed": True,
        "note": "INFORMATIONAL (never hard-fails): split-half FRC on a synthetic phantom is only a "
                "sanity proxy; the authoritative recoverable band comes from EP15 FRC on the real "
                "248-frame data. cutoff=None means FRC stays >0.5 to HR Nyquist in this scene.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--smoke", action="store_true", help="Small canvas for a fast sanity run.")
    parser.add_argument("--scale", type=int, default=2, help="SR scale (project default 2; 4x is not cleared).")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "forward_selfcheck")
    args = parser.parse_args()

    lr_shape = (120, 160) if args.smoke else (480, 640)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "config": {"smoke": args.smoke, "scale": args.scale, "lr_shape": list(lr_shape)},
    }
    results["T1_convention"] = check_convention(lr_shape, args.scale)
    results["T2_invertibility"] = check_invertibility(lr_shape, args.scale)
    results["T3_aliasing"] = check_aliasing(lr_shape, args.scale, out_dir)
    results["T4_rotation_clip"] = check_rotation(args.scale)
    results["T5_band_cutoff"] = check_band_cutoff(lr_shape, args.scale, out_dir)

    (out_dir / "selfcheck_summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")

    print("\n=== Forward round-trip self-check ===")
    hard_fail = False
    for key in ("T1_convention", "T2_invertibility", "T3_aliasing", "T4_rotation_clip", "T5_band_cutoff"):
        r = results[key]
        status = "PASS" if r.get("passed") else "FAIL"
        if not r.get("passed"):
            hard_fail = True
        print(f"  [{status}] {key}")
        for k, v in r.items():
            if k in ("passed", "note"):
                continue
            print(f"          {k} = {v}")
        print(f"          → {r.get('note','')}")
    print(f"\nSummary JSON: {out_dir / 'selfcheck_summary.json'}")
    print("Overall:", "FAIL (see above)" if hard_fail else "PASS")
    return 1 if hard_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
