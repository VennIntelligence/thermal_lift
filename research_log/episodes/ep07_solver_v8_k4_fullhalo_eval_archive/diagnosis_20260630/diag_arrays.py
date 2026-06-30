"""EP07 solver V8/K4 offline render-array diagnosis (no GPU, no checkpoint needed).

Tests, using ONLY the saved full-frame render arrays:
  (A) Grid origin: is the visible grid locked to the TILE STEP (=> patch/stitch
      boundary artifact) rather than a fixed-frequency target/data artifact?
      Prediction: tiled_p192_o128 grid period = 64 HR px; dense_p192_o160 = 32 HR px;
      full_halo96 has NO seam peak; tile_halo* keep the 64 HR step but weaker.
  (B) Flocculent texture: broadband mid/high-frequency energy + flat-ROI texture std.
      Prediction: full_halo96 >> tiled in flat-region texture.
  (C) Line/edge sharpness: P95 gradient on structure pixels (tiled sharpest).
  (D) Low-frequency DC shift: does full_halo lift/renormalize the background
      (signature of an extent-dependent global op, not a neutral boundary fix)?
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import gaussian_filter

HR = 1  # arrays already in HR space
def _repo_root():
    from pathlib import Path as _P
    p = _P(__file__).resolve()
    for q in [p, *p.parents]:
        if (q / "AGENTS.md").exists():
            return q
    return p.parents[3]
ROOT = _repo_root()
OUT = ROOT / "outputs" / "ep07_solver_diag"; OUT.mkdir(parents=True, exist_ok=True)
def _find_npz():
    c = sorted(ROOT.glob("remote_inbox/**/v8k4_step10000_render_arrays.npz"))
    if not c:
        raise FileNotFoundError("v8k4_step10000_render_arrays.npz not found under remote_inbox/")
    return c[0]
NPZ = _find_npz()

z = np.load(NPZ)
KEYS = ["aligned_mean", "tiled_p192_o128", "dense_p192_o160",
        "full_halo96", "tile_halo32", "tile_halo64", "tile_halo96"]
R = {k: z[k].astype(np.float64) for k in KEYS}
H, W = R["aligned_mean"].shape

def highpass(a, sigma):
    return a - gaussian_filter(a, sigma, mode="nearest")

def axis_seam_spectrum(diff_hp):
    """Return (periods, Px, Py): power along fx (fy~0) and fy (fx~0) axes.
    Vertical seam lines (periodic in x) show up as peaks in Px at fx=k/step."""
    F = np.fft.fft2(diff_hp)
    P = (F * np.conj(F)).real
    # marginal along x: average a thin band of low vertical-frequency rows
    band = 3
    Px = (P[:band].sum(0) + P[-band+1:].sum(0)) / (2 * band - 1)
    Py = (P[:, :band].sum(1) + P[:, -band+1:].sum(1)) / (2 * band - 1)
    fx = np.fft.fftfreq(W)
    fy = np.fft.fftfreq(H)
    return fx, Px, fy, Py

def peak_period(freqs, power, pmin=6, pmax=220):
    """Find the dominant spatial period (HR px) within [pmin,pmax] from a 1D spectrum."""
    f = freqs.copy()
    pos = f > 0
    f = f[pos]; p = power[pos]
    per = 1.0 / f
    sel = (per >= pmin) & (per <= pmax)
    if not sel.any():
        return None, 0.0
    per_s = per[sel]; p_s = p[sel]
    i = int(np.argmax(p_s))
    # prominence vs local median background
    med = np.median(p_s)
    prom = p_s[i] / (med + 1e-12)
    return float(per_s[i]), float(prom)

def flattest_roi(base, size=96):
    """Locate the lowest-local-std square ROI in the baseline (a flat background patch)."""
    loc = gaussian_filter(base, 2, mode="nearest")
    var = gaussian_filter(base*base, 16, mode="nearest") - gaussian_filter(base, 16, mode="nearest")**2
    var = np.clip(var, 0, None)
    # restrict to interior, avoid borders
    m = size
    var[:m] = var[-m:] = var[:, :m] = var[:, -m:] = np.inf
    iy, ix = np.unravel_index(np.argmin(var), var.shape)
    y0 = max(0, iy - size//2); x0 = max(0, ix - size//2)
    return y0, x0, size

def sharpest_roi(base, size=192):
    g = gaussian_filter(np.abs(np.gradient(base)[0]) + np.abs(np.gradient(base)[1]), 8, mode="nearest")
    m = size
    g[:m] = g[-m:] = g[:, :m] = g[:, -m:] = -np.inf
    iy, ix = np.unravel_index(np.argmax(g), g.shape)
    y0 = max(0, iy - size//2); x0 = max(0, ix - size//2)
    return y0, x0, size

def grad_mag(a):
    gy, gx = np.gradient(a)
    return np.hypot(gx, gy)

base = R["aligned_mean"]
fy0, fx0, fsz = flattest_roi(base, 96)
sy0, sx0, ssz = sharpest_roi(base, 192)
print(f"Flat ROI  @ (y={fy0},x={fx0}) size {fsz}")
print(f"Sharp ROI @ (y={sy0},x={sx0}) size {ssz}")

# Mid/high band for flocculence: periods 3..16 HR px
fy_full = np.fft.fftfreq(H); fx_full = np.fft.fftfreq(W)
metrics = {}
print("\n%-18s %8s %8s | %7s %7s %7s | %8s %8s | %8s" % (
    "render","seamPx","prom","floROIσ","midPSD","P95grad","dMean","bgLift","seamPy"))
for k in KEYS:
    diff = R[k] - base
    diff_hp = highpass(diff, 12)            # isolate seams/texture the solver ADDED
    fxv, Px, fyv, Py = axis_seam_spectrum(diff_hp)
    perx, promx = peak_period(fxv, Px)
    pery, promy = peak_period(fyv, Py)

    # flat-ROI texture amplitude (flocculence proxy)
    self_hp = highpass(R[k], 4)
    flo = float(np.std(self_hp[fy0:fy0+fsz, fx0:fx0+fsz]))

    # mid/high broadband PSD (periods 3..16) via radial-ish band on self_hp
    F = np.fft.fft2(self_hp); P = (F*np.conj(F)).real / (H*W)
    FX, FY = np.meshgrid(fx_full, fy_full)
    rad = np.hypot(FX, FY)
    band = (rad >= 1/16) & (rad <= 1/3)
    midpsd = float(P[band].mean())

    # edge sharpness on structure ROI
    gm = grad_mag(R[k][sy0:sy0+ssz, sx0:sx0+ssz])
    p95 = float(np.percentile(gm, 95))

    # low-frequency background shift vs baseline
    dmean = float(np.mean(R[k] - base))
    lp = gaussian_filter(R[k]-base, 24, mode="nearest")
    bg = R[k] < np.percentile(R[k], 40)     # background pixels
    bglift = float(np.mean((R[k]-base)[bg]))

    metrics[k] = dict(seam_period_x=perx, seam_prom_x=promx,
                      seam_period_y=pery, seam_prom_y=promy,
                      flat_roi_texture_std=flo, mid_psd=midpsd,
                      edge_p95_grad=p95, mean_shift=dmean, bg_lift=bglift)
    print("%-18s %8s %8.1f | %7.4f %7.2e %7.4f | %8.4f %8.4f | %8s" % (
        k, f"{perx:.0f}" if perx else "-", promx,
        flo, midpsd, p95, dmean, bglift,
        f"{pery:.0f}" if pery else "-"))

(OUT / "metrics_arrays.json").write_text(json.dumps(metrics, indent=2))
np.savez(OUT / "rois.npz", flat=np.array([fy0,fx0,fsz]), sharp=np.array([sy0,sx0,ssz]))
print(f"\nWrote {OUT/'metrics_arrays.json'}")
