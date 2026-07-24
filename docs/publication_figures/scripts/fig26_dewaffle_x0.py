#!/usr/bin/env python
"""fig26 — De-waffle warm start: drizzle's checkerboard coverage artifact and its removal from x0.

ACL-032 (research_log/algorithm_changelog.md, 2026-06-29): the hybrid solver warm-starts
x0 from obs ch5 = phase_bin_drizzle[0]. Phase-bin drizzle routes each burst frame to one of
four sub-pixel-phase bins and drizzles each bin separately onto the 2x HR grid; on flat
background with uneven phase coverage the 2x2 HR sub-positions in each block are filled from
different frame subsets, producing a 2-HR-px-period (= 1 detector pitch = 20 um) checkerboard
("waffle") that survives into the solver output (the DC step is band-limited and flat regions
have no data anchor). The de-waffle fix is NOT a filter: it swaps the warm-start source to
obs ch0 = fused aligned_mean (bilinear-upsampled 2x), which is smooth, while keeping all nine
conditioning channels (--solver-warmstart aligned_mean). That swap is reimplemented here
exactly (ch5 -> ch0 + scipy zoom order=1), matching scripts/diagnose_drizzle_waffle.py.

Panels (top row: sigma=2 Gaussian-highpass flat-background crops, bottom row: Hann-windowed
log power spectra of a 161-px flat window with the waffle peaks at the half-sampling
frequency |f| = 0.5 cyc/HR-px marked):
  (a)/(d) real drizzle composite remote_inbox/20260716_v8_verdict/drizzle_a.npy at the
          fig10 flat ROI (y0=791, x0=869) — the faint waffle on real data;
  (b)/(e) synthetic current warm start x0 = phase_bin_drizzle[0] (ch5),
          outputs/data_gen_v6_cpu_preview/pool_sample/scene_0003, flattest-GT-window ROI
          (y0=265, x0=106) — the waffle at full strength;
  (c)/(f) the de-waffled warm start x0 = upsampled aligned_mean (ch0) of the SAME scene and
          ROI — Nyquist peaks gone. De-waffle cannot be shown on the real composite alone
          (drizzle_a.npy has no aligned_mean channel), hence the synthetic before/after pair.
grid_score = fraction of spectral power with max(|fy|,|fx|) >= 0.45 (Nyquist shell), same
definition as scripts/diagnose_drizzle_waffle.py.

Run from repo root:  uv run python docs/publication_figures/scripts/fig26_dewaffle_x0.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, zoom as ndi_zoom

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pubfig_style import (  # noqa: E402
    CMAP_COVERAGE,
    CMAP_RESID_DIV,
    REPO_ROOT,
    W_DOUBLE,
    save_fig,
    setup_academic_style,
)

# ── Data locations / ROIs ────────────────────────────────────────────
REAL_DRIZZLE = REPO_ROOT / "remote_inbox/20260716_v8_verdict/drizzle_a.npy"
SYNTH_SCENE = REPO_ROOT / "outputs/data_gen_v6_cpu_preview/pool_sample/scene_0003"
REAL_ROI = (791, 869)     # fig10 flat ROI (y0, x0) on the 960x1280 2x grid
SYNTH_ROI = (265, 106)    # flattest 161-px GT window of scene_0003 (diagnose-script method)
CROP = 64                 # displayed zoom crop [HR px]
FFT_WIN = 161             # odd -> spectrum symmetric to ~±0.497 cyc/px, Nyquist peaks on all edges
HP_SIGMA = 2.0            # light Gaussian highpass to expose the faint waffle
UM_PER_HRPX = 10.0        # 2x grid on a 20-um detector pitch


def highpass(img: np.ndarray, sigma: float = HP_SIGMA) -> np.ndarray:
    img = np.asarray(img, np.float64)
    return img - gaussian_filter(img, sigma)


def log_power(img: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Hann-windowed log10 power spectrum (fftshifted) and its frequency extent."""
    arr = np.asarray(img, np.float64)
    arr = arr - arr.mean()
    h, w = arr.shape
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    p = np.abs(np.fft.fftshift(np.fft.fft2(arr * win))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(h))
    fx = np.fft.fftshift(np.fft.fftfreq(w))
    logp = np.log10(p + 1e-12 * p.max())
    return logp, float(fx.max()), float(fy.max())


def grid_score(img: np.ndarray) -> float:
    """Nyquist-shell power fraction, identical to scripts/diagnose_drizzle_waffle.py."""
    arr = np.asarray(img, np.float64)
    arr = arr - arr.mean()
    h, w = arr.shape
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    p = np.abs(np.fft.fftshift(np.fft.fft2(arr * win))) ** 2
    fy = np.abs(np.fft.fftshift(np.fft.fftfreq(h)))[:, None]
    fx = np.abs(np.fft.fftshift(np.fft.fftfreq(w)))[None, :]
    total = float(p.sum())
    return float(p[np.maximum(fy, fx) >= 0.45].sum() / total) if total > 0 else 0.0


def main() -> int:
    setup_academic_style()
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Manual-layout script: constrained layout leaves the aspect-locked
    # imshow panels floating in oversized cells, and the tight-bbox save
    # would re-enable it and re-lay out the 2nd (pdf) save.
    mpl.rcParams["figure.constrained_layout.use"] = False

    real = np.load(REAL_DRIZZLE).astype(np.float32)
    pbd = np.load(SYNTH_SCENE / "phase_bin_drizzle_2x.npy").astype(np.float32)
    obs = np.load(SYNTH_SCENE / "obs_features_1x.npz")["obs_features"].astype(np.float32)
    aligned2x = ndi_zoom(obs[0], (2, 2), order=1)  # the de-waffled x0 (ch0, ACL-032)

    fields = [
        ("Real drizzle composite\n(warm start carries this)", real, REAL_ROI),
        ("Synthetic $x_0$ = phase-bin\ndrizzle (ch5, current)", pbd[0], SYNTH_ROI),
        ("Synthetic $x_0$ = aligned mean\n(ch0, de-waffled)", aligned2x, SYNTH_ROI),
    ]

    # Inch-based geometry: square panels + slim dedicated colorbar axes so
    # panel/colorbar positions are exact and no engine redistributes gaps.
    LEFT, RIGHT = 0.50, 0.0                    # bottom-row ylabel+ticks / edge [in]
    CB_PAD, CB_W, CB_TXT = 0.04, 0.08, 0.54    # colorbar pad / width / labels [in]
    COL_GAP = 0.24                             # holds bottom-row ytick labels [in]
    TOP, ROW_GAP, BOTTOM = 0.36, 0.16, 0.50    # titles / row gap / xlabels [in]
    SAVE_PAD = 0.06   # savefig pad_inches=0.03 on both sides of the tight bbox
    P = (W_DOUBLE - SAVE_PAD - LEFT - RIGHT
         - 3 * (CB_PAD + CB_W + CB_TXT) - 2 * COL_GAP) / 3
    COL_PITCH = P + CB_PAD + CB_W + CB_TXT + COL_GAP
    FIG_W = W_DOUBLE
    FIG_H = TOP + 2 * P + ROW_GAP + BOTTOM

    fig = plt.figure(figsize=(FIG_W, FIG_H))
    y_row = [BOTTOM + P + ROW_GAP, BOTTOM]     # row 0 = top, row 1 = bottom [in]
    axes = [[None] * 3 for _ in range(2)]
    caxes = [[None] * 3 for _ in range(2)]
    for i in range(2):
        for j in range(3):
            x0 = LEFT + j * COL_PITCH
            axes[i][j] = fig.add_axes(
                [x0 / FIG_W, y_row[i] / FIG_H, P / FIG_W, P / FIG_H])
            caxes[i][j] = fig.add_axes(
                [(x0 + P + CB_PAD) / FIG_W, y_row[i] / FIG_H,
                 CB_W / FIG_W, P / FIG_H])
    letters = "abcdef"

    for j, (title, img, (y0, x0)) in enumerate(fields):
        win = img[y0:y0 + FFT_WIN, x0:x0 + FFT_WIN]
        hp = highpass(img)[y0:y0 + CROP, x0:x0 + CROP]

        # ── top: highpass zoom crop ──
        ax = axes[0][j]
        vmax = float(np.percentile(np.abs(hp), 99))
        im = ax.imshow(hp, cmap=CMAP_RESID_DIV, vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.6)
        cb = fig.colorbar(im, cax=caxes[0][j])
        cb.set_label("Highpass residual [$^\\circ$C]", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        # scale bar: 10 HR px = 100 um = 5 waffle periods
        bar_px = 10
        ax.plot([3, 3 + bar_px], [CROP - 5, CROP - 5], color="black", lw=2, solid_capstyle="butt")
        ax.text(3 + bar_px / 2, CROP - 8, f"{bar_px * UM_PER_HRPX:.0f} $\\mu$m",
                ha="center", va="bottom", fontsize=7,
                bbox=dict(facecolor="white", alpha=0.75, pad=0.8, edgecolor="none"))
        ax.text(0.02, 0.98, f"({letters[j]})", transform=ax.transAxes,
                ha="left", va="top", fontsize=9, fontweight="bold",
                bbox=dict(facecolor="white", alpha=0.75, pad=0.8, edgecolor="none"))

        # ── bottom: log power spectrum ──
        ax = axes[1][j]
        logp, fxm, fym = log_power(win)
        lo, hi = np.percentile(logp, [5, 99.9])
        im = ax.imshow(logp, cmap=CMAP_COVERAGE, vmin=lo, vmax=hi,
                       extent=[-fxm, fxm, fym, -fym], interpolation="nearest")
        # waffle peaks: half-sampling frequency |f| = 0.5 cyc/HR-px (axes + corners)
        pk = fxm  # ~0.497 for the odd window = the outermost sampled frequency
        pts = [(pk, 0), (-pk, 0), (0, pk), (0, -pk), (pk, pk), (pk, -pk), (-pk, pk), (-pk, -pk)]
        for (px, py) in pts:
            ax.plot(px, py, marker="o", mfc="none", mec="#C44E52", mew=1.0, ms=7)
        ax.set_xlim(-0.56, 0.56); ax.set_ylim(0.56, -0.56)
        gs = grid_score(win)
        ax.set_xlabel(f"$f_x$ [cyc / HR px]\ngrid score = {gs:.3f}")
        if j == 0:
            ax.set_ylabel("$f_y$ [cyc / HR px]")
        ax.set_xticks([-0.5, 0, 0.5]); ax.set_yticks([-0.5, 0, 0.5])
        for s in ax.spines.values():
            s.set_visible(True); s.set_linewidth(0.6)
        cb = fig.colorbar(im, cax=caxes[1][j])
        cb.set_label("log$_{10}$ power [a.u.]", fontsize=8)
        cb.ax.tick_params(labelsize=7)
        ax.text(-0.02, 1.06, f"({letters[3 + j]})", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, fontweight="bold")

    paths = save_fig(fig, "fig26_dewaffle_x0")
    for p in paths:
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
