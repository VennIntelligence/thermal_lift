"""Fig 16 — Forward-operator self-check: band cutoff (FRC) + aliasing demo.

Context (ACL-023): after the detector-pitch recalibration to 20 um/px, the
physical forward operator (PSF blur -> 2x block downsample) was certified
against a synthetic self-check suite (T1-T5). This figure documents test
T5 (band-cutoff FRC, informational-only) and T3 (aliasing behaviour) on a
low-res chirp target.

(a) Split-half FRC of the calibrated forward operator on a synthetic
    phantom, plotted against period [um] on the HR (2x, 10 um/sample) grid.
    The frequency axis in band_cutoff_frc_freqs.npy runs in cycles/HR-pixel
    with 0.5 at the HR Nyquist frequency; period_um = 10.0 / freq (HR pixel
    pitch = hr_nyquist_period_um / 2 = 10.0 um). Vertical references mark
    the LR detector Nyquist period (2x20 um = 40 um) and the HR grid Nyquist
    period (20 um, matching the 20 um detector aperture zero). FRC never
    crosses the 0.5 half-bit-style reference within the resolved band in
    this synthetic scene (frc_crossed_0p5_within_band=False, cutoff=None);
    this is an informational self-test only, NOT the authoritative
    recoverable band (that comes from EP15 FRC on the real 248-frame data,
    see fig06_frc_leaderboard.py).

(b) A synthetic low-res horizontal chirp (linearly increasing spatial
    frequency, left to right) passed through the same calibrated forward
    operator. Column-wise contrast (std within a sliding window) decays
    monotonically as frequency increases -- the expected band-limiting
    signature of box-average LR sampling, not a folding/moire artifact.
    T3 quantifies this: aliased energy above 0.9 * Nyquist is a few percent
    of total LR energy (lr_aliased_energy_frac_above_0p9_nyq = 0.0291),
    consistent with "aliases by design, not a bug" per the self-check note.

Data: output/forward_selfcheck/{selfcheck_summary.json,
      band_cutoff_frc_freqs.npy, band_cutoff_frc_values.npy,
      aliasing_lr_chirp.npy}
Run:  uv run python docs/publication_figures/scripts/fig16_forward_selfcheck.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

SRC = REPO_ROOT / "output" / "forward_selfcheck"
with open(SRC / "selfcheck_summary.json") as fh:
    summary = json.load(fh)

freqs = np.load(SRC / "band_cutoff_frc_freqs.npy")     # cycles / HR-pixel, 0.5 = HR Nyquist
values = np.load(SRC / "band_cutoff_frc_values.npy")   # split-half FRC correlation
chirp = np.load(SRC / "aliasing_lr_chirp.npy")          # (120, 160) LR chirp through the operator

t5 = summary["T5_band_cutoff"]
t3 = summary["T3_aliasing"]

hr_nyq_period_um = t5["hr_nyquist_period_um"]      # 20.0
lr_nyq_period_um = t5["lr_nyquist_period_um"]       # 40.0
hr_pixel_pitch_um = hr_nyq_period_um / 2.0          # 10.0
period_um = hr_pixel_pitch_um / freqs

fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.0))

# ── (a) band-cutoff FRC vs period ──────────────────────────────────────
ax = axes[0]
ax.plot(period_um, values, color="#4C72B0", lw=1.4, label="Split-half FRC")
ax.axhline(0.5, color="#999999", ls=":", lw=1.0, label="0.5 reference")
ax.axvline(lr_nyq_period_um, color="#222222", ls="--", lw=0.9)
ax.axvline(hr_nyq_period_um, color="#666666", ls="--", lw=0.9)
ax.annotate(f"LR detector\nNyquist {lr_nyq_period_um:.0f} $\\mu$m",
            (lr_nyq_period_um * 1.06, 0.62), ha="right", va="center",
            fontsize=6.5, color="#222222")
ax.annotate(f"HR grid Nyquist /\ndetector aperture\nzero {hr_nyq_period_um:.0f} $\\mu$m",
            (hr_nyq_period_um * 1.10, 0.97), ha="left", va="top",
            fontsize=6.5, color="#555555")
ax.annotate(f"FRC={t5['frc_at_hr_nyquist']:.3f} at HR Nyquist\n"
            f"(never crosses 0.5 in-band;\ncutoff=None, informational only)",
            xy=(period_um[-1], values[-1]), xytext=(700, 0.18),
            fontsize=6.5, color="#333333", ha="left",
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.7))
ax.set_xscale("log")
ax.set_xlim(period_um.max() * 1.05, period_um.min() * 0.95)
ticks = [3000, 1000, 300, 100, 40, 20]
ax.set_xticks(ticks)
ax.set_xticklabels([str(t) for t in ticks])
ax.set_xlabel("Period [$\\mu$m]")
ax.set_ylabel("Split-half FRC")
ax.set_ylim(-0.02, 1.05)
ax.set_title(f"(a) Band-cutoff FRC ($\\sigma_{{PSF}}$={t5['calibrated_psf_sigma_lr_px']:.4f} LR-px)",
             loc="left", fontsize=9)
ax.legend(loc="lower left", fontsize=6.5)

# ── (b) aliasing chirp demo ─────────────────────────────────────────────
ax = axes[1]
im = ax.imshow(chirp, cmap="inferno", aspect="auto",
                extent=[0, chirp.shape[1], chirp.shape[0], 0])
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("LR intensity [a.u.]", fontsize=8)
ax.set_xlabel("LR column (increasing spatial frequency $\\rightarrow$)")
ax.set_ylabel("LR row")
ax.annotate("full contrast\n(low freq)", (8, 12), ha="left", va="top",
            fontsize=6.5, color="white",
            bbox=dict(boxstyle="round,pad=0.15", fc="#00000090", ec="none"))
ax.annotate("contrast rolls off\n(high freq, band-limited)", (155, 12),
            ha="right", va="top", fontsize=6.5, color="white",
            bbox=dict(boxstyle="round,pad=0.15", fc="#00000090", ec="none"))
ax.annotate(f"aliased energy $>0.9\\,f_{{Nyq}}$: {t3['lr_aliased_energy_frac_above_0p9_nyq']*100:.1f}% "
            f"of LR energy\n(few-% is physical box-sampling, not folding)",
            (0.5, -0.28), xycoords="axes fraction", ha="center", va="top", fontsize=6.5)
ax.set_title("(b) LR chirp through calibrated forward operator", loc="left", fontsize=9)

paths = save_fig(fig, "fig16_forward_selfcheck")
print("\n".join(str(p) for p in paths))
