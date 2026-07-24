"""Fig 66 — Where the FRC gain lives: band decomposition of a structured
region.

The fig10 "center detail" ROI decomposed by FFT annulus into (a) the
temperature image, (b) the genuine-gain acceptance band (25-40 um period —
the band where neural arms genuinely add transfer over drizzle, ACL-071/072
band analysis), and (c) the sub-acceptance band (20-25 um, above the
detector-aperture zero at 20 um but below the authoritative recoverable
cutoff 25.45 +/- 0.73 um — content here is never claimed as recovered).
Drizzle's 25-40 um content is visibly attenuated relative to TGV/v6
(band energy, not per-se validated transfer); notably TGV also carries
substantial 20-25 um energy (rms 0.070) that is below the claims cutoff
and therefore never validated, while v9-3k is nearly clean there.

Per-column shared symmetric color scales (temperature column median-
aligned per arm).

Data: remote_inbox/20260710_expab (seed-42 a-halves). ACL-071/072 band
metrology; recoverable-band verdict ACL-049/059.
Run:  uv run python docs/publication_figures/scripts/fig66_band_decomposition.py
"""

import matplotlib.pyplot as plt
import numpy as np

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

SRC = REPO_ROOT / "remote_inbox/20260710_expab"
ARMS = [
    ("Drizzle", "drizzle_a.npy"),
    ("TGV", "tgv_a.npy"),
    ("Ours, v6 pool", "depb9v6_a_corrected.npy"),
    ("Ours, v9 3k", "depb9v9_3k_a_corrected.npy"),
]
Y, X, S = 361, 522, 128  # fig10's center-detail ROI
PX_UM = 10.0


def bandpass(img, lo_um, hi_um):
    h, w = img.shape
    fy = np.fft.fftfreq(h)
    fx = np.fft.rfftfreq(w)
    rad = np.hypot(fy[:, None], fx[None, :])
    m = (rad >= PX_UM / hi_um) & (rad <= PX_UM / lo_um)
    spec = np.fft.rfft2(img)
    spec[~m] = 0
    return np.fft.irfft2(spec, s=img.shape)


COLS = [
    ("Temperature", None),
    ("25\u201340 $\\mu$m band\n(acceptance band)", (25.0, 40.0)),
    ("20\u201325 $\\mu$m band\n(below claims)", (20.0, 25.0)),
]

panels = {}
for name, fname in ARMS:
    full = np.load(SRC / fname).astype(np.float64)
    crop = full[Y:Y + S, X:X + S]
    panels[name, 0] = crop - np.median(crop)
    for k, (_, band) in enumerate(COLS[1:], start=1):
        panels[name, k] = bandpass(full, *band)[Y:Y + S, X:X + S]

VLIM = []
for k in range(3):
    vals = np.concatenate([panels[n, k].ravel() for n, _ in ARMS])
    VLIM.append(np.percentile(np.abs(vals), 99.0))

# layout="none" at creation time: set_layout_engine("none") after the fact
# leaves a placeholder engine that silently blocks subplots_adjust.
fig, axes = plt.subplots(len(ARMS), 3, figsize=(W_DOUBLE * 0.52, 5.08),
                         layout="none")

for i, (name, _) in enumerate(ARMS):
    for k in range(3):
        ax = axes[i, k]
        cmap = "inferno" if k == 0 else "RdBu_r"
        v = VLIM[k]
        kw = dict(vmin=-v, vmax=v)
        ax.imshow(panels[name, k], cmap=cmap, interpolation="nearest", **kw)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#aaaaaa"), s.set_linewidth(0.4)
        if k > 0:
            ax.annotate(f"rms {np.std(panels[name, k]):.3f}",
                        (0.97, 0.03), xycoords="axes fraction", ha="right",
                        va="bottom", fontsize=5.2,
                        color="#222222",
                        bbox=dict(fc="white", ec="none", pad=0.15,
                                  alpha=0.8))
    axes[i, 0].set_ylabel(name, fontsize=7)

for k, (title, _) in enumerate(COLS):
    axes[0, k].set_title(title, fontsize=6.5, pad=3)

# scale bar
axes[-1, 0].plot([4, 34], [136, 136], color="#222222", lw=1.8,
                 solid_capstyle="butt", clip_on=False)
axes[-1, 0].annotate("300 $\\mu$m", (40, 137), ha="left", va="center",
                     fontsize=5.8, color="#222222", annotation_clip=False)
axes[-1, 0].set_xlim(-0.5, S - 0.5)
axes[-1, 0].set_ylim(S - 0.5, -0.5)

fig.subplots_adjust(left=0.07, right=0.995, top=0.945, bottom=0.035,
                    wspace=0.02, hspace=0.02)

paths = save_fig(fig, "fig66_band_decomposition")
print("\n".join(str(p) for p in paths))
