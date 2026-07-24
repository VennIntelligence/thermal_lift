"""Fig 63 — What self split-half FRC rewards: split-stable texture in a
flat region.

Both frame-halves band-passed to the acceptance band (25-40 um period,
2.5-4 px at 10 um/px) and cropped to the flat ROI of fig10 (no real
structure expected). Drizzle's halves are built from disjoint frames, so
its band content decorrelates (r = 0.15) — honest noise. TGV (r = 0.84)
and the v6 arm (r = 0.70) produce texture that is nearly the same in both
halves: exactly the split-stable content that inflates a self split-half
FRC, whether or not it is real. This is the visual counterpart of the
fig31/32 verdict that self-FRC is invalid metrology and cross-FRC against
drizzle is used instead. Note v9-3k (r = 0.11) decorrelates like drizzle —
its band content in flat regions behaves noise-like.

Half-to-half Pearson r is computed on the displayed crops. Shared
symmetric color scale across all panels.

Data: remote_inbox/20260710_expab (a/b half reconstructions, seed-42
phase-stratified split). ACL context: split-half controls (fig31/32).
Run:  uv run python docs/publication_figures/scripts/fig63_splithalf_flat_band.py
"""

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from pubfig_style import REPO_ROOT, save_fig, setup_academic_style

setup_academic_style()
# Manual-layout script: keep the constrained engine off, otherwise the
# tight-bbox save restores it and silently re-lays out the 2nd (pdf) save.
mpl.rcParams["figure.constrained_layout.use"] = False

SRC = REPO_ROOT / "remote_inbox/20260710_expab"
ARMS = [
    ("Drizzle", "drizzle_{h}.npy"),
    ("TGV", "tgv_{h}.npy"),
    ("Ours, v6 pool", "depb9v6_{h}_corrected.npy"),
    ("Ours, v9 3k", "depb9v9_3k_{h}_corrected.npy"),
]
Y, X, S = 791, 869, 128  # fig10's flat ROI
LO_UM, HI_UM, PX_UM = 25.0, 40.0, 10.0


def bandpass(img):
    h, w = img.shape
    fy = np.fft.fftfreq(h)
    fx = np.fft.rfftfreq(w)
    rad = np.hypot(fy[:, None], fx[None, :])  # cycles / px
    m = (rad >= PX_UM / HI_UM) & (rad <= PX_UM / LO_UM)
    spec = np.fft.rfft2(img)
    spec[~m] = 0
    return np.fft.irfft2(spec, s=img.shape)


crops = {}   # (arm, half) -> band crop
for name, pat in ARMS:
    for h in "ab":
        full = np.load(SRC / pat.format(h=h)).astype(np.float64)
        crops[name, h] = bandpass(full)[Y:Y + S, X:X + S]

vmax = 2.5 * np.median([np.std(c) for c in crops.values()])

# Geometry derived from a fixed square panel edge so every grid cell is
# exactly square: no dead space around the equal-aspect imshow panels.
P = 1.22                      # square panel edge [in]
GAP = 0.05                    # inter-panel gap [in]
LEFT, RIGHT = 0.18, 0.03      # room for "half A/B" ylabels / edge pad [in]
TOP, BOTTOM = 0.38, 0.44      # room for 2-line titles / scale bar + caption [in]
FIG_W = LEFT + len(ARMS) * P + (len(ARMS) - 1) * GAP + RIGHT
FIG_H = TOP + 2 * P + GAP + BOTTOM

fig, axes = plt.subplots(2, len(ARMS), figsize=(FIG_W, FIG_H))

for j, (name, _) in enumerate(ARMS):
    a, b = crops[name, "a"], crops[name, "b"]
    r = np.corrcoef(a.ravel(), b.ravel())[0, 1]
    for i, c in enumerate([a, b]):
        ax = axes[i, j]
        ax.imshow(c, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#aaaaaa"), s.set_linewidth(0.5)
    axes[0, j].set_title(f"{name}\nhalf-half $r$ = {r:.2f}", fontsize=8)

axes[0, 0].set_ylabel("half A", fontsize=8)
axes[1, 0].set_ylabel("half B", fontsize=8)

# scale bar below the bottom-left panel (the band image itself is too
# noisy for a legible in-panel bar)
axes[1, 0].plot([2, 32], [136, 136], color="#222222", lw=1.8,
                solid_capstyle="butt", clip_on=False)
axes[1, 0].annotate("300 $\\mu$m", (37, 137), ha="left", va="center",
                    fontsize=6.5, color="#222222", annotation_clip=False)
axes[1, 0].set_xlim(-0.5, S - 0.5)
axes[1, 0].set_ylim(S - 0.5, -0.5)

panel_span_center = (LEFT + (len(ARMS) * P + (len(ARMS) - 1) * GAP) / 2) / FIG_W
fig.text(panel_span_center, 0.012,
         "Flat ROI, 25-40 $\\mu$m band. Split-stable texture (TGV, v6) is "
         "what self split-half FRC rewards\n--- hence cross-FRC vs drizzle "
         "as metrology (fig31/32).",
         ha="center", va="bottom", fontsize=7, style="italic",
         color="#444444", linespacing=1.4)
fig.subplots_adjust(left=LEFT / FIG_W, right=1 - RIGHT / FIG_W,
                    top=1 - TOP / FIG_H, bottom=BOTTOM / FIG_H,
                    wspace=GAP / P, hspace=GAP / P)

paths = save_fig(fig, "fig63_splithalf_flat_band")
print("\n".join(str(p) for p in paths))
