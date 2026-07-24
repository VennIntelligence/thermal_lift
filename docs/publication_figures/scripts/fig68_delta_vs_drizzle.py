"""Fig 68 — What each arm changes relative to drizzle, spatially.

Difference maps (arm minus drizzle, after per-crop median alignment) on the
fig10 center-detail ROI. Drizzle is the honest-but-soft anchor every
cross-FRC is measured against, so these maps show WHERE each method's
claimed detail actually sits: all three arms sharpen the same trace edges
(red/blue fringes hugging the layout), i.e. added contrast is anchored to
real structure rather than free-floating texture; TGV shows the strongest
fringes plus staircase-artifact blocking, v9-3k is the most conservative.
Top row gives the median-aligned temperature crops for context.

Data: remote_inbox/20260710_expab (seed-42 a-halves, registration-
corrected onto the drizzle grid). ACL context: cross-FRC metrology arms
(ACL-071/076).
Run:  uv run python docs/publication_figures/scripts/fig68_delta_vs_drizzle.py
"""

import matplotlib.pyplot as plt
import numpy as np

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

SRC = REPO_ROOT / "remote_inbox/20260710_expab"
ARMS = [
    ("TGV", "tgv_a.npy"),
    ("Ours, v6 pool", "depb9v6_a_corrected.npy"),
    ("Ours, v9 3k", "depb9v9_3k_a_corrected.npy"),
]
Y, X, S = 361, 522, 128  # fig10's center-detail ROI


def crop(path):
    c = np.load(SRC / path).astype(np.float64)[Y:Y + S, X:X + S]
    return c - np.median(c)


drz = crop("drizzle_a.npy")
crops = {name: crop(f) for name, f in ARMS}
diffs = {name: c - drz for name, c in crops.items()}

vt = np.percentile(np.abs(np.concatenate(
    [c.ravel() for c in crops.values()] + [drz.ravel()])), 99.5)
vd = np.percentile(np.abs(np.concatenate(
    [d.ravel() for d in diffs.values()])), 99.0)

# layout="none" at creation time: set_layout_engine("none") after the fact
# leaves a placeholder engine that silently blocks subplots_adjust.
fig, axes = plt.subplots(2, 4, figsize=(W_DOUBLE * 0.72, 2.86),
                         layout="none")

axes[0, 0].imshow(drz, cmap="inferno", vmin=-vt, vmax=vt,
                  interpolation="nearest")
axes[0, 0].set_title("Drizzle (anchor)", fontsize=7.5)
axes[1, 0].axis("off")

im_d = None
for j, (name, _) in enumerate(ARMS, start=1):
    axes[0, j].imshow(crops[name], cmap="inferno", vmin=-vt, vmax=vt,
                      interpolation="nearest")
    axes[0, j].set_title(name, fontsize=7.5)
    im_d = axes[1, j].imshow(diffs[name], cmap="RdBu_r", vmin=-vd, vmax=vd,
                             interpolation="nearest")
    axes[1, j].annotate(f"rms {np.std(diffs[name]):.3f}",
                        (0.97, 0.03), xycoords="axes fraction", ha="right",
                        va="bottom", fontsize=5.5, color="#222222",
                        bbox=dict(fc="white", ec="none", pad=0.15,
                                  alpha=0.85))

for ax in axes.ravel():
    if ax.axison:
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#aaaaaa"), s.set_linewidth(0.4)

axes[1, 1].set_ylabel("arm $-$ drizzle", fontsize=7.5)

# horizontal colorbar in the otherwise-empty bottom-left grid cell
cax = fig.add_axes([0.035, 0.22, 0.185, 0.028])
cb = fig.colorbar(im_d, cax=cax, orientation="horizontal")
cb.set_label("$\\Delta$ [$^\\circ$C]", fontsize=6.5, labelpad=2)
cb.ax.tick_params(labelsize=6)

# scale bar
axes[0, 0].plot([4, 34], [138, 138], color="#222222", lw=1.8,
                solid_capstyle="butt", clip_on=False)
axes[0, 0].annotate("300 $\\mu$m", (40, 139), ha="left", va="center",
                    fontsize=6, color="#222222", annotation_clip=False)
# the scale-bar plot() autoscales ylim beyond the image -> restore
axes[0, 0].set_xlim(-0.5, S - 0.5)
axes[0, 0].set_ylim(S - 0.5, -0.5)

fig.subplots_adjust(left=0.01, right=0.995, top=0.93, bottom=0.03,
                    wspace=0.03, hspace=0.08)

paths = save_fig(fig, "fig68_delta_vs_drizzle")
print("\n".join(str(p) for p in paths))
