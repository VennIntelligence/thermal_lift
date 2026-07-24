"""Fig 61 — What the DC-residual self-audit actually sees (ACL-075).

Companion to fig04 (population AUCs): the residual maps themselves.
Each column is one probe dot; rows show (1) the drizzle reference where the
physical dot is visible, (2) the depb9v9s2 reconstruction where the dot is
erased or attenuated, (3) the absolute held-out data-consistency residual
|A x_hat - y| back-projected to the HR grid, where the erased dot re-appears
as a localized bump at the crosshair. The right-most column is a preserved
control dot: the reconstruction keeps it and the residual stays at the
background level.

These are the CLEAREST examples (top resid_win_max among erased dots) —
the per-dot signal is weak on average (erased-vs-kept AUC 0.68-0.84 across
arms, fig04), so this figure shows what a positive detection looks like,
not the typical case.

Data: output/dc_residual_confidence/ (residual maps + per-dot stats, built
against remote_inbox/20260710_expab reconstructions with the deployable
sigma=0.5 placeholder PSF). ACL-074/075.
Run:  uv run python docs/publication_figures/scripts/fig61_dc_residual_maps.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

ARM = "depb9v9s2"
STATS = REPO_ROOT / "output/dc_residual_confidence/per_dot_residual_stats.csv"
EXPAB = REPO_ROOT / "remote_inbox/20260710_expab"

df = pd.read_csv(STATS)
df = df[df.arm == ARM].set_index("dot_id")

# hand-picked after visual triage of the top-resid_win_max erased dots:
# d23/d20/d14 single-dot bumps, d640 a multi-dot constellation; d2471 is a
# preserved isolated control (visible dot, background-level residual).
DOTS = [
    (23, "erased"), (20, "erased"), (14, "erased"), (640, "erased"),
    (2471, "preserved"),
]
W = 16  # half-window, HR px

drz = np.load(EXPAB / "drizzle_a.npy")
rec = np.load(EXPAB / f"{ARM}_a_corrected.npy")
res = np.abs(np.load(
    REPO_ROOT / f"output/dc_residual_confidence/{ARM}_residmap_a.npy"))

fig, axes = plt.subplots(3, len(DOTS), figsize=(W_DOUBLE * 0.8, 3.95))
fig.set_layout_engine("none")

ROWS = ["Drizzle\n(dot present)", "Recon v9s2", "|DC residual|"]
last_im = None
for j, (dot, cls) in enumerate(DOTS):
    y, x = int(df.loc[dot, "y"]), int(df.loc[dot, "x"])
    sl = np.s_[y - W:y + W + 1, x - W:x + W + 1]
    for i, (img, cm) in enumerate([(drz, "gray"), (rec, "gray"),
                                   (res, "magma")]):
        ax = axes[i, j]
        c = img[sl]
        if i < 2:
            kw = dict(vmin=np.percentile(c, 2), vmax=np.percentile(c, 98))
        else:
            kw = dict(vmin=0.0, vmax=0.36)
        im = ax.imshow(c, cmap=cm, interpolation="nearest", **kw)
        if i == 2:
            last_im = im
        ax.plot(W, W, "+", color="#00C2C7", ms=7, mew=1.1)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#999999"), s.set_linewidth(0.5)
    wm = df.loc[dot, "resid_win_max"]
    tag = "erased" if cls == "erased" else "ctrl"
    axes[0, j].set_title(f"dot {dot} $\\cdot$ {tag}\nwin-max {wm:.2f}",
                         fontsize=7)

for i, lab in enumerate(ROWS):
    axes[i, 0].set_ylabel(lab, fontsize=7.5)

cax = fig.add_axes([0.905, 0.095, 0.013, 0.20])
cb = fig.colorbar(last_im, cax=cax)
cb.set_label("|residual| [DN]", fontsize=6.5)
cb.ax.tick_params(labelsize=6)

# scale bar: 10 px = 100 um in a 33-px window
axes[2, 0].plot([2, 12], [30, 30], color="white", lw=1.8,
                solid_capstyle="butt")
axes[2, 0].annotate("100 $\\mu$m", (7, 28.4), ha="center", va="bottom",
                    fontsize=6, color="white")

fig.suptitle("Held-out DC residual re-detects erased dots "
             "(clearest cases; population AUC in fig04)",
             x=0.02, y=0.985, ha="left", fontsize=9)
fig.subplots_adjust(left=0.075, right=0.895, top=0.855, bottom=0.03,
                    wspace=0.08, hspace=0.10)

paths = save_fig(fig, "fig61_dc_residual_maps")
print("\n".join(str(p) for p in paths))
