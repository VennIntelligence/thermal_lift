"""Fig 60 — Optical ground truth registered onto the HR thermal grid.

The optical microscope view of the board (0.211 um/px) was registered onto
the 10 um/px HR reconstruction grid by optical_register.py (similarity
transform, theta=225.2 deg, NCC peak 0.985, residual NCC 0.951 after
blur-1.5; see output/visboard_stage2b/optical_register_result.json). The
warped optical footprint is a ~46 px diamond covering a serpentine-trace
region.

Top row: the warped optical patch (gray) next to the same crop from four
reconstruction arms (shared temperature scale, inferno). Bottom row: the
optical trace boundary (contour at the mid-gray threshold, cyan) overlaid
on each arm — geometric agreement between reconstructed hot traces and the
optically-true trace layout is directly visible; disagreement would show as
contour lines cutting through thermal structure.

Data: remote_inbox/20260713_dotprobe/optical_warp_hr.npy (warped optical),
remote_inbox/20260710_expab/*_a*.npy (reconstructions, registration-
corrected onto the drizzle grid). ACL context: optical registration era
(stage0h dot probe); champion arms of ACL-071/076.
Run:  uv run python docs/publication_figures/scripts/fig60_optical_registration.py
"""

import matplotlib.pyplot as plt
import numpy as np

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

OPT = np.load(REPO_ROOT / "remote_inbox/20260713_dotprobe/optical_warp_hr.npy")
EXPAB = REPO_ROOT / "remote_inbox/20260710_expab"
ARMS = [
    ("drizzle_a.npy", "Drizzle"),
    ("tgv_a.npy", "TGV"),
    ("depb9v6_a_corrected.npy", "Ours, v6 pool"),
    ("depb9v9_3k_a_corrected.npy", "Ours, v9 3k"),
]

Y0, Y1, X0, X1 = 391, 457, 540, 606  # footprint bbox from the register json


def crop(a):
    return a[Y0:Y1, X0:X1]


opt = crop(OPT).astype(float)
mask = np.isfinite(opt) & (opt > 0)
opt_m = np.where(mask, opt, np.nan)
# mid-gray threshold between the trace (dark) and substrate (light) modes
vals = opt[mask]
thr = 0.5 * (np.percentile(vals, 15) + np.percentile(vals, 85))

# arms carry different DC offsets -> compare median-removed crops on a
# single shared scale
arm_crops = {lab: crop(np.load(EXPAB / f)) for f, lab in ARMS}
arm_crops = {lab: c - np.median(c) for lab, c in arm_crops.items()}
allv = np.concatenate([c.ravel() for c in arm_crops.values()])
vmin, vmax = np.percentile(allv, [1, 99.5])

fig, axes = plt.subplots(2, 5, figsize=(W_DOUBLE, 3.55))
fig.set_layout_engine("none")

# ── top row: optical + temperature crops ─────────────────────────────
ax = axes[0, 0]
ax.imshow(opt_m, cmap="gray", interpolation="nearest")
ax.set_title("Optical (warped)", fontsize=8)
for (f, lab), ax in zip(ARMS, axes[0, 1:]):
    ax.imshow(arm_crops[lab], cmap="inferno", vmin=vmin, vmax=vmax,
              interpolation="nearest")
    ax.set_title(lab, fontsize=8)

# ── bottom row: optical trace contour over each arm ──────────────────
ax = axes[1, 0]
ax.imshow(opt_m, cmap="gray", interpolation="nearest")
ax.contour(opt_m, levels=[thr], colors="#00C2C7", linewidths=0.55)
ax.set_title("+ trace contour", fontsize=7.5, color="#008B8F")
for (f, lab), ax in zip(ARMS, axes[1, 1:]):
    ax.imshow(arm_crops[lab], cmap="inferno", vmin=vmin, vmax=vmax,
              interpolation="nearest")
    ax.contour(opt_m, levels=[thr], colors="#00C2C7", linewidths=0.55)

for ax in axes.ravel():
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

# scale bar: 30 HR px = 300 um on top-left panel
# scale bar just below the optical panel (footprint is rotated -> no clean
# in-panel corner)
axes[0, 0].plot([1, 31], [70, 70], color="#222222", lw=2,
                solid_capstyle="butt", clip_on=False)
axes[0, 0].annotate("300 $\\mu$m", (35, 70.5), ha="left", va="center",
                    fontsize=6.5, color="#222222", annotation_clip=False)
axes[0, 0].set_xlim(-0.5, X1 - X0 - 0.5)
axes[0, 0].set_ylim(Y1 - Y0 - 0.5, -0.5)

axes[1, 0].annotate(
    "similarity transform: 0.211 $\\mu$m/opt px, $\\theta$=225.2$^\\circ$, "
    "NCC 0.985 (residual 0.951)",
    (0.0, -0.10), xycoords="axes fraction", ha="left", va="top", fontsize=6.5,
    color="#555555", annotation_clip=False)

fig.suptitle("Registered optical ground truth vs reconstructed traces "
             "(serpentine region)", x=0.01, ha="left", fontsize=9)
fig.subplots_adjust(top=0.87, bottom=0.075, wspace=0.06, hspace=0.14,
                    left=0.01, right=0.99)

paths = save_fig(fig, "fig60_optical_registration")
print("\n".join(str(p) for p in paths))
