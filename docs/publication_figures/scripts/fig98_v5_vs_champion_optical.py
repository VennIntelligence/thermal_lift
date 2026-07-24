"""fig98 — v5-hybrid vs champion vs classical, over the registered optical GT.

Owner request 2026-07-13: a center-detail zoom comparison with the optical
ground truth on the left and TGV + the freshly re-inferred v5_hybrid + the
champion depb9v6 side by side, on the same serpentine ROI as fig60.

v5_hybrid (solver_v5_sharp, the fig12 checkpoint, ACL-029) was re-reconstructed
on the seed-42 real halves through the CURRENT inference pipeline using the
checkpoint's own config (unroll=4, phase_bins=4, SE+GroupNorm prox; state_dict
loaded with 0 missing / 0 unexpected). Its raw output sits within 0.03 HR px of
the offset-corrected champion arms (empirically verified), so no extra
correction is applied for this visual side-by-side.

Layout (2x5): [Optical | Drizzle | TGV | v5 hybrid | depb9v6].
Top row: temperature crops (median-removed, shared inferno scale).
Bottom row: optical trace contour (cyan) overlaid on each — geometric fidelity
of the reconstructed hot traces vs the optically-true layout.

NOTE: this is a VISUAL comparison. Whether v5's apparent crispness is recovered
structure or the ACL-029 waffle/hallucination is adjudicated by corrected
cross-FRC vs drizzle (computed separately), not by eye.

Run:  uv run python docs/publication_figures/scripts/fig98_v5_vs_champion_optical.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

OPT = np.load(REPO_ROOT / "remote_inbox/20260713_dotprobe/optical_warp_hr.npy")
EXPAB = REPO_ROOT / "remote_inbox/20260710_expab"

Y0, Y1, X0, X1 = 391, 457, 540, 606  # fig60 serpentine footprint bbox
# v5 ROI crop was saved padded at this global origin (emit_v5roi.py)
V5_ROI_Y0, V5_ROI_X0 = 351, 500


def crop(a):
    return a[Y0:Y1, X0:X1]


# full-array arms (champions + classical references, all on the same grid)
V5_LAB = "v5 hybrid\n(re-inferred)"
CHAMP_LAB = "Ours, v6 pool\n(champion)"

# v5 comes from the padded ROI crop of the grid-ALIGNED reconstruction
# (raw v5 sat ~0.6 HR px off the champion grid; that registration offset was
# removed before this crop, matching the corrected cross-FRC measurement).
v5_roi = np.load(EXPAB / "v5sharp_a_corrected_roi.npy").astype(float)
v5_crop = v5_roi[Y0 - V5_ROI_Y0:Y1 - V5_ROI_Y0, X0 - V5_ROI_X0:X1 - V5_ROI_X0]

# corrected cross-FRC@30um vs drizzle (same leaderboard as the champion anchors,
# which reproduced exactly: depb9v6 0.6611, tgv 0.7017)
FRC30 = {"TGV": 0.7017, V5_LAB: 0.6256, CHAMP_LAB: 0.6611}

# raw per-arm crops (v6 & v5 are absolute temperature; TGV/Drizzle are high-pass)
raw_crops = {
    "Drizzle": crop(np.load(EXPAB / "drizzle_a.npy")).astype(float),
    "TGV": crop(np.load(EXPAB / "tgv_a.npy")).astype(float),
    V5_LAB: v5_crop.astype(float),
    CHAMP_LAB: crop(np.load(EXPAB / "depb9v6_a_corrected.npy")).astype(float),
}
ORDER = ["Drizzle", "TGV", V5_LAB, CHAMP_LAB]

# Composition (matches fig96, owner-approved "clean" look): every panel =
# a single SHARED smooth absolute-temperature scene background  +  that panel's
# OWN recovered structure. Median-removal alone (no background) shows only the
# high-frequency content and reads as speckle even for the clean champion; the
# shared background restores the smooth thermal field so the genuinely-clean
# champion looks clean while v5's ACL-029 waffle still rides on top as texture.
BG_SIGMA = 8.0  # HR px
bg = gaussian_filter(raw_crops[CHAMP_LAB], BG_SIGMA)  # v6 absolute-temp background
arm_crops = {k: bg + (raw_crops[k] - gaussian_filter(raw_crops[k], BG_SIGMA))
             for k in ORDER}

opt = crop(OPT).astype(float)
mask = np.isfinite(opt) & (opt > 0)
opt_m = np.where(mask, opt, np.nan)
vals = opt[mask]
thr = 0.5 * (np.percentile(vals, 15) + np.percentile(vals, 85))

allv = np.concatenate([arm_crops[k].ravel() for k in ORDER])
vmin, vmax = np.percentile(allv, [1, 99.5])

fig, axes = plt.subplots(2, 5, figsize=(W_DOUBLE, 3.55))
fig.set_layout_engine("none")

# top row: optical + temperature crops
axes[0, 0].imshow(opt_m, cmap="gray", interpolation="nearest")
axes[0, 0].set_title("Optical (warped)", fontsize=8)
for lab, ax in zip(ORDER, axes[0, 1:]):
    ax.imshow(arm_crops[lab], cmap="inferno", vmin=vmin, vmax=vmax,
              interpolation="nearest")
    sub = f"\nFRC {FRC30[lab]:.3f}" if lab in FRC30 else ""
    ax.set_title(lab + sub, fontsize=8)

# bottom row: optical trace contour over each arm
axes[1, 0].imshow(opt_m, cmap="gray", interpolation="nearest")
axes[1, 0].contour(opt_m, levels=[thr], colors="#00C2C7", linewidths=0.55)
axes[1, 0].set_title("+ trace contour", fontsize=7.5, color="#008B8F")
for lab, ax in zip(ORDER, axes[1, 1:]):
    ax.imshow(arm_crops[lab], cmap="inferno", vmin=vmin, vmax=vmax,
              interpolation="nearest")
    ax.contour(opt_m, levels=[thr], colors="#00C2C7", linewidths=0.55)

for ax in axes.ravel():
    ax.set_xticks([]), ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)

axes[0, 0].plot([1, 31], [70, 70], color="#222222", lw=2,
                solid_capstyle="butt", clip_on=False)
axes[0, 0].annotate("300 $\\mu$m", (35, 70.5), ha="left", va="center",
                    fontsize=6.5, color="#222222", annotation_clip=False)
axes[0, 0].set_xlim(-0.5, X1 - X0 - 0.5)
axes[0, 0].set_ylim(Y1 - Y0 - 0.5, -0.5)

axes[1, 0].annotate(
    "v5 re-inferred through current pipeline (0 param mismatch), aligned to the "
    "champion grid (0.6 HR px registration offset removed).  Punchline: v5 looks "
    "the sharpest by eye yet scores 0.626 — below the champion 0.661 and TGV 0.702. "
    "Its crispness is largely the ACL-029 background waffle, which corrected "
    "cross-FRC vs independent drizzle does not credit.",
    (0.0, -0.12), xycoords="axes fraction", ha="left", va="top", fontsize=6.0,
    color="#555555", annotation_clip=False, wrap=True)

fig.suptitle("v5-hybrid vs champion vs classical over registered optical GT "
             "(serpentine detail) — sharpest ≠ best on the honest metric "
             "(FRC = corrected cross-FRC@30µm vs drizzle)",
             x=0.01, ha="left", fontsize=9)
fig.subplots_adjust(top=0.87, bottom=0.11, wspace=0.06, hspace=0.14,
                    left=0.01, right=0.99)

paths = save_fig(fig, "fig98_v5_vs_champion_optical")
print("\n".join(str(p) for p in paths))
