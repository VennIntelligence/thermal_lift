"""Fig 10 — Real-data visual comparison montage (half-A reconstructions).

3 ROIs (center detail / high-frequency traces / flat region) × 4 arms
(drizzle, TGV, ours-v6, ours-v8) on the 248-frame real session, split A.
Classical arrays are highpass residuals by construction; the neural arms are
absolute temperature, so they are AC-coupled by subtracting a σ=10 HR px
Gaussian background before display — the same matched-highpass display
transform documented in output/visboard_stage2b/source_crops/manifest.json.
Crop windows are reused verbatim from that manifest. One shared symmetric
scale (percentile-based) for all panels; scale bar = 200 µm (20 HR px at
10 µm/px output grid).

Data: remote_inbox/20260716_v8_verdict/*.npy (drizzle_a, tgv_a,
      depb9v6_a_corrected, depb9v8_a_corrected).
Run:  uv run python docs/publication_figures/scripts/fig10_real_visual_montage.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

SRC = REPO_ROOT / "remote_inbox/20260716_v8_verdict"
ARMS = [
    ("drizzle_a.npy", "Drizzle", False),
    ("tgv_a.npy", "TGV", False),
    ("depb9v6_a_corrected.npy", "Ours (v6 pool)", True),
    ("depb9v8_a_corrected.npy", "Ours (v8 pool)", True),
]
ROIS = [
    ("Center detail", 361, 522, 128),
    ("Fine traces", 316, 443, 128),
    ("Flat region", 791, 869, 128),
]

arrays = []
for fname, label, is_abs in ARMS:
    a = np.load(SRC / fname).astype(np.float64)
    # Matched-highpass display transform applied to EVERY arm: the classical
    # arrays are already highpass residuals but keep different low-frequency
    # remnants (drizzle especially), so one shared sigma=10 HR px AC-coupling
    # puts all four in the same display domain.
    a = a - gaussian_filter(a, sigma=10.0)
    arrays.append((label, a))

# shared symmetric display scale for structured ROIs (rows 0-1);
# the flat ROI gets its own tighter scale so the noise floor is visible.
pooled = np.concatenate([
    a[y0:y0 + s, x0:x0 + s].ravel()
    for _, a in arrays for (_, y0, x0, s) in ROIS[:2]
])
vmax = float(np.percentile(np.abs(pooled), 99.5))
flat_pooled = np.concatenate([
    a[ROIS[2][1]:ROIS[2][1] + ROIS[2][3], ROIS[2][2]:ROIS[2][2] + ROIS[2][3]].ravel()
    for _, a in arrays
])
vmax_flat = float(np.percentile(np.abs(flat_pooled), 99.5))

fig, axes = plt.subplots(len(ROIS), len(ARMS), figsize=(W_DOUBLE, 5.6))

for i, (roi_name, y0, x0, s) in enumerate(ROIS):
    for j, (label, a) in enumerate(arrays):
        ax = axes[i, j]
        crop = a[y0:y0 + s, x0:x0 + s]
        vm = vmax_flat if i == 2 else vmax
        img = ax.imshow(crop, cmap="inferno", vmin=-vm, vmax=vm,
                        interpolation="nearest")
        if i == 1 and j == len(arrays) - 1:
            im = img          # handle for the structured-ROI colorbar
        if i == 2 and j == len(arrays) - 1:
            im_flat = img     # handle for the flat-ROI colorbar
        ax.set_xticks([])
        ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_visible(False)
        if i == 0:
            ax.set_title(label)
        if j == 0:
            ax.set_ylabel(roi_name, fontsize=9)

# 200 µm scale bar (20 HR px at 10 µm/px), bottom-left panel only
ax_sb = axes[-1, 0]
ax_sb.plot([6, 26], [120, 120], color="white", lw=2.2, solid_capstyle="butt")
ax_sb.annotate("200 $\\mu$m", (16, 116), color="white", ha="center",
               va="bottom", fontsize=7)

cbar = fig.colorbar(im, ax=axes[:2, :], fraction=0.032, pad=0.02)
cbar.set_label("Highpass residual [$^\\circ$C]")
cbar_f = fig.colorbar(im_flat, ax=axes[2:, :], fraction=0.065, pad=0.02)
cbar_f.set_label("Flat ROI [$^\\circ$C]")

paths = save_fig(fig, "fig10_real_visual_montage")
print("\n".join(str(p) for p in paths))
