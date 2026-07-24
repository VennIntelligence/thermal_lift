"""Fig 39 — 300-scene micro-arm endpoint visuals (half-A reconstructions).

Visual companion to fig08's prior-emergence result (ACL-069): even 4k-8k-step
micro arms trained on only 300 scenes produce reasonable structure and (per
fig08's dot-fidelity probes) zero dot erasure, while the production 30k arm is
sharper but erases dots. One row of 6 half-A "Center detail" crops:
drizzle / TGV / micro v6-end 8k / micro v7-end 8k / micro v7-end 4k /
production depb9v6 30k.

Classical arrays (tgv/drizzle) are highpass residuals by construction; the
neural arms are absolute temperature, so EVERY arm gets the same matched
sigma=10 HR px Gaussian highpass display transform used in
fig10_real_visual_montage.py. Crop window reuses fig10's "Center detail" ROI
(y0=361, x0=522, size=128); one shared symmetric percentile scale from the
pooled crops; scale bar = 200 um (20 HR px at 10 um/px output grid).

Data: remote_inbox/20260716_micro_calib/*.npy (drizzle_a, tgv_a,
      micro_v6end_a_corrected, micro_v7end_a_corrected,
      micro_v7end_4k_a_corrected, depb9v6_a_corrected).
Run:  uv run python docs/publication_figures/scripts/fig39_micro_endpoint_visuals.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

SRC = REPO_ROOT / "remote_inbox/20260716_micro_calib"
ARMS = [
    ("drizzle_a.npy", "Drizzle"),
    ("tgv_a.npy", "TGV"),
    ("micro_v6end_a_corrected.npy", "Micro v6-end (8k)"),
    ("micro_v7end_a_corrected.npy", "Micro v7-end (8k)"),
    ("micro_v7end_4k_a_corrected.npy", "Micro v7-end (4k)"),
    ("depb9v6_a_corrected.npy", "Production (30k)"),
]
# fig10 "Center detail" ROI, reused verbatim
Y0, X0, S = 361, 522, 128

crops = []
for fname, label in ARMS:
    a = np.load(SRC / fname).astype(np.float64)
    # Matched-highpass display transform applied to EVERY arm (see fig10):
    # classical arrays are already highpass residuals but keep different
    # low-frequency remnants; one shared sigma=10 HR px AC-coupling puts
    # all six arms in the same display domain.
    a = a - gaussian_filter(a, sigma=10.0)
    crops.append((label, a[Y0:Y0 + S, X0:X0 + S]))

pooled = np.concatenate([c.ravel() for _, c in crops])
vmax = float(np.percentile(np.abs(pooled), 99.5))

fig, axes = plt.subplots(1, len(ARMS), figsize=(W_DOUBLE, 1.55))

for j, (label, crop) in enumerate(crops):
    ax = axes[j]
    im = ax.imshow(crop, cmap="inferno", vmin=-vmax, vmax=vmax,
                   interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(label, fontsize=8)

# 200 um scale bar (20 HR px at 10 um/px), first panel only
ax_sb = axes[0]
ax_sb.plot([6, 26], [120, 120], color="white", lw=2.2, solid_capstyle="butt")
ax_sb.annotate("200 $\\mu$m", (16, 116), color="white", ha="center",
               va="bottom", fontsize=7)

cbar = fig.colorbar(im, ax=axes, fraction=0.032, pad=0.02)
cbar.set_label("Highpass residual [$^\\circ$C]")

paths = save_fig(fig, "fig39_micro_endpoint_visuals")
print("\n".join(str(p) for p in paths))
