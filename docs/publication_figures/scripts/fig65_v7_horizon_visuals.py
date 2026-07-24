"""Fig 65 — v7 training-horizon visual evolution (half-A reconstructions).

One structure-rich 120x120 HR px ROI (fine serpentine traces, y=371 x=514)
rendered across the v7-era micro-horizon sweep (12k / 16k / 20k / 24k
training steps) against the classical anchors (drizzle, TGV). Top row:
temperature-like values, DC-aligned per panel (crop median subtracted --
the classical arrays are highpass residuals by construction while the
neural arms are absolute temperature, so a raw shared scale would wash
out all contrast) on a shared robust scale (inferno).
Bottom row: matched highpass — a sigma=10 HR px Gaussian background is
subtracted from EACH panel identically (the established display convention,
cf. fig10) — on a shared symmetric scale (RdBu_r), so fine detail
differences along the horizon are visible. Scale bar: 30 HR px = 300 um at
10 um/px HR sample pitch (20 um detector pitch).

Data: remote_inbox/20260716_micro_horizon/*.npy (drizzle_a, tgv_a,
      v7e{12,16,20,24}k_a_corrected; "_corrected" = registration-corrected
      onto the drizzle grid). v7-era micro-horizon verdict, ~ACL-060s.
Run:  uv run python docs/publication_figures/scripts/fig65_v7_horizon_visuals.py
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

SRC = REPO_ROOT / "remote_inbox/20260716_micro_horizon"
ARMS = [
    ("drizzle_a.npy", "Drizzle"),
    ("v7e12k_a_corrected.npy", "v7 @ 12k"),
    ("v7e16k_a_corrected.npy", "v7 @ 16k"),
    ("v7e20k_a_corrected.npy", "v7 @ 20k"),
    ("v7e24k_a_corrected.npy", "v7 @ 24k"),
    ("tgv_a.npy", "TGV"),
]
Y0, X0, S = 371, 514, 120  # fine serpentine-trace ROI (HR px)

crops, crops_hp = [], []
for fname, _ in ARMS:
    a = np.load(SRC / fname).astype(np.float64)
    # matched-highpass display transform, identical for every arm
    hp = a - gaussian_filter(a, sigma=10.0)
    c = a[Y0:Y0 + S, X0:X0 + S]
    # DC-align: classical arrays are highpass residuals (~0 mean) while the
    # neural arms are absolute temperature; remove each crop's median so a
    # single shared scale shows within-panel contrast for all six arms.
    crops.append(c - np.median(c))
    crops_hp.append(hp[Y0:Y0 + S, X0:X0 + S])

# shared robust scales within each row, pooled across all panels of the row
pooled = np.concatenate([c.ravel() for c in crops])
vmin_t, vmax_t = np.percentile(pooled, [0.5, 99.5])
pooled_hp = np.concatenate([c.ravel() for c in crops_hp])
vmax_hp = float(np.percentile(np.abs(pooled_hp), 99.5))

fig, axes = plt.subplots(2, len(ARMS), figsize=(W_DOUBLE, 2.9))

for j, (_, label) in enumerate(ARMS):
    ax = axes[0, j]
    im_t = ax.imshow(crops[j], cmap="inferno", vmin=vmin_t, vmax=vmax_t,
                     interpolation="nearest")
    ax.set_title(label)
    ax_h = axes[1, j]
    im_hp = ax_h.imshow(crops_hp[j], cmap="RdBu_r", vmin=-vmax_hp,
                        vmax=vmax_hp, interpolation="nearest")
    for a_ in (ax, ax_h):
        a_.set_xticks([])
        a_.set_yticks([])
        for sp in a_.spines.values():
            sp.set_visible(False)

axes[0, 0].set_ylabel("$\\Delta$T (DC-aligned)", fontsize=8)
axes[1, 0].set_ylabel("Highpass ($\\sigma$=10)", fontsize=8)

# 300 um scale bar (30 HR px at 10 um/px), first panel of top row
ax_sb = axes[0, 0]
ax_sb.plot([6, 36], [112, 112], color="white", lw=2.2, solid_capstyle="butt")
ax_sb.annotate("300 $\\mu$m", (21, 108), color="white", ha="center",
               va="bottom", fontsize=7)

cbar_t = fig.colorbar(im_t, ax=axes[0, :], fraction=0.046, pad=0.015)
cbar_t.set_label("$\\Delta$T [$^\\circ$C]", fontsize=8)
cbar_hp = fig.colorbar(im_hp, ax=axes[1, :], fraction=0.046, pad=0.015)
cbar_hp.set_label("Highpass [$^\\circ$C]", fontsize=8)

paths = save_fig(fig, "fig65_v7_horizon_visuals")
print("\n".join(str(p) for p in paths))
