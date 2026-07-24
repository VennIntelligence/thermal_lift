"""Fig 06 — Real-data symmetrized cross-FRC: neural solver vs classical SR.

Cross-method FRC (mean of X_a–Y_b / X_b–Y_a, drizzle as the independent
reference) on the 248-frame real session with refined alignment and the
+0.5 px grid correction (ACL-048/049 measurement convention). Shaded: the
25–40 µm genuine-gain band; hatched: periods below the 20 µm detector
aperture zero (audit-only, never trusted). The half-bit criterion gives the
per-method cutoffs listed in the legend.

Data: remote_inbox/20260708_stage0j/{frc_curves_long,method_summary}.csv.
Run:  uv run python docs/publication_figures/scripts/fig06_frc_leaderboard.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import REPO_ROOT, W_1P5, save_fig, setup_academic_style

setup_academic_style()

SRC = REPO_ROOT / "remote_inbox/20260708_stage0j"
df = pd.read_csv(SRC / "frc_curves_long.csv")

METHODS = {
    "tgv_x_drz":      dict(color="#333333", ls="-",  label="TGV (cutoff 23.0 $\\mu$m)"),
    "maptv_x_drz":    dict(color="#937860", ls="-",  label="MAP-TV (cutoff 24.6 $\\mu$m)"),
    "v14_50k_x_drz":  dict(color="#4C72B0", ls="-",  label="Ours v14, 50k (cutoff 25.5 $\\mu$m)"),
    "v14_20k_x_drz":  dict(color="#4C72B0", ls="--", label="Ours v14, 20k (cutoff 25.5 $\\mu$m)"),
}

fig, ax = plt.subplots(figsize=(W_1P5, 3.2))

PMIN, PMAX = 15.0, 150.0
PCLIP = 20.0  # curves stop at the detector aperture zero; below is audit-only
for m, st in METHODS.items():
    d = df[(df["method"] == m) & (df["period_um"] >= PCLIP) & (df["period_um"] <= PMAX)]
    d = d.sort_values("period_um", ascending=False)
    ax.plot(d["period_um"], d["frc"], color=st["color"], ls=st["ls"],
            lw=1.3, label=st["label"])

# half-bit threshold (same for all methods at a given ring)
d0 = df[(df["method"] == "tgv_x_drz") & (df["period_um"] >= PCLIP) & (df["period_um"] <= PMAX)]
d0 = d0.sort_values("period_um", ascending=False)
ax.plot(d0["period_um"], d0["threshold_half_bit"], color="#999999", ls=":",
        lw=1.0, label="Half-bit criterion")

# genuine-gain band 25-40 µm
ax.axvspan(25, 40, color="#4C72B0", alpha=0.08, zorder=0)
ax.annotate("genuine-gain\nband 25–40 $\\mu$m", (np.sqrt(25 * 40), 0.06),
            ha="center", va="bottom", fontsize=7, color="#4C72B0")
# aperture zero: below 20 µm audit-only
ax.axvspan(PMIN, 20, color="#888888", alpha=0.12, zorder=0, hatch="///", lw=0)
ax.annotate("detector aperture\nzero (20 $\\mu$m):\naudit only", (17.2, 0.60),
            ha="center", va="center", fontsize=6.5, color="#555555")

ax.set_xscale("log")
ax.set_xlim(PMAX, PMIN)  # high period (low frequency) on the left
ticks = [150, 100, 80, 60, 40, 30, 24, 20, 16]
ax.set_xticks(ticks)
ax.set_xticklabels([str(t) for t in ticks])
ax.set_xlabel("Period [$\\mu$m]")
ax.set_ylabel("Cross-method FRC")
ax.set_ylim(-0.05, 1.02)
ax.axhline(0, color="#cccccc", lw=0.6, zorder=0)
ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.42), fontsize=7)
ax.set_title("Real-data resolution: symmetrized cross-FRC vs drizzle", loc="left")

paths = save_fig(fig, "fig06_frc_leaderboard")
print("\n".join(str(p) for p in paths))
