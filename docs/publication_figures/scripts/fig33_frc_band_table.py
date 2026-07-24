"""Fig 33 — Cross-FRC band table across methods (stage0j leaderboard, extended view).

Companion to fig06_frc_leaderboard.py: fig06 shows continuous FRC curves for
the four drizzle-referenced methods; this figure is the discrete "band table"
view used in the stage0j leaderboard, plotted across all five method pairs
(including v14_50k x tgv) at the six audited FRC evaluation periods
(80/60/40/30/24/20 um). Each point is the symmetrized cross-mean FRC
(mean of X_a-vs-Y_b and X_b-vs-Y_a); the light whisker around each point
spans the two individual directions, exposing direction asymmetry that the
mean alone hides. The 20 um column sits at/below the 20 um detector aperture
zero (current_spatial_resolution_um == pixel_size_um == 20.0 in
method_summary.csv) and its per-direction values disagree in sign for
several methods (e.g. tgv_x_drz: +0.99 vs 0.42 mean up at 24um but xa_yb
-0.61 / xb_ya -0.83 at 20um) -- it is shaded/hatched and annotated as
audit-only, values kept visible rather than dropped.

Message: methods agree closely in the low-frequency bands (80-40 um) and
separate meaningfully in the 30-24 um decision band; the 20 um column is
not to be trusted for ranking.

Data: remote_inbox/20260708_stage0j/method_summary.csv
Run:  uv run python docs/publication_figures/scripts/fig33_frc_band_table.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import REPO_ROOT, W_1P5, save_fig, setup_academic_style

setup_academic_style()

SRC = REPO_ROOT / "remote_inbox/20260708_stage0j" / "method_summary.csv"
df = pd.read_csv(SRC).set_index("method")

BANDS = [80, 60, 40, 30, 24, 20]
X = np.arange(len(BANDS))

# color/style per user spec: tgv black solid, maptv taupe, v14 blue
# solid/dashed for 50k/20k, v14 x tgv gray dotted
METHODS = {
    "tgv_x_drz":     dict(color="#333333", ls="-",  marker="s", label="TGV $\\times$ drizzle"),
    "maptv_x_drz":   dict(color="#937860", ls="-",  marker="D", label="MAP-TV $\\times$ drizzle"),
    "v14_50k_x_drz": dict(color="#4C72B0", ls="-",  marker="o", label="Ours v14, 50k $\\times$ drizzle"),
    "v14_20k_x_drz": dict(color="#4C72B0", ls="--", marker="o", label="Ours v14, 20k $\\times$ drizzle"),
    "v14_50k_x_tgv": dict(color="#888888", ls=":",  marker="^", label="Ours v14, 50k $\\times$ TGV"),
}
# small horizontal jitter so the five whiskers at a given band don't overlap
JITTER = {m: (i - (len(METHODS) - 1) / 2) * 0.09 for i, m in enumerate(METHODS)}

fig, ax = plt.subplots(figsize=(W_1P5, 3.7))

# shade + hatch the 20 um audit-only column
ax.axvspan(X[-1] - 0.5, X[-1] + 0.5, color="#888888", alpha=0.10, hatch="///", lw=0, zorder=0)

for m, st in METHODS.items():
    row = df.loc[m]
    mean_vals = np.array([row[f"frc_at_{b}um"] for b in BANDS])
    xa_yb = np.array([row[f"frc_at_{b}um_xa_yb"] for b in BANDS])
    xb_ya = np.array([row[f"frc_at_{b}um_xb_ya"] for b in BANDS])
    lo = np.minimum(xa_yb, xb_ya)
    hi = np.maximum(xa_yb, xb_ya)
    xj = X + JITTER[m]

    # per-direction whisker (light, thin, behind the main marker/line)
    ax.errorbar(
        xj, mean_vals,
        yerr=np.vstack([mean_vals - lo, hi - mean_vals]),
        fmt="none", ecolor=st["color"], elinewidth=0.7, capsize=2.5,
        capthick=0.7, alpha=0.35, zorder=1,
    )
    ax.plot(xj, mean_vals, color=st["color"], ls=st["ls"], marker=st["marker"],
            markersize=4, lw=1.3, label=st["label"], zorder=3)

ax.axhline(0, color="#cccccc", lw=0.6, zorder=0)

ax.set_xticks(X)
ax.set_xticklabels([f"{b}" if b != 20 else "20\n(audit)" for b in BANDS])
ax.set_xlim(-0.5, len(BANDS) - 0.5)
ax.set_xlabel("Band period [$\\mu$m]")
ax.set_ylabel("Cross-method FRC (mean of directions)")
ax.set_ylim(-1.08, 1.12)

ax.annotate(
    "20 $\\mu$m = detector aperture\nzero: values audit-only,\nnot used for ranking",
    xy=(X[-1], -1.0), xytext=(X[-1] - 1.55, -0.95),
    ha="left", va="center", fontsize=6.5, color="#555555",
    arrowprops=dict(arrowstyle="->", color="#777777", lw=0.7),
)
ax.annotate(
    "decision band:\nmethods separate",
    xy=(np.mean([X[3], X[4]]), 0.55), xytext=(np.mean([X[3], X[4]]) - 0.05, 0.30),
    ha="center", va="top", fontsize=6.5, color="#4C72B0",
    arrowprops=dict(arrowstyle="->", color="#4C72B0", lw=0.7),
)

ax.set_title("Cross-FRC band table across methods (stage0j leaderboard)", loc="left")
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.20), ncol=2, fontsize=7)

paths = save_fig(fig, "fig33_frc_band_table")
print("\n".join(str(p) for p in paths))
