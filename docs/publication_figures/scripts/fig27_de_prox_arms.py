"""Fig 27 -- D-E line: solver prox high-frequency-residual arms (ACL-042/043).

D-E restructures the unrolled solver's learned prox as `delta <- delta -
gaussian_blur(delta, sigma_hr)` before the update, so the prox can only inject
high-frequency detail while the noSE/noGN low-frequency anchor (ACL-041,
arm V11) stays structurally protected from extent drift. ACL-042 swept
sigma_hr in {4,5,8} and a wider prox (E6); ACL-043 additionally fed richer
on-the-fly phase-bin conditioning (9-/16-bin, E4/E5) on top of the same D-E
residual. None of the D-E arms recovered V9/V11-class synthetic fidelity
(PSNR ~32.5 vs V11 35.17): the entry's verdict is negative overall. Among the
D-E arms, ACL-042 names E2 (sigma_hr=4) the "most balanced" configuration
(essentially tied best on all three headline metrics), so it is highlighted
here; V11 is drawn as the pre-D-E reference for scale.

Data: data/de_prox_arms.csv (transcribed from ACL-042 / ACL-043 result
tables). Run:
uv run python docs/publication_figures/scripts/fig27_de_prox_arms.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import DATA_DIR, METHOD_PALETTE, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

df = pd.read_csv(DATA_DIR / "de_prox_arms.csv", comment="#")
order = ["V11", "E1", "E2", "E3", "E6", "E4", "E5"]
df = df.set_index("arm").loc[order].reset_index()

BLUE = METHOD_PALETTE["primary"]
GRAY = "#B0B0B0"
REF_GRAY = "#777777"


def bar_colors(is_ref, is_winner):
    if is_ref:
        return REF_GRAY
    if is_winner:
        return BLUE
    return GRAY


colors = [bar_colors(a == "V11", w) for a, w in zip(df["arm"], df["winner"])]
markers = ["s" if a == "V11" else "o" for a in df["arm"]]

panels = [
    ("synth_psnr_db", "Synthetic PSNR [dB]", "higher is better"),
    ("region_rmse", "Region RMSE [-]", "lower is better"),
    ("real_artifact", "Real artifact score [-]", "lower is better"),
]

# Dot plot (not bars): metrics are tightly clustered across D-E arms, so a
# zero-baseline bar chart would be unreadable / misleadingly flat. Points are
# comparable at their own y-position; a dashed line marks the V11 (pre-D-E)
# reference level for each metric.
fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 2.7), sharex=True)

x = np.arange(len(df))
for ax, (col, ylabel, direction) in zip(axes, panels):
    v11_val = df[col].iloc[0]
    ax.axhline(v11_val, color=REF_GRAY, ls="--", lw=0.8, zorder=1)
    for xi, v, c, m in zip(x, df[col], colors, markers):
        ax.scatter(xi, v, c=c, marker=m, s=42, zorder=3,
                    edgecolors="white", linewidths=0.6)
        va = "bottom" if col != "region_rmse" or xi != 6 else "top"
        dy = 6 if va == "bottom" else -7
        ax.annotate(f"{v:.3g}", (xi, v), xytext=(0, dy),
                    textcoords="offset points", ha="center", va=va,
                    fontsize=6.5, color="#333333")
    ax.set_xticks(x)
    ax.set_xticklabels(df["arm"], fontsize=8)
    ax.set_ylabel(ylabel)
    ax.set_title(direction, fontsize=8, fontweight="normal", color="#555555",
                 loc="right")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5, zorder=0)
    ymin, ymax = df[col].min(), df[col].max()
    pad = 0.22 * (ymax - ymin) if ymax > ymin else 0.05 * ymax
    ax.set_xlim(-0.6, len(df) - 0.4)
    ax.set_ylim(ymin - pad, ymax + pad)

axes[0].annotate("V11 = pre-D-E reference", xy=(0.30, 0.90),
                  xycoords="axes fraction", fontsize=6.8, color=REF_GRAY,
                  ha="left", va="top")
axes[1].annotate("E2 $\\sigma_{hr}$=4 (winner)", xy=(2, df["region_rmse"].iloc[2]),
                  xytext=(2.3, 0.129), fontsize=6.8, color=BLUE, ha="left",
                  arrowprops=dict(arrowstyle="-", color=BLUE, lw=0.6))

fig.suptitle("D-E solver-prox high-frequency-residual arms (ACL-042/043)",
              x=0.01, ha="left", fontsize=10)

paths = save_fig(fig, "fig27_de_prox_arms")
print("\n".join(str(p) for p in paths))
