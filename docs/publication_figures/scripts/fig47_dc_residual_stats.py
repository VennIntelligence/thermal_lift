"""Fig 47 — Which residual statistic detects erased dots (ACL-075 mechanism detail).

Companion to fig04_dc_residual_audit.py: fig04 shows the headline detectability
result for a single statistic (window max); this figure decomposes *why* that
statistic was chosen by comparing all four candidate summary statistics of the
held-out data-consistency residual |y - A(x_hat)| computed per dot.

Per research_log/dc_residual_confidence_analysis.md (step 5): for each labelled
dot, a window of half-width = clip(ceil(2*sigma_dot)+2, 3, 8) px is taken around
its centroid in the drizzle-registered residual map, on two versions of that map:
  - win_* : the raw residual map.
  - bs_*  : the same map after subtracting a Gaussian background estimate
            (sigma_bg = 8 px), i.e. a locally background-subtracted residual.
For each version, both the window max and window mean are recorded (four
statistics total: win_max, win_mean, bs_max, bs_mean). AUC = Mann-Whitney
erased-vs-kept (and erased-vs-null) computed per arm.

Message: max statistics beat mean statistics at every arm, and window max (raw,
no background subtraction) is the single best statistic overall. The erased-dot
signal is a point-like local residual peak, not a broad elevation of the window
mean — consistent with the physical picture of a small missing point source
leaving a localized footprint after the forward operator, not a diffuse
regional shift.

Data: output/dc_residual_confidence/auc_table.csv
Run:  uv run python docs/publication_figures/scripts/fig47_dc_residual_stats.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

SRC = REPO_ROOT / "output/dc_residual_confidence"
auc = pd.read_csv(SRC / "auc_table.csv")

ARMS = ["depb9v6", "depb9v9s2", "depb9v9_3k"]
ARM_COLOR = {
    "depb9v6": "#4C72B0",
    "depb9v9s2": "#C44E52",
    "depb9v9_3k": "#55A868",
}
ARM_LABEL = {
    "depb9v6": "v6",
    "depb9v9s2": "v9 seed-2",
    "depb9v9_3k": "v9 3k",
}
STATS = ["win_max", "win_mean", "bs_max", "bs_mean"]
STAT_LABEL = {
    "win_max": "raw\nmax",
    "win_mean": "raw\nmean",
    "bs_max": "bg-sub\nmax",
    "bs_mean": "bg-sub\nmean",
}


def _grouped_bars(ax, value_col, ylabel, title):
    n_stats = len(STATS)
    n_arms = len(ARMS)
    group_w = 0.78
    bar_w = group_w / n_arms
    x = np.arange(n_stats)

    bars_out = []
    for j, arm in enumerate(ARMS):
        offs = (j - (n_arms - 1) / 2) * bar_w
        vals = [
            auc[(auc["stat"] == s) & (auc["arm"] == arm)][value_col].iloc[0]
            for s in STATS
        ]
        b = ax.bar(
            x + offs, vals, width=bar_w * 0.92,
            color=ARM_COLOR[arm], zorder=3, label=ARM_LABEL[arm],
        )
        bars_out.append(b)

    ax.axhline(0.5, color="#666666", ls="--", lw=0.9, zorder=2)
    ax.annotate(
        "chance", (1.0, 0.5), xycoords=("data", "data"),
        xytext=(4, 4), textcoords="offset points", fontsize=7,
        color="#666666", ha="left", va="bottom",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([STAT_LABEL[s] for s in STATS], fontsize=7.5)
    ax.set_ylim(0.45, 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)
    return bars_out


fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.1))

bars_a = _grouped_bars(
    ax_a, "auc_erased_vs_kept", "Mann-Whitney AUC (erased vs. kept)",
    "(a) Erased vs. kept dots",
)
_grouped_bars(
    ax_b, "auc_erased_vs_null", "Mann-Whitney AUC (erased vs. null)",
    "(b) Erased vs. background null",
)

fig.suptitle(
    "Local peak beats window mean: max statistics detect erased dots best",
    fontsize=11, fontweight="bold", x=0.02, ha="left",
)
fig.legend(
    bars_a, [ARM_LABEL[a] for a in ARMS], loc="lower center",
    bbox_to_anchor=(0.5, -0.06), ncol=3, fontsize=8, frameon=False,
)

paths = save_fig(fig, "fig47_dc_residual_stats")
print("\n".join(str(p) for p in paths))
