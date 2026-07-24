"""Fig 04 — DC-residual "self-doubt gauge" (ACL-075).

Erased dots leave a detectable footprint in the held-out data-consistency
residual |y − A(x̂)| (112 held-out frames, outside the reconstruction's DC
subset). (a) Mann-Whitney AUC per arm — detection strengthens monotonically
as the arm's dot fidelity improves. (b) Residual distributions (window max)
by dot class for each arm.

Data: output/dc_residual_confidence/{auc_table,per_dot_residual_stats}.csv.
Run:  uv run python docs/publication_figures/scripts/fig04_dc_residual_audit.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import METHOD_PALETTE, REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

SRC = REPO_ROOT / "output/dc_residual_confidence"
auc = pd.read_csv(SRC / "auc_table.csv")
dots = pd.read_csv(SRC / "per_dot_residual_stats.csv")

# order arms by improving dot fidelity (isolated erased%: 4.66 → 1.55 → 0.00)
ARMS = ["depb9v6", "depb9v9s2", "depb9v9_3k"]
ARM_LABEL = {
    "depb9v6": "v6\n(isol. erased 4.66%)",
    "depb9v9s2": "v9 seed-2\n(isol. erased 1.55%)",
    "depb9v9_3k": "v9 3k\n(isol. erased 0.00%)",
}
awm = auc[auc["stat"] == "win_max"].set_index("arm").loc[ARMS]

fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(W_DOUBLE, 2.8), gridspec_kw=dict(width_ratios=[1.0, 1.5]))

# ── (a) AUC bars ─────────────────────────────────────────────────────
x = np.arange(len(ARMS))
w = 0.36
b1 = ax_a.bar(x - w / 2, awm["auc_erased_vs_kept"], width=w * 0.92,
              color=METHOD_PALETTE["primary"], zorder=3, label="erased vs. kept")
b2 = ax_a.bar(x + w / 2, awm["auc_erased_vs_null"], width=w * 0.92,
              color=METHOD_PALETTE["accent_3"], zorder=3, label="erased vs. null")
for bars in (b1, b2):
    for rect in bars:
        ax_a.annotate(f"{rect.get_height():.2f}",
                      (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                      xytext=(0, 2), textcoords="offset points",
                      ha="center", va="bottom", fontsize=7)
ax_a.axhline(0.5, color="#666666", ls="--", lw=0.9, zorder=2)
ax_a.annotate("chance", (0.44, 0.5), xycoords=("axes fraction", "data"),
              xytext=(0, -3), textcoords="offset points", fontsize=7,
              color="#666666", ha="center", va="top",
              bbox=dict(fc="white", ec="none", pad=0.2))
ax_a.set_xticks(x)
ax_a.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=7.5)
ax_a.set_ylim(0, 1.0)
ax_a.set_ylabel("Mann–Whitney AUC (window max)")
ax_a.set_title("(a) Erased-dot detectability", loc="left")
ax_a.legend(loc="upper left", fontsize=7)

# ── (b) residual distributions by class ──────────────────────────────
CLASSES = ["preserved", "blurred", "erased"]
CLASS_COLOR = {
    "preserved": METHOD_PALETTE["secondary"],
    "blurred": METHOD_PALETTE["neutral"],
    "erased": METHOD_PALETTE["accent_1"],
}
positions, data, colors = [], [], []
for i, arm in enumerate(ARMS):
    for j, cls in enumerate(CLASSES):
        vals = dots[(dots["arm"] == arm) & (dots["dot_class"] == cls)]["resid_win_max"]
        data.append(vals.to_numpy())
        positions.append(i * 4 + j)
        colors.append(CLASS_COLOR[cls])

bp = ax_b.boxplot(data, positions=positions, widths=0.72, patch_artist=True,
                  showfliers=False, medianprops=dict(color="#222222", lw=1.1),
                  whiskerprops=dict(lw=0.8), capprops=dict(lw=0.8))
for patch, c in zip(bp["boxes"], colors):
    patch.set_facecolor(c)
    patch.set_alpha(0.75)
    patch.set_edgecolor("#333333")
    patch.set_linewidth(0.6)

ax_b.set_xticks([i * 4 + 1 for i in range(len(ARMS))])
ax_b.set_xticklabels([ARM_LABEL[a] for a in ARMS], fontsize=7.5)
ax_b.set_ylabel("Held-out DC residual, window max [a.u.]")
ax_b.set_title("(b) Residual by dot class", loc="left")
handles = [plt.Rectangle((0, 0), 1, 1, fc=CLASS_COLOR[c], alpha=0.75,
                         ec="#333333", lw=0.6) for c in CLASSES]
ax_b.legend(handles, CLASSES, loc="upper left", fontsize=7, ncol=1)

paths = save_fig(fig, "fig04_dc_residual_audit")
print("\n".join(str(p) for p in paths))
