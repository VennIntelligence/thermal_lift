"""Fig 40 — The unresolved range-excursion anomaly of the v8/v9 generations.

Low-frequency stability (range excursion on held-out synthetic benches, log
scale) by arm. Classical TGV and the v6-era arm sit in the healthy 1.6–4.4
band; every v8/v9-generation 9-bin arm sits at 12–16 regardless of defect
density, seed, or pool size (the common factor is the deep-depth recipe
and/or seed — mechanism open, ACL-072 #5 / ACL-074); the v9 4-bin recipe
diverges catastrophically (~10³, meanDC-eta4-class pathology, ACL-062).

Data: data/range_exc_by_generation.csv (ACL-062/071/072/074).
Run:  uv run python docs/publication_figures/scripts/fig40_range_exc_anomaly.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import DATA_DIR, W_1P5, save_fig, setup_academic_style

setup_academic_style()

df = pd.read_csv(DATA_DIR / "range_exc_by_generation.csv", comment="#")

ORDER = ["tgv_oracle", "depb9v6", "depb9v8_9bin", "depb9v8_bin4",
         "depb9v9_9bin", "depb9v9s2", "depb9v9_3k", "depb9v9_bin4"]
LABEL = {
    "tgv_oracle": "TGV (oracle)",
    "depb9v6": "v6 9-bin",
    "depb9v8_9bin": "v8 9-bin",
    "depb9v8_bin4": "v8 4-bin",
    "depb9v9_9bin": "v9 9-bin",
    "depb9v9s2": "v9 seed-2",
    "depb9v9_3k": "v9 3k",
    "depb9v9_bin4": "v9 4-bin",
}
COLOR = {
    "classical": "#333333", "v6": "#4C72B0",
    "v8": "#8172B2", "v9": "#C44E52",
}
BENCH_MARKER = {"v6b": "o", "v8b": "s"}

fig, ax = plt.subplots(figsize=(W_1P5, 3.2))

# healthy band
ax.axhspan(1.6, 4.4, color="#55A868", alpha=0.10, zorder=0)
ax.annotate("healthy band 1.6–4.4 (v6/v7 era)", (0.98, 1.9),
            xycoords=("axes fraction", "data"), fontsize=6.5,
            color="#55A868", ha="right", va="center")

x = np.arange(len(ORDER))
for i, arm in enumerate(ORDER):
    sub = df[df["arm"] == arm]
    gen = sub["generation"].iloc[0]
    for _, r in sub.iterrows():
        dx = -0.10 if r["bench"] == "v6b" else 0.10  # avoid marker overlap
        ax.scatter(i + dx, r["range_exc"], color=COLOR[gen],
                   marker=BENCH_MARKER[r["bench"]], s=30, zorder=3)

ax.annotate("catastrophic\n(dense small dots $\\times$ 4-bin)",
            (7, 1233), xytext=(-8, -2), textcoords="offset points",
            fontsize=6.5, color="#C44E52", ha="right", va="center")
ax.annotate("v8/v9 generation: 12–16\nregardless of density/seed/size",
            (4.5, 13.5), xytext=(0, 22), textcoords="offset points",
            fontsize=6.5, color="#555555", ha="center",
            bbox=dict(fc="white", ec="none", pad=0.3), zorder=5)

ax.set_yscale("log")
ax.set_ylim(1.2, 3000)
ax.set_xticks(x)
ax.set_xticklabels([LABEL[a] for a in ORDER], rotation=30, ha="right", fontsize=7)
ax.set_ylabel("Range excursion [$^\\circ$C] (log)")
ax.set_title("Low-frequency stability across pool generations", loc="left")

# bench marker legend
h = [plt.Line2D([], [], color="#666666", marker="o", ls="none", label="v6 bench48"),
     plt.Line2D([], [], color="#666666", marker="s", ls="none", label="v8 bench48")]
ax.legend(handles=h, loc="upper left", fontsize=7)

paths = save_fig(fig, "fig40_range_exc_anomaly")
print("\n".join(str(p) for p in paths))
