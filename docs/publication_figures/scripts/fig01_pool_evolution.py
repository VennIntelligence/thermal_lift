"""Fig 01 — Dot-fidelity saga across training-pool generations.

Three panels over the pool axis (v6 → v7 → v8 → v9 → v9-3k), 9-bin recipe arms:
(a) isolated erased % (broken y-axis: v7 pathology at 43% vs healthy 0-5% band),
(b) ALL retention, (c) real cross-FRC @30µm with the TGV reference.

Data: data/pool_evolution.csv (transcribed from ACL-066/070/071/072/074).
Run:  uv run python docs/publication_figures/scripts/fig01_pool_evolution.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import (
    DATA_DIR,
    METHOD_PALETTE,
    REF_LINE,
    W_DOUBLE,
    save_fig,
    setup_academic_style,
)

setup_academic_style()

df = pd.read_csv(DATA_DIR / "pool_evolution.csv", comment="#")

POOL_LABELS = {
    "v6": "v6\n(5k)",
    "v7": "v7\n(5k)",
    "v8": "v8\n(5k)",
    "v9": "v9\n(5k)",
    "v9_3k": "v9\n(3k)",
}
# one-line recipe tags (exact knob values belong in the caption)
KNOBS = {
    "v6": "legacy",
    "v7": "dense,\nshallow",
    "v8": "sparse,\ndeep",
    "v9": "dense,\ndeep",
    "v9_3k": "dense,\ndeep",
}

x = np.arange(len(df))
blue = METHOD_PALETTE["primary"]
red = METHOD_PALETTE["accent_1"]
bar_colors = [red if p == "v7" else blue for p in df["pool"]]

fig = plt.figure(figsize=(W_DOUBLE, 2.9))
# Panel (a) is itself split into top/bottom axes to break the y-axis.
gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 2.2], hspace=0.06)
ax_a_top = fig.add_subplot(gs[0, 0])
ax_a = fig.add_subplot(gs[1, 0], sharex=ax_a_top)
ax_b = fig.add_subplot(gs[:, 1])
ax_c = fig.add_subplot(gs[:, 2])

# ── (a) isolated erased %, broken axis ───────────────────────────────
for ax in (ax_a_top, ax_a):
    ax.bar(x, df["erased_pct"], width=0.62, color=bar_colors, zorder=3)
ax_a_top.set_ylim(38, 47)
ax_a_top.set_yticks([40, 45])
ax_a.set_ylim(0, 8)
ax_a.set_yticks([0, 2, 4, 6, 8])
ax_a_top.spines["bottom"].set_visible(False)
ax_a.spines["top"].set_visible(False)
ax_a_top.tick_params(labelbottom=False, bottom=False)
# diagonal break marks
kw = dict(marker=[(-1, -0.5), (1, 0.5)], markersize=7, linestyle="none",
          color="k", mec="k", mew=0.8, clip_on=False)
ax_a_top.plot([0], [0], transform=ax_a_top.transAxes, **kw)
ax_a.plot([0], [1], transform=ax_a.transAxes, **kw)
for xi, v in zip(x, df["erased_pct"]):
    ax = ax_a_top if v > 8 else ax_a
    ax.annotate(f"{v:.2f}" if v < 1 else f"{v:.1f}",
                (xi, v), xytext=(0, 2), textcoords="offset points",
                ha="center", va="bottom", fontsize=7)
ax_a.set_ylabel("Isolated defects erased [%]")
ax_a_top.set_title("(a) Dot erasure", loc="left")

# ── (b) ALL retention ────────────────────────────────────────────────
ax_b.bar(x, df["retention"], width=0.62, color=bar_colors, zorder=3)
for xi, v in zip(x, df["retention"]):
    ax_b.annotate(f"{v:.3f}", (xi, v), xytext=(0, 2),
                  textcoords="offset points", ha="center", va="bottom", fontsize=7)
ax_b.set_ylim(0, 0.9)
ax_b.set_ylabel("Contrast retention (all defects)")
ax_b.set_title("(b) Dot retention", loc="left")

# ── (c) cross-FRC @30µm ──────────────────────────────────────────────
ax_c.bar(x, df["frc30"], width=0.62, color=bar_colors, zorder=3)
ax_c.axhline(0.7017, **REF_LINE, zorder=2)
for xi, v in zip(x, df["frc30"]):
    ax_c.annotate(f"{v:.3f}", (xi, v), xytext=(0, 2),
                  textcoords="offset points", ha="center", va="bottom", fontsize=7,
                  bbox=dict(fc="white", ec="none", pad=0.3), zorder=5)
ax_c.annotate("TGV $\\times$ drizzle (0.702)", (0.98, 0.7017),
              xycoords=("axes fraction", "data"), xytext=(0, 3),
              textcoords="offset points", fontsize=7, color="#222222",
              ha="right", va="bottom")
ax_c.set_ylim(0, 0.80)
ax_c.set_ylabel("Cross-FRC @ 30 $\\mu$m period")
ax_c.set_title("(c) Real-domain FRC", loc="left")

# shared x decoration: pool label + recipe knobs underneath
for ax in (ax_a, ax_b, ax_c):
    ax.set_xticks(x)
    ax.set_xticklabels([POOL_LABELS[p] for p in df["pool"]])
    ax.tick_params(axis="x", length=0)
    for xi, p in zip(x, df["pool"]):
        ax.annotate(KNOBS[p], (xi, 0), xycoords=("data", "axes fraction"),
                    xytext=(0, -30), textcoords="offset points",
                    ha="center", va="top", fontsize=6.5,
                    color=red if p == "v7" else "#555555")

paths = save_fig(fig, "fig01_pool_evolution")
print("\n".join(str(p) for p in paths))
