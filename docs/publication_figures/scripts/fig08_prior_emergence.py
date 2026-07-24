"""Fig 08 — The dot-erasure prior needs data scale to form (ACL-069/074).

(a) Step-horizon probe at 300 scenes (v7 recipe): isolated erased % stays at
exactly 0 from 4k to 24k steps with no upward trend, while the identical
recipe at 5000 scenes reaches 39.75% by 30k steps — erasure is not a
step-count effect but a pool-scale/diversity effect (300 scenes × ~10 epochs
memorizes; 5000 scenes × low repetition generalizes an "erase small dark
dots" prior). (b) Erased % vs pool size per recipe; pool scale itself is a
knob on the FRC↔dot-fidelity tension axis.

Data: remote_inbox/20260716_micro_{calib,horizon}/summary_micro.json,
      data/prior_emergence.csv (production points from ACL-062/067/072/074).
Run:  uv run python docs/publication_figures/scripts/fig08_prior_emergence.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import DATA_DIR, REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

calib = json.loads((REPO_ROOT / "remote_inbox/20260716_micro_calib/summary_micro.json").read_text())
horizon = json.loads((REPO_ROOT / "remote_inbox/20260716_micro_horizon/summary_micro.json").read_text())
pools = pd.read_csv(DATA_DIR / "prior_emergence.csv", comment="#")

# ── (a) step horizon at 300 scenes, v7 recipe ────────────────────────
steps_k, erased = [], []
for key, s in [("micro_v7end_4k", 4), ("micro_v7end", 8)]:
    steps_k.append(s)
    erased.append(calib["results"][key]["isolated_erased_pct"])
for key, s in [("v7e12k", 12), ("v7e16k", 16), ("v7e20k", 20), ("v7e24k", 24)]:
    steps_k.append(s)
    erased.append(horizon["results"][key]["isolated_erased_pct"])

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(W_DOUBLE, 2.7))

ax_a.plot(steps_k, erased, color="#4C72B0", marker="o", markersize=4.5,
          label="v7 recipe, 300 scenes", clip_on=False, zorder=3)
ax_a.axhline(39.75, color="#C44E52", ls="--", lw=1.0, zorder=2)
ax_a.annotate("same recipe, 5000 scenes, 30k steps: 39.75%",
              (0.97, 39.75), xycoords=("axes fraction", "data"),
              xytext=(0, -4), textcoords="offset points",
              fontsize=7, color="#C44E52", ha="right", va="top")
ax_a.annotate("0% erased at every probed step — no emerging trend",
              (0.5, 0.0), xycoords=("axes fraction", "data"),
              xytext=(0, 6), textcoords="offset points",
              fontsize=7, color="#4C72B0", ha="center", va="bottom")
ax_a.set_xlabel("Training step [$\\times 10^3$]")
ax_a.set_ylabel("Isolated defects erased [%]")
ax_a.set_xticks(steps_k)
ax_a.set_ylim(-1.5, 45)
ax_a.set_title("(a) 300 scenes: prior never forms", loc="left")
ax_a.legend(loc="center left", fontsize=7)

# ── (b) erased% vs pool size per recipe ──────────────────────────────
RECIPE_STYLE = {
    "v7": dict(color="#C44E52", marker="v", label="v7 recipe (dense, shallow)"),
    "v6": dict(color="#4C72B0", marker="o", label="v6 recipe (legacy)"),
    "v9": dict(color="#55A868", marker="s", label="v9 recipe (dense, deep)"),
}
for recipe, st in RECIPE_STYLE.items():
    d = pools[pools["recipe"] == recipe].sort_values("n_scenes")
    ax_b.plot(d["n_scenes"], d["erased_pct"], color=st["color"],
              marker=st["marker"], markersize=4.5, label=st["label"],
              clip_on=False, zorder=3)
# endpoint labels, hand-placed to avoid collisions
ANNOT = [
    ("v7", 5000, 39.75, (0, 6), "center"),
    ("v6", 5000, 4.66, (2, 7), "center"),
    ("v9", 5000, 1.55, (10, -2), "left"),
    ("v9", 3000, 0.00, (0, -10), "center"),
]
for recipe, n, v, (dx, dy), ha in ANNOT:
    ax_b.annotate(f"{v:.4g}%", (n, v), xytext=(dx, dy),
                  textcoords="offset points", ha=ha, va="center",
                  fontsize=6.5, color=RECIPE_STYLE[recipe]["color"])
ax_b.annotate("0%", (300, 0), xytext=(0, 8), textcoords="offset points",
              ha="center", fontsize=6.5, color="#555555")
ax_b.set_xscale("log")
ax_b.set_xticks([300, 3000, 5000])
ax_b.set_xticklabels(["300", "3000", "5000"])
ax_b.minorticks_off()
ax_b.set_xlabel("Training-pool size [scenes]")
ax_b.set_ylabel("Isolated defects erased [%]")
ax_b.set_ylim(-1.5, 45)
ax_b.set_title("(b) Erasure grows with pool scale", loc="left")
ax_b.legend(loc="upper left", fontsize=7)

paths = save_fig(fig, "fig08_prior_emergence")
print("\n".join(str(p) for p in paths))
