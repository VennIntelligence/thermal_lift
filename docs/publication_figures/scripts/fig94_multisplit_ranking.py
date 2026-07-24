"""Fig 94 — Champion ranking is split-choice robust (ACL-077).

(a) Real cross-FRC@30µm for the four champion-relevant arms under three
independent phase-stratified splits (seeds 42/123/456). The ordering
TGV > v6 > v9-generation holds in every split; absolute values move by
~0.01-0.03 with the split, so single-split differences of that size should
never have been read as meaningful — but the champion ranking never flips.

(b) The full symmetrized cross-FRC curves for all three splits (one thin
line per seed, fig06 axis conventions: log-period axis, 25-40 µm
genuine-gain band shaded, sub-20 µm detector-aperture zone hatched).
Per-arm curve families are tight everywhere above ~24 µm: split choice
perturbs the curve by less than the arm-to-arm separation, i.e. the
ranking verdict is curve-level, not a single-frequency accident.

Data: remote_inbox/20260713_content2ms/ms_verdict.csv (ACL-077) and
remote_inbox/20260713_content2ms/ms_curves/lb_seed{42,123,456}/
frc_curves_long.csv (fetched from 5090 output/stage2p5_multisplit_v2,
completed 2026-07-13).
Run:  uv run python docs/publication_figures/scripts/fig94_multisplit_ranking.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

SRC = REPO_ROOT / "remote_inbox/20260713_content2ms"
df = pd.read_csv(SRC / "ms_verdict.csv")

ARMS = {
    "tgv_x_drz": dict(color="#333333", marker="s", label="TGV"),
    "depb9v6_x_drz": dict(color="#4C72B0", marker="o", label="v6 9-bin"),
    "depb9v9_3k_x_drz": dict(color="#55A868", marker="P", label="v9 3k"),
    "depb9v9_9bin_x_drz": dict(color="#C44E52", marker="v", label="v9 9-bin"),
}
SEEDS = [42, 123, 456]
x = np.arange(len(SEEDS))

fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(W_DOUBLE, 2.9), width_ratios=[1.0, 1.55])

# ── (a) FRC@30um vs split seed ───────────────────────────────────────
for arm, st in ARMS.items():
    y = [df[(df["seed"] == s) & (df["arm"] == arm)]["frc_at_30um"].iloc[0]
         for s in SEEDS]
    ax_a.plot(x, y, color=st["color"], marker=st["marker"], markersize=4.5,
              lw=1.2, label=st["label"], clip_on=False)

ax_a.set_xticks(x)
ax_a.set_xticklabels([f"seed {s}" for s in SEEDS], fontsize=8)
ax_a.set_xlabel("Phase-stratified split")
ax_a.set_ylabel("Cross-FRC @ 30 $\\mu$m")
ax_a.set_ylim(0.60, 0.73)
ax_a.set_title("(a) Ranking holds across splits", loc="left")
ax_a.legend(loc="lower right", fontsize=6.5, ncol=2, columnspacing=0.8)

# ── (b) full curve families per seed (fig06 conventions) ─────────────
# curves clipped at the 20 um aperture zero: below it is audit-only and the
# ring statistics there produce +/-0.9 spikes that would dominate the panel
PMIN, PMAX, PCLIP = 15, 160, 20.2
for seed in SEEDS:
    c = pd.read_csv(SRC / f"ms_curves/lb_seed{seed}/frc_curves_long.csv")
    for arm, st in ARMS.items():
        # bare arm name = symmetrized mean; __xa_yb/__xb_ya are the lanes
        d = c[(c["method"] == arm) & (c["period_um"] >= PCLIP)
              & (c["period_um"] <= PMAX)].sort_values(
                  "period_um", ascending=False)
        ax_b.plot(d["period_um"], d["frc"], color=st["color"], lw=0.8,
                  alpha=0.75)

ax_b.axvspan(25, 40, color="#4C72B0", alpha=0.08, zorder=0)
ax_b.annotate("genuine-gain\nband 25--40 $\\mu$m", (np.sqrt(25 * 40), 0.08),
              ha="center", va="bottom", fontsize=6.5, color="#4C72B0")
ax_b.axvspan(PMIN, 20, color="#888888", alpha=0.12, zorder=0, hatch="///",
             lw=0)
ax_b.annotate("aperture zero\n(20 $\\mu$m)", (17.3, 0.62), ha="center",
              va="center", fontsize=6, color="#555555")
ax_b.axvline(30, color="#999999", lw=0.6, ls=":", zorder=0)
ax_b.annotate("30 $\\mu$m", (30, 1.005), ha="center", va="bottom",
              fontsize=6, color="#777777")

ax_b.set_xscale("log")
ax_b.set_xlim(PMAX, PMIN)
ticks = [150, 100, 80, 60, 40, 30, 24, 20, 16]
ax_b.set_xticks(ticks)
ax_b.set_xticklabels([str(t) for t in ticks])
ax_b.set_xlabel("Period [$\\mu$m]")
ax_b.set_ylabel("Cross-method FRC")
ax_b.set_ylim(-0.05, 1.02)
ax_b.axhline(0, color="#cccccc", lw=0.6, zorder=0)
ax_b.set_title("(b) Curve families, 3 splits per arm", loc="left")

paths = save_fig(fig, "fig94_multisplit_ranking")
print("\n".join(str(p) for p in paths))
