"""Fig 31 — Drizzle split-half FRC is split-choice robust (ACL-048-era instrument control).

Split-half FRC of the drizzle reconstruction under two different split
conventions (odd/even frames vs. phase-stratified seed-42 frames). Both
curves track each other closely across the full period range, showing
that the drizzle reference's self-consistency is not an artifact of the
odd/even split choice — a precondition for trusting drizzle as the
independent cross-method reference used elsewhere (e.g. fig06). Cutoffs
(half-bit criterion) are read from the method-summary CSV and annotated
in the legend. Hatched region below 20 um is the detector aperture zero
(audit-only, curves clipped there), matching fig06's convention.

Data: remote_inbox/20260705_stage0g/{drizzle_phase_stratified_seed42_frc_curve,
      drizzle_odd_even_odd_even_frc_curve,task1_3_drizzle_frc_method_summary}.csv
Run:  uv run python docs/publication_figures/scripts/fig31_drizzle_split_controls.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import REPO_ROOT, W_1P5, save_fig, setup_academic_style

setup_academic_style()

SRC = REPO_ROOT / "remote_inbox/20260705_stage0g"
summary = pd.read_csv(SRC / "task1_3_drizzle_frc_method_summary.csv")

VARIANTS = {
    "phase_stratified": dict(
        csv="drizzle_phase_stratified_seed42_frc_curve.csv",
        color="#4C72B0", ls="-",
        label_base="Phase-stratified (seed 42)",
    ),
    "odd_even": dict(
        csv="drizzle_odd_even_odd_even_frc_curve.csv",
        color="#C44E52", ls="--",
        label_base="Odd/even",
    ),
}

fig, ax = plt.subplots(figsize=(W_1P5, 3.2))

PMIN, PMAX = 15.0, 150.0
PCLIP = 20.0  # curves stop at the detector aperture zero; below is audit-only

for split_mode, st in VARIANTS.items():
    d = pd.read_csv(SRC / st["csv"])
    row = summary[summary["split_mode"] == split_mode].iloc[0]
    cutoff = row["frc_cutoff_period_um_half_bit"]
    label = f"{st['label_base']} (cutoff {cutoff:.1f} $\\mu$m)"

    dp = d[(d["period_um"] >= PCLIP) & (d["period_um"] <= PMAX)]
    dp = dp.sort_values("period_um", ascending=False)
    ax.plot(dp["period_um"], dp["frc"], color=st["color"], ls=st["ls"],
             lw=1.3, label=label)

# half-bit threshold (shared reference curve, same for both splits at a given ring)
d0 = pd.read_csv(SRC / VARIANTS["phase_stratified"]["csv"])
d0 = d0[(d0["period_um"] >= PCLIP) & (d0["period_um"] <= PMAX)]
d0 = d0.sort_values("period_um", ascending=False)
ax.plot(d0["period_um"], d0["threshold_half_bit"], color="#999999", ls=":",
        lw=1.0, label="Half-bit criterion")

# aperture zero: below 20 um audit-only
ax.axvspan(PMIN, 20, color="#888888", alpha=0.12, zorder=0, hatch="///", lw=0)
ax.annotate("detector aperture\nzero (20 $\\mu$m):\naudit only", (17.2, 0.60),
            ha="center", va="center", fontsize=6.5, color="#555555")

ax.set_xscale("log")
ax.set_xlim(PMAX, PMIN)  # high period (low frequency) on the left
ticks = [150, 100, 80, 60, 40, 30, 24, 20, 16]
ax.set_xticks(ticks)
ax.set_xticklabels([str(t) for t in ticks])
ax.set_xlabel("Period [$\\mu$m]")
ax.set_ylabel("Split-half FRC")
ax.set_ylim(-0.05, 1.02)
ax.axhline(0, color="#cccccc", lw=0.6, zorder=0)
ax.legend(loc="center left", bbox_to_anchor=(0.01, 0.40), fontsize=7)
ax.set_title("Drizzle split-half FRC: robust to split choice", loc="left")

paths = save_fig(fig, "fig31_drizzle_split_controls")
print("\n".join(str(p) for p in paths))
