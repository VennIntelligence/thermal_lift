"""Fig 17 — v21 checkpoint-convergence sweep: real-domain FRC maturity and
stability, ours vs the meanDC baseline.

Two arms are checkpoint-swept every 5k steps to 30k on the real held-out
session (N=48 pairs per checkpoint): `depb9v6` (ours, v6 pool recipe) and
`meanDC` (mean-of-DC-channels baseline). Panel (a) tracks the 25-40 um
cross-FRC band mean, the closest available proxy to the TGV FRC@30um
reference point (dashed) since the eval table does not expose a
single-frequency FRC@30um column. Panel (b) tracks range_excursion (log
scale) as a stability/divergence indicator alongside FRC.

Finding: `depb9v6` matures monotonically and plateaus by step 20k
(band-FRC 0.817 -> 0.830 from 20k to 30k, <2% drift) with range_excursion
pinned near 2-2.7 throughout -- a converged, stable checkpoint. `meanDC`
looks competitive through 10k (FRC 0.75, comparable range_excursion) but
then diverges catastrophically from 15k onward: range_excursion explodes
from ~1.4 to >10,000 and band-FRC collapses to 0.21 by 30k. This is a
training instability, not slow convergence -- meanDC should not be picked
past ~10k steps, and its early-step FRC parity with depb9v6 is not a
reliable predictor of late-step behavior.

Caveats:
  - frc_band_mean_25_40 is a 15-um-wide band mean, not the single-point
    FRC@30um reported elsewhere (e.g. TGV 0.7017 @30um); the dashed
    reference line is plotted for visual anchoring only and is not a
    like-for-like value.
  - meanDC's 15k-30k range_excursion / fullband_rmse / mean_offset values
    are extreme outliers (up to ~1e4) consistent with divergence rather
    than a plausible physical reconstruction; treat those checkpoints as
    failed runs, not as data points for model comparison.

Data: output/v21_eval/v21_convergence_table.csv
Run:  uv run python docs/publication_figures/scripts/fig17_v21_convergence.py
"""

import matplotlib.pyplot as plt
import pandas as pd

from pubfig_style import (
    REF_LINE,
    REPO_ROOT,
    W_DOUBLE,
    save_fig,
    setup_academic_style,
    ylabel_with_unit,
)

setup_academic_style()

SRC = REPO_ROOT / "output/v21_eval/v21_convergence_table.csv"
df = pd.read_csv(SRC)

TGV_FRC_30UM = 0.7017

ARMS = {
    "depb9v6": dict(color="#4C72B0", marker="o", ls="-", label="Ours (v6 pool)"),
    "meanDC":  dict(color="#C44E52", marker="s", ls="-", label="meanDC (baseline)"),
}

fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.1))
ax1, ax2 = axes

# (a) band-FRC vs step
for prefix, st in ARMS.items():
    d = df[df["prefix"] == prefix].sort_values("step")
    ax1.plot(d["step"] / 1000, d["frc_band_mean_25_40_mean"],
              color=st["color"], marker=st["marker"], ls=st["ls"],
              label=st["label"])
ax1.axhline(TGV_FRC_30UM, **REF_LINE, label=f"TGV FRC@30$\\mu$m ({TGV_FRC_30UM:.4f})")
ax1.set_title("(a) Cross-FRC band mean [25-40 $\\mu$m]", loc="left")
ax1.set_xlabel("Training step [$\\times 10^3$]")
ax1.set_ylabel("Band-FRC mean (N=48)")
ax1.set_ylim(0.15, 0.9)
ax1.legend(loc="lower left", fontsize=6.5)

# (b) range_excursion vs step, log scale (stability indicator)
for prefix, st in ARMS.items():
    d = df[df["prefix"] == prefix].sort_values("step")
    ax2.plot(d["step"] / 1000, d["range_excursion_mean"],
              color=st["color"], marker=st["marker"], ls=st["ls"],
              label=st["label"])
ax2.set_yscale("log")
ax2.set_title("(b) Range excursion (stability)", loc="left")
ax2.set_xlabel("Training step [$\\times 10^3$]")
ax2.set_ylabel(ylabel_with_unit("Range excursion (log scale)", "$^\\circ$C"))
ax2.legend(loc="upper left", fontsize=7)

paths = save_fig(fig, "fig17_v21_convergence")
print("\n".join(str(p) for p in paths))
