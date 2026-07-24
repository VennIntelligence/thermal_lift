"""Fig 21 — Per-frame shift error was the top bottleneck (ACL-047/048).

(a) Per-frame alignment corrections found by the stage0f refinement
(248 frames): scatter of (Δx, Δy) with the RMS radius — the old
contour-refined alignment carried ~0.31 px RMS per-frame error.
(b) Consequence for the information budget: EP15 M2 split-half FRC with the
old alignment cuts off at 34.07 µm; with refined alignment the authoritative
recoverable band improves to 25.45 ± 0.73 µm. Refined alignment became the
repo default (configs/alignment/stage0f_refined_alignment.csv).

Data: remote_inbox/20260704_stage0f/{t1a_best_shift_refinements,
      t0e_m2_frc_curve}.csv, remote_inbox/20260705_stage0g/
      task1_4_m2_frc_curve.csv (+ *_summary.json cutoffs).
Run:  uv run python docs/publication_figures/scripts/fig21_shift_refinement.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

t1a = pd.read_csv(REPO_ROOT / "remote_inbox/20260704_stage0f/t1a_best_shift_refinements.csv")
old_curve = pd.read_csv(REPO_ROOT / "remote_inbox/20260704_stage0f/t0e_m2_frc_curve.csv")
new_curve = pd.read_csv(REPO_ROOT / "remote_inbox/20260705_stage0g/task1_4_m2_frc_curve.csv")

fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(W_DOUBLE, 2.9), gridspec_kw=dict(width_ratios=[1, 1.5]))

# ── (a) per-frame corrections ────────────────────────────────────────
dx, dy = t1a["delta_dx_px"].to_numpy(), t1a["delta_dy_px"].to_numpy()
rms = float(np.sqrt(np.mean(dx**2 + dy**2)))
ax_a.scatter(dx, dy, s=9, color="#4C72B0", alpha=0.45, linewidths=0, zorder=3)
theta = np.linspace(0, 2 * np.pi, 200)
ax_a.plot(rms * np.cos(theta), rms * np.sin(theta), color="#C44E52", lw=1.1,
          zorder=4)
ax_a.annotate(f"RMS = {rms:.2f} LR px", (rms * 0.74, rms * 0.82),
              fontsize=7, color="#C44E52", ha="left", va="bottom",
              bbox=dict(fc="white", ec="none", pad=0.2), zorder=5)
ax_a.axhline(0, color="#cccccc", lw=0.6, zorder=1)
ax_a.axvline(0, color="#cccccc", lw=0.6, zorder=1)
ax_a.set_xlabel("Refinement $\\Delta x$ [LR px]")
ax_a.set_ylabel("Refinement $\\Delta y$ [LR px]")
ax_a.set_aspect("equal")
lim = 0.62
ax_a.set_xlim(-lim, lim)
ax_a.set_ylim(-lim, lim)
ax_a.set_title("(a) Per-frame corrections (248 frames)", loc="left")

# ── (b) M2 FRC before/after ──────────────────────────────────────────
PMIN, PMAX = 15.0, 150.0
for curve, color, ls, label, cutoff in [
    (old_curve, "#937860", "--", "Old alignment (cutoff 34.1 $\\mu$m)", 34.07),
    (new_curve, "#4C72B0", "-", "Refined alignment (cutoff 25.5 $\\pm$ 0.7 $\\mu$m)", 25.45),
]:
    d = curve[(curve["period_um"] >= PMIN) & (curve["period_um"] <= PMAX)]
    d = d.sort_values("period_um", ascending=False)
    ax_b.plot(d["period_um"], d["frc"], color=color, ls=ls, lw=1.3, label=label)
    ax_b.axvline(cutoff, color=color, ls=":", lw=0.9, zorder=1)

d0 = old_curve[(old_curve["period_um"] >= PMIN) & (old_curve["period_um"] <= PMAX)]
d0 = d0.sort_values("period_um", ascending=False)
if "threshold_half_bit" in d0.columns:
    ax_b.plot(d0["period_um"], d0["threshold_half_bit"], color="#999999",
              ls=":", lw=1.0, label="Half-bit criterion")

# below ~22µm sits at the 20µm detector aperture zero: never trusted
ax_b.axvspan(PMIN, 22, color="#888888", alpha=0.12, hatch="///", lw=0, zorder=0)
ax_b.annotate("aperture-zero\nregion:\nnot trusted", (17.5, 0.30), ha="center",
              va="center", fontsize=6.5, color="#555555",
              bbox=dict(fc="white", ec="none", pad=0.3), zorder=5)

ax_b.set_xscale("log")
ax_b.set_xlim(PMAX, PMIN)
ticks = [150, 100, 80, 60, 40, 34, 25, 20, 16]
ax_b.set_xticks(ticks)
ax_b.set_xticklabels([str(t) for t in ticks], fontsize=7)
ax_b.minorticks_off()
ax_b.set_xlabel("Period [$\\mu$m]")
ax_b.set_ylabel("Split-half FRC (M2)")
ax_b.set_ylim(-0.1, 1.02)
ax_b.axhline(0, color="#cccccc", lw=0.6, zorder=0)
leg = ax_b.legend(loc="center left", bbox_to_anchor=(0.01, 0.42), fontsize=7,
                  frameon=True, framealpha=1.0, edgecolor="none",
                  facecolor="white")
ax_b.set_title("(b) Recoverable band: 34.1 $\\rightarrow$ 25.5 $\\mu$m", loc="left")

paths = save_fig(fig, "fig21_shift_refinement")
print("\n".join(str(p) for p in paths))
