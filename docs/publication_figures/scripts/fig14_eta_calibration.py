"""Fig 14 — DC-weight (η) calibration: the one knob that moved (ACL-051/052/053).

(a) Real cross-FRC @30µm vs η (log scale): monotone improvement toward low η
with a peak plateau at η ≈ 0.0625–0.125 (η* = 0.09); the historical default
η = 0.5 was never calibrated. (b) Synthetic PSNR peaks at η = 0.25 while the
real optimum sits at 0.09 — the synth/real decoupling signature (ACL-032).
(c) Scoreboard: capacity/evidence routes (unroll depth, DC frames, inference
budget, per-frame fusion) all land in the v14 ± 0.015 plateau; only η
recalibration (+ band loss) moved the needle, closing ~35% of the TGV gap;
training the champion recipe to 50k regressed (synth↑/real↓ again).

Data: data/eta_sweep.csv, data/plateau_scoreboard.csv (ACL-051/052/053).
Run:  uv run python docs/publication_figures/scripts/fig14_eta_calibration.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import DATA_DIR, REF_LINE, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

eta = pd.read_csv(DATA_DIR / "eta_sweep.csv", comment="#")
board = pd.read_csv(DATA_DIR / "plateau_scoreboard.csv", comment="#")

fig, (ax_a, ax_b, ax_c) = plt.subplots(
    1, 3, figsize=(W_DOUBLE, 2.9), gridspec_kw=dict(width_ratios=[1.1, 1.0, 1.35]))

# ── (a) FRC vs eta ───────────────────────────────────────────────────
ax_a.plot(eta["eta"], eta["frc30"], color="#4C72B0", marker="o",
          markersize=4.5, zorder=3, clip_on=False)
star = eta.loc[eta["eta"] == 0.09]
ax_a.scatter(star["eta"], star["frc30"], marker="*", s=110, color="#C44E52",
             zorder=4, clip_on=False)
ax_a.annotate("$\\eta^*=0.09$", (0.09, 0.6675), xytext=(0, 7),
              textcoords="offset points", ha="center", fontsize=7,
              color="#C44E52")
ax_a.annotate("old default\n$\\eta=0.5$", (0.5, 0.649), xytext=(0, -22),
              textcoords="offset points", ha="center", fontsize=7,
              color="#555555",
              arrowprops=dict(arrowstyle="-", color="#999999", lw=0.6))
ax_a.axhline(0.7017, **REF_LINE)
ax_a.annotate("TGV 0.702", (0.98, 0.7017), xycoords=("axes fraction", "data"),
              xytext=(0, -3), textcoords="offset points", fontsize=7,
              ha="right", va="top", color="#222222")
ax_a.set_xscale("log")
ax_a.set_xticks([0.0625, 0.125, 0.25, 0.5, 1.0, 2.0])
ax_a.set_xticklabels(["0.0625", "0.125", "0.25", "0.5", "1", "2"], fontsize=7)
ax_a.minorticks_off()
ax_a.set_xlabel("DC weight $\\eta$")
ax_a.set_ylabel("Cross-FRC @ 30 $\\mu$m")
ax_a.set_ylim(0.60, 0.71)
ax_a.set_title("(a) Real domain", loc="left")

# ── (b) synth PSNR vs eta ────────────────────────────────────────────
ax_b.plot(eta["eta"], eta["psnr"], color="#55A868", marker="s",
          markersize=4, zorder=3, clip_on=False)
ax_b.axvline(0.09, color="#C44E52", ls=":", lw=1.0)
ax_b.annotate("real optimum\n$\\eta^*=0.09$", (0.09, 0.04),
              xycoords=("data", "axes fraction"), xytext=(4, 0),
              textcoords="offset points", fontsize=6.5, color="#C44E52",
              ha="left", va="bottom")
ax_b.annotate("synth optimum\n$\\eta=0.25$", (0.25, 33.56), xytext=(0, 6),
              textcoords="offset points", ha="center", fontsize=6.5,
              color="#55A868")
ax_b.set_xscale("log")
ax_b.set_xticks([0.0625, 0.25, 1.0])
ax_b.set_xticklabels(["0.0625", "0.25", "1"], fontsize=7)
ax_b.minorticks_off()
ax_b.set_xlabel("DC weight $\\eta$")
ax_b.set_ylabel("Synthetic PSNR [dB]")
ax_b.set_ylim(21, 35.5)
ax_b.set_title("(b) Synth/real decoupling", loc="left")

# ── (c) scoreboard ───────────────────────────────────────────────────
GROUP_COLOR = {"baseline": "#888888", "capacity": "#937860",
               "eta": "#4C72B0", "regression": "#C44E52"}
y = np.arange(len(board))[::-1]
ax_c.barh(y, board["frc30"], height=0.62,
          color=[GROUP_COLOR[g] for g in board["group"]], zorder=3)
for yi, v in zip(y, board["frc30"]):
    ax_c.annotate(f"{v:.3f}", (v, yi), xytext=(2, 0),
                  textcoords="offset points", va="center", fontsize=6.5,
                  bbox=dict(fc="white", ec="none", pad=0.15), zorder=5)
ax_c.axvline(0.7017, **REF_LINE)
ax_c.annotate("TGV\n0.702", (0.7017, 0.985), xycoords=("data", "axes fraction"),
              xytext=(-3, 0), textcoords="offset points", fontsize=6.5,
              ha="right", va="top", color="#222222")
ax_c.set_yticks(y)
ax_c.set_yticklabels(board["arm"], fontsize=6.8)
# encode groups via tick-label colour instead of a legend (no space for one)
for tick, g in zip(ax_c.get_yticklabels(), board["group"]):
    tick.set_color(GROUP_COLOR[g])
ax_c.set_xlabel("Cross-FRC @ 30 $\\mu$m")
ax_c.set_xlim(0.55, 0.72)
ax_c.set_title("(c) Only $\\eta$ moved the needle", loc="left")

paths = save_fig(fig, "fig14_eta_calibration")
print("\n".join(str(p) for p in paths))
