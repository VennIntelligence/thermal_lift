"""Fig 29 — Grid-offset probe across eras: the instrument stays calibrated.

The stage0h probe (ACL-049) measured the +0.5 px corner-convention offset on
neural×classical pairs (0.6–0.8 HR px; classical pairs ~0.03). After the
correction became the pipeline default, every later era's probe measures
only residual offsets ≤ 0.10 HR px (~0.05 LR px, the expected DR floor from
ACL-049 #5) — across stage0j (v14), stage A/B (v19), D-E (de_pb9), and the
v21 arms (meanDC, depb9v6). One log-scale dot plot, grouped by era.

Data: remote_inbox/20260713_dotprobe/offset_probe_summary_{stage0h,stage0j,
      stageAB,stageDE,v21}.csv.
Run:  uv run python docs/publication_figures/scripts/fig29_offset_probe_stability.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import REPO_ROOT, W_1P5, save_fig, setup_academic_style

setup_academic_style()

SRC = REPO_ROOT / "remote_inbox/20260713_dotprobe"

GROUPS = [
    ("Stage 0h probe\n(pre-correction)", "offset_probe_summary_stage0h.csv", {
        "v11_vs_drz": "V11 $\\times$ drz",
        "c_nodr_vs_drz": "C-noDR $\\times$ drz",
        "d_dr01_vs_drz": "D-DR0.1 $\\times$ drz",
        "v11_vs_tgv": "V11 $\\times$ TGV",
        "tgv_vs_drz": "TGV $\\times$ drz",
        "maptv_vs_drz": "MAP-TV $\\times$ drz",
    }),
    ("Stage 0j (v14)", "offset_probe_summary_stage0j.csv", None),
    ("Stage A/B (v19)", "offset_probe_summary_stageAB.csv", None),
    ("Stage D-E (de_pb9)", "offset_probe_summary_stageDE.csv", None),
    ("v21 era (meanDC,\ndepb9v6)", "offset_probe_summary_v21.csv", None),
]
CLASSICAL = {"tgv_vs_drz", "maptv_vs_drz"}

rows = []  # (y, label, value, color)
y = 0
group_bounds = []
for gname, fname, label_map in GROUPS:
    df = pd.read_csv(SRC / fname)
    y_start = y
    for _, r in df.iterrows():
        pair = r["pair"]
        label = (label_map or {}).get(pair, pair.replace("_", " "))
        if pair in CLASSICAL:
            color = "#888888"
        elif label_map is not None:
            color = "#C44E52"  # pre-correction neural pairs (the artifact)
        else:
            color = "#4C72B0"  # post-correction residuals
        rows.append((y, label, float(r["offset_norm_hr_px"]), color))
        y += 1
    group_bounds.append((gname, y_start, y))
    y += 0.8  # gap between groups

fig, ax = plt.subplots(figsize=(W_1P5, 4.2))

for yi, label, v, color in rows:
    ax.plot([0.008, v], [yi, yi], color="#dddddd", lw=0.7, zorder=1)
    ax.scatter(v, yi, s=26, color=color, zorder=3)
ax.set_yticks([r[0] for r in rows])
ax.set_yticklabels([r[1] for r in rows], fontsize=6.8)
ax.invert_yaxis()

ax.axvline(np.hypot(0.5, 0.5), color="#C44E52", ls="--", lw=0.9, zorder=2)
ax.annotate("+0.5 px corner\nconvention (0.71)", (np.hypot(0.5, 0.5), 0.02),
            xycoords=("data", "axes fraction"),
            fontsize=6.5, color="#C44E52",
            ha="center", va="bottom",
            bbox=dict(fc="white", ec="none", pad=0.2), zorder=5)
ax.axvline(0.1, color="#4C72B0", ls=":", lw=0.9, zorder=2)
ax.annotate("0.1 HR px\nresidual band", (0.155, 0.02),
            xycoords=("data", "axes fraction"),
            fontsize=6.5, color="#4C72B0",
            ha="center", va="bottom",
            bbox=dict(fc="white", ec="none", pad=0.2), zorder=5)

# group labels on the right margin
for gname, y0, y1 in group_bounds:
    ax.annotate(gname, (1.02, (y0 + y1 - 1) / 2), xycoords=("axes fraction", "data"),
                fontsize=6.5, color="#555555", ha="left", va="center",
                annotation_clip=False)

ax.set_xscale("log")
ax.set_xlim(0.008, 1.2)
ax.set_xticks([0.01, 0.03, 0.1, 0.3, 0.71])
ax.set_xticklabels(["0.01", "0.03", "0.1", "0.3", "0.71"])
ax.minorticks_off()
ax.set_xlabel("Measured grid offset $\\|\\Delta\\|$ [HR px] (log)")
ax.set_title("Offset probe across eras", loc="left")

paths = save_fig(fig, "fig29_offset_probe_stability")
print("\n".join(str(p) for p in paths))
