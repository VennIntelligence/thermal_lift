"""Fig 02 — Champion selection as a Pareto problem: real-domain FRC vs dot fidelity.

x = isolated-defect erasure rate (lower is better), y = cross-FRC @30µm (higher
is better). Neural arms only (classical TGV has no dot probe → horizontal
reference). The staircase marks the non-dominated front; OOD robustness
(ACL-076) is tagged where measured.

Data: data/champion_arms.csv. Run:
uv run python docs/publication_figures/scripts/fig02_champion_pareto.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import DATA_DIR, REF_LINE, W_1P5, save_fig, setup_academic_style

setup_academic_style()

df = pd.read_csv(DATA_DIR / "champion_arms.csv", comment="#")
neural = df[df["kind"] == "neural"].copy()
tgv30 = float(df.loc[df["arm"] == "tgv", "frc30"].iloc[0])

# label, color, marker per arm; offsets tuned to avoid collisions
# dx/dy in points; arrow=True draws a thin leader line (for the dense cluster)
STYLE = {
    "depb9v6":      dict(c="#4C72B0", m="o", label="v6 9-bin",  dx=10, dy=6,   ha="left",  arrow=False),
    "depb9v8_9bin": dict(c="#8172B2", m="X", label="v8 9-bin",  dx=10, dy=-4,  ha="left",  arrow=False),
    "depb9v8_bin4": dict(c="#8172B2", m="P", label="v8 4-bin",  dx=4,  dy=-14, ha="center", arrow=False),
    "depb9v9_9bin": dict(c="#C44E52", m="v", label="v9 9-bin",  dx=26, dy=16,  ha="left",  arrow=True),
    "depb9v9_bin4": dict(c="#C44E52", m="^", label="v9 4-bin",  dx=-4, dy=8,   ha="center", arrow=False),
    "depb9v9s2":    dict(c="#C44E52", m="D", label="v9 seed-2", dx=26, dy=-16, ha="left",  arrow=True),
    "depb9v9_3k":   dict(c="#55A868", m="s", label="v9 3k",     dx=-6, dy=12,  ha="right", arrow=True),
}

fig, ax = plt.subplots(figsize=(W_1P5, 3.4))

# Pareto front: minimize erased_pct, maximize frc30
pts = neural.sort_values("erased_pct")[["erased_pct", "frc30", "arm"]].to_numpy()
front = []
best = -np.inf
for ex, fr, arm in pts:
    if fr > best:
        front.append((ex, fr))
        best = fr
front = np.array(front, dtype=float)
ax.step(front[:, 0], front[:, 1], where="post", color="#bbbbbb", lw=1.0,
        zorder=1, label="Pareto front")

for _, r in neural.iterrows():
    s = STYLE[r["arm"]]
    ax.scatter(r["erased_pct"], r["frc30"], c=s["c"], marker=s["m"], s=42,
               zorder=3, edgecolors="white", linewidths=0.5)
    tag = s["label"]
    if not pd.isna(r["ood_wins"]):
        tag += f"\nOOD {int(r['ood_wins'])}/9"
    arrowprops = (dict(arrowstyle="-", color="#999999", lw=0.6,
                       shrinkA=1, shrinkB=3) if s["arrow"] else None)
    ax.annotate(tag, (r["erased_pct"], r["frc30"]),
                xytext=(s["dx"], s["dy"]), textcoords="offset points",
                fontsize=7, ha=s["ha"], va="center", color="#333333",
                arrowprops=arrowprops,
                bbox=dict(fc="white", ec="none", pad=0.2) if s["arrow"] else None,
                zorder=4)

ax.axhline(tgv30, **REF_LINE, zorder=2)
ax.annotate("TGV $\\times$ drizzle FRC (dot probe n/a)", (0.98, tgv30),
            xycoords=("axes fraction", "data"), xytext=(0, 3),
            textcoords="offset points", fontsize=7, ha="right", va="bottom",
            color="#222222")

# ideal-direction cue
ax.annotate("ideal", xy=(0.085, 0.90), xycoords="axes fraction", fontsize=7.5,
            color="#666666", ha="left", va="center")
ax.annotate("", xy=(0.015, 0.975), xytext=(0.075, 0.89), xycoords="axes fraction",
            arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))

ax.set_xlabel("Isolated defects erased [%] (lower is better)")
ax.set_ylabel("Cross-FRC @ 30 $\\mu$m period")
ax.set_xlim(-2.5, 35)
ax.set_ylim(0.58, 0.72)
ax.set_title("Champion arms: fidelity–resolution trade-off", loc="left")

paths = save_fig(fig, "fig02_champion_pareto")
print("\n".join(str(p) for p in paths))
