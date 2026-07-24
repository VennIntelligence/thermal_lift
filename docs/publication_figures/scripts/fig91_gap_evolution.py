"""Fig 91 — Chasing TGV: the neural FRC@30µm record and its price.

Every notable neural arm (July 2026 campaign), chronologically, against the
bit-identical TGV reference (0.7017). Filled markers = arms with no known
pathology; open markers = arms whose FRC gain came bundled with a broken
axis (dot erasure or low-frequency divergence) or a training regression.
The story: the healthy-arm ceiling sits at ~0.67 (record 0.6705); every arm
that got closer to TGV paid on another axis, which is what turned champion
selection into the multi-axis Pareto problem of fig02.

Data: data/gap_evolution.csv (ACL-050..074 anchors).
Run:  uv run python docs/publication_figures/scripts/fig91_gap_evolution.py
"""

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd

from pubfig_style import REF_LINE, W_DOUBLE, DATA_DIR, save_fig, setup_academic_style

setup_academic_style()

df = pd.read_csv(DATA_DIR / "gap_evolution.csv", comment="#", parse_dates=["date"])
# spread same-day arms horizontally for readability
df["t"] = df["date"] + pd.to_timedelta(df.groupby("date").cumcount() * 4, unit="h")

HEALTH_STYLE = {
    "ok":      dict(fc="#4C72B0", ec="#4C72B0", label="healthy on all measured axes"),
    "dots":    dict(fc="white", ec="#C44E52", label="dot-fidelity pathology"),
    "lowfreq": dict(fc="white", ec="#8172B2", label="low-frequency divergence"),
    "regress": dict(fc="white", ec="#937860", label="50k regression"),
}

fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.4))

ax.axhline(0.7017, **REF_LINE)
ax.annotate("TGV $\\times$ drizzle 0.7017 (bit-identical in every rerun)",
            (0.99, 0.7017), xycoords=("axes fraction", "data"), xytext=(0, 2),
            textcoords="offset points", fontsize=7, ha="right", va="bottom",
            color="#222222")

# running best among healthy arms
healthy = df[df["health"] == "ok"].sort_values("t")
ax.step(healthy["t"], healthy["frc30"].cummax(), where="post",
        color="#4C72B0", lw=0.9, alpha=0.35, zorder=1)

for health, st in HEALTH_STYLE.items():
    sub = df[df["health"] == health]
    ax.scatter(sub["t"], sub["frc30"], s=40, facecolor=st["fc"],
               edgecolor=st["ec"], linewidth=1.2, zorder=3, label=st["label"])

LABEL_OFFSETS = {
    "v14 (20k)": (0, -11), "eta*=0.09": (-2, 8), "eta* x band": (0, 9),
    "champion 50k": (0, -11), "de_pb9": (18, 8), "depb9v6 9-bin": (0, -12),
    "depb9v6 4-bin": (-14, -11), "v7 9-bin": (8, 6), "v8 4-bin": (0, 8),
    "v8 9-bin": (0, -11), "v9 4-bin": (0, 8), "v9 9-bin": (12, 5),
    "v9 3k": (10, -8),
}
for _, r in df.iterrows():
    dx, dy = LABEL_OFFSETS.get(r["arm"], (0, 8))
    ax.annotate(r["arm"], (r["t"], r["frc30"]), xytext=(dx, dy),
                textcoords="offset points", ha="center", fontsize=6.3,
                color="#444444")

ax.set_ylim(0.585, 0.715)
ax.set_ylabel("Cross-FRC @ 30 $\\mu$m period")
ax.set_xlabel("Verdict date (2026)")
ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
ax.set_title("Neural arms vs the classical reference over the campaign", loc="left")
ax.legend(loc="lower left", fontsize=6.5)

paths = save_fig(fig, "fig91_gap_evolution")
print("\n".join(str(p) for p in paths))
