"""Fig 03 — OOD robustness verdict (ACL-076, updated ACL-079).

Top row: absolute band-FRC (25-40µm mean) of the 5 arms on the content-axis
(4 extreme pools) and noise-axis (5 extreme pools) OOD benchmarks.
Bottom row: neural − TGV(oracle) difference; bars above zero = neural wins.
depb9v6 is 9/9 positive (zero sign-flips); depb9v9_9bin loses 7/9;
depb9v9_3k loses ALL 9/9 and is the worst-OOD arm found to date (ACL-079),
with its steepest drop on stripe_x4 (Delta FRC = -0.38 vs oracle).

Data: remote_inbox/20260712_oodC/ood_degradation_summary.csv (48 scenes/pool).
Run:  uv run python docs/publication_figures/scripts/fig03_ood_robustness.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

SRC = REPO_ROOT / "remote_inbox/20260712_oodC/ood_degradation_summary.csv"
df = pd.read_csv(SRC, header=[0, 1], index_col=[0, 1, 2, 3, 4])
df.index.names = ["ood_axis", "level", "level_label", "pool", "arm"]
band = df[("frc_band_mean_25_40", "mean")].rename("frc").reset_index()

POOL_SHORT = {
    "pool_2x_ood_content_texroles": "textured\nroles",
    "pool_2x_ood_content_xlmerge": "XL-tier\nmerge",
    "pool_2x_ood_content_legacymix": "legacy\nmix",
    "pool_2x_ood_content_tracebus": "trace-bus\nonly",
    "pool_2x_ood_noise_amp_x2": "all-noise\n$\\times$2",
    "pool_2x_ood_noise_amp_x4": "all-noise\n$\\times$4",
    "pool_2x_ood_noise_stripe_x4": "stripes\n$\\times$4",
    "pool_2x_ood_noise_oneoverf_x4": "1/f\n$\\times$4",
    "pool_2x_ood_noise_hotpixel": "hot\npixels",
}
ARMS = {
    "tgv__oracle":   dict(color="#333333", marker="s", ls="-",  label="TGV (oracle)"),
    "tgv__portable": dict(color="#888888", marker="s", ls="--", label="TGV (portable)"),
    "depb9v6":       dict(color="#4C72B0", marker="o", ls="-",  label="Ours, v6 pool"),
    "depb9v9_9bin":  dict(color="#C44E52", marker="v", ls="-",  label="Ours, v9 pool (9-bin)"),
    "depb9v9_3k":    dict(color="#55A868", marker="P", ls="-",  label="Ours, v9 pool (3k)"),
}

fig, axes = plt.subplots(
    2, 2, figsize=(W_DOUBLE, 4.4), sharex="col",
    gridspec_kw=dict(width_ratios=[4, 5], height_ratios=[1.25, 1.0]),
)

# win/loss counts vs tgv__oracle, accumulated across both axes (9 pools total)
pos_counts: dict[str, int] = {}
n_pools: dict[str, int] = {}

for j, axis_name in enumerate(["content", "noise"]):
    sub = band[band["ood_axis"] == axis_name]
    pools = sub.sort_values("level")["pool"].unique()
    x = np.arange(len(pools))
    ax_abs, ax_diff = axes[0, j], axes[1, j]

    for arm, st in ARMS.items():
        y = [sub[(sub["pool"] == p) & (sub["arm"] == arm)]["frc"].iloc[0] for p in pools]
        ax_abs.plot(x, y, color=st["color"], marker=st["marker"], ls=st["ls"],
                    label=st["label"], markersize=4.5, clip_on=False)

    oracle = np.array([sub[(sub["pool"] == p) & (sub["arm"] == "tgv__oracle")]["frc"].iloc[0]
                       for p in pools])
    width = 0.26
    diff_arms = ["depb9v6", "depb9v9_9bin", "depb9v9_3k"]
    for k, arm in enumerate(diff_arms):
        y = np.array([sub[(sub["pool"] == p) & (sub["arm"] == arm)]["frc"].iloc[0]
                      for p in pools])
        d = y - oracle
        pos_counts[arm] = pos_counts.get(arm, 0) + int((d > 0).sum())
        n_pools[arm] = n_pools.get(arm, 0) + len(d)
        ax_diff.bar(x + (k - 1) * width, d, width=width * 0.92,
                    color=ARMS[arm]["color"], zorder=3)
        for xi, di in zip(x + (k - 1) * width, d):
            va = "bottom" if di >= 0 else "top"
            off = 1.2 if di >= 0 else -1.2
            ax_diff.annotate(f"{di:+.2f}", (xi, di), xytext=(0, off),
                             textcoords="offset points", ha="center", va=va,
                             fontsize=5.6, color="#333333")
    ax_diff.axhline(0, color="#222222", lw=0.8)
    ax_diff.set_xticks(x)
    ax_diff.set_xticklabels([POOL_SHORT[p] for p in pools], fontsize=7)
    ax_diff.set_ylim(-0.42, 0.27)

    ax_abs.set_ylim(0, 0.9)
    ax_abs.set_title(f"({'ab'[j]}) {axis_name.capitalize()}-axis OOD pools", loc="left")

axes[0, 0].set_ylabel("Band FRC (25–40 $\\mu$m)")
axes[1, 0].set_ylabel("$\\Delta$FRC vs TGV (oracle)")
axes[0, 1].tick_params(labelleft=True)
axes[1, 1].tick_params(labelleft=True)
axes[0, 0].legend(loc="lower left", ncol=2, fontsize=7, columnspacing=1.0)

# verdict annotations on the diff row — counts computed from data above,
# fail loud if the arm went missing from the CSV rather than silently
# falling back to stale hardcoded numbers.
for arm in ["depb9v6", "depb9v9_9bin", "depb9v9_3k"]:
    if n_pools.get(arm, 0) != 9:
        raise RuntimeError(
            f"expected 9 OOD pools for {arm}, found {n_pools.get(arm, 0)} — "
            "check ood_degradation_summary.csv for missing arm/pool rows"
        )
v6_pos, bin9_pos, k3_pos = (pos_counts[a] for a in
                            ("depb9v6", "depb9v9_9bin", "depb9v9_3k"))

# placed over the "all-noise x4" column (x=1), the only column where all
# three arms' bars stay small (<=0.10) and leave clear headroom to the
# ylim ceiling — avoids the tall +0.17..+0.22 bars/labels elsewhere.
axes[1, 1].annotate(f"v6: {v6_pos}/9 pools positive", (1.0, 0.245),
                    xycoords=("data", "data"), fontsize=6.5, color="#4C72B0",
                    ha="center", va="top")
axes[1, 1].annotate(f"v9 9-bin: loses {9 - bin9_pos}/9", (1.0, 0.205),
                    xycoords=("data", "data"), fontsize=6.5, color="#C44E52",
                    ha="center", va="top")
axes[1, 1].annotate(f"v9 3k: {k3_pos}/9 — worst OOD (ACL-079)", (1.0, 0.165),
                    xycoords=("data", "data"), fontsize=6.5, color="#55A868",
                    ha="center", va="top")

paths = save_fig(fig, "fig03_ood_robustness")
print("\n".join(str(p) for p in paths))
