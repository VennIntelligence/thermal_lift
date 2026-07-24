"""Fig 44 — OOD secondary metrics: DC offset drift and low-frequency stability.

Companion to fig03 (band-FRC verdict). Across the 9 OOD pools:
(a) mean reconstruction offset — both depb9v9_9bin AND depb9v9_3k drift
systematically negative (roughly −2...−5.5 °C for 9bin, −2...−5.0 °C for
3k) on every pool while v6 and the classical arms stay near zero: the
point-fidelity-oriented v9 arms share a failure mode under distribution
shift, a global DC pull rather than just texture loss — and this is not
specific to the 9-bin recipe. (b) range excursion (log scale) — v9_9bin
and v9_3k both sit an order of magnitude above v6/TGV on every pool
(3k's peak is even slightly higher than 9bin's, 10.96 vs 10.43 at
tracebus), extending the in-distribution range_exc anomaly (ACL-072 open
item) to the OOD axes for both v9 variants. Combined with fig03's
band-FRC collapse, depb9v9_3k (ACL-079) is confirmed as the worst-OOD
arm on every metric measured here, not merely on point fidelity.

Data: remote_inbox/20260712_oodC/ood_degradation_summary.csv (ACL-076, ACL-079).
Run:  uv run python docs/publication_figures/scripts/fig44_ood_secondary_metrics.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

SRC = REPO_ROOT / "remote_inbox/20260712_oodC/ood_degradation_summary.csv"
df = pd.read_csv(SRC, header=[0, 1], index_col=[0, 1, 2, 3, 4])
df.index.names = ["ood_axis", "level", "level_label", "pool", "arm"]
d = df[[("mean_offset", "mean"), ("range_excursion", "mean")]]
d.columns = ["mean_offset", "range_exc"]
d = d.reset_index()

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

# fixed pool order: content by level, then noise by level
pools = (d[d["ood_axis"] == "content"].sort_values("level")["pool"].unique().tolist()
         + d[d["ood_axis"] == "noise"].sort_values("level")["pool"].unique().tolist())
x = np.arange(len(pools))

fig, (ax_a, ax_b) = plt.subplots(2, 1, figsize=(W_DOUBLE, 4.6), sharex=True)

for arm, st in ARMS.items():
    off = [d[(d["pool"] == p) & (d["arm"] == arm)]["mean_offset"].iloc[0] for p in pools]
    exc = [d[(d["pool"] == p) & (d["arm"] == arm)]["range_exc"].iloc[0] for p in pools]
    ax_a.plot(x, off, color=st["color"], marker=st["marker"], ls=st["ls"],
              markersize=4.5, label=st["label"], clip_on=False)
    ax_b.plot(x, exc, color=st["color"], marker=st["marker"], ls=st["ls"],
              markersize=4.5, clip_on=False)

ax_a.axhline(0, color="#cccccc", lw=0.8, zorder=0)
ax_a.set_ylabel("Mean offset [$^\\circ$C]")
ax_a.set_title("(a) Global DC drift under distribution shift", loc="left")
ax_a.legend(loc="lower left", ncol=2, fontsize=7, columnspacing=1.0)

ax_b.set_yscale("log")
ax_b.set_ylabel("Range excursion [$^\\circ$C] (log)")
ax_b.set_title("(b) Low-frequency stability", loc="left")
ax_b.set_xticks(x)
ax_b.set_xticklabels([POOL_SHORT[p] for p in pools], fontsize=7)

# separator between content and noise pool groups
for ax in (ax_a, ax_b):
    ax.axvline(3.5, color="#dddddd", lw=0.8, zorder=0)
ax_b.annotate("— content axis —", (1.5, -0.30), xycoords=("data", "axes fraction"),
              ha="center", va="top", fontsize=7.5, color="#666666")
ax_b.annotate("— noise axis —", (6.5, -0.30), xycoords=("data", "axes fraction"),
              ha="center", va="top", fontsize=7.5, color="#666666")

paths = save_fig(fig, "fig44_ood_secondary_metrics")
print("\n".join(str(p) for p in paths))
