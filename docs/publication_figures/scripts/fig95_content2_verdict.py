"""Fig 95 — Out-of-grammar content axis + the portable-baseline correction
(ACL-078), extended with the depb9v9_3k arm (ACL-079).

Four motif families the generator grammar has never produced (organic
blobs / text rows / concentric rings / voronoi cells):
(a) absolute band-FRC of the five arms;
(b) paired differences per pool — v6 − oracle is positive 4/4 (replicating
ACL-076's zero-sign-flip verdict) and v9_9bin − oracle is negative 4/4;
BUT against the stronger tgv_portable baseline v6 only ties (Δ ≈ 0):
the honest boundary is "v6's OOD advantage shrinks to parity with the best
classical arm on unfamiliar geometry". The systematic oracle < portable
inversion across all 13 OOD pools (oracle semantics under review) is why
both baselines are shown.

v9_3k − oracle is also negative 4/4, and more negative than v9_9bin − oracle
on every pool — consistent with ACL-079's headline finding that depb9v9_3k
is the worst-OOD neural arm overall (0/13 vs tgv_oracle across all OOD
pools, roughly 2x the degradation of the 9-bin arm). The point-fidelity
champion pays for it with the steepest out-of-grammar collapse.

Data: remote_inbox/20260713_content2ms/ood2_degradation_summary.csv.
Run:  uv run python docs/publication_figures/scripts/fig95_content2_verdict.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

SRC = REPO_ROOT / "remote_inbox/20260713_content2ms/ood2_degradation_summary.csv"
df = pd.read_csv(SRC, header=[0, 1], index_col=[0, 1, 2, 3, 4])
df.index.names = ["ood_axis", "level", "level_label", "pool", "arm"]
band = df[("frc_band_mean_25_40", "mean")].rename("frc").reset_index()

POOL_SHORT = {
    "pool_2x_ood_content2_organicblobs": "organic\nblobs",
    "pool_2x_ood_content2_textserial": "text\nrows",
    "pool_2x_ood_content2_rings": "concentric\nrings",
    "pool_2x_ood_content2_voronoi": "voronoi\ncells",
}
ARMS = {
    "tgv__oracle":   dict(color="#333333", marker="s", ls="-",  label="TGV (oracle)"),
    "tgv__portable": dict(color="#888888", marker="s", ls="--", label="TGV (portable)"),
    "depb9v6":       dict(color="#4C72B0", marker="o", ls="-",  label="Ours, v6 pool"),
    "depb9v9_9bin":  dict(color="#C44E52", marker="v", ls="-",  label="Ours, v9 pool (9-bin)"),
    "depb9v9_3k":    dict(color="#55A868", marker="P", ls="-",  label="Ours, v9 pool (3k)"),
}

pools = band.sort_values("level")["pool"].unique()
x = np.arange(len(pools))

def val(pool, arm):
    return band[(band["pool"] == pool) & (band["arm"] == arm)]["frc"].iloc[0]

fig, (ax_a, ax_b) = plt.subplots(
    2, 1, figsize=(W_DOUBLE * 0.72, 4.6), sharex=True)

for arm, st in ARMS.items():
    y = [val(p, arm) for p in pools]
    ax_a.plot(x, y, color=st["color"], marker=st["marker"], ls=st["ls"],
              markersize=4.5, label=st["label"], clip_on=False)
ax_a.set_ylim(0, 0.85)
ax_a.set_ylabel("Band FRC (25–40 $\\mu$m)")
ax_a.set_title("(a) Out-of-grammar motif families", loc="left")
ax_a.legend(loc="lower left", ncol=2, fontsize=6.5, columnspacing=1.0)

w = 0.2
series = [
    ("depb9v6", "tgv__oracle", "#4C72B0", 1.0, "v6 $-$ oracle"),
    ("depb9v6", "tgv__portable", "#4C72B0", 0.45, "v6 $-$ portable"),
    ("depb9v9_9bin", "tgv__oracle", "#C44E52", 1.0, "v9 9-bin $-$ oracle"),
    ("depb9v9_3k", "tgv__oracle", "#55A868", 1.0, "v9 3k $-$ oracle"),
]
for k, (arm, ref, color, alpha, label) in enumerate(series):
    d = np.array([val(p, arm) - val(p, ref) for p in pools])
    ax_b.bar(x + (k - 1.5) * w, d, width=w * 0.9, color=color, alpha=alpha,
             zorder=3, label=label)
    for xi, di in zip(x + (k - 1.5) * w, d):
        ax_b.annotate(f"{di:+.2f}", (xi, di),
                      xytext=(0, 2 if di >= 0 else -2),
                      textcoords="offset points", ha="center",
                      va="bottom" if di >= 0 else "top", fontsize=5.2,
                      color="#333333")
ax_b.axhline(0, color="#222222", lw=0.8)
ax_b.set_xticks(x)
ax_b.set_xticklabels([POOL_SHORT[p] for p in pools], fontsize=7.5)
ax_b.set_ylim(-0.40, 0.30)
ax_b.set_ylabel("$\\Delta$ band FRC")
ax_b.set_title("(b) vs both classical baselines", loc="left")
ax_b.legend(loc="lower left", fontsize=6.0, ncol=2, columnspacing=1.0)

paths = save_fig(fig, "fig95_content2_verdict")
print("\n".join(str(p) for p in paths))
