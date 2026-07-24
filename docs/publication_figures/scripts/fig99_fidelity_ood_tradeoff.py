"""Fig 99 — Point-fidelity vs OOD-robustness trade-off (ACL-079 headline verdict).

(a) Trade-off scatter: x = dot-probe retention fidelity (real-domain 3562-dot
probe, ACL-074), y = mean band-FRC (25-40um) advantage over TGV(oracle) across
all 13 OOD pools (9 round-1 pools ACL-076/079 + 4 out-of-grammar pools
ACL-078). The three neural arms fall on a MONOTONE trade: the point-fidelity
champion depb9v9_3k (0.00% erased, retention 0.798) is simultaneously the
worst arm out-of-distribution (0/13 wins, mean dFRC -0.275) -- "the
point-fidelity crown comes at the cost of the steepest OOD collapse"
(ACL-079). depb9v6 is the balanced champion (13/13 wins, +0.171);
depb9v9_9bin wins neither axis (fidelity-dominated by 3k, robustness-
dominated by v6).
(b) Substantiation: per-pool dFRC vs oracle for the three arms (gray
connectors); 3k is below 9bin in all 13 pools, and v6 tops 9bin in 12/13
(sole exception all-noise x4, where both still beat the oracle).

Robustness axis = mean dFRC (continuous); win counts (13/2/0 out of 13,
recomputed here, fail-loud) are annotated per arm. Delta is vs the TGV oracle
anchor; the stronger portable baseline tightens all bounds (ACL-078, fig95).

Data:
  docs/publication_figures/data/champion_arms.csv          (dot probe, ACL-074)
  remote_inbox/20260712_oodC/ood_degradation_summary.csv   (9 round-1 pools)
  remote_inbox/20260713_content2ms/ood2_degradation_summary.csv (4 OOG pools)
Run:  uv run python docs/publication_figures/scripts/fig99_fidelity_ood_tradeoff.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import (
    DATA_DIR,
    METHOD_STYLE,
    REF_LINE,
    REPO_ROOT,
    W_DOUBLE,
    save_fig,
    setup_academic_style,
)

setup_academic_style()

ARMS = ["depb9v6", "depb9v9_9bin", "depb9v9_3k"]
ORACLE = "tgv__oracle"

# ── Axis 1: dot fidelity (transcribed verdicts, ACL-074) ─────────────
champ = pd.read_csv(DATA_DIR / "champion_arms.csv", comment="#").set_index("arm")
retention = champ.loc[ARMS, "retention"].astype(float)
erased = champ.loc[ARMS, "erased_pct"].astype(float)

# ── Axis 2: OOD robustness from the two degradation summaries ────────
SOURCES = [
    REPO_ROOT / "remote_inbox/20260712_oodC/ood_degradation_summary.csv",
    REPO_ROOT / "remote_inbox/20260713_content2ms/ood2_degradation_summary.csv",
]
frames = []
for src in SOURCES:
    df = pd.read_csv(src, header=[0, 1], index_col=[0, 1, 2, 3, 4])
    df.index.names = ["ood_axis", "level", "level_label", "pool", "arm"]
    frames.append(df[("frc_band_mean_25_40", "mean")].rename("frc").reset_index())
band = pd.concat(frames, ignore_index=True)
piv = band.pivot_table(index=["ood_axis", "pool"], columns="arm", values="frc")

if len(piv) != 13:
    raise RuntimeError(f"expected 13 OOD pools, found {len(piv)} — check summaries")
delta = piv[ARMS].sub(piv[ORACLE], axis=0)          # per-pool dFRC vs oracle
wins = (delta > 0).sum()                            # v6 13, 9bin 2, 3k 0
mean_d = delta.mean()

# per-pool ordering claims (panel b) — recomputed, fail loud (ACL-079:
# 3k below 9bin in 13/13; v6 above 9bin everywhere except all-noise x4)
n_3k_worst = int((delta["depb9v9_9bin"] > delta["depb9v9_3k"]).sum())
n_v6_top = int((delta["depb9v6"] > delta["depb9v9_9bin"]).sum())
if n_3k_worst != 13 or n_v6_top != 12:
    raise RuntimeError(
        f"per-pool ordering drifted: 3k worst in {n_3k_worst}/13, "
        f"v6>9bin in {n_v6_top}/13 — recheck summaries vs ACL-079"
    )

# ── Figure ───────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(W_DOUBLE, 3.4), sharey=True,
    gridspec_kw=dict(width_ratios=[1.45, 1.0]),
)
YLIM = (-0.43, 0.30)

# ── (a) the trade-off scatter ────────────────────────────────────────
ax_a.axhspan(YLIM[0], 0, color="#f4f4f4", zorder=0)
ax_a.axhline(0, **REF_LINE, zorder=1)
ax_a.annotate("TGV (oracle) parity", (0.982, 0.0), xycoords=("axes fraction", "data"),
              xytext=(0, 3), textcoords="offset points", fontsize=7,
              ha="right", va="bottom", color="#222222")
ax_a.annotate("loses to oracle", (0.982, -0.012), xycoords=("axes fraction", "data"),
              xytext=(0, -3), textcoords="offset points", fontsize=7, style="italic",
              ha="right", va="top", color="#999999")

# monotone trade connector (drawn under the markers)
xs = retention[ARMS].to_numpy()
ys = mean_d[ARMS].to_numpy()
ax_a.plot(xs, ys, color="#bbbbbb", ls="--", lw=1.0, zorder=2)
t = 0.38  # fraction along the 9bin->3k segment; tag sits below the line
ax_a.annotate("monotone trade-off",
              (xs[1] + t * (xs[2] - xs[1]), ys[1] + t * (ys[2] - ys[1])),
              xytext=(0, -10), textcoords="offset points", fontsize=7,
              color="#888888", style="italic", ha="center", va="top",
              rotation=-16)

# per-arm annotation: (headline, dx, dy, ha, va) — offsets keep the labels
# clear of the dashed connector and of each other (maintenance note #1)
NOTES = {
    "depb9v6":      ("balanced champion", 9, 4, "left", "center"),
    "depb9v9_9bin": ("wins neither axis", 9, 6, "left", "bottom"),
    "depb9v9_3k":   ("fidelity champion,\nworst OOD arm", 0, -9, "center", "top"),
}
for arm in ARMS:
    st = METHOD_STYLE[arm]
    x, y = retention[arm], mean_d[arm]
    ax_a.scatter(x, y, c=st["color"], marker=st["marker"], s=58, zorder=4,
                 edgecolors="white", linewidths=0.6)
    head, dx, dy, ha, va = NOTES[arm]
    tag = (f"{st['label']} — {head}\n"
           f"{erased[arm]:.2f}% erased · OOD {wins[arm]}/13 · "
           f"$\\Delta$ {mean_d[arm]:+.3f}")
    ax_a.annotate(tag, (x, y), xytext=(dx, dy), textcoords="offset points",
                  fontsize=7, ha=ha, va=va, color="#333333", zorder=5)

# ideal-direction cue (matches fig02)
ax_a.annotate("ideal", xy=(0.895, 0.865), xycoords="axes fraction", fontsize=7.5,
              color="#666666", ha="right", va="center")
ax_a.annotate("", xy=(0.985, 0.975), xytext=(0.905, 0.895), xycoords="axes fraction",
              arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))

ax_a.set_xlim(0.555, 0.90)
ax_a.set_ylim(*YLIM)
ax_a.set_xlabel("Dot-probe retention fidelity (higher is better)")
ax_a.set_ylabel("$\\Delta$FRC (25–40 $\\mu$m) vs TGV (oracle)")
ax_a.set_title("(a) Fidelity vs OOD robustness (mean, 13 pools)", loc="left")

# ── (b) per-pool substantiation strip ────────────────────────────────
X = np.arange(len(ARMS))
oog = piv.index.get_level_values("ood_axis") == "content2"  # out-of-grammar

for i in range(len(piv)):                       # per-pool connectors
    ax_b.plot(X, delta.iloc[i, :].to_numpy(), color="#999999", lw=0.6,
              alpha=0.45, zorder=1)
for k, arm in enumerate(ARMS):
    st = METHOD_STYLE[arm]
    for mask, filled in ((~oog, True), (oog, False)):
        ax_b.scatter(np.full(mask.sum(), X[k]), delta.loc[mask, arm],
                     marker=st["marker"], s=26, zorder=3,
                     facecolors=st["color"] if filled else "white",
                     edgecolors=st["color"], linewidths=0.9)
    ax_b.plot([X[k] - 0.17, X[k] + 0.17], [mean_d[arm]] * 2, color="#222222",
              lw=1.5, zorder=4)

ax_b.axhline(0, **REF_LINE, zorder=2)
ax_b.set_xlim(-0.55, 2.55)
ax_b.set_xticks(X)
ax_b.set_xticklabels(["v6", "v9 9-bin", "v9 3k"], fontsize=8)
for k, arm in enumerate(ARMS):
    ax_b.get_xticklabels()[k].set_color(METHOD_STYLE[arm]["color"])
ax_b.tick_params(labelleft=True)
ax_b.set_title("(b) Per-pool $\\Delta$FRC vs oracle", loc="left")
ax_b.annotate(f"3k worst in {n_3k_worst}/13 pools;\n"
              f"v6 > 9-bin in {n_v6_top}/13 (exception:\n"
              "all-noise $\\times$4, both beat oracle)",
              (0.03, 0.035), xycoords="axes fraction", fontsize=7,
              ha="left", va="bottom", color="#444444", style="italic")

# marker-fill legend (colors already carry arm identity)
handles = [
    plt.Line2D([], [], ls="none", marker="o", mfc="#666666", mec="#666666",
               ms=4.5, label="in-grammar pool (9)"),
    plt.Line2D([], [], ls="none", marker="o", mfc="white", mec="#666666",
               ms=4.5, mew=0.9, label="out-of-grammar pool (4)"),
    plt.Line2D([], [], color="#222222", lw=1.5, label="mean (panel a)"),
]
ax_b.legend(handles=handles, loc="upper right", fontsize=6.5, handlelength=1.2,
            labelspacing=0.3, borderaxespad=0.2)

fig.suptitle("The point-fidelity crown costs the steepest OOD collapse (ACL-074/077/079)",
             x=0.005, ha="left", fontsize=10, fontweight="bold")
fig.text(0.005, -0.03,
         "$\\Delta$FRC is vs the TGV oracle anchor (ACL-076 convention); the stronger "
         "portable baseline tightens all bounds (ACL-078, fig95). Fidelity = ALL-dot "
         "retention score, real-domain 3562-dot probe (ACL-074).",
         ha="left", va="top", fontsize=7, style="italic", color="#555555")

paths = save_fig(fig, "fig99_fidelity_ood_tradeoff")
print(f"wins/13: {wins.to_dict()}  mean dFRC: {mean_d.round(4).to_dict()}")
print("\n".join(str(p) for p in paths))
