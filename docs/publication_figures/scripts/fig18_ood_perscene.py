"""Fig 18 — per-scene OOD paired-difference contrast: v6 vs v9-3k (ACL-076, ACL-079).

Companion to fig03_ood_robustness.py, which shows POOL MEANS only. This
figure exists to prove two per-scene claims at once, as a stacked
two-panel mirror (shared x-axis, 9 OOD pools):

  (a) the depb9v6-vs-tgv__oracle advantage documented in fig03 is not
      driven by a handful of outlier scenes: for each pool we plot the
      full per-scene paired distribution of
          delta = frc_band_mean_25_40(depb9v6) - frc_band_mean_25_40(tgv__oracle)
      computed scene-by-scene (48 scenes per pool, matched on scene_id).
      v6 beats the oracle on the large majority of scenes in every pool
      (per-pool win% 68.8-97.9; 8 of 9 pools >= 81).
  (b) the point-fidelity champion depb9v9_3k collapses under OOD (ACL-079:
      3k is the worst OOD arm): the same paired delta vs tgv__oracle is
      almost entirely NEGATIVE — per-pool win% 0-12.5 in 8 of 9 pools
      (43.8 in all-noise x4, the one pool where oracle itself degrades),
      mirroring panel (a). NOTE: the literal "wins/loses at EVERY scene"
      phrasing is NOT supported scene-by-scene (v6 amp_x4 win% = 68.8,
      3k amp_x4 win% = 43.8); titles say "nearly every scene" and the
      per-pool win% row carries the exact numbers.

Both panels share the same y-limits so the up/down mirror reads at a
glance. Box + jittered-strip form with a zero reference line, as before:
a half-violin-per-arm alternative was tried first and rejected — with
9 pools x 2 arms side by side the panel became too dense to read at CVPR
column width; the paired-difference form reads cleanly and is also the
more direct statistical statement (paired test, not two unpaired
distributions).

Data: remote_inbox/20260712_oodC/ood_degradation_long.csv — per-scene rows,
      columns include scene_id, arm, pool, ood_axis, level,
      frc_band_mean_25_40. NOW the 5-arm ACL-079 rebuild (2026-07-12):
      arms {depb9v6, depb9v9_9bin, depb9v9_3k, tgv__oracle, tgv__portable},
      9 pools x 5 arms x 48 scenes = 2160 rows (2161 lines incl. header),
      md5 c1f6f716bcab42b229201d44c6f1ae5c. The 4 pre-existing arms are
      bit-identical to the earlier 4-arm file, so panel (a) is unchanged
      from the previous single-panel version of this figure.
      HISTORY (2026-07-13, earlier 4-arm file): the file initially present
      in the repo was corrupted (an accidental shell-error-message stub,
      363 bytes) from a prior failed base64 transfer off the Windows/WSL
      5090 box (macOS `base64 -d` is not a valid flag; must be `base64 -D`,
      and large payloads must be chunked — a single `base64 -w0 file` ssh
      round-trip silently truncated at ~463KB of an expected ~987KB).
      Re-fetched by splitting the source file into 5 line-chunks,
      transferring each with explicit BEGIN/END markers, base64-decoding
      in Python, and concatenating; that 4-arm file's md5 was
      794278fa3a0331649b944542a7c5b3f6 (740531 bytes, 1729 lines).
Run:  uv run python docs/publication_figures/scripts/fig18_ood_perscene.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

SRC = REPO_ROOT / "remote_inbox/20260712_oodC/ood_degradation_long.csv"
df = pd.read_csv(SRC)

METRIC = "frc_band_mean_25_40"
V6, V9_3K, ORACLE = "depb9v6", "depb9v9_3k", "tgv__oracle"

COLOR_V6 = "#4C72B0"   # blue
COLOR_3K = "#55A868"   # green

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
CONTENT_POOLS = [
    "pool_2x_ood_content_texroles",
    "pool_2x_ood_content_xlmerge",
    "pool_2x_ood_content_legacymix",
    "pool_2x_ood_content_tracebus",
]
NOISE_POOLS = [
    "pool_2x_ood_noise_amp_x2",
    "pool_2x_ood_noise_amp_x4",
    "pool_2x_ood_noise_stripe_x4",
    "pool_2x_ood_noise_oneoverf_x4",
    "pool_2x_ood_noise_hotpixel",
]
POOLS = CONTENT_POOLS + NOISE_POOLS


# ── paired per-scene deltas ────────────────────────────────────────────
def paired_deltas(arm: str) -> list[np.ndarray]:
    """Per-pool arrays of per-scene delta = metric(arm) - metric(oracle)."""
    a = df[df["arm"] == arm].set_index(["pool", "scene_id"])[METRIC]
    orc = df[df["arm"] == ORACLE].set_index(["pool", "scene_id"])[METRIC]
    paired = pd.concat([a, orc], axis=1, keys=[arm, ORACLE], join="inner").reset_index()
    paired["delta"] = paired[arm] - paired[ORACLE]
    out = [paired.loc[paired["pool"] == p, "delta"].to_numpy() for p in POOLS]
    for p, d in zip(POOLS, out):
        assert len(d) == 48, f"{arm} vs {ORACLE}, {p}: expected 48 paired scenes, got {len(d)}"
    return out


deltas_v6 = paired_deltas(V6)
deltas_3k = paired_deltas(V9_3K)

# ── figure ──────────────────────────────────────────────────────────
fig, (ax_a, ax_b) = plt.subplots(
    2, 1, figsize=(W_DOUBLE, 4.8), sharex=True, gridspec_kw=dict(hspace=0.10)
)

GAP = 0.9  # extra x-offset between content and noise groups
x_content = np.arange(len(CONTENT_POOLS))
x_noise = np.arange(len(NOISE_POOLS)) + len(CONTENT_POOLS) + GAP
x = np.concatenate([x_content, x_noise])

# IDENTICAL y-limits on both panels so the +/− mirror reads at a glance
all_d = np.concatenate(deltas_v6 + deltas_3k)
pad = 0.06 * (all_d.max() - all_d.min())
Y_LO, Y_HI = all_d.min() - pad, all_d.max() + pad
HEADROOM = 0.16 * (Y_HI - Y_LO)  # room for the win-rate row inside each panel
Y_TOP = Y_HI + 1.55 * HEADROOM   # extra row above win% for group labels (panel a)


def draw_panel(ax, deltas, color, seed):
    box = ax.boxplot(
        deltas,
        positions=x,
        widths=0.5,
        patch_artist=True,
        showfliers=False,
        medianprops=dict(color="#222222", linewidth=1.2),
        boxprops=dict(facecolor=color, alpha=0.28, edgecolor=color, linewidth=0.9),
        whiskerprops=dict(color=color, linewidth=0.9),
        capprops=dict(color=color, linewidth=0.9),
        zorder=3,
    )
    rng = np.random.default_rng(seed)
    for xi, d in zip(x, deltas):
        jitter = rng.uniform(-0.16, 0.16, size=len(d))
        ax.scatter(xi + jitter, d, s=5, color="#333333", alpha=0.45, linewidths=0, zorder=4)
    ax.axhline(0, color="#222222", ls="--", lw=0.9, zorder=2)
    ax.axvline((x_content[-1] + x_noise[0]) / 2, color="#cccccc", lw=0.8, zorder=1)
    ax.set_ylim(Y_LO, Y_TOP)
    # per-pool win-rate annotation, in a row along the panel top
    wins = []
    for xi, d in zip(x, deltas):
        win = (d > 0).mean() * 100
        wins.append(win)
        ax.annotate(f"{win:.0f}%", (xi, Y_HI + 0.25 * HEADROOM), ha="center",
                    va="bottom", fontsize=6.3, color=color)
    return box, wins


_, wins_v6 = draw_panel(ax_a, deltas_v6, COLOR_V6, seed=0)
_, wins_3k = draw_panel(ax_b, deltas_3k, COLOR_3K, seed=1)

ax_a.set_title(
    "(a) Ours (v6): $+\\Delta$ on nearly every scene", loc="left", color=COLOR_V6
)
ax_b.set_title(
    "(b) Ours (v9, 3k): $-\\Delta$ on nearly every scene — worst-OOD collapse (ACL-079)",
    loc="left", color="#3B7A4A",
)

ax_a.set_ylabel("Per-scene $\\Delta$ Band FRC\n(25–40 $\\mu$m), v6 $-$ oracle")
ax_b.set_ylabel("Per-scene $\\Delta$ Band FRC\n(25–40 $\\mu$m), 3k $-$ oracle")

ax_b.set_xticks(x)
ax_b.set_xticklabels([POOL_SHORT[p] for p in POOLS], fontsize=7)

# group-name labels above the top panel's win-rate row (panel (a) only)
GROUP_Y = Y_HI + 0.62 * HEADROOM
ax_a.text(x_content.mean(), GROUP_Y, "content-axis OOD pools", ha="center",
          va="bottom", fontsize=8, fontweight="bold", color="#333333")
ax_a.text(x_noise.mean(), GROUP_Y, "noise-axis OOD pools", ha="center",
          va="bottom", fontsize=8, fontweight="bold", color="#333333")

# annotation row label (win% = fraction of scenes where the arm beats oracle)
legend_handles = [
    plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#333333",
               markersize=4, alpha=0.6, label="scene ($\\Delta$FRC)"),
    plt.Rectangle((0, 0), 1, 1, facecolor=COLOR_V6, alpha=0.28,
                  edgecolor=COLOR_V6, label="box: IQR, median"),
    plt.Line2D([0], [0], marker="$\\%$", color="none", markerfacecolor="#555555",
               markeredgecolor="#555555", markersize=6,
               label="win% vs. oracle, per pool"),
]
ax_a.legend(handles=legend_handles, loc="lower left", fontsize=6.5, ncol=1)

print("per-pool win% (v6 > oracle):")
for p, w in zip(POOLS, wins_v6):
    print(f"  {p}: {w:.1f}%")
print("per-pool win% (v9_3k > oracle):")
for p, w in zip(POOLS, wins_3k):
    print(f"  {p}: {w:.1f}%")

paths = save_fig(fig, "fig18_ood_perscene")
print("\n".join(str(p) for p in paths))
