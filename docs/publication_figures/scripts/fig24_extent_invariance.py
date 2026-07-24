"""Fig 24 — GroupNorm+SE break extent-invariance: far-field perturbation
leaks into local predictions.

Context (EP07 solver V8/K4 diagnosis, 2026-06-30; ACL-040/041): the K4
grid-vs-flocculence tradeoff was root-caused to the prox UNet not being
"extent-invariant" — its prediction at any pixel depends on how large a
region is solved, because GroupNorm normalizes over the whole image and the
SEBlock gates on a global average pool. The causal test (diag_extent.py)
needs NO trained weights: extent-invariance is an architectural property,
so a random-init net with the global ops selectively ablated isolates the
mechanism. This finding drove the noSE+noGN mainline architecture.

Experiment (diag_extent.py, torch seed 0, random init, identical conv
weights across variants; only GroupNorm/SE are ablated to Identity):
  D2 far-field : warm (+3 K) ONLY the outer 64-px frame of a 768^2 input
      field and measure the RMS output change inside a central 96^2 window,
      normalized by that window's own output std. The gap between the
      perturbation and the window edge is 272 px — far beyond the conv
      receptive field. Pure conv (noGN+noSE): exactly 0.0 (machine-exact,
      validates the margin). GroupNorm only: 0.673. GroupNorm+SE (the real
      architecture): 1.412 — a far-field perturbation 272 px away shifts
      the local prediction by ~1.4 sigma.
  D1 crop/full : same window predicted from the full 768^2 field vs from
      the 192^2 training-size tile around it (interior 96^2 compared).
      GN+SE: 1.41, GN only: 1.50, pure conv: 1.3e-5 (~0) — switching
      192-tile -> full-frame inference necessarily changes every pixel.
  K-amp        : residual prox loop x_{k+1} = x_k + 0.1*net(x_k, cond)
      with the full (GN+SE) architecture; the crop-vs-full interior gap
      grows ~linearly with K (0.018/0.037/0.055/0.074 for K=1..4),
      reproducing ACL-037's "K4 worse than K2" from first principles.

Panels:
  (a) schematic of the far-field test geometry (768^2 field, perturbed
      64-px frame, 96^2 measurement window, 272-px gap; the 192^2
      training-tile outline used by the crop-vs-full test is dashed).
  (b) paired bars: normalized RMS output change in the window for both
      tests x three architecture variants.
  (c) K-amplification of the crop-vs-full gap over the recurrent prox loop.

Data: outputs/ep07_solver_diag/metrics_extent.json (D2/D1 scalars; written
      by outputs/ep07_solver_diag/diag_extent.py, archived with DIAGNOSIS.md
      under research_log/episodes/ep07_solver_v8_k4_fullhalo_eval_archive/
      diagnosis_20260630/). The K-amp series is printed but not saved by
      diag_extent.py; values below were re-verified bit-identically by
      re-running the deterministic script on 2026-07-13 (uv env of
      algos/ep07_unet_sr, CPU): 1.8209e-2 / 3.6614e-2 / 5.5211e-2 /
      7.3993e-2, matching DIAGNOSIS.md section 3.
Run:  uv run python docs/publication_figures/scripts/fig24_extent_invariance.py
"""

import json

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

with open(REPO_ROOT / "outputs" / "ep07_solver_diag" / "metrics_extent.json") as fh:
    metrics = json.load(fh)

d2 = metrics["D2_farfield"]    # far-field frame perturbation -> window change
d1 = metrics["D1_interior"]    # 192-tile vs full-frame interior change

# K-amplification series (full GN+SE arch, alpha=0.1 residual prox loop);
# printed by diag_extent.py, re-verified deterministically (see docstring).
K_STEPS = np.array([1, 2, 3, 4])
K_GAP = np.array([1.8209e-2, 3.6614e-2, 5.5211e-2, 7.3993e-2])

VARIANTS = [
    ("full", "GroupNorm + SE", "#C44E52", None),
    ("noSE", "GroupNorm only", "#DD8452", "///"),
    ("noGN_noSE", "Pure conv", "#4C72B0", None),
]

fig, axes = plt.subplots(
    1, 3, figsize=(W_DOUBLE, 2.9), gridspec_kw={"width_ratios": [1.0, 1.15, 0.95]}
)

# ── (a) experiment geometry ────────────────────────────────────────────
ax = axes[0]
N, RING, WIN, TILE = 768, 64, 96, 192
c = N // 2
# perturbed outer frame (+3 K): orange field with unperturbed interior on top
ax.add_patch(mpatches.Rectangle((0, 0), N, N, fc="#DD8452", ec="none", alpha=0.75))
ax.add_patch(mpatches.Rectangle((RING, RING), N - 2 * RING, N - 2 * RING,
                                fc="#f4f2ef", ec="#bbbbbb", lw=0.5))
# 192^2 training-tile outline (crop-vs-full test)
ax.add_patch(mpatches.Rectangle((c - TILE // 2, c - TILE // 2), TILE, TILE,
                                fc="none", ec="#937860", lw=0.9, ls="--"))
# 96^2 measurement window
ax.add_patch(mpatches.Rectangle((c - WIN // 2, c - WIN // 2), WIN, WIN,
                                fc="#4C72B0", ec="#4C72B0", lw=1.2, alpha=0.35))
# 272-px gap arrow (window bottom edge -> frame inner edge)
ax.annotate("", xy=(c + 150, RING), xytext=(c + 150, c - WIN // 2),
            arrowprops=dict(arrowstyle="<->", color="#222222", lw=0.9))
ax.annotate("272 px\n$\\gg$ conv RF", (c + 175, (RING + c - WIN // 2) / 2),
            ha="left", va="center", fontsize=7)
ax.annotate("perturbed frame\n(outer 64 px, +3 K)", (c, N - RING // 2),
            ha="center", va="center", fontsize=7, color="#5a2d14")
ax.annotate("96$^2$ window:\nmeasure $\\Delta$output", (c, c),
            xytext=(120, 620), fontsize=7, ha="left", va="center",
            arrowprops=dict(arrowstyle="-", color="#4C72B0", lw=0.7))
ax.annotate("192$^2$ training tile\n(crop-vs-full test)",
            (c - TILE // 2, c - TILE // 2), xytext=(90, 108),
            fontsize=7, ha="left", va="center", color="#5c4a3d",
            arrowprops=dict(arrowstyle="-", color="#937860", lw=0.7))
ax.set_xlim(-10, N + 10)
ax.set_ylim(-10, N + 10)
ax.set_aspect("equal")
ax.set_xticks([0, c, N])
ax.set_yticks([0, c, N])
ax.tick_params(labelsize=6.5)
ax.set_xlabel("x [px]")
ax.set_ylabel("y [px]")
ax.set_title("(a) Far-field test geometry", loc="left", fontsize=9)

# ── (b) paired bars: both tests x three variants ───────────────────────
ax = axes[1]
groups = [
    ("Far-field frame\nperturbation", d2),
    ("Crop (192$^2$) vs\nfull-frame interior", d1),
]
width = 0.26
xg = np.arange(len(groups))
for i, (key, label, color, hatch) in enumerate(VARIANTS):
    vals = [g[1][key] for g in groups]
    xs = xg + (i - 1) * width
    ax.bar(xs, vals, width=width * 0.92, color=color, label=label,
           hatch=hatch, edgecolor="white", linewidth=0.4)
    for x, v in zip(xs, vals):
        txt = f"{v:.2f}" if v > 5e-3 else ("0 (exact)" if v == 0.0 else "$\\approx$0")
        ax.annotate(txt, (x, v), xytext=(0, 2), textcoords="offset points",
                    ha="center", va="bottom", fontsize=7)
ax.annotate("272 px away shifts\noutput $\\sim$1.4$\\sigma$ (GN+SE);\npure conv: exactly 0",
            (0.98, 0.99), xycoords="axes fraction", fontsize=7,
            ha="right", va="top", color="#8a2f33")
ax.set_xticks(xg)
ax.set_xticklabels([g[0] for g in groups], fontsize=7.5)
ax.set_ylabel("RMS output change in window [window $\\sigma$]")
ax.set_ylim(0, 2.1)
ax.grid(axis="y", alpha=0.3, linewidth=0.5)
ax.legend(loc="upper left", bbox_to_anchor=(0.0, 1.02), fontsize=6.5)
ax.set_title("(b) Locality violation by variant", loc="left", fontsize=9)

# ── (c) K-recurrence amplification ─────────────────────────────────────
ax = axes[2]
ax.plot(K_STEPS, K_GAP, color="#C44E52", marker="o", lw=1.4,
        label="GroupNorm+SE prox ($\\alpha$=0.1)")
for k, v in zip(K_STEPS, K_GAP):
    ha = "left" if k == 1 else ("right" if k == 4 else "center")
    dx = 4 if k == 1 else (-2 if k == 4 else 0)
    ax.annotate(f"{v:.3f}", (k, v), xytext=(dx, 5), textcoords="offset points",
                ha=ha, va="bottom", fontsize=7)
ax.annotate("extent bias accumulates\n$\\approx$linearly over the shared\n"
            "prox loop (ACL-037:\nK4 worse than K2)",
            (0.97, 0.06), xycoords="axes fraction", ha="right", va="bottom",
            fontsize=7, color="#333333")
ax.set_xticks(K_STEPS)
ax.set_xlabel("Unroll steps $K$ (residual prox)")
ax.set_ylabel("Crop-vs-full interior gap [window $\\sigma$]")
ax.set_ylim(0, 0.085)
ax.grid(axis="y", alpha=0.3, linewidth=0.5)
ax.legend(loc="upper left", fontsize=6.5)
ax.set_title("(c) K-recurrence amplification", loc="left", fontsize=9)

paths = save_fig(fig, "fig24_extent_invariance")
print("\n".join(str(p) for p in paths))
