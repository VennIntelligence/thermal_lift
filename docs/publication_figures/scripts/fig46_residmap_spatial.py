"""fig46 — Where the self-doubt gauge fires: held-out DC residual map with dot classes.

Spatial companion to fig04/fig47 (ACL-075, research_log/dc_residual_confidence_analysis.md).

Provenance
----------
Data:  output/dc_residual_confidence/ produced by
       algos/ep07_unet_sr/scripts/analyze_dc_residual_confidence.py (2026-07-11,
       seed=42, psf_sigma=0.5 placeholder operator, 112 held-out frames/half,
       see run_meta.json in that directory).
  - depb9v9_3k_residmap_a.npy : per-pixel mean |y - A(x_hat)| over the 112
    held-out frames of split half a, splatted to the drizzle HR grid (960x1280).
  - per_dot_residual_stats.csv : dot_id, y, x (probe-frame = drizzle HR grid
    coordinates), sigma, arm, dot_class (from the dot-fidelity probe, not
    human GT), window residual stats.

Coordinate verification (done empirically before this script was finalized):
  - CSV resid_win_max reproduces EXACTLY (median |err| = 0.0) when the
    (a+b)/2 residual map is indexed as map[y, x] with the analysis window
    half = clip(ceil(2*sigma)+2, 3, 8); transposed indexing gives median
    |err| = 0.10. So (y, x) index the arrays directly, no flip/transpose.
  - On half a alone (the map shown), erased-dot window maxima beat 3000
    random positions with AUC = 0.887 (median 0.243 vs 0.132), i.e. erased
    dots land on local residual maxima far more often than chance.

Design:
  (a) full half-a residual map for depb9v9_3k (YlOrRd, vmax = p99), erased
      dots circled in red (all 139), a seeded random subsample of preserved
      dots in thin blue (150 of 2547; stated on the figure).
  (b) four zoom crops centered on the strongest erased-dot residual peaks.
      Ranking uses the CSV's background-subtracted window peak resid_bs_max
      (raw window max is dominated by the sigma-mismatch edge systematics,
      not by point-like dot footprints), deduplicated to >=80 px apart.
      Dot position marked; crops show the same half-a raw residual map.

Run:   uv run python docs/publication_figures/scripts/fig46_residmap_spatial.py
Output: docs/publication_figures/figures/fig46_residmap_spatial.{png,pdf}
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Circle, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pubfig_style import CMAP_RESID_POS, W_DOUBLE, save_fig, setup_academic_style

DATA_DIR = Path(__file__).resolve().parents[3] / "output" / "dc_residual_confidence"
ARM = "depb9v9_3k"
HALF = "a"
N_PRESERVED_SHOWN = 150
SUBSAMPLE_SEED = 0
N_CROPS = 4
CROP_HALF = 24          # 49x49 px zoom windows
MIN_PEAK_SEP = 80       # px, spatial separation between chosen crops

WHITE_HALO = [pe.withStroke(linewidth=1.8, foreground="white", alpha=0.75)]


def main() -> None:
    setup_academic_style()

    resid = np.load(DATA_DIR / f"{ARM}_residmap_{HALF}.npy")
    df = pd.read_csv(DATA_DIR / "per_dot_residual_stats.csv")
    d = df[df["arm"] == ARM].reset_index(drop=True)
    erased = d[d["dot_class"] == "erased"]
    preserved = d[d["dot_class"] == "preserved"]

    rng = np.random.default_rng(SUBSAMPLE_SEED)
    pres_sub = preserved.iloc[
        rng.choice(len(preserved), size=N_PRESERVED_SHOWN, replace=False)
    ]

    # Rank erased dots by background-subtracted window peak (localized
    # point footprint, not the edge systematics that dominate raw max).
    er = erased.sort_values("resid_bs_max", ascending=False)
    picks: list = []
    for r in er.itertuples():
        if all(abs(r.y - p.y) + abs(r.x - p.x) > MIN_PEAK_SEP for p in picks):
            picks.append(r)
        if len(picks) == N_CROPS:
            break

    vmax = float(np.percentile(resid, 99))

    fig = plt.figure(figsize=(W_DOUBLE, 3.55))
    gs = fig.add_gridspec(
        2, 3, width_ratios=[2.75, 1.0, 1.0], wspace=0.06, hspace=0.14
    )
    ax_map = fig.add_subplot(gs[:, 0])
    crop_axes = [fig.add_subplot(gs[i // 2, 1 + i % 2]) for i in range(N_CROPS)]

    # ── (a) full residual map ────────────────────────────────────────
    im = ax_map.imshow(
        resid, cmap=CMAP_RESID_POS, vmin=0.0, vmax=vmax,
        origin="upper", interpolation="nearest",
    )
    for r in erased.itertuples():
        c = Circle((r.x, r.y), radius=11, fill=False, ec="#C44E52", lw=1.0)
        c.set_path_effects(WHITE_HALO)
        ax_map.add_patch(c)
    for r in pres_sub.itertuples():
        c = Circle((r.x, r.y), radius=8, fill=False, ec="#4C72B0", lw=0.45)
        c.set_path_effects([pe.withStroke(linewidth=1.2, foreground="white",
                                          alpha=0.7)])
        ax_map.add_patch(c)
    for k, p in enumerate(picks, start=1):
        ax_map.add_patch(Rectangle(
            (p.x - CROP_HALF, p.y - CROP_HALF), 2 * CROP_HALF, 2 * CROP_HALF,
            fill=False, ec="#222222", lw=0.8, ls="--",
        ))
        ax_map.annotate(
            str(k), (p.x + CROP_HALF + 4, p.y - CROP_HALF), fontsize=8,
            color="#222222", va="top", fontweight="bold",
        )
    ax_map.set_title(f"(a) Held-out DC residual, {ARM.replace('_', ' ')} half a")
    ax_map.set_xlabel("x [HR px]")
    ax_map.set_ylabel("y [HR px]")
    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.02,
                        location="left")
    cbar.set_label(r"$|y - A(\hat{x})|$ per-frame mean [norm. counts]")
    ax_map.legend(
        handles=[
            plt.Line2D([], [], marker="o", mfc="none", mec="#C44E52", ls="none",
                       ms=6, label=f"erased dots (all {len(erased)})"),
            plt.Line2D([], [], marker="o", mfc="none", mec="#4C72B0", ls="none",
                       ms=5, mew=0.6,
                       label=f"preserved dots ({N_PRESERVED_SHOWN}/{len(preserved)}, random subsample)"),
        ],
        loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2,
    )

    # ── (b) zoom crops on strongest erased-dot peaks ─────────────────
    for k, (ax, p) in enumerate(zip(crop_axes, picks), start=1):
        y0, y1 = int(p.y) - CROP_HALF, int(p.y) + CROP_HALF + 1
        x0, x1 = int(p.x) - CROP_HALF, int(p.x) + CROP_HALF + 1
        ax.imshow(resid[y0:y1, x0:x1], cmap=CMAP_RESID_POS, vmin=0.0,
                  vmax=vmax, origin="upper", interpolation="nearest",
                  extent=[x0, x1, y1, y0])
        c = Circle((p.x, p.y), radius=6, fill=False, ec="#C44E52", lw=1.1)
        c.set_path_effects(WHITE_HALO)
        ax.add_patch(c)
        ax.plot(p.x, p.y, "+", color="#4C72B0", ms=6, mew=1.0,
                path_effects=WHITE_HALO)
        ax.set_title(f"{k}: dot {int(p.dot_id)}, bs peak {p.resid_bs_max:.2f}",
                     fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(True)
    fig.text(
        0.80, 0.015,
        "(b) 4 strongest erased-dot peaks\n(background-subtracted; dashed boxes in a)",
        ha="center", fontsize=8,
    )

    save_fig(fig, "fig46_residmap_spatial")


if __name__ == "__main__":
    main()
