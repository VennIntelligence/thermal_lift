"""Fig 25 -- Tile-seam spectral diagnostic (EP07 solver diagnosis, ACL-037/038).

Context: the V8/K4 unrolled solver renders a regular grid artifact when
inference is done as independent 192-px tiles; enlarging the per-solve
context (a bigger outer halo, up to solving the whole 960x1280 frame in one
shot) suppresses the grid, at the cost of a background-lift / over-sharpening
"flocculence" tradeoff (root-caused in DIAGNOSIS.md to GroupNorm+SE breaking
extent-invariance in the prox UNet; see fig24). This figure isolates the
GRID side of that tradeoff in the spatial-frequency domain: the visible grid
is a tile-boundary artifact locked to the tile step, not a target/data
feature, and it collapses monotonically as the solve context grows.

Data provenance / honest adaptation note
-----------------------------------------
outputs/ep07_solver_diag/diag_arrays.py and diag_arrays2.py compute these
metrics from the archived full-frame render arrays
(remote_inbox/ep07_solver_v8_k4_slim_20260630/.../v8k4_step10000_render_arrays.npz,
NOT present in this checkout) and write only SCALAR summaries to JSON --
metrics_arrays.json (per-render dominant seam period + prominence, the
Wiener-Khinchin peak-height-vs-background-noise number DIAGNOSIS.md Table 1
is built from) and metrics_arrays2.json (per-render seam AUTOCORRELATION
sampled at 6 fixed lags: 16/32/48/64/96/128 HR px). Neither file stores the
continuous FFT curve the original quick-look figC_seam_spectrum.png plotted
(that script recomputed rfft() directly from the render arrays, which are
not archived here). So this figure is built from the two JSON scalar/sample
summaries rather than by reproducing figC's exact curve:
  (a) is a genuine data-derived "spectrum along the tiling axis" -- the
      lag-sampled autocorrelation acts as a period-domain proxy for spectral
      power at those periods (Wiener-Khinchin), now shown for all 7 render
      variants (figC only had 3). CAVEAT: this raw |diff|-autocorrelation
      is dominated by the overall smoothness/background-lift of each
      variant's diff field, so it does NOT by itself fall off for
      full_halo96 -- full_halo96 has the highest lift and is in fact
      dragged up at every lag. That is why (b) is needed.
  (b) plots the actual peak-vs-local-background PROMINENCE metric
      (metrics_arrays.json), which is what isolates the periodic grid
      component from the smooth background trend and reproduces
      DIAGNOSIS.md's headline collapse: prominence falls ~16x
      (2556 -> 161) from tiled_p192_o128 to full_halo96 as solve context
      grows, while the measured period stays pinned at the tile
      pitch (32 HR px) for every tiled/halo variant and only becomes
      untethered (~48-49 px, i.e. no real peak) once the grid is gone.

Tile pitch: dense_p192_o160/tiled_p192_o128 stride is nominally 32/64 HR px,
but the empirically dominant seam period measured by diag_arrays.py is
32 HR px for both (a border/interior alternation sub-harmonic of the 64-px
stride) -- so 32 HR px is marked as the tile-pitch fundamental, with 64/96/128
as its harmonics, matching the dashed references drawn on figC.

Run:
    cd /Users/ujs/mycode/thermal_lift && uv run python \
        docs/publication_figures/scripts/fig25_seam_spectrum.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

from pubfig_style import REPO_ROOT, REF_LINE_GRAY, W_DOUBLE, save_fig, setup_academic_style

DIAG_DIR = REPO_ROOT / "outputs" / "ep07_solver_diag"

# Ordered by increasing per-solve context (tile-local -> whole-frame).
ORDER = [
    "aligned_mean",
    "tiled_p192_o128",
    "dense_p192_o160",
    "tile_halo32",
    "tile_halo64",
    "tile_halo96",
    "full_halo96",
]
LABELS = {
    "aligned_mean": "aligned input (no solve)",
    "tiled_p192_o128": "tiled p192/o128",
    "dense_p192_o160": "dense p192/o160",
    "tile_halo32": "tile halo 32",
    "tile_halo64": "tile halo 64",
    "tile_halo96": "tile halo 96",
    "full_halo96": "full halo 96 (whole frame)",
}
LAGS = [16, 32, 48, 64, 96, 128]
TILE_PITCH = 32
HARMONICS = [64, 96, 128]


def main() -> None:
    setup_academic_style()

    with open(DIAG_DIR / "metrics_arrays.json") as fh:
        m1 = json.load(fh)
    with open(DIAG_DIR / "metrics_arrays2.json") as fh:
        m2 = json.load(fh)

    cmap = plt.cm.viridis(np.linspace(0.0, 0.88, len(ORDER)))
    color = dict(zip(ORDER, cmap))
    color["aligned_mean"] = "0.35"

    fig, axes = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.1), width_ratios=[1.15, 1.0])

    # ── (a) seam autocorrelation vs lag: period-domain spectrum proxy ──
    ax = axes[0]
    for k in ORDER:
        ac = m2[k]["seam_ac"]
        vals = [ac[str(L)] for L in LAGS]
        ax.plot(
            LAGS, vals, marker="o", ms=3.2, lw=1.2,
            color=color[k], label=LABELS[k],
            ls="-" if k != "aligned_mean" else "--",
        )
    ax.axvline(TILE_PITCH, **REF_LINE_GRAY)
    ax.annotate("tile pitch\n(32 px)", (TILE_PITCH, 0.06), fontsize=6.5,
                color="#666666", ha="left", va="bottom", xytext=(3, 0),
                textcoords="offset points")
    for h in HARMONICS:
        ax.axvline(h, color="#bbbbbb", ls=":", lw=0.8)
    ax.set_xticks(LAGS)
    ax.set_xlabel("Spatial lag / period [HR px]")
    ax.set_ylabel("Seam autocorrelation (normalized)")
    ax.set_ylim(0, 1.0)
    ax.set_title("(a) Seam autocorrelation along tiling axis", loc="left", fontsize=9)
    ax.legend(loc="lower left", fontsize=6, ncol=1, frameon=False)
    ax.annotate(
        "raw |diff| autocorr rises with\nbackground lift too (not periodicity-\n"
        "specific) -- see (b) for the isolated\ngrid-only signal",
        (0.98, 0.06), xycoords="axes fraction", ha="right", va="bottom",
        fontsize=6.3, color="#555555",
    )

    # ── (b) seam-peak prominence vs solve-context order (log y) ──
    ax = axes[1]
    ctx_order = [k for k in ORDER if k != "aligned_mean"]
    prom = [max(m1[k]["seam_prom_x"], 1e-3) for k in ctx_order]
    periods = [m1[k]["seam_period_x"] for k in ctx_order]
    xs = np.arange(len(ctx_order))
    bar_colors = [color[k] for k in ctx_order]
    ax.bar(xs, prom, color=bar_colors, width=0.62, edgecolor="white", linewidth=0.4)
    ax.set_yscale("log")
    for x, p, per in zip(xs, prom, periods):
        real_peak = per is not None and abs(per - TILE_PITCH) < 4
        txt = f"{p:.0f}" + ("" if real_peak else "\n(no real peak,\nperiod~{:.0f})".format(per))
        ax.annotate(txt, (x, p), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=6.2)
    ax.set_xticks(xs)
    ax.set_xticklabels([LABELS[k].replace(" (whole frame)", "") for k in ctx_order],
                        rotation=38, ha="right", fontsize=6.5)
    ax.set_ylabel("Seam-peak prominence (peak / local background)")
    ax.set_ylim(1e-1, 1e4)
    ax.grid(axis="y", which="major", alpha=0.3, linewidth=0.5)
    ax.set_title("(b) Grid strength collapses with solve context", loc="left", fontsize=9)
    ax.annotate(
        "tiled$\\rightarrow$full-halo: 16$\\times$ drop\n(DIAGNOSIS.md Table 1)",
        (0.97, 0.95), xycoords="axes fraction", ha="right", va="top",
        fontsize=6.5, color="#333333",
    )

    paths = save_fig(fig, "fig25_seam_spectrum")
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
