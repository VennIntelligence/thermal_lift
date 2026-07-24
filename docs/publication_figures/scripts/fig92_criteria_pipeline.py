"""fig92 — The measurement pipeline: how neural and classical reconstructions
are compared honestly (methodology schematic, no data).

Provenance
----------
Content encodes the canonical criteria pipeline from
`research_log/algorithm_changelog.md` 当前有效结论速览 (2026-07-07 snapshot):
  * #4 判据管线: neural x classical comparison requires per-half measured-offset
    correction (`probe_pair_offset.py --save-corrected-dir` or
    `real_eval.to_center_grid`, removing the +0.5 HR px neural grid corner
    convention, ACL-049) BEFORE cross-FRC; self split-half FRC is invalid for
    neural methods (rewards reproducible hallucination, ACL-047; see fig32).
  * #1 (ACL-049): skipping the offset correction depressed all earlier
    neural x classical cross-FRC by ~0.44-0.49 FRC points (rounded to ~0.4
    in the forbidden-branch label; see fig07).
  * #2 (ACL-048): refined per-frame alignment
    `configs/alignment/stage0f_refined_alignment.csv` is the repo default asset.
  * #3 (ACL-048): 20 um is the detector aperture zero; FRC values at 20 um are
    never trusted; real gain band ~25-40 um period.
Input data: 248 real frames; phase-stratified split A/B; each method
reconstructs both halves; symmetrized cross-method FRC vs drizzle
(mean of X_a x Y_b and X_b x Y_a); readout = cutoff + FRC@30um.

Pure matplotlib schematic (FancyBboxPatch + annotate); no numerical inputs.

Run from repo root:
    uv run python docs/publication_figures/scripts/fig92_criteria_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from pubfig_style import W_DOUBLE, save_fig, setup_academic_style

BLUE = "#4C72B0"   # data / pipeline steps
RED = "#C44E52"    # forbidden shortcuts
GRAY = "#666666"   # notes
BOX_FS = 7         # annotation-level text inside boxes


def _box(ax, xc, yc, w, h, text, edge, face, textcolor="#222222",
         lw=1.0, ls="-", fontsize=BOX_FS):
    ax.add_patch(FancyBboxPatch(
        (xc - w / 2, yc - h / 2), w, h,
        boxstyle="round,pad=0.05,rounding_size=0.09",
        linewidth=lw, linestyle=ls, edgecolor=edge, facecolor=face,
        mutation_aspect=1.0, zorder=3))
    if text:
        ax.text(xc, yc, text, ha="center", va="center", fontsize=fontsize,
                color=textcolor, zorder=4, linespacing=1.3)


def _arrow(ax, p0, p1, color, ls="-", rad=0.0, lw=1.1, zorder=2):
    ax.annotate(
        "", xy=p1, xytext=p0,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, linestyle=ls,
                        shrinkA=1.0, shrinkB=1.0, mutation_scale=9,
                        connectionstyle=f"arc3,rad={rad}"),
        zorder=zorder)


def _red_x(ax, x, y, s=0.11):
    """Small bold red X marking a forbidden branch."""
    # White disc behind so the X stays legible on top of the dashed arrow.
    ax.plot([x], [y], marker="o", ms=11, mfc="white", mec="none", zorder=5)
    for sx in (-1, 1):
        ax.plot([x - sx * s, x + sx * s], [y - s, y + s],
                color=RED, lw=1.8, solid_capstyle="round", zorder=6)


def build_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 3.35))
    ax.set_xlim(0, 10.8)
    ax.set_ylim(0, 5.05)
    ax.axis("off")

    # ── Band 1: main pipeline row ────────────────────────────────────
    y0, bh = 4.28, 1.02
    widths = [1.40, 1.18, 1.48, 1.48, 1.62, 1.06]
    gap = (10.8 - 0.5 - sum(widths)) / 5          # 0.25 margin each side
    xs, x = [], 0.25
    for w in widths:
        xs.append(x + w / 2)
        x += w + gap
    steps = [
        "248 real frames\n+ refined alignment\n(stage0f asset)",
        "phase-stratified\nsplit A / B",
        "each method\nreconstructs halves\n$X_a,X_b$ (drz: $Y_a,Y_b$)",
        "per-half offset corr.\n(to_center_grid,\nremoves +0.5 px)",
        "symmetrized cross-FRC\nvs drizzle: mean of\n$X_a{\\times}Y_b$, $X_b{\\times}Y_a$",
        "read out\ncutoff, FRC\n@30 $\\mu$m",
    ]
    for xc, w, txt in zip(xs, widths, steps):
        _box(ax, xc, y0, w, bh, txt, BLUE, "#E7EDF6")
    for i in range(5):
        _arrow(ax, (xs[i] + widths[i] / 2 + 0.06, y0),
               (xs[i + 1] - widths[i + 1] / 2 - 0.06, y0), BLUE)

    # ── Band 2: forbidden shortcuts (both branch off reconstruction) ─
    fy, fh = 2.32, 0.94
    bx_skip, bw_skip = 4.55, 2.40     # skip-offset box
    bx_self, bw_self = 8.15, 2.45     # self split-half FRC box
    _box(ax, bx_skip, fy, bw_skip, fh,
         "skip offset correction\n$\\Rightarrow$ ~0.4 FRC penalty\n"
         "artifact (see fig07)", RED, "#FAEDEE", textcolor=RED, ls="--")
    _box(ax, bx_self, fy, bw_self, fh,
         "self split-half FRC ($X_a{\\times}X_b$)\nrewards reproducible\n"
         "hallucination (see fig32)", RED, "#FAEDEE", textcolor=RED, ls="--")

    src = (xs[2], y0 - bh / 2 - 0.06)             # bottom of reconstruct box
    _arrow(ax, (src[0] - 0.25, src[1]), (bx_skip - 0.35, fy + fh / 2 + 0.10),
           RED, ls="--", rad=-0.08)
    _red_x(ax, 4.02, 3.36)
    _arrow(ax, (src[0] + 0.10, src[1]), (bx_self - 0.55, fy + fh / 2 + 0.10),
           RED, ls="--", rad=0.12)
    _red_x(ax, 6.10, 3.20)

    # ── Band 3: caveat note under the readout + legend at left ──────
    ny, nh = 0.78, 0.78
    _box(ax, 8.35, ny, 4.35, nh,
         "FRC values at 20 $\\mu$m sit on the detector-aperture zero\n"
         "and are never trusted (real gain band: 25–40 $\\mu$m)",
         GRAY, "#F2F2F2", textcolor="#444444", lw=0.8)
    _arrow(ax, (xs[5] + 0.30, y0 - bh / 2 - 0.06), (10.10, ny + nh / 2 + 0.10),
           GRAY, lw=0.8, rad=-0.12)

    lx, ly, dy = 0.62, 2.62, 0.62
    _box(ax, lx, ly, 0.52, 0.28, "", BLUE, "#E7EDF6")
    ax.text(lx + 0.42, ly, "pipeline step (mandatory)", fontsize=7,
            va="center", color="#222222")
    _box(ax, lx, ly - dy, 0.52, 0.28, "", RED, "#FAEDEE", ls="--")
    _red_x(ax, lx, ly - dy, s=0.07)
    ax.text(lx + 0.42, ly - dy, "forbidden shortcut", fontsize=7,
            va="center", color=RED)
    _box(ax, lx, ly - 2 * dy, 0.52, 0.28, "", GRAY, "#F2F2F2", lw=0.8)
    ax.text(lx + 0.42, ly - 2 * dy, "caveat note", fontsize=7,
            va="center", color="#444444")

    return fig


def main() -> None:
    setup_academic_style()
    fig = build_figure()
    paths = save_fig(fig, "fig92_criteria_pipeline")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
