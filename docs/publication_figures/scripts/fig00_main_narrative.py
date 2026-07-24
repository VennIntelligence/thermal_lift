"""fig00 — The research storyline in four acts, and the verdict (schematic).

Opening figure of the gallery (owner request 2026-07-24): one page that
strings the whole campaign together — physical grounding, the honest
instrument, algorithms & priors, the trade-off law — converging on the
champion verdict. Pure matplotlib schematic in the style of fig92; every
number is transcribed from changelog verdicts and appears again in the
cited gallery chapters:

  * Act 1: 20 µm pitch + forward-operator certification (ACL-023, fig16/19);
    phase occupancy 11/25 (EP15-M1, fig34); sigma robust band (ACL-056..059,
    fig48); noise floor 0.0724 °C (AGENTS.md ground truth).
  * Act 2: authoritative band 25.45 µm (ACL-048, fig21); self-FRC invalid
    (ACL-047, fig32); +0.5 px correction 0.04 -> 0.83 (ACL-049, fig07);
    cross-FRC-vs-drizzle criteria pipeline (fig92).
  * Act 3: solver mainline noSE/noGN + halo-96 + eta*=0.09 (ACL-041/053);
    pool evolution erasure 43% -> 0.00% (ACL-066/070/072/074, fig01);
    prior needs pool scale (ACL-069, fig08).
  * Act 4: fidelity-robustness monotonic trade-off (ACL-079, fig99);
    healthy ceiling 0.6705 (ACL-053, fig91); Pareto choice (fig02/93).
  * Verdict: depb9v6 0.6611 vs TGV 0.7017 (ACL-080 table); OOD 13/13,
    mean delta +0.171 (ACL-079, fig99); erasure 4.66% (ACL-070/071);
    range-exc 2-2.7 (fig17); DC self-audit (ACL-075, fig04); gains in the
    25-40 um band anchored on optical GT (fig66/60).

Run from repo root:
    uv run python docs/publication_figures/scripts/fig00_main_narrative.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

from pubfig_style import W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

BLACK = "#222222"
GRAY = "#555555"
BLUE = "#4C72B0"          # v6 semantic color — reserved for the verdict box

ACTS = [
    ("Act 1 · Physical grounding",
     ["\u2022 20 µm pitch recalibrated; forward",
      "   operator certified, T1–T5 (023)",
      "\u2022 phase occupancy measured 11/25",
      "   → 2x is the supportable factor",
      "\u2022 PSF σ: robust band [0.1, 0.4] px;",
      "   noise floor 0.072 °C"],
     "Gallery Ch. 1"),
    ("Act 2 · An honest instrument",
     ["\u2022 per-frame alignment refined →",
      "   authoritative band 25.45 µm",
      "\u2022 self split-half FRC banned:",
      "   it rewards hallucination",
      "\u2022 +0.5 px grid-convention fix:",
      "   neural cross-FRC 0.04 → 0.83",
      "\u2022 verdict metric = cross-FRC",
      "   vs independent drizzle"],
     "Gallery Ch. 2"),
    ("Act 3 · Algorithms & priors",
     ["\u2022 unrolled prox–DC solver",
      "   (noSE/noGN, halo-96, η* = 0.09)",
      "\u2022 pools v6 → v9-3k: isolated-dot",
      "   erasure 43% → 0.00%",
      "\u2022 erasure prior emerges only at",
      "   pool scale (5k vs 300 scenes)"],
     "Gallery Ch. 3–5"),
    ("Act 4 · The trade-off law",
     ["\u2022 point fidelity vs OOD robustness:",
      "   a monotonic trade-off",
      "\u2022 healthy-arm FRC ceiling ≈ 0.67,",
      "   TGV 0.702 still unbeaten",
      "\u2022 no arm dominates → champion",
      "   selection is a Pareto choice"],
     "Gallery Ch. 6"),
]

VERDICT_TITLE = ("Verdict — champion depb9v6 (v21 solver × v6 pool): "
                 "the balanced arm")
VERDICT_LINES = [
    "\u2022 in-domain: cross-FRC@30µm 0.661 vs TGV 0.702 — the classical "
    "baseline is still unbeaten on the primary metric",
    "\u2022 out-of-domain: wins 13/13 OOD pools vs TGV(oracle), mean Δ +0.171 "
    "— the only robust neural arm",
    "\u2022 fit for inspection: isolated-dot erasure 4.66%; clean low "
    "frequencies (range-exc 2–2.7); auditable via the DC-residual self-check",
    "\u2022 the 2x gains are real and localized: 25–40 µm band, anchored on "
    "real trace edges (optical ground truth)",
]


def _box(ax, x0, y0, w, h, edge, face, lw=1.0):
    ax.add_patch(FancyBboxPatch(
        (x0, y0), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.10",
        linewidth=lw, edgecolor=edge, facecolor=face, zorder=2))


def _arrow(ax, p0, p1, lw=1.2):
    ax.annotate("", xy=p1, xytext=p0,
                arrowprops=dict(arrowstyle="-|>", color=BLACK, lw=lw,
                                shrinkA=1.5, shrinkB=1.5, mutation_scale=11),
                zorder=3)


def build_figure() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(W_DOUBLE, 4.35))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 7.1)
    ax.axis("off")

    # ── Four act boxes ────────────────────────────────────────────────
    box_w, gap, y0, box_h = 2.77, 0.27, 4.35, 2.25
    for i, (title, lines, pointer) in enumerate(ACTS):
        x0 = 0.03 + i * (box_w + gap)
        _box(ax, x0, y0, box_w, box_h, edge="#333333", face="white")
        ax.text(x0 + 0.10, y0 + box_h - 0.14, title,
                fontsize=7.6, fontweight="bold", color="black",
                ha="left", va="top")
        ax.text(x0 + 0.10, y0 + box_h - 0.52, "\n".join(lines),
                fontsize=6.2, color=BLACK, ha="left", va="top",
                linespacing=1.35)
        ax.text(x0 + box_w / 2, y0 - 0.16, pointer,
                fontsize=6.0, color=GRAY, ha="center", va="top",
                style="italic")
        if i < 3:
            _arrow(ax, (x0 + box_w + 0.03, y0 + box_h / 2),
                   (x0 + box_w + gap - 0.03, y0 + box_h / 2))

    # ── Convergence arrow into the verdict ────────────────────────────
    _arrow(ax, (6.0, y0 - 0.42), (6.0, 2.40), lw=1.4)
    ax.text(6.18, 3.15, "multi-axis Pareto decision",
            fontsize=6.4, color=BLACK, ha="left", va="center",
            style="italic")

    # ── Verdict box (blue = v6 semantic color) ────────────────────────
    vy0, vh = 0.42, 1.85
    _box(ax, 0.03, vy0, 11.94, vh, edge=BLUE, face="#EEF2F8", lw=1.5)
    ax.text(0.20, vy0 + vh - 0.15, VERDICT_TITLE,
            fontsize=8.2, fontweight="bold", color="black",
            ha="left", va="top")
    ax.text(0.20, vy0 + vh - 0.58, "\n".join(VERDICT_LINES),
            fontsize=6.5, color=BLACK, ha="left", va="top", linespacing=1.5)
    ax.text(11.90, vy0 - 0.16, "Gallery Ch. 6–7", fontsize=6.0, color=GRAY,
            ha="right", va="top", style="italic")

    ax.set_title("The storyline in four acts — and the verdict", pad=6)

    fig.text(0.01, -0.020,
             "Schematic summary; every number is a transcribed changelog "
             "verdict (ACL-023…080) and reappears with full context in the "
             "cited gallery chapters.",
             fontsize=6.0, color=GRAY, ha="left", va="top")
    return fig


def main() -> None:
    fig = build_figure()
    paths = save_fig(fig, "fig00_main_narrative")
    print("saved:", *paths)


if __name__ == "__main__":
    main()
