"""fig34 — Sub-pixel phase occupancy at 5x: why 4x/5x SR is starved.

EP15 M1 phase-structure measurement under stage0f-era instruments
(20 um pixel pitch, 248 clean SR frames, reference frame 6_16_0.txt).

Provenance
----------
Data (remote_inbox/20260704_stage0f/):
  - t0e_m1_phase_summary.csv            per-alignment-source occupancy/entropy/match-rate
  - t0e_m1_detector_bin_counts_5x.csv   5x5 detector-axis phase-bin counts per source
  - t0e_m1_stage_lattice_counts_5x.csv  5x5 stage-coordinate lattice counts (not plotted;
                                        all sources fill 25/25 there)
  - t0e_m1_phase_structure_summary.json full M1 decision record
Context: research_log/episodes/ep15_info_limit/README.md (M1 section).

Story
-----
All three alignment sources fill the stage-coordinate 5x lattice (25/25), so
2 um-equivalent phase diversity exists in the sampling geometry — enough for 2x.
But on the detector axis (the axis that actually feeds SR reconstruction), the
measured contour_refined alignment collapses to 11/25 occupied bins with counts
piled in the corners (max 62 of 248 frames in one bin), entropy fraction 0.52,
and only 6.5% agreement with command cell labels. The near-uniform command_prior
grid is a commanded prior, not a measurement. Hence >2x SR is phase-starved.

Run from repo root:
  uv run python docs/publication_figures/scripts/fig34_phase_occupancy.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pubfig_style import (  # noqa: E402
    CMAP_COVERAGE,
    METHOD_PALETTE,
    REPO_ROOT,
    W_DOUBLE,
    save_fig,
    setup_academic_style,
)

DATA_DIR = REPO_ROOT / "remote_inbox" / "20260704_stage0f"

HEATMAP_METHODS = ["command_prior", "contour_refined"]
HEATMAP_TITLES = {
    "command_prior": "(a) Command prior\n(commanded, not measured)",
    "contour_refined": "(b) Contour-refined\n(measured)",
}
BAR_METHODS = ["command_prior", "ncc_init", "contour_refined"]
BAR_LABELS = ["Command\nprior", "NCC\ninit", "Contour\nrefined"]


def load_detector_grids() -> dict[str, np.ndarray]:
    grids: dict[str, np.ndarray] = {}
    with open(DATA_DIR / "t0e_m1_detector_bin_counts_5x.csv", newline="") as f:
        for row in csv.DictReader(f):
            g = grids.setdefault(row["method"], np.zeros((5, 5), dtype=int))
            g[int(row["detector_bin_y"]), int(row["detector_bin_x"])] = int(row["count"])
    return grids


def load_summary() -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    with open(DATA_DIR / "t0e_m1_phase_summary.csv", newline="") as f:
        for row in csv.DictReader(f):
            out[row["method"]] = {k: float(v) for k, v in row.items() if k != "method"}
    return out


def main() -> None:
    setup_academic_style()
    import matplotlib.pyplot as plt

    grids = load_detector_grids()
    summ = load_summary()
    n_frames = int(summ["command_prior"]["n_frames"])

    fig = plt.figure(figsize=(W_DOUBLE, 2.9))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.55], wspace=0.16)

    # ── (a) detector-axis 5x phase-bin occupancy heatmaps ──────────────
    vmax = max(int(grids[m].max()) for m in HEATMAP_METHODS)
    im = None
    for i, method in enumerate(HEATMAP_METHODS):
        ax = fig.add_subplot(gs[0, i])
        g = grids[method]
        im = ax.imshow(g, cmap=CMAP_COVERAGE, vmin=0, vmax=vmax, origin="upper")
        for y in range(5):
            for x in range(5):
                c = g[y, x]
                ax.text(
                    x, y, str(c),
                    ha="center", va="center", fontsize=7,
                    color="white" if c < 0.55 * vmax else "black",
                )
        occ = int(summ[method]["detector_bin_occupied"])
        ax.set_title(HEATMAP_TITLES[method], fontsize=9)
        ax.set_xlabel(f"Phase bin x  ({occ}/25 occ.)")
        if i == 0:
            ax.set_ylabel("Phase bin y")
        else:
            ax.set_yticklabels([])
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        for s in ax.spines.values():
            s.set_visible(False)
        ax.tick_params(length=0)
    cbar = fig.colorbar(im, ax=[fig.axes[0], fig.axes[1]], fraction=0.046,
                        pad=0.03, location="right")
    cbar.set_label(f"Frame count (of {n_frames})", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    # ── (c) summary metrics across alignment sources ───────────────────
    axc = fig.add_subplot(gs[0, 2])
    metrics = [
        ("detector_bin_occupied", 25.0, "Detector 5$\\times$ occupancy (/25)", METHOD_PALETTE["primary"]),
        ("detector_bin_entropy_fraction", 1.0, "Detector entropy fraction", METHOD_PALETTE["secondary"]),
        ("stage_lattice_match_rate", 1.0, "Command-label match rate", METHOD_PALETTE["accent_1"]),
    ]
    xg = np.arange(len(BAR_METHODS))
    bw = 0.26
    for j, (key, denom, label, color) in enumerate(metrics):
        vals = [summ[m][key] / denom for m in BAR_METHODS]
        bars = axc.bar(xg + (j - 1) * bw, vals, width=bw - 0.02, color=color, label=label)
        for b, v in zip(bars, vals):
            axc.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=7)
    axc.axhline(1.0, color="#666666", ls="--", lw=0.9)
    axc.text(len(BAR_METHODS) - 0.55, 1.02, "full (needed for 5$\\times$)",
             fontsize=7, color="#666666", ha="right", va="bottom")
    axc.set_xticks(xg)
    axc.set_xticklabels(BAR_LABELS)
    axc.set_ylim(0, 1.30)
    axc.set_ylabel("Fraction of full [-]")
    axc.set_title("(c) Coverage vs. alignment source", fontsize=9)
    axc.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=1,
               frameon=False)
    axc.grid(axis="y", alpha=0.3, linewidth=0.5)
    axc.set_axisbelow(True)

    paths = save_fig(fig, "fig34_phase_occupancy")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
