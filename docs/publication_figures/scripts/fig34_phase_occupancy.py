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
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    # Manual-layout script: keep the constrained engine off, otherwise the
    # tight-bbox save restores it and warns/re-lays out the 2nd (pdf) save.
    mpl.rcParams["figure.constrained_layout.use"] = False

    grids = load_detector_grids()
    summ = load_summary()
    n_frames = int(summ["command_prior"]["n_frames"])

    # Manual inch-based geometry: three identical square panel boxes with
    # aligned tops/bottoms; slim shared colorbar right of (b); legend as a
    # side column right of (c) so nothing hangs below the panel row.
    P = 1.38                  # square panel edge [in]
    LEFT = 0.44               # ylabel "Phase bin y" + tick labels [in]
    GAP_AB = 0.18             # gap between (a) and (b) [in]
    CB_PAD, CB_W = 0.06, 0.09   # colorbar pad / width [in]
    CB_TXT = 0.44             # colorbar tick labels + rotated label [in]
    C_AXIS = 0.52             # panel-(c) ytick labels + ylabel [in]
    LEGEND_W = 1.30           # side legend column [in]
    TOP, BOTTOM = 0.40, 0.38  # 2-line titles / xlabels + tick labels [in]
    x_a = LEFT
    x_b = x_a + P + GAP_AB
    x_cb = x_b + P + CB_PAD
    x_c = x_cb + CB_W + CB_TXT + C_AXIS
    FIG_W = x_c + P + LEGEND_W
    FIG_H = BOTTOM + P + TOP

    fig = plt.figure(figsize=(FIG_W, FIG_H))

    def add_panel(x_in: float):
        return fig.add_axes([x_in / FIG_W, BOTTOM / FIG_H, P / FIG_W, P / FIG_H])

    # ── (a) detector-axis 5x phase-bin occupancy heatmaps ──────────────
    vmax = max(int(grids[m].max()) for m in HEATMAP_METHODS)
    im = None
    for i, method in enumerate(HEATMAP_METHODS):
        ax = add_panel([x_a, x_b][i])
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
    cax = fig.add_axes([x_cb / FIG_W, BOTTOM / FIG_H, CB_W / FIG_W, P / FIG_H])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(f"Frame count (of {n_frames})", fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    # ── (c) summary metrics across alignment sources ───────────────────
    axc = add_panel(x_c)
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
        # stagger the middle metric's value label so neighbours don't merge
        dy = 0.10 if j == 1 else 0.02
        for b, v in zip(bars, vals):
            axc.text(b.get_x() + b.get_width() / 2, v + dy, f"{v:.2f}",
                     ha="center", va="bottom", fontsize=6.5)
    axc.axhline(1.0, color="#666666", ls="--", lw=0.9)
    axc.annotate("full (needed for 5$\\times$)",
                 xy=(1.02, 1.0), xycoords=("axes fraction", "data"),
                 fontsize=7, color="#666666", ha="left", va="center")
    axc.set_xticks(xg)
    axc.set_xticklabels(BAR_LABELS)
    axc.set_ylim(0, 1.30)
    axc.set_ylabel("Fraction of full [-]")
    axc.set_title("(c) Coverage vs. alignment source", fontsize=9)
    axc.legend(loc="center left", bbox_to_anchor=(1.04, 0.5), ncol=1,
               frameon=False, fontsize=7, handlelength=1.2,
               handletextpad=0.5, borderaxespad=0.0)
    axc.grid(axis="y", alpha=0.3, linewidth=0.5)
    axc.set_axisbelow(True)

    paths = save_fig(fig, "fig34_phase_occupancy")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
