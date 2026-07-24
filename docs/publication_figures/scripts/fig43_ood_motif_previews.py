"""Fig 43 -- OUT-OF-GRAMMAR motif-family previews (content-axis OOD test).

Visual preview of the FOUR out-of-grammar motif families used in the
ACL-073/ACL-078 content-axis OOD stress test: procedural scene families the
training generator grammar NEVER produces (eval-only pools, seeds
20260930-33). ACL-073 landed the four generator families; ACL-078 ran the
verdict on them (depb9v6 zero sign-flips vs oracle 4/4; the v9 arms lose).
This figure gives the single-glance visual answer to "what do these OOD
scenes actually look like" before the quantitative verdicts (fig03, fig44).

Layout (4 rows x 4 cols, W_DOUBLE): each ROW is one motif family
(top->bottom: Organic blobs / Serial text rows / Concentric rings /
Voronoi cells), each COLUMN one preview scene (0000/0012/0024/0036) from
its eval pool. All tiles are HR-truth temperature maps (cmap=inferno).

Colour-scale choice (stated per plotting-standards honesty rule): scales
are PER-TILE, vmin = tile min, vmax = tile 99.8th percentile, with the
resulting numeric range annotated on every tile. Rationale, from the
actual data: within a family the scenes differ by several degC in both
offset and span (e.g. text_serial scene 0012 spans 19.1-20.3 degC while
scene 0000 spans 22.0-26.5 degC), so a per-row shared scale would render
the low-dT scenes near-black and hide exactly the motif geometry this
figure exists to show -- same precedent as fig20's per-panel scaling.
The 99.8th-pct upper clip exists because every tile carries a sparse
hot-dot defect tail 1-3.5 degC above the structural range (<0.2% of
pixels); clipping lets the motif background structure use the full
colormap while the dots simply saturate to the brightest colour and stay
visible as bright specks. A tick-free reference colorbar (labelled tile
min -> tile p99.8) makes the per-tile normalisation explicit.

Data: remote_inbox/20260713_content2ms/motif_previews/fig43_previews.npz
      (pulled from the Windows-5090 box). 16 keys "{family}__{scene:04d}",
      each a (240, 320) float16 HR-truth temperature array [degC],
      decimated /4 from the 960x1280 hr_temperature_2x.npy of the
      corresponding eval pool. Script fails loud if any key is missing.
Refs: ACL-073 (families + round-2 pool configs landed, eval-only),
      ACL-078 (content-axis OOD verdict).
Run:  uv run python docs/publication_figures/scripts/fig43_ood_motif_previews.py
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from pubfig_style import (
    CMAP_TEMPERATURE,
    REPO_ROOT,
    W_DOUBLE,
    save_fig,
    setup_academic_style,
)

NPZ_PATH = (
    REPO_ROOT / "remote_inbox" / "20260713_content2ms" / "motif_previews"
    / "fig43_previews.npz"
)

# Row order + human-readable family labels (top -> bottom).
FAMILIES = [
    ("organic_blobs", "Organic\nblobs"),
    ("text_serial", "Serial\ntext rows"),
    ("concentric_rings", "Concentric\nrings"),
    ("voronoi_cells", "Voronoi\ncells"),
]
SCENES = [0, 12, 24, 36]
UPPER_PCT = 99.8  # clip the sparse hot-dot tail (<0.2% of pixels)


def main() -> None:
    setup_academic_style()

    data = np.load(NPZ_PATH)
    # Fail loud on any missing key -- never render a partial montage.
    missing = [
        f"{fam}__{s:04d}"
        for fam, _ in FAMILIES
        for s in SCENES
        if f"{fam}__{s:04d}" not in data.files
    ]
    if missing:
        raise KeyError(f"fig43_previews.npz is missing keys: {missing}")

    fig, axes = plt.subplots(4, 4, figsize=(W_DOUBLE, 5.9))

    for r, (fam, label) in enumerate(FAMILIES):
        for c, s in enumerate(SCENES):
            ax = axes[r, c]
            T = data[f"{fam}__{s:04d}"].astype(np.float32)
            vmin = float(T.min())
            vmax = float(np.percentile(T, UPPER_PCT))
            ax.imshow(
                T, cmap=CMAP_TEMPERATURE, vmin=vmin, vmax=vmax,
                interpolation="nearest",
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.5)
            # Per-tile numeric range so the per-tile scaling stays honest.
            ax.text(
                0.03, 0.04, f"{vmin:.1f}–{vmax:.1f} °C",
                transform=ax.transAxes, ha="left", va="bottom",
                fontsize=6.2, color="white",
                bbox=dict(facecolor="black", alpha=0.45, pad=1.2,
                          edgecolor="none"),
            )
            if r == 0:
                ax.set_title(f"scene {s:04d}", fontsize=8, pad=4)
        axes[r, 0].set_ylabel(label, fontsize=9, labelpad=6)

    # Tick-free reference colorbar: scales are per-tile, so numeric ticks
    # would be misleading -- endpoints name the per-tile mapping instead.
    sm = ScalarMappable(norm=Normalize(0.0, 1.0), cmap=CMAP_TEMPERATURE)
    cb = fig.colorbar(
        sm, ax=axes, location="bottom", shrink=0.42, aspect=40, pad=0.015,
    )
    cb.set_ticks([0.0, 1.0])
    cb.set_ticklabels(["tile min", "tile p99.8"])
    cb.ax.tick_params(labelsize=7, length=0)
    cb.set_label(
        "Temperature, per-tile inferno scale (numeric range on each tile)",
        fontsize=7.5,
    )

    fig.suptitle(
        "Four out-of-grammar motif families (never produced by the training "
        "generator grammar):\nthe content-axis OOD stress test of ACL-078 "
        "(eval-only pools, seeds 20260930–33; ACL-073)",
        fontsize=11, y=1.045,
    )

    paths = save_fig(fig, "fig43_ood_motif_previews")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
