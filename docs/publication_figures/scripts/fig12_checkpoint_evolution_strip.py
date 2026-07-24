"""fig12_checkpoint_evolution_strip: checkpoint-evolution image strip, real-data eval.

Two-row strip of real-data center-zoom (3x) temperature-map reconstructions,
sampled across training checkpoints, comparing the unrolled physics solver
against the residual UNet baseline (V10) on the identical eval hook.

Top row:    unrolled solver (v5_sharp_hybrid_solver), steps 5k/10k/15k/20k.
Bottom row: residual UNet (V10, v5_sharp_unet), steps 5k/15k/30k/50k
            (evenly spaced across its longer 5k-50k training run).

Source PNGs are pre-rendered temperature maps (already colormapped with
inferno at generation time) produced by the checkpoint-evolution eval sweep:
  remote_inbox/20260627_checkpoint_evolution/20260628_hybrid_solver/eval_real_png/
    solver_v5_sharp_hybrid_solver_step{5000,10000,15000,20000}_center_zoom3x_temperature.png
    v10_v5_sharp_unet_step{5000,15000,30000,50000}_center_zoom3x_temperature.png

These images are loaded with matplotlib.image.imread and displayed as-is via
imshow (no re-colormapping) since they are already rendered RGB temperature
maps, not raw scalar fields. Each source PNG is a full matplotlib export
(title + axes + colorbar), so we auto-detect and crop to the inner axes
bounding box (locating the black spine rectangle around the image data) to
get a tight temperature-map-only panel for the strip; pixel values inside
that box are untouched.

Context: ACL-029-era hybrid solver vs V10 UNet baseline, both evaluated on
the same real-data eval hook at matched checkpoints, illustrating how each
model's reconstruction sharpens (or fails to) over the course of training.

Run:
    cd /Users/ujs/mycode/thermal_lift && uv run python \
        docs/publication_figures/scripts/fig12_checkpoint_evolution_strip.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import numpy as np

from pubfig_style import save_fig, setup_academic_style, strip_montage

SRC_DIR = (
    Path(__file__).resolve().parent.parent.parent.parent
    / "remote_inbox"
    / "20260627_checkpoint_evolution"
    / "20260628_hybrid_solver"
    / "eval_real_png"
)

SOLVER_STEPS = [5000, 10000, 15000, 20000]
UNET_STEPS = [5000, 15000, 30000, 50000]

SOLVER_TMPL = "solver_v5_sharp_hybrid_solver_step{step}_center_zoom3x_temperature.png"
UNET_TMPL = "v10_v5_sharp_unet_step{step}_center_zoom3x_temperature.png"

ROW_LABELS = ["Unrolled solver (ours)", "Residual UNet (V10)"]


def _step_label(step: int) -> str:
    return f"{step // 1000}k steps"


def _crop_to_axes(img: np.ndarray) -> np.ndarray:
    """Crop a full matplotlib export (title+axes+colorbar) to just the
    imshow panel, by detecting the black axes-spine rectangle.

    Returns the pixel sub-array inside the main axes; the colorbar and
    title text are excluded. Pixel values inside the crop are untouched.
    """
    gray = img[..., :3].mean(axis=2)
    dark = gray < 0.15
    # The axes spine is the single darkest (near-full-width / near-full-height)
    # line; title text glyphs are shorter runs, so a threshold near the max
    # isolates the spine cleanly from the title.
    rowsum = dark.sum(axis=1)
    thresh_row = 0.85 * rowsum.max()
    rows_ok = np.where(rowsum > thresh_row)[0]
    top, bottom = rows_ok.min(), rows_ok.max()

    band = dark[top : bottom + 1, :]
    colsum = band.sum(axis=0)
    thresh_col = 0.85 * colsum.max()
    cols_ok = np.where(colsum > thresh_col)[0]

    clusters: list[list[int]] = [[cols_ok[0]]]
    for c in cols_ok[1:]:
        if c - clusters[-1][-1] <= 3:
            clusters[-1].append(c)
        else:
            clusters.append([c])
    centers = [int(np.mean(c)) for c in clusters]
    left, right = centers[0], centers[1]

    return img[top + 1 : bottom, left + 1 : right]


def main() -> None:
    setup_academic_style()

    rows = [
        (SOLVER_STEPS, SOLVER_TMPL),
        (UNET_STEPS, UNET_TMPL),
    ]

    panels = []
    col_titles = []
    for steps, tmpl in rows:
        panels.append(
            [_crop_to_axes(mpimg.imread(SRC_DIR / tmpl.format(step=step)))
             for step in steps]
        )
        col_titles.append([_step_label(s) for s in steps])

    # Shared checkpoint-strip template (pubfig_style.strip_montage): identical
    # row-label / step-title styling with fig28. Unlike fig28, the sources
    # here are pre-rendered RGB exports auto-scaled per panel at generation
    # time, so no honest shared deg-C colorbar exists; the display convention
    # is disclosed in a small in-figure footnote instead.
    fig, _ = strip_montage(
        panels,
        col_titles=col_titles,
        row_labels=ROW_LABELS,
        panel_aspect=panels[0][0].shape[1] / panels[0][0].shape[0],
        note="Display: pre-rendered inferno temperature maps, auto-scaled per "
             "panel at export; absolute $^\\circ$C scale is not shared across panels.",
    )

    save_fig(fig, "fig12_checkpoint_evolution_strip")


if __name__ == "__main__":
    main()
