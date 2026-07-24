"""fig64_halo_training_zoom: training-step evolution of the halo=96 full-frame solve.

2x3 montage of archival de_pb9 probe renders showing how the unrolled solver's
full-frame output (halo=96 padding, the EP07-validated inference setting; see
fig23_halo_sweep for the quantitative halo story) evolves over training.
Top row: center 3x-zoom temperature renders at steps 5k / 20k / 40k, exactly
as emitted by the probe (baked-in title + colorbar, colormap inherited from
the probe, not restyled). Bottom row: TensorBoard high-pass detail boards at
the same steps, cropped to one consistent centered band (identical pixel crop
across all three steps, chosen to match the temperature panels' aspect ratio
and centered on the interconnect zigzag region) so the steps remain
pixel-comparable.

Sources
-------
output/de_pb9_probe/
    solver_step{5000,20000,40000}_full_halo96_center_zoom3x_temperature.jpg
    tb_highpass_step{5000,20000,40000}.jpg
Provenance: de_pb9 probe era, early July 2026 (training-step evolution
visuals of the unrolled solver evaluated full-frame with halo=96). These are
already-rendered archival JPGs; they are shown as-is (crop only, no
re-colormapping, no absolute scale added).

Run:
    cd /Users/ujs/mycode/thermal_lift && uv run python \
        docs/publication_figures/scripts/fig64_halo_training_zoom.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

PROBE_DIR = REPO_ROOT / "output" / "de_pb9_probe"

STEPS = (5000, 20000, 40000)
STEP_LABELS = ("step 5k", "step 20k", "step 40k")
ROW_LABELS = ("temperature\n(center 3x zoom)", "TB high-pass\nboard (crop)")


def _crop_highpass(im, target_aspect: float):
    """Crop a vertically-centered band so the board matches target W/H aspect.

    The 1281x960 TensorBoard high-pass boards are taller (aspect 1.33) than the
    1288x795 temperature renders (aspect 1.62). One fixed centered band -- same
    pixel offsets for every step -- keeps the three steps pixel-comparable and
    centers on the interconnect zigzag region.
    """
    h, w = im.shape[:2]
    target_h = int(round(w / target_aspect))
    if target_h >= h:
        return im
    y0 = (h - target_h) // 2
    return im[y0 : y0 + target_h]


def main() -> None:
    setup_academic_style()

    temp_ims = [
        mpimg.imread(
            PROBE_DIR / f"solver_step{s}_full_halo96_center_zoom3x_temperature.jpg"
        )
        for s in STEPS
    ]
    hp_ims = [mpimg.imread(PROBE_DIR / f"tb_highpass_step{s}.jpg") for s in STEPS]

    temp_aspect = temp_ims[0].shape[1] / temp_ims[0].shape[0]
    hp_ims = [_crop_highpass(im, temp_aspect) for im in hp_ims]

    # Exact panel geometry (constrained layout disabled below): row-label
    # gutter left, column-label band top, footnote band bottom.
    left_in, right_in = 0.38, 0.02
    top_in, bottom_in, row_gap_in, col_gap_in = 0.24, 0.30, 0.06, 0.05
    panel_w = (W_DOUBLE - left_in - right_in - 2 * col_gap_in) / 3.0
    panel_h = panel_w / temp_aspect
    fig_h = top_in + 2 * panel_h + row_gap_in + bottom_in
    fig, axes = plt.subplots(2, 3, figsize=(W_DOUBLE, fig_h), layout="none")
    fig.subplots_adjust(
        left=left_in / W_DOUBLE,
        right=1 - right_in / W_DOUBLE,
        top=1 - top_in / fig_h,
        bottom=bottom_in / fig_h,
        wspace=col_gap_in / panel_w,
        hspace=row_gap_in / panel_h,
    )

    for col, (t_im, h_im, lbl) in enumerate(zip(temp_ims, hp_ims, STEP_LABELS)):
        for row, im in enumerate((t_im, h_im)):
            ax = axes[row, col]
            ax.imshow(im)
            ax.set_aspect("equal")
            ax.set_axis_off()
        axes[0, col].set_title(lbl, fontsize=9, pad=4)

    for row, lbl in enumerate(ROW_LABELS):
        axes[row, 0].set_axis_on()
        axes[row, 0].set_xticks([])
        axes[row, 0].set_yticks([])
        for spine in axes[row, 0].spines.values():
            spine.set_visible(False)
        axes[row, 0].set_ylabel(lbl, fontsize=8)

    fig.text(
        0.5,
        0.02,
        "Archival TensorBoard / de_pb9 probe renders (full-frame solve, halo=96); "
        "colormaps inherited from the probe, not restyled.",
        ha="center",
        va="bottom",
        fontsize=7,
        style="italic",
        color="0.35",
    )

    save_fig(fig, "fig64_halo_training_zoom")


if __name__ == "__main__":
    main()
