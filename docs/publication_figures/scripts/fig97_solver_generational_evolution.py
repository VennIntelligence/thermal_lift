"""fig97_solver_generational_evolution: primitive-ancestor -> champion strip.

Companion to fig12 (owner request 2026-07-13). A 1x3 strip of real-data
center-zoom (3x) temperature-map reconstructions showing the unrolled
solver across THREE training generations, earliest -> champion:

  Panel 1: solver_v4_acl027            step 20k  (ACL-027 era, ~2026-06-26)
  Panel 2: solver_v5_sharp_hybrid      step 20k  (ACL-029 era, THE fig12 one)
  Panel 3: solver_v21_depb9v6 (v6 pool) step 30k final  (the champion, ACL-062+)

HONESTY: this is NOT a controlled ablation. Between panels the pipeline
changed on many axes at once -- training pool (v4 -> v5_sharp -> v6), the
loss/metric redesign (ACL-027) and the grid-convention correction
(ACL-049, worth ~0.44-0.49 FRC on its own), and the matured halo=96
full-frame inference (v21 uses full_halo96; v4/v5 use the older hook).
So the strip shows CUMULATIVE project progress from the primitive ancestor
to the champion, not the effect of any single change. Each panel is a
pre-rendered temperature map with its OWN auto-scale (identical convention
to fig12) -- absolute Celsius levels are not comparable across panels; the
comparison is about structure/sharpness, not absolute temperature.

Source PNGs are full matplotlib exports (title+axes+colorbar); we crop to
the inner imshow panel via the same black-spine detection as fig12. Pixel
values inside the crop are untouched.

Run:
    cd /Users/ujs/mycode/thermal_lift && uv run python \
        docs/publication_figures/scripts/fig97_solver_generational_evolution.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np

from pubfig_style import W_DOUBLE, save_fig, setup_academic_style

ROOT = Path(__file__).resolve().parent.parent.parent.parent
INBOX = ROOT / "remote_inbox"

# (source PNG path, two-line panel title)
PANELS = [
    (INBOX / "20260627_checkpoint_evolution" / "solver_v4_acl027"
     / "solver_step020000_center_zoom3x_temperature.png",
     "v4 solver\n(ACL-027, 20k)"),
    (INBOX / "20260627_checkpoint_evolution" / "20260628_hybrid_solver"
     / "eval_real_png"
     / "solver_v5_sharp_hybrid_solver_step20000_center_zoom3x_temperature.png",
     "v5 hybrid solver\n(ACL-029, 20k) — Fig 12"),
    (INBOX / "20260705_depb9v6_champion"
     / "solver_step30000_full_halo96_center_zoom3x_temperature.png",
     "depb9v6 champion\n(v21 / v6 pool, 30k)"),
]


def _crop_to_axes(img: np.ndarray) -> np.ndarray:
    """Crop a full matplotlib export (title+axes+colorbar) to just the imshow
    panel, by detecting the black axes-spine rectangle (same as fig12)."""
    gray = img[..., :3].mean(axis=2)
    dark = gray < 0.15
    rowsum = dark.sum(axis=1)
    rows_ok = np.where(rowsum > 0.85 * rowsum.max())[0]
    top, bottom = rows_ok.min(), rows_ok.max()

    band = dark[top : bottom + 1, :]
    colsum = band.sum(axis=0)
    cols_ok = np.where(colsum > 0.85 * colsum.max())[0]

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

    imgs = [_crop_to_axes(mpimg.imread(p)) for p, _ in PANELS]
    # width ratios follow each crop's native aspect so all three panels share
    # one common height and fill the strip without residual whitespace
    aspects = [im.shape[1] / im.shape[0] for im in imgs]

    gap_in, lr_in, top_in, bot_in = 0.035, 0.01, 0.36, 0.02
    h_ax = (W_DOUBLE - 2 * lr_in - 2 * gap_in) / sum(aspects)
    fig_h = h_ax + top_in + bot_in
    avg_w = (W_DOUBLE - 2 * lr_in - 2 * gap_in) / 3

    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, fig_h), layout="none",
                             gridspec_kw={"width_ratios": aspects})
    fig.subplots_adjust(left=lr_in / W_DOUBLE, right=1 - lr_in / W_DOUBLE,
                        top=1 - top_in / fig_h, bottom=bot_in / fig_h,
                        wspace=gap_in / avg_w)

    for ax, img, (_, title) in zip(axes, imgs, PANELS):
        ax.imshow(img)
        ax.set_xticks([]), ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(title, fontsize=9)

    save_fig(fig, "fig97_solver_generational_evolution")


if __name__ == "__main__":
    main()
