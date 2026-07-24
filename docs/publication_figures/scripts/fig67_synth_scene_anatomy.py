"""Fig 67 -- anatomy of a synthetic training scene (v7 pool demo).

Shows, for three visually distinct scenes from the v7-era demo pool, the
three raster fields the procedural scene generator ("grammar") emits per
training scene:

  col 1 -- HR ground-truth temperature T (960x1280, 10 um/px; cmap=inferno)
  col 2 -- LR forward-model observation `lr` (480x640, 20 um/px;
           cmap=inferno on the SAME shared temperature scale as T, so the
           blur/downsample loss of contrast is directly readable)
  col 3 -- soft panel-coverage map `cov` (960x1280, 0=background,
           1=panel interior; cmap=viridis). This is the module-layout
           occupancy mask used to compute the `occ` scalar, not a defect
           mask.

Scenes were hand-picked from the 50-scene minipool to span the pool's
structural range (metadata printed above each row):

  scene_006 -- mid tier,  occ=0.02 (sparsest layout in the pool)
  scene_022 -- high tier, dT=2.94 degC (largest thermal swing)
  scene_048 -- high tier, occ=0.33 (densest layout in the pool)

All temperature panels (T and lr) share ONE colorbar (global min/max across
all three scenes' T; percentiles are background-dominated and clip whole
scenes), drawn bottom-left at one column width. The coverage column carries
no second colorbar; its 0-1 mapping is stated in a one-line text note
spanning the remaining two columns of the bottom strip. Scale bar on the
top-left panel: 30 HR px = 300 um (HR grid 10 um/px; detector pitch 20 um,
SCALE=2x -- see repo memory note on the 20 um pixel-pitch recalibration).

Data: outputs/v7_demo_minipool/scene_NNN.npz (v7-era pool demo generated
      by scripts/generate_v7_demo_minipool.py; keys T/lr/cov float16
      rasters + occ/dT/sigma/angle/tier scalars).
Run:  uv run python docs/publication_figures/scripts/fig67_synth_scene_anatomy.py
"""

from __future__ import annotations

import numpy as np

from pubfig_style import (
    REPO_ROOT,
    W_DOUBLE,
    plt,
    save_fig,
    setup_academic_style,
)

POOL_DIR = REPO_ROOT / "outputs" / "v7_demo_minipool"
SCENES = [6, 22, 48]  # sparse-mid / high-dT-high / dense-high
UM_PER_HR_PX = 10.0   # HR grid pitch (detector 20 um, SCALE=2x)
SCALEBAR_PX = 30      # 30 px * 10 um/px = 300 um


def main() -> None:
    setup_academic_style()

    scenes = []
    for i in SCENES:
        d = np.load(POOL_DIR / f"scene_{i:03d}.npz")
        scenes.append(
            dict(
                i=i,
                T=d["T"].astype(np.float32),
                lr=d["lr"].astype(np.float32),
                cov=d["cov"].astype(np.float32),
                occ=float(d["occ"]),
                dT=float(d["dT"]),
                sigma=float(d["sigma"]),
                angle=float(d["angle"]),
                tier=str(d["tier"]),
            )
        )

    # Shared temperature scale across ALL T and lr panels. Full min/max
    # (not percentiles): scenes differ by degC in mean level, and pixel
    # percentiles are background-dominated, which clips whole scenes.
    vmin = min(s["T"].min() for s in scenes)
    vmax = max(s["T"].max() for s in scenes)

    # Manual grid geometry (no constrained layout): panel boxes carry the
    # raster aspect (960x1280 -> 3:4) so every image fills its box exactly,
    # and the figure height is derived from the content instead of guessed.
    aspect = scenes[0]["T"].shape[0] / scenes[0]["T"].shape[1]
    n_rows = n_cols = 3
    ws, hs = 0.04, 0.13            # grid gaps, fraction of panel width/height
    left, right = 0.004, 0.996
    top_in, bot_in = 0.40, 0.44    # inches: col titles + row meta / colorbar strip
    ax_w = W_DOUBLE * (right - left) / (n_cols + (n_cols - 1) * ws)
    ax_h = ax_w * aspect
    grid_h = ax_h * (n_rows + (n_rows - 1) * hs)
    fig_h = grid_h + top_in + bot_in

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(W_DOUBLE, fig_h),
        sharex=False, sharey=False,
        gridspec_kw=dict(
            left=left, right=right, top=1.0 - top_in / fig_h,
            bottom=bot_in / fig_h, wspace=ws, hspace=hs,
        ),
    )
    # Freeze the manual geometry: downgrading the rcParams-provided
    # constrained-layout engine to the do-nothing placeholder (a bare None
    # engine would be resurrected to constrained layout by savefig's
    # rcParams round-trip, re-laying the grid out in the saved PDF).
    fig.set_layout_engine("none")

    col_titles = ["HR truth $T$", "LR observation", "Coverage map"]
    im_T = None
    for r, s in enumerate(scenes):
        im_T = axes[r, 0].imshow(s["T"], cmap="inferno", vmin=vmin, vmax=vmax)
        axes[r, 1].imshow(s["lr"], cmap="inferno", vmin=vmin, vmax=vmax)
        axes[r, 2].imshow(s["cov"], cmap="viridis", vmin=0.0, vmax=1.0)
        for c in range(3):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
            for sp in axes[r, c].spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.5)
        # Row metadata label ABOVE the row (left-aligned on left panel),
        # kept below the column titles on the top row via title pad.
        meta = (
            f"scene {s['i']:03d}:  {s['tier']} tier,  "
            f"$\\Delta T$={s['dT']:.2f} °C,  $\\sigma$={s['sigma']:.2f},  "
            f"$\\theta$={s['angle']:.0f}°,  occ={s['occ']:.2f}"
        )
        axes[r, 0].text(
            0.0, 1.03, meta, transform=axes[r, 0].transAxes,
            ha="left", va="bottom", fontsize=7, clip_on=False,
        )

    for c, t in enumerate(col_titles):
        axes[0, c].set_title(t, pad=13)

    # Scale bar on the top-left panel: 30 HR px = 300 um.
    ax0 = axes[0, 0]
    H, W = scenes[0]["T"].shape
    x0, y0 = 0.05 * W, 0.92 * H
    ax0.plot([x0, x0 + SCALEBAR_PX], [y0, y0], color="white", lw=2.5,
             solid_capstyle="butt")
    ax0.text(x0 + SCALEBAR_PX + 0.015 * W, y0, "300 µm", color="white",
             ha="left", va="center", fontsize=7)

    # ONE colorbar only (temperature, shared by all T/lr panels), bottom-left
    # at one column width; the coverage 0-1 mapping is a text note spanning
    # the remaining two columns instead of a second stacked colorbar.
    col_w = (right - left) / (n_cols + (n_cols - 1) * ws)  # figure fraction
    cax = fig.add_axes(
        [left + 0.02 * col_w, 0.28 / fig_h, 0.96 * col_w, 0.10 / fig_h]
    )
    cb_T = fig.colorbar(im_T, cax=cax, orientation="horizontal")
    cb_T.set_label("Temperature [°C]", fontsize=8)
    cb_T.ax.tick_params(labelsize=7)

    note_x0 = left + col_w * (1.0 + ws)  # left edge of the middle column
    fig.text(
        (note_x0 + right) / 2.0, 0.29 / fig_h,
        "HR truth $T$ and LR observation share the temperature scale (left).\n"
        "Coverage map: viridis, 0 = background $\\rightarrow$ 1 = panel interior.",
        ha="center", va="center", fontsize=7,
    )

    paths = save_fig(fig, "fig67_synth_scene_anatomy")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
