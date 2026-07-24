"""fig23_halo_sweep: outer solve halo suppresses the prox boundary box; quality vs cost.

Panel (a): image strip on the identical real-data flat-ROI crop, sweeping the
solve-context halo (0 / 64 / 96 / 128 HR px) next to the aligned multi-frame
input baseline. The learned proximal UNet's square "glow box" boundary
response is strong at halo 0, visibly reduced at 64, suppressed at 96, and
128 adds no visible gain over 96 (EP07 core finding, ACL-038).

Panel (b): cost of the full-frame outer-halo mitigation on an RTX 5090 -
wall time and peak memory vs halo size, as two separately-scaled mini-panels
(never dual axes). Peak memory is a run-local diagnostic: CUDA allocator
reuse / kernel plan caching affect the reported peak (halo128 reads below
halo0), so the halo128 point is drawn with an open marker and flagged.

Sources
-------
Image strip:
  research_log/episodes/ep07_solver_boundary_artifact/figures/
    08_flatroi_halo_temp_aligned_halo0_64_96_128.png
  (TensorBoard tag eval_solver_halo_flatroi/temp__aligned_halo0_halo64_halo96_halo128,
   step 2003, per figure_manifest.tsv; run solver_v7_k2_nodrizzle_flat005_smoke,
   checkpoint solver_step_002000.pt, K=2, no-drizzle). The archived PNG is a
   raw 1600x320 TensorBoard strip of five 320x320 pre-rendered inferno panels
   with no matplotlib chrome, so panels are sliced at fixed 320-px offsets
   (no _crop_to_axes detection needed) and shown as-is without re-colormapping.
   The TB export carries no absolute temperature scale, hence no colorbar.

Cost table:
  docs/publication_figures/data/halo_sweep.csv, transcribed (ACL-038) from the
  "Full-Frame Outer-Halo Memory Notes" table in
  research_log/episodes/ep07_solver_boundary_artifact/README.md.

Run:
    cd /Users/ujs/mycode/thermal_lift && uv run python \
        docs/publication_figures/scripts/fig23_halo_sweep.py
"""

from __future__ import annotations

import csv

import matplotlib.image as mpimg
import matplotlib.pyplot as plt

from pubfig_style import (
    DATA_DIR,
    METHOD_PALETTE,
    REF_LINE_GRAY,
    REPO_ROOT,
    W_DOUBLE,
    save_fig,
    setup_academic_style,
)

STRIP_PNG = (
    REPO_ROOT
    / "research_log"
    / "episodes"
    / "ep07_solver_boundary_artifact"
    / "figures"
    / "08_flatroi_halo_temp_aligned_halo0_64_96_128.png"
)
CSV_PATH = DATA_DIR / "halo_sweep.csv"

PANEL_PX = 320  # each TB sub-panel is 320x320 px, concatenated horizontally
PANEL_TITLES = [
    "Aligned input",
    "Halo 0",
    "Halo 64",
    "Halo 96",
    "Halo 128",
]
RECOMMENDED_HALO = 96  # EP07 recommendation; 128 shows no visible gain


def _load_cost_table() -> dict[str, list[float]]:
    cols: dict[str, list[float]] = {
        "halo_hr_px": [],
        "wall_time_s": [],
        "peak_mem_gb": [],
    }
    with open(CSV_PATH, newline="") as fh:
        rows = [r for r in csv.reader(fh) if r and not r[0].startswith("#")]
    header = rows[0]
    for row in rows[1:]:
        rec = dict(zip(header, row))
        for key in cols:
            cols[key].append(float(rec[key]))
    return cols


def main() -> None:
    setup_academic_style()

    strip = mpimg.imread(STRIP_PNG)
    n_panels = strip.shape[1] // PANEL_PX
    assert n_panels == len(PANEL_TITLES), (n_panels, strip.shape)

    cost = _load_cost_table()
    halo = cost["halo_hr_px"]

    # ── Layout: top row = 5-panel image strip, bottom row = 2 cost panels ──
    img_row_h = W_DOUBLE / 5  # square panels
    cost_row_h = 1.35
    fig_h = img_row_h + cost_row_h + 0.55
    fig = plt.figure(figsize=(W_DOUBLE, fig_h), constrained_layout=True)
    fig.get_layout_engine().set(h_pad=0.02, w_pad=0.02, hspace=0.04)
    top_fig, bot_fig = fig.subfigures(
        2, 1, height_ratios=[img_row_h + 0.25, cost_row_h]
    )

    # ── (a) image strip ───────────────────────────────────────────────
    axes_img = top_fig.subplots(1, n_panels)
    for i, ax in enumerate(axes_img):
        panel = strip[:, i * PANEL_PX : (i + 1) * PANEL_PX]
        ax.imshow(panel)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(PANEL_TITLES[i], fontsize=9, fontweight="normal", pad=3)
    axes_img[0].set_ylabel("Flat-ROI crop", fontsize=9)
    # Point at the square boundary-box edge in the halo-0 panel.
    axes_img[1].annotate(
        "box edge",
        xy=(185, 300),
        xytext=(70, 240),
        color="white",
        fontsize=7,
        ha="center",
        arrowprops=dict(arrowstyle="-", color="white", lw=0.7),
    )
    axes_img[0].text(
        -0.32, 1.14, "(a)", transform=axes_img[0].transAxes,
        fontsize=10, fontweight="bold", va="top",
    )

    # ── (b) cost mini-panels: wall time / peak memory (no dual axes) ──
    ax_t, ax_m = bot_fig.subplots(1, 2)

    ax_t.plot(halo, cost["wall_time_s"], marker="o",
              color=METHOD_PALETTE["primary"])
    ax_t.set_xlabel("Outer halo [HR px]")
    ax_t.set_ylabel("Wall time [s]")
    ax_t.set_ylim(5.5, 7.3)
    ax_t.text(
        -0.28, 1.18, "(b)", transform=ax_t.transAxes,
        fontsize=10, fontweight="bold", va="top",
    )

    mem = cost["peak_mem_gb"]
    ax_m.plot(halo[:-1], mem[:-1], marker="s",
              color=METHOD_PALETTE["accent_3"])
    # halo128 peak-memory reading is allocator-dependent (below halo0):
    # open marker + dashed link, flagged in the annotation.
    ax_m.plot(halo[-2:], mem[-2:], ls=":", lw=1.0,
              color=METHOD_PALETTE["accent_3"])
    ax_m.plot(halo[-1], mem[-1], marker="s", mfc="white",
              color=METHOD_PALETTE["accent_3"], ls="none")
    ax_m.annotate(
        "allocator-\ndependent",
        xy=(halo[-1], mem[-1]),
        xytext=(halo[-1] - 26, mem[-1] + 1.6),
        fontsize=7,
        ha="right",
        va="center",
        arrowprops=dict(arrowstyle="-", color="#666666", lw=0.6),
    )
    ax_m.set_xlabel("Outer halo [HR px]")
    ax_m.set_ylabel("Peak GPU memory [GB]")
    ax_m.set_ylim(8, 17.5)

    for ax in (ax_t, ax_m):
        ax.set_xticks(halo)
        ax.axvline(RECOMMENDED_HALO, **REF_LINE_GRAY)
    ax_t.text(RECOMMENDED_HALO - 4, 5.62, "recommended",
              fontsize=7, color="#666666", ha="right")

    save_fig(fig, "fig23_halo_sweep")


if __name__ == "__main__":
    main()
