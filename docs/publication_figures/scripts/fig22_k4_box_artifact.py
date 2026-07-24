"""fig22: K4 glow-box artifact — root cause via unroll-step decomposition.

Shows the EP07 solver step decomposition (x0 -> prox1 -> DC1 -> prox2 -> DC2)
on the diagnostic flat ROI: the learned proximal UNet introduces a regular
square "glow box" at its patch-solve boundary at every prox step, the
data-consistency step only partially suppresses it, so the artifact
re-accumulates over unroll steps (motivating the K2 mainline + halo eval).

Data provenance
---------------
Source image (TensorBoard export, archived PNG, already inferno-colormapped):
  research_log/episodes/ep07_solver_boundary_artifact/figures/
      05_step_decompose_temp_x0_prox1_dc1_prox2_dc2.png
  TensorBoard tag: eval_solver_step_decompose_flatroi/temp__x0_prox1_dc1_prox2_dc2
  step 2002 (see figure_manifest.tsv)
Run: algos/ep07_unet_sr/outputs/solver_v7_k2_nodrizzle_flat005_smoke,
  checkpoint solver_step_002000.pt, no-drizzle, patch_size_hr=192, overlap=160.
Context: research_log/episodes/ep07_solver_boundary_artifact/README.md
ACL refs: ACL-037 era (EP07 solver boundary artifact diagnosis).

Caveat: the archived PNG is an 8-bit colormapped TensorBoard export without a
value scale, so no temperature colorbar can be attached; panels are shown as
qualitative temperature maps (inferno).

Run:
  cd /Users/ujs/mycode/thermal_lift && \
      uv run python docs/publication_figures/scripts/fig22_k4_box_artifact.py
"""

from __future__ import annotations

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

SRC = (
    REPO_ROOT
    / "research_log/episodes/ep07_solver_boundary_artifact/figures"
    / "05_step_decompose_temp_x0_prox1_dc1_prox2_dc2.png"
)

PANEL = 320  # each stage tile is 320x320, five tiles side by side, no chrome

STAGES = [
    ("$x_0$ (aligned input)", "clean"),
    ("$\\mathrm{prox}_1$", "glow box appears"),
    ("$\\mathrm{DC}_1$", "partly suppressed"),
    ("$\\mathrm{prox}_2$", "box returns"),
    ("$\\mathrm{DC}_2$", "residual box"),
]

# Zoom window (panel-local px): bottom-left corner of the square glow box
ZX0, ZX1 = 6, 132
ZY0, ZY1 = 194, 320


def main() -> None:
    setup_academic_style()

    strip = np.asarray(Image.open(SRC).convert("RGB"))
    panels = [strip[:, i * PANEL : (i + 1) * PANEL] for i in range(5)]

    fig, axes = plt.subplots(2, 5, figsize=(W_DOUBLE, 3.0))
    # constrained_layout is on repo-wide; tighten the inter-panel gaps
    fig.get_layout_engine().set(w_pad=0.01, h_pad=0.01, wspace=0.015, hspace=0.01)

    for col, (img, (title, verdict)) in enumerate(zip(panels, STAGES)):
        ax_full, ax_zoom = axes[0, col], axes[1, col]

        ax_full.imshow(img)
        ax_full.set_title(title, pad=3)
        ax_full.add_patch(
            mpatches.Rectangle(
                (ZX0, ZY0), ZX1 - ZX0, ZY1 - ZY0,
                fill=False, edgecolor="white", linestyle=(0, (3, 2)), linewidth=0.9,
            )
        )

        ax_zoom.imshow(img[ZY0:ZY1, ZX0:ZX1])
        ax_zoom.set_xlabel(verdict, fontsize=8, fontstyle="italic", labelpad=3)

        for ax in (ax_full, ax_zoom):
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_linewidth(0.5)
                sp.set_color("0.25")

    # Row labels on the left edge
    axes[0, 0].set_ylabel("flat ROI", fontsize=8)
    axes[1, 0].set_ylabel("zoom", fontsize=8)

    # Arrow marking the box edge in the prox1 zoom (zoom-local coords)
    ax = axes[1, 1]
    # box corner in zoom coords: vertical edge x ~= 30-ZX0, bottom edge y ~= 290-ZY0
    ax.annotate(
        "box edge",
        xy=(30 - ZX0, 258 - ZY0), xytext=(66 - ZX0, 226 - ZY0),
        color="white", fontsize=7.5, ha="left", va="center",
        arrowprops=dict(arrowstyle="-|>", color="white", lw=0.9,
                        shrinkA=1, shrinkB=1),
    )
    ax.annotate(
        "",
        xy=(64 - ZX0, 292 - ZY0), xytext=(72 - ZX0, 238 - ZY0),
        arrowprops=dict(arrowstyle="-|>", color="white", lw=0.9,
                        shrinkA=1, shrinkB=1),
    )

    save_fig(fig, "fig22_k4_box_artifact")


if __name__ == "__main__":
    main()
