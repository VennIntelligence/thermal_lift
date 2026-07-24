"""fig28_v4era_checkpoint_strip: solver checkpoint evolution from raw temperature arrays.

Predecessor to fig12_checkpoint_evolution_strip.py, one training arm earlier
(ACL-027-era solver run, right after the ACL-027 loss/metric redesign, vs.
fig12's ACL-029-era hybrid solver). Unlike fig12 -- which composites
pre-rendered TensorBoard PNG exports and has to crop out an embedded
colorbar/axes frame -- this figure reads the underlying full-frame absolute
temperature .npz arrays directly, so it gets one real, shared, honestly-scaled
colorbar in deg C instead of a per-panel auto-scaled PNG export.

One row of 5 panels, single arm (solver_v4_acl027), steps 2.5k/5k/10k/15k/20k.
Each panel is the same center-detail crop used in fig10_real_visual_montage.py
(y0=361, x0=522, size=128 HR px) -- verified below that this checkpoint-eval
array grid (960x1280) matches fig10's real-data HR grid, so the same crop
window lands on the same physical scene region (fine PCB-trace detail).

Data: remote_inbox/20260627_checkpoint_evolution/solver_v4_acl027/
    solver_step{002500,005000,010000,015000,020000}_temperature_c.npz
    (key "temperature_c", float32, full-frame absolute temperature in deg C,
    shape (960, 1280) -- confirmed by direct inspection).
Context: remote_inbox/20260627_checkpoint_evolution/README.md + manifest.json
    -- this arm trained immediately after the ACL-027 loss/metric redesign;
    solver real-data rendering uses the configured scalar Gaussian PSF
    because real data carries no synthetic per-scene PSF metadata.

Colormap: inferno (temperature). vmin/vmax shared across all 5 panels, set
from the actual min/max of the 5 crop regions (not a cherry-picked per-panel
scale) -- this is the "honest" comparison the raw-array route buys us over
fig12's already-colormapped PNGs.

Scale bar: 200 um = 20 HR px at 10 um/px output grid (2x SR grid on a 20 um
detector pitch; same convention as fig10), drawn on the first panel only.

Run:
    cd /Users/ujs/mycode/thermal_lift && uv run python \
        docs/publication_figures/scripts/fig28_v4era_checkpoint_strip.py
"""

from __future__ import annotations

import numpy as np

from pubfig_style import CMAP_TEMPERATURE, REPO_ROOT, save_fig, setup_academic_style, strip_montage

setup_academic_style()

SRC_DIR = REPO_ROOT / "remote_inbox" / "20260627_checkpoint_evolution" / "solver_v4_acl027"

STEPS = [2500, 5000, 10000, 15000, 20000]
STEP_LABELS = ["2.5k steps", "5k steps", "10k steps", "15k steps", "20k steps"]

# Same center-detail crop as fig10_real_visual_montage.py (verified against
# this array's (960, 1280) grid below).
CROP_Y0, CROP_X0, CROP_SIZE = 361, 522, 128


def _load_crop(step: int) -> np.ndarray:
    path = SRC_DIR / f"solver_step{step:06d}_temperature_c.npz"
    with np.load(path) as d:
        arr = d["temperature_c"]
    assert arr.shape == (960, 1280), f"unexpected grid {arr.shape} for step {step}"
    return arr[CROP_Y0 : CROP_Y0 + CROP_SIZE, CROP_X0 : CROP_X0 + CROP_SIZE]


def main() -> None:
    crops = [_load_crop(s) for s in STEPS]

    vmin = float(min(c.min() for c in crops))
    vmax = float(max(c.max() for c in crops))

    # Shared checkpoint-strip template (pubfig_style.strip_montage): identical
    # row-label / step-title / colorbar styling with fig12. Figure title lives
    # in the caption, not inside the figure.
    fig, axes = strip_montage(
        [crops],
        col_titles=[STEP_LABELS],
        row_labels=["Unrolled solver (v4)"],
        cmap=CMAP_TEMPERATURE,
        vmin=vmin,
        vmax=vmax,
        cbar_label="Temperature [$^\\circ$C]",
        panel_aspect=1.0,
    )

    # 200 um scale bar (20 HR px at 10 um/px output grid), first panel only.
    ax_sb = axes[0][0]
    ax_sb.plot([6, 26], [120, 120], color="white", lw=2.2, solid_capstyle="butt")
    ax_sb.annotate("200 $\\mu$m", (16, 116), color="white", ha="center",
                    va="bottom", fontsize=7)

    paths = save_fig(fig, "fig28_v4era_checkpoint_strip")
    print("\n".join(str(p) for p in paths))


if __name__ == "__main__":
    main()
