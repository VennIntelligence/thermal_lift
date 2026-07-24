"""Fig 20 -- v7 TCForge synthetic training-scene showcase.

Gallery figure for the v7-generation composer (scripts/v7_composer_demo.py,
run mechanically for a 50-scene demo pool by
scripts/generate_v7_demo_minipool.py -> outputs/v7_demo_minipool/). Purpose:
give a single-glance visual sense of what the production training pools
actually contain -- multi-panel PV-module assemblies with occlusion,
irregular-edge / broken-trace geometry defects, and hot/dark thermal dot
defects -- before diving into any quantitative pool-composition figures
(fig01, fig09, etc).

Layout (2 rows x 4 cols, W_DOUBLE):
  Row 1 -- HR ground-truth temperature maps (cmap=inferno, PER-PANEL 1-99
           pct colour scale -- scenes differ by tens of degC in mean level
           so a shared scale would wash out the low-dT scenes; a per-panel
           colorbar with numeric range keeps each panel honest) for 4
           scenes hand-picked from outputs/v7_demo_minipool/index.json to
           span the pool's structural range:
             i=6  (mid tier,  occ=0.02 -- sparse, 3 panels)
             i=48 (high tier, occ=0.33 -- dense, 8 panels, most coverage)
             i=22 (high tier, dT=2.94 degC -- largest thermal swing)
             i=4  (mid tier, rich defect mix: 7 hot + 2 dark + 5 notch +
                   2 break -- also the subject of row 2)
  Row 2 -- one illustrative decomposition of scene i=4:
             (a) HR temperature (same as its row-1 panel, for reference)
             (b) panel/coverage mask `cov` from the npz (soft occupancy,
                 0=background 1=panel interior; this is NOT a defect mask,
                 it is the module-layout mask used to compute `occ`)
             (c) the single LR forward-model frame `lr` (480x640, half the
                 HR grid) -- what the detector actually sees before any
                 super-resolution
             (d) a high-pass residual of T (T minus a wide Gaussian blur of
                 T) as a defect-enhancement view: hot/dark dot defects and
                 sharp panel-edge/trace anomalies pop out against the
                 smooth thermal background. This is a derived visualisation
                 for this figure only, not a field stored in the npz (the
                 per-defect polygon list `dfx` from the generator is not
                 persisted to the npz; only the composited raster fields
                 cov/T/lr are saved by generate_v7_demo_minipool.py).

Scale bar: HR grid is 10 um/px (detector pitch 20 um, SCALE=2x; see
scripts/v7_composer_demo.py UM_PER_HR / HR_SHAPE, and repo memory note on
the 20 um pixel-pitch recalibration). LR grid is 20 um/px. A scale bar is
drawn on one representative HR panel and one LR panel; image panels have
axis ticks off per plotting_standards.md (image panels don't carry a
physically meaningful tick grid, and captions/scale bars communicate scale
more cleanly than a dense µm tick axis at this size).

Data: outputs/v7_demo_minipool/scene_0NN.npz (50 scenes, keys cov/T/lr as
      float16 rasters + occ/dT/sigma/angle/tier scalars) and
      outputs/v7_demo_minipool/index.json (per-scene metadata incl. defect
      counts n_dots/n_hots/n_darks/n_notch/n_break used to pick scenes).
      Note: index.json/npz carry no explicit "motif family" label (the
      composer's family/tier vocabulary lives in code, not metadata), so
      scene diversity here is selected via occ/dT/tier/defect-count spread
      rather than a family tag.
Run:  uv run python docs/publication_figures/scripts/fig20_synthetic_showcase.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

from pubfig_style import (
    CMAP_TEMPERATURE,
    REPO_ROOT,
    W_DOUBLE,
    save_fig,
    setup_academic_style,
)

setup_academic_style()

POOL_DIR = REPO_ROOT / "outputs" / "v7_demo_minipool"
UM_PER_HR = 10.0   # HR grid pitch [um]; detector pitch 20um, SCALE=2x
UM_PER_LR = 20.0   # LR (detector) grid pitch [um]

TOP_SCENES = [6, 48, 22, 4]     # sparse | dense | max-dT | rich-defect (=row2 subject)
DECOMP_SCENE = 4


def _load(i: int) -> dict:
    d = np.load(POOL_DIR / f"scene_{i:03d}.npz")
    return {k: d[k] for k in d.files}


def _interior_crop(img: np.ndarray) -> np.ndarray:
    """Centre crop to the side=min(h,w)/sqrt(2) square, matching the
    interior_crop() convention used by scripts/v7_composer_demo.py's own
    eyeball sheets -- avoids the mostly-empty black margin around the
    (typically centred, rotated) panel assembly and lets the gallery show
    scene content at a legible size."""
    h, w = img.shape
    side = int(min(h, w) / np.sqrt(2.0)) & ~1
    cy, cx = h // 2, w // 2
    return img[cy - side // 2: cy + side // 2, cx - side // 2: cx + side // 2]


def _pct(img: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
    return float(np.percentile(img, lo)), float(np.percentile(img, hi))


def _add_scalebar(ax, um_per_px: float, bar_um: float, label: str,
                   loc_frac=(0.06, 0.06), color="white") -> None:
    """Bottom-left scale bar in axes (data) coordinates for an imshow panel."""
    h, w = ax.images[0].get_array().shape[:2]
    bar_px = bar_um / um_per_px
    x0 = loc_frac[0] * w
    y0 = (1.0 - loc_frac[1]) * h
    ax.plot([x0, x0 + bar_px], [y0, y0], color=color, lw=2.2, solid_capstyle="butt")
    ax.text(x0 + bar_px / 2, y0 - 0.02 * h, label, color=color, fontsize=7,
             ha="center", va="bottom")


with open(POOL_DIR / "index.json") as f:
    index = {r["i"]: r for r in json.load(f)}

fig, axes = plt.subplots(2, 4, figsize=(W_DOUBLE, 4.0))

# ── Row 1: HR temperature gallery, 4 diverse scenes ────────────────────
for c, i in enumerate(TOP_SCENES):
    ax = axes[0, c]
    s = _load(i)
    rec = index[i]
    T = _interior_crop(s["T"].astype(np.float32))
    lo, hi = _pct(T)
    im = ax.imshow(T, cmap=CMAP_TEMPERATURE, vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    tag = {6: "sparse", 48: "dense", 22: "max $\\Delta T$", 4: "rich defects\n(row 2 subject)"}[i]
    ax.set_title(
        f"scene {i:03d} ({rec['tier']}) -- {tag}\n"
        f"occ={rec['occ']:.2f}  $\\Delta T$={rec['dT']:.1f}$^\\circ$C  "
        f"panels={rec['n_panels']}",
        fontsize=7.3,
    )
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("T [$^\\circ$C]", fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    if c == 0:
        _add_scalebar(ax, UM_PER_HR, 2000.0, "2 mm")

# ── Row 2: decomposition of DECOMP_SCENE ────────────────────────────────
s4 = _load(DECOMP_SCENE)
rec4 = index[DECOMP_SCENE]
T4 = _interior_crop(s4["T"].astype(np.float32))
cov4 = _interior_crop(s4["cov"].astype(np.float32))
lr4 = _interior_crop(s4["lr"].astype(np.float32))
hp4 = T4 - gaussian_filter(T4, sigma=25.0)  # high-pass: defect-enhancement view

panels = [
    ("(a) HR temperature $T$", T4, CMAP_TEMPERATURE, _pct(T4), "T [$^\\circ$C]", UM_PER_HR),
    ("(b) panel/coverage mask", cov4, "viridis", (0.0, 1.0), "coverage [0-1]", UM_PER_HR),
    ("(c) LR forward-model frame", lr4, CMAP_TEMPERATURE, _pct(lr4), "T [$^\\circ$C]", UM_PER_LR),
    ("(d) high-pass residual $T-G_\\sigma(T)$\n(defect enhancement)", hp4, "RdBu_r",
     (-np.percentile(np.abs(hp4), 99), np.percentile(np.abs(hp4), 99)),
     "$\\Delta T$ [$^\\circ$C]", UM_PER_HR),
]
for c, (title, img, cmap, (lo, hi), cblabel, pitch) in enumerate(panels):
    ax = axes[1, c]
    im = ax.imshow(img, cmap=cmap, vmin=lo, vmax=hi, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=7.3)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label(cblabel, fontsize=7)
    cb.ax.tick_params(labelsize=6.5)
    if c in (0, 2):
        bar_um = 2000.0
        _add_scalebar(ax, pitch, bar_um, "2 mm", color="white" if c == 0 else "black")

# Details (seed, defect counts) live in the caption/docstring, not the title.
fig.suptitle(
    f"TCForge synthetic training scenes (top) and decomposition of scene "
    f"{DECOMP_SCENE:03d} (bottom)",
    fontsize=11, y=1.03,
)

paths = save_fig(fig, "fig20_synthetic_showcase")
print("\n".join(str(p) for p in paths))
