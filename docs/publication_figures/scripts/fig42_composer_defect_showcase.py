"""Fig 42 -- v7 composer defect-family showcase (ACL-065).

Companion to fig20 (whole-scene gallery): fig20 shows full assemblies at
scene scale, this figure zooms into the five defect families the v7 composer
(scripts/v7_composer_demo.py) injects on top of those assemblies, one row per
family with 3-4 examples spanning each family's parameter range:
  Row 1 -- small dark dots      (temp_render TempDefects.dots): r=1-4 HR px,
           depth=0.3-1.0 fractional pull toward T_bg, soft edge ~1 HR px.
  Row 2 -- hot spots            (TempDefects.hots): r=1-4 HR px,
           amp=0.3-1.0 x local dT, soft edge ~1 HR px.
  Row 3 -- shallow dark blobs   (TempDefects.darks): r=8-16 HR px (an order
           larger than row-1 dots), depth=0.15-0.4, soft edge 3 HR px --
           the "softness variant" of the dot family: same dark-pull
           mechanism, larger radius + wider soft edge -> reads as a blur
           rather than a point.
  Row 4 -- irregular edge notches (meta.notch_pts): mask-level union-of-discs
           bites carved out of a panel edge before rotation.
  Row 5 -- broken traces        (meta.break_pts): mask-level gap cut across
           a bright trace before rotation.

Path taken: RAN THE GENERATOR. scripts/v7_composer_demo.py was executed
mechanically (uv run python scripts/v7_composer_demo.py --out <scratch>) to
confirm it still produces sheet1/sheet2 PNGs with no missing deps; this
figure then imports v7_composer_demo directly (sys.path trick, no package
install needed) and calls its own _make_scene()/compose_scene() machinery to
harvest defect instances from a scan of scenes, choosing per family the
examples that are >=~35 HR px apart in radius/depth/log-metrics so the row
shows a real spread rather than four near-identical draws. Crops are taken
from the HR ground-truth temperature field `T` (defects are additive/
subtractive on this field or its supporting coverage mask, so all five
families are visible in T alone); a per-crop 0.5-99.5 percentile inferno
scale is used exactly as v7_composer_demo.sheet_defect_zoom does, because
absolute scale varies scene-to-scene (T_bg 19-23 C, dT 1-3 C) and a shared
scale would wash out the small defect contrast this figure exists to show.
Crop size is per family (10 um/px HR grid): 0.7 mm for the point defects
(dark dots / hot spots, r=1-4 HR px -- a wide crop would make them
invisible and ambiguous among scene look-alikes), 1.6 mm for the larger
shallow dark blobs, 2.0 mm for the mask-level notch/break families which
need panel-edge/trace context to read as such. Every crop carries a cyan
ring at the exact labelled (y, x), since a single crop can contain more
than one similar-looking defect instance.

Seeds: scan seeds v7demo_seed0 + k for k=0..N_SCAN-1 (v7demo_seed0 = 4200000,
disjoint from the module's own default demo seed 1010001 and from other
fig4x seeds already in use) until every family has >=4 candidates, tiers
alternate mid/high as in the module's own sheet_defect_zoom().

Run: uv run python docs/publication_figures/scripts/fig42_composer_defect_showcase.py
"""

from __future__ import annotations

import sys

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

from pubfig_style import (
    CMAP_TEMPERATURE,
    REPO_ROOT,
    W_DOUBLE,
    save_fig,
    setup_academic_style,
)

setup_academic_style()

sys.path.insert(0, str(REPO_ROOT / "scripts"))
import v7_composer_demo as vc  # noqa: E402

UM_PER_HR = vc.UM_PER_HR   # 10 um/px, HR grid (detector pitch 20um, SCALE=2x)
SEED0 = 4200000
N_SCAN = 40
N_PER_ROW = 4


def _pct(img: np.ndarray, lo: float = 0.5, hi: float = 99.5) -> tuple[float, float]:
    return float(np.percentile(img, lo)), float(np.percentile(img, hi))


def _crop(T: np.ndarray, cy: float, cx: float, side: int):
    """Crop a `side`-px square centred as close to (cy, cx) as the frame
    allows; also returns the defect's (row, col) position inside the crop
    so callers can ring it (multiple similar-looking defects can appear in
    one crop, so the raw crop alone doesn't say which one is being labelled)."""
    h, w = T.shape
    ccy = int(np.clip(cy, side // 2, h - side // 2))
    ccx = int(np.clip(cx, side // 2, w - side // 2))
    crop = T[ccy - side // 2: ccy + side // 2, ccx - side // 2: ccx + side // 2]
    local_y = cy - (ccy - side // 2)
    local_x = cx - (ccx - side // 2)
    return crop, local_y, local_x


def _spread_pick(items: list[tuple], key, n: int) -> list[tuple]:
    """Greedily pick n items whose `key` values are maximally spread out
    (so a 4-example row shows real parameter range, not four near-duplicates)."""
    if len(items) <= n:
        return items
    items = sorted(items, key=key)
    idx = np.linspace(0, len(items) - 1, n).round().astype(int)
    seen: set[int] = set()
    out = []
    for i in idx:
        i = int(i)
        while i in seen and i < len(items) - 1:
            i += 1
        seen.add(i)
        out.append(items[i])
    return out


# ── scan scenes, harvest defect instances per family ──────────────────────
dots_pool: list[tuple] = []    # (T, y, x, r_px, depth)
hots_pool: list[tuple] = []    # (T, y, x, r_px, amp)
darks_pool: list[tuple] = []   # (T, y, x, r_px, depth)
notch_pool: list[tuple] = []   # (T, y, x)
break_pool: list[tuple] = []   # (T, y, x)

for k in range(N_SCAN):
    tier = "high" if k % 3 != 2 else "mid"
    s = vc._make_scene(SEED0 + k, tier, angle_mode="zero")
    T, m, d = s["T"], s["meta"], s["dfx"]
    for (y, x, r, depth) in d.dots:
        dots_pool.append((T, y, x, r, depth))
    for (y, x, r, amp) in d.hots:
        hots_pool.append((T, y, x, r, amp))
    for (y, x, r, depth) in d.darks:
        darks_pool.append((T, y, x, r, depth))
    for (ny, nx) in m.notch_pts:
        notch_pool.append((T, ny / UM_PER_HR, nx / UM_PER_HR))
    for (by, bx) in m.break_pts:
        break_pool.append((T, by / UM_PER_HR, bx / UM_PER_HR))

dots_ex = _spread_pick(dots_pool, key=lambda t: (t[3], t[4]), n=N_PER_ROW)
hots_ex = _spread_pick(hots_pool, key=lambda t: (t[3], t[4]), n=N_PER_ROW)
darks_ex = _spread_pick(darks_pool, key=lambda t: (t[3], t[4]), n=N_PER_ROW)
notch_ex = _spread_pick(notch_pool, key=lambda t: (t[1], t[2]), n=N_PER_ROW)
break_ex = _spread_pick(break_pool, key=lambda t: (t[1], t[2]), n=N_PER_ROW)

# Crop side [HR px] is tuned per family: the point defects (dots/hots) have
# radius 1-4 HR px, so a wide context crop makes them nearly invisible and,
# worse, one of several similar dots scattered through the same scene --
# these rows use a tight 0.7 mm crop. The larger/mask-level families need
# panel context to read as "an edge notch" / "a gap in a trace" at all, so
# they keep a 1.6-2.0 mm crop. Every crop also gets a cyan ring marking the
# exact (y, x) the label describes, since several look-alike defects can sit
# in the same crop.
ROWS = [
    ("dark dots", dots_ex, 70,
     lambda t: f"r={t[3]:.1f}px\ndepth={t[4]:.2f}"),
    ("hot spots", hots_ex, 70,
     lambda t: f"r={t[3]:.1f}px\namp={t[4]:.2f}"),
    ("shallow dark blobs\n(softness variant)", darks_ex, 160,
     lambda t: f"r={t[3]:.1f}px\ndepth={t[4]:.2f}"),
    ("irregular edge notch", notch_ex, 200, lambda t: "mask-level bite"),
    ("broken trace", break_ex, 200, lambda t: "mask-level gap"),
]

fig, axes = plt.subplots(len(ROWS), N_PER_ROW, figsize=(W_DOUBLE, 1.7 * len(ROWS)))

for r, (fam_name, examples, side, label_fn) in enumerate(ROWS):
    for c in range(N_PER_ROW):
        ax = axes[r, c]
        if c >= len(examples):
            ax.axis("off")
            continue
        item = examples[c]
        T, y, x = item[0], item[1], item[2]
        crop, ly, lx = _crop(T, y, x, side=side)
        lo, hi = _pct(crop)
        ax.imshow(crop, cmap=CMAP_TEMPERATURE, vmin=lo, vmax=hi, interpolation="nearest")
        ring_r = max(side * 0.06, 5.0)
        ax.add_patch(Circle((lx, ly), ring_r, facecolor="none", edgecolor="#00e5ff",
                             linewidth=1.3))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(label_fn(item), fontsize=7)
    axes[r, 0].set_ylabel(fam_name, fontsize=8, rotation=90, labelpad=4)

fig.suptitle(
    "v7 composer defect families (ACL-065): HR ground truth, per-row crop size "
    "(0.7/1.6/2.0 mm), per-crop 0.5-99.5 pct inferno scale, cyan ring = labelled "
    "defect location",
    fontsize=10.5,
)

paths = save_fig(fig, "fig42_composer_defect_showcase")
print("\n".join(str(p) for p in paths))
print(f"pool sizes: dots={len(dots_pool)} hots={len(hots_pool)} "
      f"darks={len(darks_pool)} notch={len(notch_pool)} break={len(break_pool)}")
