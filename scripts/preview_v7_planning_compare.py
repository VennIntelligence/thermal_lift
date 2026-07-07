#!/usr/bin/env python3
"""v7 planning: content-axis comparison sheets (geometry masks only, no physics).

Renders HR coverage masks straight from tcforge.geometry for three content
recipes, so the owner can anchor the "v6 规则密度纹理过多 / 芯片应有疏有密"
discussion on pictures instead of memory:

  sheet_v5_legacy.png     — legacy generic IC layout (v3–v5 pools; the look the
                            deleted v5 pool trained de_pb9 on, ACL-061/064)
  sheet_v6_current.png    — v6 pool motif weights verbatim (pool_2x_v6_cpu.json)
  sheet_v6_reweighted.png — v6 composer, config-only motif reweight toward
                            multi_die/generic (one candidate v7 lever, NO new code)

Display follows synthetic_data_realism.md: crop the interior rectangle of the
inscribed disc, nearest interpolation. Panel titles carry family / density /
occupancy (mask fraction inside the crop) so the sparse↔dense spread is legible.

Usage: uv run python scripts/preview_v7_planning_compare.py \
           [--out research_log/assets/v7_planning] [--n 8]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

TCFORGE_SRC = Path(__file__).resolve().parents[1] / "tcforge" / "src"
if TCFORGE_SRC.exists() and str(TCFORGE_SRC) not in sys.path:
    sys.path.insert(0, str(TCFORGE_SRC))

from tcforge.geometry import build_scene_mask_with_metadata  # noqa: E402

HR_SHAPE = (960, 1280)
PIXEL_SIZE_UM = 20.0
SCALE = 2

V6_WEIGHTS = {  # pool_2x_v6_cpu.json verbatim
    "pga_grid": 0.28, "die_bga": 0.24, "multi_die": 0.18,
    "trace_bus": 0.14, "heat_spreader": 0.10, "generic": 0.06,
}
# Candidate v7 lever 1 (config-only): demote the full-canvas periodic-texture
# families (pga_grid / die_bga / heat_spreader), promote the mixed block+trace
# families. Values are a discussion starting point, not a decision.
REWEIGHTED = {
    "pga_grid": 0.10, "die_bga": 0.14, "multi_die": 0.30,
    "trace_bus": 0.12, "heat_spreader": 0.04, "generic": 0.30,
}
# v5 pool difficulty mix (150/400/350/100) compressed to 8 panels.
V5_DIFFICULTIES = ["easy", "medium", "medium", "medium", "hard", "hard", "hard", "stress"]


def _render(seed: int, difficulty: str, motif_weights: dict | None,
            rotation_deg: float) -> tuple[np.ndarray, dict]:
    mask, meta = build_scene_mask_with_metadata(
        difficulty, seed,
        rotation_deg_center=rotation_deg, rotation_jitter_deg=0.0,
        canvas_shape=HR_SHAPE, pixel_size_um=PIXEL_SIZE_UM, scale=SCALE,
        antialias=True, ssaa_factor=4, inscribe_disc=True,
        motif_weights=motif_weights,
    )
    return mask, meta


def _interior_crop(mask: np.ndarray) -> np.ndarray:
    # Largest axis-aligned square inside the inscribed disc: side = 2r/sqrt(2).
    h, w = mask.shape
    side = int(min(h, w) / np.sqrt(2.0)) & ~1
    cy, cx = h // 2, w // 2
    return mask[cy - side // 2: cy + side // 2, cx - side // 2: cx + side // 2]


def _sheet(title: str, panels: list[tuple[np.ndarray, str]], out: Path) -> None:
    n = len(panels)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.5 * rows))
    fig.suptitle(title, fontsize=13, y=0.995)
    for i, ax in enumerate(np.atleast_2d(axes).ravel()):
        if i >= n:
            ax.axis("off")
            continue
        img, label = panels[i]
        ax.imshow(img, cmap="inferno", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(label, fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out.resolve())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=Path("research_log/assets/v7_planning"))
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--seed0", type=int, default=770001)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed0)
    rots = rng.uniform(0.0, 360.0, size=3 * args.n)

    sets = [
        ("sheet_v5_legacy.png",
         "v5 legacy generic IC layout (blocks + pillars + L-traces + frames + fine detail) "
         "— the deleted-v5-pool look; interior crop, nearest",
         None),
        ("sheet_v6_current.png",
         "v6 current motif weights (pga 0.28 / bga 0.24 / multi_die 0.18 / bus 0.14 / "
         "fins 0.10 / generic 0.06) — interior crop, nearest",
         V6_WEIGHTS),
        ("sheet_v6_reweighted.png",
         "v6 composer, REWEIGHTED only (pga 0.10 / bga 0.14 / multi_die 0.30 / bus 0.12 / "
         "fins 0.04 / generic 0.30) — candidate v7 lever 1, no code change",
         REWEIGHTED),
    ]
    for si, (fname, title, weights) in enumerate(sets):
        panels: list[tuple[np.ndarray, str]] = []
        for i in range(args.n):
            seed = args.seed0 + si * 1000 + i
            difficulty = V5_DIFFICULTIES[i % len(V5_DIFFICULTIES)] if weights is None else "medium"
            mask, meta = _render(seed, difficulty, weights, float(rots[si * args.n + i]))
            crop = _interior_crop(mask)
            occ = float((crop >= 0.5).mean())
            if weights is None:
                label = f"{difficulty}  occ={occ:.2f}  seed={seed}"
            else:
                dens = meta.get("density")
                label = (f"{meta.get('scene_family', '?')}  d={dens:.2f}  "
                         f"occ={occ:.2f}  seed={seed}")
            panels.append((crop, label))
        _sheet(title, panels, args.out / fname)


if __name__ == "__main__":
    main()
