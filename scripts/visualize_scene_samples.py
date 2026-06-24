#!/usr/bin/env python3
"""Quick scene sample generator — 6 scenes, mask + blurred LR preview."""
import sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tcforge" / "src"))

from tcforge.geometry import build_scene_mask_with_metadata
from tcforge.physics import render_temperature_field

OUT = ROOT / "output" / "ep07" / "scene_samples"
OUT.mkdir(parents=True, exist_ok=True)

CANVAS = (1920, 2560)
SCALE = 4

scenes = [
    ("easy",   42),
    ("medium", 123),
    ("medium", 456),
    ("hard",   789),
    ("hard",   999),
    ("stress", 1337),
]

fig, axes = plt.subplots(3, 4, figsize=(24, 18))
fig.suptitle("TCForge Randomized Scenes — HR Mask | Blurred LR", fontsize=14, y=0.99)

for i, (diff, seed) in enumerate(scenes):
    print(f"Generating {diff} seed={seed} ...")
    mask, meta = build_scene_mask_with_metadata(
        diff, seed, canvas_shape=CANVAS, pixel_size_um=20.0, scale=SCALE,
    )
    hr_temp = render_temperature_field(
        mask, t_bg_c=21.0, delta_t_c=3.0,
        low_freq_amplitude_c=0.2, low_freq_sigma_px=96.0, seed=seed,
    )
    blurred = ndimage.gaussian_filter(hr_temp, sigma=0.226 * SCALE)
    lr = blurred[::SCALE, ::SCALE]

    prims = [p["type"] for p in meta["primitives"]]
    counts = {}
    for p in prims:
        counts[p] = counts.get(p, 0) + 1
    label = ", ".join(f"{k}×{v}" if v > 1 else k for k, v in counts.items())

    row = i // 2
    col = (i % 2) * 2

    axes[row, col].imshow(mask, cmap="gray", interpolation="nearest")
    axes[row, col].set_title(f"[{diff}] seed={seed}  ({len(prims)} prims)\n{label}", fontsize=9)
    axes[row, col].axis("off")

    axes[row, col+1].imshow(lr, cmap="inferno", interpolation="nearest")
    axes[row, col+1].set_title(f"LR {lr.shape[1]}×{lr.shape[0]}", fontsize=9)
    axes[row, col+1].axis("off")

    plt.imsave(str(OUT / f"s{i}_mask_{diff}.png"), mask, cmap="gray")
    plt.imsave(str(OUT / f"s{i}_lr_{diff}.png"), lr, cmap="inferno")
    print(f"  → {len(prims)} primitives: {label}")

plt.tight_layout(rect=[0, 0, 1, 0.97])
grid_path = OUT / "scene_grid.png"
fig.savefig(str(grid_path), dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"\nDone! Grid: {grid_path}")
