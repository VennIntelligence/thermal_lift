#!/usr/bin/env python
"""Measure real-chip structure geometry from the data/optical micrographs.

Owner supplied 7 optical micrographs of the imaged chip (data/optical/,
2026-07-05; data/ is gitignored — keep a backup). Same device region at three
magnifications; three frames carry annotation scale bars (solid white box =
stated length):

  0021.jpg.jpg  20um  / 456 px  -> 0.0439 um/px (high mag)
  21.jpg.jpg    50um  / 576 px  -> 0.0868 um/px (mid mag)
  2.jpg.jpg    100um  / 446 px  -> 0.2242 um/px (low mag, full 4-quadrant cell)

MEASURED RESULTS (2026-07-07, run-length profiles below; the central finger
stack measures identically at 20um and 50um calibrations, agreement <0.2um):

  central cross arms   4 metal fingers w=4.3-4.5um, gaps 4.0-5.0um
                       (pitch ~8.7um; SUB-floor -> at thermal resolution the
                       stack merges into one ~31um arm)
  arm feed traces      2-3 traces w=2.2-3.4um, gaps ~1.9um (sub-floor bundle
                       ~12um wide)
  quadrant meanders    nested-L (greek key) bands: bar 33-55um wide,
                       gaps 20-55um (right quadrant also shows 10.8um gaps —
                       sub-floor, floored to 28um in the v7 motif)
  end pads             136-261um
  unit cell            ~500um, tiled over the die

These numbers parameterize the v7 `quad_meander` real-structure motif
(scripts/v7_content_demo.py; todos/dataset.md §2.2).

用法: uv run python scripts/measure_optical_reference.py（无参数；剖面定义在 SPECS 常量）
输入: data/optical/ 三张带标尺的光学显微图（0021.jpg.jpg / 21.jpg.jpg / 2.jpg.jpg）
输出: 终端打印各剖面 run-length 测量（um）；
      research_log/assets/optical_measurement/line_*.png 剖面位置标注图
"""
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

OPT = Path(__file__).resolve().parent.parent / "data" / "optical"
OUT = Path(__file__).resolve().parent.parent / "research_log" / "assets" / "optical_measurement"

UM_PER_PX = {"2.jpg.jpg": 100 / 446.0, "21.jpg.jpg": 50 / 576.0, "0021.jpg.jpg": 20 / 456.0}

# (image, 'v'|'h', fixed_coord, lo, hi, label) — original 2592x1944 px coords
SPECS = [
    ("2.jpg.jpg", "v", 650, 60, 1880, "left quadrants, vertical cut through horizontal bars"),
    ("2.jpg.jpg", "h", 1296, 60, 2530, "horizontal cut mid-height through cross arm + quadrant verticals"),
    ("2.jpg.jpg", "v", 1970, 60, 1880, "right quadrants, vertical cut"),
    ("21.jpg.jpg", "h", 500, 850, 1400, "vertical-arm finger stack above center"),
    ("21.jpg.jpg", "v", 600, 850, 1200, "left-arm feed traces, vertical cut"),
    ("0021.jpg.jpg", "h", 500, 700, 1900, "high-mag: cut through vertical arm fingers"),
    ("0021.jpg.jpg", "v", 1296, 700, 1300, "high-mag: vertical cut through horizontal arm"),
]


def find_scale_bar(img: np.ndarray) -> tuple[int, int, int, int] | None:
    """Solid near-saturated rectangle in the bottom quarter -> (x0, y0, w, h)."""
    H = img.shape[0]
    region = img[int(0.75 * H):, :]
    lab, n = ndimage.label(region > 245)
    best = None
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        h = ys.max() - ys.min() + 1
        w = xs.max() - xs.min() + 1
        fill = len(ys) / (h * w)
        if w > 150 and 20 < h < 300 and fill > 0.6 and (best is None or len(ys) > best[0]):
            best = (len(ys), xs.min(), ys.min() + int(0.75 * H), w, h)
    return None if best is None else best[1:]


def runs(mask: np.ndarray) -> list[tuple[bool, int, int]]:
    out, start = [], 0
    for i in range(1, len(mask) + 1):
        if i == len(mask) or mask[i] != mask[start]:
            out.append((bool(mask[start]), start, i - start))
            start = i
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name in UM_PER_PX:
        img = np.asarray(Image.open(OPT / name).convert("L"), dtype=np.float32)
        bar = find_scale_bar(img)
        print(f"{name}: scale bar {bar} -> {UM_PER_PX[name]:.4f} um/px (calibrated)")
    for name, axis, c, lo, hi, label in SPECS:
        um = UM_PER_PX[name]
        img = np.asarray(Image.open(OPT / name).convert("L"), dtype=np.float32)
        prof = (img[lo:hi, c - 2:c + 3].mean(axis=1) if axis == "v"
                else img[c - 2:c + 3, lo:hi].mean(axis=0))
        thr = 0.5 * (np.percentile(prof, 5) + np.percentile(prof, 95))
        rl = runs(prof > thr)
        print(f"\n== {name} {axis}@{c} [{lo}:{hi}] {label}  ({um:.4f} um/px)")
        print("  runs(um): " + " ".join(f"{'B' if v else 'd'}{l * um:.1f}" for v, s, l in rl))
        im = Image.open(OPT / name).convert("RGB")
        dr = ImageDraw.Draw(im)
        xy = [(c, lo), (c, hi)] if axis == "v" else [(lo, c), (hi, c)]
        dr.line(xy, fill=(255, 0, 0), width=6)
        im.thumbnail((900, 900))
        im.save(OUT / f"line_{name}_{axis}{c}.png")


if __name__ == "__main__":
    main()
