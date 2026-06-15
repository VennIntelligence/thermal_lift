#!/usr/bin/env python3
"""Generate the merged F5 visual figure: center comb ROI + held-out ROI2.

This consolidates the former two-panel main visual (center comb ROI) and the
held-out ROI2 audit figure into one 4-row figure so the two independent ROIs
sit directly above/below each other and stay visually comparable:

    row 0: center ROI temperature      row 1: center ROI highpass
    row 2: ROI2  temperature           row 3: ROI2  highpass

Both ROIs are fixed-geometry fractional windows on the 2x HR grid (no proxy
search, no method-output tuning).  The temperature row checks "is it just edge
enhancement?"; the highpass row carries the soft / staircase / over-thickened /
sharp-low-grain differentiation.  Keeping both domains for both ROIs is the
point of the figure.

Inputs:
    output/ep11_unified_harness/hr/{drizzle,tgv,v9a_late_60k,
    v10_lam120_15k}_{temperature,highpass}.npy

Outputs:
    output/paper_figures/fig05_combined_visual.{png,pdf}

The figure is task-level visual/proxy evidence only: it does not establish
resolution, temperature metrology, or GT fidelity.

Run from the repository root:
    uv run python scripts/paper_figures/fig05_combined_visual.py
"""

from __future__ import annotations

import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLBACKEND", "Agg")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

from thermal_core.plotting import COLORMAPS, savefig_academic, setup_academic_style

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HR_DIR = PROJECT_ROOT / "output" / "ep11_unified_harness" / "hr"
PAPER_FIGURE_DIR = PROJECT_ROOT / "output" / "paper_figures"

# Fixed fractional ROI windows on the 2x HR grid (960x1280); identical to the
# windows used by the former F5 (center comb) and F5b (held-out ROI2).
CENTER_ROI_FRAC = {"row0": 384.0 / 960.0, "row1": 518.0 / 960.0, "col0": 478.0 / 1280.0, "col1": 674.0 / 1280.0}
ROI2_FRAC = {"row0": 0.270, "row1": 0.415, "col0": 0.530, "col1": 0.685}

ARMS = [
    ("drizzle", "Drizzle"),
    ("tgv", "TGV"),
    ("v9a_late_60k", "Hybrid 60K"),
    ("v10_lam120_15k", "Hybrid+ResObs\nlambda=1.2 15K"),
]


def fill_nan(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    fill = float(np.nanmedian(arr[finite])) if bool(finite.any()) else 0.0
    return np.where(finite, arr, fill).astype(np.float32, copy=False)


def crop_roi(image: np.ndarray, frac: dict[str, float]) -> np.ndarray:
    rows, cols = np.asarray(image).shape
    y0 = int(round(rows * frac["row0"]))
    y1 = int(round(rows * frac["row1"]))
    x0 = int(round(cols * frac["col0"]))
    x1 = int(round(cols * frac["col1"]))
    if not (0 <= y0 < y1 <= rows and 0 <= x0 < x1 <= cols):
        raise ValueError(f"Invalid ROI bounds {(y0, y1, x0, x1)} for shape {(rows, cols)}")
    return np.asarray(image)[y0:y1, x0:x1]


def load_arm_arrays() -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for arm_id, _label in ARMS:
        temp_path = HR_DIR / f"{arm_id}_temperature.npy"
        hp_path = HR_DIR / f"{arm_id}_highpass.npy"
        if not temp_path.exists() or not hp_path.exists():
            raise FileNotFoundError(f"Missing cached HR pair for {arm_id}: {temp_path}, {hp_path}")
        out[arm_id] = {
            "temperature": np.load(temp_path).astype(np.float32, copy=False),
            "highpass": np.load(hp_path).astype(np.float32, copy=False),
        }
    return out


def _scales(crops: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([c[np.isfinite(c)].ravel() for c in crops if np.isfinite(c).any()])
    return float(np.percentile(values, 1.0)), float(np.percentile(values, 99.0))


def _abs_scale(crops: list[np.ndarray]) -> float:
    values = np.concatenate([c[np.isfinite(c)].ravel() for c in crops if np.isfinite(c).any()])
    return max(float(np.percentile(np.abs(values), 99.0)), 1e-6)


def draw_combined(arrays: dict[str, dict[str, np.ndarray]], output_path: Path) -> Path:
    labels = [label for _arm_id, label in ARMS]

    # Per-ROI crops; each ROI keeps its own temperature/highpass color scale.
    center_temp = [crop_roi(arrays[a]["temperature"], CENTER_ROI_FRAC) for a, _ in ARMS]
    center_hp = [crop_roi(arrays[a]["highpass"], CENTER_ROI_FRAC) for a, _ in ARMS]
    roi2_temp = [crop_roi(arrays[a]["temperature"], ROI2_FRAC) for a, _ in ARMS]
    roi2_hp = [crop_roi(arrays[a]["highpass"], ROI2_FRAC) for a, _ in ARMS]

    center_tmin, center_tmax = _scales(center_temp)
    roi2_tmin, roi2_tmax = _scales(roi2_temp)
    center_hpmax = _abs_scale(center_hp)
    roi2_hpmax = _abs_scale(roi2_hp)

    row_specs = [
        ("Center temp.", center_temp, COLORMAPS["temperature"], center_tmin, center_tmax),
        ("Center HP", center_hp, COLORMAPS["residual_diff"], -center_hpmax, center_hpmax),
        ("ROI2 temp.", roi2_temp, COLORMAPS["temperature"], roi2_tmin, roi2_tmax),
        ("ROI2 HP", roi2_hp, COLORMAPS["residual_diff"], -roi2_hpmax, roi2_hpmax),
    ]

    fig, axes = plt.subplots(4, len(labels), figsize=(7.2, 4.8), constrained_layout=True)
    for r, (ylabel, crops, cmap, vmin, vmax) in enumerate(row_specs):
        im = None
        for col in range(len(labels)):
            im = axes[r, col].imshow(
                fill_nan(crops[col]),
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
            )
            axes[r, col].set_xticks([])
            axes[r, col].set_yticks([])
            for spine in axes[r, col].spines.values():
                spine.set_visible(False)
            if r == 0:
                axes[r, col].set_title(labels[col])
        axes[r, 0].set_ylabel(ylabel)
        cbar = fig.colorbar(im, ax=axes[r, :].tolist(), fraction=0.022, pad=0.01, shrink=0.75)
        cbar.outline.set_visible(False)
        cbar.set_label("deg C")

    return savefig_academic(fig, output_path)


def main() -> int:
    setup_academic_style()
    arrays = load_arm_arrays()
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    png = PAPER_FIGURE_DIR / "fig05_combined_visual.png"
    pdf = PAPER_FIGURE_DIR / "fig05_combined_visual.pdf"
    draw_combined(arrays, png)
    draw_combined(arrays, pdf)
    print(f"[fig05c] wrote {png.relative_to(PROJECT_ROOT)}")
    print(f"[fig05c] wrote {pdf.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
