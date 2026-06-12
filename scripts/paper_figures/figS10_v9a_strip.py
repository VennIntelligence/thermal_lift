"""Supplementary figure S-F10: V9A fine-window evolution strip (temperature domain).

A 3x3 panel grid on the center fine-line window (2x grid rows 384:518,
cols 478:674): classical TGV reference, the 1x-input UNet endpoint (v8.1a 60K),
and the V9A checkpoint sequence 5K-60K.  Each panel is annotated with the
fine-window fidelity (hp_corr_input) and sharpness proxy (sharp_p95).

Each panel is normalized to its own robust 1-99 percentile range: the
reconstruction pipelines carry different global offsets (per-frame median
offset correction in the classical chain), so absolute temperatures are not
comparable across panels and only relative structure is shown.

Data sources (regenerate with algos/ep07_unet_sr/scripts/v9_review/run_pareto_sweep.py):
    output/ep07_v9_review/cache/*_temperature.npy
    output/ep07_v9_review/v9a_pareto_metrics.csv
    output/ep10_tgv_sr/best_hr_temperature.npy

Run from the repository root:
    uv run python scripts/paper_figures/figS10_v9a_strip.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thermal_core.plotting import format_colorbar, savefig_academic, setup_academic_style

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / "output" / "ep07_v9_review" / "cache"
METRICS_CSV = PROJECT_ROOT / "output" / "ep07_v9_review" / "v9a_pareto_metrics.csv"
TGV_NPY = PROJECT_ROOT / "output" / "ep10_tgv_sr" / "best_hr_temperature.npy"
OUT_DIR = PROJECT_ROOT / "output" / "paper_figures"

# Fine window on the 960x1280 2x grid (same definition as v9_review/common.py).
ROWS = slice(384, 518)
COLS = slice(478, 674)

PANELS = [
    ("tgv", "TGV (classical ref.)", TGV_NPY),
    ("v8_1a_60k", "v8.1a 60K (1x input)", CACHE_DIR / "v8_1a_60k_step60000_temperature.npy"),
    ("v9a_5k", "V9A 5K", CACHE_DIR / "v9a_5k_step5000_temperature.npy"),
    ("v9a_10k", "V9A 10K", CACHE_DIR / "v9a_10k_step10000_temperature.npy"),
    ("v9a_20k", "V9A 20K (most faithful)", CACHE_DIR / "v9a_20k_step20000_temperature.npy"),
    ("v9a_25k", "V9A 25K", CACHE_DIR / "v9a_25k_step25000_temperature.npy"),
    ("v9a_30k", "V9A 30K (fidelity cliff)", CACHE_DIR / "v9a_30k_step30000_temperature.npy"),
    ("v9a_40k", "V9A 40K", CACHE_DIR / "v9a_40k_step40000_temperature.npy"),
    ("v9a_60k", "V9A 60K", CACHE_DIR / "v9a_60k_step60000_temperature.npy"),
]


def _metrics_lookup() -> dict[str, tuple[float, float]]:
    df = pd.read_csv(METRICS_CSV)
    out: dict[str, tuple[float, float]] = {}
    for _, row in df.iterrows():
        out[row["name"]] = (float(row["hp_corr_input"]), float(row["sharp_p95"]))
    return out


def main() -> None:
    setup_academic_style()
    metrics = _metrics_lookup()

    crops = {name: np.load(path)[ROWS, COLS] for name, _, path in PANELS}

    fig, axes = plt.subplots(3, 3, figsize=(7.2, 4.9))
    im = None
    for ax, (name, title, _) in zip(axes.ravel(), PANELS):
        crop = crops[name]
        lo, hi = np.nanpercentile(crop, [1.0, 99.0])
        norm = np.clip((crop - lo) / (hi - lo), 0.0, 1.0)
        im = ax.imshow(norm, cmap="inferno", vmin=0.0, vmax=1.0, interpolation="nearest")
        ax.set_title(title, fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        # v8_1a metrics were measured against its own input mode; pareto CSV
        # only carries v9a/tgv/input rows, so fall back gracefully.
        if name in metrics:
            corr, sharp = metrics[name]
            ax.text(
                0.02, 0.035, f"fid {corr:.3f} | sharp {sharp:.2f}",
                transform=ax.transAxes, fontsize=6.5, color="white", va="bottom",
                bbox=dict(facecolor="black", alpha=0.45, pad=1.2, lw=0),
            )

    cbar = fig.colorbar(im, ax=axes, fraction=0.030, pad=0.02)
    format_colorbar(cbar, "Normalized temperature (per-panel robust 1-99%)")

    savefig_academic(fig, OUT_DIR / "figS10_v9a_strip.png", close=False)
    savefig_academic(fig, OUT_DIR / "figS10_v9a_strip.pdf")
    print(f"saved {OUT_DIR / 'figS10_v9a_strip.png'} (+.pdf)")


if __name__ == "__main__":
    main()
