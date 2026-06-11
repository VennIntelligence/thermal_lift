#!/usr/bin/env python3
"""Build a 4x shifted center-zoom temperature comparison for EP11.

UNet panel: crop/zoom the existing eval PNG (no checkpoint re-inference).
Baseline panel: load cached bicubic mean 2x HR temperature NPY.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

from thermal_core.notebook_cache import project_root
from thermal_core.plotting import COLORMAPS, savefig_academic, setup_academic_style


PROJECT_ROOT = project_root(Path(__file__).resolve().parents[3])
DEFAULT_UNET_PNG = (
    PROJECT_ROOT
    / "algos"
    / "ep07_unet_sr"
    / "outputs"
    / "ep07_v8_aa"
    / "eval_real"
    / "unet_step40000_center_zoom3x_temperature.png"
)
DEFAULT_BICUBIC_NPY = PROJECT_ROOT / "output" / "ep11_dl_benchmark" / "bicubic_mean_2x_hr_temp.npy"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "ep11_dl_benchmark"


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def shifted_center_fraction_crop(
    image: np.ndarray,
    fraction: float,
    *,
    shift_fraction_of_crop: float = 1.0 / 3.0,
) -> np.ndarray:
    """Center crop with the anchor shifted toward upper-left."""

    rows, cols = image.shape[:2]
    crop_rows = max(1, int(round(rows * fraction)))
    crop_cols = max(1, int(round(cols * fraction)))
    cy = rows // 2 - int(round(crop_rows * shift_fraction_of_crop))
    cx = cols // 2 - int(round(crop_cols * shift_fraction_of_crop))
    y0 = min(max(0, cy - crop_rows // 2), max(0, rows - crop_rows))
    x0 = min(max(0, cx - crop_cols // 2), max(0, cols - crop_cols))
    return image[y0 : y0 + crop_rows, x0 : x0 + crop_cols]


def zoom_shifted_crop(
    image: np.ndarray,
    *,
    fraction: float,
    zoom: float,
    shift_fraction_of_crop: float = 1.0 / 3.0,
) -> np.ndarray:
    crop = shifted_center_fraction_crop(
        np.asarray(image),
        fraction,
        shift_fraction_of_crop=shift_fraction_of_crop,
    )
    if crop.ndim == 2:
        return ndimage.zoom(crop, zoom=float(zoom), order=1).astype(np.float32, copy=False)
    return ndimage.zoom(crop, (float(zoom), float(zoom), 1), order=1).astype(np.float32, copy=False)


def extract_temperature_panel(rgb: np.ndarray) -> np.ndarray:
    """Strip title/colorbar margins from an academic temperature figure PNG."""

    arr = np.asarray(rgb, dtype=np.float32)
    if arr.ndim == 2:
        return arr
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    gray = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
    mask = gray < 0.985
    if not mask.any():
        return arr

    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    panel = arr[y0:y1, x0:x1]

    # Drop the vertical colorbar strip on the right, if present.
    col_energy = panel.mean(axis=(0, 2))
    if col_energy.size >= 24:
        tail = col_energy[-max(8, col_energy.size // 10) :]
        body = col_energy[: max(8, int(col_energy.size * 0.82))]
        if float(tail.mean()) > float(body.mean()) + 0.04:
            trim = max(8, int(col_energy.size * 0.12))
            panel = panel[:, :-trim]
    return panel


def temperature_limits(images: list[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([img[np.isfinite(img)].ravel() for img in images if np.isfinite(img).any()])
    if values.size == 0:
        return 0.0, 1.0
    return float(np.percentile(values, 1.0)), float(np.percentile(values, 99.0))


def display_from_png_panel(
    panel_rgb: np.ndarray,
    *,
    zoom: float,
    source_zoom: float,
    shift_fraction_of_crop: float,
) -> np.ndarray:
    """Map a 3x center-third PNG panel to the same 4x shifted view used on HR arrays."""

    extra_zoom = float(zoom) / float(source_zoom)
    return zoom_shifted_crop(
        panel_rgb,
        fraction=1.0,
        zoom=extra_zoom,
        shift_fraction_of_crop=shift_fraction_of_crop,
    )


def save_comparison(
    unet_panel_rgb: np.ndarray,
    bicubic_temp: np.ndarray,
    output_dir: Path,
    *,
    zoom: float,
    source_zoom: float,
    center_fraction: float,
    shift_fraction_of_crop: float,
    unet_title: str,
    baseline_title: str,
) -> tuple[Path, Path]:
    setup_academic_style()
    output_dir.mkdir(parents=True, exist_ok=True)

    unet_rgb = display_from_png_panel(
        unet_panel_rgb,
        zoom=zoom,
        source_zoom=source_zoom,
        shift_fraction_of_crop=shift_fraction_of_crop,
    )
    bicubic_crop = zoom_shifted_crop(
        bicubic_temp,
        fraction=center_fraction,
        zoom=zoom,
        shift_fraction_of_crop=shift_fraction_of_crop,
    )
    vmin, vmax = temperature_limits([bicubic_crop])

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1), squeeze=True)
    axes[0].imshow(unet_rgb, interpolation="nearest")
    axes[0].set_title(unet_title)
    axes[0].set_xticks([])
    axes[0].set_yticks([])

    im = axes[1].imshow(
        bicubic_crop,
        cmap=COLORMAPS["temperature"],
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
    )
    axes[1].set_title(baseline_title)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.03).set_label("Temperature [deg C]")
    fig.suptitle(
        (
            f"Center ROI {center_fraction:.3f} with upper-left shift "
            f"{shift_fraction_of_crop:.3f}×crop, {zoom:.1f}x display zoom"
        ),
        fontsize=10,
        fontweight="bold",
    )

    compare_path = output_dir / "unet_v8aa_vs_bicubic_4x_shifted_center_temperature.png"
    unet_only_path = output_dir / "unet_v8aa_step40000_center_zoom4x_shifted_temperature.png"
    savefig_academic(fig, compare_path)

    fig_unet, ax_unet = plt.subplots(1, 1, figsize=(4.1, 3.0), squeeze=True)
    ax_unet.imshow(unet_rgb, interpolation="nearest")
    ax_unet.set_title(unet_title)
    ax_unet.set_xticks([])
    ax_unet.set_yticks([])
    savefig_academic(fig_unet, unet_only_path)
    return compare_path, unet_only_path


def run(args: argparse.Namespace) -> dict[str, Any]:
    if not args.unet_png.exists():
        raise FileNotFoundError(f"UNet PNG not found: {args.unet_png}")
    if not args.bicubic_npy.exists():
        raise FileNotFoundError(
            f"Bicubic baseline NPY not found: {args.bicubic_npy}\n"
            "Run EP11 benchmark once to cache bicubic_mean_2x_hr_temp.npy."
        )

    png_rgb = plt.imread(args.unet_png)
    panel_rgb = extract_temperature_panel(png_rgb)
    bicubic_temp = np.load(args.bicubic_npy).astype(np.float32, copy=False)

    compare_path, unet_only_path = save_comparison(
        panel_rgb,
        bicubic_temp,
        args.output_dir,
        zoom=float(args.zoom),
        source_zoom=float(args.source_zoom),
        center_fraction=float(args.center_fraction),
        shift_fraction_of_crop=float(args.shift_fraction_of_crop),
        unet_title=str(args.unet_title),
        baseline_title=str(args.baseline_title),
    )

    manifest = {
        "unet_png_source": _relative(args.unet_png),
        "bicubic_npy": _relative(args.bicubic_npy),
        "output_dir": _relative(args.output_dir),
        "center_fraction": float(args.center_fraction),
        "zoom": float(args.zoom),
        "source_zoom": float(args.source_zoom),
        "shift_fraction_of_crop": float(args.shift_fraction_of_crop),
        "figures": [_relative(compare_path), _relative(unet_only_path)],
        "notes": (
            "UNet panel is cropped from the existing eval PNG; "
            "bicubic panel is rendered from cached HR temperature NPY."
        ),
    }
    manifest_path = args.output_dir / "center_zoom4x_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--unet-png", type=Path, default=DEFAULT_UNET_PNG)
    parser.add_argument("--bicubic-npy", type=Path, default=DEFAULT_BICUBIC_NPY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--zoom", type=float, default=4.0)
    parser.add_argument("--source-zoom", type=float, default=3.0)
    parser.add_argument("--center-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--shift-fraction-of-crop", type=float, default=1.0 / 3.0)
    parser.add_argument(
        "--unet-title",
        default="UNet 2x @ EP07 v8_aa step 40000 (PNG crop)",
    )
    parser.add_argument("--baseline-title", default="Bicubic mean 2x (cached)")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (0.0 < float(args.center_fraction) <= 1.0):
        raise ValueError("--center-fraction must be in (0, 1]")
    if float(args.zoom) <= 0 or float(args.source_zoom) <= 0:
        raise ValueError("--zoom and --source-zoom must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
