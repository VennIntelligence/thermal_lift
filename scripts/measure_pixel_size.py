"""Measure detector sampling pitch from a BMP export with mm axes.

The BMP export contains a rendered 640 x 480 data raster plus axes, labels,
and a colorbar. The paired TXT file is the raw 480 x 640 temperature matrix.
This script measures the mm-per-pixel scale from the BMP axis ticks, then
cross-checks it with the Otsu outer contour detected in both representations.
The result is a detector sampling pitch measurement; it must not be confused
with the calibrated 20 um spatial resolution.

用法: uv run python scripts/measure_pixel_size.py
      [--bmp data/data_raw/infrared_avi/10_16_0.bmp] [--txt .../10_16_0.txt] [--output-dir DIR]
输入: 成对的 BMP 渲染图与 TXT 温度矩阵（默认 10_16_0）
输出: output/ep03_theoretical_limits/pixel_size_measurement.png 与 pixel_size_measurement.json
关联: EP03
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from thermal_core.ep03 import detect_outer_contour
from thermal_core.io import load_frame
from thermal_core.plotting import COLORMAPS, savefig_academic, setup_academic_style


CURRENT_SPATIAL_RESOLUTION_UM = 20.0
TARGET_CONTOUR_GRID_UM = 5.0
EXPLORATORY_4X_GRID_UM = 2.5


@dataclass(frozen=True)
class DataRaster:
    """Rendered data raster location in BMP pixel coordinates."""

    x0: int
    y0: int
    x1: int
    y1: int

    @property
    def width(self) -> int:
        return self.x1 - self.x0 + 1

    @property
    def height(self) -> int:
        return self.y1 - self.y0 + 1

    @property
    def left_spine_x(self) -> int:
        return self.x0 - 1

    @property
    def bottom_spine_y(self) -> int:
        return self.y1 + 1


@dataclass(frozen=True)
class AxisScale:
    """Axis-derived physical scale."""

    x_ticks_px: np.ndarray
    y_ticks_px: np.ndarray
    x_px_per_mm: float
    y_px_per_mm: float
    x_origin_px: float
    y_origin_px: float
    pixel_size_x_um: float
    pixel_size_y_um: float

    @property
    def pixel_size_um(self) -> float:
        return 0.5 * (self.pixel_size_x_um + self.pixel_size_y_um)


@dataclass(frozen=True)
class ContourCrossCheck:
    """Contour-based scale cross-check."""

    txt_bbox: tuple[int, int, int, int]
    bmp_bbox: tuple[int, int, int, int]
    txt_width_px: int
    txt_height_px: int
    bmp_width_px: int
    bmp_height_px: int
    pixel_size_x_um: float
    pixel_size_y_um: float
    iou: float

    @property
    def pixel_size_um(self) -> float:
        return 0.5 * (self.pixel_size_x_um + self.pixel_size_y_um)


def project_root() -> Path:
    root = Path(__file__).resolve()
    while root != root.parent and not (root / "AGENTS.md").exists():
        root = root.parent
    if not (root / "AGENTS.md").exists():
        raise RuntimeError("Could not find project root containing AGENTS.md")
    return root


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """Return inclusive runs of True values."""
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return []
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []

    runs: list[tuple[int, int]] = []
    start = int(idx[0])
    prev = int(idx[0])
    for value in idx[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask)
    if xs.size == 0:
        raise RuntimeError("Cannot compute bounding box for an empty mask")
    return int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())


def bbox_size(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x0, y0, x1, y1 = bbox
    return x1 - x0 + 1, y1 - y0 + 1


def detect_data_raster(gray: np.ndarray, *, detector_rows: int, detector_cols: int) -> DataRaster:
    """Detect the rendered 640 x 480 raster inside the BMP export."""
    non_background = gray < 239
    col_counts = non_background.sum(axis=0)
    row_counts = non_background.sum(axis=1)

    col_runs = [
        run
        for run in contiguous_runs(col_counts >= int(0.75 * detector_rows))
        if (run[1] - run[0] + 1) >= int(0.85 * detector_cols)
    ]
    row_runs = [
        run
        for run in contiguous_runs(row_counts >= int(0.75 * detector_cols))
        if (run[1] - run[0] + 1) >= int(0.85 * detector_rows)
    ]
    if not col_runs or not row_runs:
        raise RuntimeError("Could not detect the rendered BMP data raster")

    col_run = min(col_runs, key=lambda run: abs((run[1] - run[0] + 1) - (detector_cols + 1)))
    row_run = min(row_runs, key=lambda run: abs((run[1] - run[0] + 1) - (detector_rows + 1)))

    x0 = col_run[0] + 1
    y0 = row_run[0]
    x1 = x0 + detector_cols - 1
    y1 = y0 + detector_rows - 1

    if x1 > col_run[1]:
        x0 = col_run[1] - detector_cols + 1
        x1 = col_run[1]
    if y1 > row_run[1]:
        y0 = row_run[1] - detector_rows + 1
        y1 = row_run[1]

    raster = DataRaster(x0=x0, y0=y0, x1=x1, y1=y1)
    if raster.width != detector_cols or raster.height != detector_rows:
        raise RuntimeError(f"Unexpected BMP raster size: {raster.width} x {raster.height}")
    return raster


def detect_x_tick_centers(gray: np.ndarray, raster: DataRaster) -> np.ndarray:
    """Detect x-axis tick marks for the visible 1.0 ... 6.0 mm labels."""
    y0 = raster.bottom_spine_y + 1
    y1 = raster.bottom_spine_y + 5
    region = gray[y0:y1, raster.x0 : raster.x1 + 1]
    tick_columns = (region < 235).sum(axis=0) >= 2
    runs = contiguous_runs(tick_columns)
    centers = []
    for start, end in runs:
        width = end - start + 1
        center = raster.x0 + 0.5 * (start + end)
        if 1 <= width <= 4 and center > raster.x0 + 50:
            centers.append(center)
    centers = np.asarray(centers, dtype=float)
    if centers.size < 5:
        raise RuntimeError(f"Detected too few x-axis ticks: {centers}")
    return centers


def detect_y_tick_centers(gray: np.ndarray, raster: DataRaster) -> np.ndarray:
    """Detect y-axis tick marks for the visible 1.0 ... 4.0 mm labels."""
    x0 = max(0, raster.left_spine_x - 8)
    x1 = raster.left_spine_x
    region = gray[raster.y0 : raster.y1 + 1, x0:x1]
    tick_rows = (region < 235).sum(axis=1) >= 3
    runs = contiguous_runs(tick_rows)
    centers = []
    for start, end in runs:
        height = end - start + 1
        center = raster.y0 + 0.5 * (start + end)
        if 1 <= height <= 4:
            centers.append(center)
    centers = np.asarray(centers, dtype=float)
    if centers.size < 4:
        raise RuntimeError(f"Detected too few y-axis ticks: {centers}")
    return centers


def measure_axis_scale(gray: np.ndarray, raster: DataRaster, *, detector_rows: int, detector_cols: int) -> AxisScale:
    x_ticks = detect_x_tick_centers(gray, raster)
    y_ticks = detect_y_tick_centers(gray, raster)

    x_labels_mm = np.arange(1, x_ticks.size + 1, dtype=float)
    x_px_per_mm, x_origin_px = np.polyfit(x_labels_mm, x_ticks, deg=1)

    y_ticks_sorted = np.sort(y_ticks)
    y_labels_mm = np.arange(y_ticks_sorted.size, 0, -1, dtype=float)
    y_slope, y_origin_px = np.polyfit(y_labels_mm, y_ticks_sorted, deg=1)
    y_px_per_mm = abs(float(y_slope))

    x_rendered_px_per_txt_px = raster.width / float(detector_cols)
    y_rendered_px_per_txt_px = raster.height / float(detector_rows)
    pixel_size_x_um = 1000.0 * x_rendered_px_per_txt_px / float(abs(x_px_per_mm))
    pixel_size_y_um = 1000.0 * y_rendered_px_per_txt_px / y_px_per_mm

    return AxisScale(
        x_ticks_px=x_ticks,
        y_ticks_px=y_ticks_sorted,
        x_px_per_mm=float(abs(x_px_per_mm)),
        y_px_per_mm=y_px_per_mm,
        x_origin_px=float(x_origin_px),
        y_origin_px=float(y_origin_px),
        pixel_size_x_um=float(pixel_size_x_um),
        pixel_size_y_um=float(pixel_size_y_um),
    )


def contour_cross_check(frame: np.ndarray, gray: np.ndarray, raster: DataRaster, scale: AxisScale) -> tuple[ContourCrossCheck, np.ndarray, np.ndarray]:
    """Cross-check scale with the sample outer contour in TXT and BMP crop."""
    txt_mask, txt_contour, _ = detect_outer_contour(frame)

    bmp_crop = gray[raster.y0 : raster.y1 + 1, raster.x0 : raster.x1 + 1]
    bmp_temperature_like = 255.0 - bmp_crop.astype(float)
    bmp_mask, bmp_contour, _ = detect_outer_contour(bmp_temperature_like)

    txt_bbox = mask_bbox(txt_mask)
    bmp_bbox = mask_bbox(bmp_mask)
    txt_width, txt_height = bbox_size(txt_bbox)
    bmp_width, bmp_height = bbox_size(bmp_bbox)

    pixel_size_x_um = 1000.0 * (bmp_width / scale.x_px_per_mm) / txt_width
    pixel_size_y_um = 1000.0 * (bmp_height / scale.y_px_per_mm) / txt_height
    union = np.logical_or(txt_mask, bmp_mask).sum()
    iou = float(np.logical_and(txt_mask, bmp_mask).sum() / union)

    return (
        ContourCrossCheck(
            txt_bbox=txt_bbox,
            bmp_bbox=bmp_bbox,
            txt_width_px=txt_width,
            txt_height_px=txt_height,
            bmp_width_px=bmp_width,
            bmp_height_px=bmp_height,
            pixel_size_x_um=float(pixel_size_x_um),
            pixel_size_y_um=float(pixel_size_y_um),
            iou=iou,
        ),
        txt_contour,
        bmp_contour,
    )


def contour_to_mm(contour: np.ndarray, *, pixel_size_um: float, detector_rows: int) -> tuple[np.ndarray, np.ndarray]:
    pixel_size_mm = pixel_size_um / 1000.0
    fov_y_mm = detector_rows * pixel_size_mm
    x_mm = (contour[:, 0] + 0.5) * pixel_size_mm
    y_mm = fov_y_mm - (contour[:, 1] + 0.5) * pixel_size_mm
    return x_mm, y_mm


def find_corners(contour: np.ndarray, theta_deg: float = 47.6) -> list[int]:
    """Find indices of the 4 outer corner points of the rotated square contour."""
    contour_cartesian = contour.copy()
    contour_cartesian[:, 1] = -contour_cartesian[:, 1]
    
    theta_rad = np.radians(theta_deg)
    angles = [
        theta_rad - np.pi / 4,
        theta_rad + np.pi / 4,
        theta_rad + 3 * np.pi / 4,
        theta_rad - 3 * np.pi / 4,
    ]
    
    corners_idx = []
    for angle in angles:
        d = np.array([np.cos(angle), np.sin(angle)])
        proj = contour_cartesian[:, 0] * d[0] + contour_cartesian[:, 1] * d[1]
        corners_idx.append(np.argmax(proj))
    return corners_idx


def make_visualization(
    bmp_rgb: np.ndarray,
    frame: np.ndarray,
    raster: DataRaster,
    scale: AxisScale,
    pixel_size_um: float,
    txt_contour: np.ndarray,
    bmp_contour: np.ndarray,
    output_path: Path,
) -> Path:
    setup_academic_style()
    detector_rows, detector_cols = frame.shape
    pixel_size_mm = pixel_size_um / 1000.0
    fov_x_mm = detector_cols * pixel_size_mm
    fov_y_mm = detector_rows * pixel_size_mm

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.8), constrained_layout=True)

    ax = axes[0]
    ax.imshow(bmp_rgb, origin="upper")
    ax.set_axis_off()
    rect = patches.Rectangle(
        (raster.x0 - 0.5, raster.y0 - 0.5),
        raster.width,
        raster.height,
        fill=False,
        edgecolor="#C44E52",
        linewidth=2.0,
    )
    ax.add_patch(rect)
    ax.plot(
        bmp_contour[:, 0] + raster.x0,
        bmp_contour[:, 1] + raster.y0,
        color="#55A868",
        linewidth=1.5,
    )
    ax.scatter(scale.x_ticks_px, np.full_like(scale.x_ticks_px, raster.bottom_spine_y + 3.0), s=8, c="#4C72B0")
    ax.scatter(np.full_like(scale.y_ticks_px, raster.left_spine_x - 4.0), scale.y_ticks_px, s=8, c="#4C72B0")

    ax = axes[1]
    im = ax.imshow(
        frame,
        extent=[0.0, fov_x_mm, 0.0, fov_y_mm],
        origin="upper",
        cmap=COLORMAPS["temperature"],
        aspect="equal",
    )
    x_mm, y_mm = contour_to_mm(txt_contour, pixel_size_um=pixel_size_um, detector_rows=detector_rows)
    ax.plot(x_mm, y_mm, color="#55A868", linewidth=1.5)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_xlim(0.0, fov_x_mm)
    ax.set_ylim(0.0, fov_y_mm)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, label="Temperature (Celsius)")

    # Find matching corner points to show correspondence
    bmp_corners_idx = find_corners(bmp_contour, theta_deg=47.6)
    txt_corners_idx = find_corners(txt_contour, theta_deg=47.6)

    # Plot connecting lines and highlight points on both axes
    for b_idx, t_idx in zip(bmp_corners_idx, txt_corners_idx):
        p_bmp = bmp_contour[b_idx]
        p_txt = txt_contour[t_idx]
        
        # Left subplot (pixel coords)
        x_left = p_bmp[0] + raster.x0
        y_left = p_bmp[1] + raster.y0
        
        # Right subplot (mm coords)
        x_right = (p_txt[0] + 0.5) * pixel_size_mm
        y_right = fov_y_mm - (p_txt[1] + 0.5) * pixel_size_mm
        
        # Draw red dot markers
        axes[0].scatter(x_left, y_left, color="#C44E52", s=18, zorder=5)
        axes[1].scatter(x_right, y_right, color="#C44E52", s=18, zorder=5)
        
        # Connect the points across subplots
        con = patches.ConnectionPatch(
            xyA=(x_left, y_left),
            xyB=(x_right, y_right),
            coordsA="data",
            coordsB="data",
            axesA=axes[0],
            axesB=axes[1],
            color="#C44E52",
            linestyle="--",
            linewidth=0.9,
            alpha=0.6,
        )
        fig.add_artist(con)

    return savefig_academic(fig, output_path)


def resolution_distinction(pixel_size_um: float, *, frame_shape: tuple[int, int]) -> dict[str, float]:
    """Summarize detector pitch, calibrated resolution, and SR grid targets."""
    rows, cols = frame_shape
    return {
        "detector_sampling_pitch_um": float(pixel_size_um),
        "current_spatial_resolution_um": float(CURRENT_SPATIAL_RESOLUTION_UM),
        "target_contour_grid_um": float(TARGET_CONTOUR_GRID_UM),
        "target_grid_factor": float(pixel_size_um / TARGET_CONTOUR_GRID_UM),
        "exploratory_4x_grid_um": float(EXPLORATORY_4X_GRID_UM),
        "spatial_resolution_in_detector_pixels": float(CURRENT_SPATIAL_RESOLUTION_UM / pixel_size_um),
        "fov_x_mm": float(cols * pixel_size_um / 1000.0),
        "fov_y_mm": float(rows * pixel_size_um / 1000.0),
    }


def print_results(
    bmp_path: Path,
    txt_path: Path,
    bmp_shape: tuple[int, int, int],
    frame_shape: tuple[int, int],
    raster: DataRaster,
    scale: AxisScale,
    contour_check: ContourCrossCheck,
    distinction: dict[str, float],
    output_path: Path,
) -> None:
    pixel_size_um = scale.pixel_size_um
    fov_x_mm = frame_shape[1] * pixel_size_um / 1000.0
    fov_y_mm = frame_shape[0] * pixel_size_um / 1000.0

    print("Pixel size measurement from BMP axes")
    print(f"  BMP: {bmp_path}")
    print(f"  TXT: {txt_path}")
    print(f"  BMP shape: {bmp_shape[1]} x {bmp_shape[0]} px")
    print(f"  TXT shape: {frame_shape[0]} x {frame_shape[1]} pixels")
    print(f"  Rendered data bbox: x={raster.x0}..{raster.x1}, y={raster.y0}..{raster.y1} ({raster.width} x {raster.height})")
    print("")
    print("Method A: BMP coordinate-axis ticks")
    print(f"  X tick centers for 1.0..{scale.x_ticks_px.size:.1f} mm: {np.array2string(scale.x_ticks_px, precision=2)}")
    print(f"  Y tick centers for {scale.y_ticks_px.size:.1f}..1.0 mm: {np.array2string(scale.y_ticks_px, precision=2)}")
    print(f"  X scale: {scale.x_px_per_mm:.6f} BMP px/mm -> {scale.pixel_size_x_um:.6f} um/TXT pixel")
    print(f"  Y scale: {scale.y_px_per_mm:.6f} BMP px/mm -> {scale.pixel_size_y_um:.6f} um/TXT pixel")
    print(f"  FOV: {fov_x_mm:.6f} mm x {fov_y_mm:.6f} mm")
    print("")
    print("Method B: sample outer-contour cross-check")
    print(f"  TXT contour bbox: x={contour_check.txt_bbox[0]}..{contour_check.txt_bbox[2]}, y={contour_check.txt_bbox[1]}..{contour_check.txt_bbox[3]} ({contour_check.txt_width_px} x {contour_check.txt_height_px})")
    print(f"  BMP contour bbox in data crop: x={contour_check.bmp_bbox[0]}..{contour_check.bmp_bbox[2]}, y={contour_check.bmp_bbox[1]}..{contour_check.bmp_bbox[3]} ({contour_check.bmp_width_px} x {contour_check.bmp_height_px})")
    print(f"  Mask IoU after crop alignment: {contour_check.iou:.6f}")
    print(f"  Contour-derived pixel size: X={contour_check.pixel_size_x_um:.6f} um, Y={contour_check.pixel_size_y_um:.6f} um, mean={contour_check.pixel_size_um:.6f} um")
    print("")
    print("Final result")
    print(f"  measured_pixel_size_um = {pixel_size_um:.6f}")
    print(f"  detector FOV = {distinction['fov_x_mm']:.6f} mm x {distinction['fov_y_mm']:.6f} mm")
    print("")
    print("Resolution distinction")
    print(f"  detector sampling pitch = {distinction['detector_sampling_pitch_um']:.6f} um/pixel")
    print(f"  calibrated spatial resolution = {distinction['current_spatial_resolution_um']:.6f} um")
    print(f"  spatial resolution spans {distinction['spatial_resolution_in_detector_pixels']:.3f} detector pixels")
    print(f"  5 um contour-grid target requires {distinction['target_grid_factor']:.3f}x sampling")
    print(f"  4x exploratory grid sample = {distinction['exploratory_4x_grid_um']:.6f} um")
    print("")
    print(f"Saved visualization: {output_path}")


def write_measurement_json(
    output_path: Path,
    *,
    raster: DataRaster,
    scale: AxisScale,
    contour_check: ContourCrossCheck,
    distinction: dict[str, float],
) -> Path:
    payload = {
        "data_raster_bbox_xyxy": [raster.x0, raster.y0, raster.x1, raster.y1],
        "axis_method": {
            "x_px_per_mm": scale.x_px_per_mm,
            "y_px_per_mm": scale.y_px_per_mm,
            "pixel_size_x_um": scale.pixel_size_x_um,
            "pixel_size_y_um": scale.pixel_size_y_um,
            "pixel_size_mean_um": scale.pixel_size_um,
            "x_tick_centers_px": scale.x_ticks_px.tolist(),
            "y_tick_centers_px": scale.y_ticks_px.tolist(),
        },
        "contour_cross_check": {
            "txt_bbox_xyxy": list(contour_check.txt_bbox),
            "bmp_bbox_xyxy": list(contour_check.bmp_bbox),
            "txt_width_px": contour_check.txt_width_px,
            "txt_height_px": contour_check.txt_height_px,
            "bmp_width_px": contour_check.bmp_width_px,
            "bmp_height_px": contour_check.bmp_height_px,
            "pixel_size_x_um": contour_check.pixel_size_x_um,
            "pixel_size_y_um": contour_check.pixel_size_y_um,
            "pixel_size_mean_um": contour_check.pixel_size_um,
            "mask_iou": contour_check.iou,
        },
        "resolution_distinction": distinction,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def build_parser() -> argparse.ArgumentParser:
    root = project_root()
    default_data_dir = root / "data" / "data_raw" / "infrared_avi"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bmp", type=Path, default=default_data_dir / "10_16_0.bmp")
    parser.add_argument("--txt", type=Path, default=default_data_dir / "10_16_0.txt")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "ep03_theoretical_limits")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    bmp_path = args.bmp.resolve()
    txt_path = args.txt.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    bmp_rgb = np.asarray(Image.open(bmp_path).convert("RGB"))
    gray = np.asarray(Image.fromarray(bmp_rgb).convert("L"))
    frame = load_frame(txt_path)
    detector_rows, detector_cols = frame.shape

    raster = detect_data_raster(gray, detector_rows=detector_rows, detector_cols=detector_cols)
    scale = measure_axis_scale(gray, raster, detector_rows=detector_rows, detector_cols=detector_cols)
    contour_check, txt_contour, bmp_contour = contour_cross_check(frame, gray, raster, scale)
    distinction = resolution_distinction(scale.pixel_size_um, frame_shape=frame.shape)

    figure_path = output_dir / "pixel_size_measurement.png"
    saved_figure = make_visualization(
        bmp_rgb,
        frame,
        raster,
        scale,
        scale.pixel_size_um,
        txt_contour,
        bmp_contour,
        figure_path,
    )
    write_measurement_json(
        output_dir / "pixel_size_measurement.json",
        raster=raster,
        scale=scale,
        contour_check=contour_check,
        distinction=distinction,
    )

    print_results(
        bmp_path,
        txt_path,
        bmp_rgb.shape,
        frame.shape,
        raster,
        scale,
        contour_check,
        distinction,
        saved_figure,
    )


if __name__ == "__main__":
    main()
