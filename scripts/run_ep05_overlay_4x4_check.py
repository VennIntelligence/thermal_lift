"""Build a 4x4 TXT/BMP overlay grid for global alignment sanity checking.

Columns are alignment methods.  Rows are:
1. TXT mean alpha-composite style overlay;
2. TXT edge persistence;
3. BMP rendered-data-crop mean overlay;
4. BMP rendered-data-crop edge persistence.

用法（项目根目录）::

    uv run python scripts/run_ep05_overlay_4x4_check.py [--edge-percentile 93.0]

输入依赖: output/ep01_data_processing/frame_audit.csv、EP05 contour-refined 对齐 CSV
    （默认由 configs/alignment/paths.json 解析，--alignment-csv 可覆盖）、
    configs/stage_calibration.json、data/data_raw/infrared_avi/（TXT + BMP）
输出: output/ep05_overlay_alignment/（--output-dir 可覆盖）4x4 叠加对比图

关联: EP05
"""

from __future__ import annotations

from thermal_core.alignment_paths import default_contour_alignment_csv
import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter, shift as ndi_shift, sobel

from thermal_core.displacement import coordinate_to_shift
from thermal_core.ep05 import affine_shift, filename_affine_diagnostics, fit_filename_affine
from thermal_core.io import load_frame
from thermal_core.plotting import COLORMAPS, savefig_academic, setup_academic_style


@dataclass(frozen=True)
class DataRaster:
    x0: int
    y0: int
    x1: int
    y1: int


def project_root() -> Path:
    root = Path.cwd()
    while root != root.parent and not (root / "AGENTS.md").exists():
        root = root.parent
    return root


def boolish(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def contiguous_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    runs = []
    start = prev = int(idx[0])
    for value in idx[1:]:
        value = int(value)
        if value == prev + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def detect_data_raster(gray: np.ndarray, detector_rows: int = 480, detector_cols: int = 640) -> DataRaster:
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
        raise RuntimeError("Could not detect BMP rendered data raster")
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
    return DataRaster(x0=x0, y0=y0, x1=x1, y1=y1)


def robust_norm(frame: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(frame, [1, 99])
    return np.clip((frame - lo) / (hi - lo + 1e-9), 0, 1)


def highpass(frame: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    return np.asarray(frame, dtype=np.float32) - gaussian_filter(frame, sigma=sigma, mode="nearest")


def gradient_magnitude(frame: np.ndarray) -> np.ndarray:
    hp = highpass(np.asarray(frame, dtype=np.float32))
    gx = sobel(hp, axis=1, mode="nearest")
    gy = sobel(hp, axis=0, mode="nearest")
    return np.hypot(gx, gy).astype(np.float32, copy=False)


def edge_mask(frame: np.ndarray, percentile: float) -> np.ndarray:
    grad = gradient_magnitude(frame)
    threshold = float(np.percentile(grad, percentile))
    mask = grad >= threshold
    mask[:2, :] = False
    mask[-2:, :] = False
    mask[:, :2] = False
    mask[:, -2:] = False
    return mask.astype(np.float32)


def center_crop(image: np.ndarray, zoom_factor: float) -> np.ndarray:
    """Crop the central field of view so displayed structure is magnified."""
    if zoom_factor <= 1.0:
        return image
    rows, cols = image.shape[:2]
    crop_rows = max(1, int(round(rows / zoom_factor)))
    crop_cols = max(1, int(round(cols / zoom_factor)))
    y0 = max(0, (rows - crop_rows) // 2)
    x0 = max(0, (cols - crop_cols) // 2)
    return image[y0 : y0 + crop_rows, x0 : x0 + crop_cols]


def load_main_frames(audit_csv: Path, data_dir: Path, alignment: pd.DataFrame) -> pd.DataFrame:
    audit = pd.read_csv(audit_csv)
    main = audit[boolish(audit["is_main_session"])].copy()
    available = set(alignment.loc[boolish(alignment["success"]), "file"].astype(str))
    main = main[main["file"].astype(str).isin(available)].copy()
    main = main[(main["data_bmp_exists"] if "data_bmp_exists" in main else True)] if False else main
    main = main[[Path(data_dir / (Path(name).stem + ".bmp")).exists() for name in main["file"].astype(str)]]
    return main.sort_values("acquisition_order").reset_index(drop=True)


def build_shift_tables(
    frames: pd.DataFrame,
    alignment: pd.DataFrame,
    stage_config: dict,
) -> dict[str, dict[str, tuple[float, float]]]:
    ref_file = str(alignment["reference_file"].dropna().iloc[0])
    ref = frames[frames["file"].eq(ref_file)]
    if ref.empty:
        raise RuntimeError(f"Reference file {ref_file} is not in selected frame set")
    ref_row = ref.iloc[0]
    affine_fit = fit_filename_affine(alignment, robust=True)
    diagnosis = filename_affine_diagnostics(alignment, affine_fit)
    excluded_from_overlay = set(
        diagnosis.loc[diagnosis["residual_gate_outlier"], "file"].astype(str).tolist()
    )
    filename_label = f"Filename affine\n(n_excluded={len(excluded_from_overlay)})"
    print(
        "Filename affine robust fit: "
        f"fit_rows={affine_fit.fit_count}, clean_rows={affine_fit.clean_count}, "
        f"median_res={affine_fit.median_residual_px:.4f}px, "
        f"threshold={affine_fit.outlier_threshold_px:.4f}px, "
        f"fit_excluded={list(affine_fit.excluded_files)}, "
        f"overlay_excluded={sorted(excluded_from_overlay)}"
    )
    align_lookup = {
        str(row["file"]): (float(row["refined_align_dx_px"]), float(row["refined_align_dy_px"]))
        for _, row in alignment[boolish(alignment["success"])].iterrows()
    }
    theta_deg = float(stage_config["theta_deg"])
    pixel_size_um = float(stage_config["pixel_size_um"])

    tables: dict[str, dict[str, tuple[float, float]]] = {
        "No alignment": {},
        "Old stage model": {},
        filename_label: {},
        "Data-driven contour": {},
    }
    for _, row in frames.iterrows():
        name = str(row["file"])
        tables["No alignment"][name] = (0.0, 0.0)

        dx_stage, dy_stage_math = coordinate_to_shift(
            float(row["X"]) - float(ref_row["X"]),
            float(row["Y"]) - float(ref_row["Y"]),
            theta_deg=theta_deg,
            pixel_size_um=pixel_size_um,
        )
        tables["Old stage model"][name] = (-float(dx_stage), float(dy_stage_math))

        if name not in excluded_from_overlay:
            tables[filename_label][name] = affine_shift(row, affine_fit.beta_dx, affine_fit.beta_dy)
        tables["Data-driven contour"][name] = align_lookup[name]
    return tables


def load_bmp_crop(path: Path, raster: DataRaster | None = None) -> tuple[np.ndarray, DataRaster]:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if raster is None:
        gray = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).convert("L"))
        raster = detect_data_raster(gray)
    crop = rgb[raster.y0 : raster.y1 + 1, raster.x0 : raster.x1 + 1]
    return crop, raster


def make_overlays(
    frames: pd.DataFrame,
    data_dir: Path,
    shifts: dict[str, tuple[float, float]],
    edge_percentile: float,
) -> dict[str, np.ndarray]:
    txt_sum = None
    txt_edge_sum = None
    bmp_sum = None
    bmp_edge_sum = None
    raster = None
    n = 0
    for _, row in frames.iterrows():
        name = str(row["file"])
        if name not in shifts:
            continue
        dx, dy = shifts[name]

        txt = load_frame(data_dir / name).astype(np.float32, copy=False)
        txt_shifted = ndi_shift(txt, shift=(dy, dx), order=1, mode="nearest")
        txt_norm = robust_norm(txt_shifted)
        txt_edges = edge_mask(txt_shifted, edge_percentile)

        bmp_name = Path(name).stem + ".bmp"
        bmp, raster = load_bmp_crop(data_dir / bmp_name, raster)
        bmp_gray = np.dot(bmp[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
        bmp_shifted = ndi_shift(bmp, shift=(dy, dx, 0), order=1, mode="nearest")
        bmp_gray_shifted = ndi_shift(bmp_gray, shift=(dy, dx), order=1, mode="nearest")
        bmp_edges = edge_mask(bmp_gray_shifted, edge_percentile)

        txt_sum = txt_norm if txt_sum is None else txt_sum + txt_norm
        txt_edge_sum = txt_edges if txt_edge_sum is None else txt_edge_sum + txt_edges
        bmp_sum = bmp_shifted if bmp_sum is None else bmp_sum + bmp_shifted
        bmp_edge_sum = bmp_edges if bmp_edge_sum is None else bmp_edge_sum + bmp_edges
        n += 1

    if n == 0:
        raise RuntimeError("No frames were available for overlay construction")
    return {
        "txt_mean": txt_sum / n,
        "txt_edges": txt_edge_sum / n,
        "bmp_mean": bmp_sum / n,
        "bmp_edges": bmp_edge_sum / n,
    }


def plot_grid(method_outputs: dict[str, dict[str, np.ndarray]], output_path: Path, n_frames: int) -> None:
    setup_academic_style()
    methods = list(method_outputs)
    row_labels = [
        "TXT mean overlay",
        "TXT edge persistence",
        "BMP mean overlay",
        "BMP edge persistence",
    ]
    fig, axes = plt.subplots(4, 4, figsize=(10.2, 8.3), constrained_layout=True, facecolor="white")
    for col, method in enumerate(methods):
        outputs = method_outputs[method]
        axes[0, col].imshow(center_crop(outputs["txt_mean"], 3.0), cmap=COLORMAPS["temperature"], interpolation="nearest")
        axes[1, col].imshow(
            center_crop(outputs["txt_edges"], 3.0), cmap=COLORMAPS["coverage"], vmin=0, vmax=1, interpolation="nearest"
        )
        axes[2, col].imshow(center_crop(outputs["bmp_mean"], 3.0), interpolation="nearest")
        axes[3, col].imshow(
            center_crop(outputs["bmp_edges"], 3.0), cmap=COLORMAPS["coverage"], vmin=0, vmax=1, interpolation="nearest"
        )
        axes[0, col].set_title(method, fontsize=11, pad=4)
        for row in range(4):
            axes[row, col].axis("off")
            axes[row, col].set_facecolor("white")
            if col == 0:
                axes[row, col].set_ylabel(row_labels[row], fontsize=11)
    savefig_academic(fig, output_path)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-audit-csv", type=Path, default=root / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--alignment-csv", type=Path, default=default_contour_alignment_csv(project_root_path=root))
    parser.add_argument("--stage-config", type=Path, default=root / "configs" / "stage_calibration.json")
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "data_raw" / "infrared_avi")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "ep05_overlay_alignment")
    parser.add_argument("--edge-percentile", type=float, default=93.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    alignment = pd.read_csv(args.alignment_csv)
    frames = load_main_frames(args.frame_audit_csv, args.data_dir, alignment)
    with open(args.stage_config, encoding="utf-8") as f:
        stage_config = json.load(f)
    shift_tables = build_shift_tables(frames, alignment, stage_config)

    method_outputs = {}
    for method, shifts in shift_tables.items():
        print(f"Building {method} overlay for {len(frames)} paired frames")
        method_outputs[method] = make_overlays(frames, args.data_dir, shifts, args.edge_percentile)

    output_path = args.output_dir / "all_main_4x4_txt_bmp_overlay.png"
    plot_grid(method_outputs, output_path, len(frames))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
