"""Build transparent edge-line overlays for TXT and BMP frames.

This is a visual sanity check for apparent motion.  It keeps only detected
edge pixels from each frame; all non-edge pixels are transparent.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter, shift as ndi_shift

from thermal_core.displacement import coordinate_to_shift
from thermal_core.io import load_frame
from thermal_core.plotting import savefig_academic, setup_academic_style


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


def robust_u8(frame: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(frame, [1, 99])
    norm = np.clip((frame - lo) / (hi - lo + 1e-9), 0, 1)
    return (255.0 * norm).astype(np.uint8)


def auto_canny(image_u8: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    blurred = cv2.GaussianBlur(image_u8, (3, 3), 0)
    median = float(np.median(blurred))
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    if upper <= lower:
        lower, upper = 40, 120
    edges = cv2.Canny(blurred, lower, upper)
    return edges > 0


def txt_edge_mask(frame: np.ndarray) -> np.ndarray:
    # Remove slow thermal background first; this keeps structure edges dominant.
    hp = frame.astype(np.float32) - gaussian_filter(frame.astype(np.float32), sigma=8.0, mode="nearest")
    return auto_canny(robust_u8(hp))


def bmp_edge_mask(rgb_crop: np.ndarray) -> np.ndarray:
    gray = np.dot(rgb_crop[..., :3], np.array([0.299, 0.587, 0.114], dtype=np.float32))
    return auto_canny(robust_u8(gray))


def load_bmp_crop(path: Path, raster: DataRaster | None) -> tuple[np.ndarray, DataRaster]:
    rgb = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if raster is None:
        gray = np.asarray(Image.fromarray((rgb * 255).astype(np.uint8)).convert("L"))
        raster = detect_data_raster(gray)
    crop = rgb[raster.y0 : raster.y1 + 1, raster.x0 : raster.x1 + 1]
    return crop, raster


def load_main_frames(audit_csv: Path, data_dir: Path, alignment: pd.DataFrame) -> pd.DataFrame:
    audit = pd.read_csv(audit_csv)
    main = audit[boolish(audit["is_main_session"])].copy()
    available = set(alignment.loc[boolish(alignment["success"]), "file"].astype(str))
    main = main[main["file"].astype(str).isin(available)].copy()
    main = main[[Path(data_dir / (Path(name).stem + ".bmp")).exists() for name in main["file"].astype(str)]]
    return main.sort_values("acquisition_order").reset_index(drop=True)


def fit_filename_affine(alignment: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    valid = alignment[boolish(alignment["success"]) & alignment["R"].eq(0)].copy()
    design = np.column_stack([np.ones(len(valid)), valid["X"].to_numpy(), valid["Y"].to_numpy()])
    beta_dx = np.linalg.lstsq(design, valid["refined_align_dx_px"].to_numpy(), rcond=None)[0]
    beta_dy = np.linalg.lstsq(design, valid["refined_align_dy_px"].to_numpy(), rcond=None)[0]
    return beta_dx, beta_dy


def build_shift_tables(
    frames: pd.DataFrame,
    alignment: pd.DataFrame,
    stage_config: dict,
) -> dict[str, dict[str, tuple[float, float]]]:
    ref_file = str(alignment["reference_file"].dropna().iloc[0])
    ref_row = frames[frames["file"].eq(ref_file)].iloc[0]
    beta_dx, beta_dy = fit_filename_affine(alignment)
    align_lookup = {
        str(row["file"]): (float(row["refined_align_dx_px"]), float(row["refined_align_dy_px"]))
        for _, row in alignment[boolish(alignment["success"])].iterrows()
    }
    theta_deg = float(stage_config["theta_deg"])
    pixel_size_um = float(stage_config["pixel_size_um"])
    tables = {
        "No alignment": {},
        "Old stage model": {},
        "Filename affine": {},
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
        design = np.array([1.0, float(row["X"]), float(row["Y"])])
        tables["Filename affine"][name] = (float(design @ beta_dx), float(design @ beta_dy))
        tables["Data-driven contour"][name] = align_lookup[name]
    return tables


def composite_edges(
    frames: pd.DataFrame,
    data_dir: Path,
    shifts: dict[str, tuple[float, float]],
    modality: str,
    alpha_per_frame: float,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = 480, 640
    rgba = np.zeros((rows, cols, 4), dtype=np.float32)
    persistence = np.zeros((rows, cols), dtype=np.float32)
    cmap = plt.get_cmap("turbo")
    raster = None

    for idx, (_, row) in enumerate(frames.iterrows()):
        name = str(row["file"])
        dx, dy = shifts[name]
        if modality == "txt":
            frame = load_frame(data_dir / name).astype(np.float32, copy=False)
            edges = txt_edge_mask(frame)
        elif modality == "bmp":
            bmp_name = Path(name).stem + ".bmp"
            bmp, raster = load_bmp_crop(data_dir / bmp_name, raster)
            edges = bmp_edge_mask(bmp)
        else:
            raise ValueError("modality must be 'txt' or 'bmp'")

        shifted = ndi_shift(edges.astype(np.float32), shift=(dy, dx), order=0, mode="constant", cval=0.0) > 0.5
        if not np.any(shifted):
            continue
        color = np.asarray(cmap(idx / max(1, len(frames) - 1))[:3], dtype=np.float32)
        src_alpha = alpha_per_frame
        dst_alpha = rgba[..., 3]
        mask = shifted
        out_alpha = src_alpha + dst_alpha[mask] * (1.0 - src_alpha)
        rgba[mask, :3] = (
            color * src_alpha + rgba[mask, :3] * dst_alpha[mask, None] * (1.0 - src_alpha)
        ) / np.maximum(out_alpha[:, None], 1e-6)
        rgba[mask, 3] = out_alpha
        persistence[mask] += 1.0

    persistence /= float(len(frames))
    return rgba, persistence


def persistence_rgba(persistence: np.ndarray) -> np.ndarray:
    cmap = plt.get_cmap("magma")
    norm = np.clip(persistence / max(float(np.percentile(persistence[persistence > 0], 99)) if np.any(persistence > 0) else 1.0, 1e-6), 0, 1)
    rgba = cmap(norm).astype(np.float32)
    rgba[..., 3] = np.clip(norm * 1.2, 0, 1)
    rgba[persistence <= 0, 3] = 0.0
    return rgba


def plot_grid(outputs: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]], output_path: Path, n_frames: int) -> None:
    setup_academic_style()
    methods = list(outputs)
    fig, axes = plt.subplots(4, 4, figsize=(13.8, 12.0), constrained_layout=True)
    row_specs = [
        ("txt", 0, "TXT edge lines"),
        ("txt", 1, "TXT edge persistence"),
        ("bmp", 0, "BMP edge lines"),
        ("bmp", 1, "BMP edge persistence"),
    ]
    for col, method in enumerate(methods):
        axes[0, col].set_title(method)
        for row_idx, (modality, kind, label) in enumerate(row_specs):
            edge_rgba, persistence = outputs[method][modality]
            image = edge_rgba if kind == 0 else persistence_rgba(persistence)
            axes[row_idx, col].imshow(image, interpolation="nearest")
            axes[row_idx, col].axis("off")
            axes[row_idx, col].set_facecolor((1, 1, 1, 0))
            if col == 0:
                axes[row_idx, col].set_ylabel(label, fontsize=9)
    fig.suptitle(f"Transparent edge-only overlays, n={n_frames} paired main-session frames", fontsize=12, fontweight="bold")
    fig.patch.set_alpha(0.0)
    savefig_academic(fig, output_path, close=True)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-audit-csv", type=Path, default=root / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--alignment-csv", type=Path, default=root / "output" / "ep05_contour_alignment" / "contour_alignment_results.csv")
    parser.add_argument("--stage-config", type=Path, default=root / "configs" / "stage_calibration.json")
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "data_raw" / "infrared_avi")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "ep05_overlay_alignment")
    parser.add_argument("--alpha-per-frame", type=float, default=0.055)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    alignment = pd.read_csv(args.alignment_csv)
    frames = load_main_frames(args.frame_audit_csv, args.data_dir, alignment)
    with open(args.stage_config, encoding="utf-8") as f:
        stage_config = json.load(f)
    shift_tables = build_shift_tables(frames, alignment, stage_config)

    outputs = {}
    for method, shifts in shift_tables.items():
        print(f"Building edge-line overlays: {method}")
        outputs[method] = {
            "txt": composite_edges(frames, args.data_dir, shifts, "txt", args.alpha_per_frame),
            "bmp": composite_edges(frames, args.data_dir, shifts, "bmp", args.alpha_per_frame),
        }
    output_path = args.output_dir / "all_main_4x4_edge_line_overlay.png"
    plot_grid(outputs, output_path, len(frames))
    print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
