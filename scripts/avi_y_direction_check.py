#!/usr/bin/env python3
"""EP02 AVI motion registration check.

This script uses the AVI files as continuous-motion diagnostic evidence.
AVI is rendered video, so it is not SR input and not alignment truth. The
output focuses on per-axis motion direction, speed stability, duplicate-frame
rate, and abnormal segments.

Usage:
    uv run python scripts/avi_y_direction_check.py
    uv run python scripts/avi_y_direction_check.py --only y0um.avi x.avi

Outputs:
    output/ep02_displacement_calibration/avi_direction_summary.csv
    output/ep02_displacement_calibration/avi_registration_pairs.csv
    output/ep02_displacement_calibration/avi_direction_comparison.png
    output/ep02_displacement_calibration/avi_cumulative_motion_paths.png
    output/ep02_displacement_calibration/avi_y0um_displacement_timeseries.png
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "data_raw" / "infrared_avi"
OUTPUT_DIR = PROJECT_ROOT / "output" / "ep02_displacement_calibration"

sys.path.insert(0, str(PROJECT_ROOT / "core" / "src"))
from thermal_core.plotting import METHOD_COLOR_LIST, make_figure, savefig_academic, setup_academic_style


# The AVI frame includes plot axes and a colorbar. The thermal image itself is
# the calibrated 640x480 detector area embedded at this fixed location.
DEFAULT_CROP_XYWH = (50, 45, 640, 480)


@dataclass(frozen=True)
class AviSpec:
    path: Path
    scan_axis: str
    fixed_um: float | None


def natural_avi_key(path: Path) -> tuple[str, float, str]:
    stem = path.stem.lower()
    if stem == "x":
        return ("x", -1.0, stem)
    match = re.fullmatch(r"([xy])(\d+)um", stem)
    if match:
        return (match.group(1), float(match.group(2)), stem)
    return ("z", float("inf"), stem)


def parse_avi_spec(path: Path) -> AviSpec | None:
    stem = path.stem.lower()
    if stem == "x":
        return AviSpec(path=path, scan_axis="x", fixed_um=None)
    match = re.fullmatch(r"([xy])(\d+)um", stem)
    if not match:
        return None
    return AviSpec(path=path, scan_axis=match.group(1), fixed_um=float(match.group(2)))


def parse_crop(value: str) -> tuple[int, int, int, int]:
    parts = [int(item.strip()) for item in value.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("crop must be 'x,y,w,h'")
    x, y, width, height = parts
    if min(parts) < 0 or width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError("crop values must be non-negative with positive width/height")
    return x, y, width, height


def zscore(frame: np.ndarray) -> np.ndarray:
    data = frame.astype(np.float32, copy=False)
    centered = data - float(np.mean(data))
    scale = float(np.std(centered))
    if scale < 1e-8:
        return centered
    return centered / scale


def preprocess_frame(frame: np.ndarray, mode: str, highpass_sigma: float) -> np.ndarray:
    if mode == "raw":
        return zscore(frame)
    if mode == "highpass":
        baseline = cv2.GaussianBlur(frame, (0, 0), sigmaX=highpass_sigma, sigmaY=highpass_sigma)
        return zscore(frame - baseline)
    if mode == "gradient":
        gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
        return zscore(cv2.magnitude(gx, gy))
    raise ValueError(f"unknown preprocess mode: {mode}")


def read_avi_frames(path: Path, crop_xywh: tuple[int, int, int, int], max_frames: int | None) -> tuple[list[np.ndarray], dict[str, float | int]]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open AVI: {path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    declared_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    crop_x, crop_y, crop_w, crop_h = crop_xywh
    if crop_x + crop_w > frame_width or crop_y + crop_h > frame_height:
        raise ValueError(
            f"crop {crop_xywh} exceeds frame size {(frame_width, frame_height)} for {path.name}"
        )

    frames: list[np.ndarray] = []
    while max_frames is None or len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        frames.append(gray[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w])
    cap.release()

    meta = {
        "fps": fps,
        "declared_frames": declared_count,
        "read_frames": len(frames),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "crop_x": crop_x,
        "crop_y": crop_y,
        "crop_w": crop_w,
        "crop_h": crop_h,
    }
    return frames, meta


def remove_duplicate_frames(frames: list[np.ndarray], threshold: float) -> tuple[list[np.ndarray], list[int], np.ndarray]:
    if not frames:
        return [], [], np.array([], dtype=float)

    unique_frames = [frames[0]]
    unique_indices = [0]
    diffs = np.zeros(len(frames), dtype=float)
    for idx in range(1, len(frames)):
        diff = float(np.mean(np.abs(frames[idx] - frames[idx - 1])))
        diffs[idx] = diff
        if diff > threshold:
            unique_frames.append(frames[idx])
            unique_indices.append(idx)
    return unique_frames, unique_indices, diffs


def parabolic_offset(left: float, center: float, right: float) -> float:
    denom = left - 2.0 * center + right
    if abs(denom) < 1e-12:
        return 0.0
    offset = 0.5 * (left - right) / denom
    if not np.isfinite(offset) or abs(offset) > 1.0:
        return 0.0
    return float(offset)


def ncc_register_preprocessed(
    ref: np.ndarray,
    target: np.ndarray,
    *,
    search_radius: int,
) -> dict[str, float | bool | int]:
    if ref.shape != target.shape:
        raise ValueError(f"frame shapes differ: {ref.shape} vs {target.shape}")

    rows, cols = ref.shape
    radius = int(search_radius)
    if radius < 2 or 2 * radius >= rows or 2 * radius >= cols:
        raise ValueError(f"invalid search_radius={search_radius} for frame shape {ref.shape}")

    template = ref[radius:rows - radius, radius:cols - radius]
    corr = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
    _, peak, _, max_loc = cv2.minMaxLoc(corr)
    peak_col, peak_row = max_loc

    sub_x = 0.0
    sub_y = 0.0
    fit_ok = False
    edge_peak = (
        peak_row == 0
        or peak_col == 0
        or peak_row == corr.shape[0] - 1
        or peak_col == corr.shape[1] - 1
    )
    if not edge_peak:
        center = float(corr[peak_row, peak_col])
        sub_x = parabolic_offset(float(corr[peak_row, peak_col - 1]), center, float(corr[peak_row, peak_col + 1]))
        sub_y = parabolic_offset(float(corr[peak_row - 1, peak_col]), center, float(corr[peak_row + 1, peak_col]))
        fit_ok = True

    dx = (peak_col + sub_x) - radius
    dy = (peak_row + sub_y) - radius

    return {
        "dx_px": float(dx),
        "dy_px": float(dy),
        "peak_ncc": float(peak),
        "integer_dx_px": int(peak_col - radius),
        "integer_dy_px": int(peak_row - radius),
        "fit_ok": bool(fit_ok),
        "edge_peak": bool(edge_peak),
    }


def close_short_false_gaps(mask: np.ndarray, max_gap: int) -> np.ndarray:
    closed = mask.copy()
    n = closed.size
    idx = 0
    while idx < n:
        if closed[idx]:
            idx += 1
            continue
        start = idx
        while idx < n and not closed[idx]:
            idx += 1
        end = idx
        left_true = start > 0 and closed[start - 1]
        right_true = end < n and closed[end]
        if left_true and right_true and end - start <= max_gap:
            closed[start:end] = True
    return closed


def longest_true_run(mask: np.ndarray) -> tuple[int, int]:
    best_start = 0
    best_end = 0
    idx = 0
    n = mask.size
    while idx < n:
        if not mask[idx]:
            idx += 1
            continue
        start = idx
        while idx < n and mask[idx]:
            idx += 1
        end = idx
        if end - start > best_end - best_start:
            best_start = start
            best_end = end
    return best_start, best_end


def detect_motion_segment(magnitudes: np.ndarray, threshold_px: float, max_gap: int = 2) -> tuple[int, int, float]:
    thresholds = [threshold_px, max(0.25, threshold_px * 0.5), 0.1]
    for threshold in thresholds:
        mask = close_short_false_gaps(magnitudes > threshold, max_gap=max_gap)
        start, end = longest_true_run(mask)
        if end - start >= 3:
            return start, end, threshold
    return 0, 0, thresholds[-1]


def wrap180(degrees: np.ndarray | float) -> np.ndarray | float:
    return (np.asarray(degrees) + 180.0) % 360.0 - 180.0


def circular_center_deg(angles_deg: np.ndarray) -> float:
    if angles_deg.size == 0:
        return float("nan")
    radians = np.radians(angles_deg)
    center = np.degrees(np.angle(np.mean(np.exp(1j * radians))))
    return float(wrap180(center))


def circular_median_deg(angles_deg: np.ndarray) -> float:
    if angles_deg.size == 0:
        return float("nan")
    center = circular_center_deg(angles_deg)
    shifted = np.asarray(wrap180(angles_deg - center), dtype=float)
    return float(wrap180(center + np.median(shifted)))


def circular_mad_deg(angles_deg: np.ndarray) -> float:
    if angles_deg.size == 0:
        return float("nan")
    center = circular_median_deg(angles_deg)
    deviations = np.abs(np.asarray(wrap180(angles_deg - center), dtype=float))
    return float(np.median(deviations))


def robust_mad(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.median(np.abs(values - np.median(values))))


def analyze_avi(
    spec: AviSpec,
    *,
    crop_xywh: tuple[int, int, int, int],
    duplicate_threshold: float,
    search_radius: int,
    motion_threshold: float,
    preprocess: str,
    highpass_sigma: float,
    max_frames: int | None,
) -> tuple[dict[str, float | int | str | None], pd.DataFrame, tuple[int, int]]:
    frames, meta = read_avi_frames(spec.path, crop_xywh, max_frames)
    unique_frames, unique_indices, raw_diffs = remove_duplicate_frames(frames, duplicate_threshold)

    if len(unique_frames) < 4:
        raise RuntimeError(f"{spec.path.name}: not enough unique frames ({len(unique_frames)})")

    processed = [preprocess_frame(frame, preprocess, highpass_sigma) for frame in unique_frames]

    rows: list[dict[str, float | int | str | bool | None]] = []
    for pair_idx in range(len(processed) - 1):
        reg = ncc_register_preprocessed(processed[pair_idx], processed[pair_idx + 1], search_radius=search_radius)
        dx = float(reg["dx_px"])
        dy = float(reg["dy_px"])
        mag = float(np.hypot(dx, dy))
        angle_row_down = float(wrap180(np.degrees(np.arctan2(dy, dx))))
        angle_y_up = float(wrap180(np.degrees(np.arctan2(-dy, dx))))
        rows.append({
            "avi": spec.path.name,
            "scan_axis": spec.scan_axis,
            "fixed_um": spec.fixed_um,
            "pair_index": pair_idx,
            "frame_a": unique_indices[pair_idx],
            "frame_b": unique_indices[pair_idx + 1],
            "raw_frame_gap": unique_indices[pair_idx + 1] - unique_indices[pair_idx],
            "dx_px": dx,
            "dy_px": dy,
            "magnitude_px": mag,
            "angle_row_down_deg": angle_row_down,
            "angle_y_up_deg": angle_y_up,
            **reg,
        })

    pair_df = pd.DataFrame(rows)
    magnitudes = pair_df["magnitude_px"].to_numpy(dtype=float)
    start, end, used_motion_threshold = detect_motion_segment(magnitudes, motion_threshold)
    pair_df["is_motion_segment"] = False
    if end > start:
        pair_df.loc[start:end - 1, "is_motion_segment"] = True

    moving = pair_df[pair_df["is_motion_segment"]].copy()
    if len(moving) < 3:
        raise RuntimeError(f"{spec.path.name}: no reliable moving segment detected")

    dx = moving["dx_px"].to_numpy(dtype=float)
    dy = moving["dy_px"].to_numpy(dtype=float)
    mag = moving["magnitude_px"].to_numpy(dtype=float)
    angles_row_down = moving["angle_row_down_deg"].to_numpy(dtype=float)
    median_vector = np.array([np.median(dx), np.median(dy)], dtype=float)
    median_norm = float(np.hypot(median_vector[0], median_vector[1]))
    if median_norm > 0:
        dot = (dx * median_vector[0] + dy * median_vector[1]) / median_norm
        reversal_fraction = float(np.mean(dot < 0))
    else:
        reversal_fraction = float("nan")

    cumulative_dx = np.cumsum(dx)
    cumulative_dy = np.cumsum(dy)
    path_length = float(np.sum(mag))
    net_length = float(np.hypot(cumulative_dx[-1], cumulative_dy[-1]))
    straightness = net_length / path_length if path_length > 0 else float("nan")

    mag_median = float(np.median(mag))
    mag_mad = robust_mad(mag)
    summary = {
        "avi": spec.path.name,
        "scan_axis": spec.scan_axis,
        "fixed_um": spec.fixed_um,
        "preprocess": preprocess,
        "total_frames": int(meta["read_frames"]),
        "declared_frames": int(meta["declared_frames"]),
        "unique_frames": len(unique_frames),
        "duplicate_rate": 1.0 - len(unique_frames) / len(frames),
        "raw_diff_median": float(np.median(raw_diffs[1:])) if len(raw_diffs) > 1 else float("nan"),
        "raw_diff_p90": float(np.quantile(raw_diffs[1:], 0.90)) if len(raw_diffs) > 1 else float("nan"),
        "motion_start_pair": int(start),
        "motion_end_pair": int(end),
        "motion_pairs": int(len(moving)),
        "motion_threshold_px": float(used_motion_threshold),
        "median_dx_px": float(np.median(dx)),
        "median_dy_px": float(np.median(dy)),
        "median_magnitude_px": mag_median,
        "magnitude_mad_px": mag_mad,
        "magnitude_robust_cv": float(1.4826 * mag_mad / mag_median) if mag_median > 0 else float("nan"),
        "median_angle_row_down_deg": circular_median_deg(angles_row_down),
        "angle_row_down_mad_deg": circular_mad_deg(angles_row_down),
        "median_angle_y_up_deg": circular_median_deg(moving["angle_y_up_deg"].to_numpy(dtype=float)),
        "median_peak_ncc": float(moving["peak_ncc"].median()),
        "edge_peak_fraction": float(moving["edge_peak"].mean()),
        "fit_ok_fraction": float(moving["fit_ok"].mean()),
        "reversal_fraction": reversal_fraction,
        "path_straightness": straightness,
        "net_motion_dx_px": float(cumulative_dx[-1]),
        "net_motion_dy_px": float(cumulative_dy[-1]),
        "net_motion_px": net_length,
        **meta,
    }
    return summary, pair_df, (start, end)


def plot_direction_summary(summary_df: pd.DataFrame, output_path: Path) -> None:
    ordered = summary_df.sort_values(["scan_axis", "fixed_um"], na_position="first").reset_index(drop=True)
    labels = ordered["avi"].tolist()
    colors = [
        METHOD_COLOR_LIST[0] if axis == "x" else METHOD_COLOR_LIST[1]
        for axis in ordered["scan_axis"].tolist()
    ]

    fig, axes = make_figure("double_col", nrows=1, ncols=2, height=4.8)
    ypos = np.arange(len(ordered))

    axes[0].barh(
        ypos,
        ordered["median_angle_row_down_deg"],
        xerr=ordered["angle_row_down_mad_deg"],
        color=colors,
        height=0.65,
    )
    axes[0].set_yticks(ypos)
    axes[0].set_yticklabels(labels, fontsize=7)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Median direction, row-down [deg]")
    axes[0].set_title("AVI Motion Direction")
    axes[0].axvline(0.0, color="#777777", lw=0.8, ls=":")

    axes[1].barh(
        ypos,
        ordered["median_magnitude_px"],
        xerr=ordered["magnitude_mad_px"],
        color=colors,
        height=0.65,
    )
    axes[1].set_yticks(ypos)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Median frame-to-frame magnitude [px]")
    axes[1].set_title("AVI Motion Speed")

    savefig_academic(fig, output_path)


def plot_cumulative_paths(pair_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = make_figure("double_col", nrows=1, ncols=2, height=3.8)
    for ax, axis in zip(axes, ["x", "y"], strict=True):
        subset = pair_df[(pair_df["scan_axis"] == axis) & (pair_df["is_motion_segment"])].copy()
        for idx, (avi, group) in enumerate(subset.groupby("avi", sort=False)):
            group = group.sort_values("pair_index")
            x = np.r_[0.0, np.cumsum(group["dx_px"].to_numpy(dtype=float))]
            y = np.r_[0.0, np.cumsum(group["dy_px"].to_numpy(dtype=float))]
            ax.plot(x, y, lw=1.0, alpha=0.75, label=avi if idx < 5 else None)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("Cumulative dx [px]")
        ax.set_ylabel("Cumulative dy [px]")
        ax.set_title(f"{axis.upper()}-scan cumulative paths")
        ax.legend(fontsize=6, ncols=1, loc="best")
    savefig_academic(fig, output_path)


def plot_timeseries(pair_df: pd.DataFrame, avi_name: str, output_path: Path) -> None:
    subset = pair_df[pair_df["avi"] == avi_name].sort_values("pair_index")
    if subset.empty:
        return

    motion = subset["is_motion_segment"].to_numpy(dtype=bool)
    motion_indices = np.where(motion)[0]
    start = int(motion_indices[0]) if motion_indices.size else 0
    end = int(motion_indices[-1] + 1) if motion_indices.size else 0

    fig, axes = make_figure("double_col", nrows=2, ncols=2, height=5.5)
    x = subset["pair_index"].to_numpy(dtype=int)
    series = [
        ("dx [px]", subset["dx_px"].to_numpy(dtype=float), METHOD_COLOR_LIST[0]),
        ("dy [px]", subset["dy_px"].to_numpy(dtype=float), METHOD_COLOR_LIST[1]),
        ("Magnitude [px]", subset["magnitude_px"].to_numpy(dtype=float), METHOD_COLOR_LIST[2]),
        ("Direction, row-down [deg]", subset["angle_row_down_deg"].to_numpy(dtype=float), METHOD_COLOR_LIST[3]),
    ]

    for ax, (ylabel, values, color) in zip(axes.ravel(), series, strict=True):
        ax.plot(x, values, lw=0.8, color=color)
        if end > start:
            ax.axvspan(start, end, color="#cccccc", alpha=0.3, label="motion segment")
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Unique-frame pair index")
        ax.set_title(f"{avi_name}: {ylabel}")
    axes[0, 0].legend(fontsize=7, loc="best")
    savefig_academic(fig, output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--crop", type=parse_crop, default=DEFAULT_CROP_XYWH, help="thermal crop as x,y,w,h")
    parser.add_argument("--duplicate-threshold", type=float, default=0.3)
    parser.add_argument("--search-radius", type=int, default=40)
    parser.add_argument("--motion-threshold", type=float, default=0.04)
    parser.add_argument("--preprocess", choices=["raw", "highpass", "gradient"], default="highpass")
    parser.add_argument("--highpass-sigma", type=float, default=12.0)
    parser.add_argument("--max-frames", type=int, default=None)
    parser.add_argument("--only", nargs="*", default=None, help="optional AVI filenames to analyze")
    parser.add_argument("--timeseries-avi", default="y0um.avi")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    setup_academic_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    all_paths = sorted(args.data_dir.glob("*.avi"), key=natural_avi_key)
    if args.only:
        requested = set(args.only)
        all_paths = [path for path in all_paths if path.name in requested]
        missing = requested - {path.name for path in all_paths}
        if missing:
            raise FileNotFoundError(f"requested AVI files not found: {sorted(missing)}")

    specs = [spec for path in all_paths if (spec := parse_avi_spec(path)) is not None]
    if not specs:
        raise RuntimeError(f"no x/y scan AVI files found in {args.data_dir}")

    print(f"Analyzing {len(specs)} AVI files with crop={args.crop}, preprocess={args.preprocess}")
    summaries: list[dict[str, float | int | str | None]] = []
    pair_tables: list[pd.DataFrame] = []

    for spec in specs:
        print(f"  {spec.path.name} ...", flush=True)
        summary, pairs, motion_range = analyze_avi(
            spec,
            crop_xywh=args.crop,
            duplicate_threshold=args.duplicate_threshold,
            search_radius=args.search_radius,
            motion_threshold=args.motion_threshold,
            preprocess=args.preprocess,
            highpass_sigma=args.highpass_sigma,
            max_frames=args.max_frames,
        )
        summaries.append(summary)
        pair_tables.append(pairs)
        print(
            "    unique={unique_frames}/{total_frames}, motion_pairs={motion_pairs}, "
            "angle={median_angle_row_down_deg:.2f} deg, mag={median_magnitude_px:.3f} px, "
            "straightness={path_straightness:.3f}".format(**summary),
            f"segment={motion_range}",
            flush=True,
        )

    summary_df = pd.DataFrame(summaries).sort_values(["scan_axis", "fixed_um"], na_position="first")
    pair_df = pd.concat(pair_tables, ignore_index=True)

    summary_path = args.output_dir / "avi_direction_summary.csv"
    pairs_path = args.output_dir / "avi_registration_pairs.csv"
    summary_df.to_csv(summary_path, index=False)
    pair_df.to_csv(pairs_path, index=False)
    print(f"Saved: {summary_path}")
    print(f"Saved: {pairs_path}")

    plot_direction_summary(summary_df, args.output_dir / "avi_direction_comparison.png")
    plot_cumulative_paths(pair_df, args.output_dir / "avi_cumulative_motion_paths.png")
    plot_timeseries(pair_df, args.timeseries_avi, args.output_dir / "avi_y0um_displacement_timeseries.png")
    print(f"Saved: {args.output_dir / 'avi_direction_comparison.png'}")
    print(f"Saved: {args.output_dir / 'avi_cumulative_motion_paths.png'}")
    print(f"Saved: {args.output_dir / 'avi_y0um_displacement_timeseries.png'}")

    display_cols = [
        "avi",
        "scan_axis",
        "motion_pairs",
        "median_angle_row_down_deg",
        "angle_row_down_mad_deg",
        "median_magnitude_px",
        "magnitude_robust_cv",
        "path_straightness",
        "edge_peak_fraction",
        "median_peak_ncc",
    ]
    print("\nSummary:")
    print(summary_df[display_cols].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
