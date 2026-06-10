#!/usr/bin/env python3
"""Check whether y-scan AVI files match the corresponding TXT coordinate lines.

The goal is diagnostic, not SR reconstruction:

1. Test whether `yNum.avi` is consistent with TXT fixed-X=N lines or fixed-Y=N
   rows.
2. Use contour centroids and high-pass/gradient NCC directions to distinguish
   coordinate naming/order problems from thermal-evolution bias.
3. Treat X/Y stage orthogonality as an instrument guarantee; compare axis
   directions modulo 180 degrees because the AVI motion may be recorded in the
   opposite command direction.

Usage:
    uv run python scripts/avi_txt_yline_match_check.py
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "data_raw" / "infrared_avi"
OUTPUT_DIR = PROJECT_ROOT / "output" / "ep02_displacement_calibration"
FRAME_AUDIT_PATH = PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv"

sys.path.insert(0, str(PROJECT_ROOT / "core" / "src"))
from thermal_core.io import load_frame
from thermal_core.plotting import METHOD_COLOR_LIST, make_figure, savefig_academic, setup_academic_style


THERMAL_CROP_XYWH = (50, 45, 640, 480)
CONTOUR_ROI_XYWH = (120, 60, 340, 300)


def wrap180(angle_deg: float | np.ndarray) -> float | np.ndarray:
    return (np.asarray(angle_deg) + 180.0) % 360.0 - 180.0


def axis_angle_diff_deg(a_deg: float, b_deg: float) -> float:
    """Smallest direction difference when sign is allowed to flip."""
    signed = abs(float(wrap180(a_deg - b_deg)))
    return min(signed, abs(180.0 - signed))


def vector_angle(dx: float, dy: float) -> float:
    return float(wrap180(np.degrees(np.arctan2(dy, dx))))


def zscore(frame: np.ndarray) -> np.ndarray:
    data = frame.astype(np.float32, copy=False)
    centered = data - float(np.mean(data))
    scale = float(np.std(centered))
    if scale < 1e-8:
        return centered
    return centered / scale


def preprocess_frame(frame: np.ndarray, mode: str, highpass_sigma: float = 12.0) -> np.ndarray:
    if mode == "raw":
        return zscore(frame)
    if mode == "highpass":
        baseline = cv2.GaussianBlur(frame.astype(np.float32), (0, 0), sigmaX=highpass_sigma, sigmaY=highpass_sigma)
        return zscore(frame - baseline)
    if mode == "gradient":
        gx = cv2.Sobel(frame.astype(np.float32), cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(frame.astype(np.float32), cv2.CV_32F, 0, 1, ksize=3)
        return zscore(cv2.magnitude(gx, gy))
    raise ValueError(f"unknown preprocess mode: {mode}")


def parabolic_offset(left: float, center: float, right: float) -> float:
    denom = left - 2.0 * center + right
    if abs(denom) < 1e-12:
        return 0.0
    offset = 0.5 * (left - right) / denom
    if not np.isfinite(offset) or abs(offset) > 1.0:
        return 0.0
    return float(offset)


def ncc_register(ref: np.ndarray, target: np.ndarray, *, search_radius: int = 8) -> dict[str, float | bool | int]:
    rows, cols = ref.shape
    radius = int(search_radius)
    template = ref[radius:rows - radius, radius:cols - radius]
    corr = cv2.matchTemplate(target, template, cv2.TM_CCOEFF_NORMED)
    _, peak, _, max_loc = cv2.minMaxLoc(corr)
    peak_col, peak_row = max_loc
    edge_peak = (
        peak_row == 0
        or peak_col == 0
        or peak_row == corr.shape[0] - 1
        or peak_col == corr.shape[1] - 1
    )
    sub_x = 0.0
    sub_y = 0.0
    fit_ok = False
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
        "edge_peak": bool(edge_peak),
        "fit_ok": bool(fit_ok),
    }


def parse_y_avi_fixed_um(path: Path) -> float | None:
    match = re.fullmatch(r"y(\d+)um", path.stem.lower())
    if not match:
        return None
    return float(match.group(1))


def read_avi_frames_at(path: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
    wanted = set(int(i) for i in frame_indices)
    if not wanted:
        return {}

    crop_x, crop_y, crop_w, crop_h = THERMAL_CROP_XYWH
    frames: dict[int, np.ndarray] = {}
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise OSError(f"cannot open AVI: {path}")

    idx = 0
    while len(frames) < len(wanted):
        ok, frame = cap.read()
        if not ok:
            break
        if idx in wanted:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
            frames[idx] = gray[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        idx += 1
    cap.release()
    return frames


def contour_centroid(frame: np.ndarray, roi_xywh: tuple[int, int, int, int] = CONTOUR_ROI_XYWH) -> dict[str, float | str] | None:
    """Return a robust central contour centroid from an Otsu foreground mask.

    The TXT and AVI renderings have opposite foreground polarity, so both
    above-threshold and below-threshold masks are tested. The mask with a
    plausible central foreground fraction nearest 0.10 is selected.
    """
    x, y, width, height = roi_xywh
    crop = frame[y:y + height, x:x + width].astype(np.float32)
    norm = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX).astype("uint8")
    threshold, _ = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    candidates: list[tuple[float, dict[str, float | str]]] = []
    for polarity, mask in (
        ("above", norm >= threshold),
        ("below", norm < threshold),
    ):
        fraction = float(np.mean(mask))
        if not (0.03 <= fraction <= 0.35):
            continue
        yy, xx = np.nonzero(mask)
        if xx.size < 500:
            continue
        score = abs(fraction - 0.10)
        candidates.append((
            score,
            {
                "centroid_x_px": float(x + np.mean(xx)),
                "centroid_y_px": float(y + np.mean(yy)),
                "foreground_fraction": fraction,
                "foreground_polarity": polarity,
                "otsu_threshold": float(threshold),
            },
        ))

    if not candidates:
        return None
    return sorted(candidates, key=lambda item: item[0])[0][1]


def line_records(
    frame_audit: pd.DataFrame,
    *,
    mapping: str,
    fixed_um: float,
    main_session_only: bool,
) -> pd.DataFrame:
    if mapping == "fixed_x":
        subset = frame_audit[(frame_audit["R"].eq(0)) & (frame_audit["X"].eq(fixed_um))].copy()
        coordinate = "Y"
    elif mapping == "fixed_y":
        subset = frame_audit[(frame_audit["R"].eq(0)) & (frame_audit["Y"].eq(fixed_um))].copy()
        coordinate = "X"
    else:
        raise ValueError("mapping must be fixed_x or fixed_y")

    if main_session_only:
        subset = subset[subset["is_main_session"].astype(bool)].copy()
    return subset.sort_values(coordinate).reset_index(drop=True)


def compute_txt_line_contours(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int | bool]] = []
    for row in records.itertuples(index=False):
        frame = load_frame(DATA_DIR / row.file).astype(np.float32)
        centroid = contour_centroid(frame)
        if centroid is None:
            rows.append({
                "file": row.file,
                "X": int(row.X),
                "Y": int(row.Y),
                "acquisition_order": int(row.acquisition_order),
                "has_contour": False,
            })
            continue
        rows.append({
            "file": row.file,
            "X": int(row.X),
            "Y": int(row.Y),
            "acquisition_order": int(row.acquisition_order),
            "has_contour": True,
            **centroid,
        })
    return pd.DataFrame(rows)


def summarize_contour_path(contours: pd.DataFrame, coordinate_col: str) -> dict[str, float | int | bool | str]:
    valid = contours[contours["has_contour"].astype(bool)].copy()
    if len(valid) < 2:
        return {
            "contour_points": int(len(valid)),
            "contour_angle_deg": np.nan,
            "contour_net_motion_px": np.nan,
            "contour_monotonic_fraction": np.nan,
        }

    points = valid[["centroid_x_px", "centroid_y_px"]].to_numpy(dtype=float)
    deltas = np.diff(points, axis=0)
    net = points[-1] - points[0]
    net_norm = float(np.hypot(net[0], net[1]))
    angle = vector_angle(float(net[0]), float(net[1]))
    if net_norm > 0:
        unit = net / net_norm
        projections = deltas @ unit
        monotonic_fraction = float(np.mean(projections > 0))
    else:
        monotonic_fraction = np.nan

    order_values = valid["acquisition_order"].to_numpy(dtype=float)
    coordinate_values = valid[coordinate_col].to_numpy(dtype=float)
    order_gaps = np.diff(order_values)
    coordinate_gaps = np.diff(coordinate_values)

    return {
        "contour_points": int(len(valid)),
        "contour_angle_deg": angle,
        "contour_net_motion_px": net_norm,
        "contour_monotonic_fraction": monotonic_fraction,
        "acquisition_order_monotonic": bool(np.all(order_gaps > 0)),
        "acquisition_gap_median": float(np.median(order_gaps)) if order_gaps.size else np.nan,
        "acquisition_gap_min": float(np.min(order_gaps)) if order_gaps.size else np.nan,
        "acquisition_gap_max": float(np.max(order_gaps)) if order_gaps.size else np.nan,
        "coordinate_gap_min": float(np.min(coordinate_gaps)) if coordinate_gaps.size else np.nan,
        "coordinate_gap_max": float(np.max(coordinate_gaps)) if coordinate_gaps.size else np.nan,
        "first_file": str(valid.iloc[0]["file"]),
        "last_file": str(valid.iloc[-1]["file"]),
    }


def compute_pair_registrations(
    records: pd.DataFrame,
    *,
    mapping: str,
    coordinate_col: str,
    avi_name: str,
    avi_angle_highpass_deg: float,
    avi_angle_gradient_deg: float,
) -> pd.DataFrame:
    rows: list[dict[str, float | str | int | bool]] = []
    frame_cache: dict[str, np.ndarray] = {}

    for a, b in zip(records.itertuples(index=False), records.iloc[1:].itertuples(index=False)):
        frame_a = frame_cache.setdefault(a.file, load_frame(DATA_DIR / a.file).astype(np.float32))
        frame_b = frame_cache.setdefault(b.file, load_frame(DATA_DIR / b.file).astype(np.float32))
        for method, avi_angle in (
            ("highpass", avi_angle_highpass_deg),
            ("gradient", avi_angle_gradient_deg),
        ):
            reg = ncc_register(
                preprocess_frame(frame_a, method),
                preprocess_frame(frame_b, method),
                search_radius=8,
            )
            dx = float(reg["dx_px"])
            dy = float(reg["dy_px"])
            angle = vector_angle(dx, dy)
            rows.append({
                "avi": avi_name,
                "mapping": mapping,
                "coordinate_col": coordinate_col,
                "coord_a": float(getattr(a, coordinate_col)),
                "coord_b": float(getattr(b, coordinate_col)),
                "file_a": a.file,
                "file_b": b.file,
                "order_gap": int(b.acquisition_order - a.acquisition_order),
                "method": method,
                "dx_px": dx,
                "dy_px": dy,
                "magnitude_px": float(np.hypot(dx, dy)),
                "angle_deg": angle,
                "axis_diff_to_avi_deg": axis_angle_diff_deg(angle, avi_angle),
                **reg,
            })
    return pd.DataFrame(rows)


def avi_contour_path(avi_path: Path, motion_pairs: pd.DataFrame) -> pd.DataFrame:
    selected = [int(motion_pairs.iloc[0]["frame_a"])] + [int(v) for v in motion_pairs["frame_b"].tolist()]
    frames = read_avi_frames_at(avi_path, selected)
    rows: list[dict[str, float | int | bool | str]] = []
    for frame_index in selected:
        centroid = contour_centroid(frames[frame_index])
        rows.append({
            "avi": avi_path.name,
            "frame_index": int(frame_index),
            "has_contour": centroid is not None,
            **(centroid or {}),
        })
    return pd.DataFrame(rows)


def plot_axis_match(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = make_figure("single_col", height=3.3)
    pivot = summary.pivot(index="fixed_um", columns="mapping", values="contour_axis_diff_to_avi_highpass_deg").sort_index()
    x = np.arange(len(pivot.index))
    width = 0.36
    ax.bar(x - width / 2, pivot.get("fixed_x", pd.Series(index=pivot.index, dtype=float)), width, label="TXT fixed X=N", color=METHOD_COLOR_LIST[1])
    ax.bar(x + width / 2, pivot.get("fixed_y", pd.Series(index=pivot.index, dtype=float)), width, label="TXT fixed Y=N", color=METHOD_COLOR_LIST[2])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in pivot.index])
    ax.set_xlabel("N in yNum.avi [um]")
    ax.set_ylabel("Axis difference to AVI [deg]")
    ax.axhline(10, color="#444444", ls=":", lw=0.9)
    ax.legend(loc="best")
    savefig_academic(fig, output_path)


def plot_txt_projection(summary_paths: pd.DataFrame, output_path: Path) -> None:
    fig, ax = make_figure("single_col", height=3.5)
    subset = summary_paths[summary_paths["mapping"].eq("fixed_x")].copy()
    for fixed_um, group in subset.groupby("fixed_um"):
        group = group.sort_values("coordinate_value")
        points = group[["centroid_x_px", "centroid_y_px"]].to_numpy(dtype=float)
        net = points[-1] - points[0]
        norm = float(np.hypot(net[0], net[1]))
        if norm == 0:
            continue
        projection = (points - points[0]) @ (net / norm)
        ax.plot(group["coordinate_value"], projection, marker="o", ms=2.5, lw=0.9, label=f"X={int(fixed_um)}")
    ax.set_xlabel("TXT Y coordinate [um]")
    ax.set_ylabel("Contour projection along line [px]")
    ax.legend(fontsize=6, ncols=2, loc="best")
    savefig_academic(fig, output_path)


def plot_contour_paths(path_df: pd.DataFrame, output_path: Path) -> None:
    fig, axes = make_figure("double_col", nrows=1, ncols=2, height=3.8)
    for ax, source in zip(axes, ["txt_fixed_x", "avi"], strict=True):
        subset = path_df[path_df["source"].eq(source)].copy()
        for label, group in subset.groupby("label", sort=False):
            group = group.sort_values("sequence_index")
            x = group["centroid_x_px"].to_numpy(dtype=float)
            y = group["centroid_y_px"].to_numpy(dtype=float)
            ax.plot(x - x[0], y - y[0], lw=1.0, alpha=0.8, label=label)
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("Relative contour x [px]")
        ax.set_ylabel("Relative contour y [px]")
        ax.set_title("TXT fixed-X paths" if source == "txt_fixed_x" else "AVI y-scan paths")
        ax.legend(fontsize=6, ncols=1, loc="best")
    savefig_academic(fig, output_path)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--frame-audit", type=Path, default=FRAME_AUDIT_PATH)
    parser.add_argument("--main-session-only", action=argparse.BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    setup_academic_style()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame_audit = pd.read_csv(args.frame_audit)
    avi_summary = pd.read_csv(args.output_dir / "avi_direction_summary.csv")
    avi_pairs = pd.read_csv(args.output_dir / "avi_registration_pairs.csv")
    gradient_summary = pd.read_csv(args.output_dir / "avi_gradient_check" / "avi_direction_summary.csv")

    y_avi_paths = sorted(
        [path for path in args.data_dir.glob("y*um.avi") if parse_y_avi_fixed_um(path) is not None],
        key=lambda path: parse_y_avi_fixed_um(path) or 0.0,
    )

    summary_rows: list[dict[str, float | str | int | bool]] = []
    pair_tables: list[pd.DataFrame] = []
    contour_path_rows: list[pd.DataFrame] = []

    for avi_path in y_avi_paths:
        fixed_um = parse_y_avi_fixed_um(avi_path)
        if fixed_um is None:
            continue
        avi_name = avi_path.name
        avi_row = avi_summary[avi_summary["avi"].eq(avi_name)].iloc[0]
        grad_row = gradient_summary[gradient_summary["avi"].eq(avi_name)].iloc[0]
        avi_angle = float(avi_row["median_angle_row_down_deg"])
        grad_angle = float(grad_row["median_angle_row_down_deg"])

        moving_pairs = avi_pairs[avi_pairs["avi"].eq(avi_name) & avi_pairs["is_motion_segment"].astype(bool)].copy()
        avi_contours = avi_contour_path(avi_path, moving_pairs)
        avi_valid = avi_contours[avi_contours["has_contour"].astype(bool)].copy()
        avi_points = avi_valid[["centroid_x_px", "centroid_y_px"]].to_numpy(dtype=float)
        avi_net = avi_points[-1] - avi_points[0]
        avi_contour_angle = vector_angle(float(avi_net[0]), float(avi_net[1]))
        avi_contour_motion = float(np.hypot(avi_net[0], avi_net[1]))
        contour_path_rows.append(pd.DataFrame({
            "source": "avi",
            "label": avi_name,
            "fixed_um": fixed_um,
            "mapping": "avi",
            "sequence_index": np.arange(len(avi_valid)),
            "coordinate_value": avi_valid["frame_index"].to_numpy(dtype=float),
            "centroid_x_px": avi_valid["centroid_x_px"].to_numpy(dtype=float),
            "centroid_y_px": avi_valid["centroid_y_px"].to_numpy(dtype=float),
        }))

        for mapping, coordinate_col in (("fixed_x", "Y"), ("fixed_y", "X")):
            records = line_records(frame_audit, mapping=mapping, fixed_um=fixed_um, main_session_only=args.main_session_only)
            contours = compute_txt_line_contours(records)
            contour_summary = summarize_contour_path(contours, coordinate_col)

            if len(contours) > 0:
                valid = contours[contours["has_contour"].astype(bool)].copy()
                contour_path_rows.append(pd.DataFrame({
                    "source": "txt_fixed_x" if mapping == "fixed_x" else "txt_fixed_y",
                    "label": f"{mapping}={int(fixed_um)}",
                    "fixed_um": fixed_um,
                    "mapping": mapping,
                    "sequence_index": np.arange(len(valid)),
                    "coordinate_value": valid[coordinate_col].to_numpy(dtype=float),
                    "centroid_x_px": valid["centroid_x_px"].to_numpy(dtype=float),
                    "centroid_y_px": valid["centroid_y_px"].to_numpy(dtype=float),
                }))

            pair_df = compute_pair_registrations(
                records,
                mapping=mapping,
                coordinate_col=coordinate_col,
                avi_name=avi_name,
                avi_angle_highpass_deg=avi_angle,
                avi_angle_gradient_deg=grad_angle,
            )
            pair_tables.append(pair_df)

            pair_summary = {}
            for method, group in pair_df.groupby("method"):
                pair_summary[f"{method}_pair_count"] = int(len(group))
                pair_summary[f"{method}_axis_diff_to_avi_median_deg"] = float(group["axis_diff_to_avi_deg"].median())
                pair_summary[f"{method}_magnitude_median_px"] = float(group["magnitude_px"].median())
                pair_summary[f"{method}_peak_median"] = float(group["peak_ncc"].median())
                pair_summary[f"{method}_edge_peak_fraction"] = float(group["edge_peak"].mean())

            contour_angle = float(contour_summary["contour_angle_deg"])
            summary_rows.append({
                "avi": avi_name,
                "fixed_um": fixed_um,
                "mapping": mapping,
                "coordinate_col": coordinate_col,
                "main_session_only": bool(args.main_session_only),
                "avi_highpass_angle_deg": avi_angle,
                "avi_gradient_angle_deg": grad_angle,
                "avi_contour_angle_deg": avi_contour_angle,
                "avi_contour_motion_px": avi_contour_motion,
                "contour_axis_diff_to_avi_highpass_deg": axis_angle_diff_deg(contour_angle, avi_angle),
                "contour_axis_diff_to_avi_contour_deg": axis_angle_diff_deg(contour_angle, avi_contour_angle),
                **contour_summary,
                **pair_summary,
            })

    summary = pd.DataFrame(summary_rows)
    pairs = pd.concat(pair_tables, ignore_index=True)
    paths = pd.concat(contour_path_rows, ignore_index=True)

    summary_path = args.output_dir / "avi_txt_yline_match_summary.csv"
    pair_path = args.output_dir / "avi_txt_yline_pair_measurements.csv"
    path_path = args.output_dir / "avi_txt_yline_contour_paths.csv"
    summary.to_csv(summary_path, index=False)
    pairs.to_csv(pair_path, index=False)
    paths.to_csv(path_path, index=False)

    plot_axis_match(summary, args.output_dir / "avi_txt_yline_axis_match.png")
    plot_txt_projection(paths, args.output_dir / "avi_txt_yline_projection_monotonicity.png")
    plot_contour_paths(paths[paths["source"].isin(["txt_fixed_x", "avi"])], args.output_dir / "avi_txt_yline_contour_paths.png")

    display_cols = [
        "avi",
        "mapping",
        "contour_points",
        "contour_axis_diff_to_avi_highpass_deg",
        "contour_monotonic_fraction",
        "acquisition_gap_median",
        "acquisition_gap_max",
        "highpass_axis_diff_to_avi_median_deg",
        "gradient_axis_diff_to_avi_median_deg",
    ]
    print("Saved:", summary_path)
    print("Saved:", pair_path)
    print("Saved:", path_path)
    print("Saved:", args.output_dir / "avi_txt_yline_axis_match.png")
    print("Saved:", args.output_dir / "avi_txt_yline_projection_monotonicity.png")
    print("Saved:", args.output_dir / "avi_txt_yline_contour_paths.png")
    print()
    print(summary[display_cols].to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
