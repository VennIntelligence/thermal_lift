#!/usr/bin/env python3
"""Check whether x-scan AVI files match TXT fixed-Y coordinate rows.

This is the X-axis counterpart of `avi_txt_yline_match_check.py`.

Hypotheses tested:

1. `xNum.avi -> TXT fixed Y=N`, then TXT frames vary along X.
2. `xNum.avi -> TXT fixed X=N`, then TXT frames vary along Y.

For `x.avi`, N is treated as 0 um.

Usage:
    uv run python scripts/avi_txt_xline_match_check.py
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "ep02_displacement_calibration"
DATA_DIR = PROJECT_ROOT / "data" / "data_raw" / "infrared_avi"
FRAME_AUDIT_PATH = PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv"

sys.path.insert(0, str(PROJECT_ROOT / "core" / "src"))
from thermal_core.plotting import METHOD_COLOR_LIST, make_figure, savefig_academic, setup_academic_style


def load_yline_module():
    path = PROJECT_ROOT / "scripts" / "avi_txt_yline_match_check.py"
    spec = importlib.util.spec_from_file_location("avi_txt_yline_match_check", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["avi_txt_yline_match_check"] = module
    spec.loader.exec_module(module)
    return module


def parse_x_avi_fixed_um(path: Path) -> float | None:
    stem = path.stem.lower()
    if stem == "x":
        return 0.0
    if stem.startswith("x") and stem.endswith("um"):
        try:
            return float(stem[1:-2])
        except ValueError:
            return None
    return None


def plot_axis_match(summary: pd.DataFrame, output_path: Path) -> None:
    fig, ax = make_figure("single_col", height=3.3)
    pivot = summary.pivot(index="fixed_um", columns="mapping", values="contour_axis_diff_to_avi_highpass_deg").sort_index()
    x = np.arange(len(pivot.index))
    width = 0.36
    ax.bar(x - width / 2, pivot.get("fixed_y", pd.Series(index=pivot.index, dtype=float)), width, label="TXT fixed Y=N", color=METHOD_COLOR_LIST[1])
    ax.bar(x + width / 2, pivot.get("fixed_x", pd.Series(index=pivot.index, dtype=float)), width, label="TXT fixed X=N", color=METHOD_COLOR_LIST[2])
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in pivot.index])
    ax.set_xlabel("N in xNum.avi [um]")
    ax.set_ylabel("Axis difference to AVI [deg]")
    ax.axhline(10, color="#444444", ls=":", lw=0.9)
    ax.legend(loc="best")
    savefig_academic(fig, output_path)


def plot_fixed_y_projection(paths: pd.DataFrame, output_path: Path) -> None:
    fig, ax = make_figure("single_col", height=3.5)
    subset = paths[paths["source"].eq("txt_fixed_y")].copy()
    for fixed_um, group in subset.groupby("fixed_um"):
        group = group.sort_values("coordinate_value")
        points = group[["centroid_x_px", "centroid_y_px"]].to_numpy(dtype=float)
        if len(points) < 2:
            continue
        net = points[-1] - points[0]
        norm = float(np.hypot(net[0], net[1]))
        if norm == 0:
            continue
        projection = (points - points[0]) @ (net / norm)
        ax.plot(group["coordinate_value"], projection, marker="o", ms=2.5, lw=0.9, label=f"Y={int(fixed_um)}")
    ax.set_xlabel("TXT X coordinate [um]")
    ax.set_ylabel("Contour projection along row [px]")
    ax.legend(fontsize=6, ncols=2, loc="best")
    savefig_academic(fig, output_path)


def main() -> None:
    setup_academic_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ymod = load_yline_module()

    frame_audit = pd.read_csv(FRAME_AUDIT_PATH)
    avi_summary = pd.read_csv(OUTPUT_DIR / "avi_direction_summary.csv")
    avi_pairs = pd.read_csv(OUTPUT_DIR / "avi_registration_pairs.csv")
    gradient_summary = pd.read_csv(OUTPUT_DIR / "avi_gradient_check" / "avi_direction_summary.csv")

    x_avi_paths = sorted(
        [path for path in DATA_DIR.glob("x*.avi") if parse_x_avi_fixed_um(path) is not None],
        key=lambda path: parse_x_avi_fixed_um(path) or 0.0,
    )

    summary_rows = []
    pair_tables = []
    contour_path_rows = []

    for avi_path in x_avi_paths:
        fixed_um = parse_x_avi_fixed_um(avi_path)
        if fixed_um is None:
            continue
        avi_name = avi_path.name
        avi_row = avi_summary[avi_summary["avi"].eq(avi_name)].iloc[0]
        grad_row = gradient_summary[gradient_summary["avi"].eq(avi_name)].iloc[0]
        avi_angle = float(avi_row["median_angle_row_down_deg"])
        grad_angle = float(grad_row["median_angle_row_down_deg"])

        moving_pairs = avi_pairs[avi_pairs["avi"].eq(avi_name) & avi_pairs["is_motion_segment"].astype(bool)].copy()
        avi_contours = ymod.avi_contour_path(avi_path, moving_pairs)
        avi_valid = avi_contours[avi_contours["has_contour"].astype(bool)].copy()
        avi_points = avi_valid[["centroid_x_px", "centroid_y_px"]].to_numpy(dtype=float)
        avi_net = avi_points[-1] - avi_points[0]
        avi_contour_angle = ymod.vector_angle(float(avi_net[0]), float(avi_net[1]))
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

        for mapping, coordinate_col in (("fixed_y", "X"), ("fixed_x", "Y")):
            records = ymod.line_records(frame_audit, mapping=mapping, fixed_um=fixed_um, main_session_only=True)
            contours = ymod.compute_txt_line_contours(records)
            contour_summary = ymod.summarize_contour_path(contours, coordinate_col)

            valid = contours[contours["has_contour"].astype(bool)].copy()
            if len(valid):
                contour_path_rows.append(pd.DataFrame({
                    "source": "txt_fixed_y" if mapping == "fixed_y" else "txt_fixed_x",
                    "label": f"{mapping}={int(fixed_um)}",
                    "fixed_um": fixed_um,
                    "mapping": mapping,
                    "sequence_index": np.arange(len(valid)),
                    "coordinate_value": valid[coordinate_col].to_numpy(dtype=float),
                    "centroid_x_px": valid["centroid_x_px"].to_numpy(dtype=float),
                    "centroid_y_px": valid["centroid_y_px"].to_numpy(dtype=float),
                }))

            pair_df = ymod.compute_pair_registrations(
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
                "avi_highpass_angle_deg": avi_angle,
                "avi_gradient_angle_deg": grad_angle,
                "avi_contour_angle_deg": avi_contour_angle,
                "avi_contour_motion_px": avi_contour_motion,
                "contour_axis_diff_to_avi_highpass_deg": ymod.axis_angle_diff_deg(contour_angle, avi_angle),
                "contour_axis_diff_to_avi_contour_deg": ymod.axis_angle_diff_deg(contour_angle, avi_contour_angle),
                **contour_summary,
                **pair_summary,
            })

    summary = pd.DataFrame(summary_rows)
    pairs = pd.concat(pair_tables, ignore_index=True)
    paths = pd.concat(contour_path_rows, ignore_index=True)

    summary_path = OUTPUT_DIR / "avi_txt_xline_match_summary.csv"
    pair_path = OUTPUT_DIR / "avi_txt_xline_pair_measurements.csv"
    path_path = OUTPUT_DIR / "avi_txt_xline_contour_paths.csv"
    summary.to_csv(summary_path, index=False)
    pairs.to_csv(pair_path, index=False)
    paths.to_csv(path_path, index=False)

    plot_axis_match(summary, OUTPUT_DIR / "avi_txt_xline_axis_match.png")
    plot_fixed_y_projection(paths, OUTPUT_DIR / "avi_txt_xline_projection_monotonicity.png")

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
    print("Saved:", OUTPUT_DIR / "avi_txt_xline_axis_match.png")
    print("Saved:", OUTPUT_DIR / "avi_txt_xline_projection_monotonicity.png")
    print()
    print(summary[display_cols].to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
