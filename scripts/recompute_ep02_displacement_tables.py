"""Recompute EP02 TXT displacement diagnostic tables from raw frames.

The X-step scale fit is kept as a visibility-response diagnostic only. It must
not be interpreted as a replacement detector pitch or as alignment truth. BMP
millimeter axes remain the detector-pitch reference, and stage coordinates stay
as reconstruction priors.

用法（项目根目录，无 CLI 参数）::

    uv run python scripts/recompute_ep02_displacement_tables.py

输入依赖: data/data_raw/infrared_avi/ 原始 TXT 帧、
    output/ep01_data_processing/frame_audit.csv、configs/stage_calibration.json
输出: output/ep02_displacement_calibration/ 下的诊断 CSV 表
    （time_adjacent_* / y_coordinate_* / coordinate_pair_time_gap_audit.csv 等）

关联: EP02
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from thermal_core.displacement import (
    bootstrap_theta_ci,
    build_frame_pairs,
    build_time_adjacent_pairs,
    coordinate_to_shift,
    fit_rotation_angle,
    measure_frame_pairs,
)
from thermal_core.ep02 import clean_sr_input


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "data_raw" / "infrared_avi"
EP01_OUTPUT_DIR = PROJECT_ROOT / "output" / "ep01_data_processing"
OUTPUT_DIR = PROJECT_ROOT / "output" / "ep02_displacement_calibration"


METHODS = [
    ("raw_ncc", "ncc", "raw"),
    ("highpass_ncc", "ncc", "highpass"),
    ("gradient_ncc", "ncc", "gradient"),
]

# Planned within-row X steps for small-step smoke tests (exclude path anomalies such as 6 um).
PLANNED_X_STEP_UM = {2, 4}


def add_order_gap(pairs: pd.DataFrame, audit_df: pd.DataFrame) -> pd.DataFrame:
    order_by_file = audit_df.set_index("file")["acquisition_order"].to_dict()
    out = pairs.copy()
    out["order_a"] = out["file_a"].map(order_by_file).astype(int)
    out["order_b"] = out["file_b"].map(order_by_file).astype(int)
    out["order_gap"] = (out["order_b"] - out["order_a"]).abs().astype(int)
    return out


def annotate_reference(
    df: pd.DataFrame,
    *,
    theta_deg: float,
    pixel_size_um: float,
) -> pd.DataFrame:
    out = df.copy()
    ref_dx, ref_dy = coordinate_to_shift(
        out["delta_X_um"].to_numpy(float),
        out["delta_Y_um"].to_numpy(float),
        theta_deg=theta_deg,
        pixel_size_um=pixel_size_um,
    )
    out["dy_y_up_px"] = -out["dy_px"].to_numpy(float)
    out["ref_dx_px"] = ref_dx
    out["ref_dy_px"] = ref_dy
    ref_vec = np.column_stack([out["ref_dx_px"].to_numpy(float), out["ref_dy_px"].to_numpy(float)])
    meas_vec = np.column_stack([out["dx_px"].to_numpy(float), out["dy_y_up_px"].to_numpy(float)])
    ref_mag = np.linalg.norm(ref_vec, axis=1)
    meas_mag = np.linalg.norm(meas_vec, axis=1)
    unit = np.divide(ref_vec, ref_mag[:, None], out=np.zeros_like(ref_vec), where=ref_mag[:, None] > 0)
    normal = np.column_stack([-unit[:, 1], unit[:, 0]])

    out["ref_mag_px"] = ref_mag
    out["measured_mag_px"] = meas_mag
    out["parallel_px"] = np.sum(meas_vec * unit, axis=1)
    out["perpendicular_px"] = np.sum(meas_vec * normal, axis=1)
    out["projection_ratio"] = np.divide(out["parallel_px"], ref_mag, out=np.full_like(ref_mag, np.nan), where=ref_mag > 0)
    out["ref_residual_px"] = np.linalg.norm(meas_vec - ref_vec, axis=1)
    if "peak_ncc" in out.columns:
        out["peak_score"] = out["peak_ncc"]
    elif "peak_phase" in out.columns:
        out["peak_score"] = out["peak_phase"]
    out["angle_deg_y_up"] = np.degrees(np.arctan2(out["dy_y_up_px"], out["dx_px"])) % 360.0
    return out


def summarize_time_adjacent(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method_label, move_type), group in df.groupby(["method_label", "move_type"], sort=True):
        rows.append(
            {
                "method_label": method_label,
                "move_type": move_type,
                "n_pairs": float(len(group)),
                "median_ref_mag_px": float(group["ref_mag_px"].median()),
                "median_measured_mag_px": float(group["measured_mag_px"].median()),
                "median_projection_ratio": float(group["projection_ratio"].median()),
                "median_abs_perpendicular_px": float(group["perpendicular_px"].abs().median()),
                "rms_ref_residual_px": float(np.sqrt(np.mean(np.square(group["ref_residual_px"])))),
                "median_peak_score": float(group["peak_score"].median()),
            }
        )
    return pd.DataFrame(rows)


def summarize_y_coordinate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (method_label, delta_um), group in df.groupby(["method_label", "delta_um"], sort=True):
        rows.append(
            {
                "method_label": method_label,
                "delta_um": float(delta_um),
                "n_pairs": float(len(group)),
                "median_ref_mag_px": float(group["ref_mag_px"].median()),
                "median_parallel_px": float(group["parallel_px"].median()),
                "median_projection_ratio": float(group["projection_ratio"].median()),
                "median_measured_mag_px": float(group["measured_mag_px"].median()),
                "median_abs_perpendicular_px": float(group["perpendicular_px"].abs().median()),
                "rms_ref_residual_px": float(np.sqrt(np.mean(np.square(group["ref_residual_px"])))),
                "median_angle_deg_y_up": float(group["angle_deg_y_up"].median()),
                "median_peak_ncc": float(group["peak_ncc"].median()),
            }
        )
    return pd.DataFrame(rows)


def measure_methods(pairs: pd.DataFrame, *, include_phase: bool) -> pd.DataFrame:
    frames = []
    for method_label, method, preprocess in METHODS:
        measured = measure_frame_pairs(
            pairs,
            DATA_DIR,
            roi_size=320,
            search_radius=5,
            method=method,
            preprocess=preprocess,
        )
        measured["method_label"] = method_label
        frames.append(measured)

    if include_phase:
        phase = measure_frame_pairs(
            pairs,
            DATA_DIR,
            roi_size=320,
            search_radius=5,
            method="phase",
            preprocess="raw",
        )
        phase["method_label"] = "phase_corr"
        frames.append(phase)

    return pd.concat(frames, ignore_index=True)


def fit_x_steps(time_adjacent: pd.DataFrame, *, pixel_size_um: float) -> pd.DataFrame:
    rows = []
    for method_label, group in time_adjacent[time_adjacent["move_type"].eq("x_step")].groupby("method_label", sort=True):
        valid = group.copy()
        if "fit_ok" in valid.columns:
            valid = valid[(valid["fit_ok"].fillna(True)) & (~valid["edge_peak"].fillna(False))]
        if method_label == "phase_corr" or valid.empty or np.allclose(valid["dx_px"], 0.0):
            rows.append(
                {
                    "method_label": method_label,
                    "theta_from_x_steps_deg": np.nan,
                    "x_step_rms_px": np.nan,
                    "x_step_apparent_um_per_visible_px": np.nan,
                    "scale_interpretation": "not_fit",
                    "n_pairs": int(len(group)),
                    "fit_ok": False,
                }
            )
            continue
        fit = fit_rotation_angle(
            valid["dx_px"].to_numpy(float),
            valid["dy_y_up_px"].to_numpy(float),
            valid["delta_X_um"].to_numpy(float),
            valid["delta_Y_um"].to_numpy(float),
            pixel_size_um=pixel_size_um,
        )
        rows.append(
            {
                "method_label": method_label,
                "theta_from_x_steps_deg": float(fit["theta_deg"]),
                "x_step_rms_px": float(fit["rms_error_px"]),
                "x_step_apparent_um_per_visible_px": float(fit["effective_pixel_size_um"]),
                "scale_interpretation": "diagnostic_visible_response_not_detector_pitch",
                "n_pairs": int(len(valid)),
                "fit_ok": True,
            }
        )
    return pd.DataFrame(rows)


def write_time_gap_audit(audit_df: pd.DataFrame) -> None:
    x_pairs = add_order_gap(build_frame_pairs(audit_df, axis="x", r_value=0, session_col="session", max_delta_um=4), audit_df)
    y_pairs = add_order_gap(build_frame_pairs(audit_df, axis="y", r_value=0, session_col="session", max_delta_um=4), audit_df)
    gap = pd.concat([x_pairs, y_pairs], ignore_index=True, sort=False)
    gap.to_csv(OUTPUT_DIR / "coordinate_pair_time_gap_audit.csv", index=False)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    audit_df = pd.read_csv(EP01_OUTPUT_DIR / "frame_audit.csv")
    with open(PROJECT_ROOT / "configs" / "stage_calibration.json", encoding="utf-8") as f:
        stage_config = json.load(f)
    theta_deg = float(stage_config["theta_deg"])
    pixel_size_um = float(stage_config["pixel_size_um"])

    main_df = clean_sr_input(audit_df)
    if main_df.empty:
        raise RuntimeError("No EP02 clean input frames found; expected is_sr_usable == True.")

    time_pairs = build_time_adjacent_pairs(
        main_df,
        r_value=0,
        session_col="session",
        max_order_gap=1,
    )
    time_pairs = time_pairs[
        (time_pairs["move_type"] != "x_step") | time_pairs["delta_X_um"].isin(PLANNED_X_STEP_UM)
    ].copy()
    time_pairs.to_csv(OUTPUT_DIR / "time_adjacent_registration_pairs.csv", index=False)
    time_pairs.to_csv(OUTPUT_DIR / "time_adjacent_pairs_r0.csv", index=False)

    y_pairs = build_frame_pairs(
        main_df,
        axis="y",
        r_value=0,
        session_col="session",
        max_delta_um=4,
    )
    y_pairs.to_csv(OUTPUT_DIR / "y_coordinate_pairs.csv", index=False)
    write_time_gap_audit(main_df)

    time_measured = annotate_reference(
        measure_methods(time_pairs, include_phase=True),
        theta_deg=theta_deg,
        pixel_size_um=pixel_size_um,
    )
    time_measured.to_csv(OUTPUT_DIR / "time_adjacent_method_measurements.csv", index=False)
    summarize_time_adjacent(time_measured).to_csv(OUTPUT_DIR / "time_adjacent_method_summary.csv", index=False)
    fit_x_steps(time_measured, pixel_size_um=pixel_size_um).to_csv(OUTPUT_DIR / "time_adjacent_x_step_fit.csv", index=False)

    y_measured = annotate_reference(
        measure_methods(y_pairs, include_phase=False),
        theta_deg=theta_deg,
        pixel_size_um=pixel_size_um,
    )
    y_measured.to_csv(OUTPUT_DIR / "y_coordinate_method_measurements.csv", index=False)
    y_summary = summarize_y_coordinate(y_measured)
    y_summary.to_csv(OUTPUT_DIR / "y_coordinate_method_summary.csv", index=False)
    y_summary.assign(
        monotonic_ok=lambda df: df.groupby("method_label")["median_parallel_px"].transform(lambda values: values.iloc[-1] > values.iloc[0])
    ).to_csv(OUTPUT_DIR / "y_coordinate_monotonic_summary.csv", index=False)

    print(f"Recomputed EP02 displacement tables with pixel_size_um={pixel_size_um:.1f}")
    print(f"Clean EP02 input frames: {len(main_df)}")


if __name__ == "__main__":
    main()
