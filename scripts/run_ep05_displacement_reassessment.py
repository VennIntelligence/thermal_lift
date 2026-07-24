"""Reassess TXT micro-scan displacement with global frame-to-frame registration.

This script intentionally separates several displacement definitions that were
mixed together in earlier episodes:

* acquisition-adjacent visible 2D shifts;
* full scanline endpoint shifts;
* cumulative frame-to-frame trajectory inside the main session;
* EP04 contour-normal phase coverage.

It does not use any single reference frame as a universal truth source.

用法（项目根目录）::

    uv run python scripts/run_ep05_displacement_reassessment.py \
        [--roi-size 320] [--n-jobs N] [--methods highpass gradient] [--skip-figures]

输入依赖: output/ep01_data_processing/frame_audit.csv、data/data_raw/infrared_avi/、
    configs/stage_calibration.json（--frame-audit-csv / --data-dir / --stage-config 可覆盖）
输出: output/ep05_sr_reassessment/（--output-dir 可覆盖）位移评估 CSV 与图表

关联: EP05
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from thermal_core.displacement import coordinate_to_shift, subpixel_ncc
from thermal_core.io import load_frame
from thermal_core.plotting import (
    COLORMAPS,
    FIGURE_SIZES,
    METHOD_COLOR_LIST,
    savefig_academic,
    setup_academic_style,
)


@dataclass(frozen=True)
class PairTask:
    pair_set: str
    pair_id: int
    file_a: str
    file_b: str
    acq_a: int
    acq_b: int
    session_a: int
    session_b: int
    x_a: float
    y_a: float
    r_a: int
    x_b: float
    y_b: float
    r_b: int
    method: str
    search_radius: int
    roi_size: int
    data_dir: str
    theta_deg: float
    pixel_size_um: float


def project_root() -> Path:
    root = Path.cwd()
    while not (root / "AGENTS.md").exists() and root != root.parent:
        root = root.parent
    return root


def default_workers() -> int:
    return max(1, min(os.cpu_count() or 1, 16))


def boolish(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def pair_class(row_a: pd.Series, row_b: pd.Series) -> str:
    dx = float(row_b["X"]) - float(row_a["X"])
    dy = float(row_b["Y"]) - float(row_a["Y"])
    same_r = int(row_a["R"]) == int(row_b["R"])
    if dx == 0.0 and dy == 0.0:
        return "same_coordinate"
    if same_r and dy == 0.0 and dx != 0.0:
        return "same_y_x_motion"
    if same_r and dx == 0.0 and dy != 0.0:
        return "same_x_y_motion"
    if same_r and dx < 0.0 and dy > 0.0:
        return "raster_row_reset"
    if not same_r:
        return "repeat_boundary"
    return "mixed_xy_motion"


def make_pair_record(pair_set: str, pair_id: int, row_a: pd.Series, row_b: pd.Series) -> dict:
    return {
        "pair_set": pair_set,
        "pair_id": int(pair_id),
        "file_a": str(row_a["file"]),
        "file_b": str(row_b["file"]),
        "acq_a": int(row_a["acquisition_order"]),
        "acq_b": int(row_b["acquisition_order"]),
        "session_a": int(row_a["session"]),
        "session_b": int(row_b["session"]),
        "cross_session": int(row_a["session"]) != int(row_b["session"]),
        "X_a": float(row_a["X"]),
        "Y_a": float(row_a["Y"]),
        "R_a": int(row_a["R"]),
        "X_b": float(row_b["X"]),
        "Y_b": float(row_b["Y"]),
        "R_b": int(row_b["R"]),
        "delta_X_um": float(row_b["X"]) - float(row_a["X"]),
        "delta_Y_um": float(row_b["Y"]) - float(row_a["Y"]),
        "pair_class": pair_class(row_a, row_b),
    }


def build_pair_table(audit: pd.DataFrame) -> pd.DataFrame:
    audit = audit.sort_values("acquisition_order").reset_index(drop=True)
    if "is_main_session" in audit:
        main = audit[boolish(audit["is_main_session"])].copy()
    else:
        main_session = audit.groupby("session")["file"].count().idxmax()
        main = audit[audit["session"].eq(main_session)].copy()
    main = main.sort_values("acquisition_order").reset_index(drop=True)

    records: list[dict] = []
    pair_id = 0

    for idx in range(len(audit) - 1):
        records.append(make_pair_record("all_acq_adjacent", pair_id, audit.iloc[idx], audit.iloc[idx + 1]))
        pair_id += 1

    for idx in range(len(main) - 1):
        records.append(make_pair_record("main_acq_adjacent", pair_id, main.iloc[idx], main.iloc[idx + 1]))
        pair_id += 1

    group_cols = ["session", "R", "Y"]
    for _, group in main.groupby(group_cols):
        group = group.sort_values("X")
        if len(group) < 2:
            continue
        first = group.iloc[0]
        last = group.iloc[-1]
        records.append(make_pair_record("same_y_scanline_endpoint", pair_id, first, last))
        pair_id += 1

    group_cols = ["session", "R", "X"]
    for _, group in main.groupby(group_cols):
        group = group.sort_values("Y")
        if len(group) < 2:
            continue
        first = group.iloc[0]
        last = group.iloc[-1]
        records.append(make_pair_record("same_x_column_endpoint", pair_id, first, last))
        pair_id += 1

    table = pd.DataFrame(records)
    return table.drop_duplicates(
        subset=["pair_set", "file_a", "file_b"],
        keep="first",
    ).reset_index(drop=True)


def task_from_pair(
    row: pd.Series,
    method: str,
    data_dir: Path,
    theta_deg: float,
    pixel_size_um: float,
    roi_size: int,
) -> PairTask:
    if str(row["pair_set"]).endswith("_endpoint"):
        search_radius = 12
    elif str(row["pair_set"]) == "all_acq_adjacent":
        search_radius = 10
    else:
        search_radius = 8
    return PairTask(
        pair_set=str(row["pair_set"]),
        pair_id=int(row["pair_id"]),
        file_a=str(row["file_a"]),
        file_b=str(row["file_b"]),
        acq_a=int(row["acq_a"]),
        acq_b=int(row["acq_b"]),
        session_a=int(row["session_a"]),
        session_b=int(row["session_b"]),
        x_a=float(row["X_a"]),
        y_a=float(row["Y_a"]),
        r_a=int(row["R_a"]),
        x_b=float(row["X_b"]),
        y_b=float(row["Y_b"]),
        r_b=int(row["R_b"]),
        method=method,
        search_radius=search_radius,
        roi_size=roi_size,
        data_dir=str(data_dir),
        theta_deg=theta_deg,
        pixel_size_um=pixel_size_um,
    )


def run_registration_task(task: PairTask) -> dict:
    frame_a = load_frame(Path(task.data_dir) / task.file_a).astype(np.float32, copy=False)
    frame_b = load_frame(Path(task.data_dir) / task.file_b).astype(np.float32, copy=False)
    estimate = subpixel_ncc(
        frame_a,
        frame_b,
        search_radius=task.search_radius,
        fit_radius=2,
        roi_size=task.roi_size,
        preprocess=task.method,
        highpass_sigma=6.0,
    )

    dx_a, dy_a = coordinate_to_shift(
        task.x_a,
        task.y_a,
        theta_deg=task.theta_deg,
        pixel_size_um=task.pixel_size_um,
    )
    dx_b, dy_b = coordinate_to_shift(
        task.x_b,
        task.y_b,
        theta_deg=task.theta_deg,
        pixel_size_um=task.pixel_size_um,
    )
    model_dx = float(dx_b - dx_a)
    model_dy_math = float(dy_b - dy_a)
    model_dy_image = -model_dy_math
    model_norm = float(np.hypot(model_dx, model_dy_image))

    measured_dx = float(estimate["dx_px"])
    measured_dy = float(estimate["dy_px"])
    measured_norm = float(np.hypot(measured_dx, measured_dy))
    if model_norm > 0:
        model_projection = (measured_dx * model_dx + measured_dy * model_dy_image) / model_norm
        model_projection_abs = abs(model_projection)
        norm_ratio = measured_norm / model_norm
        projection_ratio_abs = model_projection_abs / model_norm
    else:
        model_projection = np.nan
        model_projection_abs = np.nan
        norm_ratio = np.nan
        projection_ratio_abs = np.nan

    return {
        "pair_set": task.pair_set,
        "pair_id": task.pair_id,
        "method": task.method,
        "file_a": task.file_a,
        "file_b": task.file_b,
        "acq_a": task.acq_a,
        "acq_b": task.acq_b,
        "session_a": task.session_a,
        "session_b": task.session_b,
        "cross_session": task.session_a != task.session_b,
        "X_a": task.x_a,
        "Y_a": task.y_a,
        "R_a": task.r_a,
        "X_b": task.x_b,
        "Y_b": task.y_b,
        "R_b": task.r_b,
        "delta_X_um": task.x_b - task.x_a,
        "delta_Y_um": task.y_b - task.y_a,
        "model_dx_px": model_dx,
        "model_dy_math_px": model_dy_math,
        "model_dy_image_px": model_dy_image,
        "model_norm_px": model_norm,
        "measured_dx_px": measured_dx,
        "measured_dy_px": measured_dy,
        "measured_norm_px": measured_norm,
        "measured_projection_on_model_px": model_projection,
        "measured_abs_projection_on_model_px": model_projection_abs,
        "measured_norm_over_model_norm": norm_ratio,
        "measured_abs_projection_over_model_norm": projection_ratio_abs,
        "peak_ncc": float(estimate["peak_ncc"]),
        "integer_dx_px": int(estimate["integer_dx_px"]),
        "integer_dy_px": int(estimate["integer_dy_px"]),
        "fit_ok": bool(estimate["fit_ok"]),
        "edge_peak": bool(estimate["edge_peak"]),
        "search_radius": task.search_radius,
        "roi_size": task.roi_size,
    }


def quantile_dict(values: pd.Series) -> dict[str, float]:
    clean = pd.to_numeric(values, errors="coerce").dropna()
    if clean.empty:
        return {"count": 0}
    q = clean.quantile([0.0, 0.1, 0.5, 0.9, 1.0])
    return {
        "count": int(clean.size),
        "min": float(q.loc[0.0]),
        "p10": float(q.loc[0.1]),
        "median": float(q.loc[0.5]),
        "p90": float(q.loc[0.9]),
        "max": float(q.loc[1.0]),
    }


def summarize_measurements(measurements: pd.DataFrame) -> pd.DataFrame:
    rows = []
    group_cols = ["pair_set", "method", "pair_class"]
    for key, group in measurements.groupby(group_cols, dropna=False):
        pair_set, method, cls = key
        model_nonzero = group[pd.to_numeric(group["model_norm_px"], errors="coerce") > 1e-9]
        rows.append({
            "pair_set": pair_set,
            "method": method,
            "pair_class": cls,
            "n": int(len(group)),
            "measured_norm_median_px": float(group["measured_norm_px"].median()),
            "measured_norm_p90_px": float(group["measured_norm_px"].quantile(0.9)),
            "measured_norm_max_px": float(group["measured_norm_px"].max()),
            "model_norm_median_px": float(group["model_norm_px"].median()),
            "abs_projection_median_px": float(group["measured_abs_projection_on_model_px"].median()),
            "norm_ratio_median": float(model_nonzero["measured_norm_over_model_norm"].median()) if not model_nonzero.empty else np.nan,
            "abs_projection_ratio_median": float(model_nonzero["measured_abs_projection_over_model_norm"].median()) if not model_nonzero.empty else np.nan,
            "peak_ncc_median": float(group["peak_ncc"].median()),
            "edge_peak_count": int(group["edge_peak"].sum()),
        })
    return pd.DataFrame(rows).sort_values(["pair_set", "method", "pair_class"]).reset_index(drop=True)


def build_cumulative_trajectory(measurements: pd.DataFrame, method: str) -> pd.DataFrame:
    data = measurements[
        measurements["pair_set"].eq("main_acq_adjacent")
        & measurements["method"].eq(method)
        & ~measurements["edge_peak"].astype(bool)
    ].copy()
    data = data.sort_values(["acq_a", "acq_b"]).reset_index(drop=True)
    if data.empty:
        return pd.DataFrame()

    start = data.iloc[0]
    rows = [{
        "acquisition_order": int(start["acq_a"]),
        "file": str(start["file_a"]),
        "X": float(start["X_a"]),
        "Y": float(start["Y_a"]),
        "R": int(start["R_a"]),
        "cum_dx_px": 0.0,
        "cum_dy_px": 0.0,
        "step_norm_px": 0.0,
        "pair_class": "start",
    }]
    cum_dx = 0.0
    cum_dy = 0.0
    for _, row in data.iterrows():
        cum_dx += float(row["measured_dx_px"])
        cum_dy += float(row["measured_dy_px"])
        rows.append({
            "acquisition_order": int(row["acq_b"]),
            "file": str(row["file_b"]),
            "X": float(row["X_b"]),
            "Y": float(row["Y_b"]),
            "R": int(row["R_b"]),
            "cum_dx_px": cum_dx,
            "cum_dy_px": cum_dy,
            "step_norm_px": float(row["measured_norm_px"]),
            "pair_class": str(row["pair_class"]),
        })
    return pd.DataFrame(rows)


def summarize_ep04_phase(output_root: Path) -> dict:
    summaries: dict[str, dict] = {}
    files = {
        "outer_rows": output_root / "ep04_global_validation" / "segment_validation_results.csv",
        "outer_segments": output_root / "ep04_global_validation" / "segment_summary.csv",
        "inner_rows": output_root / "ep04_global_validation" / "inner" / "inner_segment_validation_results.csv",
        "inner_segments": output_root / "ep04_global_validation" / "inner" / "inner_segment_summary.csv",
    }
    for label, path in files.items():
        if not path.exists():
            continue
        df = pd.read_csv(path)
        phase_col = "phase_coverage_px" if "phase_coverage_px" in df else "phase_coverage_median_px"
        split_col = "split_half_diff_px" if "split_half_diff_px" in df else "split_half_median_px"
        entry = {
            "path": str(path),
            "n": int(len(df)),
            "phase_coverage_px": quantile_dict(df[phase_col]) if phase_col in df else {"count": 0},
        }
        if split_col in df:
            entry["split_half_px"] = quantile_dict(df[split_col])
        if "pass_fail" in df:
            passed = df[boolish(df["pass_fail"])]
            entry["pass_count"] = int(len(passed))
            entry["pass_rate"] = float(len(passed) / len(df)) if len(df) else np.nan
            entry["passed_phase_coverage_px"] = quantile_dict(passed[phase_col]) if phase_col in passed else {"count": 0}
        summaries[label] = entry
    return summaries


def plot_trajectory(trajectory: pd.DataFrame, output_dir: Path) -> None:
    if trajectory.empty:
        return
    setup_academic_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["one_half_col"])
    sc = ax.scatter(
        trajectory["cum_dx_px"],
        trajectory["cum_dy_px"],
        c=trajectory["acquisition_order"],
        cmap=COLORMAPS["coverage"],
        s=16,
        linewidths=0,
    )
    ax.plot(trajectory["cum_dx_px"], trajectory["cum_dy_px"], color="#444444", linewidth=0.7, alpha=0.45)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Cumulative measured dx [px]")
    ax.set_ylabel("Cumulative measured dy [px]")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Acquisition order")
    savefig_academic(fig, output_dir / "main_session_cumulative_trajectory.png")


def plot_norm_summary(measurements: pd.DataFrame, output_dir: Path) -> None:
    data = measurements[
        measurements["method"].eq("highpass")
        & measurements["pair_set"].isin([
            "main_acq_adjacent",
            "same_y_scanline_endpoint",
            "same_x_column_endpoint",
        ])
    ].copy()
    if data.empty:
        return
    setup_academic_style()
    preferred_order = [
        "same_y_x_motion",
        "raster_row_reset",
        "same_x_y_motion",
        "mixed_xy_motion",
    ]
    extra_order = sorted(set(data["pair_class"]) - set(preferred_order))
    order = preferred_order + extra_order
    data["plot_class"] = pd.Categorical(data["pair_class"], categories=order, ordered=True)
    data = data.sort_values("plot_class")
    classes = [cls for cls in order if cls in set(data["pair_class"])]
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["one_half_col"])
    values = [data.loc[data["pair_class"].eq(cls), "measured_norm_px"].dropna().to_numpy() for cls in classes]
    ax.boxplot(values, tick_labels=classes, showfliers=False)
    ax.set_ylabel("Measured 2D shift magnitude [px]")
    ax.tick_params(axis="x", rotation=25)
    savefig_academic(fig, output_dir / "visible_shift_by_pair_class.png")


def plot_endpoint_vectors(measurements: pd.DataFrame, output_dir: Path) -> None:
    data = measurements[
        measurements["method"].eq("highpass")
        & measurements["pair_set"].isin(["same_y_scanline_endpoint", "same_x_column_endpoint"])
    ].copy()
    if data.empty:
        return
    setup_academic_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["one_half_col"])
    for idx, (pair_set, group) in enumerate(data.groupby("pair_set")):
        ax.scatter(
            group["measured_dx_px"],
            group["measured_dy_px"],
            s=20,
            color=METHOD_COLOR_LIST[idx],
            label=pair_set.replace("_", " "),
            alpha=0.8,
        )
    ax.axhline(0, color="#888888", linewidth=0.7)
    ax.axvline(0, color="#888888", linewidth=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Measured dx [px]")
    ax.set_ylabel("Measured dy [px]")
    ax.legend()
    savefig_academic(fig, output_dir / "endpoint_displacement_vectors.png")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-audit-csv", type=Path, default=root / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "data_raw" / "infrared_avi")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "ep05_sr_reassessment")
    parser.add_argument("--stage-config", type=Path, default=root / "configs" / "stage_calibration.json")
    parser.add_argument("--roi-size", type=int, default=320)
    parser.add_argument("--n-jobs", type=int, default=default_workers())
    parser.add_argument("--methods", nargs="+", default=["highpass", "gradient"])
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    root = project_root()
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_academic_style()

    with open(args.stage_config, encoding="utf-8") as f:
        stage_config = json.load(f)
    theta_deg = float(stage_config["theta_deg"])
    pixel_size_um = float(stage_config["pixel_size_um"])

    audit = pd.read_csv(args.frame_audit_csv)
    pair_table = build_pair_table(audit)
    pair_table.to_csv(args.output_dir / "registration_pair_table.csv", index=False)

    tasks: list[PairTask] = []
    for _, row in pair_table.iterrows():
        for method in args.methods:
            tasks.append(task_from_pair(row, method, args.data_dir, theta_deg, pixel_size_um, args.roi_size))

    print("EP05 displacement reassessment")
    print(f"pairs: {len(pair_table)}; tasks: {len(tasks)}; methods: {args.methods}")
    print(f"data: {args.data_dir}")
    print(f"output: {args.output_dir}")
    print(f"n_jobs={args.n_jobs}, roi_size={args.roi_size}, theta={theta_deg:.1f}, pixel={pixel_size_um:.1f}")

    records = []
    if args.n_jobs == 1:
        iterator: Iterable[PairTask] = tqdm(tasks, desc="register")
        for task in iterator:
            records.append(run_registration_task(task))
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = [executor.submit(run_registration_task, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="register"):
                records.append(future.result())

    measurements = pd.DataFrame(records)
    measurements = measurements.merge(
        pair_table[["pair_set", "pair_id", "pair_class", "cross_session"]],
        on=["pair_set", "pair_id"],
        how="left",
    )
    measurements = measurements.sort_values(["pair_set", "method", "acq_a", "acq_b"]).reset_index(drop=True)
    measurements.to_csv(args.output_dir / "displacement_measurements.csv", index=False)

    summary_table = summarize_measurements(measurements)
    summary_table.to_csv(args.output_dir / "displacement_summary_by_class.csv", index=False)

    trajectory = build_cumulative_trajectory(measurements, method="highpass")
    trajectory.to_csv(args.output_dir / "main_session_cumulative_trajectory.csv", index=False)

    trajectory_summary = {}
    if not trajectory.empty:
        span_x = float(trajectory["cum_dx_px"].max() - trajectory["cum_dx_px"].min())
        span_y = float(trajectory["cum_dy_px"].max() - trajectory["cum_dy_px"].min())
        trajectory_summary = {
            "n_frames": int(len(trajectory)),
            "span_dx_px": span_x,
            "span_dy_px": span_y,
            "span_norm_px": float(np.hypot(span_x, span_y)),
            "path_length_px": float(trajectory["step_norm_px"].sum()),
            "x_coordinate_count": int(trajectory["X"].nunique()),
            "y_coordinate_count": int(trajectory["Y"].nunique()),
        }

    ep04_phase = summarize_ep04_phase(root / "output")
    summary = {
        "theta_deg": theta_deg,
        "pixel_size_um": pixel_size_um,
        "roi_size": int(args.roi_size),
        "n_pairs": int(len(pair_table)),
        "n_registration_tasks": int(len(tasks)),
        "methods": list(args.methods),
        "trajectory": trajectory_summary,
        "ep04_phase_coverage": ep04_phase,
        "key_highpass_summaries": summary_table[
            summary_table["method"].eq("highpass")
            & summary_table["pair_set"].isin([
                "main_acq_adjacent",
                "same_y_scanline_endpoint",
                "same_x_column_endpoint",
            ])
        ].to_dict(orient="records"),
    }
    with open(args.output_dir / "displacement_reassessment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not args.skip_figures:
        plot_trajectory(trajectory, args.output_dir)
        plot_norm_summary(measurements, args.output_dir)
        plot_endpoint_vectors(measurements, args.output_dir)

    print("\nKey high-pass summaries")
    cols = [
        "pair_set",
        "pair_class",
        "n",
        "measured_norm_median_px",
        "measured_norm_p90_px",
        "model_norm_median_px",
        "norm_ratio_median",
        "peak_ncc_median",
    ]
    display = summary_table[summary_table["method"].eq("highpass")][cols]
    print(display.to_string(index=False))
    if trajectory_summary:
        print("\nCumulative trajectory")
        for key, value in trajectory_summary.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
