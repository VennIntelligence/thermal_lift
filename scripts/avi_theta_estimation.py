#!/usr/bin/env python3
"""Estimate theta direction from EP02 AVI motion directions.

The AVI continuous scans provide an auxiliary directional check, not an
alignment-truth source and not a replacement for the configured calibration:

- X-scan direction = theta
- Y-scan direction = theta + 90 deg

Usage:
    uv run python scripts/avi_theta_estimation.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = PROJECT_ROOT / "output" / "ep02_displacement_calibration"
REFERENCE_THETA_DEG = 47.6

sys.path.insert(0, str(PROJECT_ROOT / "core" / "src"))
from thermal_core.plotting import METHOD_COLORS, make_figure, savefig_academic, setup_academic_style
from thermal_core.viz import plot_avi_theta_bracket_summary


def wrap_angle_180(angle_deg: float | np.ndarray) -> float | np.ndarray:
    """Wrap angle(s) to [-180, 180)."""
    return (np.asarray(angle_deg) + 180.0) % 360.0 - 180.0


def infer_scan_axis(avi_name: str) -> str:
    stem = Path(str(avi_name)).stem.lower()
    if stem.startswith("x"):
        return "x"
    if stem.startswith("y"):
        return "y"
    raise ValueError(f"cannot infer scan axis from AVI name: {avi_name}")


def normal_ci(values: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    """Normal-approximation confidence interval for the mean."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return float("nan"), float("nan")
    if values.size == 1:
        value = float(values[0])
        return value, value
    z = 1.959963984540054 if math.isclose(confidence, 0.95) else 1.96
    mean = float(np.mean(values))
    std = float(np.std(values, ddof=1))
    half_width = z * std / math.sqrt(values.size)
    return mean - half_width, mean + half_width


def robust_per_avi_ci(theta_est: float, angle_mad: float, n_pairs: float) -> tuple[float, float]:
    """Approximate per-AVI CI from angular MAD over motion pairs."""
    if not np.isfinite(angle_mad) or not np.isfinite(n_pairs) or n_pairs <= 1:
        return theta_est, theta_est
    robust_sigma = 1.4826 * float(angle_mad)
    half_width = 1.959963984540054 * robust_sigma / math.sqrt(float(n_pairs))
    return theta_est - half_width, theta_est + half_width


def theta_from_direction(direction_deg: float, scan_axis: str) -> float:
    if scan_axis == "x":
        return float(wrap_angle_180(direction_deg))
    if scan_axis == "y":
        return float(wrap_angle_180(direction_deg - 90.0))
    raise ValueError(f"unknown scan axis: {scan_axis}")


def find_gradient_summary(output_dir: Path) -> Path | None:
    candidates = [
        output_dir / "avi_gradient_check" / "avi_direction_summary.csv",
        output_dir / "avi_method_check_gradient" / "avi_direction_summary.csv",
    ]
    for path in candidates:
        if path.exists():
            return path
    for path in sorted(output_dir.glob("avi_*gradient*/avi_direction_summary.csv")):
        return path
    return None


def read_method_summary(output_dir: Path, method: str) -> pd.DataFrame:
    if method == "highpass":
        path = output_dir / "avi_direction_summary.csv"
    elif method == "gradient":
        path = find_gradient_summary(output_dir)
        if path is None:
            root = output_dir / "avi_direction_summary.csv"
            if root.exists():
                root_df = pd.read_csv(root)
                if "preprocess" in root_df.columns and (root_df["preprocess"] == "gradient").any():
                    df = root_df[root_df["preprocess"] == "gradient"].copy()
                    df["method_source"] = str(root)
                    return df
            raise FileNotFoundError("could not find gradient AVI direction summary CSV")
    else:
        raise ValueError(f"unknown method: {method}")

    if path is None or not path.exists():
        raise FileNotFoundError(f"missing AVI direction summary: {path}")

    df = pd.read_csv(path)
    if "preprocess" in df.columns:
        wanted = "highpass" if method == "highpass" else "gradient"
        if (df["preprocess"] == wanted).any():
            df = df[df["preprocess"] == wanted].copy()
    df["method_source"] = str(path)
    return df


def read_method_pairs(output_dir: Path, method: str) -> pd.DataFrame | None:
    if method == "highpass":
        path = output_dir / "avi_registration_pairs.csv"
    else:
        summary_path = find_gradient_summary(output_dir)
        path = summary_path.with_name("avi_registration_pairs.csv") if summary_path is not None else None
    if path is None or not path.exists():
        return None
    df = pd.read_csv(path)
    return df if {"avi", "dx_px", "dy_px"}.issubset(df.columns) else None


def cumulative_slope_directions(pair_df: pd.DataFrame | None) -> dict[str, float]:
    if pair_df is None:
        return {}

    directions: dict[str, float] = {}
    for avi_name, group in pair_df.groupby("avi"):
        subset = group.copy()
        if "is_motion_segment" in subset.columns:
            subset = subset[subset["is_motion_segment"].astype(bool)]
        if "fit_ok" in subset.columns:
            subset = subset[subset["fit_ok"].astype(bool)]
        if "edge_peak" in subset.columns:
            subset = subset[~subset["edge_peak"].astype(bool)]
        if subset.empty:
            continue
        dx = float(subset["dx_px"].sum())
        dy = float(subset["dy_px"].sum())
        if not np.isfinite(dx + dy) or (abs(dx) + abs(dy)) <= 0:
            continue
        directions[str(avi_name)] = float(np.degrees(np.arctan2(dy, dx)))
    return directions


def build_theta_estimates(output_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, float | str | int | None]] = []
    for method in ["highpass", "gradient"]:
        summary = read_method_summary(output_dir, method)
        pair_directions = cumulative_slope_directions(read_method_pairs(output_dir, method))

        for record in summary.to_dict("records"):
            avi_name = str(record.get("avi", record.get("avi_name", "")))
            scan_axis = str(record.get("scan_axis") or infer_scan_axis(avi_name)).lower()
            direction = float(record.get("median_angle_row_down_deg", record.get("direction_deg")))
            theta_est = theta_from_direction(direction, scan_axis)

            angle_mad = float(record.get("angle_row_down_mad_deg", np.nan))
            n_pairs = float(record.get("motion_pairs", np.nan))
            ci_lower, ci_upper = robust_per_avi_ci(theta_est, angle_mad, n_pairs)

            slope_direction = pair_directions.get(avi_name)
            slope_theta = (
                theta_from_direction(slope_direction, scan_axis)
                if slope_direction is not None
                else np.nan
            )

            rows.append(
                {
                    "avi_name": avi_name,
                    "scan_axis": scan_axis,
                    "method": method,
                    "direction_deg": direction,
                    "theta_est_deg": theta_est,
                    "theta_ci_lower_deg": ci_lower,
                    "theta_ci_upper_deg": ci_upper,
                    "n_motion_pairs": int(n_pairs) if np.isfinite(n_pairs) else None,
                    "angle_mad_deg": angle_mad,
                    "slope_direction_deg": slope_direction if slope_direction is not None else np.nan,
                    "slope_theta_est_deg": slope_theta,
                }
            )

    estimates = pd.DataFrame(rows)
    estimates = estimates.sort_values(["method", "scan_axis", "avi_name"]).reset_index(drop=True)
    return estimates


def summarize_theta(estimates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | str | int | bool]] = []
    source_masks = {
        "x-only": estimates["scan_axis"] == "x",
        "y-only": estimates["scan_axis"] == "y",
        "combined": estimates["scan_axis"].isin(["x", "y"]),
    }

    for method, method_df in estimates.groupby("method", sort=False):
        for source, mask in source_masks.items():
            subset = method_df[mask.loc[method_df.index]]
            values = subset["theta_est_deg"].to_numpy(dtype=float)
            values = values[np.isfinite(values)]
            if values.size == 0:
                continue
            ci_lower, ci_upper = normal_ci(values)
            rows.append(
                {
                    "method": method,
                    "source": source,
                    "n": int(values.size),
                    "mean_deg": float(np.mean(values)),
                    "median_deg": float(np.median(values)),
                    "std_deg": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
                    "ci_lower_deg": ci_lower,
                    "ci_upper_deg": ci_upper,
                    "min_deg": float(np.min(values)),
                    "max_deg": float(np.max(values)),
                    "within_ci_47_6": bool(ci_lower <= REFERENCE_THETA_DEG <= ci_upper),
                }
            )
    return pd.DataFrame(rows)


def plot_forest(estimates: pd.DataFrame, output_path: Path) -> None:
    setup_academic_style()
    methods = ["highpass", "gradient"]
    titles = {"highpass": "High-pass NCC", "gradient": "Gradient NCC"}
    colors = {
        "x": METHOD_COLORS["primary"],
        "y": METHOD_COLORS["accent_1"],
        "combined": METHOD_COLORS["secondary"]
    }

    # Disable constrained layout temporarily so we can manually adjust subplots_adjust
    _cl_backup = plt.rcParams.get("figure.constrained_layout.use", False)
    plt.rcParams["figure.constrained_layout.use"] = False

    fig, axes = make_figure("double_col", nrows=1, ncols=2, height=4.5, sharex=True, constrained_layout=False)
    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    plt.rcParams["figure.constrained_layout.use"] = _cl_backup

    for ax, method in zip(axes.ravel(), methods, strict=True):
        subset = estimates[estimates["method"] == method].copy()
        subset = subset.sort_values(["scan_axis", "avi_name"]).reset_index(drop=True)
        
        n_individual = len(subset)
        y_pos = np.arange(n_individual)

        # Plot individual files
        for axis in ["x", "y"]:
            axis_subset = subset[subset["scan_axis"] == axis]
            idx = axis_subset.index.to_numpy()
            theta = axis_subset["theta_est_deg"].to_numpy(dtype=float)
            lower = axis_subset["theta_ci_lower_deg"].to_numpy(dtype=float)
            upper = axis_subset["theta_ci_upper_deg"].to_numpy(dtype=float)
            xerr = np.vstack([theta - lower, upper - theta])
            ax.errorbar(
                theta,
                y_pos[idx],
                xerr=xerr,
                fmt="o",
                ms=4.0,
                capsize=2.0,
                lw=0.8,
                color=colors[axis],
                ecolor=colors[axis],
                label=f"{axis.upper()}-scan study",
                zorder=3,
            )

        # Draw separation line between individual estimates and pooled summaries
        ax.axhline(n_individual - 0.5, color="#888888", linestyle="-", linewidth=0.7, alpha=0.6)

        # Calculate and plot pooled summaries
        # 1. X-scan Pooled
        x_vals = subset[subset["scan_axis"] == "x"]["theta_est_deg"].to_numpy(dtype=float)
        x_mean = np.mean(x_vals)
        x_lo, x_hi = normal_ci(x_vals)
        y_x = n_individual + 0.5
        ax.errorbar(
            [x_mean],
            [y_x],
            xerr=[[x_mean - x_lo], [x_hi - x_mean]],
            fmt="D",
            ms=5.5,
            capsize=3.0,
            lw=1.2,
            color=colors["x"],
            ecolor=colors["x"],
            label="X-scan Pooled",
            zorder=4,
        )

        # 2. Y-scan Pooled
        y_vals = subset[subset["scan_axis"] == "y"]["theta_est_deg"].to_numpy(dtype=float)
        y_mean = np.mean(y_vals)
        y_lo, y_hi = normal_ci(y_vals)
        y_y = n_individual + 1.5
        ax.errorbar(
            [y_mean],
            [y_y],
            xerr=[[y_mean - y_lo], [y_hi - y_mean]],
            fmt="D",
            ms=5.5,
            capsize=3.0,
            lw=1.2,
            color=colors["y"],
            ecolor=colors["y"],
            label="Y-scan Pooled",
            zorder=4,
        )

        # 3. Overall Combined Pooled
        all_vals = subset["theta_est_deg"].to_numpy(dtype=float)
        comb_mean = np.mean(all_vals)
        comb_lo, comb_hi = normal_ci(all_vals)
        y_comb = n_individual + 2.5
        ax.errorbar(
            [comb_mean],
            [y_comb],
            xerr=[[comb_mean - comb_lo], [comb_hi - comb_mean]],
            fmt="D",
            ms=6.5,
            capsize=4.0,
            lw=1.5,
            color=colors["combined"],
            ecolor=colors["combined"],
            label="Combined Pooled",
            zorder=4,
        )

        # Vertical reference line
        ax.axvline(REFERENCE_THETA_DEG, color="#333333", ls="--", lw=1.0, label=f"Ref ({REFERENCE_THETA_DEG:.1f} deg)")
        
        # Shade Combined Pooled 95% CI to show how it covers the reference
        ax.axvspan(comb_lo, comb_hi, color=colors["combined"], alpha=0.08, zorder=0, label="Combined 95% CI")

        # Layout setup
        ax.set_title(titles[method])
        ax.set_xlabel("Theta estimate [deg]")
        
        all_y_pos = list(y_pos) + [y_x, y_y, y_comb]
        all_y_labels = subset["avi_name"].tolist() + ["X-scan Pooled", "Y-scan Pooled", "Combined Pooled"]
        
        ax.set_yticks(all_y_pos)
        ax.set_yticklabels(all_y_labels, fontsize=7)
        ax.grid(axis="x", alpha=0.25, linewidth=0.5)
        ax.set_xlim(39.0, 54.0)
        ax.invert_yaxis()
        ax.legend(loc="upper left", fontsize=6.5, frameon=False)
        
    axes[0].set_ylabel("AVI Source")
    fig.subplots_adjust(left=0.14, right=0.98, bottom=0.12, top=0.88, wspace=0.22)
    savefig_academic(fig, output_path)


def write_outputs(estimates: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    estimates_path = output_dir / "avi_theta_estimates.csv"
    summary_path = output_dir / "avi_theta_summary.csv"
    bracket_path = output_dir / "avi_theta_bracket_plot.png"
    figure_path = output_dir / "avi_theta_forest_plot.png"
    result_path = output_dir / "avi_theta_result.json"

    estimates.to_csv(estimates_path, index=False)
    summary.to_csv(summary_path, index=False)
    plot_avi_theta_bracket_summary(
        summary,
        method="gradient",
        reference_deg=REFERENCE_THETA_DEG,
        output_path=bracket_path,
    )
    plot_forest(estimates, figure_path)

    best_row = summary[(summary["method"] == "gradient") & (summary["source"] == "combined")]
    if best_row.empty:
        best_row = summary[summary["source"] == "combined"].head(1)
    best = best_row.iloc[0].to_dict()

    result = {
        "best_theta": float(best["mean_deg"]),
        "best_theta_median": float(best["median_deg"]),
        "ci_lower": float(best["ci_lower_deg"]),
        "ci_upper": float(best["ci_upper_deg"]),
        "method": str(best["method"]),
        "source": str(best["source"]),
        "n_samples": int(best["n"]),
        "within_ci_47_6": bool(best["within_ci_47_6"]),
        "reference_theta": REFERENCE_THETA_DEG,
        "outputs": {
            "estimates_csv": str(estimates_path),
            "summary_csv": str(summary_path),
            "bracket_plot": str(bracket_path),
            "forest_plot": str(figure_path),
        },
    }
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result


def print_summary(summary: pd.DataFrame, result: dict[str, object]) -> None:
    display = summary.copy()
    float_cols = ["mean_deg", "median_deg", "std_deg", "ci_lower_deg", "ci_upper_deg", "min_deg", "max_deg"]
    for col in float_cols:
        display[col] = display[col].map(lambda value: f"{value:.3f}")

    print("\nAVI theta estimates by method/source")
    print(display.to_string(index=False))
    print("\nBest estimate:")
    print(f"  method/source: {result['method']} / {result['source']}")
    print(f"  theta mean: {result['best_theta']:.3f} deg")
    print(f"  95% CI: [{result['ci_lower']:.3f}, {result['ci_upper']:.3f}] deg")
    print(f"  n: {result['n_samples']}")
    print(f"  47.6 deg within CI: {result['within_ci_47_6']}")

    delta = abs(float(result["best_theta"]) - REFERENCE_THETA_DEG)
    if bool(result["within_ci_47_6"]) and delta < 1.0:
        recommendation = (
            "支持当前 theta=47.6 deg；AVI 是独立方向证据，但精度不足以替换配置值。"
        )
    else:
        recommendation = (
            "不建议仅凭 AVI 更新 theta；需要回到温度矩阵和主 session 对齐诊断复核。"
        )
    print(f"\n诊断建议: {recommendation}")


def main() -> None:
    estimates = build_theta_estimates(OUTPUT_DIR)
    summary = summarize_theta(estimates)
    result = write_outputs(estimates, summary, OUTPUT_DIR)
    print_summary(summary, result)


if __name__ == "__main__":
    main()
