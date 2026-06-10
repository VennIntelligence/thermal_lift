#!/usr/bin/env python3
"""Measure EP15 M1 sub-pixel phase structure from command and measured shifts.

M1 checks whether the clean 248-frame main session still provides the expected
5x5 micro-scan phase lattice after replacing stage-command shifts with the
EP05 data-driven contour-refined alignment shifts.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCRIPT_PATH = Path(__file__).resolve()
ALGO_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[3]
EP06_SRC = PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src"

for path in (ALGO_ROOT / "src", EP06_SRC, PROJECT_ROOT / "core" / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from thermal_core.alignment_paths import default_contour_alignment_csv  # noqa: E402

from common.alignment import load_alignment_shifts  # noqa: E402
from common.data_loader import load_main_session_metadata  # noqa: E402
from thermal_core.displacement import coordinate_to_shift  # noqa: E402
from thermal_core.plotting import COLORMAPS, METHOD_COLOR_LIST, savefig_academic, setup_academic_style  # noqa: E402


EXPECTED_CLEAN_SR_FRAMES = 248


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _load_stage_config(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        config = json.load(f)
    return {
        "theta_deg": float(config["theta_deg"]),
        "pixel_size_um": float(config["pixel_size_um"]),
    }


def _reference_file(alignment_csv: Path) -> str:
    alignment = pd.read_csv(alignment_csv, usecols=lambda col: col in {"reference_file"})
    if "reference_file" not in alignment or alignment["reference_file"].dropna().empty:
        raise ValueError(f"{alignment_csv} does not contain a reference_file column")
    return str(alignment["reference_file"].dropna().iloc[0])


def command_alignment_prior(
    metadata: pd.DataFrame,
    reference_file: str,
    *,
    theta_deg: float,
    pixel_size_um: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return command relative stage coordinates and EP05-convention shifts."""

    ref_rows = metadata[metadata["file"].astype(str).eq(reference_file)]
    if ref_rows.empty:
        raise ValueError(f"reference_file={reference_file!r} is not present in clean metadata")
    ref = ref_rows.iloc[0]
    rel_x_um = metadata["X"].to_numpy(dtype=float) - float(ref["X"])
    rel_y_um = metadata["Y"].to_numpy(dtype=float) - float(ref["Y"])
    detector_dx, detector_dy_math = coordinate_to_shift(
        rel_x_um,
        rel_y_um,
        theta_deg=theta_deg,
        pixel_size_um=pixel_size_um,
    )
    # EP05 alignment convention: positive align_dx/align_dy moves the frame
    # into the reference coordinate system. This is the old_stage_model formula.
    align_dx = -np.asarray(detector_dx, dtype=float)
    align_dy = np.asarray(detector_dy_math, dtype=float)
    return rel_x_um, rel_y_um, align_dx, align_dy


def alignment_shift_to_stage_um(
    align_dx: np.ndarray,
    align_dy: np.ndarray,
    *,
    theta_deg: float,
    pixel_size_um: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Invert EP05-convention alignment shifts into stage-equivalent micrometers."""

    theta = np.radians(theta_deg)
    detector_dx = -np.asarray(align_dx, dtype=float)
    detector_dy_math = np.asarray(align_dy, dtype=float)
    stage_x_um = pixel_size_um * (detector_dx * np.cos(theta) - detector_dy_math * np.sin(theta))
    stage_y_um = pixel_size_um * (detector_dx * np.sin(theta) + detector_dy_math * np.cos(theta))
    return stage_x_um, stage_y_um


def phase_fraction(values_um_or_px: np.ndarray, period: float) -> np.ndarray:
    return np.mod(np.asarray(values_um_or_px, dtype=float), period) / period


def circular_phase_delta(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return (np.asarray(lhs, dtype=float) - np.asarray(rhs, dtype=float) + 0.5) % 1.0 - 0.5


def nearest_lattice_index(phase: np.ndarray, scale: int) -> np.ndarray:
    return np.mod(np.floor(np.asarray(phase, dtype=float) * scale + 0.5).astype(int), scale)


def floor_bin_index(phase: np.ndarray, scale: int) -> np.ndarray:
    return np.clip(np.floor(np.asarray(phase, dtype=float) * scale).astype(int), 0, scale - 1)


def nearest_lattice_distance_um(phase_x: np.ndarray, phase_y: np.ndarray, *, scale: int, period_um: float) -> np.ndarray:
    nearest_x = np.round(np.asarray(phase_x, dtype=float) * scale) / scale
    nearest_y = np.round(np.asarray(phase_y, dtype=float) * scale) / scale
    delta_x = circular_phase_delta(phase_x, nearest_x)
    delta_y = circular_phase_delta(phase_y, nearest_y)
    return np.hypot(delta_x, delta_y) * period_um


def count_matrix(x_index: np.ndarray, y_index: np.ndarray, scale: int) -> np.ndarray:
    counts = np.zeros((scale, scale), dtype=int)
    np.add.at(counts, (np.asarray(y_index, dtype=int), np.asarray(x_index, dtype=int)), 1)
    return counts


def occupancy_metrics(counts: np.ndarray) -> dict[str, Any]:
    flat = np.asarray(counts, dtype=float).ravel()
    n = int(flat.sum())
    probs = flat[flat > 0] / max(n, 1)
    entropy = float(-(probs * np.log(probs)).sum() / np.log(flat.size)) if probs.size else 0.0
    return {
        "n_frames": n,
        "occupied_cells": int(np.count_nonzero(flat)),
        "empty_cells": int(np.count_nonzero(flat == 0)),
        "total_cells": int(flat.size),
        "min_count": int(flat.min()) if flat.size else 0,
        "max_count": int(flat.max()) if flat.size else 0,
        "mean_count": float(flat.mean()) if flat.size else 0.0,
        "count_cv": float(flat.std() / flat.mean()) if flat.size and flat.mean() > 0 else 0.0,
        "entropy_fraction": entropy,
    }


def quantile_summary(values: np.ndarray) -> dict[str, float]:
    clean = np.asarray(values, dtype=float)
    clean = clean[np.isfinite(clean)]
    if clean.size == 0:
        return {"median": float("nan"), "p90": float("nan"), "p95": float("nan"), "max": float("nan")}
    return {
        "median": float(np.quantile(clean, 0.5)),
        "p90": float(np.quantile(clean, 0.9)),
        "p95": float(np.quantile(clean, 0.95)),
        "max": float(np.max(clean)),
    }


def method_records(
    metadata: pd.DataFrame,
    *,
    method: str,
    align_dx: np.ndarray,
    align_dy: np.ndarray,
    stage_x_um: np.ndarray,
    stage_y_um: np.ndarray,
    command_stage_x_um: np.ndarray,
    command_stage_y_um: np.ndarray,
    command_align_dx: np.ndarray,
    command_align_dy: np.ndarray,
    scale: int,
    pixel_size_um: float,
) -> pd.DataFrame:
    stage_phase_x = phase_fraction(stage_x_um, pixel_size_um)
    stage_phase_y = phase_fraction(stage_y_um, pixel_size_um)
    command_stage_phase_x = phase_fraction(command_stage_x_um, pixel_size_um)
    command_stage_phase_y = phase_fraction(command_stage_y_um, pixel_size_um)
    detector_phase_x = phase_fraction(align_dx, 1.0)
    detector_phase_y = phase_fraction(align_dy, 1.0)

    stage_idx_x = nearest_lattice_index(stage_phase_x, scale)
    stage_idx_y = nearest_lattice_index(stage_phase_y, scale)
    command_stage_idx_x = nearest_lattice_index(command_stage_phase_x, scale)
    command_stage_idx_y = nearest_lattice_index(command_stage_phase_y, scale)
    detector_bin_x = floor_bin_index(detector_phase_x, scale)
    detector_bin_y = floor_bin_index(detector_phase_y, scale)

    delta_phase_x = circular_phase_delta(stage_phase_x, command_stage_phase_x)
    delta_phase_y = circular_phase_delta(stage_phase_y, command_stage_phase_y)
    delta_stage_um = np.hypot(delta_phase_x, delta_phase_y) * pixel_size_um
    delta_shift_px = np.hypot(align_dx - command_align_dx, align_dy - command_align_dy)
    nearest_grid_um = nearest_lattice_distance_um(
        stage_phase_x,
        stage_phase_y,
        scale=scale,
        period_um=pixel_size_um,
    )

    out = pd.DataFrame(
        {
            "method": method,
            "frame_index": metadata["frame_index"].to_numpy(dtype=int),
            "file": metadata["file"].astype(str).to_numpy(),
            "acquisition_order": metadata["acquisition_order"].to_numpy(dtype=int),
            "X": metadata["X"].to_numpy(dtype=float),
            "Y": metadata["Y"].to_numpy(dtype=float),
            "R": metadata["R"].to_numpy(dtype=int),
            "align_dx_px": align_dx,
            "align_dy_px": align_dy,
            "stage_x_equiv_um": stage_x_um,
            "stage_y_equiv_um": stage_y_um,
            "stage_phase_x": stage_phase_x,
            "stage_phase_y": stage_phase_y,
            "stage_lattice_x": stage_idx_x,
            "stage_lattice_y": stage_idx_y,
            "command_stage_phase_x": command_stage_phase_x,
            "command_stage_phase_y": command_stage_phase_y,
            "command_stage_lattice_x": command_stage_idx_x,
            "command_stage_lattice_y": command_stage_idx_y,
            "detector_phase_x": detector_phase_x,
            "detector_phase_y": detector_phase_y,
            "detector_bin_x": detector_bin_x,
            "detector_bin_y": detector_bin_y,
            "delta_stage_phase_x": delta_phase_x,
            "delta_stage_phase_y": delta_phase_y,
            "delta_stage_um_wrapped": delta_stage_um,
            "delta_shift_px_from_command": delta_shift_px,
            "nearest_stage_lattice_um": nearest_grid_um,
        }
    )
    out["stage_lattice_matches_command"] = (
        out["stage_lattice_x"].eq(out["command_stage_lattice_x"])
        & out["stage_lattice_y"].eq(out["command_stage_lattice_y"])
    )
    return out


def summarize_method(group: pd.DataFrame, scale: int) -> dict[str, Any]:
    stage_counts = count_matrix(group["stage_lattice_x"], group["stage_lattice_y"], scale)
    detector_counts = count_matrix(group["detector_bin_x"], group["detector_bin_y"], scale)
    stage_occ = occupancy_metrics(stage_counts)
    detector_occ = occupancy_metrics(detector_counts)
    return {
        "method": str(group["method"].iloc[0]),
        "n_frames": int(len(group)),
        "scale": int(scale),
        "stage_lattice_occupied": stage_occ["occupied_cells"],
        "stage_lattice_empty": stage_occ["empty_cells"],
        "stage_lattice_min_count": stage_occ["min_count"],
        "stage_lattice_max_count": stage_occ["max_count"],
        "stage_lattice_entropy_fraction": stage_occ["entropy_fraction"],
        "stage_lattice_count_cv": stage_occ["count_cv"],
        "detector_bin_occupied": detector_occ["occupied_cells"],
        "detector_bin_empty": detector_occ["empty_cells"],
        "detector_bin_min_count": detector_occ["min_count"],
        "detector_bin_max_count": detector_occ["max_count"],
        "detector_bin_entropy_fraction": detector_occ["entropy_fraction"],
        "stage_lattice_match_rate": float(group["stage_lattice_matches_command"].mean()),
        "nearest_stage_lattice_um_median": quantile_summary(group["nearest_stage_lattice_um"])["median"],
        "nearest_stage_lattice_um_p90": quantile_summary(group["nearest_stage_lattice_um"])["p90"],
        "delta_stage_um_wrapped_median": quantile_summary(group["delta_stage_um_wrapped"])["median"],
        "delta_stage_um_wrapped_p90": quantile_summary(group["delta_stage_um_wrapped"])["p90"],
        "delta_shift_px_from_command_median": quantile_summary(group["delta_shift_px_from_command"])["median"],
        "delta_shift_px_from_command_p90": quantile_summary(group["delta_shift_px_from_command"])["p90"],
    }


def make_count_tables(samples: pd.DataFrame, scale: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    stage_rows: list[dict[str, Any]] = []
    detector_rows: list[dict[str, Any]] = []
    for method, group in samples.groupby("method", sort=False):
        stage_counts = count_matrix(group["stage_lattice_x"], group["stage_lattice_y"], scale)
        detector_counts = count_matrix(group["detector_bin_x"], group["detector_bin_y"], scale)
        for y_idx in range(scale):
            for x_idx in range(scale):
                stage_rows.append(
                    {
                        "method": method,
                        "stage_lattice_y": y_idx,
                        "stage_lattice_x": x_idx,
                        "count": int(stage_counts[y_idx, x_idx]),
                    }
                )
                detector_rows.append(
                    {
                        "method": method,
                        "detector_bin_y": y_idx,
                        "detector_bin_x": x_idx,
                        "count": int(detector_counts[y_idx, x_idx]),
                    }
                )
    return pd.DataFrame(stage_rows), pd.DataFrame(detector_rows)


def plot_stage_lattice_heatmaps(samples: pd.DataFrame, output_path: Path, scale: int) -> None:
    setup_academic_style()
    methods = list(samples["method"].drop_duplicates())
    fig, axes = plt.subplots(1, len(methods), figsize=(7.2, 2.55), squeeze=False)
    max_count = 1
    counts_by_method: dict[str, np.ndarray] = {}
    for method in methods:
        group = samples[samples["method"].eq(method)]
        counts = count_matrix(group["stage_lattice_x"], group["stage_lattice_y"], scale)
        counts_by_method[method] = counts
        max_count = max(max_count, int(counts.max()))

    for ax, method in zip(axes.ravel(), methods, strict=True):
        counts = counts_by_method[method]
        im = ax.imshow(counts, origin="lower", cmap=COLORMAPS["coverage"], vmin=0, vmax=max_count)
        for y_idx in range(scale):
            for x_idx in range(scale):
                value = int(counts[y_idx, x_idx])
                color = "black" if value > max_count * 0.55 else "white"
                ax.text(x_idx, y_idx, str(value), ha="center", va="center", fontsize=7, color=color)
        ax.set_title(method.replace("_", " "))
        ax.set_xlabel("stage phase-x lattice")
        ax.set_xticks(range(scale))
        ax.set_yticks(range(scale))
        if ax is axes.ravel()[0]:
            ax.set_ylabel("stage phase-y lattice")
        else:
            ax.set_yticklabels([])
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("frame count")
    savefig_academic(fig, output_path)


def plot_detector_bin_heatmaps(samples: pd.DataFrame, output_path: Path, scale: int) -> None:
    setup_academic_style()
    methods = list(samples["method"].drop_duplicates())
    fig, axes = plt.subplots(1, len(methods), figsize=(7.2, 2.55), squeeze=False)
    max_count = 1
    counts_by_method: dict[str, np.ndarray] = {}
    for method in methods:
        group = samples[samples["method"].eq(method)]
        counts = count_matrix(group["detector_bin_x"], group["detector_bin_y"], scale)
        counts_by_method[method] = counts
        max_count = max(max_count, int(counts.max()))

    for ax, method in zip(axes.ravel(), methods, strict=True):
        counts = counts_by_method[method]
        im = ax.imshow(counts, origin="lower", cmap=COLORMAPS["coverage"], vmin=0, vmax=max_count)
        for y_idx in range(scale):
            for x_idx in range(scale):
                value = int(counts[y_idx, x_idx])
                color = "black" if value > max_count * 0.55 else "white"
                ax.text(x_idx, y_idx, str(value), ha="center", va="center", fontsize=7, color=color)
        ax.set_title(method.replace("_", " "))
        ax.set_xlabel("detector phase-x bin")
        ax.set_xticks(range(scale))
        ax.set_yticks(range(scale))
        if ax is axes.ravel()[0]:
            ax.set_ylabel("detector phase-y bin")
        else:
            ax.set_yticklabels([])
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02)
    cbar.set_label("frame count")
    savefig_academic(fig, output_path)


def plot_stage_phase_scatter(samples: pd.DataFrame, output_path: Path, scale: int) -> None:
    setup_academic_style()
    methods = list(samples["method"].drop_duplicates())
    fig, axes = plt.subplots(1, len(methods), figsize=(7.2, 2.55), squeeze=False)
    grid = np.arange(scale) / scale
    gx, gy = np.meshgrid(grid, grid)

    for ax, method, color in zip(axes.ravel(), methods, METHOD_COLOR_LIST):
        group = samples[samples["method"].eq(method)]
        ax.scatter(gx.ravel(), gy.ravel(), marker="+", s=45, color="#333333", linewidth=0.8, label="command lattice")
        ax.scatter(
            group["stage_phase_x"],
            group["stage_phase_y"],
            s=9,
            alpha=0.72,
            color=color,
            edgecolor="none",
            label="frame phase",
        )
        ax.set_title(method.replace("_", " "))
        ax.set_xlabel("stage phase-x")
        ax.set_xlim(-0.04, 1.04)
        ax.set_ylim(-0.04, 1.04)
        ax.set_xticks(np.linspace(0.0, 1.0, 6))
        ax.set_yticks(np.linspace(0.0, 1.0, 6))
        ax.set_aspect("equal", adjustable="box")
        ax.grid(alpha=0.25)
        if ax is axes.ravel()[0]:
            ax.set_ylabel("stage phase-y")
            ax.legend(loc="upper right", fontsize=7)
        else:
            ax.set_yticklabels([])
    savefig_academic(fig, output_path)


def plot_deviation_histogram(samples: pd.DataFrame, output_path: Path) -> None:
    setup_academic_style()
    measured = samples[~samples["method"].eq("command_prior")].copy()
    fig, ax = plt.subplots(figsize=(3.5, 2.65))
    bins = np.linspace(0.0, max(0.05, float(measured["delta_stage_um_wrapped"].quantile(0.99))), 34)
    for method, color in zip(measured["method"].drop_duplicates(), METHOD_COLOR_LIST[1:], strict=False):
        group = measured[measured["method"].eq(method)]
        ax.hist(
            group["delta_stage_um_wrapped"],
            bins=bins,
            histtype="step",
            linewidth=1.4,
            color=color,
            label=method.replace("_", " "),
        )
    ax.set_xlabel("Wrapped deviation from command phase [um]")
    ax.set_ylabel("Frame count")
    ax.set_title("Measured Phase Deviation")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.25)
    savefig_academic(fig, output_path)


def decision_text(summary: pd.DataFrame, scale: int) -> str:
    command = summary[summary["method"].eq("command_prior")].iloc[0]
    refined = summary[summary["method"].eq("contour_refined")].iloc[0]
    detector_note = ""
    if int(refined["detector_bin_empty"]) > 0:
        detector_note = (
            f" Detector-axis 5x bins are sparse for contour_refined "
            f"({int(refined['detector_bin_occupied'])}/{scale * scale} occupied), so detector-bin occupancy "
            "should be treated as a high-magnification risk diagnostic."
        )
    if int(command["stage_lattice_occupied"]) == scale * scale and int(refined["stage_lattice_occupied"]) == scale * scale:
        return (
            f"PASS WITH CAVEATS: command and contour_refined both occupy all {scale * scale} "
            "stage-coordinate 5x lattice cells. "
            f"Contour_refined is within {float(refined['nearest_stage_lattice_um_p90']):.2f} um of the "
            f"nearest 2 um lattice node at P90, but it does not preserve command cell labels "
            f"(match rate {100.0 * float(refined['stage_lattice_match_rate']):.1f}%)."
            f"{detector_note} This supports phase diversity; it does not by itself prove 5x optical recoverability."
        )
    if int(command["stage_lattice_occupied"]) == scale * scale:
        return (
            f"RISK: command prior occupies all {scale * scale} stage-coordinate cells, but "
            f"contour_refined occupies only {int(refined['stage_lattice_occupied'])}/{scale * scale}. "
            "Measured alignment has degraded the nominal phase lattice."
        )
    return (
        f"FAIL: command prior itself occupies only {int(command['stage_lattice_occupied'])}/{scale * scale} "
        "stage-coordinate cells; the nominal 5x lattice assumption is not present in the clean input."
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-audit-csv", type=Path, default=PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--alignment-csv", type=Path, default=default_contour_alignment_csv(project_root_path=PROJECT_ROOT))
    parser.add_argument("--stage-config", type=Path, default=PROJECT_ROOT / "configs" / "stage_calibration.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "ep15_info_limit" / "m1_phase_structure")
    parser.add_argument("--scale", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.scale < 2:
        raise ValueError("--scale must be >= 2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_academic_style()

    stage = _load_stage_config(args.stage_config)
    metadata = load_main_session_metadata(args.frame_audit_csv)
    if len(metadata) != EXPECTED_CLEAN_SR_FRAMES:
        raise ValueError(f"Expected {EXPECTED_CLEAN_SR_FRAMES} clean SR frames; got {len(metadata)}")

    reference_file = _reference_file(args.alignment_csv)
    command_x_um, command_y_um, command_dx, command_dy = command_alignment_prior(
        metadata,
        reference_file,
        theta_deg=stage["theta_deg"],
        pixel_size_um=stage["pixel_size_um"],
    )

    ncc_shifts = load_alignment_shifts("ncc_init", metadata=metadata, alignment_csv=args.alignment_csv)
    refined_shifts = load_alignment_shifts("contour_refined", metadata=metadata, alignment_csv=args.alignment_csv)

    all_records: list[pd.DataFrame] = []
    method_inputs = [
        ("command_prior", command_dx, command_dy, command_x_um, command_y_um),
        (
            "ncc_init",
            ncc_shifts[:, 0],
            ncc_shifts[:, 1],
            *alignment_shift_to_stage_um(
                ncc_shifts[:, 0],
                ncc_shifts[:, 1],
                theta_deg=stage["theta_deg"],
                pixel_size_um=stage["pixel_size_um"],
            ),
        ),
        (
            "contour_refined",
            refined_shifts[:, 0],
            refined_shifts[:, 1],
            *alignment_shift_to_stage_um(
                refined_shifts[:, 0],
                refined_shifts[:, 1],
                theta_deg=stage["theta_deg"],
                pixel_size_um=stage["pixel_size_um"],
            ),
        ),
    ]
    for method, align_dx, align_dy, stage_x_um, stage_y_um in method_inputs:
        all_records.append(
            method_records(
                metadata,
                method=method,
                align_dx=np.asarray(align_dx, dtype=float),
                align_dy=np.asarray(align_dy, dtype=float),
                stage_x_um=np.asarray(stage_x_um, dtype=float),
                stage_y_um=np.asarray(stage_y_um, dtype=float),
                command_stage_x_um=command_x_um,
                command_stage_y_um=command_y_um,
                command_align_dx=command_dx,
                command_align_dy=command_dy,
                scale=args.scale,
                pixel_size_um=stage["pixel_size_um"],
            )
        )

    samples = pd.concat(all_records, ignore_index=True)
    summary = pd.DataFrame([summarize_method(group, args.scale) for _, group in samples.groupby("method", sort=False)])
    stage_counts, detector_counts = make_count_tables(samples, args.scale)

    samples.to_csv(args.output_dir / "m1_phase_samples.csv", index=False)
    summary.to_csv(args.output_dir / "m1_phase_summary.csv", index=False)
    stage_counts.to_csv(args.output_dir / "m1_stage_lattice_counts_5x.csv", index=False)
    detector_counts.to_csv(args.output_dir / "m1_detector_bin_counts_5x.csv", index=False)

    figures = {
        "stage_lattice_heatmap": args.output_dir / "m1_stage_lattice_occupancy_5x.png",
        "detector_bin_heatmap": args.output_dir / "m1_detector_bin_occupancy_5x.png",
        "stage_phase_scatter": args.output_dir / "m1_stage_phase_scatter_5x.png",
        "phase_deviation_histogram": args.output_dir / "m1_phase_deviation_histogram.png",
    }
    plot_stage_lattice_heatmaps(samples, figures["stage_lattice_heatmap"], args.scale)
    plot_detector_bin_heatmaps(samples, figures["detector_bin_heatmap"], args.scale)
    plot_stage_phase_scatter(samples, figures["stage_phase_scatter"], args.scale)
    plot_deviation_histogram(samples, figures["phase_deviation_histogram"])

    decision = decision_text(summary, args.scale)
    manifest = {
        "task": "EP15 M1 phase structure",
        "frame_audit_csv": _rel(args.frame_audit_csv),
        "alignment_csv": _rel(args.alignment_csv),
        "stage_config": _rel(args.stage_config),
        "output_dir": _rel(args.output_dir),
        "reference_file": reference_file,
        "n_clean_sr_frames": int(len(metadata)),
        "scale": int(args.scale),
        "theta_deg": stage["theta_deg"],
        "pixel_size_um": stage["pixel_size_um"],
        "shift_convention": "EP05 align_dx/align_dy move each frame into the reference coordinate system; command prior uses (-dx_model, dy_model).",
        "decision": decision,
        "summary": summary.to_dict(orient="records"),
        "figures": {name: _rel(path) for name, path in figures.items()},
        "outputs": {
            "samples_csv": _rel(args.output_dir / "m1_phase_samples.csv"),
            "summary_csv": _rel(args.output_dir / "m1_phase_summary.csv"),
            "stage_lattice_counts_csv": _rel(args.output_dir / "m1_stage_lattice_counts_5x.csv"),
            "detector_bin_counts_csv": _rel(args.output_dir / "m1_detector_bin_counts_5x.csv"),
        },
        "interpretation_boundary": (
            "Full 5x phase occupancy is sampling-geometry evidence only. M2/FRC, M3 sigma arbitration, "
            "and M4 deconvolution anchor are still required before claiming recoverable 4x/5x structure."
        ),
    }
    (args.output_dir / "m1_phase_structure_summary.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(summary.round(4).to_string(index=False))
    print(f"\n{decision}")
    print(f"Saved M1 outputs to {_rel(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
