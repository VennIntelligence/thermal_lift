#!/usr/bin/env python3
"""Run EP06 alignment-strategy ablation with 2x SAA metrics.

The experiment keeps the SR method fixed to SAA and changes only the alignment
shift source. It reports split-half stability, gradient/artifact proxies,
difference to the default contour-refined reconstruction, and phase-bin
coverage. The default scope is the central 360x360 LR ROI used by the EP05
alignment checks; pass ``--roi-size-lr 0`` for full-frame evaluation.
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bootstrap_paths() -> Path:
    script_path = Path(__file__).resolve()
    root = script_path.parent
    while not (root / "AGENTS.md").exists() and root != root.parent:
        root = root.parent
    for path in (root / "algos" / "ep06_sr_poc" / "src", root / "core" / "src"):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return root


PROJECT_ROOT = bootstrap_paths()

from common.alignment import load_alignment_shifts, load_quality_weights  # noqa: E402
from common.data_loader import highpass_preprocess, load_main_session_frames  # noqa: E402
from common.metrics import artifact_score, gradient_magnitude, split_half_consistency  # noqa: E402
from saa.saa import reconstruct_saa  # noqa: E402
from thermal_core.plotting import (  # noqa: E402
    COLORMAPS,
    FIGURE_SIZES,
    METHOD_COLOR_LIST,
    savefig_academic,
    setup_academic_style,
)


@dataclass(frozen=True)
class StrategySpec:
    strategy: str
    label: str
    method: str
    input_kind: str
    input_path: Path
    optional: bool = False


def default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def path_arg(value: Path) -> str:
    try:
        return str(value.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(value.resolve())


def json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return path_arg(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def finite_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(aa) & np.isfinite(bb)
    if int(valid.sum()) < 2:
        return float("nan")
    x = aa[valid].ravel()
    y = bb[valid].ravel()
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def nrmse_to_reference(reference: np.ndarray, estimate: np.ndarray) -> tuple[float, float]:
    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    valid = np.isfinite(ref) & np.isfinite(est)
    if not np.any(valid):
        return float("nan"), float("nan")
    diff = est[valid] - ref[valid]
    rmse = float(np.sqrt(np.mean(diff * diff)))
    denom = float(np.std(ref[valid]))
    return rmse, rmse / max(denom, 1e-12)


def weighted_average(frames: np.ndarray, weights: np.ndarray | None) -> np.ndarray:
    arr = np.asarray(frames, dtype=np.float32)
    if weights is None:
        return np.nanmean(arr, axis=0).astype(np.float32, copy=False)
    w = np.asarray(weights, dtype=np.float64)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    if float(w.sum()) <= 0.0:
        return np.nanmean(arr, axis=0).astype(np.float32, copy=False)
    finite = np.isfinite(arr)
    numer = np.nansum(np.where(finite, arr, 0.0) * w[:, None, None], axis=0)
    denom = np.sum(finite * w[:, None, None], axis=0)
    return np.divide(numer, denom, out=np.zeros_like(numer, dtype=np.float64), where=denom > 0).astype(np.float32)


def center_crop_stack(frames: np.ndarray, roi_size_lr: int) -> tuple[np.ndarray, dict[str, int | str]]:
    if roi_size_lr <= 0:
        return frames, {
            "mode": "full_frame",
            "r0": 0,
            "r1": int(frames.shape[1]),
            "c0": 0,
            "c1": int(frames.shape[2]),
        }
    rows, cols = frames.shape[1:]
    size = min(int(roi_size_lr), rows, cols)
    r0 = max(0, (rows - size) // 2)
    c0 = max(0, (cols - size) // 2)
    return frames[:, r0 : r0 + size, c0 : c0 + size], {
        "mode": "center_crop",
        "r0": int(r0),
        "r1": int(r0 + size),
        "c0": int(c0),
        "c1": int(c0 + size),
    }


def phase_occupancy(shifts: np.ndarray, scale: int) -> tuple[dict[str, Any], np.ndarray]:
    dx = np.asarray(shifts[:, 0], dtype=float)
    dy = np.asarray(shifts[:, 1], dtype=float)
    fx = np.mod(dx, 1.0)
    fy = np.mod(dy, 1.0)
    bin_x = np.clip(np.floor(fx * scale).astype(int), 0, scale - 1)
    bin_y = np.clip(np.floor(fy * scale).astype(int), 0, scale - 1)
    counts = np.zeros((scale, scale), dtype=int)
    np.add.at(counts, (bin_y, bin_x), 1)
    flat = counts.ravel()
    n = int(flat.sum())
    occupied = int(np.count_nonzero(flat))
    probs = flat[flat > 0] / n if n else np.array([], dtype=float)
    entropy = float(-(probs * np.log2(probs)).sum() / np.log2(scale * scale)) if probs.size else 0.0
    summary = {
        "scale": int(scale),
        "n_frames": n,
        "occupied_bins": occupied,
        "bad_bins": int(scale * scale - occupied),
        "total_bins": int(scale * scale),
        "min_count": int(flat.min()) if flat.size else 0,
        "max_count": int(flat.max()) if flat.size else 0,
        "expected_count": float(n / (scale * scale)) if scale > 0 else 0.0,
        "entropy_fraction": entropy,
    }
    return summary, counts


def build_strategy_specs(args: argparse.Namespace) -> list[StrategySpec]:
    specs = [
        StrategySpec(
            "default_contour_refined",
            "Default contour refined",
            "contour_refined",
            "alignment_csv",
            args.default_alignment_csv,
            optional=False,
        ),
        StrategySpec(
            "ncc_init",
            "NCC init",
            "ncc_init",
            "alignment_csv",
            args.default_alignment_csv,
            optional=False,
        ),
        StrategySpec(
            "filename_affine_fit",
            "Filename affine fit",
            "filename_affine_fit",
            "alignment_scores_path",
            args.affine_scores_csv,
            optional=False,
        ),
    ]
    if args.tuned_alignment_csv is not None:
        specs.insert(
            2,
            StrategySpec(
                "tuned_contour_refined",
                "Tuned contour refined",
                "contour_refined",
                "alignment_csv",
                args.tuned_alignment_csv,
                optional=True,
            ),
        )
    return specs


def load_strategy_inputs(
    spec: StrategySpec,
    metadata: pd.DataFrame,
    weight_mode: str,
) -> tuple[np.ndarray, np.ndarray | None, dict[str, Any]]:
    if not spec.input_path.exists():
        if spec.optional:
            raise FileNotFoundError(f"Optional input not found: {spec.input_path}")
        raise FileNotFoundError(f"Required input not found: {spec.input_path}")

    kwargs: dict[str, Any]
    if spec.input_kind == "alignment_scores_path":
        kwargs = {"alignment_scores_path": spec.input_path}
    else:
        kwargs = {"alignment_csv": spec.input_path}

    shifts = load_alignment_shifts(
        spec.method,
        metadata=metadata,
        strict=True,
        **kwargs,
    ).astype(np.float32, copy=False)
    if shifts.shape != (len(metadata), 2):
        raise ValueError(f"{spec.strategy} shifts have shape {shifts.shape}; expected {(len(metadata), 2)}")

    weights: np.ndarray | None
    if weight_mode == "quality":
        weights = load_quality_weights(
            spec.method,
            metadata=metadata,
            **kwargs,
        ).astype(np.float32, copy=False)
        if weights.shape != (len(metadata),):
            raise ValueError(f"{spec.strategy} weights have shape {weights.shape}; expected {(len(metadata),)}")
    elif weight_mode == "uniform":
        weights = None
    else:
        raise ValueError(f"Unsupported weight mode: {weight_mode}")

    finite = np.isfinite(shifts).all(axis=1)
    if not bool(finite.all()):
        raise ValueError(f"{spec.strategy} has {int((~finite).sum())} non-finite shifts")

    meta = {
        "strategy": spec.strategy,
        "label": spec.label,
        "method": spec.method,
        "input_kind": spec.input_kind,
        "input_alignment_csv": path_arg(spec.input_path),
        "status": "loaded",
        "n_frames": int(len(metadata)),
        "weight_mode": weight_mode,
        "weight_min": float(np.min(weights)) if weights is not None else 1.0,
        "weight_max": float(np.max(weights)) if weights is not None else 1.0,
        "weight_mean": float(np.mean(weights)) if weights is not None else 1.0,
    }
    return shifts, weights, meta


def summarize_reconstruction(
    strategy: str,
    label: str,
    shifts: np.ndarray,
    weights: np.ndarray | None,
    recon: np.ndarray,
    frames: np.ndarray,
    split_summary: dict[str, float],
    phase_summary_2x: dict[str, Any],
    input_csv: str,
) -> dict[str, Any]:
    grad = gradient_magnitude(recon)
    lr_avg = weighted_average(frames, weights)
    artifact = artifact_score(recon, lr_img=lr_avg, scale=2, return_components=True)
    shift_norm = np.hypot(shifts[:, 0], shifts[:, 1])
    row: dict[str, Any] = {
        "strategy": strategy,
        "label": label,
        "input_alignment_csv": input_csv,
        "n_frames": int(len(shifts)),
        "mean_gradient": float(np.nanmean(grad)),
        "p95_gradient": float(np.nanpercentile(grad, 95.0)),
        "artifact_score": float(artifact["score"]),
        "artifact_ringing": float(artifact["ringing"]),
        "artifact_blockiness": float(artifact["blockiness"]),
        "artifact_overshoot": float(artifact["overshoot"]),
        "shift_dx_span_px": float(np.nanmax(shifts[:, 0]) - np.nanmin(shifts[:, 0])),
        "shift_dy_span_px": float(np.nanmax(shifts[:, 1]) - np.nanmin(shifts[:, 1])),
        "shift_norm_median_px": float(np.nanmedian(shift_norm)),
        "shift_norm_p90_px": float(np.nanpercentile(shift_norm, 90.0)),
        "phase_2x_occupied_bins": int(phase_summary_2x["occupied_bins"]),
        "phase_2x_bad_bins": int(phase_summary_2x["bad_bins"]),
        "phase_2x_entropy_fraction": float(phase_summary_2x["entropy_fraction"]),
        "phase_2x_min_count": int(phase_summary_2x["min_count"]),
        "phase_2x_max_count": int(phase_summary_2x["max_count"]),
    }
    row.update(split_summary)
    return row


def split_summary(split_df: pd.DataFrame) -> dict[str, float]:
    return {
        "split_half_nrmse_median": float(split_df["nrmse"].median()),
        "split_half_nrmse_p10": float(split_df["nrmse"].quantile(0.10)),
        "split_half_nrmse_p90": float(split_df["nrmse"].quantile(0.90)),
        "split_half_rmse_median": float(split_df["rmse"].median()),
        "split_half_corr_median": float(split_df["corr"].median()),
        "split_half_psnr_db_median": float(split_df["psnr_db"].median()),
    }


def plot_split_half(metrics: pd.DataFrame, output_path: Path) -> None:
    setup_academic_style()
    labels = metrics["label"].tolist()
    x = np.arange(len(metrics))
    med = metrics["split_half_nrmse_median"].to_numpy(dtype=float)
    low = med - metrics["split_half_nrmse_p10"].to_numpy(dtype=float)
    high = metrics["split_half_nrmse_p90"].to_numpy(dtype=float) - med
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["one_half_col"])
    ax.bar(x, med, color=METHOD_COLOR_LIST[: len(metrics)], alpha=0.9)
    ax.errorbar(x, med, yerr=np.vstack([low, high]), fmt="none", ecolor="#333333", capsize=3, linewidth=0.9)
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Split-half NRMSE")
    ax.grid(axis="y", alpha=0.25)
    savefig_academic(fig, output_path)


def plot_gradient_artifact(metrics: pd.DataFrame, output_path: Path) -> None:
    setup_academic_style()
    labels = metrics["label"].tolist()
    x = np.arange(len(metrics))
    fig, axes = plt.subplots(1, 3, figsize=(8.0, 3.0))
    specs = [
        ("mean_gradient", "Mean gradient", "Mean |grad|"),
        ("p95_gradient", "P95 gradient", "P95 |grad|"),
        ("artifact_score", "Artifact score", "Lower is better"),
    ]
    for ax, (col, title, ylabel) in zip(axes, specs, strict=True):
        ax.bar(x, metrics[col].to_numpy(dtype=float), color=METHOD_COLOR_LIST[: len(metrics)], alpha=0.9)
        ax.set_xticks(x, labels, rotation=25, ha="right")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    savefig_academic(fig, output_path)


def plot_difference_to_default(metrics: pd.DataFrame, output_path: Path) -> None:
    setup_academic_style()
    labels = metrics["label"].tolist()
    x = np.arange(len(metrics))
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["double_col"])
    axes[0].bar(x, metrics["nrmse_to_default"].to_numpy(dtype=float), color=METHOD_COLOR_LIST[: len(metrics)], alpha=0.9)
    axes[0].set_title("Difference to default")
    axes[0].set_ylabel("NRMSE to default")
    axes[0].set_xticks(x, labels, rotation=25, ha="right")
    axes[0].grid(axis="y", alpha=0.25)

    axes[1].bar(x, metrics["p95_abs_diff_to_default"].to_numpy(dtype=float), color=METHOD_COLOR_LIST[: len(metrics)], alpha=0.9)
    axes[1].set_title("P95 absolute difference")
    axes[1].set_ylabel("Highpass units")
    axes[1].set_xticks(x, labels, rotation=25, ha="right")
    axes[1].grid(axis="y", alpha=0.25)
    savefig_academic(fig, output_path)


def plot_phase_coverage(phase_counts: pd.DataFrame, phase_summary: pd.DataFrame, output_path: Path) -> None:
    setup_academic_style()
    plot_summary = phase_summary[phase_summary["scale"].eq(2)].copy()
    plot_counts = phase_counts[phase_counts["scale"].eq(2)].copy()
    labels = plot_summary["label"].tolist()
    x = np.arange(len(labels))
    phase_order = [(0, 0), (0, 1), (1, 0), (1, 1)]
    phase_labels = [f"bin ({y},{x})" for y, x in phase_order]
    colors = METHOD_COLOR_LIST[:4]

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["double_col"])
    bottom = np.zeros(len(labels), dtype=float)
    indexed = plot_counts.set_index(["strategy", "phase_y_bin", "phase_x_bin"])
    for (phase_y, phase_x), phase_label, color in zip(phase_order, phase_labels, colors, strict=True):
        values = []
        for strategy in plot_summary["strategy"]:
            values.append(float(indexed.loc[(strategy, phase_y, phase_x), "count"]))
        values_arr = np.asarray(values, dtype=float)
        ax.bar(x, values_arr, bottom=bottom, color=color, edgecolor="white", linewidth=0.7, label=phase_label)
        for xi, base, val in zip(x, bottom, values_arr, strict=True):
            if val >= 12:
                ax.text(xi, base + val / 2.0, f"{int(val)}", ha="center", va="center", fontsize=7, color="white")
        bottom += values_arr

    for xi, row in enumerate(plot_summary.itertuples(index=False)):
        ax.text(
            xi,
            float(row.n_frames) + max(2.0, float(row.n_frames) * 0.025),
            f"occ {int(row.occupied_bins)}/{int(row.total_bins)}\nH={float(row.entropy_fraction):.2f}",
            ha="center",
            va="bottom",
            fontsize=7,
            color="#333333",
        )
    ax.set_xticks(x, labels, rotation=25, ha="right")
    ax.set_ylabel("Frame count")
    ax.set_ylim(0, max(float(plot_summary["n_frames"].max()) * 1.17, 1.0))
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=4)
    ax.grid(axis="y", alpha=0.2)
    savefig_academic(fig, output_path)


def plot_default_difference_panels(
    reconstructions: dict[str, np.ndarray],
    metrics: pd.DataFrame,
    output_path: Path,
) -> None:
    setup_academic_style()
    default = reconstructions["default_contour_refined"]
    others = [s for s in metrics["strategy"] if s != "default_contour_refined"]
    if not others:
        return

    # --- 3X center crop to reveal chip structure details, shifted slightly to the bottom-right ---
    zoom_factor = 3
    h, w = default.shape[:2]
    crop_h, crop_w = h // zoom_factor, w // zoom_factor
    shift_r = int(h * 0.06)  # Shift window down
    shift_c = int(w * 0.06)  # Shift window right
    r0 = (h - crop_h) // 2 + shift_r
    c0 = (w - crop_w) // 2 + shift_c
    r1 = r0 + crop_h
    c1 = c0 + crop_w

    ncols = len(others)
    fig, axes = plt.subplots(1, ncols, figsize=(3.5 * ncols, 3.6))
    axes_arr = np.atleast_1d(axes)
    diffs = [reconstructions[name][r0:r1, c0:c1] - default[r0:r1, c0:c1] for name in others]
    limit = float(np.nanpercentile(np.abs(np.concatenate([d.ravel() for d in diffs])), 99.0))
    limit = max(limit, 1e-6)
    for ax, strategy, diff in zip(axes_arr, others, diffs, strict=True):
        label = str(metrics.loc[metrics["strategy"].eq(strategy), "label"].iloc[0])
        im = ax.imshow(diff, cmap=COLORMAPS["residual_diff"], vmin=-limit, vmax=limit, interpolation="nearest")
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
        cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Highpass Δ (°C)")
    savefig_academic(fig, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "data_raw" / "infrared_avi")
    parser.add_argument("--frame-audit-csv", type=Path, default=PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--default-alignment-csv", type=Path, default=PROJECT_ROOT / "output" / "ep05_contour_alignment" / "contour_alignment_results.csv")
    parser.add_argument(
        "--tuned-alignment-csv",
        type=Path,
        default=None,
        help=(
            "Optional tuned contour alignment CSV. Must be an explicitly provided, validated 248-frame "
            "candidate, for example under output/ep05_alignment_tuning_study/ after a full-frame run."
        ),
    )
    parser.add_argument("--affine-scores-csv", type=Path, default=PROJECT_ROOT / "output" / "ep05_alignment_sr_capacity" / "alignment_method_holdout_scores.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "ep06_alignment_ablation")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--roi-size-lr", type=int, default=360, help="Central LR ROI size; 0 means full frame.")
    parser.add_argument("--highpass-sigma", type=float, default=5.0)
    parser.add_argument("--weight-mode", choices=["quality", "uniform"], default="quality")
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--seed", type=int, default=606)
    parser.add_argument("--save-npy", action="store_true", help="Also save per-strategy SAA reconstructions.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scale != 2:
        raise ValueError("EP06 alignment ablation is scoped to 2x SAA; keep --scale 2.")
    if args.n_splits < 1:
        raise ValueError("--n-splits must be >= 1")

    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_academic_style()

    frames, metadata = load_main_session_frames(
        args.data_dir,
        args.frame_audit_csv,
        workers=args.workers,
    )
    highpass_frames = highpass_preprocess(frames, sigma_bg=args.highpass_sigma, workers=args.workers)
    del frames
    highpass_frames, roi = center_crop_stack(highpass_frames, args.roi_size_lr)

    manifest_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    strategy_metrics: list[dict[str, Any]] = []
    split_tables: list[pd.DataFrame] = []
    phase_summary_rows: list[dict[str, Any]] = []
    phase_count_rows: list[dict[str, Any]] = []
    reconstructions: dict[str, np.ndarray] = {}

    for spec in build_strategy_specs(args):
        try:
            shifts, weights, manifest = load_strategy_inputs(spec, metadata, args.weight_mode)
        except FileNotFoundError as exc:
            if spec.optional:
                skipped = {
                    "strategy": spec.strategy,
                    "label": spec.label,
                    "method": spec.method,
                    "input_kind": spec.input_kind,
                    "input_alignment_csv": path_arg(spec.input_path),
                    "status": "skipped",
                    "reason": str(exc),
                }
                skipped_rows.append(skipped)
                manifest_rows.append(skipped)
                continue
            raise

        print(f"Running SAA alignment ablation: {spec.strategy}")
        recon = reconstruct_saa(
            highpass_frames,
            shifts,
            weights=weights,
            scale=args.scale,
            workers=args.workers,
        ).astype(np.float32, copy=False)
        reconstructions[spec.strategy] = recon
        if args.save_npy:
            np.save(args.output_dir / f"saa_{spec.strategy}.npy", recon)

        split_df = split_half_consistency(
            highpass_frames,
            shifts,
            reconstruct_saa,
            n_splits=args.n_splits,
            random_state=args.seed,
            weights=weights,
            scale=args.scale,
            workers=args.workers,
        )
        split_df.insert(0, "label", spec.label)
        split_df.insert(0, "strategy", spec.strategy)
        split_tables.append(split_df)

        phase2_summary: dict[str, Any] | None = None
        for scale in (2, 3, 4):
            summary, counts = phase_occupancy(shifts, scale)
            row = {"strategy": spec.strategy, "label": spec.label, **summary}
            phase_summary_rows.append(row)
            if scale == 2:
                phase2_summary = summary
            for y_bin in range(scale):
                for x_bin in range(scale):
                    phase_count_rows.append(
                        {
                            "strategy": spec.strategy,
                            "label": spec.label,
                            "scale": int(scale),
                            "phase_y_bin": int(y_bin),
                            "phase_x_bin": int(x_bin),
                            "count": int(counts[y_bin, x_bin]),
                            "bad_bin": bool(counts[y_bin, x_bin] == 0),
                            "expected_count": float(summary["expected_count"]),
                        }
                    )
        if phase2_summary is None:
            raise RuntimeError(f"No 2x phase summary was produced for {spec.strategy}")

        strategy_metrics.append(
            summarize_reconstruction(
                spec.strategy,
                spec.label,
                shifts,
                weights,
                recon,
                highpass_frames,
                split_summary(split_df),
                phase2_summary,
                manifest["input_alignment_csv"],
            )
        )
        manifest_rows.append(manifest)

    if "default_contour_refined" not in reconstructions:
        raise RuntimeError("Default contour-refined strategy was not produced; cannot compute default deltas.")

    default = reconstructions["default_contour_refined"]
    for row in strategy_metrics:
        image = reconstructions[str(row["strategy"])]
        rmse, nrmse = nrmse_to_reference(default, image)
        diff = np.asarray(image, dtype=np.float64) - np.asarray(default, dtype=np.float64)
        valid = np.isfinite(diff)
        row["rmse_to_default"] = rmse
        row["nrmse_to_default"] = nrmse
        row["corr_to_default"] = finite_corr(default, image)
        row["mean_abs_diff_to_default"] = float(np.nanmean(np.abs(diff[valid]))) if np.any(valid) else float("nan")
        row["p95_abs_diff_to_default"] = float(np.nanpercentile(np.abs(diff[valid]), 95.0)) if np.any(valid) else float("nan")

    metrics = pd.DataFrame(strategy_metrics)
    split_all = pd.concat(split_tables, ignore_index=True) if split_tables else pd.DataFrame()
    phase_summary = pd.DataFrame(phase_summary_rows)
    phase_counts = pd.DataFrame(phase_count_rows)
    manifest = pd.DataFrame(manifest_rows)

    output_files: list[str] = []
    for name, df in [
        ("strategy_metrics.csv", metrics),
        ("split_half_metrics.csv", split_all),
        ("phase_coverage.csv", phase_summary),
        ("phase_bin_counts.csv", phase_counts),
        ("alignment_inputs.csv", manifest),
    ]:
        path = args.output_dir / name
        df.to_csv(path, index=False)
        output_files.append(path_arg(path))

    plot_specs = [
        ("strategy_split_half_nrmse.png", lambda path: plot_split_half(metrics, path)),
        ("strategy_gradient_artifact.png", lambda path: plot_gradient_artifact(metrics, path)),
        ("difference_to_default.png", lambda path: plot_difference_to_default(metrics, path)),
        ("phase_coverage_2x.png", lambda path: plot_phase_coverage(phase_counts, phase_summary, path)),
        ("difference_to_default_panels.png", lambda path: plot_default_difference_panels(reconstructions, metrics, path)),
    ]
    for name, plotter in plot_specs:
        path = args.output_dir / name
        plotter(path)
        output_files.append(path_arg(path))

    summary_path = args.output_dir / "alignment_ablation_summary.json"
    output_files_with_summary = [*output_files, path_arg(summary_path)]
    summary = {
        "created_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "elapsed_sec": float(time.perf_counter() - start),
        "command": " ".join(shlex.quote(part) for part in [sys.executable, *sys.argv]),
        "cwd": path_arg(Path.cwd()),
        "script": path_arg(Path(__file__).resolve()),
        "parameters": {key: json_ready(value) for key, value in vars(args).items()},
        "inputs": {
            "data_dir": path_arg(args.data_dir),
            "frame_audit_csv": path_arg(args.frame_audit_csv),
            "roi": roi,
            "highpass_sigma": float(args.highpass_sigma),
            "shift_convention": "EP05 shifts move each LR frame into the reference coordinate system; SAA uses positive shifts directly.",
        },
        "alignment_inputs": manifest.to_dict(orient="records"),
        "skipped_strategies": skipped_rows,
        "strategy_metrics": metrics.to_dict(orient="records"),
        "outputs": output_files_with_summary,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=json_ready), encoding="utf-8")
    output_files = output_files_with_summary

    print("\nEP06 alignment ablation summary")
    print(
        metrics[
            [
                "strategy",
                "split_half_nrmse_median",
                "mean_gradient",
                "p95_gradient",
                "artifact_score",
                "nrmse_to_default",
                "phase_2x_occupied_bins",
                "phase_2x_entropy_fraction",
            ]
        ].to_string(index=False)
    )
    print(f"\nSaved {len(output_files)} files to {path_arg(args.output_dir)}")


if __name__ == "__main__":
    main()
