"""Route C: short-budget joint MAP-TV PSF cross-validation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thermal_core.plotting import FIGURE_SIZES, METHOD_COLORS, savefig_academic, setup_academic_style

from .data import DEFAULT_ALIGNMENT_CSV, DEFAULT_DATA_DIR, DEFAULT_FRAME_AUDIT_CSV, load_main_inputs
from .forward_sweep import residuals_for_sigma
from .utils import (
    OUTPUT_DIR,
    default_workers,
    deterministic_split,
    ensure_dir,
    parabolic_minimum,
    relative,
    write_json,
)

from .utils import bootstrap_project_paths

bootstrap_project_paths()

from map_tv.map_tv import reconstruct_map_tv  # noqa: E402
from saa.saa import reconstruct_saa  # noqa: E402


@dataclass(frozen=True)
class JointSweepResult:
    """Outputs from route C."""

    sweep: pd.DataFrame
    summary: dict


def plot_joint_curve(sweep: pd.DataFrame, summary: dict, output_path: str | Path) -> Path:
    """Plot route-C hold-out residual curve."""

    setup_academic_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["one_half_col"])
    ax.plot(
        sweep["sigma_lr_px"],
        sweep["holdout_mse"],
        marker="o",
        color=METHOD_COLORS["primary"],
        label="hold-out residual",
    )
    sigma = float(summary["sigma_joint_lr_px"])
    ax.axvline(sigma, color=METHOD_COLORS["accent_1"], linestyle="--", linewidth=1.0, label=f"best={sigma:.3f} px")
    ax.set_xlabel("Gaussian PSF sigma (LR px)")
    ax.set_ylabel("Validation forward residual MSE")
    ax.set_title("EP09 Route C Joint MAP-TV Sweep")
    ax.legend()
    return savefig_academic(fig, output_path)


def run_joint_sweep(
    *,
    output_dir: str | Path = OUTPUT_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    frame_audit_csv: str | Path = DEFAULT_FRAME_AUDIT_CSV,
    alignment_csv: str | Path = DEFAULT_ALIGNMENT_CSV,
    alignment_method: str = "contour_refined",
    highpass_sigma: float = 5.0,
    sigmas: list[float] | np.ndarray | None = None,
    val_stride: int = 5,
    max_train: int = 48,
    max_val: int = 32,
    max_iter: int = 20,
    lambda_tv: float = 0.01,
    step_size: float = 0.25,
    tol: float = 1e-4,
    tv_inner_iter: int = 15,
    crop_margin: int = 8,
    workers: int = default_workers(),
    seed: int = 2909,
    save_best_hr: bool = True,
    limit: int | None = None,
    scale: int = 2,
) -> JointSweepResult:
    """Run a deterministic short-budget MAP-TV hold-out sweep."""

    del seed
    start = time.perf_counter()
    out_dir = ensure_dir(output_dir)
    inputs = load_main_inputs(
        data_dir=data_dir,
        frame_audit_csv=frame_audit_csv,
        alignment_csv=alignment_csv,
        alignment_method=alignment_method,
        highpass_sigma=highpass_sigma,
        workers=workers,
        limit=limit,
    )
    train_idx, val_idx = deterministic_split(
        len(inputs.metadata),
        val_stride=val_stride,
        max_train=max_train,
        max_val=max_val,
    )
    grid = np.asarray(sigmas if sigmas is not None else np.arange(0.10, 0.6001, 0.05), dtype=float)
    train_frames = inputs.frames_highpass[train_idx]
    train_shifts = inputs.shifts[train_idx]
    val_frames = inputs.frames_highpass[val_idx]
    val_shifts = inputs.shifts[val_idx]
    initial = reconstruct_saa(train_frames, train_shifts, scale=scale, workers=workers).astype(np.float64, copy=False)

    rows: list[dict[str, float | int | bool]] = []
    best_hr: np.ndarray | None = None
    best_score = float("inf")
    for sigma in grid:
        hr, records = reconstruct_map_tv(
            train_frames,
            train_shifts,
            initial=initial,
            lambda_tv=lambda_tv,
            max_iter=max_iter,
            step_size=step_size,
            psf_sigma=float(sigma),
            scale=scale,
            workers=workers,
            tol=tol,
            tv_inner_iter=tv_inner_iter,
            use_fista=False,
            return_dataframe=True,
        )
        holdout_scores = residuals_for_sigma(
            float(sigma),
            hr,
            val_frames,
            val_shifts,
            indices=np.arange(len(val_frames)),
            scale=scale,
            crop_margin=crop_margin,
        )
        train_scores = residuals_for_sigma(
            float(sigma),
            hr,
            train_frames,
            train_shifts,
            indices=np.arange(len(train_frames)),
            scale=scale,
            crop_margin=crop_margin,
        )
        last = records.iloc[-1].to_dict() if len(records) else {}
        holdout_mse = float(np.mean(holdout_scores))
        rows.append(
            {
                "sigma_lr_px": float(sigma),
                "holdout_mse": holdout_mse,
                "train_mse": float(np.mean(train_scores)),
                "holdout_median_mse": float(np.median(holdout_scores)),
                "train_median_mse": float(np.median(train_scores)),
                "n_train": int(len(train_idx)),
                "n_val": int(len(val_idx)),
                "iterations": int(last.get("iteration", len(records))),
                "final_residual_mse_internal": float(last.get("residual_mse", np.nan)),
                "final_tv": float(last.get("tv", np.nan)),
                "final_relative_update": float(last.get("relative_update", np.nan)),
                "stopped": bool(last.get("stopped", False)),
            }
        )
        if holdout_mse < best_score:
            best_score = holdout_mse
            best_hr = np.asarray(hr, dtype=np.float32)

    sweep = pd.DataFrame(rows).sort_values("sigma_lr_px").reset_index(drop=True)
    sigmas_arr = sweep["sigma_lr_px"].to_numpy(dtype=float)
    values = sweep["holdout_mse"].to_numpy(dtype=float)
    sigma_joint, parabolic_ok = parabolic_minimum(sigmas_arr, values)
    grid_min = float(sigmas_arr[int(np.argmin(values))])
    edge = bool(np.argmin(values) in {0, len(values) - 1})
    summary = {
        "route": "C_joint_map_tv_holdout",
        "sigma_joint_lr_px": sigma_joint,
        "sigma_joint_grid_min_lr_px": grid_min,
        "sigma_joint_hr_px_at_2x": sigma_joint * scale,
        "parabolic_ok": bool(parabolic_ok),
        "minimum_at_grid_edge": edge,
        "best_holdout_mse": best_score,
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "train_indices": train_idx.tolist(),
        "val_indices": val_idx.tolist(),
        "lambda_tv": float(lambda_tv),
        "max_iter": int(max_iter),
        "step_size": float(step_size),
        "tv_inner_iter": int(tv_inner_iter),
        "crop_margin_lr_px": int(crop_margin),
        "highpass_sigma_lr_px": float(highpass_sigma),
        "alignment_method": alignment_method,
        "elapsed_sec": float(time.perf_counter() - start),
    }
    sweep.to_csv(out_dir / "joint_sigma_sweep.csv", index=False)
    write_json(out_dir / "sigma_joint.json", summary)
    if save_best_hr and best_hr is not None:
        np.save(out_dir / "joint_best_hr_highpass.npy", best_hr.astype(np.float32, copy=False))
        summary["best_hr_output"] = relative(out_dir / "joint_best_hr_highpass.npy")
        write_json(out_dir / "sigma_joint.json", summary)
    plot_joint_curve(sweep, summary, out_dir / "joint_sigma_curve.png")
    return JointSweepResult(sweep=sweep, summary=summary)
