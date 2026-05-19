"""Route A: forward-model PSF sigma residual sweep."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import minimize_scalar

from thermal_core.plotting import FIGURE_SIZES, METHOD_COLORS, savefig_academic, setup_academic_style

from .data import DEFAULT_ALIGNMENT_CSV, DEFAULT_DATA_DIR, DEFAULT_FRAME_AUDIT_CSV, load_ep06_hr, load_main_inputs
from .utils import (
    OUTPUT_DIR,
    bootstrap_curve_minima,
    crop_slices,
    default_workers,
    deterministic_split,
    ensure_dir,
    parabolic_minimum,
    relative,
    write_json,
)


@dataclass(frozen=True)
class ForwardSweepResult:
    """Outputs from route A."""

    coarse_curve: pd.DataFrame
    fine_curve: pd.DataFrame
    frame_scores: pd.DataFrame
    summary: dict


def _sigma_hr(psf_sigma_lr_px: float, scale: int) -> float:
    return max(0.0, float(psf_sigma_lr_px) * float(scale))


def _sample_reference_to_lr(blurred_hr: np.ndarray, shift: np.ndarray, *, scale: int) -> np.ndarray:
    h_hr, w_hr = blurred_hr.shape
    h_lr, w_lr = h_hr // scale, w_hr // scale
    dx, dy = np.asarray(shift, dtype=np.float64)
    yy = scale * (np.arange(h_lr, dtype=np.float64) + dy)
    xx = scale * (np.arange(w_lr, dtype=np.float64) + dx)
    coords = np.meshgrid(yy, xx, indexing="ij")
    return ndimage.map_coordinates(blurred_hr, coords, order=1, mode="constant", cval=0.0, prefilter=False)


def _blur_hr(hr_image: np.ndarray, sigma_lr_px: float, *, scale: int, mode: str) -> np.ndarray:
    sigma = _sigma_hr(sigma_lr_px, scale)
    image = np.asarray(hr_image, dtype=np.float64)
    if sigma <= 0:
        return image
    return ndimage.gaussian_filter(image, sigma=sigma, mode=mode, cval=0.0)


def residuals_for_sigma(
    sigma_lr_px: float,
    hr_image: np.ndarray,
    lr_observations: np.ndarray,
    shifts: np.ndarray,
    *,
    indices: Iterable[int] | None = None,
    scale: int = 2,
    mode: str = "constant",
    crop_margin: int = 8,
) -> np.ndarray:
    """Return one cropped MSE residual per selected frame for a candidate sigma."""

    obs = np.asarray(lr_observations, dtype=np.float32)
    selected = np.arange(obs.shape[0], dtype=int) if indices is None else np.asarray(list(indices), dtype=int)
    sl_y, sl_x = crop_slices(tuple(obs.shape[1:]), crop_margin)
    blurred = _blur_hr(hr_image, sigma_lr_px, scale=scale, mode=mode)
    scores = np.empty(len(selected), dtype=np.float64)
    for out_idx, frame_idx in enumerate(selected):
        pred = _sample_reference_to_lr(blurred, shifts[int(frame_idx)], scale=scale)
        residual = pred[sl_y, sl_x] - obs[int(frame_idx), sl_y, sl_x]
        scores[out_idx] = float(np.mean(residual * residual))
    return scores


def _score_grid(
    sigmas: np.ndarray,
    hr_image: np.ndarray,
    lr_observations: np.ndarray,
    shifts: np.ndarray,
    *,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    scale: int,
    mode: str,
    crop_margin: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, float | int | str]] = []
    frame_rows: list[dict[str, float | int | str]] = []
    split_map = {"train": train_indices, "val": val_indices}
    for sigma in sigmas:
        for split, indices in split_map.items():
            scores = residuals_for_sigma(
                float(sigma),
                hr_image,
                lr_observations,
                shifts,
                indices=indices,
                scale=scale,
                mode=mode,
                crop_margin=crop_margin,
            )
            rows.append(
                {
                    "sigma_lr_px": float(sigma),
                    "split": split,
                    "n_frames": int(len(scores)),
                    "mean_mse": float(np.mean(scores)),
                    "median_mse": float(np.median(scores)),
                    "std_mse": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
                    "p95_mse": float(np.percentile(scores, 95)),
                }
            )
            for frame_idx, mse in zip(indices, scores, strict=True):
                frame_rows.append(
                    {
                        "sigma_lr_px": float(sigma),
                        "split": split,
                        "frame_index": int(frame_idx),
                        "mse": float(mse),
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(frame_rows)


def _curve_for_split(curve: pd.DataFrame, split: str) -> tuple[np.ndarray, np.ndarray]:
    subset = curve[curve["split"].eq(split)].sort_values("sigma_lr_px")
    return subset["sigma_lr_px"].to_numpy(dtype=float), subset["mean_mse"].to_numpy(dtype=float)


def _single_minimum_diagnostic(sigmas: np.ndarray, values: np.ndarray) -> dict[str, float | bool]:
    idx = int(np.argmin(values))
    left_ok = bool(idx == 0 or np.all(np.diff(values[: idx + 1]) <= 1e-10))
    right_ok = bool(idx == len(values) - 1 or np.all(np.diff(values[idx:]) >= -1e-10))
    edge = bool(idx == 0 or idx == len(values) - 1)
    edge_min = min(float(values[0]), float(values[-1]))
    depth = float((edge_min - values[idx]) / max(edge_min, 1e-12))
    return {
        "grid_min_sigma_lr_px": float(sigmas[idx]),
        "grid_min_mse": float(values[idx]),
        "minimum_at_grid_edge": edge,
        "monotone_to_minimum": bool(left_ok and right_ok),
        "relative_depth_vs_best_edge": depth,
    }


def plot_forward_curve(curve: pd.DataFrame, summary: dict, output_path: str | Path) -> Path:
    """Plot route-A residual curves."""

    setup_academic_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["one_half_col"])
    for split, color in [("train", METHOD_COLORS["primary"]), ("val", METHOD_COLORS["secondary"])]:
        subset = curve[curve["split"].eq(split)].sort_values("sigma_lr_px")
        ax.plot(subset["sigma_lr_px"], subset["mean_mse"], marker="o", color=color, label=split)
    sigma = float(summary["sigma_forward_lr_px"])
    ax.axvline(sigma, color=METHOD_COLORS["accent_1"], linestyle="--", linewidth=1.0, label=f"best={sigma:.3f} px")
    ci = summary.get("ci95_lr_px")
    if isinstance(ci, list) and len(ci) == 2 and np.all(np.isfinite(ci)):
        ax.axvspan(float(ci[0]), float(ci[1]), color=METHOD_COLORS["accent_1"], alpha=0.12, linewidth=0)
    ax.set_xlabel("Gaussian PSF sigma (LR px)")
    ax.set_ylabel("Forward residual MSE")
    ax.set_title("EP09 Route A Forward Residual")
    ax.legend()
    return savefig_academic(fig, output_path)


def run_forward_sweep(
    *,
    output_dir: str | Path = OUTPUT_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    frame_audit_csv: str | Path = DEFAULT_FRAME_AUDIT_CSV,
    alignment_csv: str | Path = DEFAULT_ALIGNMENT_CSV,
    alignment_method: str = "contour_refined",
    hr_path: str | Path | None = None,
    highpass_sigma: float = 5.0,
    coarse_min: float = 0.10,
    coarse_max: float = 0.60,
    coarse_step: float = 0.02,
    fine_half_width: float = 0.05,
    fine_step: float = 0.005,
    val_stride: int = 5,
    crop_margin: int = 8,
    bootstrap: int = 300,
    seed: int = 909,
    workers: int = default_workers(),
    limit: int | None = None,
    scale: int = 2,
    mode: str = "constant",
) -> ForwardSweepResult:
    """Run the primary forward-model residual sigma sweep."""

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
    hr_image, resolved_hr = load_ep06_hr(hr_path)
    expected = (inputs.frames_highpass.shape[1] * scale, inputs.frames_highpass.shape[2] * scale)
    if tuple(hr_image.shape) != expected:
        raise ValueError(f"HR image shape {hr_image.shape} is inconsistent with expected {expected}")

    train_idx, val_idx = deterministic_split(len(inputs.metadata), val_stride=val_stride)
    coarse_sigmas = np.round(np.arange(coarse_min, coarse_max + 0.5 * coarse_step, coarse_step), 6)
    coarse_curve, coarse_frame_scores = _score_grid(
        coarse_sigmas,
        hr_image,
        inputs.frames_highpass,
        inputs.shifts,
        train_indices=train_idx,
        val_indices=val_idx,
        scale=scale,
        mode=mode,
        crop_margin=crop_margin,
    )
    train_sigmas, train_values = _curve_for_split(coarse_curve, "train")
    coarse_diag = _single_minimum_diagnostic(train_sigmas, train_values)
    best_coarse = float(coarse_diag["grid_min_sigma_lr_px"])
    fine_lo = max(coarse_min, best_coarse - fine_half_width)
    fine_hi = min(coarse_max, best_coarse + fine_half_width)
    fine_sigmas = np.round(np.arange(fine_lo, fine_hi + 0.5 * fine_step, fine_step), 6)
    fine_curve, fine_frame_scores = _score_grid(
        fine_sigmas,
        hr_image,
        inputs.frames_highpass,
        inputs.shifts,
        train_indices=train_idx,
        val_indices=val_idx,
        scale=scale,
        mode=mode,
        crop_margin=crop_margin,
    )
    fine_train_sigmas, fine_train_values = _curve_for_split(fine_curve, "train")
    fine_val_sigmas, fine_val_values = _curve_for_split(fine_curve, "val")
    sigma_parabolic, parabolic_ok = parabolic_minimum(fine_train_sigmas, fine_train_values)

    def objective(sigma: float) -> float:
        scores = residuals_for_sigma(
            float(sigma),
            hr_image,
            inputs.frames_highpass,
            inputs.shifts,
            indices=train_idx,
            scale=scale,
            mode=mode,
            crop_margin=crop_margin,
        )
        return float(np.mean(scores))

    bounded = minimize_scalar(objective, bounds=(fine_lo, fine_hi), method="bounded", options={"xatol": 1e-3})
    sigma_opt = float(bounded.x if bounded.success else sigma_parabolic)
    sigma_opt = float(np.clip(sigma_opt, fine_lo, fine_hi))
    val_opt, val_parabolic_ok = parabolic_minimum(fine_val_sigmas, fine_val_values)

    train_matrix = (
        fine_frame_scores[fine_frame_scores["split"].eq("train")]
        .pivot(index="sigma_lr_px", columns="frame_index", values="mse")
        .sort_index()
        .to_numpy(dtype=float)
    )
    boot = bootstrap_curve_minima(fine_sigmas, train_matrix, n_bootstrap=bootstrap, seed=seed)
    if len(boot) >= 20:
        ci95 = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))]
    else:
        ci95 = [float("nan"), float("nan")]

    fine_diag = _single_minimum_diagnostic(fine_train_sigmas, fine_train_values)
    summary = {
        "route": "A_forward_residual",
        "sigma_forward_lr_px": sigma_opt,
        "sigma_forward_hr_px_at_2x": sigma_opt * scale,
        "ci95_lr_px": ci95,
        "bootstrap_n": int(len(boot)),
        "train_grid_min_sigma_lr_px": float(fine_train_sigmas[int(np.argmin(fine_train_values))]),
        "train_parabolic_sigma_lr_px": sigma_parabolic,
        "train_parabolic_ok": bool(parabolic_ok),
        "val_parabolic_sigma_lr_px": val_opt,
        "val_parabolic_ok": bool(val_parabolic_ok),
        "train_val_abs_delta_lr_px": float(abs(sigma_opt - val_opt)),
        "coarse_min_sigma_lr_px": float(best_coarse),
        "coarse_diagnostic": coarse_diag,
        "fine_diagnostic": fine_diag,
        "optimizer_success": bool(bounded.success),
        "optimizer_message": str(bounded.message),
        "n_frames": int(len(inputs.metadata)),
        "n_train": int(len(train_idx)),
        "n_val": int(len(val_idx)),
        "crop_margin_lr_px": int(crop_margin),
        "highpass_sigma_lr_px": float(highpass_sigma),
        "alignment_method": alignment_method,
        "hr_input": relative(resolved_hr),
        "elapsed_sec": float(time.perf_counter() - start),
    }

    coarse_curve.to_csv(out_dir / "forward_residual_sweep.csv", index=False)
    fine_curve.to_csv(out_dir / "forward_residual_fine_sweep.csv", index=False)
    pd.concat([coarse_frame_scores.assign(stage="coarse"), fine_frame_scores.assign(stage="fine")], ignore_index=True).to_csv(
        out_dir / "forward_residual_frame_scores.csv",
        index=False,
    )
    write_json(out_dir / "sigma_forward.json", summary)
    plot_forward_curve(fine_curve, summary, out_dir / "forward_residual_curve.png")
    return ForwardSweepResult(
        coarse_curve=coarse_curve,
        fine_curve=fine_curve,
        frame_scores=fine_frame_scores,
        summary=summary,
    )
