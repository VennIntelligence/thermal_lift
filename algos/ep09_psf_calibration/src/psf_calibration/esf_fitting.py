"""Route B: one-dimensional ESF fitting on EP04 contour anchors."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.optimize import curve_fit
from scipy.special import erf

from thermal_core.plotting import FIGURE_SIZES, METHOD_COLORS, savefig_academic, setup_academic_style

from .data import (
    DEFAULT_ALIGNMENT_CSV,
    DEFAULT_CONTOUR_SEGMENTS_CSV,
    DEFAULT_DATA_DIR,
    DEFAULT_FRAME_AUDIT_CSV,
    load_reference_frame,
    load_segments,
)
from .utils import OUTPUT_DIR, default_workers, ensure_dir, relative, write_json


@dataclass(frozen=True)
class EsfFitResult:
    """Outputs from route B."""

    fits: pd.DataFrame
    summary: dict


def erf_model(x: np.ndarray, amplitude: float, center: float, sigma: float, baseline: float) -> np.ndarray:
    """Gaussian-convolved step edge model."""

    sigma = np.maximum(float(sigma), 1e-6)
    return baseline + amplitude * 0.5 * (1.0 + erf((x - center) / (sigma * np.sqrt(2.0))))


def extract_normal_profile(
    image: np.ndarray,
    *,
    x_px: float,
    y_px: float,
    nx: float,
    ny: float,
    tx: float,
    ty: float,
    half_width: float = 10.0,
    step: float = 0.25,
    tangent_half_width: int = 2,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample and tangent-average a normal edge profile from a temperature image."""

    x_coords = np.arange(-float(half_width), float(half_width) + 0.5 * float(step), float(step), dtype=float)
    tangent_offsets = np.arange(-int(tangent_half_width), int(tangent_half_width) + 1, dtype=float)
    profiles = []
    for tangent_offset in tangent_offsets:
        cols = float(x_px) + x_coords * float(nx) + tangent_offset * float(tx)
        rows = float(y_px) + x_coords * float(ny) + tangent_offset * float(ty)
        values = ndimage.map_coordinates(
            np.asarray(image, dtype=np.float64),
            [rows, cols],
            order=1,
            mode="nearest",
            prefilter=False,
        )
        profiles.append(values)
    profile = np.nanmean(np.vstack(profiles), axis=0)
    return x_coords, profile


def fit_esf_profile(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Fit the ESF model and return ``(params, covariance, r2, rmse)``."""

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    left = float(np.median(y_arr[: max(3, len(y_arr) // 5)]))
    right = float(np.median(y_arr[-max(3, len(y_arr) // 5) :]))
    amplitude0 = right - left
    if abs(amplitude0) < 1e-3:
        amplitude0 = float(y_arr[-1] - y_arr[0])
    baseline0 = left
    data_min = float(np.min(y_arr))
    data_max = float(np.max(y_arr))
    data_range = max(data_max - data_min, 1e-3)
    bounds = (
        [-10.0, float(x_arr.min()) * 0.75, 0.05, data_min - 2.0 * data_range],
        [10.0, float(x_arr.max()) * 0.75, 5.0, data_max + 2.0 * data_range],
    )
    p0 = [float(np.clip(amplitude0, -10.0, 10.0)), 0.0, 0.7, baseline0]
    params, cov = curve_fit(erf_model, x_arr, y_arr, p0=p0, bounds=bounds, maxfev=20000)
    pred = erf_model(x_arr, *params)
    residual = y_arr - pred
    ss_res = float(np.sum(residual * residual))
    ss_tot = float(np.sum((y_arr - np.mean(y_arr)) ** 2))
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    rmse = float(np.sqrt(np.mean(residual * residual)))
    return params, cov, float(r2), rmse


def _plot_fit(x: np.ndarray, y: np.ndarray, params: np.ndarray, row: pd.Series, output_path: Path) -> Path:
    setup_academic_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_col"])
    ax.plot(x, y, "o", color=METHOD_COLORS["primary"], markersize=2.5, label="profile")
    dense_x = np.linspace(float(np.min(x)), float(np.max(x)), 400)
    ax.plot(dense_x, erf_model(dense_x, *params), color=METHOD_COLORS["accent_1"], label="erf fit")
    ax.axvline(float(params[1]), color="#666666", linestyle="--", linewidth=0.8)
    ax.set_xlabel("Normal distance (LR px)")
    ax.set_ylabel("Temperature (deg C)")
    ax.set_title(f"Segment {int(row.segment_id)} ESF")
    ax.legend()
    return savefig_academic(fig, output_path)


def plot_esf_histogram(fits: pd.DataFrame, summary: dict, output_path: str | Path) -> Path:
    """Plot valid ESF sigma distribution."""

    setup_academic_style()
    valid = fits[fits["valid"]].copy()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["one_half_col"])
    ax.hist(valid["sigma_lr_px"], bins=14, color=METHOD_COLORS["primary"], edgecolor="white", alpha=0.9)
    sigma = float(summary["sigma_esf_median_lr_px"])
    ax.axvline(sigma, color=METHOD_COLORS["accent_1"], linestyle="--", linewidth=1.0, label=f"median={sigma:.3f} px")
    ci = summary.get("bootstrap_ci95_lr_px")
    if isinstance(ci, list) and len(ci) == 2 and np.all(np.isfinite(ci)):
        ax.axvspan(float(ci[0]), float(ci[1]), color=METHOD_COLORS["accent_1"], alpha=0.12, linewidth=0)
    ax.set_xlabel("Gaussian ESF sigma (LR px)")
    ax.set_ylabel("Valid segments")
    ax.set_title("EP09 Route B ESF Sigma")
    ax.legend()
    return savefig_academic(fig, output_path)


def _bootstrap_median(values: np.ndarray, *, n_bootstrap: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    arr = np.asarray(values, dtype=float)
    if arr.size < 2:
        return np.asarray([], dtype=float)
    out = np.empty(int(n_bootstrap), dtype=float)
    for idx in range(int(n_bootstrap)):
        sample = rng.choice(arr, size=arr.size, replace=True)
        out[idx] = float(np.median(sample))
    return out[np.isfinite(out)]


def run_esf_fitting(
    *,
    output_dir: str | Path = OUTPUT_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    frame_audit_csv: str | Path = DEFAULT_FRAME_AUDIT_CSV,
    alignment_csv: str | Path = DEFAULT_ALIGNMENT_CSV,
    alignment_method: str = "contour_refined",
    contour_segments_csv: str | Path = DEFAULT_CONTOUR_SEGMENTS_CSV,
    reference_file: str | None = None,
    min_contrast_c: float = 2.0,
    min_normal_projection: float = 0.5,
    min_r2: float = 0.95,
    half_width: float = 10.0,
    profile_step: float = 0.25,
    tangent_half_width: int = 2,
    max_fit_plots: int = 16,
    bootstrap: int = 1000,
    seed: int = 1909,
    workers: int = default_workers(),
) -> EsfFitResult:
    """Fit ESF profiles on selected EP04 outer contour segments."""

    del workers
    start = time.perf_counter()
    out_dir = ensure_dir(output_dir)
    fit_dir = ensure_dir(out_dir / "esf_fits")
    image, ref_file = load_reference_frame(
        data_dir=data_dir,
        frame_audit_csv=frame_audit_csv,
        alignment_csv=alignment_csv,
        alignment_method=alignment_method,
        reference_file=reference_file,
    )
    segments = load_segments(contour_segments_csv)
    source = (
        segments["source"].astype(str)
        if "source" in segments.columns
        else pd.Series(["outer"] * len(segments), index=segments.index)
    )
    candidates = segments[
        source.eq("outer")
        & (pd.to_numeric(segments["abs_delta_t_c"], errors="coerce") >= float(min_contrast_c))
        & (pd.to_numeric(segments["normal_projection"], errors="coerce") >= float(min_normal_projection))
    ].copy()

    rows: list[dict[str, float | int | str | bool]] = []
    for _, row in candidates.iterrows():
        result: dict[str, float | int | str | bool] = {
            "segment_id": int(row["segment_id"]),
            "x_px": float(row["x_px"]),
            "y_px": float(row["y_px"]),
            "normal_projection": float(row["normal_projection"]),
            "abs_delta_t_c": float(row["abs_delta_t_c"]),
            "quality_label": str(row.get("quality_label", "")),
        }
        try:
            x, profile = extract_normal_profile(
                image,
                x_px=float(row["x_px"]),
                y_px=float(row["y_px"]),
                nx=float(row["nx"]),
                ny=float(row["ny"]),
                tx=float(row["tx"]),
                ty=float(row["ty"]),
                half_width=half_width,
                step=profile_step,
                tangent_half_width=tangent_half_width,
            )
            params, cov, r2, rmse = fit_esf_profile(x, profile)
            sigma_err = float(np.sqrt(max(float(cov[2, 2]), 0.0))) if cov.shape == (4, 4) else float("nan")
            valid = bool(
                np.isfinite(params).all()
                and np.isfinite(r2)
                and r2 >= float(min_r2)
                and 0.05 <= abs(float(params[2])) <= 5.0
                and abs(float(params[0])) >= 0.5
            )
            result.update(
                {
                    "amplitude_c": float(params[0]),
                    "center_lr_px": float(params[1]),
                    "sigma_lr_px": abs(float(params[2])),
                    "sigma_std_err_lr_px": sigma_err,
                    "baseline_c": float(params[3]),
                    "r2": float(r2),
                    "rmse_c": rmse,
                    "valid": valid,
                    "fail_reason": "pass" if valid else "quality_gate",
                }
            )
            if len([r for r in rows if r.get("valid")]) < int(max_fit_plots):
                _plot_fit(x, profile, params, row, fit_dir / f"segment_{int(row.segment_id):03d}.png")
        except Exception as exc:
            result.update(
                {
                    "amplitude_c": float("nan"),
                    "center_lr_px": float("nan"),
                    "sigma_lr_px": float("nan"),
                    "sigma_std_err_lr_px": float("nan"),
                    "baseline_c": float("nan"),
                    "r2": float("nan"),
                    "rmse_c": float("nan"),
                    "valid": False,
                    "fail_reason": type(exc).__name__,
                }
            )
        rows.append(result)

    fits = pd.DataFrame(rows)
    valid = fits[fits["valid"] & np.isfinite(fits["sigma_lr_px"])].copy()
    values = valid["sigma_lr_px"].to_numpy(dtype=float)
    boot = _bootstrap_median(values, n_bootstrap=bootstrap, seed=seed)
    ci95 = [float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))] if len(boot) else [float("nan"), float("nan")]
    summary = {
        "route": "B_esf_fitting",
        "sigma_esf_median_lr_px": float(np.median(values)) if len(values) else float("nan"),
        "sigma_esf_mean_lr_px": float(np.mean(values)) if len(values) else float("nan"),
        "sigma_esf_iqr_lr_px": [
            float(np.percentile(values, 25)) if len(values) else float("nan"),
            float(np.percentile(values, 75)) if len(values) else float("nan"),
        ],
        "bootstrap_ci95_lr_px": ci95,
        "bootstrap_n": int(len(boot)),
        "n_candidates": int(len(candidates)),
        "n_valid": int(len(valid)),
        "min_contrast_c": float(min_contrast_c),
        "min_normal_projection": float(min_normal_projection),
        "min_r2": float(min_r2),
        "reference_file": ref_file,
        "contour_segments_csv": relative(contour_segments_csv),
        "elapsed_sec": float(time.perf_counter() - start),
    }
    fits.to_csv(out_dir / "esf_sigma_distribution.csv", index=False)
    write_json(out_dir / "sigma_esf.json", summary)
    if len(valid):
        plot_esf_histogram(fits, summary, out_dir / "esf_sigma_histogram.png")
    return EsfFitResult(fits=fits, summary=summary)
