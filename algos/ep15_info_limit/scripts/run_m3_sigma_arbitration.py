#!/usr/bin/env python3
"""Run EP15 M3 sigma arbitration between ESF, forward, and FRC evidence.

M3 checks whether EP09 Route B measured a broadened thermal edge rather than
the optical PSF alone. It builds a multi-frame drizzle mean, fits ESF profiles
on several edge families, and cross-checks the FRC curve with simple
Gaussian-PSF plus detector-aperture MTF shapes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
ALGO_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[3]
EP06_SRC = PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src"
EP09_SRC = PROJECT_ROOT / "algos" / "ep09_psf_calibration" / "src"

for path in (ALGO_ROOT / "src", EP06_SRC, EP09_SRC, PROJECT_ROOT / "core" / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from common.alignment import load_alignment_shifts  # noqa: E402
from common.data_loader import load_main_session_frames, offset_correction  # noqa: E402
from psf_calibration.esf_fitting import extract_normal_profile, fit_esf_profile  # noqa: E402
from thermal_core.plotting import (  # noqa: E402
    FIGURE_SIZES,
    METHOD_COLOR_LIST,
    METHOD_COLORS,
    get_method_style,
    savefig_academic,
    setup_academic_style,
)


EXPECTED_CLEAN_SR_FRAMES = 248
PIXEL_SIZE_UM = 10.0
DETECTOR_APERTURE_UM = 10.0
SIGMA_GRID_LR_PX = (0.2, 0.3, 0.4, 0.5, 0.7, 1.0)
ROUTE_A_SIGMA_LR_PX = 0.2257
ROUTE_B_SIGMA_LR_PX = 1.1286
ROUTE_C_SIGMA_LR_PX = 0.1190


@dataclass(frozen=True)
class EdgeCandidate:
    """A candidate ESF anchor in LR pixel coordinates."""

    edge_type: str
    edge_id: str
    x_px: float
    y_px: float
    nx: float
    ny: float
    tx: float
    ty: float
    source_score: float
    source_note: str


@dataclass(frozen=True)
class Reconstruction:
    image: np.ndarray
    zero_coverage_pct: float


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def load_grid_scale(path: Path) -> tuple[int, str]:
    """Load the M1 grid decision when present, otherwise use EP15 defaults."""

    if not path.exists():
        return 5, "default"
    data = _read_json(path)
    return int(data.get("grid_scale", data.get("scale", 5))), _rel(path)


def _splat_frame(
    accum: np.ndarray,
    weight_sum: np.ndarray,
    frame: np.ndarray,
    *,
    dx_px: float,
    dy_px: float,
    scale: int,
    y_base: np.ndarray,
    x_base: np.ndarray,
) -> None:
    """Bilinearly splat one LR frame into an HR accumulator."""

    hr_rows, hr_cols = accum.shape
    y = y_base + float(dy_px) * scale
    x = x_base + float(dx_px) * scale
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    fy = (y - y0).astype(np.float32, copy=False)
    fx = (x - x0).astype(np.float32, copy=False)

    for y_idx, wy in ((y0, 1.0 - fy), (y0 + 1, fy)):
        valid_y = (y_idx >= 0) & (y_idx < hr_rows) & (wy > 0.0)
        if not bool(valid_y.any()):
            continue
        yy = y_idx[valid_y]
        wy_valid = wy[valid_y].astype(np.float32, copy=False)
        for x_idx, wx in ((x0, 1.0 - fx), (x0 + 1, fx)):
            valid_x = (x_idx >= 0) & (x_idx < hr_cols) & (wx > 0.0)
            if not bool(valid_x.any()):
                continue
            xx = x_idx[valid_x]
            wx_valid = wx[valid_x].astype(np.float32, copy=False)
            weights = wy_valid[:, None] * wx_valid[None, :]
            values = frame[np.ix_(valid_y, valid_x)].astype(np.float32, copy=False)
            target = np.ix_(yy, xx)
            accum[target] += values * weights
            weight_sum[target] += weights


def bilinear_drizzle_mean(frames: np.ndarray, shifts: np.ndarray, *, scale: int) -> Reconstruction:
    """Build a multi-frame drizzle mean used for M3 ESF extraction."""

    if frames.ndim != 3:
        raise ValueError("frames must have shape (N, H, W)")
    if shifts.shape != (frames.shape[0], 2):
        raise ValueError(f"shifts shape {shifts.shape} does not match frame count {frames.shape[0]}")

    _, rows, cols = frames.shape
    accum = np.zeros((rows * scale, cols * scale), dtype=np.float32)
    weight_sum = np.zeros_like(accum)
    y_base = np.arange(rows, dtype=np.float64) * scale
    x_base = np.arange(cols, dtype=np.float64) * scale

    for frame, (dx_px, dy_px) in tqdm(
        zip(frames, shifts, strict=True),
        total=frames.shape[0],
        desc="M3 drizzle mean",
    ):
        _splat_frame(
            accum,
            weight_sum,
            frame,
            dx_px=float(dx_px),
            dy_px=float(dy_px),
            scale=scale,
            y_base=y_base,
            x_base=x_base,
        )

    covered = weight_sum > 1e-6
    zero_coverage_pct = 100.0 * float(1.0 - covered.mean())
    image = np.empty_like(accum)
    image[covered] = accum[covered] / weight_sum[covered]
    image[~covered] = float(np.nanmean(frames))
    return Reconstruction(image=image, zero_coverage_pct=zero_coverage_pct)


def block_mean_to_lr(image_hr: np.ndarray, *, scale: int) -> np.ndarray:
    rows = image_hr.shape[0] // scale
    cols = image_hr.shape[1] // scale
    crop = image_hr[: rows * scale, : cols * scale]
    return crop.reshape(rows, scale, cols, scale).mean(axis=(1, 3)).astype(np.float32, copy=False)


def load_segments(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Contour segment CSV not found: {path}")
    segments = pd.read_csv(path)
    required = {"segment_id", "x_px", "y_px", "nx", "ny", "tx", "ty", "abs_delta_t_c", "normal_projection"}
    missing = required - set(segments.columns)
    if missing:
        raise ValueError(f"Contour segment CSV is missing required columns: {sorted(missing)}")
    return segments


def outer_edge_candidates(
    segments: pd.DataFrame,
    *,
    min_contrast_c: float,
    min_normal_projection: float,
    max_candidates: int,
) -> list[EdgeCandidate]:
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
    candidates["rank_score"] = candidates["abs_delta_t_c"].astype(float) * candidates["normal_projection"].astype(float)
    candidates = candidates.sort_values("rank_score", ascending=False).head(int(max_candidates))

    out: list[EdgeCandidate] = []
    for _, row in candidates.iterrows():
        out.append(
            EdgeCandidate(
                edge_type="die_outer_border",
                edge_id=f"outer_{int(row['segment_id']):03d}",
                x_px=float(row["x_px"]),
                y_px=float(row["y_px"]),
                nx=float(row["nx"]),
                ny=float(row["ny"]),
                tx=float(row["tx"]),
                ty=float(row["ty"]),
                source_score=float(row["rank_score"]),
                source_note=f"EP04 outer segment {int(row['segment_id'])}",
            )
        )
    return out


def _die_bbox(segments: pd.DataFrame, image_shape: tuple[int, int], *, inset_lr_px: int) -> tuple[int, int, int, int]:
    rows, cols = image_shape
    x0 = max(0, int(np.floor(float(segments["x_px"].min()))) + int(inset_lr_px))
    x1 = min(cols - 1, int(np.ceil(float(segments["x_px"].max()))) - int(inset_lr_px))
    y0 = max(0, int(np.floor(float(segments["y_px"].min()))) + int(inset_lr_px))
    y1 = min(rows - 1, int(np.ceil(float(segments["y_px"].max()))) - int(inset_lr_px))
    if x1 <= x0 or y1 <= y0:
        raise ValueError(f"Invalid die ROI after inset={inset_lr_px}: {(x0, x1, y0, y1)}")
    return x0, x1, y0, y1


def _select_local_maxima(
    gradient: np.ndarray,
    mask: np.ndarray,
    *,
    min_distance_lr_px: int,
    max_candidates: int,
    quantile: float,
) -> list[tuple[int, int, float]]:
    values = gradient[mask]
    if values.size == 0:
        return []
    threshold = float(np.quantile(values[np.isfinite(values)], quantile))
    local = gradient == ndimage.maximum_filter(gradient, size=max(3, int(min_distance_lr_px)))
    ys, xs = np.where(mask & local & np.isfinite(gradient) & (gradient >= threshold))
    order = np.argsort(gradient[ys, xs])[::-1]
    selected: list[tuple[int, int, float]] = []
    min_dist2 = float(min_distance_lr_px) ** 2
    for idx in order:
        y = int(ys[idx])
        x = int(xs[idx])
        if all((x - sx) ** 2 + (y - sy) ** 2 >= min_dist2 for sy, sx, _ in selected):
            selected.append((y, x, float(gradient[y, x])))
        if len(selected) >= int(max_candidates):
            break
    return selected


def auto_edge_candidates(
    image_lr: np.ndarray,
    segments: pd.DataFrame,
    *,
    edge_type: str,
    inset_lr_px: int,
    background_sigma_lr_px: float,
    smooth_sigma_lr_px: float,
    min_distance_lr_px: int,
    max_candidates: int,
    quantile: float,
) -> list[EdgeCandidate]:
    """Detect strong local temperature edges inside the die bounding box."""

    base = np.asarray(image_lr, dtype=np.float32)
    if background_sigma_lr_px > 0:
        base = base - ndimage.gaussian_filter(base, sigma=float(background_sigma_lr_px), mode="nearest")
    smooth = ndimage.gaussian_filter(base, sigma=float(smooth_sigma_lr_px), mode="nearest")
    grad_y, grad_x = np.gradient(smooth)
    gradient = np.hypot(grad_x, grad_y)

    x0, x1, y0, y1 = _die_bbox(segments, image_lr.shape, inset_lr_px=int(inset_lr_px))
    mask = np.zeros_like(gradient, dtype=bool)
    mask[y0 : y1 + 1, x0 : x1 + 1] = True

    maxima = _select_local_maxima(
        gradient,
        mask,
        min_distance_lr_px=min_distance_lr_px,
        max_candidates=max_candidates,
        quantile=quantile,
    )
    out: list[EdgeCandidate] = []
    for idx, (y, x, score) in enumerate(maxima):
        gx = float(grad_x[y, x])
        gy = float(grad_y[y, x])
        norm = float(np.hypot(gx, gy))
        if norm <= 1e-9:
            continue
        nx = gx / norm
        ny = gy / norm
        tx = -ny
        ty = nx
        out.append(
            EdgeCandidate(
                edge_type=edge_type,
                edge_id=f"{edge_type}_{idx:03d}",
                x_px=float(x),
                y_px=float(y),
                nx=float(nx),
                ny=float(ny),
                tx=float(tx),
                ty=float(ty),
                source_score=score,
                source_note=(
                    f"local gradient max; bbox inset={inset_lr_px} px; "
                    f"background_sigma={background_sigma_lr_px}"
                ),
            )
        )
    return out


def fit_candidate(
    image_hr: np.ndarray,
    candidate: EdgeCandidate,
    *,
    scale: int,
    half_width_lr_px: float,
    profile_step_lr_px: float,
    tangent_half_width_lr_px: float,
    min_r2: float,
    min_amplitude_c: float,
) -> dict[str, Any]:
    """Fit one ESF candidate on the HR drizzle image and report LR-pixel sigma."""

    tangent_half_width_hr_px = max(1, int(round(float(tangent_half_width_lr_px) * scale)))
    x_hr, profile = extract_normal_profile(
        image_hr,
        x_px=float(candidate.x_px) * scale,
        y_px=float(candidate.y_px) * scale,
        nx=float(candidate.nx),
        ny=float(candidate.ny),
        tx=float(candidate.tx),
        ty=float(candidate.ty),
        half_width=float(half_width_lr_px) * scale,
        step=float(profile_step_lr_px) * scale,
        tangent_half_width=tangent_half_width_hr_px,
    )
    x_lr = x_hr / float(scale)
    params, cov, r2, rmse = fit_esf_profile(x_lr, profile)
    sigma_std_err = float(np.sqrt(max(float(cov[2, 2]), 0.0))) if cov.shape == (4, 4) else float("nan")
    sigma = abs(float(params[2]))
    amplitude = float(params[0])
    valid = bool(
        np.isfinite(params).all()
        and np.isfinite(r2)
        and r2 >= float(min_r2)
        and 0.05 <= sigma <= 5.0
        and abs(amplitude) >= float(min_amplitude_c)
    )
    return {
        "edge_type": candidate.edge_type,
        "edge_id": candidate.edge_id,
        "x_px": candidate.x_px,
        "y_px": candidate.y_px,
        "nx": candidate.nx,
        "ny": candidate.ny,
        "tx": candidate.tx,
        "ty": candidate.ty,
        "source_score": candidate.source_score,
        "source_note": candidate.source_note,
        "amplitude_c": amplitude,
        "center_lr_px": float(params[1]),
        "sigma_total_lr_px": sigma,
        "sigma_total_um": sigma * PIXEL_SIZE_UM,
        "sigma_std_err_lr_px": sigma_std_err,
        "baseline_c": float(params[3]),
        "r2": float(r2),
        "rmse_c": float(rmse),
        "valid": valid,
        "fail_reason": "pass" if valid else "quality_gate",
    }


def run_edge_fits(
    image_hr: np.ndarray,
    candidates: list[EdgeCandidate],
    *,
    scale: int,
    half_width_lr_px: float,
    profile_step_lr_px: float,
    tangent_half_width_lr_px: float,
    min_r2: float,
    min_amplitude_c: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for candidate in tqdm(candidates, desc="M3 ESF fits"):
        try:
            rows.append(
                fit_candidate(
                    image_hr,
                    candidate,
                    scale=scale,
                    half_width_lr_px=half_width_lr_px,
                    profile_step_lr_px=profile_step_lr_px,
                    tangent_half_width_lr_px=tangent_half_width_lr_px,
                    min_r2=min_r2,
                    min_amplitude_c=min_amplitude_c,
                )
            )
        except Exception as exc:
            rows.append(
                {
                    "edge_type": candidate.edge_type,
                    "edge_id": candidate.edge_id,
                    "x_px": candidate.x_px,
                    "y_px": candidate.y_px,
                    "nx": candidate.nx,
                    "ny": candidate.ny,
                    "tx": candidate.tx,
                    "ty": candidate.ty,
                    "source_score": candidate.source_score,
                    "source_note": candidate.source_note,
                    "amplitude_c": float("nan"),
                    "center_lr_px": float("nan"),
                    "sigma_total_lr_px": float("nan"),
                    "sigma_total_um": float("nan"),
                    "sigma_std_err_lr_px": float("nan"),
                    "baseline_c": float("nan"),
                    "r2": float("nan"),
                    "rmse_c": float("nan"),
                    "valid": False,
                    "fail_reason": type(exc).__name__,
                }
            )
    return pd.DataFrame(rows)


def summarize_edge_fits(fits: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    valid = fits[fits["valid"] & np.isfinite(fits["sigma_total_lr_px"])].copy()
    rows: list[dict[str, Any]] = []
    for edge_type, group_all in fits.groupby("edge_type", sort=False):
        group = valid[valid["edge_type"].eq(edge_type)]
        values = group["sigma_total_lr_px"].to_numpy(dtype=float)
        rows.append(
            {
                "edge_type": edge_type,
                "n_candidates": int(len(group_all)),
                "n_valid": int(len(group)),
                "sigma_total_median_lr_px": float(np.median(values)) if values.size else float("nan"),
                "sigma_total_mean_lr_px": float(np.mean(values)) if values.size else float("nan"),
                "sigma_total_min_lr_px": float(np.min(values)) if values.size else float("nan"),
                "sigma_total_p25_lr_px": float(np.percentile(values, 25)) if values.size else float("nan"),
                "sigma_total_p75_lr_px": float(np.percentile(values, 75)) if values.size else float("nan"),
                "median_r2": float(np.median(group["r2"].to_numpy(dtype=float))) if values.size else float("nan"),
                "median_abs_amplitude_c": (
                    float(np.median(np.abs(group["amplitude_c"].to_numpy(dtype=float)))) if values.size else float("nan")
                ),
            }
        )
    summary_table = pd.DataFrame(rows)

    if valid.empty:
        global_summary = {
            "sharpest_edge_sigma_total_lr_px": float("nan"),
            "sharpest_edge_id": None,
            "sharpest_edge_type": None,
            "outer_vs_internal_ratio": float("nan"),
            "outer_minus_internal_lr_px": float("nan"),
            "route_b_bias_confirmed_by_edges": False,
        }
        return summary_table, global_summary

    sharpest = valid.sort_values("sigma_total_lr_px", ascending=True).iloc[0]
    medians = summary_table.set_index("edge_type")["sigma_total_median_lr_px"].to_dict()
    outer = float(medians.get("die_outer_border", float("nan")))
    internal = float(medians.get("internal_metal_strong", float("nan")))
    ratio = outer / internal if np.isfinite(outer) and np.isfinite(internal) and internal > 0 else float("nan")
    diff = outer - internal if np.isfinite(outer) and np.isfinite(internal) else float("nan")
    route_b_bias = bool(np.isfinite(ratio) and ratio >= 1.25 and np.isfinite(diff) and diff >= 0.20)

    sigma_psf_proxy = float(sharpest["sigma_total_lr_px"])
    summary_table["w_edge_median_lr_px_if_sharpest_is_psf"] = np.sqrt(
        np.maximum(summary_table["sigma_total_median_lr_px"].to_numpy(dtype=float) ** 2 - sigma_psf_proxy**2, 0.0)
    )
    global_summary = {
        "sharpest_edge_sigma_total_lr_px": sigma_psf_proxy,
        "sharpest_edge_sigma_total_um": sigma_psf_proxy * PIXEL_SIZE_UM,
        "sharpest_edge_id": str(sharpest["edge_id"]),
        "sharpest_edge_type": str(sharpest["edge_type"]),
        "outer_vs_internal_ratio": float(ratio),
        "outer_minus_internal_lr_px": float(diff),
        "route_b_bias_confirmed_by_edges": route_b_bias,
    }
    return summary_table, global_summary


def _normalize_shape(values: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    arr = np.clip(arr, 0.0, 1.0)
    lo = float(np.nanmin(arr))
    hi = float(np.nanmax(arr))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-12:
        return np.zeros_like(arr, dtype=float)
    return (arr - lo) / (hi - lo)


def aperture_gaussian_mtf2(frequency_um_inv: np.ndarray, sigma_lr_px: float) -> np.ndarray:
    """Return MTF^2 for Gaussian PSF sigma and 10 um detector aperture."""

    f = np.asarray(frequency_um_inv, dtype=float)
    sigma_um = float(sigma_lr_px) * PIXEL_SIZE_UM
    mtf = np.exp(-2.0 * np.pi**2 * sigma_um**2 * f**2) * np.abs(np.sinc(DETECTOR_APERTURE_UM * f))
    return mtf**2


def fit_frc_shape(
    curve: pd.DataFrame,
    *,
    sigma_grid: tuple[float, ...],
    fit_period_min_um: float,
    fit_period_max_um: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    f = curve["frequency_um_inv"].to_numpy(dtype=float)
    period = curve["period_um"].to_numpy(dtype=float)
    frc = curve["frc"].to_numpy(dtype=float)
    fit_mask = (
        (f > 0)
        & np.isfinite(f)
        & np.isfinite(period)
        & np.isfinite(frc)
        & (period >= float(fit_period_min_um))
        & (period <= float(fit_period_max_um))
    )
    if int(fit_mask.sum()) < 8:
        raise ValueError("Not enough FRC samples in the requested fit band")

    obs_norm = _normalize_shape(frc[fit_mask])
    score_rows: list[dict[str, float]] = []
    plot_df = curve[["frequency_um_inv", "period_um", "frc"]].copy()
    plot_df["frc_shape_norm"] = np.nan
    display_mask = (f > 0) & np.isfinite(f) & np.isfinite(period) & np.isfinite(frc)
    plot_df.loc[display_mask, "frc_shape_norm"] = _normalize_shape(frc[display_mask])
    for sigma in sigma_grid:
        model_fit = aperture_gaussian_mtf2(f[fit_mask], float(sigma))
        model_norm = _normalize_shape(model_fit)
        mse = float(np.mean((obs_norm - model_norm) ** 2))
        corr = float(np.corrcoef(obs_norm, model_norm)[0, 1]) if np.std(model_norm) > 0 else float("nan")
        score_rows.append({"sigma_lr_px": float(sigma), "shape_mse": mse, "shape_corr": corr})

        model_display = aperture_gaussian_mtf2(f[display_mask], float(sigma))
        col = f"mtf2_sigma_{str(sigma).replace('.', '_')}"
        plot_df[col] = np.nan
        plot_df.loc[display_mask, col] = _normalize_shape(model_display)

    return pd.DataFrame(score_rows).sort_values("shape_mse").reset_index(drop=True), plot_df


def plot_edge_comparison(edge_summary: pd.DataFrame, fits: pd.DataFrame, output_path: Path) -> None:
    setup_academic_style()
    labels = {
        "die_outer_border": "die outer border",
        "internal_metal_strong": "internal strong edges",
        "steepest_temperature_edge": "steepest edges",
    }
    order = [edge for edge in labels if edge in set(edge_summary["edge_type"])]
    summary = edge_summary.set_index("edge_type").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["one_half_col"])
    x = np.arange(len(summary), dtype=float)
    med = summary["sigma_total_median_lr_px"].to_numpy(dtype=float)
    low = med - summary["sigma_total_p25_lr_px"].to_numpy(dtype=float)
    high = summary["sigma_total_p75_lr_px"].to_numpy(dtype=float) - med
    colors = [METHOD_COLOR_LIST[idx % len(METHOD_COLOR_LIST)] for idx in range(len(summary))]
    ax.bar(x, med, color=colors, alpha=0.82, edgecolor="white", linewidth=0.8)
    ax.errorbar(x, med, yerr=np.vstack([low, high]), fmt="none", ecolor="#333333", elinewidth=0.9, capsize=3)

    rng = np.random.default_rng(1503)
    valid = fits[fits["valid"] & np.isfinite(fits["sigma_total_lr_px"])].copy()
    for idx, edge_type in enumerate(order):
        values = valid.loc[valid["edge_type"].eq(edge_type), "sigma_total_lr_px"].to_numpy(dtype=float)
        if values.size:
            jitter = rng.uniform(-0.10, 0.10, size=values.size)
            ax.scatter(
                np.full(values.size, x[idx]) + jitter,
                values,
                s=12,
                color="#222222",
                alpha=0.55,
                linewidths=0,
                zorder=3,
            )

    ax.axhspan(0.2, 0.5, color=METHOD_COLORS["secondary"], alpha=0.12, label="credible optical range 0.2-0.5")
    ax.axhline(ROUTE_B_SIGMA_LR_PX, color=METHOD_COLORS["accent_1"], linestyle="--", linewidth=0.9, label="EP09 Route B")
    ax.axhline(ROUTE_A_SIGMA_LR_PX, color=METHOD_COLORS["primary"], linestyle=":", linewidth=0.9, label="EP09 Route A")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[item] for item in order], rotation=12, ha="right")
    ax.set_ylabel(r"ESF $\sigma_{total}$ [LR px]")
    ax.set_title("M3 Multi-edge ESF Widths")
    ax.set_ylim(0.0, max(1.35, float(np.nanmax(med)) * 1.25 if np.isfinite(med).any() else 1.35))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=6.4)
    savefig_academic(fig, output_path)


def plot_frc_shape_fit(
    plot_df: pd.DataFrame,
    score_table: pd.DataFrame,
    *,
    fit_period_min_um: float,
    fit_period_max_um: float,
    output_path: Path,
) -> None:
    setup_academic_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["double_col"])
    display = plot_df[
        (plot_df["period_um"] >= 4.0)
        & (plot_df["period_um"] <= max(100.0, float(fit_period_max_um)))
        & np.isfinite(plot_df["frc_shape_norm"])
    ].copy()
    ax.plot(
        display["period_um"],
        display["frc_shape_norm"],
        color="#222222",
        linewidth=1.6,
        label="measured FRC shape",
    )
    for idx, sigma in enumerate(SIGMA_GRID_LR_PX):
        col = f"mtf2_sigma_{str(sigma).replace('.', '_')}"
        style = get_method_style(idx)
        ax.plot(
            display["period_um"],
            display[col],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.05,
            label=rf"$\sigma={sigma:.1f}$ px",
        )
    best = float(score_table.iloc[0]["sigma_lr_px"])
    ax.axvspan(fit_period_min_um, fit_period_max_um, color="#999999", alpha=0.10, label="fit band")
    ax.axvline(10.0, color="#666666", linestyle="-.", linewidth=0.85, label="10 um aperture zero")
    ax.text(
        0.03,
        0.08,
        rf"best $\sigma={best:.1f}$ LR px",
        transform=ax.transAxes,
        fontsize=8,
        color="#222222",
    )
    ax.set_xlabel("Spatial period [um]")
    ax.set_ylabel("Normalized shape")
    ax.set_title("M3 FRC Shape Fit")
    ax.set_xlim(max(100.0, float(fit_period_max_um)), 4.0)
    ax.set_ylim(-0.06, 1.06)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=6.4, ncol=1)
    savefig_academic(fig, output_path)


def validate_inputs(frames: np.ndarray, metadata: pd.DataFrame, shifts: np.ndarray) -> None:
    if len(metadata) != EXPECTED_CLEAN_SR_FRAMES:
        raise ValueError(f"Expected {EXPECTED_CLEAN_SR_FRAMES} clean SR frames; got {len(metadata)}")
    if frames.shape[0] != EXPECTED_CLEAN_SR_FRAMES:
        raise ValueError(f"Expected {EXPECTED_CLEAN_SR_FRAMES} frame arrays; got {frames.shape[0]}")
    if frames.shape[1:] != (480, 640):
        raise ValueError(f"Expected detector frame shape (480, 640); got {frames.shape[1:]}")
    if shifts.shape != (EXPECTED_CLEAN_SR_FRAMES, 2):
        raise ValueError(f"Expected shifts shape ({EXPECTED_CLEAN_SR_FRAMES}, 2); got {shifts.shape}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "data_raw" / "infrared_avi")
    parser.add_argument("--frame-audit-csv", type=Path, default=PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--alignment-csv", type=Path, default=PROJECT_ROOT / "output" / "ep05_contour_alignment" / "contour_alignment_results.csv")
    parser.add_argument("--contour-segments-csv", type=Path, default=PROJECT_ROOT / "output" / "ep04_global_validation" / "inputs" / "contour_segments.csv")
    parser.add_argument("--m1-grid-decision-json", type=Path, default=PROJECT_ROOT / "output" / "ep15_info_limit" / "m1_phase_structure" / "grid_decision.json")
    parser.add_argument("--m2-frc-curve-csv", type=Path, default=PROJECT_ROOT / "output" / "ep15_info_limit" / "m2_frc" / "frc_curve.csv")
    parser.add_argument("--m2-frc-summary-json", type=Path, default=PROJECT_ROOT / "output" / "ep15_info_limit" / "m2_frc" / "frc_summary.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "ep15_info_limit" / "m3_sigma")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--min-r2", type=float, default=0.85)
    parser.add_argument("--min-amplitude-c", type=float, default=0.10)
    parser.add_argument("--half-width-lr-px", type=float, default=8.0)
    parser.add_argument("--profile-step-lr-px", type=float, default=0.20)
    parser.add_argument("--tangent-half-width-lr-px", type=float, default=2.0)
    parser.add_argument("--outer-min-contrast-c", type=float, default=2.0)
    parser.add_argument("--outer-min-normal-projection", type=float, default=0.5)
    parser.add_argument("--max-candidates-per-type", type=int, default=32)
    parser.add_argument("--internal-inset-lr-px", type=int, default=40)
    parser.add_argument("--steepest-inset-lr-px", type=int, default=4)
    parser.add_argument("--edge-min-distance-lr-px", type=int, default=16)
    parser.add_argument("--internal-background-sigma-lr-px", type=float, default=10.0)
    parser.add_argument("--steepest-background-sigma-lr-px", type=float, default=0.0)
    parser.add_argument("--edge-smooth-sigma-lr-px", type=float, default=1.0)
    parser.add_argument("--edge-quantile", type=float, default=0.94)
    parser.add_argument("--frc-fit-period-min-um", type=float, default=12.0)
    parser.add_argument("--frc-fit-period-max-um", type=float, default=80.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_academic_style()

    if not args.m2_frc_curve_csv.exists():
        raise FileNotFoundError(f"M2 FRC curve is required before M3: {args.m2_frc_curve_csv}")
    if not args.m2_frc_summary_json.exists():
        raise FileNotFoundError(f"M2 FRC summary is required before M3: {args.m2_frc_summary_json}")

    grid_scale, grid_source = load_grid_scale(args.m1_grid_decision_json)
    segments = load_segments(args.contour_segments_csv)

    frames_raw, metadata = load_main_session_frames(
        args.data_dir,
        args.frame_audit_csv,
        workers=args.workers,
        dtype=np.float32,
    )
    shifts = load_alignment_shifts("contour_refined", metadata=metadata, alignment_csv=args.alignment_csv)
    validate_inputs(frames_raw, metadata, shifts)
    frames = offset_correction(frames_raw, method="median").astype(np.float32, copy=False)
    rec = bilinear_drizzle_mean(frames, shifts, scale=grid_scale)
    composite_lr = block_mean_to_lr(rec.image, scale=grid_scale)

    candidates = []
    candidates.extend(
        outer_edge_candidates(
            segments,
            min_contrast_c=args.outer_min_contrast_c,
            min_normal_projection=args.outer_min_normal_projection,
            max_candidates=args.max_candidates_per_type,
        )
    )
    candidates.extend(
        auto_edge_candidates(
            composite_lr,
            segments,
            edge_type="internal_metal_strong",
            inset_lr_px=args.internal_inset_lr_px,
            background_sigma_lr_px=args.internal_background_sigma_lr_px,
            smooth_sigma_lr_px=args.edge_smooth_sigma_lr_px,
            min_distance_lr_px=args.edge_min_distance_lr_px,
            max_candidates=args.max_candidates_per_type,
            quantile=args.edge_quantile,
        )
    )
    candidates.extend(
        auto_edge_candidates(
            composite_lr,
            segments,
            edge_type="steepest_temperature_edge",
            inset_lr_px=args.steepest_inset_lr_px,
            background_sigma_lr_px=args.steepest_background_sigma_lr_px,
            smooth_sigma_lr_px=args.edge_smooth_sigma_lr_px,
            min_distance_lr_px=args.edge_min_distance_lr_px,
            max_candidates=args.max_candidates_per_type,
            quantile=args.edge_quantile,
        )
    )
    if len({candidate.edge_type for candidate in candidates}) < 3:
        raise RuntimeError("M3 requires at least three edge families")

    fits = run_edge_fits(
        rec.image,
        candidates,
        scale=grid_scale,
        half_width_lr_px=args.half_width_lr_px,
        profile_step_lr_px=args.profile_step_lr_px,
        tangent_half_width_lr_px=args.tangent_half_width_lr_px,
        min_r2=args.min_r2,
        min_amplitude_c=args.min_amplitude_c,
    )
    edge_summary, edge_global = summarize_edge_fits(fits)
    fits.to_csv(args.output_dir / "edge_fit_table.csv", index=False)
    edge_summary.to_csv(args.output_dir / "edge_summary.csv", index=False)

    frc_curve = pd.read_csv(args.m2_frc_curve_csv)
    frc_scores, frc_plot = fit_frc_shape(
        frc_curve,
        sigma_grid=SIGMA_GRID_LR_PX,
        fit_period_min_um=args.frc_fit_period_min_um,
        fit_period_max_um=args.frc_fit_period_max_um,
    )
    frc_scores.to_csv(args.output_dir / "frc_shape_fit_scores.csv", index=False)
    best_frc_fit_sigma = float(frc_scores.iloc[0]["sigma_lr_px"])

    plot_edge_comparison(edge_summary, fits, args.output_dir / "edge_comparison.png")
    plot_frc_shape_fit(
        frc_plot,
        frc_scores,
        fit_period_min_um=args.frc_fit_period_min_um,
        fit_period_max_um=args.frc_fit_period_max_um,
        output_path=args.output_dir / "frc_shape_fit.png",
    )

    route_b_bias_confirmed = bool(edge_global["route_b_bias_confirmed_by_edges"])
    m2_summary = _read_json(args.m2_frc_summary_json)
    summary = {
        "task": "EP15 M3 sigma arbitration",
        "grid_scale": int(grid_scale),
        "grid_decision_source": grid_source,
        "n_clean_sr_frames": int(len(metadata)),
        "frame_preprocess": "per-frame median offset correction followed by contour-refined bilinear drizzle mean",
        "zero_coverage_pct": rec.zero_coverage_pct,
        "route_a_forward_sigma_lr_px": ROUTE_A_SIGMA_LR_PX,
        "route_b_esf_sigma_lr_px": ROUTE_B_SIGMA_LR_PX,
        "route_c_holdout_sigma_lr_px": ROUTE_C_SIGMA_LR_PX,
        "sigma_model": "sigma_total^2 = sigma_psf^2 + w_edge^2; single-edge sigma_total is an optical-PSF upper bound only when the thermal edge width is near zero",
        "sigma_credible_range_lr_px": [0.2, 0.5],
        "sigma_credible_range_um": [2.0, 5.0],
        "sharpest_edge_sigma_total_lr_px": edge_global["sharpest_edge_sigma_total_lr_px"],
        "sharpest_edge_id": edge_global["sharpest_edge_id"],
        "sharpest_edge_type": edge_global["sharpest_edge_type"],
        "best_frc_fit_sigma": best_frc_fit_sigma,
        "best_frc_fit_sigma_um": best_frc_fit_sigma * PIXEL_SIZE_UM,
        "route_b_bias_confirmed": route_b_bias_confirmed,
        "route_b_bias_evidence": {
            "edge_evidence": edge_global,
            "edge_summary": edge_summary.to_dict(orient="records"),
            "interpretation": (
                "confirmed: die-border ESF is broader than internal sharp edges"
                if route_b_bias_confirmed
                else "partial: edge-family separation did not pass the strict ratio/difference gate"
            ),
        },
        "frc_shape_fit": {
            "model": "MTF(f)^2 = [exp(-2*pi^2*sigma^2*f^2) * abs(sinc(10um*f))]^2",
            "fit_period_band_um": [float(args.frc_fit_period_min_um), float(args.frc_fit_period_max_um)],
            "sigma_grid_lr_px": list(SIGMA_GRID_LR_PX),
            "scores": frc_scores.to_dict(orient="records"),
            "m2_cutoff_period_um": float(m2_summary.get("f_c_period_um", float("nan"))),
            "m2_theory_status": str(m2_summary.get("theory_status", "")),
        },
        "outputs": {
            "edge_comparison_png": _rel(args.output_dir / "edge_comparison.png"),
            "frc_shape_fit_png": _rel(args.output_dir / "frc_shape_fit.png"),
            "sigma_summary_json": _rel(args.output_dir / "sigma_summary.json"),
            "edge_fit_table_csv": _rel(args.output_dir / "edge_fit_table.csv"),
            "edge_summary_csv": _rel(args.output_dir / "edge_summary.csv"),
            "frc_shape_fit_scores_csv": _rel(args.output_dir / "frc_shape_fit_scores.csv"),
        },
        "elapsed_sec": float(time.perf_counter() - start),
    }
    _write_json(args.output_dir / "sigma_summary.json", summary)

    print("M3 multi-edge ESF:")
    for row in edge_summary.to_dict(orient="records"):
        print(
            f"  {row['edge_type']}: median sigma_total={row['sigma_total_median_lr_px']:.3f} LR px "
            f"(valid {row['n_valid']}/{row['n_candidates']})"
        )
    print(f"M3 route B bias confirmed: {route_b_bias_confirmed}")
    print(f"M3 sharpest edge sigma_total: {edge_global['sharpest_edge_sigma_total_lr_px']:.3f} LR px")
    print(f"M3 best FRC shape-fit sigma: {best_frc_fit_sigma:.3f} LR px")
    print(f"Saved M3 outputs to {_rel(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
