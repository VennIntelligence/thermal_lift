"""EP04 global validation helpers for data-driven multi-frame ESF."""

from __future__ import annotations

import json
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Iterable

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.collections import LineCollection
from scipy.interpolate import make_interp_spline
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.optimize import least_squares
from tqdm.auto import tqdm

from thermal_core.displacement import coordinate_to_shift, subpixel_ncc
from thermal_core.ep03 import (
    crb_multi_frame,
    detect_outer_contour,
    measure_contour_observability,
    normal_cdf,
)
from thermal_core.io import load_frame, parse_filename
from thermal_core.plotting import METHOD_COLORS, METHOD_COLOR_LIST, format_colorbar, make_figure, savefig_academic


EXPECTED_X_UM = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40)
NOISE_FLOOR_C = 0.0724


QUALITY_GATES = {
    "min_snr": 8.0,
    "min_delta_t": 0.5,
    "min_ncc_peak": 0.85,
    "min_phase_range_px": 0.15,
    # This is the fitted apparent ESF transition width, not the pure optical PSF sigma.
    "sigma_range": (0.8, 1.3),
    "max_split_half_a": 0.04,
    "max_split_half_b": 0.06,
    "max_psf_sensitivity": 0.03,
}

EP04_ANCHOR_COLOR = "#56B4E9"
EP04_REJECT_COLOR = "#C8CBD2"
EP04_OUTER_COLOR = "#3B6EA8"
EP04_INNER_COLOR = "#D0833E"
EP04_TOTAL_COLOR = "#E3E6EC"
EP04_TOTAL_EDGE = "#AEB4C0"
EP04_GATE_REJECT_COLOR = "#B8BCC6"


@dataclass(frozen=True)
class Ep04Config:
    """Runtime configuration for EP04 validation."""

    theta_deg: float = 47.6
    pixel_size_um: float = 20.0
    noise_floor_c: float = NOISE_FLOOR_C
    ncc_roi_size: int = 40
    ncc_preprocess: str = "highpass"
    ncc_search_radius: int = 2
    ncc_fit_radius: int = 1
    ncc_highpass_sigma: float = 6.0
    esf_half_width: int = 10
    esf_step_px: float = 0.5
    psf_delta_sigma_px: float = 0.15


@dataclass(frozen=True)
class ThermalBoundaryPoint:
    """One segment-level apparent thermal boundary point."""

    segment_id: int
    center_col: float
    center_row: float
    center_x_mm: float
    center_y_mm: float
    normal_angle_deg: float
    tangent_angle_deg: float
    split_half_diff_px: float
    crb_px: float
    crb_ratio: float
    confidence: str
    pass_quality_gate: bool
    fitted_sigma_px: float
    fitted_delta_t_c: float


@dataclass(frozen=True)
class EsfFitResult:
    """Single-frame ESF fit result used by EP04 joint fitting."""

    segment_id: int
    a: float
    b: float
    c: float
    s: float
    sigma: float
    rms_c: float
    s_se_px: float
    sigma_se_px: float
    success: bool
    n_points: int


def add_segment_quality_columns(segments: pd.DataFrame) -> pd.DataFrame:
    """Add EP04-compatible quality labels to current EP03 contour segments."""
    out = segments.copy()
    if out.empty:
        return out
    if "is_a_class" not in out.columns:
        if "anchor_candidate" in out.columns:
            out["is_a_class"] = out["anchor_candidate"].astype(bool)
        else:
            curvature_gate = out["curvature_proxy"] <= out["curvature_proxy"].quantile(0.70)
            out["is_a_class"] = (
                (out["snr"] >= 7.0)
                & (out["normal_projection"] >= 0.5)
                & curvature_gate
            )
    out["quality_label"] = np.select(
        [
            out["is_a_class"].astype(bool),
            (out["snr"] > 5.0) & (out["normal_projection"] > 0.35),
            out["snr"] < 3.0,
        ],
        ["A_high_snr_high_projection", "B_usable", "C_low_snr"],
        default="D_other",
    )
    if "delta_t_c" in out.columns and "abs_delta_t_c" not in out.columns:
        out["abs_delta_t_c"] = out["delta_t_c"].abs()
    if "tangent_angle_deg" not in out.columns and {"tx", "ty"}.issubset(out.columns):
        out["tangent_angle_deg"] = np.degrees(np.arctan2(out["ty"], out["tx"])) % 180.0
    return out


def esf_model(u: np.ndarray, a: float, b: float, c: float, s: float, sigma: float) -> np.ndarray:
    """Blurred step ESF with a local linear background."""
    sigma = max(float(sigma), 1e-6)
    return a + b * u + c * normal_cdf((u - s) / sigma)


def extract_normal_profile(
    frame: np.ndarray,
    segment: pd.Series,
    *,
    half_width_px: float = 10.0,
    step_px: float = 0.5,
) -> pd.DataFrame:
    """Sample one subpixel ESF profile along a segment normal."""
    u = np.arange(-half_width_px, half_width_px + 0.5 * step_px, step_px)
    x = float(segment["x_px"]) + u * float(segment["nx"])
    y = float(segment["y_px"]) + u * float(segment["ny"])
    temp = map_coordinates(np.asarray(frame, dtype=float), np.vstack([y, x]), order=1, mode="nearest")
    return pd.DataFrame(
        {
            "segment_id": int(segment["segment_id"]),
            "u_px": u,
            "x_px": x,
            "y_px": y,
            "temperature_c": temp,
        }
    )


def fit_esf_profile(
    u: np.ndarray,
    temp: np.ndarray,
    *,
    segment_id: int,
    fixed_sigma: float | None = None,
) -> EsfFitResult:
    """Fit a single-frame ESF profile."""
    u = np.asarray(u, dtype=float)
    temp = np.asarray(temp, dtype=float)
    order = np.argsort(u)
    u = u[order]
    temp = temp[order]

    left = float(np.nanmedian(temp[: max(3, len(temp) // 5)]))
    right = float(np.nanmedian(temp[-max(3, len(temp) // 5) :]))
    amp0 = right - left
    if abs(amp0) < 1e-6:
        amp0 = float(np.nanmax(temp) - np.nanmin(temp))
    unique_u, inverse = np.unique(u, return_inverse=True)
    if unique_u.size >= 3:
        temp_sum = np.bincount(inverse, weights=temp)
        temp_count = np.bincount(inverse)
        unique_temp = temp_sum / np.maximum(temp_count, 1)
        grad = np.gradient(unique_temp, unique_u)
        s0 = float(unique_u[int(np.nanargmax(np.abs(grad)))])
    elif unique_u.size == 2:
        s0 = float(np.mean(unique_u))
    else:
        s0 = float(np.nanmedian(u))

    sigma0 = 1.0 if fixed_sigma is None else float(fixed_sigma)
    p0_free = np.array([left, 0.0, amp0, s0, sigma0], dtype=float)
    s_lower = float(np.nanmin(u) - 2.0)
    s_upper = float(np.nanmax(u) + 2.0)

    if fixed_sigma is None:
        lower = np.array([np.nanmin(temp) - 2.0, -0.5, -8.0, s_lower, 0.15])
        upper = np.array([np.nanmax(temp) + 2.0, 0.5, 8.0, s_upper, 1.5])

        def residual(params: np.ndarray) -> np.ndarray:
            return esf_model(u, *params) - temp

        result = least_squares(residual, p0_free, bounds=(lower, upper), max_nfev=4000)
        params = result.x
        jac = result.jac
        n_params = 5
    else:
        sigma = float(fixed_sigma)
        p0 = p0_free[:4]
        lower = np.array([np.nanmin(temp) - 2.0, -0.5, -8.0, s_lower])
        upper = np.array([np.nanmax(temp) + 2.0, 0.5, 8.0, s_upper])

        def residual(params: np.ndarray) -> np.ndarray:
            return esf_model(u, params[0], params[1], params[2], params[3], sigma) - temp

        result = least_squares(residual, p0, bounds=(lower, upper), max_nfev=4000)
        params = np.r_[result.x, sigma]
        jac = result.jac
        n_params = 4

    resid = esf_model(u, *params) - temp
    rms = float(np.sqrt(np.mean(resid * resid)))
    dof = max(1, len(temp) - n_params)
    residual_variance = float(np.sum(resid * resid) / dof)
    s_se = np.nan
    sigma_se = np.nan
    try:
        cov = np.linalg.pinv(jac.T @ jac) * residual_variance
        s_se = float(np.sqrt(max(cov[3, 3], 0.0)))
        sigma_se = 0.0 if fixed_sigma is not None else float(np.sqrt(max(cov[4, 4], 0.0)))
    except np.linalg.LinAlgError:
        pass

    return EsfFitResult(
        segment_id=int(segment_id),
        a=float(params[0]),
        b=float(params[1]),
        c=float(params[2]),
        s=float(params[3]),
        sigma=float(params[4]),
        rms_c=rms,
        s_se_px=s_se,
        sigma_se_px=sigma_se,
        success=bool(result.success),
        n_points=int(len(temp)),
    )


def fit_joint_esf(
    multiframe_profiles: pd.DataFrame,
    *,
    frame_indices: list[int] | None = None,
) -> dict:
    """Jointly fit multi-frame ESF profiles with shared edge position and sigma."""
    data = multiframe_profiles.copy()
    if frame_indices is not None:
        data = data[data["frame_index"].isin(frame_indices)].copy()
    frames = sorted(data["frame_index"].unique())
    n_frames = len(frames)
    if n_frames == 0:
        raise ValueError("No frames available for joint ESF fit")

    single_inits = []
    sigma_inits = []
    frame_stats = []
    for frame_id in frames:
        group = data[data["frame_index"].eq(frame_id)]
        fit = fit_esf_profile(
            group["u_px"].to_numpy(),
            group["temperature_c"].to_numpy(),
            segment_id=int(group["segment_id"].iloc[0]),
        )
        delta = float(group["delta_n_px"].iloc[0])
        single_inits.append(fit.s - delta)
        sigma_inits.append(fit.sigma)
        frame_stats.append((fit.a, fit.b, fit.c))

    x0 = [float(np.nanmedian(single_inits)), float(np.clip(np.nanmedian(sigma_inits), 0.2, 1.4))]
    for a, b, c in frame_stats:
        x0.extend([a, b, c])
    x0 = np.asarray(x0, dtype=float)

    temp_min = float(data["temperature_c"].min())
    temp_max = float(data["temperature_c"].max())
    lower = [-6.0, 0.15]
    upper = [6.0, 1.5]
    for _ in frames:
        lower.extend([temp_min - 2.0, -0.5, -8.0])
        upper.extend([temp_max + 2.0, 0.5, 8.0])
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    frame_to_pos = {frame_id: i for i, frame_id in enumerate(frames)}
    u = data["u_px"].to_numpy(dtype=float)
    temp = data["temperature_c"].to_numpy(dtype=float)
    delta_n = data["delta_n_px"].to_numpy(dtype=float)
    frame_pos = data["frame_index"].map(frame_to_pos).to_numpy(dtype=int)

    def residual(params: np.ndarray) -> np.ndarray:
        s = params[0]
        sigma = max(float(params[1]), 1e-6)
        per_frame = params[2:].reshape(n_frames, 3)
        a = per_frame[frame_pos, 0]
        b = per_frame[frame_pos, 1]
        c = per_frame[frame_pos, 2]
        pred = a + b * u + c * normal_cdf((u - s - delta_n) / sigma)
        return pred - temp

    result = least_squares(residual, x0, bounds=(lower, upper), max_nfev=8000)
    resid = residual(result.x)
    return {
        "frames": frames,
        "n_frames": n_frames,
        "result": result,
        "s_px": float(result.x[0]),
        "sigma_px": float(result.x[1]),
        "rms_c": float(np.sqrt(np.mean(resid * resid))),
        "residuals": resid,
        "data": data,
        "success": bool(result.success),
    }


def build_x_scanlines(
    audit_df: pd.DataFrame,
    *,
    session: int = 2,
    r_value: int = 0,
    expected_x: Iterable[int] = EXPECTED_X_UM,
    require_sr_usable: bool = True,
) -> list[pd.DataFrame]:
    """Build complete fixed-Y X scanlines for EP04 localization validation.

    EP04 validates X-scanline localization anchors, not the full multi-frame SR
    input.  When the clean SR contract columns are present, the scanline subset
    is constrained to `is_sr_usable`/`is_main_session` in addition to
    `session=2, R=0`.
    """
    expected = set(int(v) for v in expected_x)
    mask = audit_df["session"].astype(int).eq(int(session)) & audit_df["R"].astype(int).eq(int(r_value))
    if require_sr_usable:
        if "is_sr_usable" in audit_df.columns:
            mask &= _bool_series(audit_df["is_sr_usable"])
        elif "is_main_session" in audit_df.columns:
            mask &= _bool_series(audit_df["is_main_session"])
    subset = audit_df[mask].copy()

    scanlines: list[pd.DataFrame] = []
    for _, group in subset.groupby("Y", sort=True):
        observed = set(group["X"].astype(int))
        if not expected.issubset(observed):
            continue
        line = group[group["X"].astype(int).isin(expected)].copy()
        line = line.sort_values("acquisition_order").reset_index(drop=True)
        if len(line) == len(expected):
            scanlines.append(line)
    return scanlines


def _bool_series(series: pd.Series) -> pd.Series:
    """Parse bool-like audit columns without treating the string 'False' as true."""
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False).astype(bool)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y"})


def ep04_data_contract_summary(audit_df: pd.DataFrame) -> dict:
    """Summarize how EP04's scanline subset relates to the clean SR input."""
    scanlines = build_x_scanlines(audit_df)
    scanline_files = [str(v) for line in scanlines for v in line["file"].tolist()]
    unique_scanline_files = set(scanline_files)

    raw_main_mask = audit_df["session"].astype(int).eq(2) if "session" in audit_df.columns else None
    if "is_sr_usable" in audit_df.columns:
        clean_mask = _bool_series(audit_df["is_sr_usable"])
    elif "is_clean_main_session" in audit_df.columns:
        clean_mask = _bool_series(audit_df["is_clean_main_session"])
    elif "is_main_session" in audit_df.columns:
        clean_mask = _bool_series(audit_df["is_main_session"])
    elif raw_main_mask is not None:
        clean_mask = raw_main_mask
    else:
        clean_mask = pd.Series(False, index=audit_df.index)

    scanline_mask = audit_df["file"].astype(str).isin(unique_scanline_files)
    scanline_y_values = [int(line["Y"].iloc[0]) for line in scanlines]
    scanline_lengths = sorted({int(len(line)) for line in scanlines})
    return {
        "ep04_validation_unit": "contour segment x complete R=0 X scanline",
        "raw_main_session_frame_count": int(raw_main_mask.sum()) if raw_main_mask is not None else None,
        "clean_sr_input_frame_count": int(clean_mask.sum()),
        "ep04_scanline_count": int(len(scanlines)),
        "ep04_scanline_frame_count": int(len(scanline_files)),
        "ep04_unique_frame_count": int(len(unique_scanline_files)),
        "ep04_clean_unique_frame_count": int((scanline_mask & clean_mask).sum()),
        "ep04_scanline_y_um": scanline_y_values,
        "ep04_scanline_lengths": scanline_lengths,
        "ep04_scanline_filter": "session=2, R=0, is_sr_usable=True when available",
    }


def select_clean_sr_reference_row(audit_df: pd.DataFrame) -> pd.Series:
    """Select a representative reference frame from the clean SR input contract."""
    if "is_sr_usable" in audit_df.columns:
        subset = audit_df[_bool_series(audit_df["is_sr_usable"])].copy()
    elif "is_main_session" in audit_df.columns:
        subset = audit_df[_bool_series(audit_df["is_main_session"])].copy()
    elif "session" in audit_df.columns:
        subset = audit_df[audit_df["session"].astype(int).eq(2)].copy()
    else:
        raise ValueError("frame audit must include is_sr_usable, is_main_session, or session")
    if subset.empty:
        raise ValueError("No clean SR reference frames found")
    subset = subset.sort_values(["acquisition_order", "file"]).reset_index(drop=True)
    return subset.iloc[len(subset) // 2]


def preload_frames(data_dir: Path, filenames: Iterable[str]) -> dict[str, np.ndarray]:
    """Load every requested frame once into a float32 cache."""
    cache: dict[str, np.ndarray] = {}
    for name in sorted(set(str(v) for v in filenames)):
        cache[name] = load_frame(Path(data_dir) / name).astype(np.float32, copy=False)
    return cache


def _segment_value(segment: dict | pd.Series, key: str, default=np.nan):
    if isinstance(segment, pd.Series):
        return segment[key] if key in segment.index else default
    return segment.get(key, default)


def _segment_bool(segment: dict | pd.Series, key: str, default: bool = False) -> bool:
    value = _segment_value(segment, key, default)
    if isinstance(value, str):
        return value.lower() in {"true", "1", "yes"}
    return bool(value)


def _base_result(
    segment_id: int,
    segment: dict | pd.Series,
    scanline_files: list[str],
) -> dict:
    coords = [parse_filename(name) for name in scanline_files]
    y_values = [coord[1] for coord in coords if coord is not None]
    x_values = [coord[0] for coord in coords if coord is not None]
    return {
        "segment_id": int(segment_id),
        "scanline_Y": float(y_values[0]) if y_values else np.nan,
        "scanline_y_um": float(y_values[0]) if y_values else np.nan,
        "scanline_x_min_um": float(np.min(x_values)) if x_values else np.nan,
        "scanline_x_max_um": float(np.max(x_values)) if x_values else np.nan,
        "n_frames": int(len(scanline_files)),
        "x_px": float(_segment_value(segment, "x_px")),
        "y_px": float(_segment_value(segment, "y_px")),
        "nx": float(_segment_value(segment, "nx")),
        "ny": float(_segment_value(segment, "ny")),
        "normal_projection": float(_segment_value(segment, "normal_projection")),
        "normal_angle_deg": float(_segment_value(segment, "normal_angle_deg")),
        "curvature_proxy": float(_segment_value(segment, "curvature_proxy")),
        "segment_delta_t_c": float(_segment_value(segment, "delta_t_c")),
        "segment_abs_delta_t_c": float(_segment_value(segment, "abs_delta_t_c")),
        "snr": float(_segment_value(segment, "snr")),
        "is_a_class": _segment_bool(segment, "is_a_class"),
        "quality_label": str(_segment_value(segment, "quality_label", "")),
        "ncc_quality": np.nan,
        "median_ncc_peak": np.nan,
        "min_ncc_peak": np.nan,
        "ncc_fit_ok_fraction": np.nan,
        "ncc_edge_peak_fraction": np.nan,
        "cumulative_shift_px": np.nan,
        "phase_coverage_px": np.nan,
        "stage_shift_range_px": np.nan,
        "fitted_sigma_px": np.nan,
        "fitted_delta_t": np.nan,
        "joint_s_px": np.nan,
        "joint_rms_c": np.nan,
        "odd_s_px": np.nan,
        "even_s_px": np.nan,
        "split_half_diff_px": np.nan,
        "crb_px": np.nan,
        "crb_ratio": np.nan,
        "psf_sensitivity_px": np.nan,
        "psf_s_low_px": np.nan,
        "psf_s_high_px": np.nan,
        "pass_fail": False,
        "fail_reason": "",
        "error_detail": "",
    }


def _short_exception_detail(exc: Exception, *, max_chars: int = 160) -> str:
    """Return a compact, CSV-safe exception detail for batch diagnostics."""
    message = str(exc).strip().replace("\n", " ").replace("\r", " ").replace(";", ",")
    while "  " in message:
        message = message.replace("  ", " ")
    if len(message) > max_chars:
        message = f"{message[: max_chars - 3]}..."
    return message


def _fail(result: dict, reason: str, error_detail: str = "") -> dict:
    result["pass_fail"] = False
    result["fail_reason"] = reason
    result["error_detail"] = str(error_detail)
    return result


def _centered_roi_slices(
    shape: tuple[int, int],
    x_px: float,
    y_px: float,
    roi_size: int,
) -> tuple[slice, slice] | None:
    rows, cols = shape
    half = roi_size / 2.0
    if x_px - half < 0 or x_px + half >= cols or y_px - half < 0 or y_px + half >= rows:
        return None
    c0 = int(round(x_px - half))
    r0 = int(round(y_px - half))
    return slice(r0, r0 + roi_size), slice(c0, c0 + roi_size)


def _stage_normal_deltas(
    scanline_files: list[str],
    normal: np.ndarray,
    *,
    theta_deg: float,
    pixel_size_um: float,
) -> np.ndarray:
    coords = [parse_filename(name) for name in scanline_files]
    if any(coord is None for coord in coords):
        return np.full(len(scanline_files), np.nan, dtype=float)
    x = np.asarray([coord[0] for coord in coords], dtype=float)
    y = np.asarray([coord[1] for coord in coords], dtype=float)
    dx, dy = coordinate_to_shift(x, y, theta_deg=theta_deg, pixel_size_um=pixel_size_um)
    dx = dx - float(dx[0])
    dy = dy - float(dy[0])
    return dx * normal[0] + dy * normal[1]


def _local_ncc_deltas(
    frame_list: list[np.ndarray],
    roi: tuple[slice, slice],
    normal: np.ndarray,
    *,
    config: Ep04Config,
) -> tuple[np.ndarray, pd.DataFrame]:
    cumulative_dx = 0.0
    cumulative_dy = 0.0
    deltas = [0.0]
    rows = []
    for frame_index in range(1, len(frame_list)):
        estimate = subpixel_ncc(
            frame_list[frame_index - 1],
            frame_list[frame_index],
            roi=roi,
            search_radius=config.ncc_search_radius,
            fit_radius=config.ncc_fit_radius,
            preprocess=config.ncc_preprocess,
            highpass_sigma=config.ncc_highpass_sigma,
        )
        cumulative_dx += float(estimate["dx_px"])
        cumulative_dy += float(estimate["dy_px"])
        delta_n = cumulative_dx * normal[0] + cumulative_dy * normal[1]
        deltas.append(float(delta_n))
        rows.append(
            {
                "frame_index": int(frame_index),
                "pair_dx_px": float(estimate["dx_px"]),
                "pair_dy_px": float(estimate["dy_px"]),
                "ncc_delta_n_px": float(delta_n),
                "cumulative_dx_px": float(cumulative_dx),
                "cumulative_dy_px": float(cumulative_dy),
                "peak_ncc": float(estimate["peak_ncc"]),
                "fit_ok": bool(estimate["fit_ok"]),
                "edge_peak": bool(estimate["edge_peak"]),
            }
        )
    return np.asarray(deltas, dtype=float), pd.DataFrame(rows)


def _normal_shift_metrics(delta_n_px: np.ndarray) -> tuple[float, float]:
    """Return absolute normal-path length and normal phase coverage."""
    values = np.asarray(delta_n_px, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return np.nan, np.nan
    cumulative_path = float(np.sum(np.abs(np.diff(values)))) if values.size >= 2 else 0.0
    phase_coverage = float(np.ptp(values))
    return cumulative_path, phase_coverage


def _compute_ncc_phase(
    frame_list: list[np.ndarray],
    scanline_files: list[str],
    roi: tuple[slice, slice],
    normal: np.ndarray,
    *,
    config: Ep04Config,
) -> dict:
    ncc_deltas, ncc_detail = _local_ncc_deltas(frame_list, roi, normal, config=config)
    stage_deltas = _stage_normal_deltas(
        scanline_files,
        normal,
        theta_deg=config.theta_deg,
        pixel_size_um=config.pixel_size_um,
    )
    cumulative_shift, phase_coverage = _normal_shift_metrics(ncc_deltas)
    return {
        "ncc_deltas": ncc_deltas,
        "ncc_detail": ncc_detail,
        "cumulative_shift_px": cumulative_shift,
        "phase_coverage_px": phase_coverage,
        "stage_shift_range_px": float(np.ptp(stage_deltas)),
    }


def _profiles_from_frames(
    frame_list: list[np.ndarray],
    scanline_files: list[str],
    segment: dict | pd.Series,
    delta_n_px: np.ndarray,
    *,
    config: Ep04Config,
) -> pd.DataFrame:
    frames = []
    for frame_index, (frame, filename) in enumerate(zip(frame_list, scanline_files)):
        profile = extract_normal_profile(
            frame,
            pd.Series(segment),
            half_width_px=float(config.esf_half_width),
            step_px=float(config.esf_step_px),
        )
        coord = parse_filename(filename)
        x_um, y_um = (np.nan, np.nan) if coord is None else (coord[0], coord[1])
        profile["frame_index"] = int(frame_index)
        profile["file"] = str(filename)
        profile["X_um"] = float(x_um)
        profile["Y_um"] = float(y_um)
        profile["delta_n_px"] = float(delta_n_px[frame_index])
        profile["shift_model"] = "highpass_ncc"
        frames.append(profile)
    return pd.concat(frames, ignore_index=True)


def _fit_and_split_half(profiles: pd.DataFrame) -> tuple[dict, dict, dict]:
    """Fit the full joint ESF model and odd/even split halves."""
    full_fit = fit_joint_esf(profiles)
    frame_ids = sorted(int(v) for v in profiles["frame_index"].unique())
    odd_fit = fit_joint_esf(profiles, frame_indices=[v for v in frame_ids if v % 2 == 1])
    even_fit = fit_joint_esf(profiles, frame_indices=[v for v in frame_ids if v % 2 == 0])
    return full_fit, odd_fit, even_fit


def _joint_delta_t(fit: dict) -> float:
    params = np.asarray(fit["result"].x, dtype=float)
    n_frames = int(fit["n_frames"])
    per_frame = params[2:].reshape(n_frames, 3)
    return float(np.nanmedian(np.abs(per_frame[:, 2])))


def fit_joint_esf_fixed_sigma(
    multiframe_profiles: pd.DataFrame,
    fixed_sigma: float,
    *,
    frame_indices: list[int] | None = None,
) -> dict:
    """Jointly fit multi-frame ESF profiles with fixed shared sigma."""
    data = multiframe_profiles.copy()
    if frame_indices is not None:
        data = data[data["frame_index"].isin(frame_indices)].copy()
    frames = sorted(data["frame_index"].unique())
    n_frames = len(frames)

    single_inits = []
    frame_stats = []
    for frame_id in frames:
        group = data[data["frame_index"].eq(frame_id)]
        fit = fit_esf_profile(
            group["u_px"].to_numpy(),
            group["temperature_c"].to_numpy(),
            segment_id=int(group["segment_id"].iloc[0]),
        )
        delta = float(group["delta_n_px"].iloc[0])
        single_inits.append(fit.s - delta)
        frame_stats.append((fit.a, fit.b, fit.c))

    x0 = [float(np.nanmedian(single_inits))]
    for a, b, c in frame_stats:
        x0.extend([a, b, c])
    x0 = np.asarray(x0, dtype=float)

    temp_min = float(data["temperature_c"].min())
    temp_max = float(data["temperature_c"].max())
    lower = [-6.0]
    upper = [6.0]
    for _ in frames:
        lower.extend([temp_min - 2.0, -0.5, -8.0])
        upper.extend([temp_max + 2.0, 0.5, 8.0])
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)

    frame_to_pos = {frame_id: i for i, frame_id in enumerate(frames)}
    u = data["u_px"].to_numpy(dtype=float)
    temp = data["temperature_c"].to_numpy(dtype=float)
    delta_n = data["delta_n_px"].to_numpy(dtype=float)
    frame_pos = data["frame_index"].map(frame_to_pos).to_numpy(dtype=int)
    sigma = max(float(fixed_sigma), 1e-6)

    def residual(params: np.ndarray) -> np.ndarray:
        s = params[0]
        per_frame = params[1:].reshape(n_frames, 3)
        a = per_frame[frame_pos, 0]
        b = per_frame[frame_pos, 1]
        c = per_frame[frame_pos, 2]
        pred = a + b * u + c * normal_cdf((u - s - delta_n) / sigma)
        return pred - temp

    result = least_squares(residual, x0, bounds=(lower, upper), max_nfev=8000)
    resid = residual(result.x)
    return {
        "frames": frames,
        "n_frames": n_frames,
        "result": result,
        "s_px": float(result.x[0]),
        "sigma_px": sigma,
        "rms_c": float(np.sqrt(np.mean(resid * resid))),
        "residuals": resid,
        "data": data,
        "success": bool(result.success),
    }


def _psf_sensitivity(multiframe_profiles: pd.DataFrame, full_fit: dict, *, config: Ep04Config) -> tuple[float, float, float]:
    sigma = float(full_fit["sigma_px"])
    sigma_low = max(0.15, sigma - float(config.psf_delta_sigma_px))
    sigma_high = sigma + float(config.psf_delta_sigma_px)
    low_fit = fit_joint_esf_fixed_sigma(multiframe_profiles, sigma_low)
    high_fit = fit_joint_esf_fixed_sigma(multiframe_profiles, sigma_high)
    s_full = float(full_fit["s_px"])
    sensitivity = max(abs(float(low_fit["s_px"]) - s_full), abs(float(high_fit["s_px"]) - s_full))
    return float(sensitivity), float(low_fit["s_px"]), float(high_fit["s_px"])


def _quality_reasons(
    result: dict,
    *,
    gates: dict,
    include_psf: bool = True,
) -> list[str]:
    reasons: list[str] = []
    if float(result["segment_abs_delta_t_c"]) < float(gates["min_delta_t"]):
        reasons.append("low_delta_t")
    if float(result["snr"]) < float(gates["min_snr"]):
        reasons.append("low_snr")
    if not np.isfinite(result["ncc_quality"]) or float(result["ncc_quality"]) < float(gates["min_ncc_peak"]):
        reasons.append("ncc_unreliable")
    if np.isfinite(result["ncc_fit_ok_fraction"]) and float(result["ncc_fit_ok_fraction"]) < 0.80:
        reasons.append("ncc_unreliable")
    if np.isfinite(result["ncc_edge_peak_fraction"]) and float(result["ncc_edge_peak_fraction"]) > 0.0:
        reasons.append("ncc_unreliable")
    if not np.isfinite(result["phase_coverage_px"]) or float(result["phase_coverage_px"]) < float(gates["min_phase_range_px"]):
        reasons.append("low_phase_coverage")
    sigma_lo, sigma_hi = gates["sigma_range"]
    if not np.isfinite(result["fitted_sigma_px"]) or not (float(sigma_lo) <= float(result["fitted_sigma_px"]) <= float(sigma_hi)):
        reasons.append("sigma_out_of_range")
    if include_psf and (
        not np.isfinite(result["psf_sensitivity_px"])
        or float(result["psf_sensitivity_px"]) > float(gates["max_psf_sensitivity"])
    ):
        reasons.append("psf_sensitivity_high")
    split_limit = float(gates["max_split_half_a"] if result["is_a_class"] else gates["max_split_half_b"])
    if not np.isfinite(result["split_half_diff_px"]) or float(result["split_half_diff_px"]) > split_limit:
        reasons.append("split_half_high")
    return list(dict.fromkeys(reasons))


def _record_fit_metrics(
    result: dict,
    profiles: pd.DataFrame,
    full_fit: dict,
    odd_fit: dict,
    even_fit: dict,
    ncc_deltas: np.ndarray,
    *,
    config: Ep04Config,
    gates: dict,
    compute_psf_for_all: bool = False,
) -> None:
    result["joint_s_px"] = float(full_fit["s_px"])
    result["fitted_sigma_px"] = float(full_fit["sigma_px"])
    result["fitted_delta_t"] = _joint_delta_t(full_fit)
    result["joint_rms_c"] = float(full_fit["rms_c"])
    result["odd_s_px"] = float(odd_fit["s_px"])
    result["even_s_px"] = float(even_fit["s_px"])
    result["split_half_diff_px"] = abs(float(odd_fit["s_px"]) - float(even_fit["s_px"]))
    result["crb_px"] = crb_multi_frame(
        abs(float(result["fitted_delta_t"])),
        float(result["fitted_sigma_px"]),
        config.noise_floor_c,
        ncc_deltas,
    )
    result["crb_ratio"] = (
        float(result["split_half_diff_px"]) / float(result["crb_px"])
        if np.isfinite(result["crb_px"]) and float(result["crb_px"]) > 0.0
        else np.nan
    )

    pre_psf_reasons = _quality_reasons(result, gates=gates, include_psf=False)
    if not pre_psf_reasons or compute_psf_for_all:
        sensitivity, s_low, s_high = _psf_sensitivity(profiles, full_fit, config=config)
        result["psf_sensitivity_px"] = sensitivity
        result["psf_s_low_px"] = s_low
        result["psf_s_high_px"] = s_high


def run_segment_validation(
    segment_id: int,
    segment_row: dict,
    frames: dict[str, np.ndarray],
    scanline_files: list[str],
    theta_deg: float = 47.6,
    pixel_size_um: float = 20.0,
    ncc_roi_size: int = 40,
    ncc_preprocess: str = "highpass",
    esf_half_width: int = 10,
    **kwargs,
) -> dict:
    """Run the data-driven joint ESF validation for one segment and scanline."""
    config = Ep04Config(
        theta_deg=float(theta_deg),
        pixel_size_um=float(pixel_size_um),
        noise_floor_c=float(kwargs.get("noise_floor_c", kwargs.get("noise_sigma", NOISE_FLOOR_C))),
        ncc_roi_size=int(ncc_roi_size),
        ncc_preprocess=str(ncc_preprocess),
        ncc_search_radius=int(kwargs.get("ncc_search_radius", 2)),
        ncc_fit_radius=int(kwargs.get("ncc_fit_radius", 1)),
        ncc_highpass_sigma=float(kwargs.get("ncc_highpass_sigma", 6.0)),
        esf_half_width=int(esf_half_width),
        esf_step_px=float(kwargs.get("esf_step_px", 0.5)),
        psf_delta_sigma_px=float(kwargs.get("psf_delta_sigma_px", 0.15)),
    )
    gates = QUALITY_GATES.copy()
    gates.update(kwargs.get("quality_gates", {}))
    result = _base_result(segment_id, segment_row, scanline_files)

    try:
        frame_list = [frames[str(name)] for name in scanline_files]
    except KeyError as exc:
        return _fail(result, f"missing_frame:{exc}")
    if len(frame_list) < 4:
        return _fail(result, "too_few_frames")

    x_px = float(_segment_value(segment_row, "x_px"))
    y_px = float(_segment_value(segment_row, "y_px"))
    roi = _centered_roi_slices(frame_list[0].shape, x_px, y_px, config.ncc_roi_size)
    if roi is None:
        return _fail(result, "edge_of_frame")

    normal = np.array(
        [float(_segment_value(segment_row, "nx")), float(_segment_value(segment_row, "ny"))],
        dtype=float,
    )
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal_norm) or normal_norm == 0.0:
        return _fail(result, "invalid_normal")
    normal /= normal_norm

    try:
        phase = _compute_ncc_phase(
            frame_list,
            scanline_files,
            roi,
            normal,
            config=config,
        )
        ncc_deltas = phase["ncc_deltas"]
        ncc_detail = phase["ncc_detail"]
        result["cumulative_shift_px"] = phase["cumulative_shift_px"]
        result["phase_coverage_px"] = phase["phase_coverage_px"]
        result["stage_shift_range_px"] = phase["stage_shift_range_px"]
        if ncc_detail.empty:
            return _fail(result, "ncc_unreliable")
        result["ncc_quality"] = float(ncc_detail["peak_ncc"].median())
        result["median_ncc_peak"] = float(ncc_detail["peak_ncc"].median())
        result["min_ncc_peak"] = float(ncc_detail["peak_ncc"].min())
        result["ncc_fit_ok_fraction"] = float(ncc_detail["fit_ok"].mean())
        result["ncc_edge_peak_fraction"] = float(ncc_detail["edge_peak"].mean())

        profiles = _profiles_from_frames(
            frame_list,
            scanline_files,
            pd.Series(segment_row),
            ncc_deltas,
            config=config,
        )
        full_fit, odd_fit, even_fit = _fit_and_split_half(profiles)
        _record_fit_metrics(
            result,
            profiles,
            full_fit,
            odd_fit,
            even_fit,
            ncc_deltas,
            config=config,
            gates=gates,
            compute_psf_for_all=bool(kwargs.get("compute_psf_for_all", False)),
        )
    except Exception as exc:  # noqa: BLE001 - batch validation must return a row for every attempted segment.
        return _fail(result, f"fit_error:{type(exc).__name__}", _short_exception_detail(exc))

    reasons = _quality_reasons(result, gates=gates, include_psf=np.isfinite(result["psf_sensitivity_px"]))
    result["pass_fail"] = len(reasons) == 0
    result["fail_reason"] = "pass" if not reasons else ";".join(reasons)
    return result


def summarize_segments(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate segment-scanline validation rows into one row per segment."""
    if results.empty:
        return pd.DataFrame()

    rows = []
    for segment_id, group in results.groupby("segment_id", sort=True):
        first = group.iloc[0]
        finite_split = group["split_half_diff_px"].dropna()
        segment_pass = bool(group["pass_fail"].astype(bool).mean() >= 0.5)
        failed = group[~group["pass_fail"].astype(bool)]
        fail_reason = "pass"
        if not segment_pass and not failed.empty:
            reasons = failed["fail_reason"].astype(str).str.split(";").explode()
            reasons = reasons[reasons.ne("pass") & reasons.ne("")]
            fail_reason = str(reasons.value_counts().idxmax()) if not reasons.empty else "unknown"
        rows.append(
            {
                "segment_id": int(segment_id),
                "x_px": float(first["x_px"]),
                "y_px": float(first["y_px"]),
                "nx": float(first["nx"]),
                "ny": float(first["ny"]),
                "normal_projection": float(first["normal_projection"]),
                "normal_angle_deg": float(first["normal_angle_deg"]),
                "curvature_proxy": float(first["curvature_proxy"]),
                "segment_abs_delta_t_c": float(first["segment_abs_delta_t_c"]),
                "snr": float(first["snr"]),
                "is_a_class": bool(first["is_a_class"]),
                "quality_label": str(first["quality_label"]),
                "n_scanlines": int(len(group)),
                "n_pass": int(group["pass_fail"].astype(bool).sum()),
                "pass_rate": float(group["pass_fail"].astype(bool).mean()),
                "pass_fail": segment_pass,
                "split_half_median_px": float(finite_split.median()) if not finite_split.empty else np.nan,
                "split_half_p90_px": float(finite_split.quantile(0.90)) if len(finite_split) >= 2 else np.nan,
                "crb_median_px": float(group["crb_px"].median()),
                "crb_ratio_median": float(group["crb_ratio"].median()),
                "phase_coverage_median_px": float(group["phase_coverage_px"].median()),
                "ncc_quality_median": float(group["ncc_quality"].median()),
                "joint_rms_median_c": float(group["joint_rms_c"].median()),
                "fitted_sigma_median_px": float(group["fitted_sigma_px"].median()),
                "fitted_delta_t_median_c": float(group["fitted_delta_t"].median()),
                "psf_sensitivity_median_px": float(group["psf_sensitivity_px"].median()),
                "edge_position_median_px": float(group["joint_s_px"].median()),
                "edge_position_iqr_px": float(group["joint_s_px"].quantile(0.75) - group["joint_s_px"].quantile(0.25)),
                "fail_reason_primary": fail_reason,
            }
        )
    return pd.DataFrame(rows)


def global_summary_dict(results: pd.DataFrame, segment_summary: pd.DataFrame) -> dict:
    """Build compact global EP04 summary statistics."""
    if results.empty or segment_summary.empty:
        return {}
    a_segments = segment_summary[segment_summary["is_a_class"].astype(bool)].copy()
    a_rows = results[results["is_a_class"].astype(bool)].copy()
    return {
        "n_rows": int(len(results)),
        "n_segments": int(segment_summary["segment_id"].nunique()),
        "n_scanlines": int(results["scanline_y_um"].nunique()),
        "n_a_class_segments": int(len(a_segments)),
        "segment_pass_rate": float(segment_summary["pass_fail"].mean()),
        "a_class_segment_pass_rate": float(a_segments["pass_fail"].mean()) if not a_segments.empty else np.nan,
        "row_pass_rate": float(results["pass_fail"].mean()),
        "a_class_row_pass_rate": float(a_rows["pass_fail"].mean()) if not a_rows.empty else np.nan,
        "a_class_split_half_median_px": float(a_segments["split_half_median_px"].median()) if not a_segments.empty else np.nan,
        "a_class_split_half_p90_px": float(a_segments["split_half_median_px"].quantile(0.90)) if not a_segments.empty else np.nan,
        "a_class_crb_ratio_median": float(a_segments["crb_ratio_median"].median()) if not a_segments.empty else np.nan,
        "a_class_phase_coverage_median_px": float(a_segments["phase_coverage_median_px"].median()) if not a_segments.empty else np.nan,
        "a_class_ncc_quality_median": float(a_segments["ncc_quality_median"].median()) if not a_segments.empty else np.nan,
        "all_split_half_median_px": float(segment_summary["split_half_median_px"].median()),
        "all_crb_ratio_median": float(segment_summary["crb_ratio_median"].median()),
    }


def save_validation_outputs(
    results: pd.DataFrame,
    output_dir: Path,
    *,
    extra_summary: dict | None = None,
    output_prefix: str = "",
) -> tuple[pd.DataFrame, dict]:
    """Save required EP04 CSV/JSON analysis outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = str(output_prefix)
    results.to_csv(output_dir / f"{prefix}segment_validation_results.csv", index=False)
    segment_summary = summarize_segments(results)
    segment_summary.to_csv(output_dir / f"{prefix}segment_summary.csv", index=False)
    summary = global_summary_dict(results, segment_summary)
    if extra_summary:
        summary.update(extra_summary)
    with open(output_dir / f"{prefix}global_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    return segment_summary, summary


def run_all_segments(
    segments_csv: Path,
    frame_audit_csv: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    min_snr: float = 8.0,
    min_delta_t: float = 0.5,
    max_split_half: float = 0.06,
    n_jobs: int = 1,
    theta_deg: float = 47.6,
    pixel_size_um: float = 20.0,
    noise_floor_c: float = NOISE_FLOOR_C,
    ncc_roi_size: int = 40,
    ncc_preprocess: str = "highpass",
    esf_half_width: int = 10,
    limit_segments: int | None = None,
    limit_scanlines: int | None = None,
    show_progress: bool = False,
    output_prefix: str = "",
    save_outputs: bool = True,
    extra_summary: dict | None = None,
) -> pd.DataFrame:
    """Run EP04 validation for every segment in an EP03-compatible CSV."""
    segments = pd.read_csv(segments_csv)
    if limit_segments is not None:
        segments = segments.sort_values("segment_id").head(int(limit_segments)).copy()
    audit = pd.read_csv(frame_audit_csv)
    scanlines = build_x_scanlines(audit)
    if limit_scanlines is not None:
        scanlines = scanlines[: int(limit_scanlines)]
    if not scanlines:
        raise RuntimeError("No complete session=2/R=0 X scanlines found")

    scanline_files = [[str(v) for v in line["file"].tolist()] for line in scanlines]
    filenames = [name for files in scanline_files for name in files]
    frame_cache = preload_frames(Path(data_dir), filenames)

    gates = QUALITY_GATES.copy()
    gates["min_snr"] = float(min_snr)
    gates["min_delta_t"] = float(min_delta_t)
    gates["max_split_half_b"] = float(max_split_half)

    tasks = [
        (int(segment["segment_id"]), segment.to_dict(), files)
        for _, segment in segments.sort_values("segment_id").iterrows()
        for files in scanline_files
    ]

    def evaluate(task: tuple[int, dict, list[str]]) -> dict:
        segment_id, segment_row, files = task
        return run_segment_validation(
            segment_id,
            segment_row,
            frame_cache,
            files,
            theta_deg=theta_deg,
            pixel_size_um=pixel_size_um,
            ncc_roi_size=ncc_roi_size,
            ncc_preprocess=ncc_preprocess,
            esf_half_width=esf_half_width,
            noise_floor_c=noise_floor_c,
            quality_gates=gates,
        )

    if int(n_jobs) <= 1:
        iterator = tqdm(tasks, desc="EP04 validation", unit="fit") if show_progress else tasks
        rows = [evaluate(task) for task in iterator]
    else:
        rows = []
        with ThreadPoolExecutor(max_workers=int(n_jobs)) as executor:
            futures = [executor.submit(evaluate, task) for task in tasks]
            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(futures), desc="EP04 validation", unit="fit")
            for future in iterator:
                rows.append(future.result())

    results = pd.DataFrame(rows).sort_values(["segment_id", "scanline_y_um"]).reset_index(drop=True)
    if save_outputs:
        save_validation_outputs(
            results,
            Path(output_dir),
            output_prefix=output_prefix,
            extra_summary=extra_summary,
        )
    return results


def run_all_inner_segments(
    segments_csv: Path,
    frame_audit_csv: Path,
    data_dir: Path,
    output_dir: Path,
    **kwargs,
) -> pd.DataFrame:
    """Run EP04 validation for inner contour segments with inner-prefixed outputs."""
    kwargs.setdefault("output_prefix", "inner_")
    return run_all_segments(segments_csv, frame_audit_csv, data_dir, output_dir, **kwargs)


def plot_split_half_distribution(
    segment_summary: pd.DataFrame,
    *,
    segment34_value: float = 0.0216,
) -> plt.Figure:
    """Plot A-class segment split-half distribution."""
    data = segment_summary[segment_summary["is_a_class"].astype(bool)].copy()
    values = data["split_half_median_px"].dropna().to_numpy(dtype=float)
    fig, ax = make_figure("single_col", height=3.0)
    bins = np.linspace(0.0, max(0.12, float(np.nanmax(values)) * 1.05 if values.size else 0.12), 24)
    ax.hist(values, bins=bins, color=METHOD_COLOR_LIST[0], alpha=0.82, edgecolor="white")
    if values.size:
        median = float(np.nanmedian(values))
        p90 = float(np.nanquantile(values, 0.90))
        crb = float(data["crb_median_px"].median())
        ax.axvline(median, color="#222222", linestyle="--", linewidth=0.9, label=f"median={median:.3f} px")
        ax.axvline(p90, color=METHOD_COLOR_LIST[4], linestyle="--", linewidth=0.9, label=f"P90={p90:.3f} px")
        ax.axvline(crb, color=METHOD_COLOR_LIST[1], linestyle=":", linewidth=1.1, label=f"median CRB={crb:.3f} px")
    ax.axvline(segment34_value, color=METHOD_COLOR_LIST[2], linestyle="-.", linewidth=1.0, label="EP03 seg34=0.022 px")
    ax.set_xlabel("Median split-half difference [px]")
    ax.set_ylabel("A-class segment count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=7)
    return fig


def plot_crb_ratio_scatter(segment_summary: pd.DataFrame) -> plt.Figure:
    """Plot per-segment split-half / CRB ratio."""
    data = segment_summary.sort_values("segment_id").copy()
    colors = np.where(data["pass_fail"].astype(bool), METHOD_COLOR_LIST[1], METHOD_COLOR_LIST[2])
    fig, ax = make_figure("double_col", height=3.4)
    ax.scatter(
        data["segment_id"],
        data["crb_ratio_median"],
        s=24,
        c=colors,
        edgecolor="white",
        linewidth=0.25,
        alpha=0.86,
    )
    for value, label in [(1.0, "CRB"), (1.5, "1.5x"), (3.0, "3x")]:
        ax.axhline(value, color="#666666", linestyle="--", linewidth=0.8)
        ax.text(float(data["segment_id"].min()), value * 1.03, label, fontsize=7, color="#444444", va="bottom")
    ax.set_xlabel("Segment ID")
    ax.set_ylabel("Median split-half / CRB")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.scatter([], [], c=METHOD_COLOR_LIST[1], s=24, label="pass")
    ax.scatter([], [], c=METHOD_COLOR_LIST[2], s=24, label="fail")
    ax.legend(loc="upper right", fontsize=7)
    return fig


def plot_pass_fail_contour_map(
    reference_frame: np.ndarray,
    segment_summary: pd.DataFrame,
) -> plt.Figure:
    """Overlay pass/fail segment status on the reference thermal frame."""
    data = segment_summary.sort_values("segment_id").copy()
    fig, ax = make_figure("double_col", height=4.4)
    im = ax.imshow(reference_frame, cmap="inferno", origin="upper")
    contour_like = data.sort_values("segment_id")
    ax.plot(contour_like["x_px"], contour_like["y_px"], color="#dddddd", linewidth=0.7, alpha=0.75, label="outer segment path")
    precision = data["split_half_median_px"].to_numpy(dtype=float)
    precision_scale = np.nan_to_num(precision, nan=np.nanmedian(precision))
    sizes = 16 + 48 * (1.0 - np.clip(precision_scale / max(np.nanpercentile(precision_scale, 90), 1e-6), 0.0, 1.0))
    passed = data["pass_fail"].astype(bool)
    ax.scatter(
        data.loc[~passed, "x_px"],
        data.loc[~passed, "y_px"],
        s=sizes[~passed.to_numpy()],
        c=METHOD_COLOR_LIST[2],
        edgecolor="white",
        linewidth=0.25,
        alpha=0.85,
        label=f"fail (N={(~passed).sum()})",
    )
    ax.scatter(
        data.loc[passed, "x_px"],
        data.loc[passed, "y_px"],
        s=sizes[passed.to_numpy()],
        c=METHOD_COLOR_LIST[1],
        edgecolor="white",
        linewidth=0.25,
        alpha=0.90,
        label=f"pass (N={passed.sum()})",
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    format_colorbar(cbar, "Temperature [C]")
    ax.set_xlabel("Column [px]")
    ax.set_ylabel("Row [px]")
    ax.legend(loc="upper right", fontsize=7)
    return fig


def plot_phase_coverage_vs_precision(segment_summary: pd.DataFrame) -> plt.Figure:
    """Plot phase coverage against split-half repeatability."""
    data = segment_summary.copy()
    colors = np.where(data["pass_fail"].astype(bool), METHOD_COLOR_LIST[1], METHOD_COLOR_LIST[2])
    fig, ax = make_figure("single_col", height=3.1)
    ax.scatter(
        data["phase_coverage_median_px"],
        data["split_half_median_px"],
        s=22 + 0.9 * np.clip(data["snr"], 0, 60),
        c=colors,
        edgecolor="white",
        linewidth=0.25,
        alpha=0.82,
    )
    ax.axvline(QUALITY_GATES["min_phase_range_px"], color="#666666", linestyle="--", linewidth=0.9, label="phase gate")
    ax.axhline(QUALITY_GATES["max_split_half_a"], color="#444444", linestyle=":", linewidth=1.0, label="A split gate")
    ax.set_xlabel("Median normal phase coverage [px]")
    ax.set_ylabel("Median split-half difference [px]")
    ax.grid(alpha=0.25)
    ax.legend(loc="upper right", fontsize=7)
    return fig


def plot_failure_taxonomy(segment_summary: pd.DataFrame) -> plt.Figure:
    """Plot primary failure reason counts."""
    failed = segment_summary[~segment_summary["pass_fail"].astype(bool)].copy()
    counts = failed["fail_reason_primary"].replace("", "unknown").value_counts()
    fig, ax = make_figure("single_col", height=3.0)
    if counts.empty:
        ax.text(0.5, 0.5, "No failed segments", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return fig
    labels = counts.index.tolist()
    colors = [METHOD_COLOR_LIST[i % len(METHOD_COLOR_LIST)] for i in range(len(labels))]
    ax.bar(np.arange(len(labels)), counts.to_numpy(), color=colors, alpha=0.82)
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("Segment count")
    ax.grid(axis="y", alpha=0.25)
    return fig


def plot_cross_scanline_consistency(
    results: pd.DataFrame,
    segment_summary: pd.DataFrame,
    *,
    n_segments: int = 5,
) -> plt.Figure:
    """Plot edge-position estimates across scanlines for representative segments."""
    candidates = segment_summary[segment_summary["is_a_class"].astype(bool)].copy()
    if candidates.empty:
        candidates = segment_summary.copy()
    chosen_ids: list[int] = []
    if 34 in set(candidates["segment_id"].astype(int)):
        chosen_ids.append(34)
    ranked = candidates.sort_values(["pass_fail", "split_half_median_px"], ascending=[False, True])
    for segment_id in ranked["segment_id"].astype(int):
        if segment_id not in chosen_ids:
            chosen_ids.append(int(segment_id))
        if len(chosen_ids) >= n_segments:
            break

    fig, ax = make_figure("double_col", height=3.4)
    for i, segment_id in enumerate(chosen_ids):
        group = results[results["segment_id"].astype(int).eq(int(segment_id))].sort_values("scanline_y_um")
        group = group[np.isfinite(group["joint_s_px"])]
        if group.empty:
            continue
        y = group["joint_s_px"].to_numpy(dtype=float)
        y = y - float(np.nanmedian(y))
        color = METHOD_COLOR_LIST[i % len(METHOD_COLOR_LIST)]
        ax.plot(
            group["scanline_y_um"],
            y,
            marker="o",
            markersize=3.2,
            linewidth=1.0,
            color=color,
            label=f"seg {segment_id}",
        )
    ax.axhline(0.0, color="#555555", linewidth=0.8)
    ax.set_xlabel(r"Scanline Y [$\mu$m]")
    ax.set_ylabel("Joint edge position minus segment median [px]")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=7, ncol=2)
    return fig


def create_ep04_figures(
    results: pd.DataFrame,
    segment_summary: pd.DataFrame,
    reference_frame: np.ndarray,
    output_dir: Path,
) -> dict[str, Path]:
    """Create and save the six required EP04 figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "split_half_distribution.png": plot_split_half_distribution(segment_summary),
        "crb_ratio_scatter.png": plot_crb_ratio_scatter(segment_summary),
        "pass_fail_contour_map.png": plot_pass_fail_contour_map(reference_frame, segment_summary),
        "phase_coverage_vs_precision.png": plot_phase_coverage_vs_precision(segment_summary),
        "failure_taxonomy.png": plot_failure_taxonomy(segment_summary),
        "cross_scanline_consistency.png": plot_cross_scanline_consistency(results, segment_summary),
    }
    saved = {}
    for name, fig in figures.items():
        saved[name] = savefig_academic(fig, output_dir / name)
    return saved


def prepare_ep04_segment_inputs(
    frame_audit_csv: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    outer_segments_csv: Path | None = None,
    inner_segments_csv: Path | None = None,
    theta_deg: float = 47.6,
    noise_floor_c: float = NOISE_FLOOR_C,
    force: bool = False,
) -> dict[str, Path]:
    """Return outer/inner segment CSVs, generating EP04-local copies if needed.

    EP04 is allowed to consume EP03 segment files when they exist. When those
    outputs are absent on a fresh machine, this function recreates the same
    contour segmentation inputs inside ``output/ep04_global_validation/inputs``
    without writing to the EP03 output directory.
    """
    frame_audit_csv = Path(frame_audit_csv)
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    inputs_dir = output_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    outer_requested = Path(outer_segments_csv) if outer_segments_csv is not None else None
    inner_requested = Path(inner_segments_csv) if inner_segments_csv is not None else None
    outer_path = outer_requested if outer_requested is not None and outer_requested.exists() else inputs_dir / "contour_segments.csv"
    inner_path = inner_requested if inner_requested is not None and inner_requested.exists() else inputs_dir / "inner_contour_segments.csv"

    if not force and outer_path.exists() and inner_path.exists():
        return {"outer_segments_csv": outer_path, "inner_segments_csv": inner_path}

    audit = pd.read_csv(frame_audit_csv)
    reference_row = select_clean_sr_reference_row(audit)
    reference_frame = load_frame(data_dir / str(reference_row["file"]))

    all_segments, contour_summary, outer_mask, outer_contour, inner_contours = measure_contour_observability(
        reference_frame,
        theta_deg=float(theta_deg),
        noise_sigma_c=float(noise_floor_c),
    )
    all_segments = add_segment_quality_columns(all_segments)
    source = all_segments.get("source", pd.Series("", index=all_segments.index)).astype(str)
    outer_segments = all_segments[source.eq("outer")].copy()
    inner_segments = all_segments[source.eq("inner")].copy()

    if outer_requested is not None and outer_requested.exists() and not force:
        outer_path = outer_requested
    else:
        outer_segments.to_csv(inputs_dir / "contour_segments.csv", index=False)
        outer_path = inputs_dir / "contour_segments.csv"

    if inner_requested is not None and inner_requested.exists() and not force:
        inner_path = inner_requested
    else:
        inner_segments.to_csv(inputs_dir / "inner_contour_segments.csv", index=False)
        inner_path = inputs_dir / "inner_contour_segments.csv"

    inner_info_path = inputs_dir / "inner_contour_info.csv"
    inner_info = pd.DataFrame(
        {
            "contour_id": np.arange(len(inner_contours), dtype=int),
            "n_points": [int(len(contour)) for contour in inner_contours],
        }
    )
    inner_info.to_csv(inner_info_path, index=False)
    contour_summary.to_csv(inputs_dir / "contour_observability_summary.csv", index=False)
    metadata = {
        "reference_file": str(reference_row["file"]),
        "reference_acquisition_order": int(reference_row["acquisition_order"]),
        "theta_deg": float(theta_deg),
        "noise_floor_c": float(noise_floor_c),
        "outer_contour_points": int(len(outer_contour)),
        "outer_mask_pixels": int(np.count_nonzero(outer_mask)),
        "n_outer_segments": int(len(outer_segments)),
        "n_inner_segments": int(len(inner_segments)),
        "n_inner_contours": int(len(inner_contours)),
        "outer_segments_csv": str(outer_path),
        "inner_segments_csv": str(inner_path),
        "inner_contour_info_csv": str(inner_info_path),
    }
    with open(inputs_dir / "segment_input_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    return {"outer_segments_csv": outer_path, "inner_segments_csv": inner_path}


def _contour_summary_frame(outer_segment_summary: pd.DataFrame, inner_segment_summary: pd.DataFrame) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for contour, frame in [("outer", outer_segment_summary), ("inner", inner_segment_summary)]:
        if frame is None or frame.empty:
            continue
        copy = frame.copy()
        copy["contour"] = contour
        frames.append(copy)
    return pd.concat(frames, ignore_index=True, sort=False) if frames else pd.DataFrame()


def segment_quality_distribution_table(
    outer_segment_summary: pd.DataFrame,
    inner_segment_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize precision/CRB/SNR/pass-fail by contour family and quality label."""
    combined = _contour_summary_frame(outer_segment_summary, inner_segment_summary)
    if combined.empty:
        return pd.DataFrame()
    combined["pass_bool"] = _bool_mask(combined, "pass_fail")
    grouped = (
        combined.groupby(["contour", "quality_label"], dropna=False)
        .agg(
            n_segments=("segment_id", "count"),
            pass_rate=("pass_bool", "mean"),
            median_split_half_px=("split_half_median_px", "median"),
            median_crb_ratio=("crb_ratio_median", "median"),
            median_snr=("snr", "median"),
            median_phase_coverage_px=("phase_coverage_median_px", "median"),
        )
        .reset_index()
    )
    return grouped.sort_values(["contour", "quality_label"]).reset_index(drop=True)


def combined_anchor_summary_table(
    outer_segment_summary: pd.DataFrame,
    inner_segment_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build a compact inner/outer anchor benchmark table for notebook display."""

    def stats(frame: pd.DataFrame) -> dict:
        if frame is None or frame.empty:
            return {
                "segments": 0,
                "a_class": 0,
                "passed": 0,
                "pass_rate": np.nan,
                "a_pass_rate": np.nan,
                "split": np.nan,
                "crb": np.nan,
                "snr": np.nan,
            }
        passed = _bool_mask(frame, "pass_fail")
        a_class = _bool_mask(frame, "is_a_class")
        return {
            "segments": int(len(frame)),
            "a_class": int(a_class.sum()),
            "passed": int(passed.sum()),
            "pass_rate": float(passed.mean()),
            "a_pass_rate": float(passed[a_class].mean()) if int(a_class.sum()) else np.nan,
            "split": float(pd.to_numeric(frame.loc[passed, "split_half_median_px"], errors="coerce").median()),
            "crb": float(pd.to_numeric(frame.loc[passed, "crb_ratio_median"], errors="coerce").median()),
            "snr": float(pd.to_numeric(frame["snr"], errors="coerce").median()),
        }

    def fmt(value: float, decimals: int) -> str:
        return f"{value:.{decimals}f}" if np.isfinite(value) else "n/a"

    def pct(value: float) -> str:
        return f"{100.0 * value:.1f}%" if np.isfinite(value) else "n/a"

    outer = stats(outer_segment_summary)
    inner = stats(inner_segment_summary)
    combined = stats(_contour_summary_frame(outer_segment_summary, inner_segment_summary))
    rows = [
        ("Total segments", outer["segments"], inner["segments"], combined["segments"]),
        ("A-class segments", outer["a_class"], inner["a_class"], combined["a_class"]),
        ("Passed anchor segments", outer["passed"], inner["passed"], combined["passed"]),
        ("Anchor pass rate", pct(outer["pass_rate"]), pct(inner["pass_rate"]), pct(combined["pass_rate"])),
        ("A-class anchor pass rate", pct(outer["a_pass_rate"]), pct(inner["a_pass_rate"]), pct(combined["a_pass_rate"])),
        ("Passed median split-half [px]", fmt(outer["split"], 4), fmt(inner["split"], 4), fmt(combined["split"], 4)),
        ("Passed median CRB ratio", fmt(outer["crb"], 2), fmt(inner["crb"], 2), fmt(combined["crb"], 2)),
        ("Median input SNR", fmt(outer["snr"], 1), fmt(inner["snr"], 1), fmt(combined["snr"], 1)),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Outer", "Inner", "Combined"])


def _box_values(frame: pd.DataFrame, column: str, mask: pd.Series) -> np.ndarray:
    values = pd.to_numeric(frame.loc[mask, column], errors="coerce").to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    return values if values.size else np.asarray([np.nan])


def plot_global_segment_quality_distribution(
    outer_segment_summary: pd.DataFrame,
    inner_segment_summary: pd.DataFrame,
) -> plt.Figure:
    """Plot precision, CRB, SNR, and pass/fail distributions for outer/inner segments."""
    combined = _contour_summary_frame(outer_segment_summary, inner_segment_summary)
    combined["pass_bool"] = _bool_mask(combined, "pass_fail")
    fig, axes = make_figure("double_col", nrows=2, ncols=2, height=3.1)
    axes = np.asarray(axes).ravel()
    labels = ["Outer pass", "Outer fail", "Inner pass", "Inner fail"]
    masks = [
        combined["contour"].eq("outer") & combined["pass_bool"],
        combined["contour"].eq("outer") & ~combined["pass_bool"],
        combined["contour"].eq("inner") & combined["pass_bool"],
        combined["contour"].eq("inner") & ~combined["pass_bool"],
    ]
    colors = [EP04_OUTER_COLOR, "#AFC3DE", EP04_INNER_COLOR, "#E4B991"]

    for ax, column, ylabel, title, log_y in [
        (axes[0], "split_half_median_px", "Median split-half [px]", "(a) Localization precision", True),
        (axes[1], "crb_ratio_median", "Median split-half / CRB", "(b) CRB consistency", True),
        (axes[2], "snr", "SNR", "(c) Thermal contrast", False),
    ]:
        values = [_box_values(combined, column, mask) for mask in masks]
        box = ax.boxplot(values, labels=labels, patch_artist=True, showfliers=False)
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.72)
            patch.set_edgecolor("#30343B")
            patch.set_linewidth(0.8)
        for item in box["medians"]:
            item.set_color("#20242A")
            item.set_linewidth(1.0)
        for item in box["whiskers"] + box["caps"]:
            item.set_color("#30343B")
            item.set_linewidth(0.8)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=20)
        if log_y:
            ax.set_yscale("log")
        ax.grid(axis="y", alpha=0.25)

    ax = axes[3]
    counts = (
        combined.groupby(["contour", "quality_label", "pass_bool"])
        .size()
        .rename("n")
        .reset_index()
    )
    quality_order = ["A_high_snr_high_projection", "B_usable", "C_low_snr", "D_other"]
    x_labels = []
    pass_counts = []
    fail_counts = []
    for contour in ["outer", "inner"]:
        for quality in quality_order:
            sub = counts[counts["contour"].eq(contour) & counts["quality_label"].eq(quality)]
            if sub.empty:
                continue
            x_labels.append(f"{contour}\n{quality.split('_')[0]}")
            pass_counts.append(int(sub.loc[sub["pass_bool"], "n"].sum()))
            fail_counts.append(int(sub.loc[~sub["pass_bool"], "n"].sum()))
    x = np.arange(len(x_labels))
    ax.bar(
        x,
        pass_counts,
        color=EP04_OUTER_COLOR,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.88,
        label="anchor pass",
    )
    ax.bar(
        x,
        fail_counts,
        bottom=pass_counts,
        color=EP04_GATE_REJECT_COLOR,
        edgecolor="white",
        linewidth=0.35,
        alpha=0.86,
        label="gate reject",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    ax.set_ylabel("Segment count")
    ax.set_title("(d) Pass/fail by contour class")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=7)
    return fig


def _scanline_pass_matrix(results: pd.DataFrame, segment_summary: pd.DataFrame) -> tuple[np.ndarray, list[int], list[float]]:
    if results.empty or segment_summary.empty:
        return np.empty((0, 0)), [], []
    passed_segments = segment_summary.loc[_bool_mask(segment_summary, "pass_fail"), "segment_id"].astype(int).tolist()
    if not passed_segments:
        passed_segments = segment_summary.sort_values("segment_id")["segment_id"].astype(int).head(1).tolist()
    scanlines = sorted(pd.to_numeric(results["scanline_y_um"], errors="coerce").dropna().unique().tolist())
    matrix = np.full((len(passed_segments), len(scanlines)), np.nan, dtype=float)
    lookup = {
        (int(row["segment_id"]), float(row["scanline_y_um"])): bool(row["pass_fail"])
        for _, row in results.iterrows()
    }
    for i, segment_id in enumerate(passed_segments):
        for j, scanline_y in enumerate(scanlines):
            value = lookup.get((int(segment_id), float(scanline_y)))
            matrix[i, j] = np.nan if value is None else float(value)
    return matrix, passed_segments, scanlines


def _anchor_zoom_limits(
    reference_frame: np.ndarray,
    combined: pd.DataFrame,
    *,
    zoom: float = 2.0,
    y_screen_shift_px: float = 22.0,
) -> tuple[float, float, float, float]:
    """Return a zoomed crop around the chip structure for anchor maps."""
    height, width = reference_frame.shape[:2]
    if combined.empty or not {"x_px", "y_px"}.issubset(combined.columns):
        return 0.0, float(width), float(height), 0.0

    x = pd.to_numeric(combined["x_px"], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(combined["y_px"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(x) & np.isfinite(y)
    if not np.any(finite):
        return 0.0, float(width), float(height), 0.0

    x = x[finite]
    y = y[finite]
    pad_px = 18.0
    crop_w = min(float(width), max(float(width) / zoom, float(np.ptp(x)) + 2.0 * pad_px))
    crop_h = min(float(height), max(float(height) / zoom, float(np.ptp(y)) + 2.0 * pad_px))

    center_x = float(np.mean(x))
    # A smaller image-row crop center moves the chip lower on the rendered panel
    # while preserving the original image coordinate convention.
    center_y = float(np.mean(y)) - y_screen_shift_px

    x0 = float(np.clip(center_x - crop_w / 2.0, 0.0, max(float(width) - crop_w, 0.0)))
    y0 = float(np.clip(center_y - crop_h / 2.0, 0.0, max(float(height) - crop_h, 0.0)))
    return x0, x0 + crop_w, y0 + crop_h, y0


def _anchor_scanline_support_frame(
    results: pd.DataFrame,
    segment_summary: pd.DataFrame,
    *,
    contour: str,
) -> pd.DataFrame:
    columns = ["contour", "scanline_y_um", "evaluated_rows", "passing_rows", "support_rate"]
    if results.empty or segment_summary.empty:
        return pd.DataFrame(columns=columns)

    anchor_ids = (
        pd.to_numeric(segment_summary.loc[_bool_mask(segment_summary, "pass_fail"), "segment_id"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    if anchor_ids.size == 0:
        return pd.DataFrame(columns=columns)

    data = results.copy()
    data["segment_id_int"] = pd.to_numeric(data["segment_id"], errors="coerce")
    data["scanline_y_um"] = pd.to_numeric(data["scanline_y_um"], errors="coerce")
    data = data.loc[data["segment_id_int"].isin(anchor_ids)].dropna(subset=["scanline_y_um"])
    if data.empty:
        return pd.DataFrame(columns=columns)

    data["pass_bool"] = _bool_mask(data, "pass_fail")
    support = (
        data.groupby("scanline_y_um", dropna=False)
        .agg(evaluated_rows=("pass_bool", "size"), passing_rows=("pass_bool", "sum"))
        .reset_index()
    )
    support["support_rate"] = support["passing_rows"] / support["evaluated_rows"].clip(lower=1)
    support.insert(0, "contour", contour)
    return support[columns]


def plot_anchor_coverage_map(
    reference_frame: np.ndarray,
    outer_segment_summary: pd.DataFrame,
    inner_segment_summary: pd.DataFrame,
    outer_results: pd.DataFrame,
    inner_results: pd.DataFrame,
) -> plt.Figure:
    """Plot a zoomed spatial anchor coverage map on the reference temperature frame."""
    combined = _contour_summary_frame(outer_segment_summary, inner_segment_summary)
    combined["pass_bool"] = _bool_mask(combined, "pass_fail")
    fig, ax_map = make_figure("one_half_col", height=4.4)

    im = ax_map.imshow(reference_frame, cmap="inferno", origin="upper")
    marker_map = {"outer": "o", "inner": "s"}
    for contour, data in combined.groupby("contour", sort=False):
        marker = marker_map.get(contour, "o")
        passed = data["pass_bool"]
        ax_map.scatter(
            data.loc[~passed, "x_px"],
            data.loc[~passed, "y_px"],
            s=16,
            marker=marker,
            c=EP04_REJECT_COLOR,
            edgecolor="white",
            linewidth=0.25,
            alpha=0.70,
            label=f"{contour.capitalize()} Reject",
        )
        ax_map.scatter(
            data.loc[passed, "x_px"],
            data.loc[passed, "y_px"],
            s=30,
            marker=marker,
            c=EP04_ANCHOR_COLOR,
            edgecolor="white",
            linewidth=0.35,
            alpha=0.94,
            label=f"{contour.capitalize()} Anchor",
        )
    cbar = fig.colorbar(im, ax=ax_map, fraction=0.046, pad=0.02)
    format_colorbar(cbar, "Temperature [C]")
    x0, x1, y_bottom, y_top = _anchor_zoom_limits(reference_frame, combined)
    ax_map.set_xlim(x0, x1)
    ax_map.set_ylim(y_bottom, y_top)
    ax_map.set_xlabel("Column [px]")
    ax_map.set_ylabel("Row [px]")
    legend = ax_map.legend(
        loc="upper left",
        fontsize=8,
        ncol=2,
        frameon=True,
        facecolor="#241038",
        edgecolor="none",
        framealpha=0.72,
        handletextpad=0.45,
        columnspacing=0.9,
    )
    for text in legend.get_texts():
        text.set_color("white")
    return fig


def plot_anchor_scanline_support(
    outer_results: pd.DataFrame,
    inner_results: pd.DataFrame,
    outer_segment_summary: pd.DataFrame,
    inner_segment_summary: pd.DataFrame,
) -> plt.Figure:
    """Plot scanline-level support counts for accepted anchor segments."""
    support = pd.concat(
        [
            _anchor_scanline_support_frame(outer_results, outer_segment_summary, contour="outer"),
            _anchor_scanline_support_frame(inner_results, inner_segment_summary, contour="inner"),
        ],
        ignore_index=True,
    )
    fig, ax = make_figure("double_col", height=3.2)
    if support.empty:
        ax.text(0.5, 0.5, "No accepted anchor scanline support", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return fig

    scanlines = sorted(pd.to_numeric(support["scanline_y_um"], errors="coerce").dropna().unique().tolist())
    x = np.arange(len(scanlines))
    bar_width = 0.34
    offsets = {"outer": -bar_width / 2.0, "inner": bar_width / 2.0}
    colors = {"outer": EP04_OUTER_COLOR, "inner": EP04_INNER_COLOR}

    for contour in ["outer", "inner"]:
        contour_support = (
            support.loc[support["contour"].eq(contour)]
            .set_index("scanline_y_um")
            .reindex(scanlines)
            .fillna({"evaluated_rows": 0, "passing_rows": 0, "support_rate": 0.0})
        )
        evaluated = pd.to_numeric(contour_support["evaluated_rows"], errors="coerce").fillna(0).to_numpy(dtype=float)
        passing = pd.to_numeric(contour_support["passing_rows"], errors="coerce").fillna(0).to_numpy(dtype=float)
        offset = offsets[contour]
        ax.bar(
            x + offset,
            evaluated,
            width=bar_width,
            color=EP04_TOTAL_COLOR,
            edgecolor=EP04_TOTAL_EDGE,
            linewidth=0.45,
            label="Evaluated Anchor Checks" if contour == "outer" else None,
            zorder=1,
        )
        ax.bar(
            x + offset,
            passing,
            width=bar_width * 0.70,
            color=colors[contour],
            edgecolor="white",
            linewidth=0.35,
            label=f"{contour.capitalize()} Passing Checks",
            zorder=2,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(v)}" for v in scanlines])
    ax.set_xlabel("Scanline Y [um]")
    ax.set_ylabel("Anchor checks per scanline")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=3, frameon=False, fontsize=8)
    return fig


def _segment_scanline_matrix(results: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame, list[float]]:
    """Return a pass/fail matrix ordered to expose bad segments and bad scanlines."""
    if results.empty:
        return np.empty((0, 0)), pd.DataFrame(), []

    data = results.copy()
    data["segment_id_int"] = pd.to_numeric(data["segment_id"], errors="coerce")
    data["scanline_y_um"] = pd.to_numeric(data["scanline_y_um"], errors="coerce")
    data["pass_value"] = _bool_mask(data, "pass_fail").astype(float)
    data = data.dropna(subset=["segment_id_int", "scanline_y_um"])
    if data.empty:
        return np.empty((0, 0)), pd.DataFrame(), []

    scanlines = sorted(data["scanline_y_um"].unique().tolist())
    matrix_frame = data.pivot_table(
        index="segment_id_int",
        columns="scanline_y_um",
        values="pass_value",
        aggfunc="max",
    ).reindex(columns=scanlines)
    order = (
        pd.DataFrame(
            {
                "segment_id": matrix_frame.index.astype(int),
                "segment_pass_rate": matrix_frame.mean(axis=1).to_numpy(dtype=float),
            }
        )
        .sort_values(["segment_pass_rate", "segment_id"], ascending=[True, True])
        .reset_index(drop=True)
    )
    matrix_frame = matrix_frame.reindex(order["segment_id"].astype(float).tolist())
    return matrix_frame.to_numpy(dtype=float), order, scanlines


def plot_segment_scanline_pass_heatmap(
    outer_results: pd.DataFrame,
    inner_results: pd.DataFrame,
) -> plt.Figure:
    """Plot segment x scanline pass/fail heatmaps for outer and inner gates."""
    fig, axes = make_figure("double_col", nrows=1, ncols=2, height=4.4)
    axes = np.asarray(axes).ravel()
    cmap = ListedColormap([EP04_GATE_REJECT_COLOR, EP04_ANCHOR_COLOR])
    cmap.set_bad("#F4F5F7")
    norm = BoundaryNorm([-0.5, 0.5, 1.5], cmap.N)
    last_image = None

    for ax, contour, results in [
        (axes[0], "outer", outer_results),
        (axes[1], "inner", inner_results),
    ]:
        matrix, order, scanlines = _segment_scanline_matrix(results)
        if matrix.size == 0:
            ax.text(0.5, 0.5, f"No {contour} rows", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue

        masked = np.ma.masked_invalid(matrix)
        last_image = ax.imshow(masked, aspect="auto", interpolation="nearest", cmap=cmap, norm=norm)
        ax.set_xticks(np.arange(len(scanlines)))
        ax.set_xticklabels([f"{int(v)}" for v in scanlines], rotation=45, ha="right")
        n_rows = len(order)
        y_ticks = np.unique(np.linspace(0, n_rows - 1, min(6, n_rows), dtype=int))
        ax.set_yticks(y_ticks)
        ax.set_yticklabels(order.loc[y_ticks, "segment_id"].astype(int).astype(str).tolist())
        ax.set_xlabel("Scanline Y [um]")
        ax.set_ylabel("Segment ID, sorted by pass rate")
        ax.set_title(f"{contour.title()} segment x scanline gate")
        ax.grid(False)

    if last_image is not None:
        cbar = fig.colorbar(last_image, ax=axes.tolist(), ticks=[0, 1], fraction=0.035, pad=0.02)
        cbar.ax.set_yticklabels(["reject", "pass"])
        format_colorbar(cbar, "Row-level gate")
    return fig


def scanline_segment_failure_summary_table(
    outer_results: pd.DataFrame,
    inner_results: pd.DataFrame,
) -> pd.DataFrame:
    """Summarize whether failures concentrate by scanline or by segment."""
    rows: list[dict] = []
    for contour, results in [("outer", outer_results), ("inner", inner_results)]:
        if results.empty:
            continue
        data = results.copy()
        data["pass_bool"] = _bool_mask(data, "pass_fail")
        data["segment_id_int"] = pd.to_numeric(data["segment_id"], errors="coerce")
        data["scanline_y_um"] = pd.to_numeric(data["scanline_y_um"], errors="coerce")
        data = data.dropna(subset=["segment_id_int", "scanline_y_um"])
        if data.empty:
            continue

        by_scanline = (
            data.groupby("scanline_y_um", dropna=False)
            .agg(row_count=("pass_bool", "size"), row_pass_rate=("pass_bool", "mean"))
            .reset_index()
            .sort_values(["row_pass_rate", "scanline_y_um"])
        )
        by_segment = (
            data.groupby("segment_id_int", dropna=False)
            .agg(row_count=("pass_bool", "size"), segment_pass_rate=("pass_bool", "mean"))
            .reset_index()
            .sort_values(["segment_pass_rate", "segment_id_int"])
        )
        weakest_line = by_scanline.iloc[0]
        weakest_segment = by_segment.iloc[0]
        rows.append(
            {
                "contour": contour,
                "evaluated_rows": int(len(data)),
                "overall_row_pass_rate": float(data["pass_bool"].mean()),
                "weakest_scanline_y_um": float(weakest_line["scanline_y_um"]),
                "weakest_scanline_pass_rate": float(weakest_line["row_pass_rate"]),
                "zero_pass_scanlines": int(by_scanline["row_pass_rate"].eq(0.0).sum()),
                "weakest_segment_id": int(weakest_segment["segment_id_int"]),
                "weakest_segment_pass_rate": float(weakest_segment["segment_pass_rate"]),
                "zero_pass_segments": int(by_segment["segment_pass_rate"].eq(0.0).sum()),
            }
        )
    return pd.DataFrame(rows)


def _split_fail_reasons(value: object) -> list[str]:
    reasons = [item.strip() for item in str(value).split(";")]
    return [item for item in reasons if item and item != "pass" and item.lower() != "nan"]


def failure_reason_table(results: pd.DataFrame, *, contour: str) -> pd.DataFrame:
    """Explode row-level fail_reason strings into a count table."""
    if results.empty:
        return pd.DataFrame(columns=["contour", "reason", "triggered_rows", "failed_rows", "share_of_failed_rows"])
    failed = results.loc[~_bool_mask(results, "pass_fail")].copy()
    exploded = failed["fail_reason"].map(_split_fail_reasons).explode().dropna()
    counts = exploded.value_counts()
    return pd.DataFrame(
        {
            "contour": str(contour),
            "reason": counts.index,
            "triggered_rows": counts.to_numpy(dtype=int),
            "failed_rows": int(len(failed)),
            "share_of_failed_rows": counts.to_numpy(dtype=float) / max(int(len(failed)), 1),
        }
    )


def failure_cooccurrence_table(
    outer_results: pd.DataFrame,
    inner_results: pd.DataFrame,
    *,
    top_n: int = 8,
) -> pd.DataFrame:
    """Summarize multi-label failure reasons and their strongest co-occurrences."""
    rows: list[dict] = []
    for contour, results in [("outer", outer_results), ("inner", inner_results)]:
        if results.empty:
            continue
        failed = results.loc[~_bool_mask(results, "pass_fail")].copy()
        reason_lists = [sorted(set(_split_fail_reasons(value))) for value in failed["fail_reason"]]
        reason_counts: Counter[str] = Counter()
        pair_counts: Counter[tuple[str, str]] = Counter()
        for reasons in reason_lists:
            reason_counts.update(reasons)
            pair_counts.update(combinations(reasons, 2))

        failed_rows = int(len(failed))
        for reason, count in reason_counts.most_common(int(top_n)):
            co_counts: Counter[str] = Counter()
            for pair, pair_count in pair_counts.items():
                if reason == pair[0]:
                    co_counts[pair[1]] += pair_count
                elif reason == pair[1]:
                    co_counts[pair[0]] += pair_count
            top_co_reason = ""
            top_co_count = 0
            if co_counts:
                top_co_reason, top_co_count = co_counts.most_common(1)[0]
            rows.append(
                {
                    "contour": contour,
                    "reason": reason,
                    "triggered_rows": int(count),
                    "failed_rows": failed_rows,
                    "share_of_failed_rows": float(count / max(failed_rows, 1)),
                    "top_co_reason": top_co_reason,
                    "top_co_triggered_rows": int(top_co_count),
                    "top_co_share_of_reason": float(top_co_count / max(int(count), 1)),
                }
            )
    return pd.DataFrame(rows)


def ncc_esf_failure_diagnostic_table(
    outer_results: pd.DataFrame,
    inner_results: pd.DataFrame,
) -> pd.DataFrame:
    """Separate NCC quality from ESF/model/stability failure modes."""
    rows: list[dict] = []
    esf_reasons = {
        "sigma_out_of_range",
        "split_half_high",
        "low_phase_coverage",
        "psf_sensitivity_high",
    }
    for contour, results in [("outer", outer_results), ("inner", inner_results)]:
        if results.empty:
            continue
        data = results.copy()
        data["pass_bool"] = _bool_mask(data, "pass_fail")
        failed = data.loc[~data["pass_bool"]].copy()
        reason_lists = [set(_split_fail_reasons(value)) for value in failed["fail_reason"]]

        def share_with(predicate) -> float:
            if not reason_lists:
                return np.nan
            return float(np.mean([predicate(reasons) for reasons in reason_lists]))

        ncc_peak = pd.to_numeric(failed.get("median_ncc_peak", pd.Series(index=failed.index)), errors="coerce")
        ncc_fit_ok = pd.to_numeric(failed.get("ncc_fit_ok_fraction", pd.Series(index=failed.index)), errors="coerce")
        phase = pd.to_numeric(failed.get("phase_coverage_px", pd.Series(index=failed.index)), errors="coerce")
        sigma = pd.to_numeric(failed.get("fitted_sigma_px", pd.Series(index=failed.index)), errors="coerce")
        split = pd.to_numeric(failed.get("split_half_diff_px", pd.Series(index=failed.index)), errors="coerce")
        rows.append(
            {
                "contour": contour,
                "failed_rows": int(len(failed)),
                "median_failed_ncc_peak": float(ncc_peak.median()) if not ncc_peak.dropna().empty else np.nan,
                "p10_failed_ncc_peak": float(ncc_peak.quantile(0.10)) if not ncc_peak.dropna().empty else np.nan,
                "share_failed_ncc_peak_above_gate": float(ncc_peak.ge(QUALITY_GATES["min_ncc_peak"]).mean()) if len(failed) else np.nan,
                "median_failed_ncc_fit_ok_fraction": float(ncc_fit_ok.median()) if not ncc_fit_ok.dropna().empty else np.nan,
                "ncc_unreliable_share": share_with(lambda reasons: "ncc_unreliable" in reasons),
                "fit_error_share": share_with(lambda reasons: any(reason.startswith("fit_error") for reason in reasons)),
                "sigma_out_of_range_share": share_with(lambda reasons: "sigma_out_of_range" in reasons),
                "split_half_high_share": share_with(lambda reasons: "split_half_high" in reasons),
                "low_phase_coverage_share": share_with(lambda reasons: "low_phase_coverage" in reasons),
                "esf_or_stability_share": share_with(
                    lambda reasons: bool(reasons & esf_reasons)
                    or any(reason.startswith("fit_error") for reason in reasons)
                ),
                "median_failed_phase_coverage_px": float(phase.median()) if not phase.dropna().empty else np.nan,
                "median_failed_sigma_px": float(sigma.median()) if not sigma.dropna().empty else np.nan,
                "median_failed_split_half_px": float(split.median()) if not split.dropna().empty else np.nan,
            }
        )
    return pd.DataFrame(rows)


def ep06_role_margin_table(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Audit how far each EP06 role sits from alignment-input numeric thresholds."""
    if recommendations.empty:
        return pd.DataFrame()

    data = recommendations.copy()
    data["pass_rate"] = pd.to_numeric(data.get("pass_rate"), errors="coerce")
    data["split_half_median_px"] = pd.to_numeric(data.get("split_half_median_px"), errors="coerce")
    data["crb_ratio_median"] = pd.to_numeric(data.get("crb_ratio_median"), errors="coerce")
    data["phase_coverage_median_px"] = pd.to_numeric(data.get("phase_coverage_median_px"), errors="coerce")
    data["pass_rate_margin"] = data["pass_rate"] - 0.70
    data["split_margin_px"] = 0.06 - data["split_half_median_px"]
    data["crb_ratio_margin"] = 5.0 - data["crb_ratio_median"]
    data["phase_margin_px"] = data["phase_coverage_median_px"] - QUALITY_GATES["min_phase_range_px"]

    norm_columns = {
        "pass_rate_to_0.70": data["pass_rate_margin"] / 0.70,
        "split_to_0.06px": data["split_margin_px"] / 0.06,
        "crb_to_5x": data["crb_ratio_margin"] / 5.0,
        "phase_to_0.15px": data["phase_margin_px"] / QUALITY_GATES["min_phase_range_px"],
    }
    norm = pd.DataFrame(norm_columns, index=data.index)
    data["alignment_margin_min"] = norm.min(axis=1)
    data["closest_alignment_gate"] = norm.idxmin(axis=1)

    rows: list[dict] = []
    for (contour, role), group in data.groupby(["contour", "ep06_role"], dropna=False, sort=True):
        closest = group["closest_alignment_gate"].mode()
        rows.append(
            {
                "contour": contour,
                "ep06_role": role,
                "n_segments": int(len(group)),
                "median_pass_rate_margin": float(group["pass_rate_margin"].median()),
                "median_split_margin_px": float(group["split_margin_px"].median()),
                "median_crb_ratio_margin": float(group["crb_ratio_margin"].median()),
                "median_phase_margin_px": float(group["phase_margin_px"].median()),
                "p10_alignment_margin_min": float(group["alignment_margin_min"].quantile(0.10)),
                "closest_alignment_gate": str(closest.iloc[0]) if not closest.empty else "",
                "near_threshold_segments": int(group["alignment_margin_min"].abs().le(0.15).sum()),
            }
        )
    return pd.DataFrame(rows)


def _display_gate_label(value: object) -> str:
    """Convert code-style gate names into plot labels."""
    text = str(value).replace("_", " ").replace(":", ": ").strip()
    replacements = {
        "ncc": "NCC",
        "crb": "CRB",
        "snr": "SNR",
        "esf": "ESF",
        "psf": "PSF",
        "valueerror": "ValueError",
    }
    words = [replacements.get(word.lower(), word.lower()) for word in text.split()]
    return " ".join(words)


def plot_inner_failure_reasons(
    inner_results: pd.DataFrame,
    inner_segment_summary: pd.DataFrame,
) -> plt.Figure:
    """Plot inner-contour pass rate by class and row-level failure reasons."""
    fig, axes = make_figure("double_col", nrows=1, ncols=2, height=3.3)
    ax_rate, ax_reason = np.asarray(axes).ravel()
    summary = inner_segment_summary.copy()
    summary["pass_bool"] = _bool_mask(summary, "pass_fail")
    rate = (
        summary.groupby("quality_label", dropna=False)
        .agg(n_segments=("segment_id", "count"), pass_rate=("pass_bool", "mean"))
        .reset_index()
        .sort_values("quality_label")
    )
    x = np.arange(len(rate))
    ax_rate.bar(x, 100.0 * rate["pass_rate"].to_numpy(dtype=float), color=METHOD_COLOR_LIST[0], alpha=0.82)
    for i, row in rate.iterrows():
        ax_rate.text(i, 100.0 * float(row["pass_rate"]) + 1.0, f"n = {int(row['n_segments'])}", ha="center", fontsize=8)
    ax_rate.set_xticks(x)
    ax_rate.set_xticklabels([str(v).split("_")[0] for v in rate["quality_label"]])
    ax_rate.set_ylim(0.0, min(105.0, max(20.0, 100.0 * float(rate["pass_rate"].max()) + 15.0)))
    ax_rate.set_ylabel("Segment anchor pass rate [%]")
    ax_rate.set_title("(a) Segment-level anchor pass rate")
    ax_rate.grid(axis="y", alpha=0.25)
    ax_rate.tick_params(axis="both", labelsize=8)

    reasons = failure_reason_table(inner_results, contour="inner").head(8)
    if reasons.empty:
        ax_reason.text(0.5, 0.5, "No failed rows", transform=ax_reason.transAxes, ha="center", va="center")
    else:
        y = np.arange(len(reasons))
        shares = 100.0 * reasons["share_of_failed_rows"].to_numpy(dtype=float)
        counts = reasons["triggered_rows"].to_numpy(dtype=int)
        ax_reason.barh(y, shares, color=METHOD_COLOR_LIST[3], alpha=0.82)
        ax_reason.set_yticks(y)
        ax_reason.set_yticklabels([_display_gate_label(value) for value in reasons["reason"]])
        ax_reason.invert_yaxis()
        x_max = max(10.0, float(np.nanmax(shares)) * 1.28)
        ax_reason.set_xlim(0.0, x_max)
        for yi, share, count in zip(y, shares, counts, strict=False):
            ax_reason.text(share + 0.015 * x_max, yi, f"{share:.1f}% (n = {count})", va="center", fontsize=8)
        ax_reason.set_xlabel("Share of failed row evaluations [%]")
        ax_reason.set_title("(b) Row-level gate rejection reasons")
        ax_reason.grid(axis="x", alpha=0.25)
        ax_reason.tick_params(axis="both", labelsize=8)
    return fig


def localization_vs_shape_table() -> pd.DataFrame:
    """Clarify EP04 localization metrics versus downstream shape reconstruction metrics."""
    return pd.DataFrame(
        [
            {
                "Aspect": "EP04 localization precision",
                "Measured here": "split-half edge position, CRB ratio, NCC phase coverage",
                "Use in EP06": "alignment anchors, frame/segment gates, held-out QC",
                "Not a claim": "not dense contour SR or metrology-grade 5 um temperature recovery",
            },
            {
                "Aspect": "EP06 shape reconstruction",
                "Measured here": "only anchor availability and failure modes",
                "Use in EP06": "target internal contours, evaluate LR/bicubic/SR shape stability",
                "Not a claim": "not judged failed by anchor rejection alone",
            },
        ]
    )


def build_ep06_gate_recommendations(
    outer_segment_summary: pd.DataFrame,
    inner_segment_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Assign EP06 use categories from EP04 segment-level quality evidence."""
    combined = _contour_summary_frame(outer_segment_summary, inner_segment_summary)
    if combined.empty:
        return pd.DataFrame()
    combined["pass_bool"] = _bool_mask(combined, "pass_fail")
    split = pd.to_numeric(combined["split_half_median_px"], errors="coerce")
    crb = pd.to_numeric(combined["crb_ratio_median"], errors="coerce")
    phase = pd.to_numeric(combined["phase_coverage_median_px"], errors="coerce")
    pass_rate = pd.to_numeric(combined["pass_rate"], errors="coerce")
    robust = (
        combined["pass_bool"]
        & pass_rate.ge(0.70)
        & split.le(0.06)
        & crb.le(5.0)
        & phase.ge(QUALITY_GATES["min_phase_range_px"])
    )
    moderate = combined["pass_bool"] & pass_rate.ge(0.50) & split.le(0.10) & crb.le(8.0)
    holdout_mask = robust & combined["segment_id"].astype(int).mod(4).eq(0)
    alignment_mask = robust & ~holdout_mask
    validation_mask = holdout_mask | (moderate & ~robust)

    recommendation = np.full(len(combined), "sr_target_not_truth", dtype=object)
    recommendation[validation_mask.to_numpy()] = "holdout_validation"
    recommendation[alignment_mask.to_numpy()] = "alignment_input"
    combined["ep06_role"] = recommendation

    reasons = []
    for _, row in combined.iterrows():
        if row["ep06_role"] == "alignment_input":
            reasons.append("stable split-half/CRB with scanline support; use to initialize or constrain alignment")
        elif row["ep06_role"] == "holdout_validation":
            reasons.append("quality-gated but withheld or marginal; use to validate alignment without fitting to it")
        elif row["contour"] == "inner":
            reasons.append("internal structure remains an SR target, but current localization gate should not be treated as truth")
        else:
            reasons.append("excluded from direct truth use; keep only as context or qualitative target")
    combined["ep06_reason"] = reasons
    columns = [
        "contour",
        "segment_id",
        "quality_label",
        "x_px",
        "y_px",
        "pass_rate",
        "split_half_median_px",
        "crb_ratio_median",
        "phase_coverage_median_px",
        "snr",
        "fail_reason_primary",
        "ep06_role",
        "ep06_reason",
    ]
    return combined[[col for col in columns if col in combined.columns]].sort_values(
        ["ep06_role", "contour", "segment_id"]
    ).reset_index(drop=True)


def ep06_gate_recommendation_summary(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Aggregate EP06 recommendation counts by contour and role."""
    if recommendations.empty:
        return pd.DataFrame()
    table = (
        recommendations.groupby(["contour", "ep06_role"], dropna=False)
        .agg(
            n_segments=("segment_id", "count"),
            median_split_half_px=("split_half_median_px", "median"),
            median_crb_ratio=("crb_ratio_median", "median"),
            median_pass_rate=("pass_rate", "median"),
        )
        .reset_index()
        .sort_values(["contour", "ep06_role"])
    )
    return table


def plot_ep06_gate_recommendations(recommendations: pd.DataFrame) -> plt.Figure:
    """Plot EP06 role assignment counts for inner and outer contour segments."""
    fig, ax = make_figure("single_col", height=3.1)
    if recommendations.empty:
        ax.text(0.5, 0.5, "No recommendations", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return fig
    order = ["alignment_input", "holdout_validation", "sr_target_not_truth"]
    colors = {
        "alignment_input": METHOD_COLORS["secondary"],
        "holdout_validation": METHOD_COLORS["primary"],
        "sr_target_not_truth": METHOD_COLORS["accent_1"],
    }
    labels = {
        "alignment_input": "Alignment input",
        "holdout_validation": "Holdout validation",
        "sr_target_not_truth": "SR target, not truth",
    }
    contours = ["outer", "inner"]
    bottom = np.zeros(len(contours), dtype=float)
    x = np.arange(len(contours))
    counts = recommendations.groupby(["contour", "ep06_role"]).size()
    for role in order:
        values = np.asarray([int(counts.get((contour, role), 0)) for contour in contours], dtype=float)
        ax.bar(x, values, bottom=bottom, color=colors[role], alpha=0.88, label=labels[role])
        for xi, value, base in zip(x, values, bottom, strict=False):
            if value <= 0:
                continue
            ax.text(
                xi,
                base + value / 2.0,
                f"{int(value)}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if role == "sr_target_not_truth" else "black",
            )
        bottom += values
    ax.set_xticks(x)
    ax.set_xticklabels([value.title() for value in contours])
    ax.set_ylabel("Segment count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.99),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.92,
        fontsize=7,
        handlelength=1.0,
        handletextpad=0.4,
        borderpad=0.25,
        labelspacing=0.25,
    )
    return fig


def save_ep06_gate_outputs(recommendations: pd.DataFrame, output_dir: Path) -> dict[str, Path]:
    """Save EP06 gate recommendation CSV outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    recommendations_path = output_dir / "ep06_gate_recommendations.csv"
    summary_path = output_dir / "ep06_gate_recommendation_summary.csv"
    recommendations.to_csv(recommendations_path, index=False)
    ep06_gate_recommendation_summary(recommendations).to_csv(summary_path, index=False)
    return {"recommendations": recommendations_path, "summary": summary_path}


def create_ep04_anchor_gate_figures(
    reference_frame: np.ndarray,
    outer_results: pd.DataFrame,
    outer_segment_summary: pd.DataFrame,
    inner_results: pd.DataFrame,
    inner_segment_summary: pd.DataFrame,
    recommendations: pd.DataFrame,
    output_dir: Path,
) -> dict[str, Path]:
    """Create and save the EP04 anchor/gate figures used by the notebook/report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "global_segment_quality_distribution.png": plot_global_segment_quality_distribution(
            outer_segment_summary,
            inner_segment_summary,
        ),
        "anchor_coverage_map.png": plot_anchor_coverage_map(
            reference_frame,
            outer_segment_summary,
            inner_segment_summary,
            outer_results,
            inner_results,
        ),
        "anchor_scanline_support.png": plot_anchor_scanline_support(
            outer_results,
            inner_results,
            outer_segment_summary,
            inner_segment_summary,
        ),
        "inner_failure_reasons.png": plot_inner_failure_reasons(
            inner_results,
            inner_segment_summary,
        ),
        "ep06_gate_recommendations.png": plot_ep06_gate_recommendations(recommendations),
        "segment_scanline_pass_heatmap.png": plot_segment_scanline_pass_heatmap(
            outer_results,
            inner_results,
        ),
        "normal_angle_coverage.png": plot_normal_angle_coverage_comparison(
            outer_segment_summary,
            inner_segment_summary,
        ),
    }
    saved = {}
    for name, fig in figures.items():
        saved[name] = savefig_academic(fig, output_dir / name)
    return saved


def _bool_mask(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(False, index=frame.index)
    values = frame[column]
    if values.dtype == object:
        return values.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})
    return values.astype(bool)


def _passed(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[_bool_mask(frame, "pass_fail")].copy()


def _finite_column(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.asarray([], dtype=float)
    values = pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)
    return values[np.isfinite(values)]


def _marker_sizes_from_precision(frame: pd.DataFrame) -> np.ndarray:
    precision = pd.to_numeric(frame.get("split_half_median_px", pd.Series(index=frame.index)), errors="coerce").to_numpy(dtype=float)
    finite = precision[np.isfinite(precision)]
    if finite.size == 0:
        return np.full(len(frame), 28.0)
    denom = max(float(np.nanpercentile(finite, 90)), 1e-6)
    scaled = np.nan_to_num(precision, nan=float(np.nanmedian(finite)))
    return 18.0 + 48.0 * (1.0 - np.clip(scaled / denom, 0.0, 1.0))


def plot_combined_pass_fail_contour_map(
    reference_frame: np.ndarray,
    outer_segment_summary: pd.DataFrame,
    inner_segment_summary: pd.DataFrame,
) -> plt.Figure:
    """Overlay outer and inner segment pass/fail status on the reference frame."""
    fig, ax = make_figure("double_col", height=4.6)
    im = ax.imshow(reference_frame, cmap="inferno", origin="upper")
    specs = [
        ("Outer", outer_segment_summary.sort_values("segment_id").copy(), "o"),
        ("Inner", inner_segment_summary.sort_values("segment_id").copy(), "s"),
    ]
    for scope, data, marker in specs:
        if data.empty:
            continue
        passed = _bool_mask(data, "pass_fail")
        sizes = _marker_sizes_from_precision(data)
        for state, mask, color, alpha in [
            ("fail", ~passed, METHOD_COLOR_LIST[2], 0.72),
            ("pass", passed, METHOD_COLOR_LIST[1], 0.88),
        ]:
            if int(mask.sum()) == 0:
                continue
            ax.scatter(
                data.loc[mask, "x_px"],
                data.loc[mask, "y_px"],
                s=sizes[mask.to_numpy()],
                marker=marker,
                c=color,
                edgecolor="white",
                linewidth=0.25,
                alpha=alpha,
                label=f"{scope} {state} (N={int(mask.sum())})",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    format_colorbar(cbar, "Temperature [C]")
    ax.set_xlabel("Column [px]")
    ax.set_ylabel("Row [px]")
    ax.legend(loc="upper right", fontsize=6, ncol=2)
    return fig


def plot_combined_split_half_distribution(
    outer_segment_summary: pd.DataFrame,
    inner_segment_summary: pd.DataFrame,
) -> plt.Figure:
    """Plot split-half repeatability for all passed outer and inner segments."""
    outer_values = _finite_column(_passed(outer_segment_summary), "split_half_median_px")
    inner_values = _finite_column(_passed(inner_segment_summary), "split_half_median_px")
    combined = np.concatenate([outer_values, inner_values]) if outer_values.size or inner_values.size else np.asarray([])
    fig, ax = make_figure("single_col", height=3.0)
    if combined.size == 0:
        ax.text(0.5, 0.5, "No passed segments", transform=ax.transAxes, ha="center", va="center")
        ax.set_axis_off()
        return fig
    bins = np.linspace(0.0, max(0.12, float(np.nanpercentile(combined, 95)) * 1.15), 24)
    if outer_values.size:
        ax.hist(outer_values, bins=bins, color=METHOD_COLOR_LIST[0], alpha=0.72, edgecolor="white", label=f"Outer (N={outer_values.size})")
    if inner_values.size:
        _, _, patches = ax.hist(inner_values, bins=bins, color=METHOD_COLOR_LIST[3], alpha=0.48, edgecolor="white", label=f"Inner (N={inner_values.size})")
        for patch in patches:
            patch.set_hatch("///")
    median = float(np.nanmedian(combined))
    p90 = float(np.nanpercentile(combined, 90))
    ax.axvline(median, color="#222222", linestyle="--", linewidth=0.9, label=f"combined median={median:.3f} px")
    ax.axvline(p90, color=METHOD_COLOR_LIST[4], linestyle=":", linewidth=1.1, label=f"combined P90={p90:.3f} px")
    ax.set_xlabel("Median split-half difference [px]")
    ax.set_ylabel("Passed segment count")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=7)
    return fig


def plot_normal_angle_coverage_comparison(
    outer_segment_summary: pd.DataFrame,
    inner_segment_summary: pd.DataFrame,
) -> plt.Figure:
    """Compare passed-segment normal-angle coverage for outer and inner contours."""
    fig, ax = make_figure("single_col", height=3.2, subplot_kw={"projection": "polar"})
    bins = np.linspace(0.0, np.pi, 19)
    centers = 0.5 * (bins[:-1] + bins[1:])
    width = float(np.diff(bins)[0]) * 0.42
    for offset, label, data, color in [
        (-0.23 * width, "Outer", outer_segment_summary, METHOD_COLOR_LIST[0]),
        (0.23 * width, "Inner", inner_segment_summary, METHOD_COLOR_LIST[3]),
    ]:
        angles = _finite_column(_passed(data), "normal_angle_deg")
        if angles.size == 0:
            continue
        counts, _ = np.histogram(np.deg2rad(np.mod(angles, 180.0)), bins=bins)
        heights = counts.astype(float)
        if heights.max() > 0:
            heights /= heights.max()
        ax.bar(centers + offset, heights, width=width, color=color, alpha=0.64, edgecolor="white", linewidth=0.35, label=f"{label} (N={angles.size})")
    ax.set_thetamin(0)
    ax.set_thetamax(180)
    ax.set_theta_zero_location("E")
    ax.set_theta_direction(-1)
    ax.set_rlabel_position(135)
    ax.set_ylabel("Relative count")
    ax.legend(loc="upper right", bbox_to_anchor=(1.30, 1.08), fontsize=7)
    return fig


def plot_combined_crb_ratio_scatter(
    outer_segment_summary: pd.DataFrame,
    inner_segment_summary: pd.DataFrame,
) -> plt.Figure:
    """Plot CRB ratios for all passed outer and inner segments."""
    fig, axes = make_figure("double_col", nrows=1, ncols=2, height=3.0, sharey=True)
    axes = np.asarray(axes).ravel()
    combined_y = np.concatenate(
        [
            _finite_column(_passed(outer_segment_summary), "crb_ratio_median"),
            _finite_column(_passed(inner_segment_summary), "crb_ratio_median"),
        ]
    )
    positive_y = combined_y[combined_y > 0]
    y_max = max(float(np.nanpercentile(positive_y, 98)) * 1.25, 3.5) if positive_y.size else 3.5
    for ax, label, data, marker, color in [
        (axes[0], "Outer", _passed(outer_segment_summary), "o", METHOD_COLOR_LIST[0]),
        (axes[1], "Inner", _passed(inner_segment_summary), "s", METHOD_COLOR_LIST[3]),
    ]:
        y = pd.to_numeric(data.get("crb_ratio_median", pd.Series(index=data.index)), errors="coerce").to_numpy(dtype=float)
        x = pd.to_numeric(data.get("segment_id", pd.Series(index=data.index)), errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(x) & np.isfinite(y) & (y > 0)
        ax.scatter(x[mask], y[mask], s=26, marker=marker, c=color, edgecolor="white", linewidth=0.25, alpha=0.86, label=f"{label} passed (N={int(mask.sum())})")
        for value in [1.0, 1.5, 3.0]:
            ax.axhline(value, color="#666666", linestyle="--", linewidth=0.75)
        ax.set_xlabel("Segment ID")
        ax.set_title(f"{label} passed segments")
        ax.set_yscale("log")
        ax.set_ylim(0.6, y_max)
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right", fontsize=7)
    axes[0].set_ylabel("Median split-half / CRB")
    return fig


def combined_validation_summary_table(
    outer_segment_summary: pd.DataFrame,
    inner_segment_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Build the inner/outer/combined EP04 validation summary table."""

    def median_col(frame: pd.DataFrame, column: str) -> float:
        values = _finite_column(frame, column)
        return float(np.nanmedian(values)) if values.size else np.nan

    def stats(frame: pd.DataFrame) -> dict:
        passed = _passed(frame)
        a_mask = _bool_mask(frame, "is_a_class")
        a_class = frame[a_mask].copy()
        a_passed = _passed(a_class)
        return {
            "total": int(len(frame)),
            "a_class": int(len(a_class)),
            "passed": int(len(passed)),
            "pass_rate": float(len(passed) / len(frame)) if len(frame) else np.nan,
            "a_passed": int(len(a_passed)),
            "a_pass_rate": float(len(a_passed) / len(a_class)) if len(a_class) else np.nan,
            "split": median_col(passed, "split_half_median_px"),
            "ratio": median_col(passed, "crb_ratio_median"),
            "delta": median_col(frame, "segment_abs_delta_t_c"),
            "snr": median_col(frame, "snr"),
        }

    def fmt(value: float, decimals: int) -> str:
        return f"{value:.{decimals}f}" if np.isfinite(value) else "n/a"

    def pct(value: float) -> str:
        return f"{100.0 * value:.1f}%" if np.isfinite(value) else "n/a"

    outer = stats(outer_segment_summary)
    inner = stats(inner_segment_summary)
    combined = stats(pd.concat([outer_segment_summary, inner_segment_summary], ignore_index=True))
    rows = [
        ("Total segments", f"{outer['total']}", f"{inner['total']}", f"{combined['total']}"),
        ("A-class segments", f"{outer['a_class']}", f"{inner['a_class']}", f"{combined['a_class']}"),
        ("Passed segments (all labels)", f"{outer['passed']}", f"{inner['passed']}", f"{combined['passed']}"),
        ("Pass rate (all labels)", pct(outer["pass_rate"]), pct(inner["pass_rate"]), pct(combined["pass_rate"])),
        ("A-class passed segments", f"{outer['a_passed']}", f"{inner['a_passed']}", f"{combined['a_passed']}"),
        ("A-class pass rate", pct(outer["a_pass_rate"]), pct(inner["a_pass_rate"]), pct(combined["a_pass_rate"])),
        ("Passed median split-half [px]", fmt(outer["split"], 3), fmt(inner["split"], 3), fmt(combined["split"], 3)),
        ("Passed median CRB ratio", fmt(outer["ratio"], 2), fmt(inner["ratio"], 2), fmt(combined["ratio"], 2)),
        ("Median abs Delta T [C]", fmt(outer["delta"], 2), fmt(inner["delta"], 2), "-"),
        ("Median SNR", fmt(outer["snr"], 1), fmt(inner["snr"], 1), "-"),
    ]
    return pd.DataFrame(rows, columns=["Metric", "Outer", "Inner", "Combined"])


# ---------------------------------------------------------------------------
# EP04-B: apparent thermal contour reconstruction and baselines
# ---------------------------------------------------------------------------


def _pixel_to_mm(value_px: float | np.ndarray, pixel_size_um: float) -> float | np.ndarray:
    return np.asarray(value_px, dtype=float) * float(pixel_size_um) * 1e-3


def _finite_float(value, default: float = np.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    return out if np.isfinite(out) else float(default)


def classify_boundary_confidence(
    split_half_diff_px: float,
    crb_ratio: float,
    psf_sensitivity_px: float,
    *,
    pass_quality_gate: bool,
) -> str:
    """Classify one apparent thermal boundary point."""
    if not bool(pass_quality_gate):
        return "fail"
    split = _finite_float(split_half_diff_px)
    ratio = _finite_float(crb_ratio)
    psf = _finite_float(psf_sensitivity_px)
    if split < 0.03 and ratio < 2.0 and psf < 0.02:
        return "high"
    if split < 0.06 and ratio < 5.0:
        return "medium"
    return "low"


def _merge_segment_geometry(
    segment_summary: pd.DataFrame,
    segments: pd.DataFrame | None,
) -> pd.DataFrame:
    """Attach original contour order and tangent direction when available."""
    summary = segment_summary.copy()
    if segments is None or segments.empty:
        summary["contour_index"] = summary["segment_id"].astype(int)
        summary["tangent_angle_deg"] = (summary["normal_angle_deg"].astype(float) + 90.0) % 180.0
        summary["tx"] = -summary["ny"].astype(float)
        summary["ty"] = summary["nx"].astype(float)
        return summary

    keep = [
        "segment_id",
        "contour_index",
        "tx",
        "ty",
        "tangent_angle_deg",
        "normal_angle_deg",
        "nx",
        "ny",
    ]
    geom = segments[[col for col in keep if col in segments.columns]].copy()
    rename = {
        "contour_index": "contour_index_geom",
        "tx": "tx_geom",
        "ty": "ty_geom",
        "tangent_angle_deg": "tangent_angle_deg_geom",
        "normal_angle_deg": "normal_angle_deg_geom",
        "nx": "nx_geom",
        "ny": "ny_geom",
    }
    geom = geom.rename(columns={key: value for key, value in rename.items() if key in geom.columns})
    out = summary.merge(geom, on="segment_id", how="left")
    out["contour_index"] = out.get("contour_index_geom", out["segment_id"]).fillna(out["segment_id"]).astype(int)
    out["tangent_angle_deg"] = out.get(
        "tangent_angle_deg_geom",
        (out["normal_angle_deg"].astype(float) + 90.0) % 180.0,
    ).fillna((out["normal_angle_deg"].astype(float) + 90.0) % 180.0)
    out["tx"] = out.get("tx_geom", -out["ny"].astype(float)).fillna(-out["ny"].astype(float))
    out["ty"] = out.get("ty_geom", out["nx"].astype(float)).fillna(out["nx"].astype(float))
    return out


def extract_thermal_boundary_points(
    segment_summary: pd.DataFrame,
    *,
    segments: pd.DataFrame | None = None,
    pixel_size_um: float = 20.0,
    include_failed: bool = False,
) -> pd.DataFrame:
    """Extract one apparent thermal boundary point for every accepted segment."""
    data = _merge_segment_geometry(segment_summary, segments)
    if not include_failed:
        data = data[data["pass_fail"].astype(bool)].copy()

    rows: list[dict] = []
    for _, row in data.sort_values("contour_index").iterrows():
        s_px = _finite_float(row.get("edge_position_median_px"))
        nx = _finite_float(row.get("nx"))
        ny = _finite_float(row.get("ny"))
        center_col = _finite_float(row.get("x_px")) + s_px * nx
        center_row = _finite_float(row.get("y_px")) + s_px * ny
        pass_gate = bool(row.get("pass_fail", False))
        confidence = classify_boundary_confidence(
            row.get("split_half_median_px", np.nan),
            row.get("crb_ratio_median", np.nan),
            row.get("psf_sensitivity_median_px", np.nan),
            pass_quality_gate=pass_gate,
        )
        point = ThermalBoundaryPoint(
            segment_id=int(row["segment_id"]),
            center_col=float(center_col),
            center_row=float(center_row),
            center_x_mm=float(_pixel_to_mm(center_col, pixel_size_um)),
            center_y_mm=float(_pixel_to_mm(center_row, pixel_size_um)),
            normal_angle_deg=float(row["normal_angle_deg"]),
            tangent_angle_deg=float(row["tangent_angle_deg"]),
            split_half_diff_px=float(row.get("split_half_median_px", np.nan)),
            crb_px=float(row.get("crb_median_px", np.nan)),
            crb_ratio=float(row.get("crb_ratio_median", np.nan)),
            confidence=confidence,
            pass_quality_gate=pass_gate,
            fitted_sigma_px=float(row.get("fitted_sigma_median_px", np.nan)),
            fitted_delta_t_c=float(row.get("fitted_delta_t_median_c", np.nan)),
        )
        record = asdict(point)
        record.update(
            {
                "contour_index": int(row["contour_index"]),
                "segment_center_col": float(row.get("x_px", np.nan)),
                "segment_center_row": float(row.get("y_px", np.nan)),
                "joint_s_median_px": float(s_px),
                "psf_sensitivity_px": float(row.get("psf_sensitivity_median_px", np.nan)),
                "n_scanlines": int(row.get("n_scanlines", 0)),
                "n_pass": int(row.get("n_pass", 0)),
                "pass_rate": float(row.get("pass_rate", np.nan)),
                "quality_label": str(row.get("quality_label", "")),
                "normal_projection": float(row.get("normal_projection", np.nan)),
            }
        )
        rows.append(record)

    columns = [
        "segment_id",
        "center_col",
        "center_row",
        "center_x_mm",
        "center_y_mm",
        "normal_angle_deg",
        "tangent_angle_deg",
        "split_half_diff_px",
        "crb_px",
        "crb_ratio",
        "confidence",
        "pass_quality_gate",
        "fitted_sigma_px",
        "fitted_delta_t_c",
        "contour_index",
        "segment_center_col",
        "segment_center_row",
        "joint_s_median_px",
        "psf_sensitivity_px",
        "n_scanlines",
        "n_pass",
        "pass_rate",
        "quality_label",
        "normal_projection",
    ]
    return pd.DataFrame(rows, columns=columns)


def _confidence_rank(confidence: str) -> int:
    ranks = {"fail": 0, "low": 1, "medium": 2, "high": 3}
    return ranks.get(str(confidence), 0)


def _valid_contour_runs(
    boundary_points: pd.DataFrame,
    *,
    min_confidence: str = "medium",
) -> list[pd.DataFrame]:
    if boundary_points.empty:
        return []
    min_rank = _confidence_rank(min_confidence)
    order_col = "contour_index" if "contour_index" in boundary_points.columns else "segment_id"
    valid = boundary_points[
        boundary_points["pass_quality_gate"].astype(bool)
        & (boundary_points["confidence"].map(_confidence_rank) >= min_rank)
    ].sort_values(order_col)
    order_values = np.sort(valid[order_col].dropna().astype(int).unique())
    order_diffs = np.diff(order_values)
    order_diffs = order_diffs[order_diffs > 0]
    native_step = max(int(np.gcd.reduce(order_diffs)), 1) if order_diffs.size else 1
    runs: list[pd.DataFrame] = []
    current: list[pd.Series] = []
    last_contour_index: int | None = None
    for _, row in valid.iterrows():
        contour_index = int(row[order_col])
        if last_contour_index is None or contour_index == last_contour_index + native_step:
            current.append(row)
        else:
            if len(current) >= 2:
                runs.append(pd.DataFrame(current))
            current = [row]
        last_contour_index = contour_index
    if len(current) >= 2:
        runs.append(pd.DataFrame(current))
    return runs


def stitch_thermal_contour(
    boundary_points: pd.DataFrame,
    *,
    pixel_size_um: float = 20.0,
    min_confidence: str = "medium",
    samples_per_span: int = 16,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spline-stitch high and medium apparent thermal boundary points without closing gaps."""
    rows: list[dict] = []
    for run_id, run in enumerate(_valid_contour_runs(boundary_points, min_confidence=min_confidence)):
        run = run.sort_values("contour_index")
        t = run["contour_index"].to_numpy(dtype=float)
        col = run["center_col"].to_numpy(dtype=float)
        row = run["center_row"].to_numpy(dtype=float)
        if len(run) < 2 or np.ptp(t) <= 0:
            continue

        n_samples = max(2, int((len(run) - 1) * int(samples_per_span)) + 1)
        t_new = np.linspace(float(t.min()), float(t.max()), n_samples)
        if len(run) >= 3:
            degree = min(3, len(run) - 1)
            col_new = make_interp_spline(t, col, k=degree)(t_new)
            row_new = make_interp_spline(t, row, k=degree)(t_new)
        else:
            col_new = np.interp(t_new, t, col)
            row_new = np.interp(t_new, t, row)

        step = np.hypot(np.diff(col_new, prepend=col_new[0]), np.diff(row_new, prepend=row_new[0]))
        arc = np.cumsum(step)
        for sample_index, (arc_t, c, r, length_px) in enumerate(zip(t_new, col_new, row_new, arc)):
            rows.append(
                {
                    "contour_run_id": int(run_id),
                    "sample_index": int(sample_index),
                    "source_contour_index": float(arc_t),
                    "start_segment_id": int(run["segment_id"].iloc[0]),
                    "end_segment_id": int(run["segment_id"].iloc[-1]),
                    "col_px": float(c),
                    "row_px": float(r),
                    "arc_length_px": float(length_px),
                    "n_source_points": int(len(run)),
                    "min_confidence": str(min_confidence),
                }
            )

    px_df = pd.DataFrame(rows)
    if px_df.empty:
        mm_df = pd.DataFrame(
            columns=[
                "contour_run_id",
                "sample_index",
                "source_contour_index",
                "start_segment_id",
                "end_segment_id",
                "x_mm",
                "y_mm",
                "arc_length_mm",
                "n_source_points",
                "min_confidence",
            ]
        )
        return px_df, mm_df

    mm_df = px_df.rename(columns={"col_px": "x_mm", "row_px": "y_mm", "arc_length_px": "arc_length_mm"}).copy()
    mm_df["x_mm"] = _pixel_to_mm(mm_df["x_mm"].to_numpy(dtype=float), pixel_size_um)
    mm_df["y_mm"] = _pixel_to_mm(mm_df["y_mm"].to_numpy(dtype=float), pixel_size_um)
    mm_df["arc_length_mm"] = _pixel_to_mm(mm_df["arc_length_mm"].to_numpy(dtype=float), pixel_size_um)
    return px_df, mm_df


def _robust_uint8(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image, dtype=float)
    lo, hi = np.nanpercentile(image, [1.0, 99.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo, hi = float(np.nanmin(image)), float(np.nanmax(image))
    if hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    scaled = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
    return np.round(255.0 * scaled).astype(np.uint8)


def _canny_context(frame: np.ndarray, sigma: float = 1.0) -> dict:
    smooth = gaussian_filter(np.asarray(frame, dtype=float), sigma=float(sigma))
    image_u8 = _robust_uint8(smooth)
    median = float(np.median(image_u8))
    low = int(np.clip(0.50 * median, 8, 80))
    high = int(np.clip(1.35 * median, 24, 180))
    if high <= low:
        high = min(255, low + 30)
    edges = cv2.Canny(image_u8, low, high, L2gradient=True)
    if int(np.count_nonzero(edges)) < 32:
        edges = cv2.Canny(image_u8, 12, 36, L2gradient=True)
    gx = cv2.Sobel(smooth, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(smooth, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(gx, gy)
    return {"edges": edges > 0, "gx": gx, "gy": gy, "magnitude": magnitude, "smooth": smooth}


def baseline_single_frame_canny(frame: np.ndarray, sigma: float = 1.0) -> pd.DataFrame:
    """Return global Canny edge points with gradient-direction subpixel offsets."""
    context = _canny_context(frame, sigma=sigma)
    rows, cols = np.nonzero(context["edges"])
    if rows.size == 0:
        return pd.DataFrame(columns=["row", "col", "row_subpx", "col_subpx", "gradient_magnitude"])

    gx = context["gx"][rows, cols]
    gy = context["gy"][rows, cols]
    norm = np.hypot(gx, gy)
    valid = norm > 1e-12
    gxn = np.zeros_like(norm, dtype=float)
    gyn = np.zeros_like(norm, dtype=float)
    gxn[valid] = gx[valid] / norm[valid]
    gyn[valid] = gy[valid] / norm[valid]

    mag = context["magnitude"]
    plus = map_coordinates(mag, np.vstack([rows + gyn, cols + gxn]), order=1, mode="nearest")
    minus = map_coordinates(mag, np.vstack([rows - gyn, cols - gxn]), order=1, mode="nearest")
    center = mag[rows, cols]
    denom = minus - 2.0 * center + plus
    offset = np.zeros_like(center, dtype=float)
    ok = np.abs(denom) > 1e-12
    offset[ok] = np.clip(0.5 * (minus[ok] - plus[ok]) / denom[ok], -1.0, 1.0)

    return pd.DataFrame(
        {
            "row": rows.astype(float),
            "col": cols.astype(float),
            "row_subpx": rows.astype(float) + offset * gyn,
            "col_subpx": cols.astype(float) + offset * gxn,
            "gradient_magnitude": center,
        }
    )


def baseline_multiframe_median_canny(frames_list: list[np.ndarray], sigma: float = 1.0) -> pd.DataFrame:
    """Return Canny edge points from a 16-frame median-fused temperature frame."""
    median_frame = np.median(np.asarray(frames_list, dtype=np.float32), axis=0)
    return baseline_single_frame_canny(median_frame, sigma=sigma)


def _segment_vector(segment: pd.Series | dict, key: str, fallback: float = np.nan) -> float:
    return _finite_float(_segment_value(segment, key, fallback), fallback)


def _segment_normal_and_tangent(segment: pd.Series | dict) -> tuple[np.ndarray, np.ndarray]:
    normal = np.array([_segment_vector(segment, "nx"), _segment_vector(segment, "ny")], dtype=float)
    normal_norm = float(np.linalg.norm(normal))
    if not np.isfinite(normal_norm) or normal_norm == 0.0:
        normal = np.array([1.0, 0.0], dtype=float)
    else:
        normal /= normal_norm
    if "tx" in getattr(segment, "index", segment if isinstance(segment, dict) else {}):
        tangent = np.array([_segment_vector(segment, "tx"), _segment_vector(segment, "ty")], dtype=float)
        tangent_norm = float(np.linalg.norm(tangent))
        tangent = np.array([-normal[1], normal[0]]) if tangent_norm == 0.0 else tangent / tangent_norm
    else:
        tangent = np.array([-normal[1], normal[0]], dtype=float)
    return normal, tangent


def _estimate_canny_s_from_context(
    context: dict,
    segment: pd.Series | dict,
    *,
    normal_half_width_px: float = 14.0,
    tangent_half_width_px: float = 9.0,
) -> dict:
    rows, cols = np.nonzero(context["edges"])
    if rows.size == 0:
        return {"s_px": np.nan, "center_col": np.nan, "center_row": np.nan, "success": False, "fail_reason": "no_canny_edges"}

    x0 = _segment_vector(segment, "x_px")
    y0 = _segment_vector(segment, "y_px")
    normal, tangent = _segment_normal_and_tangent(segment)
    rel_col = cols.astype(float) - x0
    rel_row = rows.astype(float) - y0
    n_coord = rel_col * normal[0] + rel_row * normal[1]
    t_coord = rel_col * tangent[0] + rel_row * tangent[1]
    local = (np.abs(n_coord) <= normal_half_width_px) & (np.abs(t_coord) <= tangent_half_width_px)
    if not np.any(local):
        local = (np.abs(n_coord) <= 1.4 * normal_half_width_px) & (np.abs(t_coord) <= 1.6 * tangent_half_width_px)
    if not np.any(local):
        return {"s_px": np.nan, "center_col": np.nan, "center_row": np.nan, "success": False, "fail_reason": "no_local_edge"}

    candidate_rows = rows[local]
    candidate_cols = cols[local]
    candidate_n = n_coord[local]
    candidate_t = t_coord[local]
    candidate_mag = context["magnitude"][candidate_rows, candidate_cols]
    mag_scale = max(float(np.nanpercentile(candidate_mag, 95)), 1e-12)
    cost = (
        np.abs(candidate_t) / float(tangent_half_width_px)
        + 0.35 * np.abs(candidate_n) / float(normal_half_width_px)
        - 0.25 * np.clip(candidate_mag / mag_scale, 0.0, 2.0)
    )
    choice = int(np.nanargmin(cost))
    row_i = int(candidate_rows[choice])
    col_i = int(candidate_cols[choice])

    gx = float(context["gx"][row_i, col_i])
    gy = float(context["gy"][row_i, col_i])
    grad_norm = float(np.hypot(gx, gy))
    col_sub = float(col_i)
    row_sub = float(row_i)
    if grad_norm > 1e-12:
        gxn = gx / grad_norm
        gyn = gy / grad_norm
        mag = context["magnitude"]
        plus = float(map_coordinates(mag, [[row_i + gyn], [col_i + gxn]], order=1, mode="nearest")[0])
        minus = float(map_coordinates(mag, [[row_i - gyn], [col_i - gxn]], order=1, mode="nearest")[0])
        center = float(mag[row_i, col_i])
        denom = minus - 2.0 * center + plus
        if abs(denom) > 1e-12:
            offset = float(np.clip(0.5 * (minus - plus) / denom, -1.0, 1.0))
            col_sub = float(col_i + offset * gxn)
            row_sub = float(row_i + offset * gyn)

    s_px = (col_sub - x0) * normal[0] + (row_sub - y0) * normal[1]
    return {
        "s_px": float(s_px),
        "center_col": float(col_sub),
        "center_row": float(row_sub),
        "success": True,
        "fail_reason": "ok",
    }


def baseline_gaussian_gradient_fit(
    frame: np.ndarray,
    segment_row: pd.Series | dict,
    *,
    half_width_px: float = 10.0,
    step_px: float = 0.5,
) -> dict:
    """Fit a Gaussian gradient profile along one segment normal."""
    profile = extract_normal_profile(
        frame,
        pd.Series(segment_row),
        half_width_px=float(half_width_px),
        step_px=float(step_px),
    )
    u = profile["u_px"].to_numpy(dtype=float)
    temp = profile["temperature_c"].to_numpy(dtype=float)
    grad = np.gradient(temp, u)
    peak_index = int(np.nanargmax(np.abs(grad)))
    amp0 = float(grad[peak_index])
    mu0 = float(u[peak_index])
    sigma0 = 1.0
    offset0 = float(np.nanmedian(grad))

    lower = np.array([float(np.nanmin(grad) - abs(amp0) - 1.0), -half_width_px - 2.0, 0.25, -abs(amp0) - 1.0])
    upper = np.array([float(np.nanmax(grad) + abs(amp0) + 1.0), half_width_px + 2.0, 3.0, abs(amp0) + 1.0])
    x0 = np.array([offset0, mu0, sigma0, amp0], dtype=float)

    def residual(params: np.ndarray) -> np.ndarray:
        offset, mu, sigma, amp = params
        sigma = max(float(sigma), 1e-6)
        return offset + amp * np.exp(-0.5 * ((u - mu) / sigma) ** 2) - grad

    try:
        fit = least_squares(residual, x0, bounds=(lower, upper), max_nfev=4000)
        offset, mu, sigma, amp = fit.x
        rms = float(np.sqrt(np.mean(residual(fit.x) ** 2)))
        normal, _ = _segment_normal_and_tangent(segment_row)
        x0_px = _segment_vector(segment_row, "x_px")
        y0_px = _segment_vector(segment_row, "y_px")
        return {
            "s_px": float(mu),
            "center_col": float(x0_px + mu * normal[0]),
            "center_row": float(y0_px + mu * normal[1]),
            "sigma_px": float(sigma),
            "amplitude": float(amp),
            "rms": rms,
            "success": bool(fit.success),
            "fail_reason": "ok" if fit.success else "least_squares_failed",
        }
    except Exception as exc:  # noqa: BLE001 - baseline diagnostics should be row-complete.
        return {
            "s_px": np.nan,
            "center_col": np.nan,
            "center_row": np.nan,
            "sigma_px": np.nan,
            "amplitude": np.nan,
            "rms": np.nan,
            "success": False,
            "fail_reason": f"fit_error:{type(exc).__name__}",
            "error_detail": _short_exception_detail(exc),
        }


def _method_row(
    *,
    segment: pd.Series,
    scanline_y_um: float,
    method: str,
    n_frames: int,
    estimate: dict,
    ours_s_px: float,
    split_half_diff_px: float = np.nan,
    crb_px: float = np.nan,
    pixel_size_um: float = 20.0,
) -> dict:
    s_px = _finite_float(estimate.get("s_px", np.nan))
    normal, _ = _segment_normal_and_tangent(segment)
    center_col = estimate.get("center_col", np.nan)
    center_row = estimate.get("center_row", np.nan)
    if not np.isfinite(_finite_float(center_col)) or not np.isfinite(_finite_float(center_row)):
        center_col = _segment_vector(segment, "x_px") + s_px * normal[0]
        center_row = _segment_vector(segment, "y_px") + s_px * normal[1]
    diff = s_px - float(ours_s_px) if np.isfinite(s_px) and np.isfinite(ours_s_px) else np.nan
    return {
        "segment_id": int(segment["segment_id"]),
        "scanline_y_um": float(scanline_y_um),
        "method": str(method),
        "n_frames": int(n_frames),
        "s_px": float(s_px),
        "center_col": float(center_col),
        "center_row": float(center_row),
        "center_x_mm": float(_pixel_to_mm(center_col, pixel_size_um)) if np.isfinite(_finite_float(center_col)) else np.nan,
        "center_y_mm": float(_pixel_to_mm(center_row, pixel_size_um)) if np.isfinite(_finite_float(center_row)) else np.nan,
        "split_half_diff_px": float(split_half_diff_px),
        "crb_px": float(crb_px),
        "diff_vs_ours_px": float(diff),
        "abs_diff_vs_ours_px": abs(float(diff)) if np.isfinite(diff) else np.nan,
        "cross_scanline_std_px": np.nan,
        "success": bool(estimate.get("success", np.isfinite(s_px))),
        "fail_reason": str(estimate.get("fail_reason", "ok")),
        "error_detail": str(estimate.get("error_detail", "")),
    }


def run_baseline_comparison(
    results: pd.DataFrame,
    segment_summary: pd.DataFrame,
    frame_audit_df: pd.DataFrame,
    data_dir: Path,
    *,
    segments: pd.DataFrame | None = None,
    pixel_size_um: float = 20.0,
    canny_sigma: float = 1.0,
    esf_half_width_px: float = 10.0,
    esf_step_px: float = 0.5,
    contour: str | None = None,
) -> pd.DataFrame:
    """Compare joint ESF apparent thermal boundary positions against three baselines."""
    passed_ids = set(segment_summary.loc[segment_summary["pass_fail"].astype(bool), "segment_id"].astype(int))
    segment_rows = _merge_segment_geometry(segment_summary, segments)
    segment_rows = segment_rows[segment_rows["segment_id"].astype(int).isin(passed_ids)].copy()
    scanlines = build_x_scanlines(frame_audit_df)
    if not scanlines or segment_rows.empty:
        return pd.DataFrame()

    needed_files = [str(name) for line in scanlines for name in line["file"].tolist()]
    frames = preload_frames(Path(data_dir), needed_files)
    result_map = {
        (int(row["segment_id"]), float(row["scanline_y_um"])): row
        for _, row in results.iterrows()
        if int(row["segment_id"]) in passed_ids
    }

    rows: list[dict] = []
    for line in scanlines:
        scanline_y = float(line["Y"].iloc[0])
        files = [str(name) for name in line["file"].tolist()]
        frame_list = [frames[name] for name in files]
        stack = np.stack(frame_list).astype(np.float32, copy=False)
        middle_frame = frame_list[len(frame_list) // 2]
        median_frame = np.median(stack, axis=0)
        even_frame = np.median(stack[::2], axis=0)
        odd_frame = np.median(stack[1::2], axis=0)

        single_context = _canny_context(middle_frame, sigma=canny_sigma)
        median_context = _canny_context(median_frame, sigma=canny_sigma)
        even_context = _canny_context(even_frame, sigma=canny_sigma)
        odd_context = _canny_context(odd_frame, sigma=canny_sigma)

        for _, segment in segment_rows.iterrows():
            key = (int(segment["segment_id"]), scanline_y)
            ours = result_map.get(key)
            if ours is None:
                continue
            ours_s = _finite_float(ours.get("joint_s_px", np.nan))
            normal, _ = _segment_normal_and_tangent(segment)
            ours_estimate = {
                "s_px": ours_s,
                "center_col": _segment_vector(segment, "x_px") + ours_s * normal[0],
                "center_row": _segment_vector(segment, "y_px") + ours_s * normal[1],
                "success": np.isfinite(ours_s),
                "fail_reason": "ok" if np.isfinite(ours_s) else "nan_joint_s",
            }
            rows.append(
                _method_row(
                    segment=segment,
                    scanline_y_um=scanline_y,
                    method="Ours (joint ESF)",
                    n_frames=len(frame_list),
                    estimate=ours_estimate,
                    ours_s_px=ours_s,
                    split_half_diff_px=float(ours.get("split_half_diff_px", np.nan)),
                    crb_px=float(ours.get("crb_px", np.nan)),
                    pixel_size_um=pixel_size_um,
                )
            )

            single = _estimate_canny_s_from_context(single_context, segment)
            rows.append(
                _method_row(
                    segment=segment,
                    scanline_y_um=scanline_y,
                    method="Single Canny",
                    n_frames=1,
                    estimate=single,
                    ours_s_px=ours_s,
                    pixel_size_um=pixel_size_um,
                )
            )

            median = _estimate_canny_s_from_context(median_context, segment)
            even = _estimate_canny_s_from_context(even_context, segment)
            odd = _estimate_canny_s_from_context(odd_context, segment)
            median_split = (
                abs(float(even["s_px"]) - float(odd["s_px"]))
                if bool(even["success"]) and bool(odd["success"])
                else np.nan
            )
            rows.append(
                _method_row(
                    segment=segment,
                    scanline_y_um=scanline_y,
                    method="Median+Canny",
                    n_frames=len(frame_list),
                    estimate=median,
                    ours_s_px=ours_s,
                    split_half_diff_px=median_split,
                    pixel_size_um=pixel_size_um,
                )
            )

            grad_fit = baseline_gaussian_gradient_fit(
                middle_frame,
                segment,
                half_width_px=esf_half_width_px,
                step_px=esf_step_px,
            )
            rows.append(
                _method_row(
                    segment=segment,
                    scanline_y_um=scanline_y,
                    method="Gaussian grad fit",
                    n_frames=1,
                    estimate=grad_fit,
                    ours_s_px=ours_s,
                    pixel_size_um=pixel_size_um,
                )
            )

    out = pd.DataFrame(rows)
    if out.empty:
        return out
    std = (
        out[out["success"].astype(bool)]
        .groupby(["segment_id", "method"])["s_px"]
        .std(ddof=1)
        .rename("cross_scanline_std_px")
        .reset_index()
    )
    out = out.drop(columns=["cross_scanline_std_px"]).merge(std, on=["segment_id", "method"], how="left")
    method_order = {
        "Single Canny": 0,
        "Median+Canny": 1,
        "Gaussian grad fit": 2,
        "Ours (joint ESF)": 3,
    }
    out["method_order"] = out["method"].map(method_order).fillna(99).astype(int)
    if contour is not None:
        out["contour"] = str(contour)
    return out.sort_values(["segment_id", "scanline_y_um", "method_order"]).reset_index(drop=True)


def contour_summary_dict(
    segment_summary: pd.DataFrame,
    boundary_points: pd.DataFrame,
    contour_coordinates_px: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
) -> dict:
    """Build EP04-B apparent thermal contour summary statistics."""
    total_segments = int(segment_summary["segment_id"].nunique()) if not segment_summary.empty else 0
    passed_segments = int(boundary_points["segment_id"].nunique()) if not boundary_points.empty else 0
    source_segments = (
        int(boundary_points[boundary_points["confidence"].isin(["high", "medium"])]["segment_id"].nunique())
        if not boundary_points.empty
        else 0
    )
    high_count = int(boundary_points["confidence"].eq("high").sum()) if not boundary_points.empty else 0

    improvement = np.nan
    if not baseline_comparison.empty:
        precision = (
            baseline_comparison.groupby(["segment_id", "method"], sort=False)["split_half_diff_px"]
            .median()
            .unstack()
        )
        if {"Median+Canny", "Ours (joint ESF)"}.issubset(precision.columns):
            ratio = precision["Median+Canny"] / precision["Ours (joint ESF)"]
            ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
            if not ratio.empty:
                improvement = float(ratio.median())
        if not np.isfinite(improvement):
            consistency = (
                baseline_comparison.groupby(["segment_id", "method"], sort=False)["cross_scanline_std_px"]
                .median()
                .unstack()
            )
            if {"Median+Canny", "Ours (joint ESF)"}.issubset(consistency.columns):
                ratio = consistency["Median+Canny"] / consistency["Ours (joint ESF)"]
                ratio = ratio[np.isfinite(ratio) & (ratio > 0)]
                if not ratio.empty:
                    improvement = float(ratio.median())

    return {
        "total_segments": total_segments,
        "passed_segments": passed_segments,
        "pass_rate": float(passed_segments / total_segments) if total_segments else np.nan,
        "high_confidence_segments": high_count,
        "medium_confidence_segments": int(boundary_points["confidence"].eq("medium").sum()) if not boundary_points.empty else 0,
        "low_confidence_segments": int(boundary_points["confidence"].eq("low").sum()) if not boundary_points.empty else 0,
        "median_split_half_px": float(boundary_points["split_half_diff_px"].median()) if not boundary_points.empty else np.nan,
        "median_crb_ratio": float(boundary_points["crb_ratio"].median()) if not boundary_points.empty else np.nan,
        "contour_coverage_fraction": float(source_segments / total_segments) if total_segments else np.nan,
        "baseline_improvement_median": improvement,
    }


def save_ep04b_outputs(
    boundary_points: pd.DataFrame,
    contour_coordinates_px: pd.DataFrame,
    contour_coordinates_mm: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    summary: dict,
    output_dir: Path,
) -> None:
    """Save all EP04-B CSV and JSON outputs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    boundary_points.to_csv(output_dir / "thermal_boundary_points.csv", index=False)
    contour_coordinates_px.to_csv(output_dir / "contour_coordinates_px.csv", index=False)
    contour_coordinates_mm.to_csv(output_dir / "contour_coordinates_mm.csv", index=False)
    baseline_comparison.to_csv(output_dir / "baseline_comparison.csv", index=False)
    with open(output_dir / "contour_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)


def run_ep04b_contour_reconstruction(
    segment_validation_csv: Path,
    segment_summary_csv: Path,
    frame_audit_csv: Path,
    data_dir: Path,
    output_dir: Path,
    *,
    segments_csv: Path | None = None,
    pixel_size_um: float = 20.0,
    force_baseline: bool = False,
    contour: str | None = None,
) -> dict[str, pd.DataFrame | dict]:
    """Run EP04-B apparent thermal contour reconstruction and baseline comparison."""
    results = pd.read_csv(segment_validation_csv)
    segment_summary = pd.read_csv(segment_summary_csv)
    frame_audit = pd.read_csv(frame_audit_csv)
    segments = pd.read_csv(segments_csv) if segments_csv is not None and Path(segments_csv).exists() else None

    boundary_points = extract_thermal_boundary_points(
        segment_summary,
        segments=segments,
        pixel_size_um=pixel_size_um,
    )
    contour_px, contour_mm = stitch_thermal_contour(
        boundary_points,
        pixel_size_um=pixel_size_um,
        min_confidence="medium",
    )

    baseline_path = Path(output_dir) / "baseline_comparison.csv"
    if baseline_path.exists() and not force_baseline:
        baseline = pd.read_csv(baseline_path)
        if contour is not None and "contour" not in baseline.columns:
            baseline["contour"] = str(contour)
    else:
        baseline = run_baseline_comparison(
            results,
            segment_summary,
            frame_audit,
            Path(data_dir),
            segments=segments,
            pixel_size_um=pixel_size_um,
            contour=contour,
        )

    summary = contour_summary_dict(segment_summary, boundary_points, contour_px, baseline)
    save_ep04b_outputs(boundary_points, contour_px, contour_mm, baseline, summary, Path(output_dir))
    return {
        "boundary_points": boundary_points,
        "contour_coordinates_px": contour_px,
        "contour_coordinates_mm": contour_mm,
        "baseline_comparison": baseline,
        "contour_summary": summary,
    }


def _outer_contour_xy(reference_frame: np.ndarray) -> np.ndarray | None:
    try:
        _, contour, _ = detect_outer_contour(reference_frame)
        return contour
    except Exception:  # noqa: BLE001 - plotting can fall back to segment centers.
        return None


def _boundary_segments_for_lines(points: pd.DataFrame, *, min_confidence: str = "medium") -> tuple[list[np.ndarray], np.ndarray]:
    line_segments: list[np.ndarray] = []
    line_values: list[float] = []
    for run in _valid_contour_runs(points, min_confidence=min_confidence):
        run = run.sort_values("contour_index")
        xy = run[["center_col", "center_row"]].to_numpy(dtype=float)
        values = run["split_half_diff_px"].to_numpy(dtype=float)
        for i in range(len(xy) - 1):
            line_segments.append(np.vstack([xy[i], xy[i + 1]]))
            line_values.append(float(np.nanmean(values[i : i + 2])))
    return line_segments, np.asarray(line_values, dtype=float)


def _plot_apparent_boundary_overlay(
    ax: plt.Axes,
    reference_frame: np.ndarray,
    boundary_points: pd.DataFrame,
    segment_summary: pd.DataFrame,
    *,
    segments: pd.DataFrame | None = None,
) -> LineCollection | None:
    im = ax.imshow(reference_frame, cmap="inferno", origin="upper")
    contour = _outer_contour_xy(reference_frame)
    if contour is not None:
        ax.plot(contour[:, 0], contour[:, 1], color="white", linestyle="--", linewidth=0.7, alpha=0.75, label="Otsu outer contour")
    else:
        ordered = segment_summary.sort_values("segment_id")
        ax.plot(ordered["x_px"], ordered["y_px"], color="white", linestyle="--", linewidth=0.7, alpha=0.75, label="Otsu outer contour")

    all_points = extract_thermal_boundary_points(
        segment_summary,
        segments=segments,
        include_failed=True,
    )
    weak = all_points[~all_points["confidence"].isin(["high", "medium"])].sort_values("contour_index")
    weak_label_used = False
    weak_run: list[pd.Series] = []
    last_id: int | None = None
    for _, weak_row in weak.iterrows():
        segment_id = int(weak_row["segment_id"])
        if last_id is None or segment_id == last_id + 1:
            weak_run.append(weak_row)
        else:
            if weak_run:
                run = pd.DataFrame(weak_run)
                ax.plot(
                    run["segment_center_col"],
                    run["segment_center_row"],
                    color="#bdbdbd",
                    linestyle="--",
                    linewidth=1.0,
                    alpha=0.78,
                    label="low/fail gap" if not weak_label_used else None,
                )
                weak_label_used = True
            weak_run = [weak_row]
        last_id = segment_id
    if weak_run:
        run = pd.DataFrame(weak_run)
        ax.plot(
            run["segment_center_col"],
            run["segment_center_row"],
            color="#bdbdbd",
            linestyle="--",
            linewidth=1.0,
            alpha=0.78,
            label="low/fail gap" if not weak_label_used else None,
        )

    line_segments, values = _boundary_segments_for_lines(boundary_points, min_confidence="medium")
    collection = None
    if line_segments:
        vmax = float(np.nanpercentile(boundary_points["split_half_diff_px"], 90))
        vmax = max(vmax, 0.04)
        collection = LineCollection(
            line_segments,
            array=values,
            cmap="viridis_r",
            linewidths=1.8,
            norm=plt.Normalize(0.0, vmax),
            label="apparent thermal boundary",
        )
        ax.add_collection(collection)
    marker_colors = {"high": METHOD_COLOR_LIST[1], "medium": METHOD_COLOR_LIST[0], "low": METHOD_COLOR_LIST[4]}
    for confidence, group in boundary_points.groupby("confidence"):
        ax.scatter(
            group["center_col"],
            group["center_row"],
            s=20 if confidence != "low" else 14,
            c=marker_colors.get(confidence, "#999999"),
            edgecolor="white",
            linewidth=0.25,
            alpha=0.92,
            label=f"{confidence} point",
        )
    ax.set_xlabel("Column [px]")
    ax.set_ylabel("Row [px]")
    ax.set_title("Apparent Thermal Contour")
    return collection if collection is not None else im


def _choose_boundary_ids(boundary_points: pd.DataFrame, n: int = 3) -> list[int]:
    if boundary_points.empty:
        return []
    chosen: list[int] = []
    high = boundary_points[boundary_points["confidence"].eq("high")].sort_values("split_half_diff_px")
    medium = boundary_points[boundary_points["confidence"].eq("medium")].sort_values("split_half_diff_px")
    for frame in (high.head(1), medium.iloc[[len(medium) // 2]] if not medium.empty else medium, boundary_points.sort_values("split_half_diff_px").tail(1)):
        for segment_id in frame.get("segment_id", pd.Series(dtype=int)).astype(int).tolist():
            if segment_id not in chosen:
                chosen.append(segment_id)
            if len(chosen) >= n:
                return chosen
    for segment_id in boundary_points.sort_values("split_half_diff_px")["segment_id"].astype(int):
        if segment_id not in chosen:
            chosen.append(segment_id)
        if len(chosen) >= n:
            break
    return chosen


def _method_median_points(baseline_comparison: pd.DataFrame, segment_id: int) -> pd.DataFrame:
    group = baseline_comparison[baseline_comparison["segment_id"].astype(int).eq(int(segment_id))]
    if group.empty:
        return pd.DataFrame()
    return (
        group[group["success"].astype(bool)]
        .groupby("method", sort=False)
        .agg(
            center_col=("center_col", "median"),
            center_row=("center_row", "median"),
            s_px=("s_px", "median"),
            cross_scanline_std_px=("cross_scanline_std_px", "median"),
        )
        .reset_index()
    )


def plot_thermal_contour_hero(
    reference_frame: np.ndarray,
    boundary_points: pd.DataFrame,
    segment_summary: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    *,
    segments: pd.DataFrame | None = None,
) -> plt.Figure:
    """Create the EP04-B hero apparent thermal contour figure."""
    fig = plt.figure(figsize=(7.2, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 0.85])
    ax_map = fig.add_subplot(gs[0, 0])
    ax_zoom = fig.add_subplot(gs[0, 1])
    ax_profile = fig.add_subplot(gs[1, :])

    mappable = _plot_apparent_boundary_overlay(ax_map, reference_frame, boundary_points, segment_summary, segments=segments)
    if mappable is not None:
        cbar = fig.colorbar(mappable, ax=ax_map, fraction=0.046, pad=0.02)
        format_colorbar(cbar, "Split-half diff [px]")
    ax_map.legend(loc="upper right", fontsize=6)

    chosen_ids = _choose_boundary_ids(boundary_points, n=1)
    if chosen_ids:
        segment_id = chosen_ids[0]
        point = boundary_points[boundary_points["segment_id"].astype(int).eq(segment_id)].iloc[0]
        half = 34.0
        x0, y0 = float(point["center_col"]), float(point["center_row"])
        ax_zoom.imshow(reference_frame, cmap="inferno", origin="upper")
        ax_zoom.set_xlim(x0 - half, x0 + half)
        ax_zoom.set_ylim(y0 + half, y0 - half)
        local_methods = _method_median_points(baseline_comparison, segment_id)
        style_map = {
            "Single Canny": ("x", METHOD_COLOR_LIST[2]),
            "Median+Canny": ("s", METHOD_COLOR_LIST[3]),
            "Gaussian grad fit": ("^", METHOD_COLOR_LIST[4]),
            "Ours (joint ESF)": ("o", METHOD_COLOR_LIST[1]),
        }
        for _, row in local_methods.iterrows():
            marker, color = style_map.get(str(row["method"]), ("o", "#333333"))
            kwargs = {"edgecolor": "white", "linewidth": 0.35} if marker != "x" else {"linewidth": 0.8}
            ax_zoom.scatter(row["center_col"], row["center_row"], marker=marker, s=34, c=color, label=str(row["method"]), **kwargs)
        normal = np.array([np.cos(np.radians(point["normal_angle_deg"])), np.sin(np.radians(point["normal_angle_deg"]))])
        bar = max(float(point["split_half_diff_px"]) * 10.0, 0.4)
        ax_zoom.plot(
            [x0 - bar * normal[0], x0 + bar * normal[0]],
            [y0 - bar * normal[1], y0 + bar * normal[1]],
            color="white",
            linewidth=1.0,
            alpha=0.9,
        )
        ax_zoom.set_title(f"Local Comparison (seg {segment_id})")
        ax_zoom.legend(loc="lower right", fontsize=5)
    else:
        ax_zoom.text(0.5, 0.5, "No passed boundary points", transform=ax_zoom.transAxes, ha="center", va="center")
    ax_zoom.set_xlabel("Column [px]")
    ax_zoom.set_ylabel("Row [px]")

    prof = boundary_points.sort_values("contour_index")
    ax_profile.plot(prof["segment_id"], prof["split_half_diff_px"], color=METHOD_COLOR_LIST[0], marker="o", markersize=3.0, linewidth=1.0, label="split-half")
    ax_profile.plot(prof["segment_id"], prof["crb_px"], color=METHOD_COLOR_LIST[2], linestyle="--", linewidth=1.0, label="CRB")
    for confidence, marker in [("high", "o"), ("medium", "s"), ("low", "^")]:
        group = prof[prof["confidence"].eq(confidence)]
        ax_profile.scatter(group["segment_id"], group["split_half_diff_px"], s=18, marker=marker, label=confidence, zorder=3)
    ax_profile.set_xlabel("Contour segment ID")
    ax_profile.set_ylabel("Uncertainty [px]")
    ax_profile.set_title("Normal-Direction Uncertainty Profile")
    ax_profile.grid(axis="y", alpha=0.25)
    ax_profile.legend(loc="upper right", fontsize=7, ncol=5)
    return fig


def plot_baseline_comparison(
    reference_frame: np.ndarray,
    boundary_points: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
) -> plt.Figure:
    """Plot local apparent thermal boundary estimates and cross-scanline baseline summary."""
    fig, axes = make_figure("double_col", nrows=2, ncols=2, height=3.0)
    axes = np.asarray(axes).ravel()
    segment_ids = _choose_boundary_ids(boundary_points, n=3)
    style_map = {
        "Single Canny": ("x", METHOD_COLOR_LIST[2]),
        "Median+Canny": ("s", METHOD_COLOR_LIST[3]),
        "Gaussian grad fit": ("^", METHOD_COLOR_LIST[4]),
        "Ours (joint ESF)": ("o", METHOD_COLOR_LIST[1]),
    }
    for ax, segment_id in zip(axes[:3], segment_ids):
        point = boundary_points[boundary_points["segment_id"].astype(int).eq(segment_id)].iloc[0]
        x0, y0 = float(point["center_col"]), float(point["center_row"])
        ax.imshow(reference_frame, cmap="inferno", origin="upper")
        ax.set_xlim(x0 - 30.0, x0 + 30.0)
        ax.set_ylim(y0 + 30.0, y0 - 30.0)
        local_methods = _method_median_points(baseline_comparison, segment_id)
        for _, row in local_methods.iterrows():
            marker, color = style_map.get(str(row["method"]), ("o", "#333333"))
            kwargs = {"edgecolor": "white", "linewidth": 0.35} if marker != "x" else {"linewidth": 0.8}
            ax.scatter(row["center_col"], row["center_row"], marker=marker, s=36, c=color, label=str(row["method"]), **kwargs)
        ax.set_title(f"Segment {segment_id}")
        ax.set_xlabel("Column [px]")
        ax.set_ylabel("Row [px]")
    if segment_ids:
        axes[0].legend(loc="lower right", fontsize=5)

    ax = axes[3]
    summary = (
        baseline_comparison.groupby(["segment_id", "method"], sort=False)["cross_scanline_std_px"]
        .median()
        .reset_index()
        .groupby("method", sort=False)["cross_scanline_std_px"]
        .median()
        .reindex(["Single Canny", "Median+Canny", "Gaussian grad fit", "Ours (joint ESF)"])
    )
    colors = [style_map.get(method, ("o", "#333333"))[1] for method in summary.index]
    ax.bar(np.arange(len(summary)), summary.to_numpy(dtype=float), color=colors, alpha=0.86)
    ax.set_xticks(np.arange(len(summary)))
    ax.set_xticklabels(summary.index, rotation=25, ha="right")
    ax.set_ylabel("Median cross-scanline std [px]")
    ax.set_title("Cross-Scanline Consistency")
    ax.grid(axis="y", alpha=0.25)
    return fig


def plot_uncertainty_map(
    reference_frame: np.ndarray,
    boundary_points: pd.DataFrame,
    segment_summary: pd.DataFrame,
    *,
    segments: pd.DataFrame | None = None,
) -> plt.Figure:
    """Plot apparent thermal boundary uncertainty and confidence classes on the reference frame."""
    fig, ax = make_figure("double_col", height=4.4)
    im = ax.imshow(reference_frame, cmap="inferno", origin="upper")
    all_points = extract_thermal_boundary_points(segment_summary, segments=segments, include_failed=True)
    fail = all_points[all_points["confidence"].eq("fail")]
    if not fail.empty:
        ax.scatter(fail["segment_center_col"], fail["segment_center_row"], marker="x", s=22, c="#bdbdbd", linewidth=0.8, label="fail")
    markers = {"high": "o", "medium": "s", "low": "^"}
    for confidence, marker in markers.items():
        group = boundary_points[boundary_points["confidence"].eq(confidence)]
        if group.empty:
            continue
        ax.scatter(
            group["center_col"],
            group["center_row"],
            c=group["split_half_diff_px"],
            cmap="viridis_r",
            vmin=0.0,
            vmax=max(float(boundary_points["split_half_diff_px"].quantile(0.90)), 0.04),
            marker=marker,
            s=34 if confidence != "low" else 25,
            edgecolor="white",
            linewidth=0.35,
            label=confidence,
        )
    fig.subplots_adjust(right=0.82)
    temp_cax = fig.add_axes([0.845, 0.18, 0.018, 0.64])
    uncertainty_cax = fig.add_axes([0.915, 0.18, 0.018, 0.64])
    cbar = fig.colorbar(im, cax=temp_cax)
    format_colorbar(cbar, "Temperature [C]")
    mappable = plt.cm.ScalarMappable(
        norm=plt.Normalize(0.0, max(float(boundary_points["split_half_diff_px"].quantile(0.90)), 0.04)),
        cmap="viridis_r",
    )
    cbar2 = fig.colorbar(mappable, cax=uncertainty_cax)
    format_colorbar(cbar2, "Split-half diff [px]")
    ax.set_xlabel("Column [px]")
    ax.set_ylabel("Row [px]")
    ax.legend(loc="upper right", fontsize=7)
    return fig


def method_summary_table(baseline_comparison: pd.DataFrame, boundary_points: pd.DataFrame) -> pd.DataFrame:
    """Create a compact method comparison table from measured EP04-B outputs."""
    methods = ["Single Canny", "Median+Canny", "Gaussian grad fit", "Ours (joint ESF)"]
    rows = []
    for method in methods:
        group = baseline_comparison[baseline_comparison["method"].eq(method)]
        n_frames = int(group["n_frames"].median()) if not group.empty else 0
        precision = (
            group.groupby("segment_id")["split_half_diff_px"].median().dropna()
            if method in {"Median+Canny", "Ours (joint ESF)"}
            else group.groupby("segment_id")["cross_scanline_std_px"].median().dropna()
        )
        if method == "Ours (joint ESF)" and not boundary_points.empty:
            precision = boundary_points.set_index("segment_id")["split_half_diff_px"].dropna()
        median = float(precision.median()) if not precision.empty else np.nan
        p90 = float(precision.quantile(0.90)) if not precision.empty else np.nan
        rows.append(
            {
                "Method": method,
                "Frames": n_frames,
                "Median precision": f"{median:.3f} px" if np.isfinite(median) else "n/a",
                "P90": f"{p90:.3f} px" if np.isfinite(p90) else "n/a",
                "Needs shift?": "Data-driven" if method == "Ours (joint ESF)" else "No",
                "Cost": "Medium" if method == "Ours (joint ESF)" else "Low",
            }
        )
    return pd.DataFrame(rows)


def precision_points_from_segment_summary(segment_summary: pd.DataFrame) -> pd.DataFrame:
    """Convert passed segment summaries into method-table precision points."""
    points = _passed(segment_summary).copy()
    if points.empty:
        return pd.DataFrame(columns=["segment_id", "split_half_diff_px"])
    return pd.DataFrame(
        {
            "segment_id": points["segment_id"].astype(int).to_numpy(),
            "split_half_diff_px": pd.to_numeric(points["split_half_median_px"], errors="coerce").to_numpy(dtype=float),
        }
    ).dropna(subset=["split_half_diff_px"])


def _choose_representative_fit_specs(
    boundary_points: pd.DataFrame,
    segment_summary: pd.DataFrame,
) -> list[tuple[str, int]]:
    specs: list[tuple[str, int]] = []
    high = boundary_points[boundary_points["confidence"].eq("high")].sort_values("split_half_diff_px")
    if not high.empty:
        specs.append(("High precision", int(high.iloc[0]["segment_id"])))
    medium = boundary_points[boundary_points["confidence"].eq("medium")].sort_values("split_half_diff_px")
    if not medium.empty:
        specs.append(("Medium precision", int(medium.iloc[len(medium) // 2]["segment_id"])))
    fail = segment_summary[~segment_summary["pass_fail"].astype(bool)].copy()
    if not fail.empty:
        fail = fail.sort_values(["snr", "normal_projection"], ascending=False)
        specs.append(("Quality-gate fail", int(fail.iloc[0]["segment_id"])))

    used: set[int] = set()
    deduped: list[tuple[str, int]] = []
    for label, segment_id in specs:
        if segment_id in used:
            continue
        used.add(segment_id)
        deduped.append((label, segment_id))
    for _, row in boundary_points.sort_values("split_half_diff_px").iterrows():
        segment_id = int(row["segment_id"])
        if segment_id not in used:
            deduped.append((str(row["confidence"]).title(), segment_id))
            used.add(segment_id)
        if len(deduped) >= 3:
            break
    return deduped[:3]


def _scanline_file_map(frame_audit_df: pd.DataFrame) -> dict[float, list[str]]:
    return {float(line["Y"].iloc[0]): [str(name) for name in line["file"].tolist()] for line in build_x_scanlines(frame_audit_df)}


def _fit_joint_for_plot(
    segment: pd.Series,
    scanline_files: list[str],
    frames: dict[str, np.ndarray],
    *,
    config: Ep04Config | None = None,
) -> tuple[pd.DataFrame, dict]:
    config = Ep04Config() if config is None else config
    frame_list = [frames[str(name)] for name in scanline_files]
    roi = _centered_roi_slices(
        frame_list[0].shape,
        float(segment["x_px"]),
        float(segment["y_px"]),
        config.ncc_roi_size,
    )
    if roi is None:
        raise RuntimeError("segment ROI is outside the frame")
    normal, _ = _segment_normal_and_tangent(segment)
    ncc_deltas, _ = _local_ncc_deltas(frame_list, roi, normal, config=config)
    profiles = _profiles_from_frames(
        frame_list,
        scanline_files,
        segment,
        ncc_deltas,
        config=config,
    )
    return profiles, fit_joint_esf(profiles)


def _normalised_joint_esf_data(fit: dict) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    data = fit["data"].copy()
    params = np.asarray(fit["result"].x, dtype=float)
    frames = list(fit["frames"])
    frame_to_pos = {frame_id: i for i, frame_id in enumerate(frames)}
    per_frame = params[2:].reshape(int(fit["n_frames"]), 3)
    pos = data["frame_index"].map(frame_to_pos).to_numpy(dtype=int)
    u = data["u_px"].to_numpy(dtype=float)
    temp = data["temperature_c"].to_numpy(dtype=float)
    a = per_frame[pos, 0]
    b = per_frame[pos, 1]
    c = per_frame[pos, 2]
    c_safe = np.where(np.abs(c) < 1e-9, np.nan, c)
    data["u_aligned_px"] = u - data["delta_n_px"].to_numpy(dtype=float)
    data["normalised_temperature"] = (temp - (a + b * u)) / c_safe

    curve_u = np.linspace(
        float(data["u_aligned_px"].quantile(0.01)),
        float(data["u_aligned_px"].quantile(0.99)),
        240,
    )
    curve = normal_cdf((curve_u - float(fit["s_px"])) / float(fit["sigma_px"]))
    return data, curve_u, curve


def plot_representative_esf_fits(
    results: pd.DataFrame,
    segment_summary: pd.DataFrame,
    boundary_points: pd.DataFrame,
    frame_audit_df: pd.DataFrame,
    data_dir: Path,
    *,
    segments: pd.DataFrame | None = None,
) -> plt.Figure:
    """Plot joint ESF fits for representative high, medium, and failed segments."""
    specs = _choose_representative_fit_specs(boundary_points, segment_summary)
    fig, axes = make_figure("double_col", nrows=1, ncols=3, height=2.8)
    axes = np.asarray(axes).ravel()
    if not specs:
        for ax in axes:
            ax.text(0.5, 0.5, "No representative segments", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
        return fig

    merged_segments = _merge_segment_geometry(segment_summary, segments)
    scanlines = _scanline_file_map(frame_audit_df)
    needed_files = [name for files in scanlines.values() for name in files]
    frame_cache = preload_frames(Path(data_dir), needed_files)

    for ax, (label, segment_id) in zip(axes, specs):
        segment = merged_segments[merged_segments["segment_id"].astype(int).eq(int(segment_id))].iloc[0]
        rows = results[results["segment_id"].astype(int).eq(int(segment_id))].copy()
        if rows.empty:
            ax.text(0.5, 0.5, f"Segment {segment_id}\nNo EP04-A row", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue
        rows = rows.sort_values(["pass_fail", "split_half_diff_px"], ascending=[False, True])
        row = rows.iloc[0]
        scanline_y = float(row["scanline_y_um"])
        files = scanlines.get(scanline_y)
        if files is None:
            ax.text(0.5, 0.5, f"Segment {segment_id}\nNo scanline", transform=ax.transAxes, ha="center", va="center")
            ax.set_axis_off()
            continue
        try:
            _, fit = _fit_joint_for_plot(segment, files, frame_cache)
            data, curve_u, curve = _normalised_joint_esf_data(fit)
            ax.scatter(
                data["u_aligned_px"],
                data["normalised_temperature"],
                s=4,
                color=METHOD_COLOR_LIST[0],
                alpha=0.22,
                linewidth=0,
                label="aligned samples",
            )
            ax.plot(curve_u, curve, color=METHOD_COLOR_LIST[2], linewidth=1.3, label="joint ESF fit")
            ax.axvline(float(fit["s_px"]), color="#333333", linestyle="--", linewidth=0.9)
            ax.set_ylim(-0.25, 1.25)
            ax.set_title(f"{label}\nseg {segment_id}, split={float(row['split_half_diff_px']):.3f}px, CRB={float(row['crb_px']):.3f}px")
        except Exception as exc:  # noqa: BLE001 - a failed representative should stay visible.
            ax.text(0.5, 0.5, f"Segment {segment_id}\n{type(exc).__name__}", transform=ax.transAxes, ha="center", va="center")
            ax.set_title(f"{label}\nseg {segment_id}")
        ax.set_xlabel("Aligned normal coordinate [px]")
        ax.set_ylabel("Normalised temperature")
        ax.grid(alpha=0.20)
    axes[0].legend(loc="lower right", fontsize=6)
    return fig


def create_ep04b_figures(
    results: pd.DataFrame,
    segment_summary: pd.DataFrame,
    boundary_points: pd.DataFrame,
    baseline_comparison: pd.DataFrame,
    reference_frame: np.ndarray,
    frame_audit_df: pd.DataFrame,
    data_dir: Path,
    output_dir: Path,
    *,
    segments: pd.DataFrame | None = None,
) -> dict[str, Path]:
    """Create and save the five required EP04-B apparent thermal contour figures."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    figures = {
        "thermal_contour_hero.png": plot_thermal_contour_hero(
            reference_frame,
            boundary_points,
            segment_summary,
            baseline_comparison,
            segments=segments,
        ),
        "baseline_comparison.png": plot_baseline_comparison(
            reference_frame,
            boundary_points,
            baseline_comparison,
        ),
        "uncertainty_map.png": plot_uncertainty_map(
            reference_frame,
            boundary_points,
            segment_summary,
            segments=segments,
        ),
        "representative_esf_fits.png": plot_representative_esf_fits(
            results,
            segment_summary,
            boundary_points,
            frame_audit_df,
            data_dir,
            segments=segments,
        ),
    }
    saved: dict[str, Path] = {}
    for name, fig in figures.items():
        saved[name] = savefig_academic(fig, output_dir / name)
    return saved
