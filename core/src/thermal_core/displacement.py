"""Displacement estimation and stage calibration utilities."""

from __future__ import annotations

from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, sobel

from thermal_core.io import load_frame


PIXEL_SIZE_UM = 10.0


def center_roi(shape: tuple[int, int], size: int = 320) -> tuple[slice, slice]:
    """Return centered row/column slices for a square ROI."""
    rows, cols = shape
    if size > rows or size > cols:
        raise ValueError(f"ROI size {size} exceeds frame shape {shape}")
    r0 = (rows - size) // 2
    c0 = (cols - size) // 2
    return slice(r0, r0 + size), slice(c0, c0 + size)


def coordinate_to_shift(
    x_um: np.ndarray | float,
    y_um: np.ndarray | float,
    theta_deg: float = 47.6,
    pixel_size_um: float = PIXEL_SIZE_UM,
) -> tuple[np.ndarray, np.ndarray]:
    """Map stage coordinates in micrometers to pixel displacement."""
    theta = np.radians(theta_deg)
    x = np.asarray(x_um, dtype=float)
    y = np.asarray(y_um, dtype=float)
    dx = (x * np.cos(theta) + y * np.sin(theta)) / pixel_size_um
    dy = (-x * np.sin(theta) + y * np.cos(theta)) / pixel_size_um
    return dx, dy


def _crop(frame: np.ndarray, roi: tuple[slice, slice] | None, roi_size: int) -> np.ndarray:
    if roi is None:
        roi = center_roi(frame.shape, roi_size)
    return np.asarray(frame[roi], dtype=np.float32)


def _znorm(frame: np.ndarray) -> np.ndarray:
    centered = frame - float(np.mean(frame))
    scale = float(np.std(centered))
    if scale == 0.0:
        return centered
    return centered / scale


def preprocess_for_registration(
    frame: np.ndarray,
    mode: str = "raw",
    *,
    highpass_sigma: float = 12.0,
) -> np.ndarray:
    """Prepare a cropped frame for displacement registration.

    ``raw`` keeps the thermal field after z-normalization, ``highpass`` removes
    a smooth thermal baseline, and ``gradient`` keeps Sobel edge magnitude.
    """
    data = np.asarray(frame, dtype=np.float32)
    if mode == "raw":
        return _znorm(data)
    if mode == "highpass":
        baseline = gaussian_filter(data, sigma=highpass_sigma, mode="nearest")
        return _znorm(data - baseline)
    if mode == "gradient":
        gx = sobel(data, axis=1, mode="nearest")
        gy = sobel(data, axis=0, mode="nearest")
        return _znorm(np.hypot(gx, gy))
    raise ValueError("preprocess mode must be 'raw', 'highpass', or 'gradient'")


def _overlap_for_shift(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    dx: int,
    dy: int,
) -> tuple[np.ndarray, np.ndarray]:
    rows, cols = frame_a.shape
    if dx >= 0:
        ax = slice(0, cols - dx)
        bx = slice(dx, cols)
    else:
        ax = slice(-dx, cols)
        bx = slice(0, cols + dx)

    if dy >= 0:
        ay = slice(0, rows - dy)
        by = slice(dy, rows)
    else:
        ay = slice(-dy, rows)
        by = slice(0, rows + dy)

    return frame_a[ay, ax], frame_b[by, bx]


def _quadratic_peak(
    corr: np.ndarray,
    x_values: np.ndarray,
    y_values: np.ndarray,
    peak_row: int,
    peak_col: int,
    fit_radius: int,
) -> tuple[float, float, bool]:
    r0 = max(0, peak_row - fit_radius)
    r1 = min(corr.shape[0], peak_row + fit_radius + 1)
    c0 = max(0, peak_col - fit_radius)
    c1 = min(corr.shape[1], peak_col + fit_radius + 1)
    if r1 - r0 < 3 or c1 - c0 < 3:
        return float(x_values[peak_col]), float(y_values[peak_row]), False

    yy, xx = np.meshgrid(y_values[r0:r1], x_values[c0:c1], indexing="ij")
    zz = corr[r0:r1, c0:c1].ravel()
    design = np.column_stack([
        xx.ravel() ** 2,
        yy.ravel() ** 2,
        xx.ravel() * yy.ravel(),
        xx.ravel(),
        yy.ravel(),
        np.ones(zz.size),
    ])
    a, b, c, d, e, _ = np.linalg.lstsq(design, zz, rcond=None)[0]
    hessian = np.array([[2.0 * a, c], [c, 2.0 * b]], dtype=float)
    rhs = np.array([-d, -e], dtype=float)

    try:
        peak = np.linalg.solve(hessian, rhs)
    except np.linalg.LinAlgError:
        return float(x_values[peak_col]), float(y_values[peak_row]), False

    x_min = float(x_values[c0])
    x_max = float(x_values[c1 - 1])
    y_min = float(y_values[r0])
    y_max = float(y_values[r1 - 1])
    if not (x_min <= peak[0] <= x_max and y_min <= peak[1] <= y_max):
        return float(x_values[peak_col]), float(y_values[peak_row]), False
    return float(peak[0]), float(peak[1]), True


def subpixel_ncc(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    roi: tuple[slice, slice] | None = None,
    search_radius: int = 5,
    fit_radius: int = 2,
    roi_size: int = 320,
    preprocess: str = "raw",
    highpass_sigma: float = 12.0,
) -> dict[str, float | int | bool]:
    """Estimate image displacement from frame A to frame B using NCC."""
    a = preprocess_for_registration(
        _crop(frame_a, roi, roi_size),
        preprocess,
        highpass_sigma=highpass_sigma,
    )
    b = preprocess_for_registration(
        _crop(frame_b, roi, roi_size),
        preprocess,
        highpass_sigma=highpass_sigma,
    )

    shifts = np.arange(-search_radius, search_radius + 1)
    corr = np.empty((shifts.size, shifts.size), dtype=float)
    for iy, dy in enumerate(shifts):
        for ix, dx in enumerate(shifts):
            aa, bb = _overlap_for_shift(a, b, int(dx), int(dy))
            corr[iy, ix] = float(np.mean(aa * bb))

    peak_row, peak_col = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dx, dy, fit_ok = _quadratic_peak(corr, shifts, shifts, peak_row, peak_col, fit_radius)

    return {
        "dx_px": dx,
        "dy_px": dy,
        "peak_ncc": float(corr[peak_row, peak_col]),
        "integer_dx_px": int(shifts[peak_col]),
        "integer_dy_px": int(shifts[peak_row]),
        "fit_ok": bool(fit_ok),
        "edge_peak": bool(
            peak_row in (0, corr.shape[0] - 1) or peak_col in (0, corr.shape[1] - 1)
        ),
    }


def _parabolic_offset(values: np.ndarray, index: int) -> float:
    if index <= 0 or index >= values.size - 1:
        return 0.0
    left, center, right = values[index - 1], values[index], values[index + 1]
    denom = left - 2.0 * center + right
    if denom == 0.0:
        return 0.0
    return float(0.5 * (left - right) / denom)


def phase_correlation(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    roi: tuple[slice, slice] | None = None,
    roi_size: int = 320,
    preprocess: str = "raw",
    highpass_sigma: float = 12.0,
) -> dict[str, float | int]:
    """Estimate image displacement from frame A to frame B by phase correlation."""
    a = preprocess_for_registration(
        _crop(frame_a, roi, roi_size),
        preprocess,
        highpass_sigma=highpass_sigma,
    )
    b = preprocess_for_registration(
        _crop(frame_b, roi, roi_size),
        preprocess,
        highpass_sigma=highpass_sigma,
    )
    wy = np.hanning(a.shape[0])[:, None]
    wx = np.hanning(a.shape[1])[None, :]
    window = wy * wx

    fa = np.fft.fft2(a * window)
    fb = np.fft.fft2(b * window)
    cross_power = fb * np.conj(fa)
    cross_power /= np.maximum(np.abs(cross_power), 1e-12)
    corr = np.abs(np.fft.ifft2(cross_power))

    peak_row, peak_col = np.unravel_index(int(np.argmax(corr)), corr.shape)
    dy = peak_row if peak_row <= corr.shape[0] // 2 else peak_row - corr.shape[0]
    dx = peak_col if peak_col <= corr.shape[1] // 2 else peak_col - corr.shape[1]

    row_profile = corr[:, peak_col]
    col_profile = corr[peak_row, :]
    dy += _parabolic_offset(row_profile, peak_row)
    dx += _parabolic_offset(col_profile, peak_col)

    return {
        "dx_px": float(dx),
        "dy_px": float(dy),
        "peak_phase": float(corr[peak_row, peak_col]),
        "integer_dx_px": int(peak_col if peak_col <= corr.shape[1] // 2 else peak_col - corr.shape[1]),
        "integer_dy_px": int(peak_row if peak_row <= corr.shape[0] // 2 else peak_row - corr.shape[0]),
    }


def _session_column(df: pd.DataFrame, session_col: str | None) -> str:
    if session_col:
        return session_col
    for candidate in ("session", "session_id"):
        if candidate in df.columns:
            return candidate
    raise ValueError("No session column found; expected 'session' or 'session_id'")


def build_frame_pairs(
    df_audit: pd.DataFrame,
    *,
    axis: str,
    r_value: int = 0,
    session_col: str | None = None,
    max_delta_um: float | None = None,
) -> pd.DataFrame:
    """Build same-session adjacent frame pairs along one stage axis."""
    if axis not in {"x", "y"}:
        raise ValueError("axis must be 'x' or 'y'")

    session = _session_column(df_audit, session_col)
    moving = "X" if axis == "x" else "Y"
    fixed = "Y" if axis == "x" else "X"
    rows = []

    subset = df_audit[df_audit["R"].astype(int) == r_value].copy()
    for keys, group in subset.groupby([session, fixed], sort=True):
        group = group.sort_values(moving)
        records = group.to_dict("records")
        for a, b in zip(records, records[1:]):
            delta = float(b[moving] - a[moving])
            if delta <= 0:
                continue
            if max_delta_um is not None and delta > max_delta_um:
                continue
            rows.append({
                "scan_axis": axis,
                "session": int(a[session]),
                "fixed_coord_um": int(a[fixed]),
                "file_a": a["file"],
                "file_b": b["file"],
                "X_a": int(a["X"]),
                "Y_a": int(a["Y"]),
                "X_b": int(b["X"]),
                "Y_b": int(b["Y"]),
                "delta_X_um": int(b["X"] - a["X"]),
                "delta_Y_um": int(b["Y"] - a["Y"]),
                "delta_um": delta,
            })

    return pd.DataFrame(rows)


def build_time_adjacent_pairs(
    df_audit: pd.DataFrame,
    *,
    r_value: int | None = None,
    session_col: str | None = None,
    max_order_gap: int = 1,
) -> pd.DataFrame:
    """Build physically time-adjacent frame pairs from acquisition order."""
    session = _session_column(df_audit, session_col)
    sorted_df = df_audit.sort_values("acquisition_order").reset_index(drop=True)
    rows = []

    for a, b in zip(sorted_df.to_dict("records"), sorted_df.iloc[1:].to_dict("records")):
        if int(a[session]) != int(b[session]):
            continue
        order_gap = int(b["acquisition_order"] - a["acquisition_order"])
        if order_gap <= 0 or order_gap > max_order_gap:
            continue
        if r_value is not None and (int(a["R"]) != r_value or int(b["R"]) != r_value):
            continue

        delta_x = int(b["X"] - a["X"])
        delta_y = int(b["Y"] - a["Y"])
        if delta_y == 0 and delta_x > 0:
            move_type = "x_step"
        elif delta_x < 0 and delta_y > 0:
            move_type = "row_transition"
        elif delta_x == 0 and delta_y == 0:
            move_type = "repeat"
        else:
            move_type = "other"

        rows.append({
            "pair_type": "time_adjacent",
            "move_type": move_type,
            "session": int(a[session]),
            "order_a": int(a["acquisition_order"]),
            "order_b": int(b["acquisition_order"]),
            "order_gap": order_gap,
            "file_a": a["file"],
            "file_b": b["file"],
            "X_a": int(a["X"]),
            "Y_a": int(a["Y"]),
            "R_a": int(a["R"]),
            "X_b": int(b["X"]),
            "Y_b": int(b["Y"]),
            "R_b": int(b["R"]),
            "delta_X_um": delta_x,
            "delta_Y_um": delta_y,
            "delta_um": float(np.hypot(delta_x, delta_y)),
            "delta_T_mean_c": float(b["T_mean"] - a["T_mean"]) if "T_mean" in a else np.nan,
        })

    return pd.DataFrame(rows)


def build_repeat_pairs(
    df_audit: pd.DataFrame,
    repeat_coords: list[tuple[int, int]] | None = None,
    session_col: str | None = None,
) -> pd.DataFrame:
    """Build all same-coordinate repeat pairs."""
    session = _session_column(df_audit, session_col)
    coord_filter = set(repeat_coords) if repeat_coords else None
    rows = []

    for (x, y), group in df_audit.groupby(["X", "Y"], sort=True):
        if coord_filter is not None and (int(x), int(y)) not in coord_filter:
            continue
        group = group.sort_values("R")
        if len(group) < 2:
            continue
        for a, b in combinations(group.to_dict("records"), 2):
            rows.append({
                "scan_axis": "repeat",
                "session_a": int(a[session]),
                "session_b": int(b[session]),
                "file_a": a["file"],
                "file_b": b["file"],
                "X_a": int(a["X"]),
                "Y_a": int(a["Y"]),
                "X_b": int(b["X"]),
                "Y_b": int(b["Y"]),
                "R_a": int(a["R"]),
                "R_b": int(b["R"]),
                "delta_X_um": 0,
                "delta_Y_um": 0,
                "delta_um": 0.0,
            })

    return pd.DataFrame(rows)


def measure_frame_pairs(
    pairs: pd.DataFrame,
    data_dir: Path,
    *,
    roi_size: int = 320,
    search_radius: int = 5,
    method: str = "ncc",
    preprocess: str = "raw",
    highpass_sigma: float = 12.0,
) -> pd.DataFrame:
    """Measure displacements for a frame-pair table."""
    if method not in {"ncc", "phase"}:
        raise ValueError("method must be 'ncc' or 'phase'")

    estimator = subpixel_ncc if method == "ncc" else phase_correlation
    cache: dict[str, np.ndarray] = {}

    def read(name: str) -> np.ndarray:
        if name not in cache:
            cache[name] = load_frame(data_dir / name).astype(np.float32, copy=False)
        return cache[name]

    rows = []
    for _, pair in pairs.iterrows():
        result = estimator(
            read(pair["file_a"]),
            read(pair["file_b"]),
            roi_size=roi_size,
            search_radius=search_radius,
            preprocess=preprocess,
            highpass_sigma=highpass_sigma,
        ) if method == "ncc" else estimator(
            read(pair["file_a"]),
            read(pair["file_b"]),
            roi_size=roi_size,
            preprocess=preprocess,
            highpass_sigma=highpass_sigma,
        )
        rows.append({**pair.to_dict(), "method": method, "preprocess": preprocess, **result})

    return pd.DataFrame(rows)


def fit_rotation_angle(
    measured_dx: np.ndarray,
    measured_dy: np.ndarray,
    delta_x_um: np.ndarray,
    delta_y_um: np.ndarray,
    pixel_size_um: float = PIXEL_SIZE_UM,
) -> dict[str, np.ndarray | float | int]:
    """Fit the stage-to-image rotation angle from measured displacements."""
    x = np.asarray(delta_x_um, dtype=float)
    y = np.asarray(delta_y_um, dtype=float)
    dx = np.asarray(measured_dx, dtype=float)
    dy = np.asarray(measured_dy, dtype=float)

    rhs_x = pixel_size_um * dx
    rhs_y = pixel_size_um * dy
    c_num = np.sum(x * rhs_x + y * rhs_y)
    s_num = np.sum(y * rhs_x - x * rhs_y)
    norm = float(np.hypot(c_num, s_num))
    if norm == 0.0:
        raise ValueError("Cannot fit rotation angle from zero displacement data")

    theta_rad = float(np.arctan2(s_num, c_num))
    theta_deg = float(np.degrees(theta_rad))
    pred_dx, pred_dy = coordinate_to_shift(x, y, theta_deg, pixel_size_um)
    residual_dx = dx - pred_dx
    residual_dy = dy - pred_dy
    residual_norm = np.hypot(residual_dx, residual_dy)
    scale = norm / float(np.sum(x ** 2 + y ** 2))

    return {
        "theta_rad": theta_rad,
        "theta_deg": theta_deg,
        "scale_um_per_um": scale,
        "effective_pixel_size_um": pixel_size_um / scale,
        "residual_dx_px": residual_dx,
        "residual_dy_px": residual_dy,
        "residual_norm_px": residual_norm,
        "rms_error_px": float(np.sqrt(np.mean(residual_norm ** 2))),
        "median_error_px": float(np.median(residual_norm)),
        "n_pairs": int(dx.size),
    }


def bootstrap_theta_ci(
    pairs_data: pd.DataFrame,
    *,
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    pixel_size_um: float = PIXEL_SIZE_UM,
    seed: int = 0,
) -> dict[str, np.ndarray | float]:
    """Estimate a bootstrap confidence interval for theta."""
    rng = np.random.default_rng(seed)
    n = len(pairs_data)
    theta = np.empty(n_bootstrap, dtype=float)
    values = pairs_data[["dx_px", "dy_px", "delta_X_um", "delta_Y_um"]].to_numpy(float)

    for i in range(n_bootstrap):
        sample = values[rng.integers(0, n, n)]
        fit = fit_rotation_angle(
            sample[:, 0],
            sample[:, 1],
            sample[:, 2],
            sample[:, 3],
            pixel_size_um,
        )
        theta[i] = float(fit["theta_deg"])

    alpha = (1.0 - ci) / 2.0
    return {
        "theta_samples": theta,
        "ci_lower": float(np.quantile(theta, alpha)),
        "ci_upper": float(np.quantile(theta, 1.0 - alpha)),
        "theta_mean": float(np.mean(theta)),
        "theta_std": float(np.std(theta, ddof=1)),
    }


def linearity_regression(nominal: np.ndarray, measured: np.ndarray) -> dict[str, float]:
    """Fit measured = slope * nominal + intercept."""
    x = np.asarray(nominal, dtype=float)
    y = np.asarray(measured, dtype=float)
    slope, intercept = np.polyfit(x, y, 1)
    pred = slope * x + intercept
    residual = y - pred
    ss_res = float(np.sum(residual ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0.0 else float("nan"),
        "residual_std_px": float(np.std(residual, ddof=1)) if y.size > 1 else 0.0,
    }
