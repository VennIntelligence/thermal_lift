"""EP03 physical limits and local observability helpers.

This module intentionally keeps EP03 focused on physical design bounds for the
2x contour-level SR proof of concept. Stage commands appear only as priors for
sampling geometry and alignment diagnostics; they are not treated as alignment
ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter, map_coordinates
from scipy.special import erf

from thermal_core.displacement import coordinate_to_shift
from thermal_core.plotting import (
    COLORMAPS,
    METHOD_COLOR_LIST,
    format_colorbar,
    make_figure,
)


SQRT2 = np.sqrt(2.0)
SQRT_2PI = np.sqrt(2.0 * np.pi)


@dataclass(frozen=True)
class PhysicalScales:
    """Project-level EP03 physical scales."""

    detector_pitch_um: float = 20.0
    spatial_resolution_um: float = 20.0
    target_grid_um: float = 5.0
    detector_rows: int = 480
    detector_cols: int = 640
    psf_sigmas_px: tuple[float, ...] = (0.2, 0.35, 0.5)
    noise_floor_c: float = 0.0724


def normal_pdf(z: np.ndarray) -> np.ndarray:
    """Standard normal probability density function."""
    z = np.asarray(z, dtype=float)
    return np.exp(-0.5 * z * z) / SQRT_2PI


def normal_cdf(z: np.ndarray) -> np.ndarray:
    """Standard normal cumulative distribution function."""
    z = np.asarray(z, dtype=float)
    return 0.5 * (1.0 + erf(z / SQRT2))


def gaussian_mtf(frequency_cyc_per_px: np.ndarray | float, sigma_px: float) -> np.ndarray:
    """Gaussian PSF modulation transfer function.

    Frequencies are expressed in cycles per detector pixel. The returned MTF is
    an amplitude transfer ratio.
    """
    f = np.asarray(frequency_cyc_per_px, dtype=float)
    return np.exp(-2.0 * (np.pi ** 2) * (float(sigma_px) ** 2) * f * f)


def build_sampling_resolution_table(
    *,
    detector_pitch_um: float = 20.0,
    spatial_resolution_um: float = 20.0,
    target_grid_um: float = 5.0,
) -> pd.DataFrame:
    """Return the detector pitch, spatial resolution, and target-grid distinction."""
    return pd.DataFrame(
        [
            {
                "quantity": "Detector sampling pitch",
                "value_um": detector_pitch_um,
                "detector_pixels": 1.0,
                "role": "TXT matrix sample spacing; 640 x 480 covers 12.8 x 9.6 mm",
                "sr_design_use": "Defines the LR pixel grid and stage prior unit conversion",
            },
            {
                "quantity": "Current spatial resolution",
                "value_um": spatial_resolution_um,
                "detector_pixels": spatial_resolution_um / detector_pitch_um,
                "role": "Calibrated optical/thermal resolving scale, not pixel pitch",
                "sr_design_use": "Sets the high-frequency information boundary",
            },
            {
                "quantity": "2x SR grid sample",
                "value_um": detector_pitch_um / 2.0,
                "detector_pixels": 0.5,
                "role": "Minimum sampling grid that can represent a 5 um target pitch",
                "sr_design_use": "Default contour-level POC grid; not a 5 um metrology claim",
            },
            {
                "quantity": "4x SR grid sample",
                "value_um": detector_pitch_um / 4.0,
                "detector_pixels": 0.25,
                "role": "Higher-density display/analysis grid",
                "sr_design_use": "High-risk exploratory setting unless MTF/SNR evidence supports it",
            },
        ]
    )


def build_output_grid_nyquist_table(
    *,
    detector_pitch_um: float = 20.0,
    spatial_resolution_um: float = 20.0,
    grid_factors: tuple[int, ...] = (1, 2, 4),
) -> pd.DataFrame:
    """Return output-grid sample spacing and the implied Nyquist period.

    The Nyquist period is twice the sample spacing of an output grid. It is a
    sampling property, not a calibrated optical-resolution claim.
    """
    rows: list[dict] = []
    for factor in grid_factors:
        grid_pitch_um = float(detector_pitch_um) / float(factor)
        nyquist_period_um = 2.0 * grid_pitch_um
        rows.append(
            {
                "grid_label": f"{int(factor)}x",
                "output_sample_um": grid_pitch_um,
                "output_sample_detector_px": 1.0 / float(factor),
                "nyquist_period_um": nyquist_period_um,
                "nyquist_period_detector_px": nyquist_period_um / float(detector_pitch_um),
                "nyquist_cyc_per_detector_px": 0.5 * float(factor),
                "current_spatial_resolution_um": float(spatial_resolution_um),
                "resolution_to_nyquist_period": float(spatial_resolution_um) / nyquist_period_um,
                "interpretation": (
                    "LR detector grid"
                    if factor == 1
                    else "default 2x contour-level POC grid"
                    if factor == 2
                    else "exploratory display/ablation grid, not a default claim"
                ),
            }
        )
    return pd.DataFrame(rows)


def build_mtf_attenuation_table(
    *,
    detector_pitch_um: float = 20.0,
    sigmas_px: tuple[float, ...] = (0.2, 0.35, 0.5),
    grid_factors: tuple[int, ...] = (1, 2, 4),
) -> pd.DataFrame:
    """Return MTF attenuation at 1x/2x/4x grid Nyquist frequencies."""
    rows: list[dict] = []
    for factor in grid_factors:
        frequency = 0.5 * float(factor)
        for sigma in sigmas_px:
            mtf = float(gaussian_mtf(frequency, sigma))
            rows.append(
                {
                    "grid_factor": int(factor),
                    "grid_label": f"{factor}x",
                    "grid_pitch_um": detector_pitch_um / float(factor),
                    "nyquist_cyc_per_detector_px": frequency,
                    "sigma_psf_px": float(sigma),
                    "mtf_amplitude": mtf,
                    "attenuation_db": 20.0 * np.log10(max(mtf, 1e-12)),
                }
            )
    return pd.DataFrame(rows)


def build_mtf_snr_recoverability_table(
    contrast_table: pd.DataFrame,
    *,
    noise_sigma_c: float,
    sigmas_px: tuple[float, ...] = (0.2, 0.35, 0.5),
    grid_factors: tuple[int, ...] = (1, 2, 4),
    detector_pitch_um: float = 20.0,
) -> pd.DataFrame:
    """Combine local contrast, Gaussian MTF, and noise into effective SNR.

    ``effective_snr = DeltaT * MTF(f, sigma) / noise`` is a necessary-condition
    risk indicator. It does not prove SR success, because it ignores alignment
    errors, model mismatch, thermal drift, and structural consistency.
    """
    if contrast_table.empty:
        raise ValueError("contrast_table must contain at least one contrast level")
    required = {"label", "delta_t_c"}
    missing = required - set(contrast_table.columns)
    if missing:
        raise ValueError(f"contrast_table missing columns: {sorted(missing)}")

    rows: list[dict] = []
    for _, contrast in contrast_table.iterrows():
        delta_t_c = float(contrast["delta_t_c"])
        source = str(contrast.get("source", "reference"))
        label = str(contrast["label"])
        for factor in grid_factors:
            frequency = 0.5 * float(factor)
            for sigma in sigmas_px:
                mtf = float(gaussian_mtf(frequency, float(sigma)))
                effective_delta_t = delta_t_c * mtf
                effective_snr = effective_delta_t / float(noise_sigma_c)
                if effective_snr >= 5.0:
                    risk_band = "observable"
                elif effective_snr >= 3.0:
                    risk_band = "borderline"
                elif effective_snr >= 1.0:
                    risk_band = "weak"
                else:
                    risk_band = "noise-dominated"
                rows.append(
                    {
                        "source": source,
                        "contrast_label": label,
                        "delta_t_c": delta_t_c,
                        "input_snr": delta_t_c / float(noise_sigma_c),
                        "grid_factor": int(factor),
                        "grid_label": f"{int(factor)}x",
                        "grid_pitch_um": float(detector_pitch_um) / float(factor),
                        "nyquist_cyc_per_detector_px": frequency,
                        "sigma_psf_px": float(sigma),
                        "mtf_amplitude": mtf,
                        "effective_delta_t_c": effective_delta_t,
                        "effective_snr": effective_snr,
                        "passes_3x_noise": bool(effective_snr >= 3.0),
                        "passes_5x_noise": bool(effective_snr >= 5.0),
                        "risk_band": risk_band,
                    }
                )
    return pd.DataFrame(rows)


def crb_single_frame(
    delta_t_c: float,
    sigma_psf_px: float,
    noise_sigma_c: float,
    *,
    edge_position_px: float = 10.3,
    n_pixels: int = 24,
) -> float:
    """Single-frame CRB for an ESF edge position in detector pixels."""
    u = np.arange(n_pixels, dtype=float)
    sigma = max(float(sigma_psf_px), 1e-6)
    deriv = (float(delta_t_c) / sigma) * normal_pdf((u - edge_position_px) / sigma)
    fisher = np.sum(deriv * deriv) / (float(noise_sigma_c) ** 2)
    return float(1.0 / np.sqrt(max(fisher, 1e-18)))


def crb_multi_frame(
    delta_t_c: float,
    sigma_psf_px: float,
    noise_sigma_c: float,
    shifts_px: np.ndarray,
    *,
    edge_position_px: float = 10.3,
    n_pixels: int = 24,
) -> float:
    """Known-shift multi-frame CRB for an ESF edge position."""
    u = np.arange(n_pixels, dtype=float)
    shifts = np.asarray(shifts_px, dtype=float)
    sigma = max(float(sigma_psf_px), 1e-6)
    deriv = (float(delta_t_c) / sigma) * normal_pdf(
        (u[None, :] - edge_position_px - shifts[:, None]) / sigma
    )
    fisher = np.sum(deriv * deriv) / (float(noise_sigma_c) ** 2)
    return float(1.0 / np.sqrt(max(fisher, 1e-18)))


def uniform_phase_deltas(n_frames: int, coverage_px: float = 1.0) -> np.ndarray:
    """Uniform subpixel phase coverage centered on zero."""
    n = int(n_frames)
    if n <= 1 or float(coverage_px) == 0.0:
        return np.zeros(max(n, 1), dtype=float)
    return np.linspace(-0.5 * float(coverage_px), 0.5 * float(coverage_px), n)


def build_crb_localization_table(
    noise_sigma_c: float,
    *,
    contrasts_c: tuple[float, ...] = (0.3, 0.7, 1.0, 2.0),
    sigma_values_px: tuple[float, ...] = (0.5, 1.0),
    n_frames: int = 16,
    phase_coverage_px: float = 1.0,
) -> pd.DataFrame:
    """Build a compact CRB table for local ESF anchor confidence."""
    rows: list[dict] = []
    shifts = uniform_phase_deltas(n_frames, phase_coverage_px)
    for sigma in sigma_values_px:
        for contrast in contrasts_c:
            single = crb_single_frame(contrast, sigma, noise_sigma_c)
            multi = crb_multi_frame(contrast, sigma, noise_sigma_c, shifts)
            rows.extend(
                [
                    {
                        "delta_t_c": float(contrast),
                        "sigma_psf_px": float(sigma),
                        "model": "single_frame",
                        "n_frames": 1,
                        "phase_coverage_px": 0.0,
                        "crb_px": single,
                    },
                    {
                        "delta_t_c": float(contrast),
                        "sigma_psf_px": float(sigma),
                        "model": f"{n_frames}_frame_known_shift",
                        "n_frames": int(n_frames),
                        "phase_coverage_px": float(phase_coverage_px),
                        "crb_px": multi,
                    },
                ]
            )
    return pd.DataFrame(rows)


def build_crb_sensitivity_table(
    noise_sigma_c: float,
    *,
    contrasts_c: tuple[float, ...] = (0.3, 0.7, 1.0, 2.0),
    sigma_values_px: tuple[float, ...] = (0.2, 0.35, 0.5, 1.0),
    n_frames_values: tuple[int, ...] = (1, 4, 16, 64, 255),
    phase_coverage_values_px: tuple[float, ...] = (0.0, 0.5, 1.0),
) -> pd.DataFrame:
    """Scan optimistic ESF CRB sensitivity over contrast, PSF, frames, phases."""
    rows: list[dict] = []
    for n_frames in n_frames_values:
        coverages = (0.0,) if int(n_frames) <= 1 else phase_coverage_values_px
        for phase_coverage in coverages:
            shifts = uniform_phase_deltas(int(n_frames), float(phase_coverage))
            for sigma in sigma_values_px:
                for contrast in contrasts_c:
                    crb_px = crb_multi_frame(
                        contrast,
                        sigma,
                        noise_sigma_c,
                        shifts,
                    )
                    if crb_px <= 0.05:
                        gate_band = "passes 0.05 px"
                    elif crb_px <= 0.10:
                        gate_band = "passes 0.10 px"
                    else:
                        gate_band = "above 0.10 px"
                    rows.append(
                        {
                            "delta_t_c": float(contrast),
                            "sigma_psf_px": float(sigma),
                            "n_frames": int(n_frames),
                            "phase_coverage_px": float(phase_coverage),
                            "crb_px": crb_px,
                            "passes_0p05_px": bool(crb_px <= 0.05),
                            "passes_0p10_px": bool(crb_px <= 0.10),
                            "gate_band": gate_band,
                        }
                    )
    return pd.DataFrame(rows)


def build_crb_gate_summary_table(
    sensitivity_table: pd.DataFrame,
    *,
    gates_px: tuple[float, ...] = (0.10, 0.05),
) -> pd.DataFrame:
    """Summarize the minimum scanned contrast needed to pass each CRB gate."""
    group_cols = ["sigma_psf_px", "n_frames", "phase_coverage_px"]
    rows: list[dict] = []
    for keys, group in sensitivity_table.groupby(group_cols, sort=True):
        row = {
            "sigma_psf_px": float(keys[0]),
            "n_frames": int(keys[1]),
            "phase_coverage_px": float(keys[2]),
        }
        for gate in gates_px:
            passing = group[group["crb_px"] <= float(gate)]
            row[f"min_delta_t_for_{gate:.2f}px_gate_c"] = (
                float(passing["delta_t_c"].min()) if not passing.empty else np.nan
            )
        row["best_crb_px_in_scan"] = float(group["crb_px"].min())
        row["worst_crb_px_in_scan"] = float(group["crb_px"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def build_snr_reference_table(
    noise_sigma_c: float,
    measured_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Reference and measured contrast scales expressed as SNR."""
    rows = [
        {"source": "reference", "label": "Noise floor", "delta_t_c": noise_sigma_c},
        {"source": "reference", "label": "3x noise gate", "delta_t_c": 3.0 * noise_sigma_c},
        {"source": "reference", "label": "Weak local contrast", "delta_t_c": 0.3},
        {"source": "reference", "label": "Nominal edge contrast", "delta_t_c": 0.7},
        {"source": "reference", "label": "Strong local contrast", "delta_t_c": 1.0},
    ]
    if measured_summary is not None and not measured_summary.empty:
        for _, row in measured_summary.iterrows():
            rows.append(
                {
                    "source": "measured",
                    "label": f"{row['source']} median edge",
                    "delta_t_c": float(row["median_abs_delta_t_c"]),
                }
            )
    out = pd.DataFrame(rows)
    out["snr"] = out["delta_t_c"] / float(noise_sigma_c)
    return out


def select_main_scan(audit_df: pd.DataFrame) -> pd.DataFrame:
    """Return the main session sorted by acquisition order."""
    if "session" in audit_df.columns:
        main = audit_df[audit_df["session"].eq(2)].copy()
    elif "is_main_session" in audit_df.columns:
        main = audit_df[audit_df["is_main_session"].astype(bool)].copy()
    else:
        raise ValueError("frame audit must include session or is_main_session")
    if main.empty:
        raise ValueError("No main session frames found")
    return main.sort_values("acquisition_order").reset_index(drop=True)


def select_reference_frame_row(main_df: pd.DataFrame) -> pd.Series:
    """Use the middle acquisition-order frame as a representative reference."""
    return main_df.iloc[len(main_df) // 2]


def load_frame_by_row(data_dir: Path, row: pd.Series, load_frame_fn) -> np.ndarray:
    """Load a temperature frame from an audit-table row."""
    return load_frame_fn(Path(data_dir) / str(row["file"]))


def _normalize_uint8(frame: np.ndarray) -> np.ndarray:
    data = np.asarray(frame, dtype=float)
    finite = np.isfinite(data)
    if not finite.any():
        raise ValueError("frame contains no finite values")
    lo = float(np.nanpercentile(data, 1.0))
    hi = float(np.nanpercentile(data, 99.0))
    if hi <= lo:
        lo = float(np.nanmin(data))
        hi = float(np.nanmax(data))
    scaled = np.clip((data - lo) / max(hi - lo, 1e-9), 0.0, 1.0)
    return np.round(255.0 * scaled).astype(np.uint8)


def detect_outer_contour(
    frame: np.ndarray,
    *,
    blur_sigma: float = 1.0,
    kernel_size: int = 7,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Segment the main device mask and return its largest external contour."""
    smooth = gaussian_filter(np.asarray(frame, dtype=float), sigma=blur_sigma)
    norm = _normalize_uint8(smooth)
    threshold, mask = cv2.threshold(norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((int(kernel_size), int(kernel_size)), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise RuntimeError("No external contour found")
    contour = max(contours, key=cv2.contourArea)[:, 0, :].astype(float)
    return mask.astype(bool), contour, float(threshold)


def detect_inner_contours(
    frame: np.ndarray,
    outer_mask: np.ndarray,
    *,
    blur_sigma: float = 1.0,
    kernel_size: int = 5,
    min_area_px: float = 100.0,
) -> list[np.ndarray]:
    """Detect internal hot/cold contours within the main device mask."""
    smooth = gaussian_filter(np.asarray(frame, dtype=float), sigma=blur_sigma)
    norm = _normalize_uint8(smooth)
    kernel = np.ones((int(kernel_size), int(kernel_size)), dtype=np.uint8)
    records: list[dict] = []
    for flag in (cv2.THRESH_BINARY, cv2.THRESH_BINARY_INV):
        _, mask = cv2.threshold(norm, 0, 255, flag + cv2.THRESH_OTSU)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        if not contours:
            continue
        largest = max(range(len(contours)), key=lambda i: cv2.contourArea(contours[i]))
        for idx, contour in enumerate(contours):
            if idx == largest:
                continue
            area = float(cv2.contourArea(contour))
            if area < float(min_area_px):
                continue
            moments = cv2.moments(contour)
            if moments["m00"] == 0.0:
                continue
            cx = float(moments["m10"] / moments["m00"])
            cy = float(moments["m01"] / moments["m00"])
            if not _point_inside(outer_mask, cx, cy):
                continue
            records.append({"contour": contour[:, 0, :].astype(float), "area": area, "cx": cx, "cy": cy})

    kept: list[dict] = []
    for record in sorted(records, key=lambda r: r["area"], reverse=True):
        center = np.array([record["cx"], record["cy"]], dtype=float)
        duplicate = False
        for existing in kept:
            existing_center = np.array([existing["cx"], existing["cy"]], dtype=float)
            area_scale = max(record["area"], existing["area"], 1.0)
            area_gap = abs(record["area"] - existing["area"]) / area_scale
            if np.linalg.norm(center - existing_center) <= 5.0 and area_gap <= 0.35:
                duplicate = True
                break
        if not duplicate:
            kept.append(record)
    return [record["contour"] for record in kept]


def _point_inside(mask: np.ndarray, x: float, y: float) -> bool:
    row = int(np.clip(round(y), 0, mask.shape[0] - 1))
    col = int(np.clip(round(x), 0, mask.shape[1] - 1))
    return bool(mask[row, col])


def _sample_image(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    coords = np.vstack([np.asarray(y, dtype=float), np.asarray(x, dtype=float)])
    return map_coordinates(np.asarray(image, dtype=float), coords, order=1, mode="nearest")


def _angle_deg(vec: np.ndarray) -> float:
    return float(np.degrees(np.arctan2(vec[1], vec[0])) % 180.0)


def _segment_one_contour(
    frame: np.ndarray,
    contour: np.ndarray,
    *,
    source: str,
    theta_deg: float,
    noise_sigma_c: float,
    outer_mask: np.ndarray | None = None,
    segment_id_offset: int = 0,
    stride_px: int = 8,
    window_px: int = 12,
) -> list[dict]:
    points = np.asarray(contour, dtype=float)
    if len(points) < max(16, 2 * window_px + 1):
        return []

    dx_scan, dy_scan = coordinate_to_shift(1.0, 0.0, theta_deg=theta_deg)
    scan_vec = np.array([float(dx_scan), float(dy_scan)], dtype=float)
    scan_vec /= max(np.linalg.norm(scan_vec), 1e-12)

    offsets = np.arange(-window_px, window_px + 1)
    sample_offsets = np.linspace(3.0, 8.0, 6)
    rows: list[dict] = []
    for local_id, idx in enumerate(range(0, len(points), stride_px)):
        local = points[(idx + offsets) % len(points)]
        center = points[idx]
        centered = local - local.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        tangent = vt[0]
        tangent /= max(np.linalg.norm(tangent), 1e-12)
        normal = np.array([-tangent[1], tangent[0]], dtype=float)
        normal /= max(np.linalg.norm(normal), 1e-12)

        plus = _sample_image(frame, center[0] + sample_offsets * normal[0], center[1] + sample_offsets * normal[1])
        minus = _sample_image(frame, center[0] - sample_offsets * normal[0], center[1] - sample_offsets * normal[1])

        if source == "outer" and outer_mask is not None:
            plus_inside = _point_inside(outer_mask, center[0] + 4.0 * normal[0], center[1] + 4.0 * normal[1])
            minus_inside = _point_inside(outer_mask, center[0] - 4.0 * normal[0], center[1] - 4.0 * normal[1])
            if minus_inside and not plus_inside:
                normal = -normal
                plus, minus = minus, plus
        elif float(np.nanmedian(plus)) < float(np.nanmedian(minus)):
            normal = -normal
            plus, minus = minus, plus

        delta_t = float(np.nanmedian(plus) - np.nanmedian(minus))
        normal_dist = centered @ normal
        tangent_dist = centered @ tangent
        span = max(float(np.ptp(tangent_dist)), 1e-6)
        curvature_proxy = float(np.sqrt(np.mean(normal_dist * normal_dist)) / span)

        rows.append(
            {
                "source": source,
                "segment_id": int(segment_id_offset + local_id),
                "contour_index": int(idx),
                "x_px": float(center[0]),
                "y_px": float(center[1]),
                "tx": float(tangent[0]),
                "ty": float(tangent[1]),
                "nx": float(normal[0]),
                "ny": float(normal[1]),
                "normal_angle_deg": _angle_deg(normal),
                "normal_projection": float(abs(np.dot(normal, scan_vec))),
                "curvature_proxy": curvature_proxy,
                "delta_t_c": delta_t,
                "abs_delta_t_c": abs(delta_t),
                "snr": abs(delta_t) / float(noise_sigma_c),
            }
        )
    return rows


def measure_contour_observability(
    frame: np.ndarray,
    *,
    theta_deg: float,
    noise_sigma_c: float,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, list[np.ndarray]]:
    """Measure local contour contrast, SNR, and normal projection."""
    outer_mask, outer_contour, _ = detect_outer_contour(frame)
    rows = _segment_one_contour(
        frame,
        outer_contour,
        source="outer",
        theta_deg=theta_deg,
        noise_sigma_c=noise_sigma_c,
        outer_mask=outer_mask,
        segment_id_offset=0,
        stride_px=8,
    )

    inner_contours = detect_inner_contours(frame, outer_mask)
    offset = 10000
    for contour in inner_contours:
        rows.extend(
            _segment_one_contour(
                frame,
                contour,
                source="inner",
                theta_deg=theta_deg,
                noise_sigma_c=noise_sigma_c,
                outer_mask=None,
                segment_id_offset=offset,
                stride_px=6,
                window_px=10,
            )
        )
        offset += 1000

    segments = pd.DataFrame(rows)
    if segments.empty:
        return segments, pd.DataFrame(), outer_mask, outer_contour, inner_contours

    segments["curvature_gate"] = False
    for source, idx in segments.groupby("source").groups.items():
        threshold = float(segments.loc[idx, "curvature_proxy"].quantile(0.70))
        segments.loc[idx, "curvature_gate"] = segments.loc[idx, "curvature_proxy"] <= threshold
    segments["anchor_candidate"] = (
        (segments["snr"] >= 7.0)
        & (segments["normal_projection"] >= 0.5)
        & segments["curvature_gate"].astype(bool)
    )

    summary = (
        segments.groupby("source")
        .agg(
            n_segments=("segment_id", "count"),
            median_abs_delta_t_c=("abs_delta_t_c", "median"),
            p10_abs_delta_t_c=("abs_delta_t_c", lambda x: float(np.quantile(x, 0.10))),
            p90_abs_delta_t_c=("abs_delta_t_c", lambda x: float(np.quantile(x, 0.90))),
            median_snr=("snr", "median"),
            snr_gt_5_fraction=("snr", lambda x: float(np.mean(np.asarray(x) > 5.0))),
            anchor_candidate_fraction=("anchor_candidate", "mean"),
            median_normal_projection=("normal_projection", "median"),
        )
        .reset_index()
    )
    return segments, summary, outer_mask, outer_contour, inner_contours


def plot_sampling_resolution_diagram(
    table: pd.DataFrame,
    *,
    detector_pitch_um: float = 20.0,
    spatial_resolution_um: float = 20.0,
    target_grid_um: float = 5.0,
    annotation_fontsize: float = 9.0,
) -> plt.Figure:
    """Visualize detector pitch, current resolution, and the 2x target grid."""
    fig, ax = make_figure("double_col", height=3.2)
    ax.set_xlim(-1.0, 41.0)
    ax.set_ylim(-0.2, 4.1)
    ax.set_xlabel("Physical distance along one image axis [um]")
    ax.set_yticks([3.2, 2.2, 1.2, 0.2])
    ax.set_yticklabels(["Detector samples", "Spatial resolution", "2x SR grid", "4x grid"])

    for x in np.arange(0.0, 40.1, detector_pitch_um):
        ax.plot([x, x], [2.9, 3.5], color=METHOD_COLOR_LIST[0], linewidth=1.2)
    ax.plot([0.0, 40.0], [3.2, 3.2], color=METHOD_COLOR_LIST[0], linewidth=0.8)
    ax.text(
        10.0,
        3.55,
        f"{detector_pitch_um:g} um/pixel",
        ha="center",
        va="bottom",
        fontsize=annotation_fontsize,
        color=METHOD_COLOR_LIST[0],
    )

    rect = patches.Rectangle(
        (0.0, 1.95),
        spatial_resolution_um,
        0.5,
        facecolor=METHOD_COLOR_LIST[2],
        alpha=0.25,
        edgecolor=METHOD_COLOR_LIST[2],
    )
    ax.add_patch(rect)
    ax.text(
        spatial_resolution_um / 2.0,
        2.52,
        f"{spatial_resolution_um:g} um calibrated resolution",
        ha="center",
        va="bottom",
        fontsize=annotation_fontsize,
        color=METHOD_COLOR_LIST[2],
    )

    for x in np.arange(0.0, 40.1, target_grid_um):
        ax.plot([x, x], [0.95, 1.45], color=METHOD_COLOR_LIST[1], linewidth=0.9)
    ax.text(
        5.0,
        1.55,
        f"{target_grid_um:g} um/sample = 2x grid",
        ha="center",
        va="bottom",
        fontsize=annotation_fontsize,
        color=METHOD_COLOR_LIST[1],
    )

    for x in np.arange(0.0, 40.1, detector_pitch_um / 4.0):
        ax.plot([x, x], [-0.05, 0.45], color=METHOD_COLOR_LIST[3], linewidth=0.55, alpha=0.75)
    ax.text(
        5.0,
        0.55,
        f"{detector_pitch_um / 4.0:g} um/sample = 4x grid",
        ha="center",
        va="bottom",
        fontsize=annotation_fontsize,
        color=METHOD_COLOR_LIST[3],
    )

    ax.grid(axis="x", alpha=0.15)
    ax.spines["left"].set_visible(False)
    return fig


def plot_mtf_psf_curves(
    table: pd.DataFrame,
    *,
    sigmas_px: tuple[float, ...] = (0.2, 0.35, 0.5),
) -> plt.Figure:
    """Plot Gaussian PSF MTF curves and 1x/2x/4x frequency markers."""
    fig, ax = make_figure("double_col", height=3.6)
    frequencies = np.linspace(0.0, 2.05, 500)
    for i, sigma in enumerate(sigmas_px):
        color = METHOD_COLOR_LIST[i % len(METHOD_COLOR_LIST)]
        ax.plot(frequencies, gaussian_mtf(frequencies, sigma), color=color, label=f"sigma={sigma:.2f} px")
        subset = table[table["sigma_psf_px"].eq(float(sigma))]
        ax.scatter(
            subset["nyquist_cyc_per_detector_px"],
            subset["mtf_amplitude"],
            color=color,
            s=22,
            zorder=3,
        )

    for factor, label in [(1, "1x Nyquist"), (2, "2x Nyquist"), (4, "4x Nyquist")]:
        x = 0.5 * factor
        ax.axvline(x, color="#666666", linestyle="--", linewidth=0.8)
        ax.text(
            x - 0.02,
            1.8e-6,
            label,
            rotation=0,
            fontsize=8,
            ha="right",
            va="bottom",
            color="#444444",
        )

    ax.set_yscale("log")
    ax.set_ylim(1e-6, 1.05)
    ax.set_xlim(0.0, 2.05)
    ax.set_xlabel("Spatial frequency [cycles / detector pixel]")
    ax.set_ylabel("MTF amplitude")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper left", fontsize=8)
    return fig


def plot_mtf_snr_recoverability_heatmap(
    recoverability_table: pd.DataFrame,
    *,
    min_log10_snr: float = -2.0,
) -> plt.Figure:
    """Plot effective SNR at grid Nyquist frequencies as a risk heatmap."""
    table = recoverability_table.copy()
    table["column_label"] = table.apply(
        lambda row: f"{row['grid_label']}\nsigma={row['sigma_psf_px']:.2f}",
        axis=1,
    )
    contrast_order = (
        table[["source", "contrast_label", "delta_t_c", "input_snr"]]
        .drop_duplicates()
        .sort_values(["source", "delta_t_c", "contrast_label"])
    )
    contrast_order["row_label"] = contrast_order.apply(
        lambda row: f"{row['contrast_label']}\n{row['delta_t_c']:.3g} C, SNR {row['input_snr']:.1f}",
        axis=1,
    )
    column_order = (
        table[["grid_factor", "sigma_psf_px", "column_label"]]
        .drop_duplicates()
        .sort_values(["grid_factor", "sigma_psf_px"])
    )

    table = table.merge(
        contrast_order[["source", "contrast_label", "delta_t_c", "row_label"]],
        on=["source", "contrast_label", "delta_t_c"],
        how="left",
    )
    pivot = table.pivot_table(
        index="row_label",
        columns="column_label",
        values="effective_snr",
        aggfunc="first",
    ).reindex(
        index=contrast_order["row_label"].to_list(),
        columns=column_order["column_label"].to_list(),
    )
    values = pivot.to_numpy(dtype=float)
    log_values = np.log10(np.maximum(values, 10.0 ** min_log10_snr))

    fig, ax = make_figure("double_col", height=max(3.6, 0.38 * len(pivot.index) + 1.4))
    im = ax.imshow(log_values, cmap=COLORMAPS["coverage"], aspect="auto", vmin=min_log10_snr, vmax=2.0)
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(pivot.columns, rotation=0, ha="center")
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("Output grid and assumed Gaussian PSF")
    ax.set_ylabel("Local contrast scale")

    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            value = values[row, col]
            if value >= 100.0:
                label = f"{value:.0f}"
            elif value >= 10.0:
                label = f"{value:.1f}"
            elif value >= 1.0:
                label = f"{value:.2f}"
            else:
                label = f"{value:.2g}"
            color = "white" if log_values[row, col] < 0.35 else "black"
            ax.text(col, row, label, ha="center", va="center", fontsize=7, color=color)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    format_colorbar(cbar, "log10 effective SNR")
    return fig


def plot_noise_floor_snr(
    snr_table: pd.DataFrame,
    segments: pd.DataFrame,
    *,
    noise_sigma_c: float,
) -> plt.Figure:
    """Plot noise-floor SNR curve and measured contour contrast distribution."""
    fig, axes = make_figure("double_col", nrows=1, ncols=2, height=3.4)
    ax0, ax1 = axes
    x = np.linspace(0.0, max(3.0, float(snr_table["delta_t_c"].max()) * 1.1), 300)
    ax0.plot(x, x / float(noise_sigma_c), color=METHOD_COLOR_LIST[0], linewidth=1.4)
    for _, row in snr_table.iterrows():
        color = METHOD_COLOR_LIST[1] if row["source"] == "measured" else METHOD_COLOR_LIST[2]
        label = str(row["label"])
        ax0.scatter(row["delta_t_c"], row["snr"], s=28, color=color, zorder=3)
        ax0.annotate(
            label,
            xy=(row["delta_t_c"], row["snr"]),
            xytext=(8, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            fontsize=8,
        )
    ax0.axhline(1.0, color="#666666", linestyle=":", linewidth=0.9)
    ax0.axhline(5.0, color="#666666", linestyle="--", linewidth=0.9)
    ax0.text(
        0.98,
        1.0,
        "1x noise",
        transform=ax0.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
    )
    ax0.text(
        0.98,
        5.0,
        "5x noise",
        transform=ax0.get_yaxis_transform(),
        ha="right",
        va="bottom",
        fontsize=8,
        color="#666666",
    )
    ax0.set_xlabel("Local temperature contrast [C]")
    ax0.set_ylabel("SNR = contrast / 0.0724 C")
    ax0.set_title("(a) Noise Floor vs Local Contrast")
    ax0.grid(axis="y", alpha=0.25)

    if segments.empty:
        ax1.text(0.5, 0.5, "No measured segments", transform=ax1.transAxes, ha="center", va="center")
    else:
        bins = np.linspace(0.0, max(3.0, float(segments["abs_delta_t_c"].quantile(0.98)) * 1.05), 32)
        for source, group in segments.groupby("source"):
            color = METHOD_COLOR_LIST[0] if source == "inner" else METHOD_COLOR_LIST[2]
            # Fill with low alpha
            ax1.hist(
                group["abs_delta_t_c"],
                bins=bins,
                histtype="stepfilled",
                alpha=0.25,
                color=color,
            )
            # Outline with high alpha for contrast
            ax1.hist(
                group["abs_delta_t_c"],
                bins=bins,
                histtype="step",
                alpha=0.9,
                color=color,
                linewidth=1.2,
            )
    for multiple, label in [(1, "1x noise"), (3, "3x noise"), (5, "5x noise")]:
        x_noise = multiple * noise_sigma_c
        ax1.axvline(x_noise, color="#666666", linestyle="--", linewidth=0.8)
        ax1.annotate(
            label,
            xy=(x_noise, ax1.get_ylim()[1] * 0.85),
            xytext=(3, 0),
            textcoords="offset points",
            ha="left",
            va="center",
            rotation=90,
            rotation_mode="anchor",
            fontsize=8,
        )
    ax1.set_xlabel("Measured local |Delta T| [C]")
    ax1.set_ylabel("Segment count")
    ax1.set_title("(b) Measured Contour Contrast Scales")

    # Custom legend with higher contrast
    handles = [
        patches.Patch(facecolor=METHOD_COLOR_LIST[0], edgecolor=METHOD_COLOR_LIST[0], alpha=0.9, label="inner segments"),
        patches.Patch(facecolor=METHOD_COLOR_LIST[2], edgecolor=METHOD_COLOR_LIST[2], alpha=0.9, label="outer segments"),
    ]
    ax1.legend(handles=handles, loc="upper right", fontsize=8)
    return fig


def plot_local_contour_candidate_map(
    frame: np.ndarray,
    outer_contour: np.ndarray,
    inner_contours: list[np.ndarray],
    segments: pd.DataFrame,
) -> plt.Figure:
    """Plot contour locations and local anchor candidates on the temperature frame."""
    fig, ax = make_figure("single_col", height=3.25)
    im = ax.imshow(frame, cmap=COLORMAPS["temperature"], origin="upper")
    ax.plot(outer_contour[:, 0], outer_contour[:, 1], color=METHOD_COLOR_LIST[2], linewidth=0.8, label="outer contour")
    for i, contour in enumerate(inner_contours[:40]):
        label = "inner contours" if i == 0 else None
        ax.plot(contour[:, 0], contour[:, 1], color=METHOD_COLOR_LIST[0], linewidth=0.45, alpha=0.8, label=label)
    if not segments.empty:
        good = segments[segments["anchor_candidate"]]
        ax.scatter(good["x_px"], good["y_px"], s=10, color=METHOD_COLOR_LIST[1], label="anchor candidates")

    # 2X Zoom ROI Crop: Center around outer contour with 30px padding
    x_min, x_max = np.min(outer_contour[:, 0]), np.max(outer_contour[:, 0])
    y_min, y_max = np.min(outer_contour[:, 1]), np.max(outer_contour[:, 1])
    pad = 30
    ax.set_xlim(max(0, x_min - pad), min(frame.shape[1], x_max + pad))
    ax.set_ylim(min(frame.shape[0], y_max + pad), max(0, y_min - pad))

    ax.set_axis_off()
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    format_colorbar(cbar, "Temperature [C]")
    ax.legend(loc="lower right", fontsize=7)
    return fig


def plot_local_anchor_confidence(segments: pd.DataFrame) -> plt.Figure:
    """Plot segment-level SNR versus X micro-scan normal projection."""
    fig, ax = make_figure("single_col", height=3.0)
    if not segments.empty:
        for source, color in [("outer", METHOD_COLOR_LIST[2]), ("inner", METHOD_COLOR_LIST[0])]:
            group = segments[segments["source"].eq(source)]
            if group.empty:
                continue
            sizes = np.where(group["anchor_candidate"], 26, 10)
            ax.scatter(
                group["normal_projection"],
                group["snr"],
                s=sizes,
                color=color,
                alpha=0.65,
                edgecolors="none",
                label=f"{source} segments",
            )
    ax.axvline(0.5, color="#666666", linestyle="--", linewidth=0.9, label="projection gate")
    ax.axhline(7.0, color="#444444", linestyle=":", linewidth=0.9, label="SNR gate")
    ax.set_xlabel("Normal projection onto X micro-scan")
    ax.set_ylabel("Local SNR")
    ax.set_yscale("log")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="lower right", fontsize=7)
    return fig


def plot_local_observability_map(
    frame: np.ndarray,
    outer_contour: np.ndarray,
    inner_contours: list[np.ndarray],
    segments: pd.DataFrame,
) -> plt.Figure:
    """Plot the legacy two-panel local observability figure."""
    fig, axes = make_figure("double_col", nrows=1, ncols=2, height=3.5)
    ax0, ax1 = axes
    im = ax0.imshow(frame, cmap=COLORMAPS["temperature"], origin="upper")
    ax0.plot(outer_contour[:, 0], outer_contour[:, 1], color=METHOD_COLOR_LIST[2], linewidth=0.8, label="outer")
    for contour in inner_contours[:40]:
        ax0.plot(contour[:, 0], contour[:, 1], color=METHOD_COLOR_LIST[0], linewidth=0.45, alpha=0.8)
    if not segments.empty:
        good = segments[segments["anchor_candidate"]]
        ax0.scatter(good["x_px"], good["y_px"], s=10, color=METHOD_COLOR_LIST[1], label="anchor candidates")

    # 2X Zoom ROI Crop: Center around outer contour with 30px padding
    x_min, x_max = np.min(outer_contour[:, 0]), np.max(outer_contour[:, 0])
    y_min, y_max = np.min(outer_contour[:, 1]), np.max(outer_contour[:, 1])
    pad = 30
    ax0.set_xlim(max(0, x_min - pad), min(frame.shape[1], x_max + pad))
    ax0.set_ylim(min(frame.shape[0], y_max + pad), max(0, y_min - pad))

    ax0.set_title("Contour-Level Local Edge Candidates")
    ax0.set_axis_off()
    cbar = fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02)
    format_colorbar(cbar, "Temperature [C]")
    ax0.legend(loc="lower right", fontsize=7)

    if not segments.empty:
        colors = segments["source"].map({"outer": METHOD_COLOR_LIST[2], "inner": METHOD_COLOR_LIST[0]}).fillna("#777777")
        sizes = np.where(segments["anchor_candidate"], 26, 10)
        ax1.scatter(segments["normal_projection"], segments["snr"], s=sizes, c=colors, alpha=0.65, edgecolors="none")
    ax1.axvline(0.5, color="#666666", linestyle="--", linewidth=0.9)
    ax1.axhline(7.0, color="#666666", linestyle="--", linewidth=0.9)
    ax1.set_xlabel("Normal projection onto X micro-scan")
    ax1.set_ylabel("Local SNR")
    ax1.set_yscale("log")
    ax1.set_title("Anchor Confidence Is Local")
    ax1.grid(axis="y", alpha=0.25)
    return fig


def plot_crb_esf_localization(
    crb_table: pd.DataFrame,
    *,
    noise_sigma_c: float,
    seed: int = 7,
) -> plt.Figure:
    """Plot synthetic ESF observability and CRB vs contrast."""
    rng = np.random.default_rng(seed)
    fig, axes = make_figure("double_col", nrows=1, ncols=2, height=3.4)
    ax0, ax1 = axes

    u_dense = np.linspace(-4.0, 4.0, 400)
    u_sample = np.arange(-4.0, 4.01, 0.5)
    for i, delta_t in enumerate((0.3, 1.0)):
        offset = i * 1.35
        clean = offset + delta_t * normal_cdf(u_dense / 0.5)
        observed = offset + delta_t * normal_cdf(u_sample / 0.5) + rng.normal(0.0, noise_sigma_c, size=u_sample.size)
        color = METHOD_COLOR_LIST[i]
        ax0.plot(u_dense, clean, color=color, label=f"Delta T={delta_t:.1f} C")
        ax0.scatter(u_sample, observed, s=14, color=color, alpha=0.75)
    ax0.axvline(0.0, color="#333333", linestyle="--", linewidth=0.9, label="edge position")
    ax0.set_xlabel("Normal coordinate [px]")
    ax0.set_ylabel("Temperature + offset [C]")
    ax0.set_title("(a) ESF Edge Position Is a Local Anchor")
    ax0.legend(loc="lower right", fontsize=7)
    ax0.grid(axis="y", alpha=0.2)

    sigma_values = sorted(crb_table["sigma_psf_px"].unique())
    model_styles = {
        "single_frame": "-",
        "16_frame_known_shift": (0, (5.0, 2.2)),
    }
    for i, sigma in enumerate(sigma_values):
        for model, linestyle in model_styles.items():
            subset = crb_table[
                crb_table["sigma_psf_px"].eq(sigma)
                & crb_table["model"].eq(model)
            ].sort_values("delta_t_c")
            if subset.empty:
                continue
            color = METHOD_COLOR_LIST[i]
            ax1.plot(
                subset["delta_t_c"],
                subset["crb_px"],
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                marker="o",
                markersize=5.8,
                markeredgewidth=0.5,
                markeredgecolor="white",
            )
    ax1.axhspan(0.03, 0.06, color="#888888", alpha=0.15)
    ax1.set_yscale("log")
    ax1.set_xlabel("Local edge contrast [C]")
    ax1.set_ylabel("CRB for edge position [px]")
    ax1.set_title("(b) CRB Supports Quality Gates, Not Shape Proof")
    ax1.grid(axis="y", alpha=0.25)

    sigma_handles = [
        Line2D(
            [0],
            [0],
            color=METHOD_COLOR_LIST[i],
            marker="o",
            linestyle="-",
            linewidth=1.8,
            markersize=6.4,
            markeredgewidth=0.5,
            markeredgecolor="white",
            label=f"sigma={float(sigma):.1f} px",
        )
        for i, sigma in enumerate(sigma_values)
    ]
    style_handles = [
        Line2D([0], [0], color="#30343B", linestyle="-", linewidth=2.0, label="single frame"),
        Line2D([0], [0], color="#30343B", linestyle=(0, (5.0, 2.2)), linewidth=2.0, label="16-frame known shift"),
        patches.Patch(facecolor="#888888", edgecolor="#888888", alpha=0.15, label="0.03-0.06 px anchor band"),
    ]
    ax1.legend(
        handles=sigma_handles + style_handles,
        loc="upper right",
        fontsize=6.8,
        handlelength=3.2,
        handletextpad=0.8,
        labelspacing=0.55,
        markerscale=1.15,
    )
    return fig


def plot_crb_sensitivity_surface(
    sensitivity_table: pd.DataFrame,
    *,
    sigma_values_px: tuple[float, ...] = (0.35, 0.5),
    phase_coverage_values_px: tuple[float, ...] = (0.5, 1.0),
) -> plt.Figure:
    """Plot CRB sensitivity over contrast and frame count for selected cases."""
    nrows = len(sigma_values_px)
    ncols = len(phase_coverage_values_px)
    fig, axes = make_figure("double_col", nrows=nrows, ncols=ncols, height=4.8)
    axes_arr = np.asarray(axes, dtype=object).reshape(nrows, ncols)

    images = []
    for r, sigma in enumerate(sigma_values_px):
        for c, coverage in enumerate(phase_coverage_values_px):
            ax = axes_arr[r, c]
            subset = sensitivity_table[
                np.isclose(sensitivity_table["sigma_psf_px"], float(sigma))
                & np.isclose(sensitivity_table["phase_coverage_px"], float(coverage))
            ]
            pivot = subset.pivot_table(
                index="n_frames",
                columns="delta_t_c",
                values="crb_px",
                aggfunc="first",
            ).sort_index().sort_index(axis=1)
            values = pivot.to_numpy(dtype=float)
            log_values = np.log10(np.maximum(values, 1e-3))
            im = ax.imshow(log_values, cmap=COLORMAPS["coverage"], aspect="auto", vmin=-2.2, vmax=0.0)
            images.append(im)
            ax.contour(
                np.arange(values.shape[1]),
                np.arange(values.shape[0]),
                values,
                levels=[0.05, 0.10],
                colors=["#FFFFFF", "#30343B"],
                linewidths=[1.0, 1.0],
            )
            for row in range(values.shape[0]):
                for col in range(values.shape[1]):
                    value = values[row, col]
                    label = f"{value:.3f}" if value < 0.1 else f"{value:.2f}"
                    color = "white" if log_values[row, col] < -0.65 else "black"
                    ax.text(col, row, label, ha="center", va="center", fontsize=6.5, color=color)

            ax.set_xticks(np.arange(len(pivot.columns)))
            ax.set_xticklabels([f"{x:.1f}" for x in pivot.columns])
            ax.set_yticks(np.arange(len(pivot.index)))
            ax.set_yticklabels([str(int(x)) for x in pivot.index])
            ax.set_title(f"sigma={sigma:.2f} px, phase={coverage:.1f} px")
            ax.set_xlabel("Delta T [C]")
            if c == 0:
                ax.set_ylabel("Frames")
            else:
                ax.set_ylabel("")

    cbar = fig.colorbar(images[-1], ax=axes_arr.ravel().tolist(), fraction=0.046, pad=0.02)
    format_colorbar(cbar, "log10 CRB [px]")
    return fig
