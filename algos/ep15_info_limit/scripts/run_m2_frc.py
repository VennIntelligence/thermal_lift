#!/usr/bin/env python3
"""Measure EP15 M2 Fourier Ring Correlation information cutoff.

This script reconstructs two independent, phase-stratified half sets with
bilinear drizzle, computes FRC curves, and runs the required controls for the
4x/5x information-limit decision.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal.windows import tukey
from tqdm import tqdm


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
from common.data_loader import bicubic_upsample, load_main_session_frames, offset_correction  # noqa: E402
from thermal_core.displacement import coordinate_to_shift  # noqa: E402
from thermal_core.plotting import FIGURE_SIZES, get_method_style, savefig_academic, setup_academic_style  # noqa: E402


EXPECTED_CLEAN_SR_FRAMES = 248
DEFAULT_SEEDS = (42, 123, 456)
PERIODS_OF_INTEREST_UM = (20.0, 16.0, 14.0, 12.0, 11.0, 10.0, 9.0, 8.0)
NOISE_FLOOR_C = 0.0724
PIXEL_SIZE_UM = 10.0


@dataclass(frozen=True)
class GridDecision:
    grid_scale: int
    splat_mode: str
    source: str


@dataclass(frozen=True)
class Reconstruction:
    image: np.ndarray
    zero_coverage_pct: float


@dataclass(frozen=True)
class Cutoff:
    frequency_um_inv: float
    period_um: float
    threshold_value: float
    frc_value: float
    crossed: bool


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_grid_decision(path: Path) -> GridDecision:
    """Load M1 grid decision if present; otherwise use the M2 defaults."""

    if not path.exists():
        return GridDecision(grid_scale=5, splat_mode="bilinear", source="default")
    data = _read_json(path)
    scale = int(data.get("grid_scale", data.get("scale", 5)))
    splat_mode = str(data.get("splat_mode", "bilinear"))
    return GridDecision(grid_scale=scale, splat_mode=splat_mode, source=_rel(path))


def load_stage_config(path: Path) -> dict[str, float]:
    data = _read_json(path)
    return {
        "theta_deg": float(data["theta_deg"]),
        "pixel_size_um": float(data["pixel_size_um"]),
    }


def command_phase_bins(
    metadata: pd.DataFrame,
    *,
    scale: int,
    theta_deg: float,
    pixel_size_um: float,
) -> np.ndarray:
    """Assign each frame to a command-prior detector phase bin."""

    missing = {"X", "Y"} - set(metadata.columns)
    if missing:
        raise ValueError(f"metadata is missing coordinate columns: {sorted(missing)}")
    dx_px, dy_px = coordinate_to_shift(
        metadata["X"].to_numpy(dtype=float),
        metadata["Y"].to_numpy(dtype=float),
        theta_deg=theta_deg,
        pixel_size_um=pixel_size_um,
    )
    phase_x = np.mod(dx_px, 1.0)
    phase_y = np.mod(dy_px, 1.0)
    bin_x = np.clip(np.floor(phase_x * scale).astype(int), 0, scale - 1)
    bin_y = np.clip(np.floor(phase_y * scale).astype(int), 0, scale - 1)
    return bin_y * scale + bin_x


def stratified_split(
    bin_ids: np.ndarray,
    *,
    scale: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Split each command phase bin into A/B by randomized odd-even alternation."""

    rng = np.random.default_rng(seed)
    a_indices: list[int] = []
    b_indices: list[int] = []
    rows: list[dict[str, int]] = []
    for bin_id in range(scale * scale):
        idx = np.flatnonzero(bin_ids == bin_id)
        if idx.size == 0:
            rows.append({"bin_id": bin_id, "count_total": 0, "count_a": 0, "count_b": 0, "abs_diff": 0})
            continue
        permuted = rng.permutation(idx)
        if rng.random() < 0.5:
            a_bin = permuted[::2]
            b_bin = permuted[1::2]
        else:
            a_bin = permuted[1::2]
            b_bin = permuted[::2]
        a_indices.extend(int(i) for i in a_bin)
        b_indices.extend(int(i) for i in b_bin)
        rows.append(
            {
                "bin_id": bin_id,
                "count_total": int(idx.size),
                "count_a": int(a_bin.size),
                "count_b": int(b_bin.size),
                "abs_diff": int(abs(a_bin.size - b_bin.size)),
            }
        )
    balance = pd.DataFrame(rows)
    if int(balance["abs_diff"].max()) > 1:
        raise AssertionError("phase-stratified A/B split violated the <=1 per-bin balance contract")
    return np.sort(np.asarray(a_indices, dtype=int)), np.sort(np.asarray(b_indices, dtype=int)), balance


def acquisition_half_split(metadata: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    order = metadata["acquisition_order"].to_numpy(dtype=float)
    sorted_indices = np.argsort(order, kind="mergesort")
    mid = len(sorted_indices) // 2
    return np.sort(sorted_indices[:mid]), np.sort(sorted_indices[mid:])


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
    """Bilinearly splat one LR frame into the HR accumulator."""

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


def bilinear_drizzle(
    frames: np.ndarray,
    shifts: np.ndarray,
    *,
    scale: int,
    desc: str,
) -> Reconstruction:
    """Reconstruct an HR image with bilinear point splatting."""

    if frames.ndim != 3:
        raise ValueError("frames must have shape (N, H, W)")
    if shifts.shape != (frames.shape[0], 2):
        raise ValueError(f"shifts shape {shifts.shape} does not match frame count {frames.shape[0]}")

    n_frames, rows, cols = frames.shape
    hr_shape = (rows * scale, cols * scale)
    accum = np.zeros(hr_shape, dtype=np.float32)
    weight_sum = np.zeros(hr_shape, dtype=np.float32)
    y_base = np.arange(rows, dtype=np.float64) * scale
    x_base = np.arange(cols, dtype=np.float64) * scale

    for frame, (dx_px, dy_px) in tqdm(
        zip(frames, shifts, strict=True),
        total=n_frames,
        desc=desc,
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
    result = np.empty(hr_shape, dtype=np.float32)
    result[covered] = accum[covered] / weight_sum[covered]
    fill = float(np.nanmean(frames))
    result[~covered] = fill
    return Reconstruction(result, zero_coverage_pct=zero_coverage_pct)


def crop_for_frc(image: np.ndarray, *, scale: int, crop_lr_px: int) -> np.ndarray:
    crop = int(crop_lr_px * scale)
    if crop <= 0:
        return np.asarray(image, dtype=np.float32)
    if image.shape[0] <= 2 * crop or image.shape[1] <= 2 * crop:
        raise ValueError(f"crop {crop} is too large for image shape {image.shape}")
    return np.asarray(image[crop:-crop, crop:-crop], dtype=np.float32)


def frc_curve(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    scale: int,
    crop_lr_px: int,
    tukey_alpha: float,
) -> pd.DataFrame:
    """Compute Fourier Ring Correlation with one-frequency-bin ring width."""

    a = crop_for_frc(image_a, scale=scale, crop_lr_px=crop_lr_px)
    b = crop_for_frc(image_b, scale=scale, crop_lr_px=crop_lr_px)
    if a.shape != b.shape:
        raise ValueError(f"FRC inputs have different shapes: {a.shape} vs {b.shape}")

    a = a - float(np.nanmean(a))
    b = b - float(np.nanmean(b))
    win_y = tukey(a.shape[0], alpha=tukey_alpha).astype(np.float32)
    win_x = tukey(a.shape[1], alpha=tukey_alpha).astype(np.float32)
    window = win_y[:, None] * win_x[None, :]

    fa = np.fft.fft2(a * window)
    fb = np.fft.fft2(b * window)
    cross = fa * np.conj(fb)
    power_a = np.abs(fa) ** 2
    power_b = np.abs(fb) ** 2

    hr_pitch_um = PIXEL_SIZE_UM / scale
    rows, cols = a.shape
    image_size = min(rows, cols)
    df = 1.0 / (image_size * hr_pitch_um)
    fy = np.fft.fftfreq(rows, d=hr_pitch_um)
    fx = np.fft.fftfreq(cols, d=hr_pitch_um)
    radial_frequency = np.hypot(fy[:, None], fx[None, :])
    ring_index = np.floor(radial_frequency / df + 1e-12).astype(np.int32)
    max_frequency = 0.5 / hr_pitch_um
    valid = radial_frequency <= max_frequency + 1e-12
    flat_ring = ring_index[valid].ravel()
    max_ring = int(flat_ring.max())

    numerator = np.bincount(flat_ring, weights=np.real(cross[valid]).ravel(), minlength=max_ring + 1)
    denom_a = np.bincount(flat_ring, weights=power_a[valid].ravel(), minlength=max_ring + 1)
    denom_b = np.bincount(flat_ring, weights=power_b[valid].ravel(), minlength=max_ring + 1)
    n_ring = np.bincount(flat_ring, minlength=max_ring + 1).astype(int)
    denominator = np.sqrt(np.maximum(denom_a * denom_b, 0.0))
    frc = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    frequencies = np.arange(max_ring + 1, dtype=float) * df
    periods = np.divide(1.0, frequencies, out=np.full_like(frequencies, np.inf), where=frequencies > 0)

    sqrt_n = np.sqrt(np.maximum(n_ring.astype(float), 1.0))
    half_bit = (0.2071 + 1.9102 / sqrt_n) / (1.2071 + 0.9102 / sqrt_n)
    return pd.DataFrame(
        {
            "frequency_um_inv": frequencies,
            "period_um": periods,
            "frc": frc,
            "threshold_1_7": np.full_like(frequencies, 1.0 / 7.0, dtype=float),
            "threshold_half_bit": half_bit,
            "n_ring_pixels": n_ring,
        }
    )


def average_curves(curves: list[pd.DataFrame]) -> pd.DataFrame:
    if not curves:
        raise ValueError("No FRC curves to average")
    base = curves[0][["frequency_um_inv", "period_um", "threshold_1_7", "threshold_half_bit", "n_ring_pixels"]].copy()
    stack = np.vstack([curve["frc"].to_numpy(dtype=float) for curve in curves])
    base["frc"] = np.nanmean(stack, axis=0)
    base = base[["frequency_um_inv", "period_um", "frc", "threshold_1_7", "threshold_half_bit", "n_ring_pixels"]]
    return base


def find_cutoff(curve: pd.DataFrame, threshold_column: str) -> Cutoff:
    valid = curve[
        (curve["frequency_um_inv"] > 0)
        & np.isfinite(curve["frc"])
        & np.isfinite(curve[threshold_column])
    ].copy()
    if valid.empty:
        return Cutoff(float("nan"), float("nan"), float("nan"), float("nan"), crossed=False)
    below = valid["frc"].to_numpy(dtype=float) < valid[threshold_column].to_numpy(dtype=float)
    if bool(below.any()):
        row = valid.iloc[int(np.argmax(below))]
        return Cutoff(
            frequency_um_inv=float(row["frequency_um_inv"]),
            period_um=float(row["period_um"]),
            threshold_value=float(row[threshold_column]),
            frc_value=float(row["frc"]),
            crossed=True,
        )
    row = valid.iloc[-1]
    return Cutoff(
        frequency_um_inv=float(row["frequency_um_inv"]),
        period_um=float(row["period_um"]),
        threshold_value=float(row[threshold_column]),
        frc_value=float(row["frc"]),
        crossed=False,
    )


def interpolate_curve(curve: pd.DataFrame, period_um: float, column: str) -> float:
    freq = curve["frequency_um_inv"].to_numpy(dtype=float)
    values = curve[column].to_numpy(dtype=float)
    target = 1.0 / float(period_um)
    valid = (freq > 0) & np.isfinite(freq) & np.isfinite(values)
    if not bool(valid.any()):
        return float("nan")
    order = np.argsort(freq[valid])
    return float(np.interp(target, freq[valid][order], values[valid][order], left=np.nan, right=np.nan))


def make_band_table(curves: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for period_um in PERIODS_OF_INTEREST_UM:
        row: dict[str, float | str] = {
            "period_um": period_um,
            "frequency_um_inv": 1.0 / period_um,
        }
        for name, curve in curves.items():
            row[f"{name}_frc"] = interpolate_curve(curve, period_um, "frc")
        rows.append(row)
    return pd.DataFrame(rows)


def aperture_dip_visible(band_table: pd.DataFrame) -> tuple[bool, float]:
    values = band_table.set_index("period_um")["main_frc"]
    if not {8.0, 10.0, 12.0}.issubset(values.index):
        return False, float("nan")
    shoulder = min(float(values.loc[8.0]), float(values.loc[12.0]))
    margin = shoulder - float(values.loc[10.0])
    return bool(np.isfinite(margin) and margin >= 0.03), float(margin)


def sigma_target_px(f_c_frequency_um_inv: float, *, mtf_val: float) -> tuple[float, float]:
    if not np.isfinite(f_c_frequency_um_inv) or f_c_frequency_um_inv <= 0:
        return float("nan"), float("nan")
    sigma_um = math.sqrt(math.log(1.0 / mtf_val) / (2.0 * math.pi**2 * f_c_frequency_um_inv**2))
    return sigma_um / PIXEL_SIZE_UM, sigma_um


def plot_frc_curve(curve: pd.DataFrame, cutoff: Cutoff, output_path: Path) -> None:
    setup_academic_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_col"])
    plot = curve[np.isfinite(curve["period_um"]) & (curve["period_um"] >= 4.0)].copy()
    ax.plot(plot["period_um"], plot["frc"], color="#4C72B0", label="phase-stratified FRC")
    ax.plot(plot["period_um"], plot["threshold_half_bit"], color="#666666", linewidth=0.9, label="half-bit")
    ax.axhline(1.0 / 7.0, color="#C44E52", linestyle="--", linewidth=0.9, label="1/7 threshold")
    if np.isfinite(cutoff.period_um):
        ax.axvline(cutoff.period_um, color="#4C72B0", linestyle=":", linewidth=1.0, label=f"f_c={cutoff.period_um:.2f} um")
    ax.axvline(10.0, color="#222222", linestyle="-.", linewidth=0.9, label="10 um aperture zero")
    ax.set_xlabel("Spatial period [um]")
    ax.set_ylabel("FRC")
    ax.set_title("M2 FRC Information Cutoff")
    x_max = max(30.0, float(cutoff.period_um) * 1.1 if np.isfinite(cutoff.period_um) else 30.0)
    ax.set_xlim(x_max, 4.0)
    ax.set_ylim(-0.08, 1.04)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=6.5)
    savefig_academic(fig, output_path)


def plot_controls(curves: dict[str, pd.DataFrame], output_path: Path) -> None:
    setup_academic_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["one_half_col"])
    labels = {
        "main": "main stratified",
        "positive_bicubic": "positive: bicubic LR",
        "negative_shift_shuffle": "negative: shuffled B shifts",
        "drift_acquisition_half": "drift: first/second half",
    }
    for idx, (name, curve) in enumerate(curves.items()):
        style = get_method_style(idx)
        plot = curve[np.isfinite(curve["period_um"]) & (curve["period_um"] >= 4.0)].copy()
        ax.plot(
            plot["period_um"],
            plot["frc"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.2,
            label=labels.get(name, name),
        )
    ax.axhline(1.0 / 7.0, color="#666666", linestyle="--", linewidth=0.9, label="1/7 threshold")
    ax.axvline(10.0, color="#222222", linestyle="-.", linewidth=0.9, label="10 um aperture zero")
    ax.set_xlabel("Spatial period [um]")
    ax.set_ylabel("FRC")
    ax.set_title("M2 FRC Controls")
    ax.set_xlim(30.0, 4.0)
    ax.set_ylim(-0.14, 1.04)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=6.5, ncol=1)
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


def control_expectations(
    *,
    main_cutoff: Cutoff,
    positive_cutoff: Cutoff,
    negative_curve: pd.DataFrame,
    drift_cutoff: Cutoff,
) -> dict[str, Any]:
    high_freq = negative_curve[
        (negative_curve["period_um"] <= 12.0)
        & (negative_curve["period_um"] >= 8.0)
        & np.isfinite(negative_curve["frc"])
    ]
    negative_high_frc = float(high_freq["frc"].median()) if not high_freq.empty else float("nan")
    return {
        "positive_control_fc_period_um": positive_cutoff.period_um,
        "positive_control_lower_resolution_than_main": bool(
            np.isfinite(positive_cutoff.period_um)
            and np.isfinite(main_cutoff.period_um)
            and positive_cutoff.period_um > main_cutoff.period_um
        ),
        "negative_control_8_12um_median_frc": negative_high_frc,
        "negative_control_high_freq_near_zero": bool(np.isfinite(negative_high_frc) and abs(negative_high_frc) < 0.10),
        "drift_control_fc_period_um": drift_cutoff.period_um,
        "drift_control_period_delta_um": (
            float(drift_cutoff.period_um - main_cutoff.period_um)
            if np.isfinite(drift_cutoff.period_um) and np.isfinite(main_cutoff.period_um)
            else float("nan")
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "data_raw" / "infrared_avi")
    parser.add_argument("--frame-audit-csv", type=Path, default=PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--alignment-csv", type=Path, default=default_contour_alignment_csv(project_root_path=PROJECT_ROOT))
    parser.add_argument("--stage-config", type=Path, default=PROJECT_ROOT / "configs" / "stage_calibration.json")
    parser.add_argument("--grid-decision-json", type=Path, default=PROJECT_ROOT / "output" / "ep15_info_limit" / "m1_phase_structure" / "grid_decision.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "ep15_info_limit" / "m2_frc")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--crop-lr-px", type=int, default=16)
    parser.add_argument("--tukey-alpha", type=float, default=0.25)
    parser.add_argument("--noise-sigma-c", type=float, default=NOISE_FLOOR_C)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_academic_style()

    grid = load_grid_decision(args.grid_decision_json)
    if grid.grid_scale < 2:
        raise ValueError("grid_scale must be >= 2")
    if grid.splat_mode != "bilinear":
        raise ValueError(f"M2 implements bilinear drizzle only; got splat_mode={grid.splat_mode!r}")
    stage = load_stage_config(args.stage_config)

    frames_raw, metadata = load_main_session_frames(
        args.data_dir,
        args.frame_audit_csv,
        workers=args.workers,
        dtype=np.float32,
    )
    shifts = load_alignment_shifts("contour_refined", metadata=metadata, alignment_csv=args.alignment_csv)
    validate_inputs(frames_raw, metadata, shifts)
    frames = offset_correction(frames_raw, method="median").astype(np.float32, copy=False)

    bin_ids = command_phase_bins(
        metadata,
        scale=grid.grid_scale,
        theta_deg=stage["theta_deg"],
        pixel_size_um=stage["pixel_size_um"],
    )
    bin_counts = pd.Series(bin_ids).value_counts().rename_axis("bin_id").reset_index(name="count").sort_values("bin_id")
    bin_counts.to_csv(args.output_dir / "command_phase_bin_counts.csv", index=False)

    main_curves: list[pd.DataFrame] = []
    repeat_rows: list[dict[str, Any]] = []
    split_balance_rows: list[pd.DataFrame] = []
    zero_coverage_values: list[float] = []
    seed42_a_image: np.ndarray | None = None
    seed42_b_indices: np.ndarray | None = None

    for seed in args.seeds:
        a_idx, b_idx, balance = stratified_split(bin_ids, scale=grid.grid_scale, seed=int(seed))
        balance.insert(0, "seed", int(seed))
        split_balance_rows.append(balance)
        rec_a = bilinear_drizzle(
            frames[a_idx],
            shifts[a_idx],
            scale=grid.grid_scale,
            desc=f"stratified A seed={seed}",
        )
        rec_b = bilinear_drizzle(
            frames[b_idx],
            shifts[b_idx],
            scale=grid.grid_scale,
            desc=f"stratified B seed={seed}",
        )
        zero_coverage_values.extend([rec_a.zero_coverage_pct, rec_b.zero_coverage_pct])
        curve = frc_curve(
            rec_a.image,
            rec_b.image,
            scale=grid.grid_scale,
            crop_lr_px=args.crop_lr_px,
            tukey_alpha=args.tukey_alpha,
        )
        main_curves.append(curve)
        cutoff_1_7 = find_cutoff(curve, "threshold_1_7")
        cutoff_half = find_cutoff(curve, "threshold_half_bit")
        repeat_rows.append(
            {
                "seed": int(seed),
                "n_a": int(a_idx.size),
                "n_b": int(b_idx.size),
                "max_phase_bin_abs_diff": int(balance["abs_diff"].max()),
                "zero_coverage_pct_a": rec_a.zero_coverage_pct,
                "zero_coverage_pct_b": rec_b.zero_coverage_pct,
                "f_c_frequency_um_inv_1_7": cutoff_1_7.frequency_um_inv,
                "f_c_period_um_1_7": cutoff_1_7.period_um,
                "f_c_crossed_1_7": cutoff_1_7.crossed,
                "f_c_frequency_um_inv_half_bit": cutoff_half.frequency_um_inv,
                "f_c_period_um_half_bit": cutoff_half.period_um,
                "f_c_crossed_half_bit": cutoff_half.crossed,
            }
        )
        if int(seed) == 42:
            seed42_a_image = rec_a.image
            seed42_b_indices = b_idx.copy()

    split_balance = pd.concat(split_balance_rows, ignore_index=True)
    split_balance.to_csv(args.output_dir / "stratified_split_balance.csv", index=False)
    repeats = pd.DataFrame(repeat_rows)
    repeats.to_csv(args.output_dir / "frc_repeats.csv", index=False)
    main_curve = average_curves(main_curves)
    main_curve[["frequency_um_inv", "period_um", "frc", "threshold_1_7", "threshold_half_bit"]].to_csv(
        args.output_dir / "frc_curve.csv",
        index=False,
    )

    main_cutoff = find_cutoff(main_curve, "threshold_1_7")
    main_half_cutoff = find_cutoff(main_curve, "threshold_half_bit")
    if seed42_a_image is None or seed42_b_indices is None:
        raise RuntimeError("Seed 42 is required for controls; include --seeds 42 ...")

    rng = np.random.default_rng(42)
    ref_idx = len(frames) // 2
    bicubic = bicubic_upsample(frames[ref_idx], scale=grid.grid_scale)
    positive_a = bicubic + rng.normal(0.0, args.noise_sigma_c, size=bicubic.shape).astype(np.float32)
    positive_b = bicubic + rng.normal(0.0, args.noise_sigma_c, size=bicubic.shape).astype(np.float32)
    positive_curve = frc_curve(
        positive_a,
        positive_b,
        scale=grid.grid_scale,
        crop_lr_px=args.crop_lr_px,
        tukey_alpha=args.tukey_alpha,
    )

    shuffled_b_shifts = shifts[seed42_b_indices][rng.permutation(seed42_b_indices.size)]
    rec_b_shuffle = bilinear_drizzle(
        frames[seed42_b_indices],
        shuffled_b_shifts,
        scale=grid.grid_scale,
        desc="negative shuffled B shifts",
    )
    zero_coverage_values.append(rec_b_shuffle.zero_coverage_pct)
    negative_curve = frc_curve(
        seed42_a_image,
        rec_b_shuffle.image,
        scale=grid.grid_scale,
        crop_lr_px=args.crop_lr_px,
        tukey_alpha=args.tukey_alpha,
    )

    first_idx, second_idx = acquisition_half_split(metadata)
    rec_first = bilinear_drizzle(
        frames[first_idx],
        shifts[first_idx],
        scale=grid.grid_scale,
        desc="drift control first half",
    )
    rec_second = bilinear_drizzle(
        frames[second_idx],
        shifts[second_idx],
        scale=grid.grid_scale,
        desc="drift control second half",
    )
    zero_coverage_values.extend([rec_first.zero_coverage_pct, rec_second.zero_coverage_pct])
    drift_curve = frc_curve(
        rec_first.image,
        rec_second.image,
        scale=grid.grid_scale,
        crop_lr_px=args.crop_lr_px,
        tukey_alpha=args.tukey_alpha,
    )

    controls = {
        "main": main_curve,
        "positive_bicubic": positive_curve,
        "negative_shift_shuffle": negative_curve,
        "drift_acquisition_half": drift_curve,
    }
    controls_csv = []
    for name, curve in controls.items():
        tmp = curve[["frequency_um_inv", "period_um", "frc", "threshold_1_7", "threshold_half_bit"]].copy()
        tmp.insert(0, "curve", name)
        controls_csv.append(tmp)
    pd.concat(controls_csv, ignore_index=True).to_csv(args.output_dir / "frc_controls.csv", index=False)

    band_table = make_band_table(controls)
    band_table.to_csv(args.output_dir / "frc_band_table.csv", index=False)
    dip_visible, dip_margin = aperture_dip_visible(band_table)

    positive_cutoff = find_cutoff(positive_curve, "threshold_1_7")
    negative_cutoff = find_cutoff(negative_curve, "threshold_1_7")
    drift_cutoff = find_cutoff(drift_curve, "threshold_1_7")
    controls_summary = control_expectations(
        main_cutoff=main_cutoff,
        positive_cutoff=positive_cutoff,
        negative_curve=negative_curve,
        drift_cutoff=drift_cutoff,
    )

    sigma_02_px, sigma_02_um = sigma_target_px(main_cutoff.frequency_um_inv, mtf_val=0.2)
    sigma_03_px, sigma_03_um = sigma_target_px(main_cutoff.frequency_um_inv, mtf_val=0.3)
    f_c_periods = repeats["f_c_period_um_1_7"].to_numpy(dtype=float)
    f_c_periods = f_c_periods[np.isfinite(f_c_periods)]

    plot_frc_curve(main_curve, main_cutoff, args.output_dir / "frc_curve.png")
    plot_controls(controls, args.output_dir / "frc_controls.png")

    theory_status = "undetermined"
    if np.isfinite(main_cutoff.period_um):
        if 10.0 <= main_cutoff.period_um <= 14.0:
            theory_status = "supports_11_14um_prediction"
        elif main_cutoff.period_um > 16.0:
            theory_status = "less_information_than_predicted"
        else:
            theory_status = "outside_primary_acceptance_band"

    summary = {
        "task": "EP15 M2 FRC information cutoff",
        "grid_scale": int(grid.grid_scale),
        "splat_mode": grid.splat_mode,
        "grid_decision_source": grid.source,
        "frame_preprocess": "per-frame median offset correction; each cropped FRC image is mean-centered before Tukey windowing",
        "n_clean_sr_frames": int(len(frames)),
        "frame_shape_lr": list(frames.shape[1:]),
        "hr_pitch_um": PIXEL_SIZE_UM / grid.grid_scale,
        "f_c_frequency_um_inv": main_cutoff.frequency_um_inv,
        "f_c_period_um": main_cutoff.period_um,
        "f_c_std_um": float(np.std(f_c_periods, ddof=1)) if f_c_periods.size > 1 else 0.0,
        "f_c_crossed_1_7": main_cutoff.crossed,
        "f_c_half_bit_frequency_um_inv": main_half_cutoff.frequency_um_inv,
        "f_c_half_bit_period_um": main_half_cutoff.period_um,
        "aperture_dip_visible": dip_visible,
        "aperture_dip_margin_frc": dip_margin,
        "sigma_target_02": sigma_02_px,
        "sigma_target_03": sigma_03_px,
        "sigma_target_02_um": sigma_02_um,
        "sigma_target_03_um": sigma_03_um,
        "zero_coverage_pct": float(np.mean(zero_coverage_values)),
        "zero_coverage_pct_max": float(np.max(zero_coverage_values)),
        "seeds": [int(seed) for seed in args.seeds],
        "controls": {
            **controls_summary,
            "negative_control_fc_period_um": negative_cutoff.period_um,
        },
        "theory_status": theory_status,
        "period_band_table": band_table.to_dict(orient="records"),
        "outputs": {
            "frc_curve_png": _rel(args.output_dir / "frc_curve.png"),
            "frc_curve_csv": _rel(args.output_dir / "frc_curve.csv"),
            "frc_controls_png": _rel(args.output_dir / "frc_controls.png"),
            "frc_controls_csv": _rel(args.output_dir / "frc_controls.csv"),
            "frc_band_table_csv": _rel(args.output_dir / "frc_band_table.csv"),
            "frc_repeats_csv": _rel(args.output_dir / "frc_repeats.csv"),
            "split_balance_csv": _rel(args.output_dir / "stratified_split_balance.csv"),
        },
        "elapsed_sec": float(time.perf_counter() - start),
    }
    (args.output_dir / "frc_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(f"FRC 1/7 cutoff period: {main_cutoff.period_um:.3f} um")
    print(f"FRC half-bit cutoff period: {main_half_cutoff.period_um:.3f} um")
    print(f"aperture dip visible: {dip_visible} (margin={dip_margin:.4f})")
    print(f"sigma_target MTF=0.2: {sigma_02_px:.4f} LR px ({sigma_02_um:.3f} um)")
    print(f"sigma_target MTF=0.3: {sigma_03_px:.4f} LR px ({sigma_03_um:.3f} um)")
    print(f"Saved M2 outputs to {_rel(args.output_dir)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
