#!/usr/bin/env python3
"""Stage 0c real split-half FRC harness for solver_v2 redesign.

The harness uses the current project truth from AGENTS.md:

* detector sampling pitch: 20 um/pixel
* current spatial resolution: 20 um
* default SR output grid: 2x, i.e. 10 um/grid-sample display pitch

The 2x output pitch is a reconstruction grid, not a claim that the real optical
resolution is better than 20 um. Period samples below 20 um are retained as
audit points only and are flagged in the CSV/JSON outputs.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal.windows import tukey
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
ALGO_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[3]
EP06_SRC = PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src"
CORE_SRC = PROJECT_ROOT / "core" / "src"

for _path in (ALGO_ROOT / "src", EP06_SRC, CORE_SRC, SCRIPT_PATH.parent):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from common.alignment import load_alignment_shifts  # noqa: E402
from common.data_loader import (  # noqa: E402
    highpass_preprocess,
    load_main_session_frames,
    load_main_session_metadata,
    offset_correction,
)
from thermal_core.alignment_paths import default_contour_alignment_csv  # noqa: E402
from run_m2_frc import bilinear_drizzle, command_phase_bins, find_cutoff, stratified_split  # noqa: E402


EXPECTED_CLEAN_SR_FRAMES = 248
LR_SHAPE = (480, 640)
DEFAULT_SCALE = 2
DETECTOR_PIXEL_SIZE_UM = 20.0
CURRENT_SPATIAL_RESOLUTION_UM = 20.0
HIGHPASS_SIGMA_LR_PX = 5.0
DEFAULT_SEEDS = (42,)
DEFAULT_PERIODS_UM = (80.0, 60.0, 40.0, 30.0, 24.0, 20.0, 16.0, 14.0, 12.0, 10.0)


@dataclass(frozen=True)
class Inputs:
    frames: np.ndarray
    metadata: pd.DataFrame
    shifts: np.ndarray
    phase_bins: np.ndarray
    stage: dict[str, Any]
    frame_domain: str
    cache_hit: bool
    cache_dir: Path


@dataclass(frozen=True)
class SplitSpec:
    split_mode: str
    seed: int | None
    a_indices: np.ndarray
    b_indices: np.ndarray
    balance: pd.DataFrame


@dataclass(frozen=True)
class ReconstructionResult:
    image: np.ndarray
    zero_coverage_pct: float
    source: str


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_csv_floats(text: str) -> tuple[float, ...]:
    values = tuple(float(part.strip()) for part in str(text).split(",") if part.strip())
    if not values:
        raise ValueError("at least one period must be supplied")
    return values


def load_stage_config(path: Path, *, allow_non20: bool = False) -> dict[str, Any]:
    data = read_json(path)
    pixel_size_um = float(data["pixel_size_um"])
    spatial_resolution_um = float(data.get("current_spatial_resolution_um", pixel_size_um))
    audit = {
        "stage_config": path,
        "theta_deg": float(data["theta_deg"]),
        "pixel_size_um": pixel_size_um,
        "current_spatial_resolution_um": spatial_resolution_um,
        "detector_rows": int(data.get("detector_rows", LR_SHAPE[0])),
        "detector_cols": int(data.get("detector_cols", LR_SHAPE[1])),
        "strict_20um_audit_passed": bool(
            math.isclose(pixel_size_um, DETECTOR_PIXEL_SIZE_UM, rel_tol=0.0, abs_tol=1e-9)
            and math.isclose(spatial_resolution_um, CURRENT_SPATIAL_RESOLUTION_UM, rel_tol=0.0, abs_tol=1e-9)
        ),
        "audit_note": (
            "20 um/pixel detector pitch and 20 um spatial resolution are required by AGENTS.md; "
            "2x output grid pitch is pixel_size_um / scale and is not a physical resolution claim."
        ),
    }
    if not audit["strict_20um_audit_passed"] and not allow_non20:
        raise ValueError(
            "Stage config does not match current 20 um truth: "
            f"pixel_size_um={pixel_size_um}, current_spatial_resolution_um={spatial_resolution_um}. "
            "Pass --allow-non20 only for explicit legacy audits."
        )
    return audit


def validate_inputs(frames: np.ndarray, metadata: pd.DataFrame, shifts: np.ndarray, *, allow_limit: bool) -> None:
    if frames.ndim != 3:
        raise ValueError(f"frames must be (N,H,W), got {frames.shape}")
    if frames.shape[1:] != LR_SHAPE:
        raise ValueError(f"expected LR frame shape {LR_SHAPE}, got {frames.shape[1:]}")
    if shifts.shape != (frames.shape[0], 2):
        raise ValueError(f"expected shifts shape ({frames.shape[0]}, 2), got {shifts.shape}")
    if len(metadata) != frames.shape[0]:
        raise ValueError(f"metadata count {len(metadata)} does not match frames {frames.shape[0]}")
    if not allow_limit and frames.shape[0] != EXPECTED_CLEAN_SR_FRAMES:
        raise ValueError(f"expected {EXPECTED_CLEAN_SR_FRAMES} clean SR frames, got {frames.shape[0]}")


def prepare_inputs(args: argparse.Namespace) -> Inputs:
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    limit_tag = "full248" if args.limit is None else f"limit{int(args.limit)}"
    frames_path = cache_dir / f"frames_{args.frame_domain}_{limit_tag}.npy"
    shifts_path = cache_dir / f"shifts_{args.alignment_method}_{limit_tag}.npy"
    bins_path = cache_dir / f"phase_bins_scale{int(args.phase_bin_scale)}_{limit_tag}.npy"
    metadata_path = cache_dir / f"metadata_{limit_tag}.csv"

    stage = load_stage_config(args.stage_config, allow_non20=bool(args.allow_non20))
    cache_hit = (
        bool(args.cache_inputs)
        and not bool(args.force_inputs)
        and frames_path.exists()
        and shifts_path.exists()
        and bins_path.exists()
        and metadata_path.exists()
    )
    if cache_hit:
        frames = np.load(frames_path, mmap_mode="r")
        shifts = np.load(shifts_path)
        phase_bins = np.load(bins_path)
        metadata = pd.read_csv(metadata_path)
        validate_inputs(frames, metadata, shifts, allow_limit=args.limit is not None)
        return Inputs(
            frames=frames,
            metadata=metadata,
            shifts=np.asarray(shifts, dtype=np.float32),
            phase_bins=np.asarray(phase_bins, dtype=np.int16),
            stage=stage,
            frame_domain=str(args.frame_domain),
            cache_hit=True,
            cache_dir=cache_dir,
        )

    raw_frames, metadata = load_main_session_frames(
        args.data_dir,
        args.frame_audit_csv,
        workers=args.workers,
        dtype=np.float32,
        limit=args.limit,
    )
    if args.limit is None:
        shifts = load_alignment_shifts(
            args.alignment_method,
            metadata=metadata,
            alignment_csv=args.alignment_csv,
        ).astype(np.float32, copy=False)
    else:
        full_metadata = load_main_session_metadata(args.frame_audit_csv)
        full_shifts = load_alignment_shifts(
            args.alignment_method,
            metadata=full_metadata,
            alignment_csv=args.alignment_csv,
        ).astype(np.float32, copy=False)
        shifts = full_shifts[: len(metadata)]
    validate_inputs(raw_frames, metadata, shifts, allow_limit=args.limit is not None)

    if args.frame_domain == "offset":
        frames = offset_correction(raw_frames, method="median").astype(np.float32, copy=False)
    elif args.frame_domain == "highpass":
        frames = highpass_preprocess(raw_frames, sigma_bg=args.highpass_sigma, workers=args.workers).astype(
            np.float32,
            copy=False,
        )
    else:
        raise ValueError(f"unknown frame_domain={args.frame_domain!r}")

    phase_bins = command_phase_bins(
        metadata,
        scale=int(args.phase_bin_scale),
        theta_deg=float(stage["theta_deg"]),
        pixel_size_um=float(stage["pixel_size_um"]),
    ).astype(np.int16, copy=False)

    if args.cache_inputs:
        np.save(frames_path, np.asarray(frames, dtype=np.float32))
        np.save(shifts_path, np.asarray(shifts, dtype=np.float32))
        np.save(bins_path, np.asarray(phase_bins, dtype=np.int16))
        metadata.to_csv(metadata_path, index=False)
        write_json(
            cache_dir / f"input_summary_{limit_tag}.json",
            {
                "frames_npy": frames_path,
                "shifts_npy": shifts_path,
                "phase_bins_npy": bins_path,
                "metadata_csv": metadata_path,
                "frame_domain": args.frame_domain,
                "alignment_method": args.alignment_method,
                "alignment_csv": args.alignment_csv,
                "stage": stage,
                "n_frames": int(frames.shape[0]),
                "frame_shape_lr": list(frames.shape[1:]),
            },
        )

    return Inputs(
        frames=frames,
        metadata=metadata,
        shifts=shifts,
        phase_bins=phase_bins,
        stage=stage,
        frame_domain=str(args.frame_domain),
        cache_hit=False,
        cache_dir=cache_dir,
    )


def phase_balance(bin_ids: np.ndarray, a_idx: np.ndarray, b_idx: np.ndarray, *, scale: int) -> pd.DataFrame:
    rows: list[dict[str, int]] = []
    bin_arr = np.asarray(bin_ids, dtype=int)
    a_set = np.zeros(bin_arr.size, dtype=bool)
    b_set = np.zeros(bin_arr.size, dtype=bool)
    a_set[np.asarray(a_idx, dtype=int)] = True
    b_set[np.asarray(b_idx, dtype=int)] = True
    for bin_id in range(int(scale) * int(scale)):
        in_bin = bin_arr == bin_id
        count_a = int(np.sum(in_bin & a_set))
        count_b = int(np.sum(in_bin & b_set))
        rows.append(
            {
                "bin_id": int(bin_id),
                "count_total": int(np.sum(in_bin)),
                "count_a": count_a,
                "count_b": count_b,
                "abs_diff": int(abs(count_a - count_b)),
            }
        )
    return pd.DataFrame(rows)


def build_splits(args: argparse.Namespace, inputs: Inputs) -> list[SplitSpec]:
    modes = ["phase_stratified", "odd_even"] if args.split_mode == "both" else [str(args.split_mode)]
    split_specs: list[SplitSpec] = []
    n_frames = int(inputs.frames.shape[0])
    for mode in modes:
        if mode == "phase_stratified":
            for seed in args.seeds:
                a_idx, b_idx, balance = stratified_split(
                    np.asarray(inputs.phase_bins, dtype=int),
                    scale=int(args.phase_bin_scale),
                    seed=int(seed),
                )
                split_specs.append(SplitSpec(mode, int(seed), a_idx, b_idx, balance))
        elif mode == "odd_even":
            order = inputs.metadata["acquisition_order"].to_numpy(dtype=np.int64)
            a_idx = np.flatnonzero(order % 2 == 1)
            b_idx = np.flatnonzero(order % 2 == 0)
            if a_idx.size == 0 or b_idx.size == 0:
                seq = np.arange(n_frames, dtype=np.int64)
                a_idx = seq[seq % 2 == 1]
                b_idx = seq[seq % 2 == 0]
            balance = phase_balance(inputs.phase_bins, a_idx, b_idx, scale=int(args.phase_bin_scale))
            split_specs.append(SplitSpec(mode, None, np.sort(a_idx), np.sort(b_idx), balance))
        else:
            raise ValueError(f"unknown split mode: {mode}")
    return split_specs


def fill_nan(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    fill = float(np.nanmedian(arr[finite])) if bool(finite.any()) else 0.0
    return np.where(finite, arr, fill).astype(np.float32, copy=False)


def aligned_mean_reconstruct(
    frames: np.ndarray,
    shifts: np.ndarray,
    *,
    scale: int,
    upsample_order: int,
    desc: str,
) -> ReconstructionResult:
    frame_arr = np.asarray(frames, dtype=np.float32)
    shift_arr = np.asarray(shifts, dtype=np.float32)
    if frame_arr.ndim != 3:
        raise ValueError(f"frames must be (N,H,W), got {frame_arr.shape}")
    if shift_arr.shape != (frame_arr.shape[0], 2):
        raise ValueError(f"shifts shape {shift_arr.shape} does not match {frame_arr.shape[0]} frames")

    rows, cols = frame_arr.shape[1:]
    yy, xx = np.meshgrid(np.arange(rows, dtype=np.float32), np.arange(cols, dtype=np.float32), indexing="ij")
    accum = np.zeros((rows, cols), dtype=np.float64)
    counts = np.zeros((rows, cols), dtype=np.float32)
    for frame, (dx, dy) in tqdm(
        zip(frame_arr, shift_arr, strict=True),
        total=frame_arr.shape[0],
        desc=desc,
        disable=frame_arr.shape[0] < 4,
    ):
        src_y = yy - float(dy)
        src_x = xx - float(dx)
        valid = (
            (src_y >= 0.0)
            & (src_y <= rows - 1)
            & (src_x >= 0.0)
            & (src_x <= cols - 1)
            & np.isfinite(frame).all()
        )
        sampled = ndimage.map_coordinates(
            frame,
            (src_y, src_x),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )
        accum += np.where(valid, sampled, 0.0)
        counts += valid.astype(np.float32, copy=False)

    covered = counts > 0
    lr = np.empty((rows, cols), dtype=np.float32)
    lr[covered] = (accum[covered] / counts[covered]).astype(np.float32, copy=False)
    fill = float(np.nanmean(frame_arr))
    lr[~covered] = fill
    zero_coverage_pct = 100.0 * float(1.0 - covered.mean())
    if int(scale) == 1:
        image = lr
    else:
        image = ndimage.zoom(
            lr,
            zoom=(int(scale), int(scale)),
            order=int(upsample_order),
            mode="nearest",
            prefilter=int(upsample_order) > 1,
        ).astype(np.float32, copy=False)
    return ReconstructionResult(
        image=image,
        zero_coverage_pct=zero_coverage_pct,
        source=f"aligned_mean_lr_then_zoom_order{int(upsample_order)}",
    )


def reconstruct_method(
    method: str,
    frames: np.ndarray,
    shifts: np.ndarray,
    *,
    indices: np.ndarray,
    scale: int,
    aligned_upsample_order: int,
    split_label: str,
) -> ReconstructionResult:
    idx = np.asarray(indices, dtype=int)
    if method == "aligned_mean":
        return aligned_mean_reconstruct(
            frames[idx],
            shifts[idx],
            scale=scale,
            upsample_order=aligned_upsample_order,
            desc=f"{method} {split_label}",
        )
    if method == "drizzle":
        rec = bilinear_drizzle(
            np.asarray(frames[idx], dtype=np.float32),
            np.asarray(shifts[idx], dtype=np.float32),
            scale=scale,
            desc=f"{method} {split_label}",
        )
        return ReconstructionResult(
            image=np.asarray(rec.image, dtype=np.float32),
            zero_coverage_pct=float(rec.zero_coverage_pct),
            source="run_m2_frc.bilinear_drizzle",
        )
    raise ValueError(f"unknown method: {method}")


def crop_for_frc(image: np.ndarray, *, scale: int, crop_lr_px: int) -> np.ndarray:
    crop = int(crop_lr_px) * int(scale)
    arr = np.asarray(image, dtype=np.float32)
    if crop <= 0:
        return arr
    if arr.shape[0] <= 2 * crop or arr.shape[1] <= 2 * crop:
        raise ValueError(f"crop_lr_px={crop_lr_px} is too large for image shape {arr.shape} at scale={scale}")
    return arr[crop:-crop, crop:-crop]


def frc_curve_v2(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    scale: int,
    pixel_size_um: float,
    crop_lr_px: int,
    tukey_alpha: float,
) -> pd.DataFrame:
    a = fill_nan(crop_for_frc(image_a, scale=scale, crop_lr_px=crop_lr_px))
    b = fill_nan(crop_for_frc(image_b, scale=scale, crop_lr_px=crop_lr_px))
    if a.shape != b.shape:
        raise ValueError(f"FRC inputs have different shapes: {a.shape} vs {b.shape}")
    a = a - float(np.nanmean(a))
    b = b - float(np.nanmean(b))
    win_y = tukey(a.shape[0], alpha=float(tukey_alpha)).astype(np.float32)
    win_x = tukey(a.shape[1], alpha=float(tukey_alpha)).astype(np.float32)
    window = win_y[:, None] * win_x[None, :]

    fa = np.fft.fft2(a * window)
    fb = np.fft.fft2(b * window)
    cross = fa * np.conj(fb)
    power_a = np.abs(fa) ** 2
    power_b = np.abs(fb) ** 2

    hr_pitch_um = float(pixel_size_um) / float(scale)
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
            "hr_pitch_um": np.full_like(frequencies, hr_pitch_um, dtype=float),
            "pixel_size_um": np.full_like(frequencies, float(pixel_size_um), dtype=float),
        }
    )


def interpolate_frc(curve: pd.DataFrame, period_um: float) -> float:
    freq = curve["frequency_um_inv"].to_numpy(dtype=float)
    values = curve["frc"].to_numpy(dtype=float)
    valid = (freq > 0) & np.isfinite(freq) & np.isfinite(values)
    if not bool(valid.any()):
        return float("nan")
    order = np.argsort(freq[valid])
    freq_sorted = freq[valid][order]
    values_sorted = values[valid][order]
    target = 1.0 / float(period_um)
    # Period=20 um at scale=2 sits exactly on the 10 um-grid Nyquist edge.
    # Floating-point roundoff can make the last ring frequency one ulp below
    # 0.05, so allow a tiny edge snap instead of reporting NaN.
    edge_eps = max(1e-12, 1e-9 * float(freq_sorted[-1]))
    if target > float(freq_sorted[-1]) and target <= float(freq_sorted[-1]) + edge_eps:
        target = float(freq_sorted[-1])
    return float(np.interp(target, freq_sorted, values_sorted, left=np.nan, right=np.nan))


def nrmse_pair(a: np.ndarray, b: np.ndarray) -> float:
    lhs = fill_nan(a)
    rhs = fill_nan(b)
    valid = np.isfinite(lhs) & np.isfinite(rhs)
    if int(valid.sum()) < 2:
        return float("nan")
    denom = float(np.nanstd(lhs[valid]) + np.nanstd(rhs[valid]))
    return float(np.sqrt(np.nanmean((lhs[valid] - rhs[valid]) ** 2)) / max(denom, 1e-12))


def pearson_finite(a: np.ndarray, b: np.ndarray) -> float:
    lhs = fill_nan(a).ravel()
    rhs = fill_nan(b).ravel()
    valid = np.isfinite(lhs) & np.isfinite(rhs)
    if int(valid.sum()) < 2:
        return float("nan")
    return float(np.corrcoef(lhs[valid], rhs[valid])[0, 1])


def period_role(period_um: float, spatial_resolution_um: float) -> str:
    if float(period_um) >= float(spatial_resolution_um):
        return "at_or_above_current_spatial_resolution"
    return "below_20um_spatial_resolution_audit_only"


def evaluate_pair(
    *,
    method: str,
    split: SplitSpec,
    image_a: np.ndarray,
    image_b: np.ndarray,
    source: str,
    zero_a: float,
    zero_b: float,
    scale: int,
    stage: dict[str, Any],
    args: argparse.Namespace,
    curve_path: Path,
) -> dict[str, Any]:
    curve = frc_curve_v2(
        image_a,
        image_b,
        scale=scale,
        pixel_size_um=float(stage["pixel_size_um"]),
        crop_lr_px=int(args.crop_lr_px),
        tukey_alpha=float(args.tukey_alpha),
    )
    curve.insert(0, "seed", -1 if split.seed is None else int(split.seed))
    curve.insert(0, "split_mode", split.split_mode)
    curve.insert(0, "method", method)
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(curve_path, index=False)

    cutoff_1_7 = find_cutoff(curve, "threshold_1_7")
    cutoff_half = find_cutoff(curve, "threshold_half_bit")
    row: dict[str, Any] = {
        "method": method,
        "status": "success",
        "source": source,
        "split_mode": split.split_mode,
        "seed": -1 if split.seed is None else int(split.seed),
        "n_a": int(split.a_indices.size),
        "n_b": int(split.b_indices.size),
        "scale": int(scale),
        "pixel_size_um": float(stage["pixel_size_um"]),
        "hr_pitch_um": float(stage["pixel_size_um"]) / float(scale),
        "current_spatial_resolution_um": float(stage["current_spatial_resolution_um"]),
        "frc_curve_csv": rel(curve_path),
        "split_half_nrmse": nrmse_pair(image_a, image_b),
        "split_half_corr": pearson_finite(image_a, image_b),
        "zero_coverage_pct_a": float(zero_a),
        "zero_coverage_pct_b": float(zero_b),
        "phase_balance_max_abs_diff": int(split.balance["abs_diff"].max()) if not split.balance.empty else 0,
        "frc_cutoff_frequency_um_inv_1_7": float(cutoff_1_7.frequency_um_inv),
        "frc_cutoff_period_um_1_7": float(cutoff_1_7.period_um),
        "frc_cutoff_crossed_1_7": bool(cutoff_1_7.crossed),
        "frc_cutoff_frequency_um_inv_half_bit": float(cutoff_half.frequency_um_inv),
        "frc_cutoff_period_um_half_bit": float(cutoff_half.period_um),
        "frc_cutoff_crossed_half_bit": bool(cutoff_half.crossed),
        "cutoff_1_7_role": period_role(float(cutoff_1_7.period_um), float(stage["current_spatial_resolution_um"]))
        if np.isfinite(cutoff_1_7.period_um)
        else "no_finite_cutoff",
        "period_um_conversion_audit": "period_um = 1 / frequency_um_inv with d = pixel_size_um / scale; pixel_size_um fixed at 20.",
    }
    for period in args.periods_um:
        key = f"frc_at_{float(period):g}um".replace(".", "p")
        row[key] = interpolate_frc(curve, float(period))
        row[f"{key}_role"] = period_role(float(period), float(stage["current_spatial_resolution_um"]))
    return row


def parse_artifact_pair(spec: str) -> tuple[str, Path, Path, int, str]:
    parts = str(spec).split(":")
    if len(parts) not in (3, 4, 5):
        raise ValueError(
            "--artifact-pair must be name:path_a:path_b[:scale[:source_note]]; "
            f"got {spec!r}"
        )
    name = parts[0]
    path_a = Path(parts[1])
    path_b = Path(parts[2])
    scale = int(parts[3]) if len(parts) >= 4 and parts[3] else DEFAULT_SCALE
    note = parts[4] if len(parts) == 5 else "external_split_artifact_pair"
    return name, path_a, path_b, scale, note


def evaluate_artifact_pairs(args: argparse.Namespace, stage: dict[str, Any]) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    curves: list[pd.DataFrame] = []
    dummy_split = SplitSpec(
        split_mode="external_artifact_pair",
        seed=None,
        a_indices=np.asarray([], dtype=int),
        b_indices=np.asarray([], dtype=int),
        balance=pd.DataFrame(columns=["abs_diff"]),
    )
    for raw_spec in args.artifact_pair:
        name, path_a, path_b, scale, note = parse_artifact_pair(raw_spec)
        if not path_a.is_absolute():
            path_a = PROJECT_ROOT / path_a
        if not path_b.is_absolute():
            path_b = PROJECT_ROOT / path_b
        row_prefix: dict[str, Any] = {
            "method": name,
            "split_mode": "external_artifact_pair",
            "seed": -1,
            "scale": int(scale),
            "pixel_size_um": float(stage["pixel_size_um"]),
            "hr_pitch_um": float(stage["pixel_size_um"]) / float(scale),
            "current_spatial_resolution_um": float(stage["current_spatial_resolution_um"]),
            "source": note,
            "artifact_a": rel(path_a),
            "artifact_b": rel(path_b),
        }
        try:
            image_a = np.load(path_a).astype(np.float32, copy=False)
            image_b = np.load(path_b).astype(np.float32, copy=False)
            if image_a.shape != image_b.shape:
                raise ValueError(f"artifact shapes differ: {image_a.shape} vs {image_b.shape}")
            curve_path = args.output_dir / "frc_curves" / f"{name}_external_pair_frc_curve.csv"
            row = evaluate_pair(
                method=name,
                split=dummy_split,
                image_a=image_a,
                image_b=image_b,
                source=note,
                zero_a=float(np.mean(~np.isfinite(image_a)) * 100.0),
                zero_b=float(np.mean(~np.isfinite(image_b)) * 100.0),
                scale=scale,
                stage=stage,
                args=args,
                curve_path=curve_path,
            )
            row.update(row_prefix)
            rows.append(row)
            curves.append(pd.read_csv(curve_path))
        except Exception as exc:  # noqa: BLE001 - keep audit row materialized.
            failed = {
                **row_prefix,
                "status": "failed",
                "error": repr(exc),
            }
            rows.append(failed)
    return rows, curves


def parse_cross_pair(spec: str) -> tuple[str, list[Path], int, str]:
    parts = str(spec).split(":")
    if len(parts) not in (5, 6, 7):
        raise ValueError(
            "--cross-pair must be name:x_a:x_b:y_a:y_b[:scale[:source_note]]; "
            f"got {spec!r}"
        )
    name = parts[0]
    paths = [Path(p) for p in parts[1:5]]
    scale = int(parts[5]) if len(parts) >= 6 and parts[5] else DEFAULT_SCALE
    note = parts[6] if len(parts) == 7 else "cross_method_split_pair"
    return name, paths, scale, note


def evaluate_cross_pairs(args: argparse.Namespace, stage: dict[str, Any]) -> tuple[list[dict[str, Any]], list[pd.DataFrame]]:
    """Symmetrized cross-method split FRC: FRC(X_a, Y_b) and FRC(X_b, Y_a).

    A deterministic reconstructor that reprints the same learned prior in both
    halves inflates its own split-half FRC, but that shared-prior content does
    not correlate with an independent method's opposite half.  The cross FRC
    therefore only credits frequency content both methods genuinely recover
    from disjoint frames.  It is still a RELATIVE ranking tool (both methods
    share the burst alignment), not an absolute resolution claim.
    """

    rows: list[dict[str, Any]] = []
    curves: list[pd.DataFrame] = []
    for raw_spec in args.cross_pair:
        name, paths, scale, note = parse_cross_pair(raw_spec)
        paths = [p if p.is_absolute() else PROJECT_ROOT / p for p in paths]
        row_prefix: dict[str, Any] = {
            "method": name,
            "split_mode": "cross_method_pair",
            "seed": -1,
            "scale": int(scale),
            "pixel_size_um": float(stage["pixel_size_um"]),
            "hr_pitch_um": float(stage["pixel_size_um"]) / float(scale),
            "current_spatial_resolution_um": float(stage["current_spatial_resolution_um"]),
            "source": note,
            "cross_x_a": rel(paths[0]),
            "cross_x_b": rel(paths[1]),
            "cross_y_a": rel(paths[2]),
            "cross_y_b": rel(paths[3]),
        }
        try:
            images = [np.load(p).astype(np.float32, copy=False) for p in paths]
            shapes = {img.shape for img in images}
            if len(shapes) != 1:
                raise ValueError(f"cross-pair arrays have mixed shapes: {sorted(shapes)}")
            x_a, x_b, y_a, y_b = images

            def _curve(image_1: np.ndarray, image_2: np.ndarray) -> pd.DataFrame:
                return frc_curve_v2(
                    image_1,
                    image_2,
                    scale=scale,
                    pixel_size_um=float(stage["pixel_size_um"]),
                    crop_lr_px=int(args.crop_lr_px),
                    tukey_alpha=float(args.tukey_alpha),
                )

            curve_ab = _curve(x_a, y_b)
            curve_ba = _curve(x_b, y_a)
            if len(curve_ab) != len(curve_ba):
                raise ValueError("cross-pair FRC curves have different ring counts")
            mean_curve = curve_ab.copy()
            mean_curve["frc"] = 0.5 * (
                curve_ab["frc"].to_numpy(dtype=float) + curve_ba["frc"].to_numpy(dtype=float)
            )
            curve_dir = args.output_dir / "frc_curves"
            curve_dir.mkdir(parents=True, exist_ok=True)
            for tag, curve in (("xa_yb", curve_ab), ("xb_ya", curve_ba), ("mean", mean_curve)):
                labeled = curve.copy()
                labeled.insert(0, "seed", -1)
                labeled.insert(0, "split_mode", "cross_method_pair")
                labeled.insert(0, "method", name if tag == "mean" else f"{name}__{tag}")
                labeled.to_csv(curve_dir / f"{name}_cross_{tag}_frc_curve.csv", index=False)
                curves.append(labeled)

            scored = mean_curve.copy()
            cutoff_1_7 = find_cutoff(scored, "threshold_1_7")
            cutoff_half = find_cutoff(scored, "threshold_half_bit")
            row: dict[str, Any] = {
                **row_prefix,
                "status": "success",
                "n_a": 0,
                "n_b": 0,
                "frc_curve_csv": rel(curve_dir / f"{name}_cross_mean_frc_curve.csv"),
                "split_half_nrmse": float(np.mean([nrmse_pair(x_a, y_b), nrmse_pair(x_b, y_a)])),
                "split_half_corr": float(np.mean([pearson_finite(x_a, y_b), pearson_finite(x_b, y_a)])),
                "zero_coverage_pct_a": float(np.mean(~np.isfinite(x_a)) * 100.0),
                "zero_coverage_pct_b": float(np.mean(~np.isfinite(y_b)) * 100.0),
                "phase_balance_max_abs_diff": 0,
                "frc_cutoff_frequency_um_inv_1_7": float(cutoff_1_7.frequency_um_inv),
                "frc_cutoff_period_um_1_7": float(cutoff_1_7.period_um),
                "frc_cutoff_crossed_1_7": bool(cutoff_1_7.crossed),
                "frc_cutoff_frequency_um_inv_half_bit": float(cutoff_half.frequency_um_inv),
                "frc_cutoff_period_um_half_bit": float(cutoff_half.period_um),
                "frc_cutoff_crossed_half_bit": bool(cutoff_half.crossed),
                "cutoff_1_7_role": period_role(
                    float(cutoff_1_7.period_um), float(stage["current_spatial_resolution_um"])
                )
                if np.isfinite(cutoff_1_7.period_um)
                else "no_finite_cutoff",
                "period_um_conversion_audit": (
                    "period_um = 1 / frequency_um_inv with d = pixel_size_um / scale; pixel_size_um fixed at 20."
                ),
                "cross_frc_interpretation": (
                    "symmetrized cross-method FRC (mean of X_a-vs-Y_b and X_b-vs-Y_a); shared-prior "
                    "hallucination does not survive; relative ranking only, alignment still shared"
                ),
            }
            for period in args.periods_um:
                key = f"frc_at_{float(period):g}um".replace(".", "p")
                row[key] = interpolate_frc(mean_curve, float(period))
                row[f"{key}_role"] = period_role(float(period), float(stage["current_spatial_resolution_um"]))
                row[f"{key}_xa_yb"] = interpolate_frc(curve_ab, float(period))
                row[f"{key}_xb_ya"] = interpolate_frc(curve_ba, float(period))
            rows.append(row)
        except Exception as exc:  # noqa: BLE001 - keep audit row materialized.
            rows.append({**row_prefix, "status": "failed", "error": repr(exc)})
    return rows, curves


def save_reconstruction_if_requested(
    args: argparse.Namespace,
    image: np.ndarray,
    *,
    method: str,
    split: SplitSpec,
    side: str,
) -> str:
    if not args.save_recons:
        return ""
    seed = "odd_even" if split.seed is None else f"seed{int(split.seed)}"
    path = args.output_dir / "recons" / f"{method}_{split.split_mode}_{seed}_{side}.npy"
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(image, dtype=np.float32))
    return rel(path)


def run_native_methods(args: argparse.Namespace, inputs: Inputs, splits: list[SplitSpec]) -> tuple[list[dict[str, Any]], list[pd.DataFrame], list[pd.DataFrame]]:
    rows: list[dict[str, Any]] = []
    curves: list[pd.DataFrame] = []
    balances: list[pd.DataFrame] = []
    for split in splits:
        balance = split.balance.copy()
        balance.insert(0, "seed", -1 if split.seed is None else int(split.seed))
        balance.insert(0, "split_mode", split.split_mode)
        balances.append(balance)
        for method in args.methods:
            started = time.perf_counter()
            try:
                label = f"{split.split_mode} seed={split.seed}" if split.seed is not None else split.split_mode
                rec_a = reconstruct_method(
                    method,
                    inputs.frames,
                    inputs.shifts,
                    indices=split.a_indices,
                    scale=int(args.scale),
                    aligned_upsample_order=int(args.aligned_upsample_order),
                    split_label=f"{label} A",
                )
                rec_b = reconstruct_method(
                    method,
                    inputs.frames,
                    inputs.shifts,
                    indices=split.b_indices,
                    scale=int(args.scale),
                    aligned_upsample_order=int(args.aligned_upsample_order),
                    split_label=f"{label} B",
                )
                a_path = save_reconstruction_if_requested(args, rec_a.image, method=method, split=split, side="a")
                b_path = save_reconstruction_if_requested(args, rec_b.image, method=method, split=split, side="b")
                seed_tag = "odd_even" if split.seed is None else f"seed{int(split.seed)}"
                curve_path = args.output_dir / "frc_curves" / f"{method}_{split.split_mode}_{seed_tag}_frc_curve.csv"
                row = evaluate_pair(
                    method=method,
                    split=split,
                    image_a=rec_a.image,
                    image_b=rec_b.image,
                    source=f"{rec_a.source}; frame_domain={inputs.frame_domain}",
                    zero_a=rec_a.zero_coverage_pct,
                    zero_b=rec_b.zero_coverage_pct,
                    scale=int(args.scale),
                    stage=inputs.stage,
                    args=args,
                    curve_path=curve_path,
                )
                row["runtime_sec"] = float(time.perf_counter() - started)
                row["reconstruction_a_npy"] = a_path
                row["reconstruction_b_npy"] = b_path
                rows.append(row)
                curves.append(pd.read_csv(curve_path))
            except Exception as exc:  # noqa: BLE001 - one failed arm should not erase the manifest.
                rows.append(
                    {
                        "method": method,
                        "status": "failed",
                        "split_mode": split.split_mode,
                        "seed": -1 if split.seed is None else int(split.seed),
                        "runtime_sec": float(time.perf_counter() - started),
                        "error": repr(exc),
                    }
                )
    return rows, curves, balances


def write_outputs(
    args: argparse.Namespace,
    *,
    rows: list[dict[str, Any]],
    curves: list[pd.DataFrame],
    balances: list[pd.DataFrame],
    inputs: Inputs,
    elapsed_sec: float,
) -> dict[str, Any]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "method_summary.csv"
    curve_path = args.output_dir / "frc_curves_long.csv"
    balance_path = args.output_dir / "split_balance.csv"
    manifest_path = args.output_dir / "run_manifest.json"

    pd.DataFrame(rows).to_csv(summary_path, index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(curve_path, index=False)
    else:
        pd.DataFrame().to_csv(curve_path, index=False)
    if balances:
        pd.concat(balances, ignore_index=True).to_csv(balance_path, index=False)
    else:
        pd.DataFrame().to_csv(balance_path, index=False)

    failed = [row for row in rows if row.get("status") != "success"]
    manifest = {
        "task": "solver_v2 redesign Stage 0c real split-half FRC harness",
        "created_or_updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "success" if not failed else "completed_with_failures",
        "elapsed_sec": float(elapsed_sec),
        "project_truth_audit": {
            "detector_pixel_size_um": DETECTOR_PIXEL_SIZE_UM,
            "current_spatial_resolution_um": CURRENT_SPATIAL_RESOLUTION_UM,
            "stage_config": inputs.stage,
            "scale": int(args.scale),
            "hr_pitch_um": float(inputs.stage["pixel_size_um"]) / float(args.scale),
            "interpretation": (
                "FRC period_um is computed with 20 um detector pitch. For scale=2, the output grid "
                "pitch is 10 um, but periods below 20 um are audit samples below the current spatial "
                "resolution and must not be reported as proven optical resolution."
            ),
        },
        "inputs": {
            "n_frames": int(inputs.frames.shape[0]),
            "expected_clean_sr_frames": EXPECTED_CLEAN_SR_FRAMES,
            "frame_shape_lr": list(inputs.frames.shape[1:]),
            "frame_domain": inputs.frame_domain,
            "frame_domain_note": {
                "offset": "per-frame scalar median offset removed, matching EP15 M2 raw-temperature FRC style",
                "highpass": "Gaussian highpass frames, sigma in LR pixels",
            }.get(inputs.frame_domain, ""),
            "highpass_sigma_lr_px": float(args.highpass_sigma),
            "data_dir": args.data_dir,
            "frame_audit_csv": args.frame_audit_csv,
            "alignment_method": args.alignment_method,
            "alignment_csv": args.alignment_csv,
            "cache_hit": bool(inputs.cache_hit),
            "cache_dir": inputs.cache_dir,
            "limit": args.limit,
        },
        "split": {
            "split_mode": args.split_mode,
            "seeds": [int(seed) for seed in args.seeds],
            "phase_bin_scale": int(args.phase_bin_scale),
        },
        "methods_requested": list(args.methods),
        "artifact_pairs_requested": list(args.artifact_pair),
        "cross_pairs_requested": list(args.cross_pair),
        "periods_um": [
            {
                "period_um": float(period),
                "role": period_role(float(period), float(inputs.stage["current_spatial_resolution_um"])),
            }
            for period in args.periods_um
        ],
        "outputs": {
            "method_summary_csv": summary_path,
            "frc_curves_long_csv": curve_path,
            "split_balance_csv": balance_path,
            "run_manifest_json": manifest_path,
        },
        "failed_rows": failed,
    }
    write_json(manifest_path, manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "data_raw" / "infrared_avi")
    parser.add_argument("--frame-audit-csv", type=Path, default=PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--alignment-csv", type=Path, default=default_contour_alignment_csv(project_root_path=PROJECT_ROOT))
    parser.add_argument("--alignment-method", default="contour_refined")
    parser.add_argument("--stage-config", type=Path, default=PROJECT_ROOT / "configs" / "stage_calibration.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "stage0c_real_split_frc_v2")
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["aligned_mean", "drizzle"],
        choices=["aligned_mean", "drizzle", "none"],
        help="Native split reconstructions to run; 'none' skips loading the real burst "
        "entirely (artifact/cross pairs only).",
    )
    parser.add_argument("--split-mode", choices=["phase_stratified", "odd_even", "both"], default="phase_stratified")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--scale", type=int, default=DEFAULT_SCALE)
    parser.add_argument("--phase-bin-scale", type=int, default=DEFAULT_SCALE)
    parser.add_argument("--frame-domain", choices=["offset", "highpass"], default="offset")
    parser.add_argument("--highpass-sigma", type=float, default=HIGHPASS_SIGMA_LR_PX)
    parser.add_argument("--crop-lr-px", type=int, default=16)
    parser.add_argument("--tukey-alpha", type=float, default=0.25)
    parser.add_argument("--periods-um", type=parse_csv_floats, default=DEFAULT_PERIODS_UM)
    parser.add_argument("--aligned-upsample-order", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="Smoke/debug only. Omit for the full 248-frame contract.")
    parser.add_argument("--cache-inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force-inputs", action="store_true")
    parser.add_argument("--save-recons", action="store_true")
    parser.add_argument(
        "--artifact-pair",
        action="append",
        default=[],
        help="Optional external split pair: name:path_a.npy:path_b.npy[:scale[:source_note]].",
    )
    parser.add_argument(
        "--cross-pair",
        action="append",
        default=[],
        help="Optional symmetrized cross-method pair: name:x_a.npy:x_b.npy:y_a.npy:y_b.npy"
        "[:scale[:source_note]]. Scores FRC(x_a,y_b) and FRC(x_b,y_a) and their mean; "
        "kills shared-prior self-correlation that inflates same-method split FRC.",
    )
    parser.add_argument("--allow-non20", action="store_true", help="Only for explicit legacy scale audits.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if int(args.scale) < 1:
        raise ValueError("--scale must be >= 1")
    if int(args.phase_bin_scale) < 1:
        raise ValueError("--phase-bin-scale must be >= 1")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    print(f"[stage0c] output_dir={rel(args.output_dir)}")
    print(
        f"[stage0c] methods={args.methods}, split_mode={args.split_mode}, "
        f"frame_domain={args.frame_domain}, scale={args.scale}"
    )

    skip_native = list(args.methods) == ["none"]
    if skip_native:
        stage = load_stage_config(args.stage_config, allow_non20=bool(args.allow_non20))
        inputs = Inputs(
            frames=np.zeros((0,) + LR_SHAPE, dtype=np.float32),
            metadata=pd.DataFrame(),
            shifts=np.zeros((0, 2), dtype=np.float32),
            phase_bins=np.zeros((0,), dtype=np.int16),
            stage=stage,
            frame_domain=str(args.frame_domain),
            cache_hit=False,
            cache_dir=args.output_dir / "cache",
        )
        splits = []
        native_rows: list[dict[str, Any]] = []
        native_curves: list[pd.DataFrame] = []
        balances: list[pd.DataFrame] = []
    else:
        if "none" in args.methods:
            raise ValueError("--methods none cannot be combined with native methods")
        inputs = prepare_inputs(args)
        splits = build_splits(args, inputs)
        native_rows, native_curves, balances = run_native_methods(args, inputs, splits)
    artifact_rows, artifact_curves = evaluate_artifact_pairs(args, inputs.stage)
    cross_rows, cross_curves = evaluate_cross_pairs(args, inputs.stage)
    rows = native_rows + artifact_rows + cross_rows
    curves = native_curves + artifact_curves + cross_curves
    manifest = write_outputs(
        args,
        rows=rows,
        curves=curves,
        balances=balances,
        inputs=inputs,
        elapsed_sec=float(time.perf_counter() - started),
    )
    print(f"[stage0c] wrote {rel(args.output_dir / 'method_summary.csv')}")
    print(f"[stage0c] wrote {rel(args.output_dir / 'frc_curves_long.csv')}")
    print(f"[stage0c] wrote {rel(args.output_dir / 'run_manifest.json')}")
    failures = [row for row in rows if row.get("status") != "success"]
    if failures:
        print(f"[stage0c] completed with {len(failures)} failed row(s)", file=sys.stderr)
        return 2
    print(f"[stage0c] status={manifest['status']}, elapsed={manifest['elapsed_sec']:.2f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
