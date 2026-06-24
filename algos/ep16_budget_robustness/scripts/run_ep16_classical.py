#!/usr/bin/env python3
"""Run EP16 CPU classical budget, robustness, and alignment-source ablations."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLBACKEND", "Agg")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import find_peaks
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
ALGO_ROOT = SCRIPT_DIR.parents[0]
PROJECT_ROOT = SCRIPT_DIR.parents[2]
for _path in (
    PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src",
    PROJECT_ROOT / "algos" / "ep10_drizzle" / "src",
    PROJECT_ROOT / "algos" / "ep15_info_limit" / "scripts",
    PROJECT_ROOT / "core" / "src",
):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from thermal_core.alignment_paths import default_contour_alignment_csv  # noqa: E402
from thermal_core.plotting import FIGURE_SIZES, savefig_academic, setup_academic_style  # noqa: E402

from common.alignment import load_alignment_shifts  # noqa: E402
from common.data_loader import bicubic_upsample, highpass_preprocess, load_main_session_frames  # noqa: E402
from common.metrics import artifact_score, split_half_consistency  # noqa: E402
from ep10_drizzle.drizzle_sr import coverage_statistics, drizzle_reconstruct  # noqa: E402
from run_m1_phase_structure import command_alignment_prior  # noqa: E402
from run_m2_frc import average_curves, command_phase_bins, find_cutoff, frc_curve, stratified_split  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "output" / "ep16_budget_robustness"
PAPER_FIGURE_DIR = PROJECT_ROOT / "output" / "paper_figures"
DATA_DIR = PROJECT_ROOT / "data" / "data_raw" / "infrared_avi"
FRAME_AUDIT_CSV = PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv"
ALIGNMENT_CSV = default_contour_alignment_csv(project_root_path=PROJECT_ROOT)
STAGE_CONFIG = PROJECT_ROOT / "configs" / "stage_calibration.json"
TGV_ENV = PROJECT_ROOT / "algos" / "ep10_tgv_sr" / ".venv"
DEFAULT_CONDA_EXE = Path("/home/ujs/miniforge3/bin/conda")
EXPECTED_CLEAN_SR_FRAMES = 248
SCALE = 2
HIGHPASS_SIGMA = 5.0
PERIODS_UM = (20.0, 16.0, 14.0, 12.0, 10.0)
DEFAULT_BUDGETS = (31, 62, 124, 248)
DEFAULT_DRIZZLE_BUDGET_SEEDS = (101, 202, 303)
DEFAULT_TGV_BUDGET_SEEDS = (101, 202)
DEFAULT_SHIFT_NOISE_SEEDS = (401, 402, 403)
DEFAULT_FRC_SEEDS = (42, 123, 456)
PIXEL_SIZE_UM = 20.0
ZIGZAG_ROI_FRACTION = 1.0 / 6.0
ZIGZAG_ROI_CENTER_YX = (0.5, 0.5)


@dataclass(frozen=True)
class RunSpec:
    arm: str
    n_frames: int
    subset_seed: int
    shift_source: str
    shift_noise_sigma_px: float = 0.0
    shift_noise_seed: int = 0
    run_id: str = ""


@dataclass(frozen=True)
class MatrixRow:
    experiment: str
    run_id: str
    arm: str
    n_frames: int
    subset_seed: int
    shift_source: str
    shift_noise_sigma_px: float
    shift_noise_seed: int
    row_role: str = ""


@dataclass
class Inputs:
    raw_frames: np.ndarray
    hp_frames: np.ndarray
    metadata: pd.DataFrame
    contour_shifts: np.ndarray
    command_shifts: np.ndarray
    phase_bins: np.ndarray
    cache_dir: Path
    raw_frames_npy: Path
    hp_frames_npy: Path


@dataclass
class Manifest:
    runs: dict[str, dict[str, Any]] = field(default_factory=dict)
    matrix_rows: list[dict[str, Any]] = field(default_factory=list)


def _token_float(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def _run_id(
    *,
    arm: str,
    n_frames: int,
    subset_seed: int,
    shift_source: str,
    shift_noise_sigma_px: float,
    shift_noise_seed: int,
) -> str:
    sigma = _token_float(shift_noise_sigma_px)
    return (
        f"{arm}_N{int(n_frames)}_{shift_source}_subset{int(subset_seed)}_"
        f"noise{sigma}_seed{int(shift_noise_seed)}"
    )


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        try:
            return str(value.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def pearson_finite(a: np.ndarray, b: np.ndarray) -> float:
    lhs = np.asarray(a, dtype=np.float64).ravel()
    rhs = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(lhs) & np.isfinite(rhs)
    if int(valid.sum()) < 2:
        return float("nan")
    return float(np.corrcoef(lhs[valid], rhs[valid])[0, 1])


def fill_nan(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    fill = float(np.nanmean(arr[finite])) if bool(finite.any()) else 0.0
    return np.where(finite, arr, fill).astype(np.float32, copy=False)


def crop_bounds_fraction(
    shape: tuple[int, int],
    *,
    fraction: float,
    y_frac: float = 0.5,
    x_frac: float = 0.5,
) -> tuple[int, int, int, int]:
    rows, cols = shape
    crop_rows = max(1, int(round(rows * float(fraction))))
    crop_cols = max(1, int(round(cols * float(fraction))))
    cy = int(round(rows * float(y_frac)))
    cx = int(round(cols * float(x_frac)))
    y0 = min(max(0, cy - crop_rows // 2), rows - crop_rows)
    x0 = min(max(0, cx - crop_cols // 2), cols - crop_cols)
    return y0, y0 + crop_rows, x0, x0 + crop_cols


def sample_line(image: np.ndarray, y0: float, x0: float, y1: float, x1: float, *, n_samples: int | None = None) -> np.ndarray:
    length = float(np.hypot(y1 - y0, x1 - x0))
    n = int(n_samples or max(16, round(length) + 1))
    ys = np.linspace(y0, y1, n)
    xs = np.linspace(x0, x1, n)
    return ndimage.map_coordinates(fill_nan(image), [ys, xs], order=1, mode="nearest")


def line_signal_for_dark_trace(profile: np.ndarray) -> np.ndarray:
    baseline = float(np.percentile(profile, 90.0))
    return (baseline - np.asarray(profile, dtype=np.float32)).astype(np.float32, copy=False)


def profile_metrics(signal: np.ndarray, *, pitch_um: float, min_spacing_um: float = 8.0) -> dict[str, float | bool | int]:
    """Copied from EP15 M4 on 2026-06-11, reduced to scalar profile metrics."""

    sig = ndimage.gaussian_filter1d(np.asarray(signal, dtype=np.float32), sigma=1.0, mode="nearest")
    base = float(np.percentile(sig, 10.0))
    height = float(np.max(sig) - base)
    if height <= 1e-8:
        return {
            "fwhm_um": float("nan"),
            "dip_depth": float("nan"),
            "lines_separated": False,
            "n_peaks": 0,
        }
    peaks, _props = find_peaks(sig, prominence=0.10 * height, distance=max(2, int(round(min_spacing_um / pitch_um))))
    if peaks.size == 0:
        peaks = np.asarray([int(np.argmax(sig))], dtype=int)
    primary = int(peaks[np.argmax(sig[peaks])])
    half = base + 0.5 * (float(sig[primary]) - base)
    left = primary
    while left > 0 and sig[left] > half:
        left -= 1
    right = primary
    while right < sig.size - 1 and sig[right] > half:
        right += 1
    fwhm_um = float(max(0, right - left) * pitch_um)
    if peaks.size >= 2:
        ranked = peaks[np.argsort(sig[peaks])[-2:]]
        p0, p1 = sorted(int(p) for p in ranked)
        valley = float(np.min(sig[p0 : p1 + 1]))
        peak_ref = min(float(sig[p0]), float(sig[p1]))
        dip_ratio = (valley - base) / max(peak_ref - base, 1e-8)
        dip_depth = float(1.0 - np.clip(dip_ratio, 0.0, 1.0))
        lines_separated = bool(dip_depth >= 0.25 and (p1 - p0) * pitch_um >= min_spacing_um)
    else:
        dip_depth = float("nan")
        lines_separated = False
    return {
        "fwhm_um": fwhm_um,
        "dip_depth": dip_depth,
        "lines_separated": lines_separated,
        "n_peaks": int(peaks.size),
    }


def zigzag_profile_specs(image_shape: tuple[int, int]) -> list[dict[str, float | str]]:
    """Copied from EP15 M4 on 2026-06-11; hard-coded center zigzag probes."""

    y0, y1, x0, x1 = crop_bounds_fraction(
        image_shape,
        fraction=ZIGZAG_ROI_FRACTION,
        y_frac=ZIGZAG_ROI_CENTER_YX[0],
        x_frac=ZIGZAG_ROI_CENTER_YX[1],
    )
    roi_h = y1 - y0
    roi_w = x1 - x0
    return [
        {"profile_id": "zigzag_upper_left", "y0": y0 + 0.24 * roi_h, "x0": x0 + 0.02 * roi_w, "y1": y0 + 0.24 * roi_h, "x1": x0 + 0.44 * roi_w},
        {"profile_id": "zigzag_mid_left", "y0": y0 + 0.36 * roi_h, "x0": x0 + 0.03 * roi_w, "y1": y0 + 0.36 * roi_h, "x1": x0 + 0.46 * roi_w},
        {"profile_id": "zigzag_lower_left", "y0": y0 + 0.49 * roi_h, "x0": x0 + 0.04 * roi_w, "y1": y0 + 0.49 * roi_h, "x1": x0 + 0.48 * roi_w},
    ]


def summarize_zigzag_image(image: np.ndarray) -> dict[str, Any]:
    pitch = PIXEL_SIZE_UM / SCALE
    rows: list[dict[str, Any]] = []
    for spec in zigzag_profile_specs(np.asarray(image).shape):
        profile = sample_line(
            image,
            float(spec["y0"]),
            float(spec["x0"]),
            float(spec["y1"]),
            float(spec["x1"]),
        )
        signal = line_signal_for_dark_trace(profile)
        rows.append({"profile_id": spec["profile_id"], **profile_metrics(signal, pitch_um=pitch)})
    table = pd.DataFrame(rows)
    return {
        "zigzag_fwhm_median_um": float(np.nanmedian(table["fwhm_um"].to_numpy(dtype=float))),
        "zigzag_dip_depth_median": float(np.nanmedian(table["dip_depth"].to_numpy(dtype=float))),
        "zigzag_profiles_separated": int(table["lines_separated"].astype(bool).sum()),
    }


def load_stage_config(path: Path) -> dict[str, float]:
    data = read_json(path)
    return {"theta_deg": float(data["theta_deg"]), "pixel_size_um": float(data["pixel_size_um"])}


def reference_file_from_alignment(path: Path) -> str:
    alignment = pd.read_csv(path, usecols=lambda col: col in {"reference_file"})
    if "reference_file" not in alignment or alignment["reference_file"].dropna().empty:
        raise ValueError(f"{path} does not contain reference_file")
    return str(alignment["reference_file"].dropna().iloc[0])


def prepare_inputs(args: argparse.Namespace) -> Inputs:
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / "raw_frames.npy"
    hp_path = cache_dir / "hp_frames.npy"

    raw_frames, metadata = load_main_session_frames(
        args.data_dir,
        args.frame_audit_csv,
        workers=args.io_workers,
        dtype=np.float32,
    )
    if len(metadata) != EXPECTED_CLEAN_SR_FRAMES or raw_frames.shape != (EXPECTED_CLEAN_SR_FRAMES, 480, 640):
        raise ValueError(f"Expected 248 clean 480x640 frames, got {raw_frames.shape} metadata={len(metadata)}")
    hp_frames = highpass_preprocess(raw_frames, sigma_bg=HIGHPASS_SIGMA, workers=args.io_workers)
    contour_shifts = load_alignment_shifts("contour_refined", metadata=metadata, alignment_csv=args.alignment_csv)
    stage = load_stage_config(args.stage_config)
    ref_file = reference_file_from_alignment(args.alignment_csv)
    _rel_x, _rel_y, command_dx, command_dy = command_alignment_prior(
        metadata,
        ref_file,
        theta_deg=stage["theta_deg"],
        pixel_size_um=stage["pixel_size_um"],
    )
    command_shifts = np.column_stack([command_dx, command_dy]).astype(np.float32, copy=False)
    phase_bins = command_phase_bins(
        metadata,
        scale=SCALE,
        theta_deg=stage["theta_deg"],
        pixel_size_um=stage["pixel_size_um"],
    )

    np.save(raw_path, raw_frames.astype(np.float32, copy=False))
    np.save(hp_path, hp_frames.astype(np.float32, copy=False))
    metadata.to_csv(cache_dir / "metadata.csv", index=False)
    np.save(cache_dir / "contour_shifts.npy", contour_shifts.astype(np.float32, copy=False))
    np.save(cache_dir / "command_shifts.npy", command_shifts.astype(np.float32, copy=False))
    np.save(cache_dir / "phase_bins_2x.npy", phase_bins.astype(np.int16, copy=False))
    write_json(
        cache_dir / "input_summary.json",
        {
            "frames_shape": list(raw_frames.shape),
            "clean_sr_frames": int(len(metadata)),
            "highpass_sigma_lr_px": HIGHPASS_SIGMA,
            "alignment_csv": args.alignment_csv,
            "stage_config": args.stage_config,
            "reference_file": ref_file,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    )
    return Inputs(raw_frames, hp_frames, metadata, contour_shifts, command_shifts, phase_bins, cache_dir, raw_path, hp_path)


def sample_phase_stratified_indices(bin_ids: np.ndarray, n_frames: int, seed: int) -> np.ndarray:
    bin_arr = np.asarray(bin_ids, dtype=int)
    n_total = int(bin_arr.size)
    if int(n_frames) >= n_total:
        return np.arange(n_total, dtype=int)
    rng = np.random.default_rng(int(seed))
    bins = np.unique(bin_arr)
    occupied = [int(b) for b in bins if np.any(bin_arr == b)]
    if n_frames < len(occupied):
        raise ValueError(f"n_frames={n_frames} is smaller than occupied phase bins={len(occupied)}")
    counts = {b: int(np.sum(bin_arr == b)) for b in occupied}
    ideal = {b: n_frames * counts[b] / n_total for b in occupied}
    take = {b: max(1, int(np.floor(ideal[b]))) for b in occupied}
    while sum(take.values()) > n_frames:
        candidates = [b for b in occupied if take[b] > 1]
        b = min(candidates, key=lambda item: (ideal[item] - math.floor(ideal[item]), take[item]))
        take[b] -= 1
    remainders = sorted(
        occupied,
        key=lambda item: (ideal[item] - math.floor(ideal[item]), rng.random()),
        reverse=True,
    )
    for b in remainders:
        if sum(take.values()) >= n_frames:
            break
        if take[b] < counts[b]:
            take[b] += 1
    selected: list[int] = []
    for b in occupied:
        idx = np.flatnonzero(bin_arr == b)
        selected.extend(int(i) for i in rng.choice(idx, size=take[b], replace=False))
    return np.sort(np.asarray(selected, dtype=int))


def shifts_for_spec(spec: RunSpec, inputs: Inputs) -> np.ndarray:
    if spec.shift_source == "contour_refined":
        shifts = np.asarray(inputs.contour_shifts, dtype=np.float32)
    elif spec.shift_source == "command_prior":
        shifts = np.asarray(inputs.command_shifts, dtype=np.float32)
    else:
        raise ValueError(f"Unknown shift_source={spec.shift_source}")
    shifts = shifts.copy()
    if float(spec.shift_noise_sigma_px) > 0:
        rng = np.random.default_rng(int(spec.shift_noise_seed))
        shifts += rng.normal(0.0, float(spec.shift_noise_sigma_px), size=shifts.shape).astype(np.float32)
    return shifts


def subset_for_spec(spec: RunSpec, inputs: Inputs) -> np.ndarray:
    return sample_phase_stratified_indices(inputs.phase_bins, spec.n_frames, spec.subset_seed)


def build_matrices(args: argparse.Namespace) -> tuple[dict[str, RunSpec], dict[str, list[MatrixRow]]]:
    arms = ["drizzle", "tgv"] if args.arms == "both" else [args.arms]
    specs: dict[str, RunSpec] = {}
    matrices: dict[str, list[MatrixRow]] = {"frame_budget": [], "shift_robustness": [], "alignment_source": []}

    def add(row: MatrixRow) -> None:
        spec = RunSpec(
            arm=row.arm,
            n_frames=row.n_frames,
            subset_seed=row.subset_seed,
            shift_source=row.shift_source,
            shift_noise_sigma_px=row.shift_noise_sigma_px,
            shift_noise_seed=row.shift_noise_seed,
            run_id=row.run_id,
        )
        specs.setdefault(row.run_id, spec)
        matrices[row.experiment].append(row)

    for arm in arms:
        budget_seeds = DEFAULT_TGV_BUDGET_SEEDS if arm == "tgv" else DEFAULT_DRIZZLE_BUDGET_SEEDS
        for n_frames in DEFAULT_BUDGETS:
            seeds = (0,) if n_frames == EXPECTED_CLEAN_SR_FRAMES else budget_seeds
            for seed in seeds:
                run_id = _run_id(
                    arm=arm,
                    n_frames=n_frames,
                    subset_seed=seed,
                    shift_source="contour_refined",
                    shift_noise_sigma_px=0.0,
                    shift_noise_seed=0,
                )
                add(MatrixRow("frame_budget", run_id, arm, n_frames, seed, "contour_refined", 0.0, 0, "budget"))

        for sigma in (0.0, 0.05, 0.1, 0.2):
            seeds = (0,) if sigma == 0.0 else DEFAULT_SHIFT_NOISE_SEEDS
            for seed in seeds:
                run_id = _run_id(
                    arm=arm,
                    n_frames=EXPECTED_CLEAN_SR_FRAMES,
                    subset_seed=0,
                    shift_source="contour_refined",
                    shift_noise_sigma_px=sigma,
                    shift_noise_seed=seed,
                )
                add(MatrixRow("shift_robustness", run_id, arm, EXPECTED_CLEAN_SR_FRAMES, 0, "contour_refined", sigma, seed, "shift_noise"))

        for source in ("command_prior", "contour_refined"):
            run_id = _run_id(
                arm=arm,
                n_frames=EXPECTED_CLEAN_SR_FRAMES,
                subset_seed=0,
                shift_source=source,
                shift_noise_sigma_px=0.0,
                shift_noise_seed=0,
            )
            add(MatrixRow("alignment_source", run_id, arm, EXPECTED_CLEAN_SR_FRAMES, 0, source, 0.0, 0, "alignment_source"))

    return specs, matrices


def raw_control_highpass(raw_subset: np.ndarray) -> np.ndarray:
    raw_mean_hr = bicubic_upsample(np.nanmean(raw_subset, axis=0), scale=SCALE)
    return highpass_preprocess(raw_mean_hr, sigma_bg=HIGHPASS_SIGMA)


def drizzle_method(frames: np.ndarray, shifts: np.ndarray, **kwargs: Any) -> np.ndarray:
    hr, _coverage = drizzle_reconstruct(frames, shifts, scale=SCALE, pixfrac=0.7, kernel="square", **kwargs)
    return hr


def compute_split_metrics(
    frames: np.ndarray,
    shifts: np.ndarray,
    *,
    n_splits: int,
    seed: int,
) -> dict[str, Any]:
    table = split_half_consistency(
        frames,
        shifts,
        drizzle_method,
        n_splits=int(n_splits),
        random_state=int(seed),
        coverage_threshold=1.0,
    )
    return {
        "split_half_n_splits_actual": int(len(table)),
        "split_half_nrmse_median": float(table["nrmse"].median()),
        "split_half_corr_median": float(table["corr"].median()),
        "split_half_rmse_median": float(table["rmse"].median()),
    }


def compute_frc_metrics(
    frames: np.ndarray,
    shifts: np.ndarray,
    bin_ids: np.ndarray,
    *,
    seeds: tuple[int, ...],
) -> dict[str, Any]:
    curves: list[pd.DataFrame] = []
    balance_max: list[int] = []
    n_a_values: list[int] = []
    n_b_values: list[int] = []
    for seed in seeds:
        a_idx, b_idx, balance = stratified_split(bin_ids, scale=SCALE, seed=int(seed))
        if a_idx.size == 0 or b_idx.size == 0:
            continue
        rec_a, _cov_a = drizzle_reconstruct(frames[a_idx], shifts[a_idx], scale=SCALE, pixfrac=0.7, kernel="square")
        rec_b, _cov_b = drizzle_reconstruct(frames[b_idx], shifts[b_idx], scale=SCALE, pixfrac=0.7, kernel="square")
        curve = frc_curve(
            fill_nan(rec_a),
            fill_nan(rec_b),
            scale=SCALE,
            crop_lr_px=16,
            tukey_alpha=0.25,
        )
        curves.append(curve)
        balance_max.append(int(balance["abs_diff"].max()))
        n_a_values.append(int(a_idx.size))
        n_b_values.append(int(b_idx.size))
    if not curves:
        return {
            "frc_seed_count": 0,
            "frc_cutoff_period_um_1_7": float("nan"),
            "frc_cutoff_crossed_1_7": False,
            **{f"frc_{int(period)}um": float("nan") for period in PERIODS_UM},
        }
    avg = average_curves(curves)
    cutoff = find_cutoff(avg, "threshold_1_7")
    out: dict[str, Any] = {
        "frc_seed_count": int(len(curves)),
        "frc_cutoff_frequency_um_inv_1_7": cutoff.frequency_um_inv,
        "frc_cutoff_period_um_1_7": cutoff.period_um,
        "frc_cutoff_crossed_1_7": cutoff.crossed,
        "frc_split_balance_max_abs_diff": int(max(balance_max)) if balance_max else 0,
        "frc_split_n_a_min": int(min(n_a_values)) if n_a_values else 0,
        "frc_split_n_b_min": int(min(n_b_values)) if n_b_values else 0,
    }
    for period in PERIODS_UM:
        freq = avg["frequency_um_inv"].to_numpy(dtype=float)
        values = avg["frc"].to_numpy(dtype=float)
        valid = (freq > 0) & np.isfinite(freq) & np.isfinite(values)
        target = 1.0 / float(period)
        if bool(valid.any()):
            order = np.argsort(freq[valid])
            out[f"frc_{int(period)}um"] = float(np.interp(target, freq[valid][order], values[valid][order], left=np.nan, right=np.nan))
        else:
            out[f"frc_{int(period)}um"] = float("nan")
    return out


def evaluate_reconstruction(
    *,
    spec: RunSpec,
    hr: np.ndarray,
    coverage: np.ndarray | None,
    frames: np.ndarray,
    raw_frames: np.ndarray,
    shifts: np.ndarray,
    bin_ids: np.ndarray,
    runtime_sec: float,
    hr_npy: Path,
    convergence_csv: Path | None,
    split_n: int,
    split_source: str,
    frc_source: str,
    tgv_child: dict[str, Any] | None = None,
) -> dict[str, Any]:
    raw_hp = raw_control_highpass(raw_frames)
    metrics: dict[str, Any] = {
        "run_id": spec.run_id,
        "arm": spec.arm,
        "status": "success",
        "n_frames": int(spec.n_frames),
        "subset_seed": int(spec.subset_seed),
        "shift_source": spec.shift_source,
        "shift_noise_sigma_px": float(spec.shift_noise_sigma_px),
        "shift_noise_seed": int(spec.shift_noise_seed),
        "low_n_flag": bool(spec.n_frames == 31),
        "runtime_sec": float(runtime_sec),
        "hr_npy": rel(hr_npy),
        "convergence_csv": rel(convergence_csv) if convergence_csv else "",
        "artifact_score": float(artifact_score(hr, lr_img=np.nanmean(frames, axis=0), scale=SCALE)),
        "raw_control_corr": pearson_finite(hr, raw_hp),
        "split_half_metric_source": split_source,
        "split_half_n_splits_requested": int(split_n),
        "frc_metric_source": frc_source,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }
    metrics.update(summarize_zigzag_image(hr))
    metrics.update(compute_split_metrics(frames, shifts, n_splits=split_n, seed=spec.subset_seed + 7001))
    metrics.update(compute_frc_metrics(frames, shifts, bin_ids, seeds=DEFAULT_FRC_SEEDS))
    if coverage is not None:
        metrics.update(coverage_statistics(coverage, threshold=1.0))
    else:
        metrics.update(
            {
                "min_coverage": float("nan"),
                "coverage_lt1_fraction": float("nan"),
                "coverage_p05": float("nan"),
                "coverage_median": float("nan"),
                "coverage_p95": float("nan"),
            }
        )
    if tgv_child:
        metrics.update(
            {
                "tgv_child_runtime_sec": tgv_child.get("runtime_sec"),
                "tgv_iterations": tgv_child.get("iterations"),
                "tgv_backend": tgv_child.get("tgv_backend", {}).get("backend") if isinstance(tgv_child.get("tgv_backend"), dict) else "",
                "tgv_backend_status": tgv_child.get("tgv_backend", {}).get("status") if isinstance(tgv_child.get("tgv_backend"), dict) else "",
                "tgv_backend_device": tgv_child.get("tgv_backend", {}).get("selected_device") if isinstance(tgv_child.get("tgv_backend"), dict) else "",
            }
        )
    return metrics


def failed_record(spec: RunSpec, started: float, exc: BaseException) -> dict[str, Any]:
    return {
        "run_id": spec.run_id,
        "arm": spec.arm,
        "status": "failed",
        "n_frames": int(spec.n_frames),
        "subset_seed": int(spec.subset_seed),
        "shift_source": spec.shift_source,
        "shift_noise_sigma_px": float(spec.shift_noise_sigma_px),
        "shift_noise_seed": int(spec.shift_noise_seed),
        "runtime_sec": float(time.perf_counter() - started),
        "error": repr(exc),
        "traceback": traceback.format_exc(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    }


def metric_path(output_dir: Path, run_id: str) -> Path:
    return output_dir / "runs" / f"{run_id}.json"


def run_drizzle_spec(spec: RunSpec, inputs: Inputs, args: argparse.Namespace) -> dict[str, Any]:
    path = metric_path(args.output_dir, spec.run_id)
    if args.resume and path.exists():
        cached = read_json(path)
        if cached.get("status") == "success":
            return cached
    started = time.perf_counter()
    try:
        idx = subset_for_spec(spec, inputs)
        shifts_all = shifts_for_spec(spec, inputs)
        frames = inputs.hp_frames[idx]
        raw_subset = inputs.raw_frames[idx]
        shifts = shifts_all[idx]
        bin_ids = inputs.phase_bins[idx]
        hr, coverage = drizzle_reconstruct(frames, shifts, scale=SCALE, pixfrac=0.7, kernel="square")
        hr_path = args.output_dir / "hr" / f"{spec.run_id}_hr.npy"
        cov_path = args.output_dir / "coverage" / f"{spec.run_id}_coverage.npy"
        hr_path.parent.mkdir(parents=True, exist_ok=True)
        cov_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(hr_path, hr.astype(np.float32, copy=False))
        np.save(cov_path, coverage.astype(np.float32, copy=False))
        row = evaluate_reconstruction(
            spec=spec,
            hr=hr,
            coverage=coverage,
            frames=frames,
            raw_frames=raw_subset,
            shifts=shifts,
            bin_ids=bin_ids,
            runtime_sec=float(time.perf_counter() - started),
            hr_npy=hr_path,
            convergence_csv=None,
            split_n=5,
            split_source="method_exact_drizzle_split_half",
            frc_source="method_exact_drizzle_phase_split",
        )
        row["coverage_npy"] = rel(cov_path)
        write_json(path, row)
        return row
    except Exception as exc:  # noqa: BLE001 - run matrix should continue.
        row = failed_record(spec, started, exc)
        write_json(path, row)
        return row


def tgv_spec_payload(spec: RunSpec, inputs: Inputs, args: argparse.Namespace) -> tuple[Path, Path]:
    idx = subset_for_spec(spec, inputs)
    shifts = shifts_for_spec(spec, inputs)[idx]
    shifts_path = args.output_dir / "tgv_specs" / f"{spec.run_id}_shifts.npy"
    spec_path = args.output_dir / "tgv_specs" / f"{spec.run_id}.json"
    result_path = args.output_dir / "tgv_results" / f"{spec.run_id}.json"
    hr_path = args.output_dir / "hr" / f"{spec.run_id}_hr.npy"
    convergence_csv = args.output_dir / "convergence" / f"{spec.run_id}_convergence.csv"
    shifts_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(shifts_path, shifts.astype(np.float32, copy=False))
    payload = {
        "run_id": spec.run_id,
        "indices": idx.astype(int).tolist(),
        "hp_frames_npy": str(inputs.hp_frames_npy.resolve()),
        "shifts_npy": str(shifts_path.resolve()),
        "hr_npy": str(hr_path.resolve()),
        "convergence_csv": str(convergence_csv.resolve()),
        "result_json": str(result_path.resolve()),
        "lambda_tv": float(args.tgv_lambda_tv),
        "psf_sigma": float(args.tgv_psf_sigma),
        "alpha_ratio": float(args.tgv_alpha_ratio),
        "max_iter": int(args.tgv_max_iter),
        "tgv_inner_iter": int(args.tgv_inner_iter),
        "aniso_ratio_y": float(args.tgv_aniso_ratio_y),
        "coverage_weighted": True,
        "workers": int(min(int(args.tgv_workers), 6)),
        "cuda_visible_devices": "",
    }
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return spec_path, result_path


def run_tgv_child(spec: RunSpec, inputs: Inputs, args: argparse.Namespace) -> dict[str, Any]:
    path = metric_path(args.output_dir, spec.run_id)
    if args.resume and path.exists():
        cached = read_json(path)
        if cached.get("status") == "success":
            return cached
    started = time.perf_counter()
    try:
        spec_path, result_path = tgv_spec_payload(spec, inputs, args)
        cmd = [
            str(args.conda_exe),
            "run",
            "-p",
            str(args.tgv_env),
            "python",
            str(SCRIPT_DIR / "run_tgv_child.py"),
            "--spec",
            str(spec_path),
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = ""
        env["OMP_NUM_THREADS"] = str(min(int(args.tgv_workers), 6))
        env.setdefault("OPENBLAS_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        proc = subprocess.run(
            cmd,
            cwd=ALGO_ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=float(args.tgv_timeout_sec),
            check=False,
        )
        child = read_json(result_path) if result_path.exists() else {
            "status": "failed",
            "error": "child result JSON missing",
            "stdout_tail": proc.stdout[-4000:],
            "stderr_tail": proc.stderr[-4000:],
        }
        if proc.returncode != 0 or child.get("status") != "success":
            raise RuntimeError(
                f"TGV child failed returncode={proc.returncode} error={child.get('error')} stderr={proc.stderr[-1000:]}"
            )

        idx = subset_for_spec(spec, inputs)
        shifts = shifts_for_spec(spec, inputs)[idx]
        hr = np.load(PROJECT_ROOT / child["hr_npy"] if not Path(child["hr_npy"]).is_absolute() else child["hr_npy"])
        row = evaluate_reconstruction(
            spec=spec,
            hr=hr,
            coverage=None,
            frames=inputs.hp_frames[idx],
            raw_frames=inputs.raw_frames[idx],
            shifts=shifts,
            bin_ids=inputs.phase_bins[idx],
            runtime_sec=float(time.perf_counter() - started),
            hr_npy=Path(child["hr_npy"]),
            convergence_csv=Path(child["convergence_csv"]),
            split_n=5,
            split_source="drizzle_proxy_same_subset_for_tgv_budget",
            frc_source="drizzle_proxy_same_subset_for_tgv_budget",
            tgv_child=child,
        )
        row["tgv_workers"] = int(min(args.tgv_workers, 6))
        row["tgv_lambda_tv"] = float(args.tgv_lambda_tv)
        row["tgv_psf_sigma"] = float(args.tgv_psf_sigma)
        row["tgv_alpha_ratio"] = float(args.tgv_alpha_ratio)
        row["tgv_inner_iter"] = int(args.tgv_inner_iter)
        row["tgv_max_iter"] = int(args.tgv_max_iter)
        write_json(path, row)
        return row
    except Exception as exc:  # noqa: BLE001
        row = failed_record(spec, started, exc)
        write_json(path, row)
        return row


def manifest_payload(args: argparse.Namespace, manifest: Manifest) -> dict[str, Any]:
    return {
        "task": "EP16 frame-budget shift-robustness alignment-source classical CPU methods",
        "created_or_updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "output_dir": args.output_dir,
        "cpu_resource_contract": {
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "tgv_parallel": int(args.tgv_parallel),
            "tgv_workers_per_run": int(min(args.tgv_workers, 6)),
            "max_total_physical_cores_requested": int(args.tgv_parallel) * int(min(args.tgv_workers, 6)),
        },
        "metric_definitions": {
            "reconstruction_domain": "highpass frames, sigma_bg=5.0 LR px",
            "raw_control_corr": "Pearson between run HR highpass and highpass(bicubic(nanmean(raw subset)), sigma_bg=5.0)",
            "drizzle_split_half": "EP06 split_half_consistency with STScI drizzle, n_splits=5",
            "tgv_split_frc": "CPU budget proxy: drizzle split/FRC on the same subset and shifts; full TGV HR still supplies artifact/raw-control/zigzag metrics",
            "frc": "phase-stratified split-half FRC with crop_lr_px=16, Tukey alpha=0.25, 1/7 cutoff",
            "zigzag": "EP15 M4 center zigzag profile probes copied on 2026-06-11",
        },
        "matrix_expected_rows": {
            "frame_budget": len([row for row in manifest.matrix_rows if row["experiment"] == "frame_budget"]),
            "shift_robustness": len([row for row in manifest.matrix_rows if row["experiment"] == "shift_robustness"]),
            "alignment_source": len([row for row in manifest.matrix_rows if row["experiment"] == "alignment_source"]),
        },
        "matrix_rows": manifest.matrix_rows,
        "runs": list(manifest.runs.values()),
    }


def write_manifest(args: argparse.Namespace, manifest: Manifest) -> None:
    write_json(args.output_dir / "run_manifest.json", manifest_payload(args, manifest))


def load_existing_run_records(output_dir: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    for path in sorted((output_dir / "runs").glob("*.json")):
        try:
            row = read_json(path)
        except json.JSONDecodeError:
            continue
        records[str(row.get("run_id", path.stem))] = row
    return records


def write_csvs(args: argparse.Namespace, matrices: dict[str, list[MatrixRow]], run_records: dict[str, dict[str, Any]]) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in matrices.items():
        out_rows: list[dict[str, Any]] = []
        for row in rows:
            metrics = run_records.get(row.run_id, {"run_id": row.run_id, "status": "missing"})
            merged = {
                "experiment": row.experiment,
                "row_role": row.row_role,
                "arm": row.arm,
                "n_frames": int(row.n_frames),
                "subset_seed": int(row.subset_seed),
                "shift_source": row.shift_source,
                "shift_noise_sigma_px": float(row.shift_noise_sigma_px),
                "shift_noise_seed": int(row.shift_noise_seed),
                **metrics,
            }
            out_rows.append(merged)
        pd.DataFrame(out_rows).to_csv(args.output_dir / f"{name}.csv", index=False)


def mean_sem(table: pd.DataFrame, group_cols: list[str], value_col: str) -> pd.DataFrame:
    ok = table[table["status"].eq("success") & np.isfinite(pd.to_numeric(table[value_col], errors="coerce"))].copy()
    if ok.empty:
        return pd.DataFrame(columns=[*group_cols, "mean", "std", "count", "sem"])
    ok[value_col] = pd.to_numeric(ok[value_col], errors="coerce")
    out = ok.groupby(group_cols, dropna=False)[value_col].agg(["mean", "std", "count"]).reset_index()
    out["std"] = out["std"].fillna(0.0)
    out["sem"] = out["std"] / np.sqrt(out["count"].clip(lower=1))
    return out


def plot_frame_budget(output_dir: Path) -> None:
    setup_academic_style()
    table = pd.read_csv(output_dir / "frame_budget.csv")
    summary = mean_sem(table, ["arm", "n_frames"], "raw_control_corr")
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_col"])
    colors = {"drizzle": "#4C72B0", "tgv": "#C44E52"}
    markers = {"drizzle": "o", "tgv": "s"}
    for arm, group in summary.groupby("arm", sort=False):
        group = group.sort_values("n_frames")
        ax.errorbar(
            group["n_frames"],
            group["mean"],
            yerr=group["sem"],
            marker=markers.get(arm, "o"),
            color=colors.get(arm, "#333333"),
            linewidth=1.2,
            capsize=2.5,
            label=arm,
        )
    ax.set_xlabel("Frame budget N")
    ax.set_ylabel("Raw-control correlation")
    ax.set_title("EP16 Frame Budget")
    ax.set_xticks(list(DEFAULT_BUDGETS))
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    savefig_academic(fig, output_dir / "fig_frame_budget.png")


def plot_shift_robustness(output_dir: Path) -> None:
    setup_academic_style()
    table = pd.read_csv(output_dir / "shift_robustness.csv")
    summary = mean_sem(table, ["arm", "shift_noise_sigma_px"], "raw_control_corr")
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_col"])
    colors = {"drizzle": "#4C72B0", "tgv": "#C44E52"}
    markers = {"drizzle": "o", "tgv": "s"}
    for arm, group in summary.groupby("arm", sort=False):
        group = group.sort_values("shift_noise_sigma_px")
        ax.errorbar(
            group["shift_noise_sigma_px"],
            group["mean"],
            yerr=group["sem"],
            marker=markers.get(arm, "o"),
            color=colors.get(arm, "#333333"),
            linewidth=1.2,
            capsize=2.5,
            label=arm,
        )
    ax.set_xlabel("Shift perturbation sigma [LR px]")
    ax.set_ylabel("Raw-control correlation")
    ax.set_title("EP16 Shift Robustness")
    ax.set_xticks([0.0, 0.05, 0.1, 0.2])
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    savefig_academic(fig, output_dir / "fig_shift_robustness.png")


def plot_alignment_source(output_dir: Path) -> None:
    setup_academic_style()
    table = pd.read_csv(output_dir / "alignment_source.csv")
    ok = table[table["status"].eq("success")].copy()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_col"])
    if not ok.empty:
        pivot = ok.pivot_table(index="shift_source", columns="arm", values="raw_control_corr", aggfunc="mean")
        x = np.arange(len(pivot.index))
        width = 0.34
        for i, arm in enumerate([c for c in ("drizzle", "tgv") if c in pivot.columns]):
            ax.bar(x + (i - 0.5) * width, pivot[arm], width=width, label=arm)
        ax.set_xticks(x)
        ax.set_xticklabels([str(v).replace("_", " ") for v in pivot.index], rotation=15, ha="right")
    ax.set_ylabel("Raw-control correlation")
    ax.set_title("EP16 Alignment Source")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="best", fontsize=7)
    savefig_academic(fig, output_dir / "fig_alignment_source.png")


def plot_paper_fig07(output_dir: Path) -> None:
    setup_academic_style()
    budget = mean_sem(pd.read_csv(output_dir / "frame_budget.csv"), ["arm", "n_frames"], "raw_control_corr")
    robust = mean_sem(pd.read_csv(output_dir / "shift_robustness.csv"), ["arm", "shift_noise_sigma_px"], "raw_control_corr")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.65), constrained_layout=True)
    colors = {"drizzle": "#4C72B0", "tgv": "#C44E52"}
    markers = {"drizzle": "o", "tgv": "s"}
    for arm, group in budget.groupby("arm", sort=False):
        group = group.sort_values("n_frames")
        axes[0].errorbar(group["n_frames"], group["mean"], yerr=group["sem"], marker=markers.get(arm, "o"), color=colors.get(arm), linewidth=1.2, capsize=2.5, label=arm)
    axes[0].set_xlabel("Frames N")
    axes[0].set_ylabel("Raw-control corr.")
    axes[0].set_title("Frame budget")
    axes[0].set_xticks(list(DEFAULT_BUDGETS))
    axes[0].grid(axis="y", alpha=0.25)
    for arm, group in robust.groupby("arm", sort=False):
        group = group.sort_values("shift_noise_sigma_px")
        axes[1].errorbar(group["shift_noise_sigma_px"], group["mean"], yerr=group["sem"], marker=markers.get(arm, "o"), color=colors.get(arm), linewidth=1.2, capsize=2.5, label=arm)
    axes[1].set_xlabel("Shift noise sigma [px]")
    axes[1].set_ylabel("Raw-control corr.")
    axes[1].set_title("Shift robustness")
    axes[1].set_xticks([0.0, 0.05, 0.1, 0.2])
    axes[1].grid(axis="y", alpha=0.25)
    axes[1].legend(loc="best", fontsize=7)
    PAPER_FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    savefig_academic(fig, PAPER_FIGURE_DIR / "fig07_budget_robustness.png", close=False)
    savefig_academic(fig, PAPER_FIGURE_DIR / "fig07_budget_robustness.pdf", close=True)


def write_figures(output_dir: Path) -> None:
    required = ["frame_budget.csv", "shift_robustness.csv", "alignment_source.csv"]
    if not all((output_dir / name).exists() for name in required):
        return
    plot_frame_budget(output_dir)
    plot_shift_robustness(output_dir)
    plot_alignment_source(output_dir)
    plot_paper_fig07(output_dir)


def run_all(args: argparse.Namespace) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    specs, matrices = build_matrices(args)
    manifest = Manifest()
    manifest.matrix_rows = [row.__dict__ for rows in matrices.values() for row in rows]
    manifest.runs.update(load_existing_run_records(args.output_dir))
    write_manifest(args, manifest)

    if args.summarize_only:
        write_csvs(args, matrices, manifest.runs)
        write_figures(args.output_dir)
        write_manifest(args, manifest)
        return

    inputs = prepare_inputs(args)

    drizzle_specs = [spec for spec in specs.values() if spec.arm == "drizzle"]
    if args.arms in {"drizzle", "both"}:
        for spec in tqdm(drizzle_specs, desc="EP16 drizzle runs"):
            row = run_drizzle_spec(spec, inputs, args)
            manifest.runs[spec.run_id] = row
            write_manifest(args, manifest)
            write_csvs(args, matrices, manifest.runs)

    tgv_specs = [spec for spec in specs.values() if spec.arm == "tgv"]
    if args.arms in {"tgv", "both"} and args.run_tgv and not args.skip_tgv:
        max_parallel = max(1, min(2, int(args.tgv_parallel)))
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            future_map = {executor.submit(run_tgv_child, spec, inputs, args): spec for spec in tgv_specs}
            for future in tqdm(as_completed(future_map), total=len(future_map), desc="EP16 TGV child runs"):
                spec = future_map[future]
                try:
                    row = future.result()
                except Exception as exc:  # noqa: BLE001
                    row = failed_record(spec, time.perf_counter(), exc)
                    write_json(metric_path(args.output_dir, spec.run_id), row)
                manifest.runs[spec.run_id] = row
                write_manifest(args, manifest)
                write_csvs(args, matrices, manifest.runs)

    write_csvs(args, matrices, manifest.runs)
    write_figures(args.output_dir)
    write_manifest(args, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--frame-audit-csv", type=Path, default=FRAME_AUDIT_CSV)
    parser.add_argument("--alignment-csv", type=Path, default=ALIGNMENT_CSV)
    parser.add_argument("--stage-config", type=Path, default=STAGE_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--arms", choices=["drizzle", "tgv", "both"], default="both")
    parser.add_argument("--run-tgv", action="store_true", help="Launch the TGV child queue.")
    parser.add_argument("--skip-tgv", action="store_true", help="Do not launch TGV even when --arms includes it.")
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--io-workers", type=int, default=4)
    parser.add_argument("--tgv-env", type=Path, default=TGV_ENV)
    parser.add_argument("--conda-exe", type=Path, default=DEFAULT_CONDA_EXE if DEFAULT_CONDA_EXE.exists() else Path("conda"))
    parser.add_argument("--tgv-parallel", type=int, default=2)
    parser.add_argument("--tgv-workers", type=int, default=6)
    parser.add_argument("--tgv-timeout-sec", type=float, default=7200.0)
    parser.add_argument("--tgv-lambda-tv", type=float, default=0.003)
    parser.add_argument("--tgv-psf-sigma", type=float, default=0.5)
    parser.add_argument("--tgv-alpha-ratio", type=float, default=2.0)
    parser.add_argument("--tgv-max-iter", type=int, default=100)
    parser.add_argument("--tgv-inner-iter", type=int, default=80)
    parser.add_argument("--tgv-aniso-ratio-y", type=float, default=1.5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    args.data_dir = args.data_dir.resolve()
    args.frame_audit_csv = args.frame_audit_csv.resolve()
    args.alignment_csv = args.alignment_csv.resolve()
    args.stage_config = args.stage_config.resolve()
    args.tgv_env = args.tgv_env.resolve()
    if args.conda_exe != Path("conda"):
        args.conda_exe = args.conda_exe.resolve()
    run_all(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
