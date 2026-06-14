#!/usr/bin/env python3
"""Unified paper harness for T1/T2 and F5.

This script is intentionally a thin orchestrator: it reuses the EP06 data
loader and metrics, EP10 drizzle, EP15 FRC/profile probes, and the EP07
real-data inference path that handles hybrid-drizzle and V10 residual-over-
observation checkpoints correctly.
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

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import ndimage
from scipy.signal import find_peaks

SCRIPT_PATH = Path(__file__).resolve()
ALGO_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[3]
EP06_SRC = PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src"
EP10_DRIZZLE_SRC = PROJECT_ROOT / "algos" / "ep10_drizzle" / "src"
EP15_SCRIPTS = PROJECT_ROOT / "algos" / "ep15_info_limit" / "scripts"
EP07_SRC = PROJECT_ROOT / "algos" / "ep07_unet_sr" / "src"
CORE_SRC = PROJECT_ROOT / "core" / "src"
for _path in (EP06_SRC, EP10_DRIZZLE_SRC, EP15_SCRIPTS, EP07_SRC, CORE_SRC):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from common.alignment import load_alignment_shifts  # noqa: E402
from common.data_loader import bicubic_upsample, highpass_preprocess, load_main_session_frames  # noqa: E402
from common.metrics import artifact_score  # noqa: E402
from ep10_drizzle.drizzle_sr import drizzle_reconstruct  # noqa: E402
from run_m2_frc import command_phase_bins, find_cutoff, frc_curve, stratified_split  # noqa: E402
from thermal_core.plotting import COLORMAPS, savefig_academic, setup_academic_style  # noqa: E402
from unet_sr.dataset import HYBRID_DRIZZLE_MEAN_CHANNEL  # noqa: E402
from unet_sr.inference import infer_from_burst  # noqa: E402
from unet_sr.model import ThermalSRUNet  # noqa: E402

EXPECTED_CLEAN_SR_FRAMES = 248
LR_SHAPE = (480, 640)
SCALE_2X = 2
SCALE_5X = 5
PIXEL_SIZE_UM = 10.0
HIGHPASS_SIGMA = 5.0
FRC_SEED = 42
FRC_PERIODS = (20.0, 16.0, 14.0, 12.0, 10.0)
FINE_ROW_FRAC = (384.0 / 960.0, 518.0 / 960.0)
FINE_COL_FRAC = (478.0 / 1280.0, 674.0 / 1280.0)


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    display_name: str
    family: str
    t1_include: bool
    t2_include: bool
    input_mode: str = ""
    anchor: str = ""
    checkpoint_run: str = ""
    checkpoint_step: int = 0
    output_grid_scale: int = SCALE_2X
    role: str = ""
    selection_note: str = ""


@dataclass
class Inputs:
    raw_frames: np.ndarray
    hp_frames: np.ndarray
    metadata: pd.DataFrame
    shifts: np.ndarray
    phase_bins_2x: np.ndarray
    raw_control_temp_2x: np.ndarray
    raw_control_hp_2x: np.ndarray
    raw_control_temp_5x: np.ndarray
    raw_control_hp_5x: np.ndarray


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fill_nan(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    fill = float(np.nanmedian(arr[finite])) if bool(finite.any()) else 0.0
    return np.where(finite, arr, fill).astype(np.float32, copy=False)


def pearson_finite(a: np.ndarray, b: np.ndarray) -> float:
    lhs = np.asarray(a, dtype=np.float64).ravel()
    rhs = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(lhs) & np.isfinite(rhs)
    if int(valid.sum()) < 2:
        return float("nan")
    return float(np.corrcoef(lhs[valid], rhs[valid])[0, 1])


def nrmse_pair(a: np.ndarray, b: np.ndarray) -> float:
    lhs = np.asarray(a, dtype=np.float32)
    rhs = np.asarray(b, dtype=np.float32)
    valid = np.isfinite(lhs) & np.isfinite(rhs)
    if int(valid.sum()) < 2:
        return float("nan")
    denom = float(np.nanstd(lhs[valid]) + np.nanstd(rhs[valid]))
    return float(np.sqrt(np.nanmean((lhs[valid] - rhs[valid]) ** 2)) / max(denom, 1e-12))


def fine_window(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    rows, cols = arr.shape
    y0 = int(round(rows * FINE_ROW_FRAC[0]))
    y1 = int(round(rows * FINE_ROW_FRAC[1]))
    x0 = int(round(cols * FINE_COL_FRAC[0]))
    x1 = int(round(cols * FINE_COL_FRAC[1]))
    return arr[y0:y1, x0:x1]


def lattice_score(image_hp: np.ndarray) -> float:
    x = np.asarray(fine_window(image_hp), dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).any():
        return float("nan")
    x = fill_nan(x).astype(np.float64, copy=False)
    x -= float(np.nanmean(x))
    power = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    rows, cols = x.shape
    fy = np.fft.fftshift(np.fft.fftfreq(rows))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(cols))[None, :]
    band = (np.abs(fy) > 0.35) | (np.abs(fx) > 0.35)
    return float(power[band].sum() / max(float(power.sum()), 1e-12))


def sharp_p95(image_temp: np.ndarray) -> float:
    crop = np.asarray(fine_window(image_temp), dtype=np.float64)
    if crop.size == 0 or not np.isfinite(crop).any():
        return float("nan")
    gy, gx = np.gradient(fill_nan(crop).astype(np.float64, copy=False))
    return float(np.nanpercentile(np.hypot(gy, gx), 95.0))


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


def sample_line(image: np.ndarray, y0: float, x0: float, y1: float, x1: float) -> np.ndarray:
    length = float(np.hypot(y1 - y0, x1 - x0))
    n = int(max(16, round(length) + 1))
    ys = np.linspace(y0, y1, n)
    xs = np.linspace(x0, x1, n)
    return ndimage.map_coordinates(fill_nan(image), [ys, xs], order=1, mode="nearest")


def line_signal_for_dark_trace(profile: np.ndarray) -> np.ndarray:
    baseline = float(np.percentile(profile, 90.0))
    return (baseline - np.asarray(profile, dtype=np.float32)).astype(np.float32, copy=False)


def profile_metrics(signal: np.ndarray, *, pitch_um: float, min_spacing_um: float = 8.0) -> dict[str, float | bool | int]:
    sig = ndimage.gaussian_filter1d(np.asarray(signal, dtype=np.float32), sigma=1.0, mode="nearest")
    base = float(np.percentile(sig, 10.0))
    height = float(np.max(sig) - base)
    if height <= 1e-8:
        return {"fwhm_um": float("nan"), "dip_depth": float("nan"), "lines_separated": False, "n_peaks": 0}
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
    return {"fwhm_um": fwhm_um, "dip_depth": dip_depth, "lines_separated": lines_separated, "n_peaks": int(peaks.size)}


def zigzag_profile_specs(image_shape: tuple[int, int]) -> list[dict[str, float | str]]:
    y0, y1, x0, x1 = crop_bounds_fraction(image_shape, fraction=1.0 / 6.0, y_frac=0.5, x_frac=0.5)
    roi_h = y1 - y0
    roi_w = x1 - x0
    return [
        {"profile_id": "zigzag_upper_left", "y0": y0 + 0.24 * roi_h, "x0": x0 + 0.02 * roi_w, "y1": y0 + 0.24 * roi_h, "x1": x0 + 0.44 * roi_w},
        {"profile_id": "zigzag_mid_left", "y0": y0 + 0.36 * roi_h, "x0": x0 + 0.03 * roi_w, "y1": y0 + 0.36 * roi_h, "x1": x0 + 0.46 * roi_w},
        {"profile_id": "zigzag_lower_left", "y0": y0 + 0.49 * roi_h, "x0": x0 + 0.04 * roi_w, "y1": y0 + 0.49 * roi_h, "x1": x0 + 0.48 * roi_w},
    ]


def summarize_zigzag(image_hp: np.ndarray, *, output_grid_scale: int) -> dict[str, Any]:
    pitch_um = PIXEL_SIZE_UM / float(output_grid_scale)
    rows: list[dict[str, Any]] = []
    for spec in zigzag_profile_specs(np.asarray(image_hp).shape):
        profile = sample_line(
            image_hp,
            float(spec["y0"]),
            float(spec["x0"]),
            float(spec["y1"]),
            float(spec["x1"]),
        )
        signal = line_signal_for_dark_trace(profile)
        rows.append({"profile_id": spec["profile_id"], **profile_metrics(signal, pitch_um=pitch_um)})
    table = pd.DataFrame(rows)
    return {
        "zigzag_fwhm_median_um": float(np.nanmedian(table["fwhm_um"].to_numpy(dtype=float))),
        "zigzag_dip_depth_median": float(np.nanmedian(table["dip_depth"].to_numpy(dtype=float))),
        "zigzag_profiles_separated": int(table["lines_separated"].astype(bool).sum()),
    }


def load_stage_config(path: Path) -> dict[str, float]:
    data = read_json(path)
    return {"theta_deg": float(data["theta_deg"]), "pixel_size_um": float(data["pixel_size_um"])}


def prepare_inputs(args: argparse.Namespace) -> Inputs:
    raw_frames, metadata = load_main_session_frames(
        args.data_dir,
        args.frame_audit_csv,
        workers=args.workers,
        dtype=np.float32,
    )
    if raw_frames.shape != (EXPECTED_CLEAN_SR_FRAMES, *LR_SHAPE):
        raise ValueError(f"Expected clean frame stack {(EXPECTED_CLEAN_SR_FRAMES, *LR_SHAPE)}, got {raw_frames.shape}")
    shifts = load_alignment_shifts("contour_refined", metadata=metadata).astype(np.float32, copy=False)
    stage = load_stage_config(args.stage_config)
    bins = command_phase_bins(metadata, scale=SCALE_2X, theta_deg=stage["theta_deg"], pixel_size_um=stage["pixel_size_um"])
    raw_mean = np.nanmean(raw_frames, axis=0)
    raw_control_temp_2x = bicubic_upsample(raw_mean, scale=SCALE_2X)
    raw_control_temp_5x = bicubic_upsample(raw_mean, scale=SCALE_5X)
    return Inputs(
        raw_frames=raw_frames,
        hp_frames=highpass_preprocess(raw_frames, sigma_bg=HIGHPASS_SIGMA, workers=args.workers),
        metadata=metadata,
        shifts=shifts,
        phase_bins_2x=bins,
        raw_control_temp_2x=raw_control_temp_2x,
        raw_control_hp_2x=highpass_preprocess(raw_control_temp_2x, sigma_bg=HIGHPASS_SIGMA),
        raw_control_temp_5x=raw_control_temp_5x,
        raw_control_hp_5x=highpass_preprocess(raw_control_temp_5x, sigma_bg=HIGHPASS_SIGMA),
    )


def checkpoint_path(spec: ArmSpec) -> Path:
    return PROJECT_ROOT / "algos" / "ep07_unet_sr" / "outputs" / spec.checkpoint_run / f"checkpoint_step_{spec.checkpoint_step:06d}.pt"


def load_checkpoint(spec: ArmSpec) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    ckpt_path = checkpoint_path(spec)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    config_path = ckpt_path.parent / "config.json"
    if config_path.exists():
        disk_cfg = read_json(config_path)
        merged = dict(disk_cfg)
        merged.update(cfg)
        cfg = merged
    return cfg, ckpt["model_state_dict"]


def infer_unet_cached(
    spec: ArmSpec,
    inputs: Inputs,
    *,
    cache_dir: Path,
    device: str,
    force: bool,
    tag: str,
    frame_indices: np.ndarray | None = None,
    overlap: int,
) -> tuple[np.ndarray, float, bool, dict[str, Any]]:
    suffix = f"{spec.arm_id}_{tag}_step{spec.checkpoint_step}_temperature.npy"
    path = cache_dir / suffix
    if path.exists() and not force:
        arr = np.load(path).astype(np.float32, copy=False)
        return arr, 0.0, True, {"cache_path": rel(path)}

    cfg, state_dict = load_checkpoint(spec)
    model_scale = 1 if spec.input_mode == "hybrid_drizzle2x" else int(cfg.get("scale", 2))
    model = ThermalSRUNet(
        in_channels=int(cfg["in_channels"]),
        out_channels=int(cfg.get("out_channels", 1)),
        base_channels=int(cfg["base_channels"]),
        scale=model_scale,
        hr_upsampler=str(cfg.get("hr_upsampler", "bilinear")),
        hr_res_blocks=int(cfg.get("hr_res_blocks", 0)),
    )
    model.load_state_dict(state_dict)
    model.eval()
    residual_mode = str(cfg.get("residual_mode", "none"))
    residual_channel = HYBRID_DRIZZLE_MEAN_CHANNEL if residual_mode == "drizzle2x" else None
    idx = np.arange(len(inputs.raw_frames)) if frame_indices is None else np.asarray(frame_indices, dtype=int)

    started = time.perf_counter()
    with torch.no_grad():
        pred = infer_from_burst(
            model,
            inputs.raw_frames[idx],
            inputs.shifts[idx],
            scale=SCALE_2X,
            patch_size_hr=int(cfg.get("patch_size_hr", 256)),
            overlap=int(overlap),
            device=device,
            residual=bool(cfg.get("residual", False)),
            sigma_bg=HIGHPASS_SIGMA,
            input_mode=spec.input_mode or str(cfg.get("input_mode", "lr")),
            residual_channel=residual_channel,
        ).astype(np.float32, copy=False)
    runtime = float(time.perf_counter() - started)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, pred)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred, runtime, False, {
        "cache_path": rel(path),
        "checkpoint": rel(checkpoint_path(spec)),
        "checkpoint_config": rel(checkpoint_path(spec).parent / "config.json"),
        "residual_mode": residual_mode,
        "residual_channel": residual_channel,
        "patch_size_hr": int(cfg.get("patch_size_hr", 256)),
    }


def tgv_highpass_to_temperature(tgv_hp: np.ndarray, raw_frames: np.ndarray) -> np.ndarray:
    ref_idx = len(raw_frames) // 2
    ref_temp_hr = bicubic_upsample(raw_frames[ref_idx], scale=SCALE_2X).astype(np.float32, copy=False)
    ref_hp_hr = highpass_preprocess(ref_temp_hr, sigma_bg=HIGHPASS_SIGMA)
    return (ref_temp_hr - ref_hp_hr + np.asarray(tgv_hp, dtype=np.float32)).astype(np.float32, copy=False)


def reconstruct_classical(spec: ArmSpec, inputs: Inputs, args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, float, dict[str, Any]]:
    started = time.perf_counter()
    if spec.arm_id == "bicubic":
        temp = inputs.raw_control_temp_2x
        hp = inputs.raw_control_hp_2x
        source = "bicubic(nanmean(raw_frames))"
    elif spec.arm_id == "drizzle":
        hp, coverage = drizzle_reconstruct(inputs.hp_frames, inputs.shifts, scale=SCALE_2X, pixfrac=0.7, kernel="square")
        temp, _ = drizzle_reconstruct(inputs.raw_frames, inputs.shifts, scale=SCALE_2X, pixfrac=0.7, kernel="square")
        source = "ep10_drizzle.drizzle_reconstruct scale=2 pixfrac=0.7 kernel=square"
        np.save(args.output_dir / "cache" / "drizzle_2x_highpass.npy", hp.astype(np.float32, copy=False))
        np.save(args.output_dir / "cache" / "drizzle_2x_temperature.npy", temp.astype(np.float32, copy=False))
        np.save(args.output_dir / "cache" / "drizzle_2x_coverage.npy", coverage.astype(np.float32, copy=False))
    elif spec.arm_id == "tgv":
        hp = np.load(args.tgv_highpass).astype(np.float32, copy=False)
        temp = tgv_highpass_to_temperature(hp, inputs.raw_frames)
        source = rel(args.tgv_highpass)
    elif spec.arm_id == "map_tv":
        hp = np.load(args.map_tv_highpass).astype(np.float32, copy=False)
        temp = np.load(args.map_tv_temperature).astype(np.float32, copy=False)
        source = f"{rel(args.map_tv_highpass)} + {rel(args.map_tv_temperature)}"
    else:
        raise ValueError(f"Unknown classical arm: {spec.arm_id}")
    return hp, temp, float(time.perf_counter() - started), {"source": source}


def reconstruct_for_split(spec: ArmSpec, inputs: Inputs, args: argparse.Namespace, idx: np.ndarray, tag: str) -> np.ndarray:
    if spec.family == "learned":
        temp, _runtime, _hit, _meta = infer_unet_cached(
            spec,
            inputs,
            cache_dir=args.output_dir / "cache",
            device=args.device,
            force=args.force,
            tag=tag,
            frame_indices=idx,
            overlap=args.overlap,
        )
        return highpass_preprocess(temp, sigma_bg=HIGHPASS_SIGMA)
    if spec.arm_id == "bicubic":
        temp = bicubic_upsample(np.nanmean(inputs.raw_frames[idx], axis=0), scale=SCALE_2X)
        return highpass_preprocess(temp, sigma_bg=HIGHPASS_SIGMA)
    if spec.arm_id == "drizzle":
        hp, _coverage = drizzle_reconstruct(inputs.hp_frames[idx], inputs.shifts[idx], scale=SCALE_2X, pixfrac=0.7, kernel="square")
        return hp
    raise ValueError(f"Split/FRC is not recomputed for precomputed arm {spec.arm_id}")


def interpolate_frc(curve: pd.DataFrame, period_um: float) -> float:
    freq = curve["frequency_um_inv"].to_numpy(dtype=float)
    values = curve["frc"].to_numpy(dtype=float)
    valid = (freq > 0) & np.isfinite(freq) & np.isfinite(values)
    if not bool(valid.any()):
        return float("nan")
    order = np.argsort(freq[valid])
    return float(np.interp(1.0 / period_um, freq[valid][order], values[valid][order], left=np.nan, right=np.nan))


def compute_split_frc(spec: ArmSpec, inputs: Inputs, args: argparse.Namespace) -> dict[str, Any]:
    if spec.arm_id == "map_tv":
        summary = read_json(args.map_tv_summary)
        parameter = pd.read_csv(args.map_tv_parameter_selection)
        selected = parameter[parameter["selected_global"].astype(bool)]
        row = selected.iloc[0] if not selected.empty else parameter.sort_values("selection_proxy").iloc[0]
        out = {
            "split_half_nrmse": float(row["split_half_nrmse"]),
            "split_half_source": rel(args.map_tv_parameter_selection),
            "frc_cutoff_period_um_1_7": float(summary["frc"]["map_tv_cutoff_period_um"]),
            "frc_source": rel(args.map_tv_summary),
        }
        for period in FRC_PERIODS:
            out[f"frc_{int(period)}um"] = float(summary["frc"][f"map_tv_frc_at_{int(period)}um"])
        return out

    if spec.arm_id == "tgv" and args.tgv_proxy_csv.exists():
        table = pd.read_csv(args.tgv_proxy_csv)
        rows = table[(table["arm"].eq("tgv")) & (table["n_frames"].eq(EXPECTED_CLEAN_SR_FRAMES)) & (table["status"].eq("success"))]
        if not rows.empty:
            row = rows.iloc[0]
            out = {
                "split_half_nrmse": float(row.get("split_half_nrmse_median", np.nan)),
                "split_half_source": f"{rel(args.tgv_proxy_csv)} (drizzle proxy on same subset/shifts; not actual TGV split)",
                "frc_cutoff_period_um_1_7": float(row.get("frc_cutoff_period_um_1_7", np.nan)),
                "frc_source": f"{rel(args.tgv_proxy_csv)} (drizzle proxy on same subset/shifts; not actual TGV split)",
            }
            for period in FRC_PERIODS:
                out[f"frc_{int(period)}um"] = float(row.get(f"frc_{int(period)}um", np.nan))
            return out

    if spec.arm_id == "tgv":
        return {
            "split_half_nrmse": float("nan"),
            "split_half_source": "not_recomputed_for_precomputed_tgv_full_hr",
            "frc_cutoff_period_um_1_7": float("nan"),
            "frc_source": "not_recomputed_for_precomputed_tgv_full_hr",
            **{f"frc_{int(period)}um": float("nan") for period in FRC_PERIODS},
        }

    a_idx, b_idx, balance = stratified_split(inputs.phase_bins_2x, scale=SCALE_2X, seed=FRC_SEED)
    hp_a = reconstruct_for_split(spec, inputs, args, a_idx, f"frc_seed{FRC_SEED}_a")
    hp_b = reconstruct_for_split(spec, inputs, args, b_idx, f"frc_seed{FRC_SEED}_b")
    curve = frc_curve(fill_nan(hp_a), fill_nan(hp_b), scale=spec.output_grid_scale, crop_lr_px=16, tukey_alpha=0.25)
    cutoff = find_cutoff(curve, "threshold_1_7")
    curve_path = args.output_dir / "frc_curves" / f"{spec.arm_id}_frc_curve.csv"
    curve_path.parent.mkdir(parents=True, exist_ok=True)
    curve.to_csv(curve_path, index=False)
    out = {
        "split_half_nrmse": nrmse_pair(hp_a, hp_b),
        "split_half_source": f"phase-stratified split seed={FRC_SEED}, max_bin_abs_diff={int(balance['abs_diff'].max())}",
        "frc_cutoff_period_um_1_7": float(cutoff.period_um),
        "frc_source": rel(curve_path),
    }
    for period in FRC_PERIODS:
        out[f"frc_{int(period)}um"] = interpolate_frc(curve, period)
    return out


def raw_control_for_scale(inputs: Inputs, scale: int) -> np.ndarray:
    if int(scale) == SCALE_2X:
        return inputs.raw_control_hp_2x
    if int(scale) == SCALE_5X:
        return inputs.raw_control_hp_5x
    raise ValueError(f"Unsupported output scale: {scale}")


def evaluate_arm(spec: ArmSpec, inputs: Inputs, args: argparse.Namespace) -> dict[str, Any]:
    print(f"[harness] evaluating {spec.arm_id}")
    row: dict[str, Any] = {
        "arm_id": spec.arm_id,
        "display_name": spec.display_name,
        "family": spec.family,
        "input_mode": spec.input_mode,
        "anchor": spec.anchor,
        "role": spec.role,
        "selection_note": spec.selection_note,
        "checkpoint_run": spec.checkpoint_run,
        "checkpoint_step": spec.checkpoint_step if spec.checkpoint_step else np.nan,
        "output_grid_scale": int(spec.output_grid_scale),
        "t1_include": bool(spec.t1_include),
        "t2_include": bool(spec.t2_include),
        "status": "success",
    }
    try:
        if spec.family == "learned":
            temp, runtime, cache_hit, meta = infer_unet_cached(
                spec,
                inputs,
                cache_dir=args.output_dir / "cache",
                device=args.device,
                force=args.force,
                tag="full",
                frame_indices=None,
                overlap=args.overlap,
            )
            hp = highpass_preprocess(temp, sigma_bg=HIGHPASS_SIGMA)
            row.update(meta)
            row["full_inference_runtime_sec"] = runtime
            row["full_cache_hit"] = bool(cache_hit)
            row["temperature_mean_c"] = float(np.nanmean(temp))
            row["temperature_min_c"] = float(np.nanmin(temp))
            row["temperature_max_c"] = float(np.nanmax(temp))
            sanity_required = spec.input_mode == "hybrid_drizzle2x" or spec.arm_id.startswith("v10")
            row["temperature_sanity_required"] = bool(sanity_required)
            row["temperature_sanity_pass"] = bool((not sanity_required) or (10.0 <= row["temperature_mean_c"] <= 35.0))
            if sanity_required and not row["temperature_sanity_pass"]:
                raise RuntimeError(f"{spec.arm_id} temperature sanity failed: mean={row['temperature_mean_c']:.3f} C")
        else:
            hp, temp, runtime, meta = reconstruct_classical(spec, inputs, args)
            row.update(meta)
            row["full_inference_runtime_sec"] = runtime
            row["full_cache_hit"] = False
            row["temperature_mean_c"] = float(np.nanmean(temp))
            row["temperature_min_c"] = float(np.nanmin(temp))
            row["temperature_max_c"] = float(np.nanmax(temp))
            row["temperature_sanity_required"] = False
            row["temperature_sanity_pass"] = True

        row["artifact_score"] = float(artifact_score(hp, scale=int(spec.output_grid_scale)))
        row["raw_control_corr"] = pearson_finite(hp, raw_control_for_scale(inputs, spec.output_grid_scale))
        row["lattice_score"] = lattice_score(hp)
        row["sharp_p95"] = sharp_p95(temp)
        row.update(summarize_zigzag(hp, output_grid_scale=int(spec.output_grid_scale)))
        row.update(compute_split_frc(spec, inputs, args))

        hp_path = args.output_dir / "hr" / f"{spec.arm_id}_highpass.npy"
        temp_path = args.output_dir / "hr" / f"{spec.arm_id}_temperature.npy"
        hp_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(hp_path, hp.astype(np.float32, copy=False))
        np.save(temp_path, temp.astype(np.float32, copy=False))
        row["highpass_npy"] = rel(hp_path)
        row["temperature_npy"] = rel(temp_path)
    except Exception as exc:  # noqa: BLE001 - keep table materialized for audit.
        row["status"] = "failed"
        row["error"] = repr(exc)
        print(f"[harness] FAILED {spec.arm_id}: {exc}", file=sys.stderr)
    return row


def build_arm_specs() -> list[ArmSpec]:
    return [
        ArmSpec("bicubic", "Bicubic raw mean", "classical", True, False, role="classical baseline", output_grid_scale=2),
        ArmSpec("drizzle", "Drizzle 2x", "classical", True, False, role="observation-injected classical baseline", output_grid_scale=2),
        ArmSpec("map_tv", "MAP-TV anchor", "classical", True, False, role="EP15 deconvolution anchor", output_grid_scale=5),
        ArmSpec("tgv", "Anisotropic TGV", "classical", True, False, role="EP10 TGV anchor", output_grid_scale=2),
        ArmSpec("v6", "V6 hot+full anchor", "learned", True, False, "lr", "full 0.1 (legacy hot)", "ep07_v6_physics", 8000, role="1x-input canonical", selection_note="EP11 canonical normalized-ideal checkpoint"),
        ArmSpec("v8_1a", "V8.1a no anchor", "learned", True, True, "lr", "none", "ep07_v8_1a_loss_cooldown", 15000, role="1x input x none", selection_note="EP11 canonical normalized-ideal checkpoint"),
        ArmSpec("v8_1b", "V8.1b PixelShuffle", "learned", False, True, "lr", "none / PixelShuffle head", "ep07_v8_1b_pixelshuffle", 5000, role="failed-head control", selection_note="EP11 canonical negative-control checkpoint"),
        ArmSpec("v9b", "V9B band anchor", "learned", True, True, "lr", "band-limited 0.1", "ep07_v9b_fwd_consistency", 11000, role="1x input x band anchor", selection_note="EP11 canonical normalized-ideal checkpoint"),
        ArmSpec("v9d", "V9D full anchor", "learned", True, True, "lr", "full-band 0.1", "ep07_v9d_fwd_fullband", 7000, role="1x input x full anchor", selection_note="normalized-ideal checkpoint over completed TB trajectory"),
        ArmSpec("v9a", "V9A hybrid", "learned", True, True, "hybrid_drizzle2x", "none", "ep07_v9a_hybrid_drizzle", 10000, role="hybrid input x none", selection_note="10K selected by 10K-25K Pareto/fidelity gate"),
        ArmSpec("v9c", "V9C hybrid legal anchor", "learned", True, True, "hybrid_drizzle2x", "legal 1x band 0.1", "ep07_v9c_hybrid_legal_fwd", 5000, role="hybrid input x legal anchor", selection_note="normalized-ideal checkpoint over completed TB trajectory"),
        ArmSpec("v10_lam120_15k", "V10 residual-over-obs lam120@15K", "learned", True, True, "hybrid_drizzle2x", "residual-over-observation lambda=1.2", "ep07_v10_resid_hl_lam120", 15000, role="residual-over-observation working point", selection_note="ACL-020 high-lambda sweep best trade-off"),
        ArmSpec("v9a_late_60k", "V9A late 60K", "learned", False, False, "hybrid_drizzle2x", "none", "ep07_v9a_hybrid_drizzle", 60000, role="F5 late-drift visual control", selection_note="visual failure-mode endpoint for F5"),
    ]


def write_metric_tables(rows: list[dict[str, Any]], output_dir: Path) -> tuple[Path, Path, Path]:
    df = pd.DataFrame(rows)
    all_path = output_dir / "all_arm_metrics.csv"
    t1_path = output_dir / "t1_metrics.csv"
    t2_path = output_dir / "t2_metrics.csv"
    df.to_csv(all_path, index=False)
    df[df["t1_include"].astype(bool)].to_csv(t1_path, index=False)
    df[df["t2_include"].astype(bool)].to_csv(t2_path, index=False)
    return all_path, t1_path, t2_path


def save_f5(output_dir: Path, paper_dir: Path) -> tuple[Path, Path]:
    arrays: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for arm_id, label in [
        ("drizzle", "Drizzle"),
        ("tgv", "TGV"),
        ("v9a_late_60k", "V9A 60K"),
        ("v10_lam120_15k", "V10 lam120 15K"),
    ]:
        temp = np.load(output_dir / "hr" / f"{arm_id}_temperature.npy").astype(np.float32, copy=False)
        hp = np.load(output_dir / "hr" / f"{arm_id}_highpass.npy").astype(np.float32, copy=False)
        arrays[label] = (fine_window(temp), fine_window(hp))

    setup_academic_style()
    labels = list(arrays)
    temp_crops = [arrays[label][0] for label in labels]
    hp_crops = [arrays[label][1] for label in labels]
    temp_values = np.concatenate([crop[np.isfinite(crop)].ravel() for crop in temp_crops if np.isfinite(crop).any()])
    temp_vmin, temp_vmax = float(np.percentile(temp_values, 1.0)), float(np.percentile(temp_values, 99.0))
    hp_vmax = max(float(np.percentile(np.abs(np.concatenate([crop[np.isfinite(crop)].ravel() for crop in hp_crops])), 99.0)), 1e-6)

    fig, axes = plt.subplots(2, len(labels), figsize=(7.2, 3.05), constrained_layout=True)
    for col, label in enumerate(labels):
        im0 = axes[0, col].imshow(fill_nan(arrays[label][0]), cmap=COLORMAPS["temperature"], vmin=temp_vmin, vmax=temp_vmax, interpolation="nearest")
        im1 = axes[1, col].imshow(fill_nan(arrays[label][1]), cmap=COLORMAPS["residual_diff"], vmin=-hp_vmax, vmax=hp_vmax, interpolation="nearest")
        axes[0, col].set_title(label)
        for row in (0, 1):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    axes[0, 0].set_ylabel("Temp.")
    axes[1, 0].set_ylabel("Highpass")
    cbar0 = fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.025, pad=0.01)
    cbar0.set_label("deg C")
    cbar1 = fig.colorbar(im1, ax=axes[1, :].tolist(), fraction=0.025, pad=0.01)
    cbar1.set_label("deg C")

    paper_dir.mkdir(parents=True, exist_ok=True)
    png = paper_dir / "fig05_main_visual.png"
    pdf = paper_dir / "fig05_main_visual.pdf"
    savefig_academic(fig, png)

    # Recreate for PDF because savefig_academic closes by default.
    fig, axes = plt.subplots(2, len(labels), figsize=(7.2, 3.05), constrained_layout=True)
    for col, label in enumerate(labels):
        im0 = axes[0, col].imshow(fill_nan(arrays[label][0]), cmap=COLORMAPS["temperature"], vmin=temp_vmin, vmax=temp_vmax, interpolation="nearest")
        im1 = axes[1, col].imshow(fill_nan(arrays[label][1]), cmap=COLORMAPS["residual_diff"], vmin=-hp_vmax, vmax=hp_vmax, interpolation="nearest")
        axes[0, col].set_title(label)
        for row in (0, 1):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    axes[0, 0].set_ylabel("Temp.")
    axes[1, 0].set_ylabel("Highpass")
    fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.025, pad=0.01).set_label("deg C")
    fig.colorbar(im1, ax=axes[1, :].tolist(), fraction=0.025, pad=0.01).set_label("deg C")
    savefig_academic(fig, pdf)
    return png, pdf


def tb_vs_harness(rows: list[dict[str, Any]], output_dir: Path) -> Path:
    tb_csv = PROJECT_ROOT / "output" / "ep11_dl_benchmark" / "checkpoint_selection" / "checkpoint_metrics.csv"
    out_path = output_dir / "tb_vs_harness_scale_check.csv"
    columns = [
        "arm_id",
        "step",
        "tb_artifact_score",
        "harness_artifact_score",
        "tb_raw_control_corr",
        "harness_raw_control_corr",
        "note",
    ]
    if not tb_csv.exists():
        pd.DataFrame(columns=columns).to_csv(out_path, index=False)
        return out_path
    tb = pd.read_csv(tb_csv)
    harness = pd.DataFrame(rows)
    records: list[dict[str, Any]] = []
    for arm_id, tb_arm in [("v9b", "v9b"), ("v8_1a", "v8.1a"), ("v6", "v6")]:
        h = harness[harness["arm_id"].eq(arm_id)]
        if h.empty:
            continue
        step = int(h.iloc[0]["checkpoint_step"])
        t = tb[(tb["arm"].eq(tb_arm)) & (tb["step"].eq(step))]
        if t.empty:
            continue
        records.append(
            {
                "arm_id": arm_id,
                "step": step,
                "tb_artifact_score": float(t.iloc[0]["artifact_score"]),
                "harness_artifact_score": float(h.iloc[0]["artifact_score"]),
                "tb_raw_control_corr": float(t.iloc[0]["raw_control_corr"]),
                "harness_raw_control_corr": float(h.iloc[0]["raw_control_corr"]),
                "note": "TB-scale real_eval artifact differs from EP11/common.metrics harness artifact; do not mix in one table.",
            }
        )
    pd.DataFrame(records, columns=columns).to_csv(out_path, index=False)
    return out_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "data_raw" / "infrared_avi")
    parser.add_argument("--frame-audit-csv", type=Path, default=PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--stage-config", type=Path, default=PROJECT_ROOT / "configs" / "stage_calibration.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "ep11_unified_harness")
    parser.add_argument("--paper-figure-dir", type=Path, default=PROJECT_ROOT / "output" / "paper_figures")
    parser.add_argument("--tgv-highpass", type=Path, default=PROJECT_ROOT / "output" / "ep10_tgv_sr" / "best_hr_highpass.npy")
    parser.add_argument("--tgv-proxy-csv", type=Path, default=PROJECT_ROOT / "output" / "ep16_budget_robustness" / "frame_budget.csv")
    parser.add_argument("--map-tv-highpass", type=Path, default=PROJECT_ROOT / "output" / "ep15_info_limit" / "m4_deconv_anchor" / "map-tv_highpass.npy")
    parser.add_argument("--map-tv-temperature", type=Path, default=PROJECT_ROOT / "output" / "ep15_info_limit" / "m4_deconv_anchor" / "map-tv_temperature.npy")
    parser.add_argument("--map-tv-summary", type=Path, default=PROJECT_ROOT / "output" / "ep15_info_limit" / "m4_deconv_anchor" / "m4_summary.json")
    parser.add_argument("--map-tv-parameter-selection", type=Path, default=PROJECT_ROOT / "output" / "ep15_info_limit" / "m4_deconv_anchor" / "parameter_selection.csv")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--only", nargs="*", default=None, help="Optional subset of arm ids for smoke/debug.")
    parser.add_argument("--skip-f5", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "cache").mkdir(parents=True, exist_ok=True)
    setup_academic_style()

    print(f"[harness] output_dir={rel(args.output_dir)}")
    print(f"[harness] device={args.device}, workers={args.workers}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '<unset>')}")
    started = time.perf_counter()
    inputs = prepare_inputs(args)
    specs = build_arm_specs()
    if args.only:
        wanted = set(args.only)
        specs = [spec for spec in specs if spec.arm_id in wanted]
    rows = [evaluate_arm(spec, inputs, args) for spec in specs]
    all_path, t1_path, t2_path = write_metric_tables(rows, args.output_dir)
    scale_check_path = tb_vs_harness(rows, args.output_dir)
    f5_paths: tuple[Path, Path] | None = None
    if not args.skip_f5 and {"drizzle", "tgv", "v9a_late_60k", "v10_lam120_15k"}.issubset({row["arm_id"] for row in rows if row.get("status") == "success"}):
        f5_paths = save_f5(args.output_dir, args.paper_figure_dir)

    manifest = {
        "task": "Task D unified paper harness for T1/T2/F5",
        "created_or_updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "elapsed_sec": float(time.perf_counter() - started),
        "metric_scale": "single unified EP11/common.metrics harness scale for T1/T2; TB-scale comparison stored separately",
        "clean_frame_contract": "248 clean main-session frames only",
        "frame_shape": list(inputs.raw_frames.shape),
        "alignment_method": "contour_refined",
        "highpass_sigma": HIGHPASS_SIGMA,
        "frc_seed": FRC_SEED,
        "device_requested": args.device,
        "workers": int(args.workers),
        "outputs": {
            "all_arm_metrics": all_path,
            "t1_metrics": t1_path,
            "t2_metrics": t2_path,
            "tb_vs_harness_scale_check": scale_check_path,
            "fig05_png": f5_paths[0] if f5_paths else "",
            "fig05_pdf": f5_paths[1] if f5_paths else "",
        },
        "known_metric_boundaries": {
            "map_tv_grid": "MAP-TV is a precomputed 5x EP15 anchor; output_grid_scale is explicit.",
            "tgv_split_frc": "TGV split/FRC columns reuse EP16 drizzle proxy on identical subset/shifts unless TGV split recomputation is added later; source columns identify this.",
            "visual_preference": "F5 is a dual-domain task-level visual gate, not fidelity or resolution evidence.",
        },
        "arms": rows,
    }
    write_json(args.output_dir / "run_manifest.json", manifest)
    print(f"[harness] wrote {rel(t1_path)}")
    print(f"[harness] wrote {rel(t2_path)}")
    if f5_paths:
        print(f"[harness] wrote {rel(f5_paths[0])} and {rel(f5_paths[1])}")
    failed = [row for row in rows if row.get("status") != "success"]
    if failed:
        print(f"[harness] failed arms: {[row['arm_id'] for row in failed]}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
