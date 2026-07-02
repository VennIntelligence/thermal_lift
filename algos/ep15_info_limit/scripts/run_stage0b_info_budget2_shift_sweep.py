#!/usr/bin/env python3
"""Stage 0b synthetic information-budget rerun with shift-error sweep.

This is a reproducible replacement scaffold for the historical
``info_budget2.py`` artifact referenced from the solver-v2 redesign notes.  It
keeps the old experiment shape -- synthetic HR truth, physical LR burst, and
classical oracle reconstructions -- but makes the currently load-bearing
variables explicit:

* detector pitch is read from ``configs/stage_calibration.json`` and defaults to
  the corrected 20 um/LR-pixel value;
* PSF sigma is an input in LR-pixel units;
* reconstruction shifts can be perturbed by an LR-pixel Gaussian error grid.

The outputs are intended for Stage 1a domain-randomization calibration, not for
claiming optical ground truth resolution.
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

import numpy as np
import pandas as pd
from scipy import ndimage
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
ALGO_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[3]
EP06_SRC = PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src"

for path in (EP06_SRC, PROJECT_ROOT / "core" / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from common.alignment import load_alignment_shifts  # noqa: E402
from common.data_loader import load_main_session_metadata  # noqa: E402
from thermal_core.alignment_paths import default_contour_alignment_csv  # noqa: E402
from thermal_core.displacement import coordinate_to_shift  # noqa: E402


EXPECTED_CLEAN_SR_FRAMES = 248
DEFAULT_COORD_VALUES_UM = (0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40)
DEFAULT_SHIFT_ERROR_GRID = (0.0, 0.05, 0.10, 0.20)
DEFAULT_FRAME_BUDGETS = (16, 64, 144, 248)
DEFAULT_PSF_SIGMA_LR_PX = 0.22572150008846692
DEFAULT_NOISE_SIGMA_C = 0.0724


@dataclass(frozen=True)
class StageConfig:
    theta_deg: float
    detector_pitch_um: float
    spatial_resolution_um: float


@dataclass(frozen=True)
class ShiftSet:
    shifts_lr_px: np.ndarray
    source: str
    note: str


@dataclass(frozen=True)
class ReconBundle:
    single_frame_wiener: np.ndarray
    aligned_mean_wiener: np.ndarray
    drizzle_wiener: np.ndarray
    aligned_mean: np.ndarray
    drizzle: np.ndarray
    zero_coverage_pct: float


@dataclass(frozen=True)
class AliasOracleBundle:
    alias_single_wiener: np.ndarray
    alias_multiframe_wiener: np.ndarray


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
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
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one float value")
    return values


def parse_int_list(text: str) -> list[int]:
    values = [int(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one integer value")
    return values


def load_stage_config(path: Path) -> StageConfig:
    data = read_json(path)
    return StageConfig(
        theta_deg=float(data["theta_deg"]),
        detector_pitch_um=float(data["pixel_size_um"]),
        spatial_resolution_um=float(data.get("current_spatial_resolution_um", data["pixel_size_um"])),
    )


def load_default_psf_sigma(path: Path) -> tuple[float, str]:
    if not path.exists():
        return DEFAULT_PSF_SIGMA_LR_PX, "built_in_default"
    data = read_json(path)
    if "psf_sigma_lr_px" not in data:
        return DEFAULT_PSF_SIGMA_LR_PX, f"{_rel(path)} missing psf_sigma_lr_px"
    return float(data["psf_sigma_lr_px"]), _rel(path)


def reference_file_from_alignment(path: Path) -> str | None:
    if not path.exists():
        return None
    table = pd.read_csv(path, usecols=lambda col: col in {"reference_file"})
    if "reference_file" not in table or table["reference_file"].dropna().empty:
        return None
    return str(table["reference_file"].dropna().iloc[0])


def command_shifts_from_metadata(
    metadata: pd.DataFrame,
    *,
    reference_file: str | None,
    stage: StageConfig,
) -> np.ndarray:
    if not {"X", "Y"}.issubset(metadata.columns):
        raise ValueError("metadata must contain X and Y columns for command-prior shifts")
    if reference_file is not None and "file" in metadata.columns and metadata["file"].astype(str).eq(reference_file).any():
        ref = metadata.loc[metadata["file"].astype(str).eq(reference_file)].iloc[0]
    else:
        ref = metadata.iloc[0]
    rel_x = metadata["X"].to_numpy(dtype=float) - float(ref["X"])
    rel_y = metadata["Y"].to_numpy(dtype=float) - float(ref["Y"])
    dx, dy = coordinate_to_shift(
        rel_x,
        rel_y,
        theta_deg=stage.theta_deg,
        pixel_size_um=stage.detector_pitch_um,
    )
    return np.column_stack([dx, dy]).astype(np.float32, copy=False)


def synthetic_command_shift_lattice(stage: StageConfig, *, limit: int | None) -> np.ndarray:
    coords = [(float(x), float(y)) for y in DEFAULT_COORD_VALUES_UM for x in DEFAULT_COORD_VALUES_UM]
    x_um = np.asarray([x for x, _y in coords], dtype=float)
    y_um = np.asarray([y for _x, y in coords], dtype=float)
    x_um = x_um - float(x_um[0])
    y_um = y_um - float(y_um[0])
    dx, dy = coordinate_to_shift(
        x_um,
        y_um,
        theta_deg=stage.theta_deg,
        pixel_size_um=stage.detector_pitch_um,
    )
    shifts = np.column_stack([dx, dy]).astype(np.float32, copy=False)
    if limit is not None:
        shifts = shifts[: int(limit)]
    return shifts


def load_shift_set(args: argparse.Namespace, stage: StageConfig) -> ShiftSet:
    source = str(args.shift_source)
    attempts: list[str] = []

    def try_contour() -> ShiftSet | None:
        try:
            metadata = load_main_session_metadata(args.frame_audit_csv)
            shifts = load_alignment_shifts("contour_refined", metadata=metadata, alignment_csv=args.alignment_csv)
            if shifts.shape[0] != len(metadata):
                raise ValueError(f"shift count {shifts.shape[0]} does not match metadata {len(metadata)}")
            return ShiftSet(
                shifts_lr_px=shifts.astype(np.float32, copy=False),
                source="contour_refined",
                note=f"Loaded {len(shifts)} EP05 contour-refined shifts from {_rel(args.alignment_csv)}",
            )
        except Exception as exc:  # noqa: BLE001 - fallback path records the failure.
            attempts.append(f"contour_refined failed: {type(exc).__name__}: {exc}")
            return None

    def try_command_from_metadata() -> ShiftSet | None:
        try:
            metadata = load_main_session_metadata(args.frame_audit_csv)
            ref_file = reference_file_from_alignment(args.alignment_csv)
            shifts = command_shifts_from_metadata(metadata, reference_file=ref_file, stage=stage)
            return ShiftSet(
                shifts_lr_px=shifts,
                source="command_prior_metadata",
                note=f"Built {len(shifts)} command-prior shifts from {_rel(args.frame_audit_csv)}",
            )
        except Exception as exc:  # noqa: BLE001 - fallback path records the failure.
            attempts.append(f"command_prior_metadata failed: {type(exc).__name__}: {exc}")
            return None

    if source in {"auto", "contour_refined"}:
        result = try_contour()
        if result is not None:
            return result
        if source == "contour_refined":
            raise RuntimeError("; ".join(attempts))

    if source in {"auto", "command_prior"}:
        result = try_command_from_metadata()
        if result is not None:
            return result
        if source == "command_prior":
            raise RuntimeError("; ".join(attempts))

    shifts = synthetic_command_shift_lattice(stage, limit=args.synthetic_shift_limit)
    note = (
        f"Generated {len(shifts)} synthetic command-lattice shifts from AGENTS coordinate set; "
        + ("; ".join(attempts) if attempts else "no metadata/alignment read attempted")
    )
    return ShiftSet(shifts_lr_px=shifts, source="synthetic_command_lattice", note=note)


def gaussian_mtf_lr(frequency_cyc_per_lr_px: float, sigma_lr_px: float) -> float:
    f = float(frequency_cyc_per_lr_px)
    sigma = float(sigma_lr_px)
    return float(math.exp(-2.0 * math.pi**2 * sigma * sigma * f * f))


def detector_box_mtf(frequency_um_inv: float, aperture_um: float) -> float:
    return float(abs(np.sinc(float(aperture_um) * float(frequency_um_inv))))


def make_truth(scene_seed: int, *, lr_size: int, scale: int, delta_temp_c: float) -> np.ndarray:
    rng = np.random.default_rng(scene_seed)
    hr_size = int(lr_size) * int(scale)
    yy, xx = np.mgrid[0:hr_size, 0:hr_size].astype(np.float32)
    y = yy / max(hr_size - 1, 1)
    x = xx / max(hr_size - 1, 1)

    field = 0.18 * x + 0.10 * y
    coarse = rng.normal(0.0, 1.0, size=(hr_size, hr_size)).astype(np.float32)
    field += 0.18 * ndimage.gaussian_filter(coarse, sigma=hr_size / 18.0, mode="reflect")

    for _ in range(14):
        cx = rng.uniform(0.15, 0.85)
        cy = rng.uniform(0.15, 0.85)
        half_w = rng.uniform(0.015, 0.090)
        half_h = rng.uniform(0.010, 0.070)
        amp = rng.uniform(-0.75, 1.0)
        mask = (np.abs(x - cx) <= half_w) & (np.abs(y - cy) <= half_h)
        field[mask] += amp

    for _ in range(8):
        angle = rng.uniform(0.0, math.pi)
        cx = rng.uniform(0.15, 0.85)
        cy = rng.uniform(0.15, 0.85)
        width = rng.uniform(0.004, 0.015)
        length = rng.uniform(0.12, 0.50)
        xr = (x - cx) * math.cos(angle) + (y - cy) * math.sin(angle)
        yr = -(x - cx) * math.sin(angle) + (y - cy) * math.cos(angle)
        mask = (np.abs(yr) <= width) & (np.abs(xr) <= length / 2.0)
        field[mask] += rng.uniform(-0.8, 0.8)

    field = ndimage.gaussian_filter(field.astype(np.float32), sigma=0.35, mode="reflect")
    p01, p99 = np.percentile(field, [1.0, 99.0])
    field = np.clip((field - p01) / max(p99 - p01, 1e-6), 0.0, 1.0)
    return (float(delta_temp_c) * field).astype(np.float32, copy=False)


def block_mean_downsample(image: np.ndarray, *, scale: int) -> np.ndarray:
    rows = image.shape[0] // scale
    cols = image.shape[1] // scale
    cropped = image[: rows * scale, : cols * scale]
    return cropped.reshape(rows, scale, cols, scale).mean(axis=(1, 3)).astype(np.float32, copy=False)


def render_burst(
    truth_hr: np.ndarray,
    true_shifts: np.ndarray,
    *,
    scale: int,
    psf_sigma_lr_px: float,
    noise_sigma_c: float,
    rng: np.random.Generator,
) -> np.ndarray:
    sigma_hr = max(0.0, float(psf_sigma_lr_px) * int(scale))
    frames: list[np.ndarray] = []
    for dx_px, dy_px in true_shifts:
        shifted = ndimage.shift(
            truth_hr,
            shift=(-float(dy_px) * scale, -float(dx_px) * scale),
            order=1,
            mode="reflect",
            prefilter=False,
        )
        if sigma_hr > 0:
            shifted = ndimage.gaussian_filter(shifted, sigma=sigma_hr, mode="reflect")
        lr = block_mean_downsample(shifted, scale=scale)
        if noise_sigma_c > 0:
            lr = lr + rng.normal(0.0, float(noise_sigma_c), size=lr.shape).astype(np.float32)
        frames.append(lr.astype(np.float32, copy=False))
    return np.stack(frames, axis=0)


def bicubic_upsample(image: np.ndarray, *, scale: int) -> np.ndarray:
    return ndimage.zoom(np.asarray(image, dtype=np.float32), zoom=(scale, scale), order=3, mode="nearest").astype(
        np.float32,
        copy=False,
    )


def align_lr_mean(frames: np.ndarray, assumed_shifts: np.ndarray) -> np.ndarray:
    aligned = []
    for frame, (dx_px, dy_px) in zip(frames, assumed_shifts, strict=True):
        aligned.append(
            ndimage.shift(
                frame,
                shift=(float(dy_px), float(dx_px)),
                order=1,
                mode="reflect",
                prefilter=False,
            ).astype(np.float32, copy=False)
        )
    return np.mean(np.stack(aligned, axis=0), axis=0, dtype=np.float32)


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
            target = np.ix_(yy, xx)
            accum[target] += frame[np.ix_(valid_y, valid_x)].astype(np.float32, copy=False) * weights
            weight_sum[target] += weights


def drizzle(frames: np.ndarray, assumed_shifts: np.ndarray, *, scale: int) -> tuple[np.ndarray, float]:
    if frames.ndim != 3:
        raise ValueError("frames must have shape (N,H,W)")
    if assumed_shifts.shape != (frames.shape[0], 2):
        raise ValueError(f"assumed shift shape {assumed_shifts.shape} does not match frames {frames.shape}")

    rows, cols = frames.shape[1:]
    accum = np.zeros((rows * scale, cols * scale), dtype=np.float32)
    weight_sum = np.zeros_like(accum)
    y_base = np.arange(rows, dtype=np.float64) * scale
    x_base = np.arange(cols, dtype=np.float64) * scale
    for frame, (dx_px, dy_px) in zip(frames, assumed_shifts, strict=True):
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
    out = np.empty_like(accum)
    out[covered] = accum[covered] / weight_sum[covered]
    out[~covered] = float(np.nanmean(frames))
    zero_coverage_pct = 100.0 * float(1.0 - covered.mean())
    return out.astype(np.float32, copy=False), zero_coverage_pct


def wiener_deconvolve_gaussian(image: np.ndarray, *, sigma_lr_px: float, scale: int, wiener_k: float) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    sigma_hr = max(0.0, float(sigma_lr_px) * int(scale))
    if sigma_hr <= 0:
        return arr.copy()
    rows, cols = arr.shape
    fy = np.fft.fftfreq(rows)
    fx = np.fft.fftfreq(cols)
    radius2 = fy[:, None] * fy[:, None] + fx[None, :] * fx[None, :]
    h = np.exp(-2.0 * np.pi**2 * sigma_hr * sigma_hr * radius2)
    f = np.fft.fft2(arr)
    denom = h * h + float(wiener_k)
    restored = np.fft.ifft2(f * h / np.maximum(denom, 1e-12)).real
    return restored.astype(np.float32, copy=False)


def reconstruct_bundle(
    frames: np.ndarray,
    assumed_shifts: np.ndarray,
    *,
    scale: int,
    psf_sigma_lr_px: float,
    wiener_k: float,
) -> ReconBundle:
    single_lr = align_lr_mean(frames[:1], assumed_shifts[:1])
    aligned_lr = align_lr_mean(frames, assumed_shifts)
    aligned_hr = bicubic_upsample(aligned_lr, scale=scale)
    single_hr = bicubic_upsample(single_lr, scale=scale)
    drizzle_hr, zero_coverage_pct = drizzle(frames, assumed_shifts, scale=scale)

    return ReconBundle(
        single_frame_wiener=wiener_deconvolve_gaussian(
            single_hr,
            sigma_lr_px=psf_sigma_lr_px,
            scale=scale,
            wiener_k=wiener_k,
        ),
        aligned_mean_wiener=wiener_deconvolve_gaussian(
            aligned_hr,
            sigma_lr_px=psf_sigma_lr_px,
            scale=scale,
            wiener_k=wiener_k,
        ),
        drizzle_wiener=wiener_deconvolve_gaussian(
            drizzle_hr,
            sigma_lr_px=psf_sigma_lr_px,
            scale=scale,
            wiener_k=wiener_k,
        ),
        aligned_mean=aligned_hr,
        drizzle=drizzle_hr,
        zero_coverage_pct=zero_coverage_pct,
    )


def crop_border(image: np.ndarray, crop: int) -> np.ndarray:
    if crop <= 0:
        return np.asarray(image)
    if image.shape[0] <= 2 * crop or image.shape[1] <= 2 * crop:
        raise ValueError(f"crop {crop} is too large for image shape {image.shape}")
    return np.asarray(image[crop:-crop, crop:-crop])


def rmse(lhs: np.ndarray, rhs: np.ndarray) -> float:
    a = np.asarray(lhs, dtype=np.float32)
    b = np.asarray(rhs, dtype=np.float32)
    return float(np.sqrt(np.mean((a - b) ** 2)))


def psnr(lhs: np.ndarray, rhs: np.ndarray, *, data_range: float) -> float:
    err = rmse(lhs, rhs)
    if err <= 0:
        return float("inf")
    return float(20.0 * math.log10(float(data_range) / err))


def highpass(image: np.ndarray, *, sigma_hr_px: float) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    return (arr - ndimage.gaussian_filter(arr, sigma=float(sigma_hr_px), mode="reflect")).astype(np.float32, copy=False)


def evaluate_images(
    truth_hr: np.ndarray,
    images: dict[str, np.ndarray],
    *,
    crop_hr_px: int,
    data_range: float,
    highpass_sigma_hr_px: float,
) -> dict[str, float]:
    truth = crop_border(truth_hr, crop_hr_px)
    truth_hp = highpass(truth, sigma_hr_px=highpass_sigma_hr_px)
    out: dict[str, float] = {}
    for name, image in images.items():
        pred = crop_border(np.asarray(image), crop_hr_px)
        pred_hp = highpass(pred, sigma_hr_px=highpass_sigma_hr_px)
        out[f"{name}_rmse_c"] = rmse(pred, truth)
        out[f"{name}_psnr_db"] = psnr(pred, truth, data_range=data_range)
        out[f"{name}_hp_rmse_c"] = rmse(pred_hp, truth_hp)
        out[f"{name}_hp_psnr_db"] = psnr(pred_hp, truth_hp, data_range=max(float(np.std(truth_hp)) * 4.0, 1e-6))
    return out


def evaluate_bundle(
    truth_hr: np.ndarray,
    bundle: ReconBundle,
    *,
    crop_hr_px: int,
    data_range: float,
    highpass_sigma_hr_px: float,
) -> dict[str, float]:
    images = {
        "single_frame_wiener": bundle.single_frame_wiener,
        "aligned_mean_wiener": bundle.aligned_mean_wiener,
        "drizzle_wiener": bundle.drizzle_wiener,
        "aligned_mean": bundle.aligned_mean,
        "drizzle": bundle.drizzle,
    }
    out: dict[str, float] = {"zero_coverage_pct": bundle.zero_coverage_pct}
    out.update(
        evaluate_images(
            truth_hr,
            images,
            crop_hr_px=crop_hr_px,
            data_range=data_range,
            highpass_sigma_hr_px=highpass_sigma_hr_px,
        )
    )
    out["drizzle_wiener_gain_vs_aligned_db"] = out["drizzle_wiener_psnr_db"] - out["aligned_mean_wiener_psnr_db"]
    out["aligned_wiener_gain_vs_single_db"] = out["aligned_mean_wiener_psnr_db"] - out["single_frame_wiener_psnr_db"]
    return out


def alias_frequency_indices(lr_size: int, scale: int) -> tuple[np.ndarray, np.ndarray]:
    base = np.arange(int(lr_size), dtype=np.int64)
    offsets = np.arange(int(scale), dtype=np.int64) * int(lr_size)
    return base[:, None] + offsets[None, :], offsets


def alias_forward_fft(
    truth_hr: np.ndarray,
    shifts_lr_px: np.ndarray,
    *,
    scale: int,
    psf_sigma_lr_px: float,
    noise_sigma_c: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Render LR observation FFTs under a periodic 2x aliasing model.

    This oracle path intentionally uses a Fourier-consistent model rather than
    the spatial drizzle approximation. It estimates the information available
    from phase diversity when the operator family is known.
    """

    hr = np.asarray(truth_hr, dtype=np.float32)
    hr_rows, hr_cols = hr.shape
    if hr_rows % scale or hr_cols % scale:
        raise ValueError(f"HR shape {hr.shape} must be divisible by scale={scale}")
    lr_rows = hr_rows // scale
    lr_cols = hr_cols // scale
    x_fft = np.fft.fft2(hr)

    row_aliases, _row_offsets = alias_frequency_indices(lr_rows, scale)
    col_aliases, _col_offsets = alias_frequency_indices(lr_cols, scale)
    fy_hr = np.fft.fftfreq(hr_rows)
    fx_hr = np.fft.fftfreq(hr_cols)
    sigma_hr = float(psf_sigma_lr_px) * int(scale)
    h = np.exp(-2.0 * np.pi**2 * sigma_hr * sigma_hr * (fy_hr[:, None] ** 2 + fx_hr[None, :] ** 2))
    xh = x_fft * h

    y_fft = np.zeros((len(shifts_lr_px), lr_rows, lr_cols), dtype=np.complex128)
    norm = float(scale * scale)
    for frame_idx, (dx_px, dy_px) in enumerate(np.asarray(shifts_lr_px, dtype=np.float64)):
        dx_hr = float(dx_px) * scale
        dy_hr = float(dy_px) * scale
        out = np.zeros((lr_rows, lr_cols), dtype=np.complex128)
        for ay in range(scale):
            ky = row_aliases[:, ay]
            fy = fy_hr[ky]
            for ax in range(scale):
                kx = col_aliases[:, ax]
                fx = fx_hr[kx]
                phase = np.exp(2j * np.pi * (fy[:, None] * dy_hr + fx[None, :] * dx_hr))
                out += xh[np.ix_(ky, kx)] * phase
        y_fft[frame_idx] = out / norm

    if noise_sigma_c > 0:
        noisy = np.fft.ifft2(y_fft, axes=(-2, -1)).real
        noisy += rng.normal(0.0, float(noise_sigma_c), size=noisy.shape)
        y_fft = np.fft.fft2(noisy, axes=(-2, -1))
    return y_fft


def alias_oracle_reconstruct(
    y_fft: np.ndarray,
    assumed_shifts_lr_px: np.ndarray,
    *,
    scale: int,
    psf_sigma_lr_px: float,
    ridge: float,
) -> np.ndarray:
    """Invert the 2x aliasing system by per-frequency ridge least squares."""

    n_frames, lr_rows, lr_cols = y_fft.shape
    hr_rows = lr_rows * scale
    hr_cols = lr_cols * scale
    row_aliases, _row_offsets = alias_frequency_indices(lr_rows, scale)
    col_aliases, _col_offsets = alias_frequency_indices(lr_cols, scale)
    fy_hr = np.fft.fftfreq(hr_rows)
    fx_hr = np.fft.fftfreq(hr_cols)
    sigma_hr = float(psf_sigma_lr_px) * int(scale)
    h = np.exp(-2.0 * np.pi**2 * sigma_hr * sigma_hr * (fy_hr[:, None] ** 2 + fx_hr[None, :] ** 2))
    shifts = np.asarray(assumed_shifts_lr_px, dtype=np.float64)
    alias_count = int(scale * scale)
    x_hat = np.zeros((hr_rows, hr_cols), dtype=np.complex128)
    norm = float(scale * scale)
    eye = np.eye(alias_count, dtype=np.complex128)

    for ky_lr in range(lr_rows):
        ky_alias = row_aliases[ky_lr]
        fy = fy_hr[ky_alias]
        for kx_lr in range(lr_cols):
            kx_alias = col_aliases[kx_lr]
            fx = fx_hr[kx_alias]
            cols: list[np.ndarray] = []
            h_vals: list[float] = []
            for ay in range(scale):
                for ax in range(scale):
                    h_val = float(h[ky_alias[ay], kx_alias[ax]]) / norm
                    h_vals.append(h_val)
                    dx_hr = shifts[:, 0] * scale
                    dy_hr = shifts[:, 1] * scale
                    phase = np.exp(2j * np.pi * (fy[ay] * dy_hr + fx[ax] * dx_hr))
                    cols.append(h_val * phase)
            a = np.stack(cols, axis=1)
            y = y_fft[:, ky_lr, kx_lr]
            ah = np.conj(a).T
            scale_ridge = float(ridge) * max(float(np.mean(np.square(h_vals))), 1e-12)
            coeffs = np.linalg.solve(ah @ a + scale_ridge * eye, ah @ y)
            idx = 0
            for ay in range(scale):
                for ax in range(scale):
                    x_hat[ky_alias[ay], kx_alias[ax]] = coeffs[idx]
                    idx += 1

    return np.fft.ifft2(x_hat).real.astype(np.float32, copy=False)


def alias_oracle_bundle(
    truth_hr: np.ndarray,
    true_shifts: np.ndarray,
    assumed_shifts: np.ndarray,
    *,
    scale: int,
    psf_sigma_lr_px: float,
    noise_sigma_c: float,
    rng: np.random.Generator,
    ridge: float,
) -> AliasOracleBundle:
    y_fft = alias_forward_fft(
        truth_hr,
        true_shifts,
        scale=scale,
        psf_sigma_lr_px=psf_sigma_lr_px,
        noise_sigma_c=noise_sigma_c,
        rng=rng,
    )
    return AliasOracleBundle(
        alias_single_wiener=alias_oracle_reconstruct(
            y_fft[:1],
            assumed_shifts[:1],
            scale=scale,
            psf_sigma_lr_px=psf_sigma_lr_px,
            ridge=ridge,
        ),
        alias_multiframe_wiener=alias_oracle_reconstruct(
            y_fft,
            assumed_shifts,
            scale=scale,
            psf_sigma_lr_px=psf_sigma_lr_px,
            ridge=ridge,
        ),
    )


def subset_indices(n_total: int, n_frames: int, *, seed: int) -> np.ndarray:
    if n_frames > n_total:
        raise ValueError(f"requested n_frames={n_frames} but only {n_total} shifts are available")
    if n_frames == n_total:
        return np.arange(n_total, dtype=int)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n_total, size=int(n_frames), replace=False).astype(int))


def perturb_shifts(
    shifts: np.ndarray,
    *,
    sigma_px: float,
    seed: int,
) -> np.ndarray:
    base = np.asarray(shifts, dtype=np.float32)
    if float(sigma_px) <= 0:
        return base.copy()
    rng = np.random.default_rng(seed)
    return (base + rng.normal(0.0, float(sigma_px), size=base.shape).astype(np.float32)).astype(np.float32, copy=False)


def add_zero_error_deltas(summary: pd.DataFrame) -> pd.DataFrame:
    out = summary.copy()
    metric_cols = [col for col in out.columns if col.endswith("_psnr_db_mean")]
    key_cols = ["psf_sigma_lr_px", "n_frames", "noise_sigma_c"]
    zero = out[out["shift_error_sigma_px"].eq(0.0)][key_cols + metric_cols].copy()
    zero = zero.rename(columns={col: f"{col}_zero_shift_error" for col in metric_cols})
    out = out.merge(zero, on=key_cols, how="left")
    for col in metric_cols:
        base_col = f"{col}_zero_shift_error"
        out[f"{col}_delta_vs_zero_error_db"] = out[col] - out[base_col]
    return out


def summarize_runs(runs: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["psf_sigma_lr_px", "n_frames", "shift_error_sigma_px", "noise_sigma_c"]
    value_cols = [
        col
        for col in runs.columns
        if col.endswith("_psnr_db")
        or col.endswith("_rmse_c")
        or col
        in {
            "zero_coverage_pct",
            "drizzle_wiener_gain_vs_aligned_db",
            "aligned_wiener_gain_vs_single_db",
            "alias_multiframe_gain_vs_single_db",
        }
    ]
    agg: dict[str, list[str]] = {col: ["mean", "std"] for col in value_cols}
    summary = runs.groupby(group_cols, dropna=False).agg(agg)
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns.to_flat_index()]
    summary = summary.reset_index()
    counts = runs.groupby(group_cols, dropna=False).size().reset_index(name="n_repeats")
    summary = summary.merge(counts, on=group_cols, how="left")
    return add_zero_error_deltas(summary)


def run_experiment(args: argparse.Namespace) -> dict[str, Any]:
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage = load_stage_config(args.stage_config)
    shift_set = load_shift_set(args, stage)
    if shift_set.shifts_lr_px.shape[0] < 2:
        raise ValueError("At least two shifts are required")

    psf_sigmas = args.psf_sigmas_lr_px
    if psf_sigmas is None:
        default_sigma, sigma_source = load_default_psf_sigma(args.psf_config)
        psf_sigmas = [default_sigma]
    else:
        sigma_source = "cli"

    budgets = [budget for budget in args.frame_budgets if budget <= shift_set.shifts_lr_px.shape[0]]
    if not budgets:
        raise ValueError(f"No frame budget fits available shift count {shift_set.shifts_lr_px.shape[0]}")

    run_rows: list[dict[str, Any]] = []
    total = (
        len(psf_sigmas)
        * len(budgets)
        * len(args.shift_error_grid)
        * len(args.scene_seeds)
        * len(args.subset_seeds)
        * max(1, len(args.shift_error_seeds))
    )
    pbar = tqdm(total=total, desc="Stage 0b info budget")
    for psf_sigma in psf_sigmas:
        for n_frames in budgets:
            for subset_seed in args.subset_seeds:
                idx = subset_indices(len(shift_set.shifts_lr_px), n_frames, seed=subset_seed)
                true_shifts = shift_set.shifts_lr_px[idx]
                for scene_seed in args.scene_seeds:
                    truth = make_truth(
                        scene_seed,
                        lr_size=args.lr_size,
                        scale=args.scale,
                        delta_temp_c=args.delta_temp_c,
                    )
                    burst_rng = np.random.default_rng(args.noise_seed + 1000003 * scene_seed + 7919 * n_frames)
                    frames = render_burst(
                        truth,
                        true_shifts,
                        scale=args.scale,
                        psf_sigma_lr_px=psf_sigma,
                        noise_sigma_c=args.noise_sigma_c,
                        rng=burst_rng,
                    )
                    for shift_error in args.shift_error_grid:
                        seeds = [0] if float(shift_error) <= 0 else args.shift_error_seeds
                        for shift_seed in seeds:
                            assumed = perturb_shifts(true_shifts, sigma_px=shift_error, seed=shift_seed)
                            bundle = reconstruct_bundle(
                                frames,
                                assumed,
                                scale=args.scale,
                                psf_sigma_lr_px=psf_sigma,
                                wiener_k=args.wiener_k,
                            )
                            metrics = evaluate_bundle(
                                truth,
                                bundle,
                                crop_hr_px=args.crop_lr_px * args.scale,
                                data_range=args.delta_temp_c,
                                highpass_sigma_hr_px=args.highpass_sigma_hr_px,
                            )
                            if not args.skip_alias_oracle:
                                oracle_rng = np.random.default_rng(
                                    args.noise_seed
                                    + 700001 * scene_seed
                                    + 9176 * n_frames
                                    + 101 * int(round(float(shift_error) * 1000.0))
                                )
                                alias_bundle = alias_oracle_bundle(
                                    truth,
                                    true_shifts,
                                    assumed,
                                    scale=args.scale,
                                    psf_sigma_lr_px=psf_sigma,
                                    noise_sigma_c=args.noise_sigma_c,
                                    rng=oracle_rng,
                                    ridge=args.oracle_ridge,
                                )
                                metrics.update(
                                    evaluate_images(
                                        truth,
                                        {
                                            "alias_single_wiener": alias_bundle.alias_single_wiener,
                                            "alias_multiframe_wiener": alias_bundle.alias_multiframe_wiener,
                                        },
                                        crop_hr_px=args.crop_lr_px * args.scale,
                                        data_range=args.delta_temp_c,
                                        highpass_sigma_hr_px=args.highpass_sigma_hr_px,
                                    )
                                )
                                metrics["alias_multiframe_gain_vs_single_db"] = (
                                    metrics["alias_multiframe_wiener_psnr_db"]
                                    - metrics["alias_single_wiener_psnr_db"]
                                )
                            run_rows.append(
                                {
                                    "psf_sigma_lr_px": float(psf_sigma),
                                    "n_frames": int(n_frames),
                                    "shift_error_sigma_px": float(shift_error),
                                    "shift_error_seed": int(shift_seed),
                                    "subset_seed": int(subset_seed),
                                    "scene_seed": int(scene_seed),
                                    "noise_sigma_c": float(args.noise_sigma_c),
                                    "delta_temp_c": float(args.delta_temp_c),
                                    "shift_source": shift_set.source,
                                    **metrics,
                                }
                            )
                            pbar.update(1)
                    skipped = max(0, len(args.shift_error_seeds) - 1)
                    if skipped and 0.0 in args.shift_error_grid:
                        pbar.update(skipped)
    pbar.close()

    runs = pd.DataFrame(run_rows)
    summary = summarize_runs(runs)
    runs_path = args.output_dir / "info_budget2_runs.csv"
    summary_path = args.output_dir / "info_budget2_summary.csv"
    runs.to_csv(runs_path, index=False)
    summary.to_csv(summary_path, index=False)

    detector_pitch = stage.detector_pitch_um
    hr_pitch = detector_pitch / args.scale
    two_x_nyquist_freq_um_inv = 1.0 / (2.0 * hr_pitch)
    lr_nyquist_freq_um_inv = 1.0 / (2.0 * detector_pitch)
    sigma_audit = []
    for sigma in psf_sigmas:
        sigma_audit.append(
            {
                "psf_sigma_lr_px": float(sigma),
                "gaussian_mtf_at_2x_nyquist_f1_cyc_per_lr_px": gaussian_mtf_lr(1.0, float(sigma)),
                "detector_box_mtf_at_2x_nyquist": detector_box_mtf(
                    two_x_nyquist_freq_um_inv,
                    aperture_um=detector_pitch,
                ),
                "gaussian_mtf_at_lr_nyquist_f0p5_cyc_per_lr_px": gaussian_mtf_lr(0.5, float(sigma)),
                "detector_box_mtf_at_lr_nyquist": detector_box_mtf(
                    lr_nyquist_freq_um_inv,
                    aperture_um=detector_pitch,
                ),
            }
        )

    dr_table_cols = [
        "psf_sigma_lr_px",
        "n_frames",
        "shift_error_sigma_px",
        "n_repeats",
        "drizzle_wiener_psnr_db_mean",
        "drizzle_wiener_psnr_db_mean_delta_vs_zero_error_db",
        "drizzle_wiener_gain_vs_aligned_db_mean",
        "drizzle_wiener_hp_psnr_db_mean",
        "alias_multiframe_wiener_psnr_db_mean",
        "alias_multiframe_wiener_psnr_db_mean_delta_vs_zero_error_db",
        "alias_multiframe_gain_vs_single_db_mean",
        "alias_multiframe_wiener_hp_psnr_db_mean",
        "zero_coverage_pct_mean",
    ]
    dr_table = summary[[col for col in dr_table_cols if col in summary.columns]].copy()

    audit = {
        "detector_pitch_um": detector_pitch,
        "current_spatial_resolution_um": stage.spatial_resolution_um,
        "scale": int(args.scale),
        "hr_pitch_um": hr_pitch,
        "lr_nyquist_period_um": 2.0 * detector_pitch,
        "two_x_output_nyquist_period_um": 2.0 * hr_pitch,
        "pitch_truth_status": (
            "PASS_20UM"
            if math.isclose(detector_pitch, 20.0, rel_tol=0.0, abs_tol=1e-6)
            and math.isclose(stage.spatial_resolution_um, 20.0, rel_tol=0.0, abs_tol=1e-6)
            else "CHECK_CONFIG_NOT_20UM"
        ),
        "historical_10um_numbers_status": "do_not_reuse_without_20um_rerun",
        "frequency_audit": sigma_audit,
        "interpretation_boundary": (
            "Synthetic information budget with known truth and controlled shift mismatch. "
            "It calibrates DR pressure tests; it is not real-data optical ground truth or a claim of 10 um metrology."
        ),
    }

    manifest = {
        "task": "solver_v2 Stage 0b info_budget2 replacement and shift-error sweep",
        "created_at_local_assumed": "2026-07-02",
        "script": _rel(SCRIPT_PATH),
        "output_dir": _rel(args.output_dir),
        "inputs": {
            "stage_config": _rel(args.stage_config),
            "psf_config": _rel(args.psf_config),
            "psf_sigma_source": sigma_source,
            "frame_audit_csv": _rel(args.frame_audit_csv),
            "alignment_csv": _rel(args.alignment_csv),
            "shift_source": shift_set.source,
            "shift_note": shift_set.note,
            "n_available_shifts": int(len(shift_set.shifts_lr_px)),
        },
        "parameters": {
            "psf_sigmas_lr_px": [float(v) for v in psf_sigmas],
            "shift_error_grid_px": [float(v) for v in args.shift_error_grid],
            "frame_budgets": [int(v) for v in budgets],
            "scale": int(args.scale),
            "lr_size": int(args.lr_size),
            "delta_temp_c": float(args.delta_temp_c),
            "noise_sigma_c": float(args.noise_sigma_c),
            "wiener_k": float(args.wiener_k),
            "skip_alias_oracle": bool(args.skip_alias_oracle),
            "oracle_ridge": float(args.oracle_ridge),
            "scene_seeds": [int(v) for v in args.scene_seeds],
            "subset_seeds": [int(v) for v in args.subset_seeds],
            "shift_error_seeds": [int(v) for v in args.shift_error_seeds],
        },
        "audit_20um": audit,
        "dr_calibration_table": dr_table.to_dict(orient="records"),
        "outputs": {
            "runs_csv": _rel(runs_path),
            "summary_csv": _rel(summary_path),
            "summary_json": _rel(args.output_dir / "info_budget2_summary.json"),
        },
        "elapsed_sec": float(time.perf_counter() - start),
    }
    write_json(args.output_dir / "info_budget2_summary.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage-config", type=Path, default=PROJECT_ROOT / "configs" / "stage_calibration.json")
    parser.add_argument("--psf-config", type=Path, default=PROJECT_ROOT / "configs" / "psf_calibration.json")
    parser.add_argument("--frame-audit-csv", type=Path, default=PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--alignment-csv", type=Path, default=default_contour_alignment_csv(project_root_path=PROJECT_ROOT))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "ep15_info_limit" / "stage0b_info_budget2")
    parser.add_argument(
        "--shift-source",
        choices=("auto", "contour_refined", "command_prior", "synthetic_command_lattice"),
        default="auto",
        help="Shift distribution for synthetic burst phases. auto tries contour, command metadata, then synthetic lattice.",
    )
    parser.add_argument("--synthetic-shift-limit", type=int, default=EXPECTED_CLEAN_SR_FRAMES)
    parser.add_argument("--psf-sigmas-lr-px", type=parse_float_list, default=None)
    parser.add_argument("--shift-error-grid", type=parse_float_list, default=list(DEFAULT_SHIFT_ERROR_GRID))
    parser.add_argument("--frame-budgets", type=parse_int_list, default=list(DEFAULT_FRAME_BUDGETS))
    parser.add_argument("--scene-seeds", type=parse_int_list, default=[11, 23, 37, 41, 53, 67])
    parser.add_argument("--subset-seeds", type=parse_int_list, default=[101])
    parser.add_argument("--shift-error-seeds", type=parse_int_list, default=[401, 402, 403])
    parser.add_argument("--noise-seed", type=int, default=1502)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--lr-size", type=int, default=128)
    parser.add_argument("--crop-lr-px", type=int, default=8)
    parser.add_argument("--delta-temp-c", type=float, default=3.0)
    parser.add_argument("--noise-sigma-c", type=float, default=DEFAULT_NOISE_SIGMA_C)
    parser.add_argument("--wiener-k", type=float, default=2.5e-3)
    parser.add_argument("--oracle-ridge", type=float, default=1e-4)
    parser.add_argument("--skip-alias-oracle", action="store_true")
    parser.add_argument("--highpass-sigma-hr-px", type=float, default=5.0)
    args = parser.parse_args()
    if args.scale != 2:
        raise ValueError("Stage 0b scope is 2x; use --scale 2")
    if args.lr_size < 32:
        raise ValueError("--lr-size must be at least 32")
    args.shift_error_grid = [float(v) for v in args.shift_error_grid]
    args.frame_budgets = [int(v) for v in args.frame_budgets]
    args.scene_seeds = [int(v) for v in args.scene_seeds]
    args.subset_seeds = [int(v) for v in args.subset_seeds]
    args.shift_error_seeds = [int(v) for v in args.shift_error_seeds]
    return args


def main() -> int:
    manifest = run_experiment(parse_args())
    print("Stage 0b info-budget2 replacement complete.")
    print(f"Shift source: {manifest['inputs']['shift_source']} ({manifest['inputs']['n_available_shifts']} shifts)")
    print(f"20um audit: {manifest['audit_20um']['pitch_truth_status']}")
    print(f"Summary CSV: {manifest['outputs']['summary_csv']}")
    print(f"Summary JSON: {manifest['outputs']['summary_json']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
