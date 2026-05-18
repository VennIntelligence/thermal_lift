#!/usr/bin/env python3
"""Run EP06 SAA baselines on highpass and raw-control tracks.

The EP05 alignment convention is preserved: shifts are LR-pixel translations
that move each LR frame into the reference coordinate system. SAA therefore
uses the shifts directly when backfilling onto the 2x HR grid.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, shift as ndi_shift, zoom
from tqdm import tqdm


def bootstrap_paths() -> tuple[Path, Path]:
    script_path = Path(__file__).resolve()
    algo_root = script_path.parents[1]
    project_root = script_path
    while not (project_root / "AGENTS.md").exists() and project_root != project_root.parent:
        project_root = project_root.parent
    for path in (algo_root / "src", project_root / "core" / "src", script_path.parent):
        text = str(path)
        if text not in sys.path:
            sys.path.insert(0, text)
    return project_root, algo_root


PROJECT_ROOT, ALGO_ROOT = bootstrap_paths()

from thermal_core.io import load_frame  # noqa: E402


def default_workers() -> int:
    return max(1, min(4, (os.cpu_count() or 2) // 2))


def boolish(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def call_if_available(
    candidates: list[tuple[str, str]],
    *positional: Any,
    **kwargs: Any,
) -> Any | None:
    """Call the first importable candidate, passing only supported kwargs."""
    for module_name, function_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        fn = getattr(module, function_name, None)
        if fn is None:
            continue
        signature = inspect.signature(fn)
        accepted = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
            or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        }
        return fn(*positional, **accepted)
    return None


def select_main_session(audit: pd.DataFrame) -> pd.DataFrame:
    if "is_main_session" in audit.columns:
        main = audit[boolish(audit["is_main_session"])].copy()
    elif "session" in audit.columns:
        if audit["session"].eq(2).any():
            main = audit[audit["session"].eq(2)].copy()
        else:
            main_session = audit.groupby("session")["file"].count().idxmax()
            main = audit[audit["session"].eq(main_session)].copy()
    else:
        raise ValueError("frame audit must include is_main_session or session")
    if main.empty:
        raise ValueError("No main-session frames found in frame audit")
    return main.sort_values("acquisition_order").reset_index(drop=True)


def load_main_session_frames(
    data_dir: Path,
    frame_audit_csv: Path,
    *,
    workers: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    result = call_if_available(
        [("common.data_loader", "load_main_session_frames")],
        data_dir=data_dir,
        frame_audit_path=frame_audit_csv,
        frame_audit_csv=frame_audit_csv,
        workers=workers,
    )
    if result is not None:
        if isinstance(result, dict):
            return np.asarray(result["frames"], dtype=np.float32), result["metadata"].copy()
        frames, metadata = result
        return np.asarray(frames, dtype=np.float32), metadata.copy()

    audit = pd.read_csv(frame_audit_csv)
    metadata = select_main_session(audit)
    frames = [
        load_frame(data_dir / str(row.file)).astype(np.float32, copy=False)
        for row in tqdm(metadata.itertuples(index=False), total=len(metadata), desc="load frames")
    ]
    return np.stack(frames).astype(np.float32, copy=False), metadata


def highpass_preprocess(frames: np.ndarray, *, sigma_bg: float, workers: int) -> np.ndarray:
    result = call_if_available(
        [("common.data_loader", "highpass_preprocess")],
        frames,
        sigma_bg=sigma_bg,
        workers=workers,
    )
    if result is not None:
        return np.asarray(result, dtype=np.float32)
    return np.stack(
        [
            frame - gaussian_filter(frame, sigma=sigma_bg, mode="nearest")
            for frame in tqdm(frames, desc="highpass")
        ]
    ).astype(np.float32, copy=False)


def offset_correction(frames: np.ndarray, *, workers: int) -> np.ndarray:
    result = call_if_available(
        [("common.data_loader", "offset_correction")],
        frames,
        workers=workers,
    )
    if result is not None:
        return np.asarray(result, dtype=np.float32)
    offsets = np.median(frames, axis=(1, 2), keepdims=True)
    return (frames - offsets).astype(np.float32, copy=False)


def load_alignment_shifts(
    metadata: pd.DataFrame,
    alignment_csv: Path,
    *,
    method: str,
) -> np.ndarray:
    result = call_if_available(
        [("common.alignment", "load_alignment_shifts")],
        metadata=metadata,
        frame_metadata=metadata,
        alignment_csv=alignment_csv,
        method=method,
    )
    if result is not None:
        shifts = np.asarray(result, dtype=np.float32)
        if shifts.shape == (len(metadata), 2):
            return shifts

    alignment = pd.read_csv(alignment_csv)
    alignment = alignment.drop_duplicates("file").set_index("file")
    if method in {"contour_refined", "data_driven_contour_refined", "refined"}:
        cols = ("refined_align_dx_px", "refined_align_dy_px")
    elif method in {"ncc_init", "data_driven_ncc_init", "init"}:
        cols = ("init_align_dx_px", "init_align_dy_px")
    elif {"align_dx_px", "align_dy_px"}.issubset(alignment.columns):
        cols = ("align_dx_px", "align_dy_px")
    else:
        raise ValueError(f"Unsupported alignment method/columns: {method}")

    missing = [str(name) for name in metadata["file"] if str(name) not in alignment.index]
    if missing:
        raise ValueError(f"Alignment table is missing {len(missing)} main frames; first: {missing[:5]}")
    shifts = alignment.loc[metadata["file"].astype(str), list(cols)].to_numpy(dtype=np.float32)
    return shifts


def load_quality_weights(metadata: pd.DataFrame, alignment_csv: Path) -> np.ndarray:
    result = call_if_available(
        [("common.alignment", "load_quality_weights")],
        metadata=metadata,
        frame_metadata=metadata,
        alignment_csv=alignment_csv,
    )
    if result is not None:
        weights = np.asarray(result, dtype=np.float32)
        if weights.shape[0] == len(metadata):
            return normalize_weights(weights)

    alignment = pd.read_csv(alignment_csv).drop_duplicates("file").set_index("file")
    table = alignment.loc[metadata["file"].astype(str)]
    if {"ncc_peak", "refined_holdout_chamfer_px"}.issubset(table.columns):
        weights = table["ncc_peak"].to_numpy(dtype=float) / (
            table["refined_holdout_chamfer_px"].to_numpy(dtype=float) + 1e-3
        )
    elif "ncc_peak" in table.columns:
        weights = table["ncc_peak"].to_numpy(dtype=float)
    else:
        weights = np.ones(len(metadata), dtype=float)
    return normalize_weights(weights)


def normalize_weights(weights: np.ndarray) -> np.ndarray:
    clean = np.asarray(weights, dtype=np.float32)
    clean = np.where(np.isfinite(clean) & (clean > 0), clean, 0.0)
    if float(clean.sum()) <= 0:
        clean = np.ones_like(clean)
    clean = clean / float(np.mean(clean))
    return np.clip(clean, 0.05, 5.0).astype(np.float32, copy=False)


def bicubic_upsample(frame: np.ndarray, *, scale: int) -> np.ndarray:
    return zoom(np.asarray(frame, dtype=np.float32), zoom=scale, order=3).astype(np.float32, copy=False)


def shift_and_add(
    frames: np.ndarray,
    shifts: np.ndarray,
    *,
    weights: np.ndarray | None,
    scale: int,
    workers: int,
    desc: str,
) -> np.ndarray:
    result = call_if_available(
        [
            ("saa.saa", "shift_and_add"),
            ("saa.saa", "reconstruct_saa"),
            ("saa", "shift_and_add"),
            ("saa", "reconstruct_saa"),
        ],
        frames,
        shifts,
        weights=weights,
        scale=scale,
        workers=workers,
    )
    if result is not None:
        if isinstance(result, dict):
            result = result.get("image", result.get("reconstruction"))
        return np.asarray(result, dtype=np.float32)

    n_frames, rows, cols = frames.shape
    hr_shape = (rows * scale, cols * scale)
    accum = np.zeros(hr_shape, dtype=np.float32)
    weight_sum = np.zeros(hr_shape, dtype=np.float32)
    frame_weights = np.ones(n_frames, dtype=np.float32) if weights is None else normalize_weights(weights)
    mask_lr = np.ones((rows, cols), dtype=np.float32)
    mask_hr = zoom(mask_lr, zoom=scale, order=0)

    for frame, (dx, dy), frame_weight in tqdm(
        zip(frames, shifts, frame_weights, strict=True),
        total=n_frames,
        desc=desc,
    ):
        up = zoom(frame, zoom=scale, order=1).astype(np.float32, copy=False)
        shift_yx = (float(dy) * scale, float(dx) * scale)
        aligned = ndi_shift(up, shift=shift_yx, order=1, mode="constant", cval=0.0)
        aligned_mask = ndi_shift(mask_hr, shift=shift_yx, order=1, mode="constant", cval=0.0)
        accum += aligned * frame_weight
        weight_sum += aligned_mask * frame_weight

    return (accum / np.maximum(weight_sum, 1e-6)).astype(np.float32, copy=False)


def synthetic_truth(shape: tuple[int, int] = (128, 160)) -> np.ndarray:
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    truth = 0.3 * np.sin(x / 8.0) + 0.2 * np.cos(y / 11.0)
    truth += ((x > 30) & (x < 120) & (y > 24) & (y < 96)).astype(float)
    truth += 0.8 * ((x - 92) ** 2 + (y - 64) ** 2 < 18**2)
    truth -= 0.5 * ((x > 54) & (x < 72) & (y > 42) & (y < 106)).astype(float)
    return truth.astype(np.float32)


def forward_observation(
    image_hr: np.ndarray,
    shift_lr: np.ndarray,
    *,
    scale: int,
    psf_sigma: float,
) -> np.ndarray:
    dx, dy = map(float, shift_lr)
    shifted = ndi_shift(image_hr, shift=(-dy * scale, -dx * scale), order=1, mode="nearest")
    blurred = gaussian_filter(shifted, sigma=psf_sigma * scale, mode="nearest")
    rows = blurred.shape[0] // scale
    cols = blurred.shape[1] // scale
    cropped = blurred[: rows * scale, : cols * scale]
    return cropped.reshape(rows, scale, cols, scale).mean(axis=(1, 3)).astype(np.float32)


def make_synthetic_observations(
    *,
    n_frames: int,
    scale: int,
    psf_sigma: float,
    noise_sigma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    truth = synthetic_truth()
    shifts = rng.uniform(-0.48, 0.48, size=(n_frames, 2)).astype(np.float32)
    frames = np.stack(
        [forward_observation(truth, shift, scale=scale, psf_sigma=psf_sigma) for shift in shifts]
    )
    frames += rng.normal(0.0, noise_sigma, size=frames.shape).astype(np.float32)
    return truth, frames.astype(np.float32), shifts


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float32)
    estimate = np.asarray(estimate, dtype=np.float32)
    mse = float(np.mean((reference - estimate) ** 2))
    if mse <= 0:
        return float("inf")
    data_range = float(np.max(reference) - np.min(reference))
    return float(20.0 * np.log10(max(data_range, 1e-6) / np.sqrt(mse)))


def run_synthetic_validation(args: argparse.Namespace) -> dict[str, Any]:
    truth, frames, shifts = make_synthetic_observations(
        n_frames=min(48, max(16, args.synthetic_frames)),
        scale=args.scale,
        psf_sigma=args.psf_sigma,
        noise_sigma=args.synthetic_noise,
        seed=args.seed,
    )
    uniform = shift_and_add(
        frames,
        shifts,
        weights=None,
        scale=args.scale,
        workers=args.workers,
        desc="synthetic SAA uniform",
    )
    weighted = shift_and_add(
        frames,
        shifts,
        weights=np.ones(len(frames), dtype=np.float32),
        scale=args.scale,
        workers=args.workers,
        desc="synthetic SAA weighted",
    )
    return {
        "synthetic_shape_hr": list(truth.shape),
        "n_frames": int(len(frames)),
        "scale": int(args.scale),
        "psf_sigma_lr_px": float(args.psf_sigma),
        "noise_sigma": float(args.synthetic_noise),
        "uniform_psnr_db": psnr(truth, uniform),
        "weighted_psnr_db": psnr(truth, weighted),
        "uniform_pass_25db": bool(psnr(truth, uniform) >= 25.0),
        "weighted_pass_25db": bool(psnr(truth, weighted) >= 25.0),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "data_raw" / "infrared_avi")
    parser.add_argument("--frame-audit-csv", type=Path, default=PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--alignment-csv", type=Path, default=PROJECT_ROOT / "output" / "ep05_contour_alignment" / "contour_alignment_results.csv")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "ep06_sr_poc")
    parser.add_argument("--alignment-method", default="contour_refined", choices=["contour_refined", "ncc_init", "data_driven_contour_refined", "data_driven_ncc_init"])
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--highpass-sigma", type=float, default=5.0)
    parser.add_argument("--psf-sigma", type=float, default=1.0)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--synthetic-frames", type=int, default=48)
    parser.add_argument("--synthetic-noise", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=606)
    parser.add_argument("--skip-synthetic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.scale != 2:
        raise ValueError("EP06 is a 2x contour-level POC; keep --scale 2.")
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames, metadata = load_main_session_frames(args.data_dir, args.frame_audit_csv, workers=args.workers)
    shifts = load_alignment_shifts(metadata, args.alignment_csv, method=args.alignment_method)
    weights = load_quality_weights(metadata, args.alignment_csv)
    highpass_frames = highpass_preprocess(frames, sigma_bg=args.highpass_sigma, workers=args.workers)
    raw_frames = offset_correction(frames, workers=args.workers)

    outputs = {
        "saa_uniform_highpass.npy": shift_and_add(highpass_frames, shifts, weights=None, scale=args.scale, workers=args.workers, desc="SAA uniform highpass"),
        "saa_weighted_highpass.npy": shift_and_add(highpass_frames, shifts, weights=weights, scale=args.scale, workers=args.workers, desc="SAA weighted highpass"),
        "saa_uniform_raw.npy": shift_and_add(raw_frames, shifts, weights=None, scale=args.scale, workers=args.workers, desc="SAA uniform raw"),
        "saa_weighted_raw.npy": shift_and_add(raw_frames, shifts, weights=weights, scale=args.scale, workers=args.workers, desc="SAA weighted raw"),
    }
    for name, image in outputs.items():
        np.save(args.output_dir / name, image.astype(np.float32, copy=False))

    reference_file = None
    if args.alignment_csv.exists():
        alignment = pd.read_csv(args.alignment_csv)
        if "reference_file" in alignment.columns and not alignment.empty:
            reference_file = str(alignment["reference_file"].dropna().iloc[0])
    if reference_file in set(metadata["file"].astype(str)):
        ref_idx = int(metadata.index[metadata["file"].astype(str).eq(reference_file)][0])
    else:
        ref_idx = len(metadata) // 2
        reference_file = str(metadata.iloc[ref_idx]["file"])
    lr_reference = highpass_frames[ref_idx]
    np.save(args.output_dir / "lr_reference.npy", lr_reference.astype(np.float32, copy=False))
    np.save(args.output_dir / "bicubic_reference.npy", bicubic_upsample(lr_reference, scale=args.scale))
    lr_raw_reference = raw_frames[ref_idx]
    np.save(args.output_dir / "lr_raw_reference.npy", lr_raw_reference.astype(np.float32, copy=False))
    np.save(args.output_dir / "bicubic_raw_reference.npy", bicubic_upsample(lr_raw_reference, scale=args.scale))

    validation = {"skipped": True} if args.skip_synthetic else run_synthetic_validation(args)
    validation.update(
        {
            "real_n_frames": int(len(frames)),
            "real_lr_shape": list(frames.shape[1:]),
            "real_hr_shape": list(outputs["saa_uniform_highpass.npy"].shape),
            "alignment_method": args.alignment_method,
            "shift_convention": "EP05 shifts move each LR frame into reference coordinates; SAA uses positive shifts directly.",
            "reference_file": reference_file,
            "elapsed_sec": float(time.perf_counter() - start),
        }
    )
    (args.output_dir / "saa_synthetic_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(f"Saved SAA outputs to {args.output_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
