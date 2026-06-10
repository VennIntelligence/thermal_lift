#!/usr/bin/env python3
"""Run EP06 iterative back-projection on highpass and raw-control tracks.

The same EP05 LR-to-reference shifts are passed to the algorithm. Forward-model
prediction uses the inverse shift internally; the fallback below mirrors that
contract when common.forward_model is not yet available.
"""

from __future__ import annotations

from thermal_core.alignment_paths import default_contour_alignment_csv
import argparse
import importlib
import inspect
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, shift as ndi_shift, zoom
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_saa import (  # noqa: E402
    PROJECT_ROOT,
    bicubic_upsample,
    default_workers,
    forward_observation,
    highpass_preprocess,
    load_alignment_shifts,
    load_main_session_frames,
    load_quality_weights,
    make_synthetic_observations,
    offset_correction,
    psnr,
    shift_and_add,
)


def call_if_available(
    candidates: list[tuple[str, str]],
    *positional: Any,
    **kwargs: Any,
) -> Any | None:
    for module_name, function_name in candidates:
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        fn = getattr(module, function_name, None)
        if fn is None:
            continue
        signature = inspect.signature(fn)
        has_var_kw = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        accepted = {key: value for key, value in kwargs.items() if has_var_kw or key in signature.parameters}
        return fn(*positional, **accepted)
    return None


def downsample2_average(image_hr: np.ndarray, *, scale: int) -> np.ndarray:
    rows = image_hr.shape[0] // scale
    cols = image_hr.shape[1] // scale
    cropped = image_hr[: rows * scale, : cols * scale]
    return cropped.reshape(rows, scale, cols, scale).mean(axis=(1, 3)).astype(np.float32)


def forward_model(image_hr: np.ndarray, shift_lr: np.ndarray, *, scale: int, psf_sigma: float) -> np.ndarray:
    result = call_if_available(
        [("common.forward_model", "forward")],
        image_hr,
        shift_lr,
        scale=scale,
        psf_sigma=psf_sigma,
    )
    if result is not None:
        return np.asarray(result, dtype=np.float32)
    return forward_observation(image_hr, shift_lr, scale=scale, psf_sigma=psf_sigma)


def adjoint_model(
    residual_lr: np.ndarray,
    shift_lr: np.ndarray,
    *,
    hr_shape: tuple[int, int],
    scale: int,
    psf_sigma: float,
    splat_sigma: float | None = None,
) -> np.ndarray:
    result = call_if_available(
        [("common.forward_model", "adjoint")],
        residual_lr,
        shift_lr,
        scale=scale,
        psf_sigma=psf_sigma,
        hr_shape=hr_shape,
        splat_sigma=splat_sigma,
    )
    if result is not None:
        return np.asarray(result, dtype=np.float32)

    dx, dy = map(float, shift_lr)
    up = zoom(np.asarray(residual_lr, dtype=np.float32), zoom=scale, order=1)
    if up.shape != hr_shape:
        fixed = np.zeros(hr_shape, dtype=np.float32)
        fixed[: min(hr_shape[0], up.shape[0]), : min(hr_shape[1], up.shape[1])] = up[
            : min(hr_shape[0], up.shape[0]),
            : min(hr_shape[1], up.shape[1]),
        ]
        up = fixed
    blurred = gaussian_filter(up, sigma=psf_sigma * scale, mode="nearest")
    return ndi_shift(blurred, shift=(dy * scale, dx * scale), order=1, mode="nearest").astype(np.float32)


def fallback_ibp(
    frames: np.ndarray,
    shifts: np.ndarray,
    init_hr: np.ndarray,
    *,
    scale: int,
    psf_sigma: float,
    max_iter: int,
    beta: float,
    tol: float,
    splat_sigma: float | None,
    track: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    image = np.asarray(init_hr, dtype=np.float32).copy()
    rows: list[dict[str, float | int | str]] = []
    n_frames = int(len(frames))

    for iteration in tqdm(range(1, max_iter + 1), desc=f"IBP {track}"):
        correction = np.zeros_like(image, dtype=np.float32)
        residual_energy = 0.0
        for frame, shift in zip(frames, shifts, strict=True):
            predicted = forward_model(image, shift, scale=scale, psf_sigma=psf_sigma)
            residual = np.asarray(frame, dtype=np.float32) - predicted
            residual_energy += float(np.mean(residual**2))
            correction += adjoint_model(
                residual,
                shift,
                hr_shape=image.shape,
                scale=scale,
                psf_sigma=psf_sigma,
                splat_sigma=splat_sigma,
            )
        correction /= max(1, n_frames)
        update = beta * correction
        image += update
        rel_update = float(np.linalg.norm(update.ravel()) / max(np.linalg.norm(image.ravel()), 1e-12))
        rows.append(
            {
                "track": track,
                "iteration": iteration,
                "mean_residual_rmse": float(np.sqrt(residual_energy / max(1, n_frames))),
                "relative_update": rel_update,
                "correction_l2": float(np.linalg.norm(correction.ravel())),
            }
        )
        if rel_update < tol:
            break
    return image.astype(np.float32, copy=False), pd.DataFrame(rows)


def run_ibp_algorithm(
    frames: np.ndarray,
    shifts: np.ndarray,
    init_hr: np.ndarray,
    *,
    scale: int,
    psf_sigma: float,
    max_iter: int,
    beta: float,
    tol: float,
    workers: int,
    splat_sigma: float | None,
    track: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    result = call_if_available(
        [
            ("ibp.ibp", "iterative_back_projection"),
            ("ibp.ibp", "reconstruct_ibp"),
            ("ibp", "iterative_back_projection"),
            ("ibp", "reconstruct_ibp"),
        ],
        frames,
        shifts,
        init_hr=init_hr,
        initial=init_hr,
        scale=scale,
        psf_sigma=psf_sigma,
        max_iter=max_iter,
        beta=beta,
        tol=tol,
        workers=workers,
        splat_sigma=splat_sigma,
        track=track,
    )
    if result is None:
        return fallback_ibp(
            frames,
            shifts,
            init_hr,
            scale=scale,
            psf_sigma=psf_sigma,
            max_iter=max_iter,
            beta=beta,
            tol=tol,
            splat_sigma=splat_sigma,
            track=track,
        )
    if isinstance(result, dict):
        image = result.get("image", result.get("reconstruction"))
        convergence = result.get("convergence", result.get("history", []))
    else:
        image, convergence = result
    convergence_df = pd.DataFrame(convergence)
    if "track" not in convergence_df.columns:
        convergence_df.insert(0, "track", track)
    return np.asarray(image, dtype=np.float32), convergence_df


def initial_image(
    output_dir: Path,
    name: str,
    frames: np.ndarray,
    shifts: np.ndarray,
    weights: np.ndarray,
    *,
    scale: int,
    workers: int,
    use_bicubic: bool,
    splat_sigma: float | None = None,
) -> np.ndarray:
    path = output_dir / name
    if path.exists() and not use_bicubic:
        return np.load(path).astype(np.float32, copy=False)
    if use_bicubic:
        return bicubic_upsample(frames[len(frames) // 2], scale=scale)
    return shift_and_add(
        frames,
        shifts,
        weights=weights,
        scale=scale,
        workers=workers,
        splat_sigma=splat_sigma,
        desc=f"init {name}",
    )


def run_synthetic_validation(args: argparse.Namespace) -> dict[str, Any]:
    truth, frames, shifts = make_synthetic_observations(
        n_frames=min(40, max(12, args.synthetic_frames)),
        scale=args.scale,
        psf_sigma=args.psf_sigma,
        noise_sigma=args.synthetic_noise,
        seed=args.seed,
    )
    init = shift_and_add(
        frames,
        shifts,
        weights=None,
        scale=args.scale,
        workers=args.workers,
        splat_sigma=args.splat_sigma,
        desc="synthetic IBP init",
    )
    recon, convergence = run_ibp_algorithm(
        frames,
        shifts,
        init,
        scale=args.scale,
        psf_sigma=args.psf_sigma,
        max_iter=min(args.max_iter, 8),
        beta=args.beta,
        tol=args.tol,
        workers=args.workers,
        splat_sigma=args.splat_sigma,
        track="synthetic",
    )
    return {
        "synthetic_shape_hr": list(truth.shape),
        "n_frames": int(len(frames)),
        "scale": int(args.scale),
        "saa_init_psnr_db": psnr(truth, init),
        "ibp_psnr_db": psnr(truth, recon),
        "ibp_beats_saa": bool(psnr(truth, recon) > psnr(truth, init)),
        "iterations": int(len(convergence)),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "data_raw" / "infrared_avi")
    parser.add_argument("--frame-audit-csv", type=Path, default=PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--alignment-csv", type=Path, default=default_contour_alignment_csv(project_root_path=PROJECT_ROOT))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "ep06_sr_poc")
    parser.add_argument("--alignment-method", default="contour_refined", choices=["contour_refined", "ncc_init", "data_driven_contour_refined", "data_driven_ncc_init"])
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--splat-sigma", type=float, default=None, help="Optional Gaussian adjoint/SAA-init splat sigma in HR pixels.")
    parser.add_argument("--highpass-sigma", type=float, default=5.0)
    parser.add_argument("--psf-sigma", type=float, default=1.0)
    parser.add_argument("--max-iter", type=int, default=8)
    parser.add_argument("--beta", type=float, default=0.35)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--bicubic-init", action="store_true")
    parser.add_argument("--synthetic-frames", type=int, default=40)
    parser.add_argument("--synthetic-noise", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=607)
    parser.add_argument("--skip-synthetic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.scale not in (2, 4):
        raise ValueError("EP06 is a 2x/4x contour-level POC; keep --scale 2 or 4.")
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frames, metadata = load_main_session_frames(args.data_dir, args.frame_audit_csv, workers=args.workers)
    shifts = load_alignment_shifts(metadata, args.alignment_csv, method=args.alignment_method)
    weights = load_quality_weights(metadata, args.alignment_csv)
    highpass_frames = highpass_preprocess(frames, sigma_bg=args.highpass_sigma, workers=args.workers)
    raw_frames = offset_correction(frames, workers=args.workers)

    init_highpass = initial_image(
        args.output_dir,
        "saa_weighted_highpass.npy",
        highpass_frames,
        shifts,
        weights,
        scale=args.scale,
        workers=args.workers,
        use_bicubic=args.bicubic_init,
        splat_sigma=args.splat_sigma,
    )
    init_raw = initial_image(
        args.output_dir,
        "saa_weighted_raw.npy",
        raw_frames,
        shifts,
        weights,
        scale=args.scale,
        workers=args.workers,
        use_bicubic=args.bicubic_init,
        splat_sigma=args.splat_sigma,
    )

    ibp_highpass, conv_highpass = run_ibp_algorithm(
        highpass_frames,
        shifts,
        init_highpass,
        scale=args.scale,
        psf_sigma=args.psf_sigma,
        max_iter=args.max_iter,
        beta=args.beta,
        tol=args.tol,
        workers=args.workers,
        splat_sigma=args.splat_sigma,
        track="highpass",
    )
    ibp_raw, conv_raw = run_ibp_algorithm(
        raw_frames,
        shifts,
        init_raw,
        scale=args.scale,
        psf_sigma=args.psf_sigma,
        max_iter=args.max_iter,
        beta=args.beta,
        tol=args.tol,
        workers=args.workers,
        splat_sigma=args.splat_sigma,
        track="raw",
    )
    np.save(args.output_dir / "ibp_highpass.npy", ibp_highpass)
    np.save(args.output_dir / "ibp_raw.npy", ibp_raw)
    pd.concat([conv_highpass, conv_raw], ignore_index=True).to_csv(args.output_dir / "ibp_convergence.csv", index=False)

    validation = {"skipped": True} if args.skip_synthetic else run_synthetic_validation(args)
    validation.update(
        {
            "real_n_frames": int(len(frames)),
            "alignment_method": args.alignment_method,
            "shift_convention": "EP05 LR-to-reference shifts passed directly; forward prediction applies inverse shift internally.",
            "max_iter_real": int(args.max_iter),
            "psf_sigma": float(args.psf_sigma),
            "splat_sigma_hr_px": None if args.splat_sigma is None else float(args.splat_sigma),
            "elapsed_sec": float(time.perf_counter() - start),
        }
    )
    (args.output_dir / "ibp_synthetic_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Saved IBP outputs to {args.output_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
