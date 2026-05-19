#!/usr/bin/env python3
"""Run EP06 MAP-TV reconstructions with split-half lambda selection."""

from __future__ import annotations

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
from scipy.ndimage import gaussian_filter, laplace, sobel
from tqdm import tqdm

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_ibp import adjoint_model, forward_model  # noqa: E402
from run_saa import (  # noqa: E402
    PROJECT_ROOT,
    default_workers,
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


def parse_lambda_grid(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("--lambda-grid must contain at least one numeric value")
    if any(value < 0 for value in values):
        raise ValueError("--lambda-grid values must be non-negative")
    return values


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gx = sobel(image, axis=1, mode="nearest")
    gy = sobel(image, axis=0, mode="nearest")
    return np.hypot(gx, gy).astype(np.float32)


def tv_value(image: np.ndarray, eps: float = 1e-6) -> float:
    gx = np.diff(image, axis=1, append=image[:, -1:])
    gy = np.diff(image, axis=0, append=image[-1:, :])
    return float(np.sum(np.sqrt(gx * gx + gy * gy + eps * eps)))


def tv_gradient(image: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    gx = np.diff(image, axis=1, append=image[:, -1:])
    gy = np.diff(image, axis=0, append=image[-1:, :])
    norm = np.sqrt(gx * gx + gy * gy + eps * eps)
    px = gx / norm
    py = gy / norm
    div = np.zeros_like(image, dtype=np.float32)
    div[:, :-1] += px[:, :-1]
    div[:, 1:] -= px[:, :-1]
    div[:-1, :] += py[:-1, :]
    div[1:, :] -= py[:-1, :]
    return -div.astype(np.float32, copy=False)


def artifact_score(image: np.ndarray) -> float:
    image = np.asarray(image, dtype=np.float32)
    if not np.isfinite(image).all():
        return float("inf")
    high_freq = image - gaussian_filter(image, sigma=1.0, mode="nearest")
    lap = laplace(image, mode="nearest")
    base = float(np.std(image))
    if base <= 1e-12:
        return 0.0
    return float((np.std(high_freq) + 0.25 * np.std(lap)) / base)


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    denom = float(np.std(a) + np.std(b))
    return float(np.sqrt(np.mean((a - b) ** 2)) / max(denom, 1e-6))


def fallback_map_tv(
    frames: np.ndarray,
    shifts: np.ndarray,
    init_hr: np.ndarray,
    *,
    lambda_tv: float,
    scale: int,
    psf_sigma: float,
    max_iter: int,
    step_size: float,
    tol: float,
    track: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    image = np.asarray(init_hr, dtype=np.float32).copy()
    rows: list[dict[str, float | int | str]] = []
    n_frames = int(len(frames))

    for iteration in tqdm(range(1, max_iter + 1), desc=f"MAP-TV {track} lambda={lambda_tv:g}"):
        data_grad = np.zeros_like(image, dtype=np.float32)
        data_mse = 0.0
        for frame, shift in zip(frames, shifts, strict=True):
            predicted = forward_model(image, shift, scale=scale, psf_sigma=psf_sigma)
            residual = predicted - np.asarray(frame, dtype=np.float32)
            data_mse += float(np.mean(residual**2))
            data_grad += adjoint_model(residual, shift, hr_shape=image.shape, scale=scale, psf_sigma=psf_sigma)
        data_grad /= max(1, n_frames)
        grad = data_grad + lambda_tv * tv_gradient(image)
        update = -step_size * grad
        image += update
        rel_update = float(np.linalg.norm(update.ravel()) / max(np.linalg.norm(image.ravel()), 1e-12))
        tv = tv_value(image)
        data_rmse = float(np.sqrt(data_mse / max(1, n_frames)))
        rows.append(
            {
                "track": track,
                "iteration": iteration,
                "lambda_tv": float(lambda_tv),
                "data_rmse": data_rmse,
                "tv_value": tv,
                "objective_proxy": float(0.5 * data_rmse**2 + lambda_tv * tv / image.size),
                "relative_update": rel_update,
            }
        )
        if rel_update < tol:
            break
    return image.astype(np.float32, copy=False), pd.DataFrame(rows)


def run_map_tv_algorithm(
    frames: np.ndarray,
    shifts: np.ndarray,
    init_hr: np.ndarray,
    *,
    lambda_tv: float,
    scale: int,
    psf_sigma: float,
    max_iter: int,
    step_size: float,
    tol: float,
    tv_inner_iter: int,
    use_fista: bool,
    workers: int,
    track: str,
) -> tuple[np.ndarray, pd.DataFrame]:
    result = call_if_available(
        [
            ("map_tv.map_tv", "reconstruct_map_tv"),
            ("map_tv.map_tv", "map_tv_reconstruct"),
            ("map_tv", "reconstruct_map_tv"),
            ("map_tv", "map_tv_reconstruct"),
        ],
        frames,
        shifts,
        init_hr=init_hr,
        initial=init_hr,
        lambda_tv=lambda_tv,
        scale=scale,
        psf_sigma=psf_sigma,
        max_iter=max_iter,
        step_size=step_size,
        tol=tol,
        tv_inner_iter=tv_inner_iter,
        use_fista=use_fista,
        workers=workers,
        track=track,
    )
    if result is None:
        return fallback_map_tv(
            frames,
            shifts,
            init_hr,
            lambda_tv=lambda_tv,
            scale=scale,
            psf_sigma=psf_sigma,
            max_iter=max_iter,
            step_size=step_size,
            tol=tol,
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
    if "lambda_tv" not in convergence_df.columns:
        convergence_df["lambda_tv"] = float(lambda_tv)
    return np.asarray(image, dtype=np.float32), convergence_df


def select_lambda(
    frames: np.ndarray,
    shifts: np.ndarray,
    weights: np.ndarray,
    *,
    lambdas: list[float],
    scale: int,
    psf_sigma: float,
    max_iter: int,
    step_size: float,
    tol: float,
    tv_inner_iter: int,
    use_fista: bool,
    selection_artifact_weight: float,
    selection_std_weight: float,
    workers: int,
    track: str,
) -> tuple[float, pd.DataFrame]:
    even = np.arange(len(frames)) % 2 == 0
    odd = ~even
    rows: list[dict[str, float | int | str | bool]] = []
    init_a = shift_and_add(frames[even], shifts[even], weights=weights[even], scale=scale, workers=workers, desc=f"{track} split A init")
    init_b = shift_and_add(frames[odd], shifts[odd], weights=weights[odd], scale=scale, workers=workers, desc=f"{track} split B init")
    init_std = 0.5 * (float(np.std(init_a)) + float(np.std(init_b)))
    for lambda_tv in lambdas:
        recon_a, _ = run_map_tv_algorithm(
            frames[even],
            shifts[even],
            init_a,
            lambda_tv=lambda_tv,
            scale=scale,
            psf_sigma=psf_sigma,
            max_iter=max_iter,
            step_size=step_size,
            tol=tol,
            tv_inner_iter=tv_inner_iter,
            use_fista=use_fista,
            workers=workers,
            track=f"{track}_split_a",
        )
        recon_b, _ = run_map_tv_algorithm(
            frames[odd],
            shifts[odd],
            init_b,
            lambda_tv=lambda_tv,
            scale=scale,
            psf_sigma=psf_sigma,
            max_iter=max_iter,
            step_size=step_size,
            tol=tol,
            tv_inner_iter=tv_inner_iter,
            use_fista=use_fista,
            workers=workers,
            track=f"{track}_split_b",
        )
        consistency = nrmse(recon_a, recon_b)
        artifacts = 0.5 * (artifact_score(recon_a) + artifact_score(recon_b))
        mean_grad = 0.5 * (float(np.mean(gradient_magnitude(recon_a))) + float(np.mean(gradient_magnitude(recon_b))))
        mean_std = 0.5 * (float(np.std(recon_a)) + float(np.std(recon_b)))
        std_excess = max(0.0, mean_std / max(init_std, 1e-12) - 1.0)
        rows.append(
            {
                "track": track,
                "lambda_tv": float(lambda_tv),
                "split_half_nrmse": consistency,
                "artifact_score": artifacts,
                "mean_gradient": mean_grad,
                "std": mean_std,
                "init_saa_std": init_std,
                "std_excess_vs_saa": std_excess,
                "selection_proxy": float(
                    consistency
                    + selection_artifact_weight * artifacts
                    + selection_std_weight * std_excess
                ),
                "n_split_a": int(even.sum()),
                "n_split_b": int(odd.sum()),
            }
        )
    table = pd.DataFrame(rows)
    best_idx = int(table["selection_proxy"].idxmin())
    table["selected"] = False
    table.loc[best_idx, "selected"] = True
    return float(table.loc[best_idx, "lambda_tv"]), table


def initial_image(
    output_dir: Path,
    name: str,
    frames: np.ndarray,
    shifts: np.ndarray,
    weights: np.ndarray,
    *,
    scale: int,
    workers: int,
) -> np.ndarray:
    path = output_dir / name
    if path.exists():
        return np.load(path).astype(np.float32, copy=False)
    return shift_and_add(frames, shifts, weights=weights, scale=scale, workers=workers, desc=f"init {name}")


def run_synthetic_validation(args: argparse.Namespace, lambda_tv: float) -> dict[str, Any]:
    truth, frames, shifts = make_synthetic_observations(
        n_frames=min(32, max(12, args.synthetic_frames)),
        scale=args.scale,
        psf_sigma=args.psf_sigma,
        noise_sigma=args.synthetic_noise,
        seed=args.seed,
    )
    weights = np.ones(len(frames), dtype=np.float32)
    init = shift_and_add(frames, shifts, weights=weights, scale=args.scale, workers=args.workers, desc="synthetic MAP-TV init")
    recon, convergence = run_map_tv_algorithm(
        frames,
        shifts,
        init,
        lambda_tv=lambda_tv,
        scale=args.scale,
        psf_sigma=args.psf_sigma,
        max_iter=min(args.max_iter, 8),
        step_size=args.step_size,
        tol=args.tol,
        tv_inner_iter=args.tv_inner_iter,
        use_fista=not args.no_fista,
        workers=args.workers,
        track="synthetic",
    )
    return {
        "synthetic_shape_hr": list(truth.shape),
        "n_frames": int(len(frames)),
        "scale": int(args.scale),
        "lambda_tv": float(lambda_tv),
        "saa_init_psnr_db": psnr(truth, init),
        "map_tv_psnr_db": psnr(truth, recon),
        "map_tv_beats_saa": bool(psnr(truth, recon) > psnr(truth, init)),
        "iterations": int(len(convergence)),
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
    parser.add_argument("--max-iter", type=int, default=6)
    parser.add_argument("--lambda-select-iter", type=int, default=2)
    parser.add_argument("--lambda-grid", default="0.0003,0.001,0.003")
    parser.add_argument("--step-size", type=float, default=0.2)
    parser.add_argument("--tol", type=float, default=1e-4)
    parser.add_argument("--tv-inner-iter", type=int, default=30)
    parser.add_argument("--no-fista", action="store_true")
    parser.add_argument("--selection-artifact-weight", type=float, default=0.05)
    parser.add_argument("--selection-std-weight", type=float, default=0.08)
    parser.add_argument("--workers", type=int, default=default_workers())
    parser.add_argument("--synthetic-frames", type=int, default=32)
    parser.add_argument("--synthetic-noise", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=608)
    parser.add_argument("--skip-synthetic", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.scale != 2:
        raise ValueError("EP06 is a 2x contour-level POC; keep --scale 2.")
    start = time.perf_counter()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lambdas = parse_lambda_grid(args.lambda_grid)

    frames, metadata = load_main_session_frames(args.data_dir, args.frame_audit_csv, workers=args.workers)
    shifts = load_alignment_shifts(metadata, args.alignment_csv, method=args.alignment_method)
    weights = load_quality_weights(metadata, args.alignment_csv)
    highpass_frames = highpass_preprocess(frames, sigma_bg=args.highpass_sigma, workers=args.workers)
    raw_frames = offset_correction(frames, workers=args.workers)

    lambda_highpass, selection_highpass = select_lambda(
        highpass_frames,
        shifts,
        weights,
        lambdas=lambdas,
        scale=args.scale,
        psf_sigma=args.psf_sigma,
        max_iter=args.lambda_select_iter,
        step_size=args.step_size,
        tol=args.tol,
        tv_inner_iter=args.tv_inner_iter,
        use_fista=not args.no_fista,
        selection_artifact_weight=args.selection_artifact_weight,
        selection_std_weight=args.selection_std_weight,
        workers=args.workers,
        track="highpass",
    )
    lambda_raw, selection_raw = select_lambda(
        raw_frames,
        shifts,
        weights,
        lambdas=lambdas,
        scale=args.scale,
        psf_sigma=args.psf_sigma,
        max_iter=args.lambda_select_iter,
        step_size=args.step_size,
        tol=args.tol,
        tv_inner_iter=args.tv_inner_iter,
        use_fista=not args.no_fista,
        selection_artifact_weight=args.selection_artifact_weight,
        selection_std_weight=args.selection_std_weight,
        workers=args.workers,
        track="raw",
    )
    pd.concat([selection_highpass, selection_raw], ignore_index=True).to_csv(
        args.output_dir / "map_tv_lambda_selection.csv",
        index=False,
    )

    init_highpass = initial_image(args.output_dir, "saa_weighted_highpass.npy", highpass_frames, shifts, weights, scale=args.scale, workers=args.workers)
    init_raw = initial_image(args.output_dir, "saa_weighted_raw.npy", raw_frames, shifts, weights, scale=args.scale, workers=args.workers)
    map_highpass, conv_highpass = run_map_tv_algorithm(
        highpass_frames,
        shifts,
        init_highpass,
        lambda_tv=lambda_highpass,
        scale=args.scale,
        psf_sigma=args.psf_sigma,
        max_iter=args.max_iter,
        step_size=args.step_size,
        tol=args.tol,
        tv_inner_iter=args.tv_inner_iter,
        use_fista=not args.no_fista,
        workers=args.workers,
        track="highpass",
    )
    map_raw, conv_raw = run_map_tv_algorithm(
        raw_frames,
        shifts,
        init_raw,
        lambda_tv=lambda_raw,
        scale=args.scale,
        psf_sigma=args.psf_sigma,
        max_iter=args.max_iter,
        step_size=args.step_size,
        tol=args.tol,
        tv_inner_iter=args.tv_inner_iter,
        use_fista=not args.no_fista,
        workers=args.workers,
        track="raw",
    )
    np.save(args.output_dir / "map_tv_highpass.npy", map_highpass)
    np.save(args.output_dir / "map_tv_raw.npy", map_raw)
    pd.concat([conv_highpass, conv_raw], ignore_index=True).to_csv(args.output_dir / "map_tv_convergence.csv", index=False)

    validation = {"skipped": True} if args.skip_synthetic else run_synthetic_validation(args, lambda_highpass)
    validation.update(
        {
            "real_n_frames": int(len(frames)),
            "alignment_method": args.alignment_method,
            "selected_lambda_highpass": float(lambda_highpass),
            "selected_lambda_raw": float(lambda_raw),
            "lambda_selection": "split-half consistency proxy with artifact penalty",
            "lambda_selection_artifact_weight": float(args.selection_artifact_weight),
            "lambda_selection_std_weight": float(args.selection_std_weight),
            "shift_convention": "EP05 LR-to-reference shifts passed directly; forward prediction applies inverse shift internally.",
            "max_iter_real": int(args.max_iter),
            "step_size": float(args.step_size),
            "psf_sigma": float(args.psf_sigma),
            "use_fista": bool(not args.no_fista),
            "elapsed_sec": float(time.perf_counter() - start),
        }
    )
    (args.output_dir / "map_tv_synthetic_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Saved MAP-TV outputs to {args.output_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
