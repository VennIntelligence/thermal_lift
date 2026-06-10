#!/usr/bin/env python3
"""Run EP10 STScI Drizzle pixfrac sweep on the 248 clean-frame real input."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


ALGO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EP06_SRC = PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src"
for path in (ALGO_ROOT / "src", EP06_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from drizzle.resample import Drizzle

from common.alignment import load_alignment_shifts
from common.data_loader import bicubic_upsample, highpass_preprocess, load_main_session_frames
from common.forward_model import forward
from common.metrics import artifact_score, split_half_consistency
from ep10_drizzle.drizzle_sr import (
    _pearson_finite,
    coverage_statistics,
    drizzle_reconstruct,
    gaussian_unsharp,
    holdout_residual_mse,
    psnr,
    raw_control_agreement,
)
from thermal_core.plotting import COLORMAPS, METHOD_COLOR_LIST, savefig_academic, setup_academic_style


DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "ep10_drizzle"
DEFAULT_PIXFRACS = (1.0, 0.8, 0.7, 0.6, 0.5)
DEFAULT_PIXFRACS_4X = (1.0, 0.8, 0.6, 0.4, 0.3)


def _default_workers() -> int:
    return max(1, min(4, os.cpu_count() or 1))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pf_label(pixfrac: float) -> str:
    return f"{float(pixfrac):.1f}"


def _center_crop(img: np.ndarray, fraction: float = 1.0 / 3.0) -> np.ndarray:
    rows, cols = img.shape
    crop_rows = max(1, int(round(rows * fraction)))
    crop_cols = max(1, int(round(cols * fraction)))
    r0 = (rows - crop_rows) // 2
    c0 = (cols - crop_cols) // 2
    return img[r0 : r0 + crop_rows, c0 : c0 + crop_cols]




def _synthetic_truth(shape: tuple[int, int] = (128, 160)) -> np.ndarray:
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    truth = 0.28 * np.sin(x / 8.0) + 0.18 * np.cos(y / 11.0)
    truth += ((x > 30) & (x < 120) & (y > 24) & (y < 96)).astype(float)
    truth += 0.75 * ((x - 92) ** 2 + (y - 64) ** 2 < 18**2)
    truth -= 0.45 * ((x > 54) & (x < 72) & (y > 42) & (y < 106)).astype(float)
    truth -= truth.min()
    truth /= max(float(truth.max()), 1e-12)
    return truth.astype(np.float32)


def run_synthetic_validation(output_dir: Path, *, scale: int, seed: int = 10010) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    truth = _synthetic_truth()
    phases = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]], dtype=np.float32)
    shifts = np.tile(phases, (12, 1))
    shifts += rng.normal(0.0, 0.01, size=shifts.shape).astype(np.float32)
    shifts = np.clip(shifts, 0.0, 0.5)
    frames = np.stack([forward(truth, shift, psf_sigma=0.0, scale=scale) for shift in shifts]).astype(np.float32)
    frames += rng.normal(0.0, 0.002, size=frames.shape).astype(np.float32)

    recon, coverage = drizzle_reconstruct(frames, shifts, scale=scale, pixfrac=0.5, coverage_threshold=1.0)
    baseline = bicubic_upsample(np.mean(frames, axis=0), scale=scale)

    point_hr = np.zeros((scale * 4, scale * 4), dtype=np.float32)
    expected_yx = (scale // 2, scale // 2)
    point_hr[expected_yx] = 1.0
    point_shift = np.array([[0.5, 0.5]], dtype=np.float32)
    point_lr = forward(point_hr, point_shift[0], psf_sigma=0.0, scale=scale).astype(np.float32)
    point_recon, _ = drizzle_reconstruct(
        point_lr[np.newaxis, ...],
        point_shift,
        scale=scale,
        pixfrac=0.01,
        coverage_threshold=0.0,
    )
    point_argmax = tuple(int(v) for v in np.unravel_index(np.nanargmax(point_recon), point_recon.shape))

    payload = {
        "drizzle_version": getattr(sys.modules.get("drizzle"), "__version__", "unknown"),
        "add_image_signature": str(inspect.signature(Drizzle.add_image)),
        "scale": int(scale),
        "truth_shape": list(truth.shape),
        "frames_shape": list(frames.shape),
        "n_frames": int(len(frames)),
        "pixfrac": 0.5,
        "baseline_psnr_db": psnr(truth, baseline),
        "drizzle_psnr_db": psnr(truth, recon),
        "coverage_min": float(np.nanmin(coverage)),
        "point_coordinate_expected_yx": [int(expected_yx[0]), int(expected_yx[1])],
        "point_coordinate_reconstructed_yx": list(point_argmax),
        "coordinate_check_pass": bool(point_argmax == expected_yx),
    }
    _write_json(output_dir / "synthetic_validation.json", payload)
    return payload


def run_small_real_check(hp_frames: np.ndarray, shifts: np.ndarray, output_dir: Path, *, scale: int) -> dict[str, Any]:
    n_small = min(5, len(hp_frames))
    small_hr, small_coverage = drizzle_reconstruct(
        hp_frames[:n_small],
        shifts[:n_small],
        scale=scale,
        pixfrac=0.7,
        coverage_threshold=0.0,
    )
    lr_reference = bicubic_upsample(np.nanmean(hp_frames[:n_small], axis=0), scale=scale)
    crop_hr = _center_crop(small_hr)
    crop_lr = _center_crop(lr_reference)
    hr_pos_local = np.unravel_index(np.nanargmax(np.abs(crop_hr)), crop_hr.shape)
    lr_pos_local = np.unravel_index(np.nanargmax(np.abs(crop_lr)), crop_lr.shape)
    delta = float(np.linalg.norm(np.asarray(hr_pos_local, dtype=float) - np.asarray(lr_pos_local, dtype=float)))
    payload = {
        "n_frames": int(n_small),
        "scale": int(scale),
        "pixfrac": 0.7,
        "center_crop_abs_peak_drizzle_yx": [int(hr_pos_local[0]), int(hr_pos_local[1])],
        "center_crop_abs_peak_bicubic_mean_yx": [int(lr_pos_local[0]), int(lr_pos_local[1])],
        "center_crop_peak_delta_px": delta,
        "center_crop_corr_vs_bicubic_mean": _pearson_finite(crop_hr, crop_lr),
        "min_coverage": float(np.nanmin(small_coverage)),
    }
    np.save(output_dir / "small_real_pf0.7_hr.npy", small_hr.astype(np.float32, copy=False))
    _write_json(output_dir / "small_real_check.json", payload)
    return payload


def save_comparison_figure(
    output_dir: Path,
    pixfracs: list[float],
    highpass_by_pf: dict[float, np.ndarray],
    unsharp_by_pf: dict[float, np.ndarray],
) -> None:
    setup_academic_style()
    crops = [_center_crop(highpass_by_pf[pf]) for pf in pixfracs] + [_center_crop(unsharp_by_pf[pf]) for pf in pixfracs]
    vmax = float(np.nanpercentile(np.abs(np.concatenate([c[np.isfinite(c)].ravel() for c in crops])), 99.0))
    vmax = max(vmax, 1e-6)

    fig, axes = plt.subplots(2, len(pixfracs), figsize=(7.2, 3.3), squeeze=False)
    for col, pf in enumerate(pixfracs):
        for row, (label, images) in enumerate((("Drizzle", highpass_by_pf), ("Unsharp", unsharp_by_pf))):
            ax = axes[row, col]
            im = ax.imshow(
                _center_crop(images[pf]),
                cmap=COLORMAPS["residual_diff"],
                vmin=-vmax,
                vmax=vmax,
                interpolation="nearest",
            )
            ax.set_title(f"{label} pf={_pf_label(pf)}")
            ax.set_xticks([])
            ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.78, label="Highpass response [deg C]")
    savefig_academic(fig, output_dir / "comparison_pixfrac.png")


def save_coverage_figure(output_dir: Path, pixfracs: list[float], coverage_by_pf: dict[float, np.ndarray]) -> None:
    setup_academic_style()
    finite_values = np.concatenate([coverage_by_pf[pf][np.isfinite(coverage_by_pf[pf])].ravel() for pf in pixfracs])
    vmax = float(np.percentile(finite_values, 95.0)) if finite_values.size else 1.0
    vmax = max(vmax, 1.0)

    fig, axes = plt.subplots(1, len(pixfracs), figsize=(7.2, 2.0), squeeze=False)
    for ax, pf in zip(axes.ravel(), pixfracs, strict=True):
        im = ax.imshow(coverage_by_pf[pf], cmap=COLORMAPS["coverage"], vmin=0.0, vmax=vmax, interpolation="nearest")
        ax.set_title(f"pf={_pf_label(pf)}")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.72, label="Drizzle out_wht")
    savefig_academic(fig, output_dir / "coverage_maps.png")

    fig, axes = plt.subplots(1, len(pixfracs), figsize=(7.2, 1.9), squeeze=False)
    for idx, (ax, pf) in enumerate(zip(axes.ravel(), pixfracs, strict=True)):
        vals = coverage_by_pf[pf][np.isfinite(coverage_by_pf[pf])]
        ax.hist(vals, bins=60, color=METHOD_COLOR_LIST[idx % len(METHOD_COLOR_LIST)], alpha=0.9)
        ax.axvline(1.0, color="#444444", linestyle="--", linewidth=0.9)
        ax.set_title(f"pf={_pf_label(pf)}")
        ax.set_yscale("log")
        ax.set_xlabel("Coverage")
        if idx == 0:
            ax.set_ylabel("Pixels")
    savefig_academic(fig, output_dir / "coverage_histograms.png")


def run_sweep(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    print("Drizzle API")
    print(f"  Drizzle: {inspect.signature(Drizzle)}")
    print(f"  Drizzle.add_image: {inspect.signature(Drizzle.add_image)}")

    print("Loading 248 clean-frame raw input and contour-refined shifts")
    raw_frames, metadata = load_main_session_frames(workers=args.workers, dtype=np.float32, limit=args.limit)
    shifts = load_alignment_shifts(args.alignment_method, metadata=metadata).astype(np.float32, copy=False)
    hp_frames = highpass_preprocess(raw_frames, sigma_bg=args.highpass_sigma, workers=args.workers)
    print(f"Loaded frames={raw_frames.shape}, shifts={shifts.shape}")

    print("Running synthetic coordinate validation")
    synthetic = run_synthetic_validation(output_dir, scale=args.scale)
    if not synthetic["coordinate_check_pass"]:
        raise RuntimeError(f"Drizzle pixmap coordinate check failed: {synthetic}")

    print("Running 5-frame real-data smoke reconstruction")
    run_small_real_check(hp_frames, shifts, output_dir, scale=args.scale)

    lr_baseline = np.nanmean(hp_frames, axis=0)

    rows: list[dict[str, Any]] = []
    highpass_by_pf: dict[float, np.ndarray] = {}
    unsharp_by_pf: dict[float, np.ndarray] = {}
    coverage_by_pf: dict[float, np.ndarray] = {}

    pixfracs = [float(pf) for pf in args.pixfracs]
    for pf in tqdm(pixfracs, desc="pixfrac sweep"):
        label = _pf_label(pf)
        print(f"pixfrac={label}: highpass Drizzle")
        hr, coverage = drizzle_reconstruct(
            hp_frames,
            shifts,
            scale=args.scale,
            pixfrac=pf,
            coverage_threshold=args.coverage_threshold,
        )
        hr_unsharp = gaussian_unsharp(hr, sigma=args.unsharp_sigma, amount=args.unsharp_amount)

        np.save(output_dir / f"drizzle_pf{label}_hr.npy", hr.astype(np.float32, copy=False))
        np.save(output_dir / f"drizzle_pf{label}_coverage.npy", coverage.astype(np.float32, copy=False))
        np.save(output_dir / f"drizzle_pf{label}_unsharp_hr.npy", hr_unsharp.astype(np.float32, copy=False))

        print(f"pixfrac={label}: raw-control track")
        raw_corr, raw_control_hp = raw_control_agreement(
            hr,
            raw_frames,
            shifts,
            scale=args.scale,
            pixfrac=pf,
            highpass_sigma_lr=args.highpass_sigma,
            coverage_threshold=args.coverage_threshold,
        )
        np.save(output_dir / f"drizzle_pf{label}_raw_control_highpass.npy", raw_control_hp.astype(np.float32, copy=False))

        print(f"pixfrac={label}: split-half consistency")
        split_half = split_half_consistency(
            hp_frames,
            shifts,
            drizzle_reconstruct,
            n_splits=args.n_splits,
            random_state=args.random_state,
            scale=args.scale,
            pixfrac=pf,
            coverage_threshold=args.coverage_threshold,
        )
        split_half.to_csv(output_dir / f"split_half_pf{label}.csv", index=False)

        print(f"pixfrac={label}: holdout residual")
        holdout_mse = holdout_residual_mse(
            hp_frames,
            shifts,
            scale=args.scale,
            pixfrac=pf,
            psf_sigma=args.psf_sigma,
            coverage_threshold=args.coverage_threshold,
        )

        cov_stats = coverage_statistics(coverage, threshold=args.coverage_threshold)
        row = {
            "pixfrac": pf,
            "scale": int(args.scale),
            "n_input_frames": int(len(hp_frames)),
            "input_frame_count": int(len(hp_frames)),
            "split_half_nrmse": float(split_half["nrmse"].median()),
            "holdout_mse": float(holdout_mse),
            "artifact_score": float(artifact_score(hr)),
            "artifact_score_with_lr_overshoot": float(artifact_score(hr, lr_baseline)),
            "min_coverage": cov_stats["min_coverage"],
            "raw_control_corr": float(raw_corr),
            "coverage_lt1_fraction": cov_stats["coverage_lt1_fraction"],
            "coverage_p05": cov_stats["coverage_p05"],
            "coverage_median": cov_stats["coverage_median"],
            "coverage_p95": cov_stats["coverage_p95"],
            "artifact_score_unsharp": float(artifact_score(hr_unsharp)),
            "artifact_score_unsharp_with_lr_overshoot": float(artifact_score(hr_unsharp, lr_baseline)),
        }
        rows.append(row)

        highpass_by_pf[pf] = hr
        unsharp_by_pf[pf] = hr_unsharp
        coverage_by_pf[pf] = coverage

    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "sweep_results.csv", index=False)
    save_comparison_figure(output_dir, pixfracs, highpass_by_pf, unsharp_by_pf)
    save_coverage_figure(output_dir, pixfracs, coverage_by_pf)

    print(f"Wrote {output_dir / 'sweep_results.csv'}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scale", type=int, choices=[2, 4], default=2)
    parser.add_argument("--pixfracs", type=float, nargs="+", default=None)
    parser.add_argument("--alignment-method", default="contour_refined")
    parser.add_argument("--highpass-sigma", type=float, default=5.0)
    parser.add_argument("--psf-sigma", type=float, default=0.5)
    parser.add_argument("--unsharp-sigma", type=float, default=1.0)
    parser.add_argument("--unsharp-amount", type=float, default=0.3)
    parser.add_argument("--coverage-threshold", type=float, default=1.0)
    parser.add_argument("--n-splits", type=int, default=10)
    parser.add_argument("--random-state", type=int, default=0)
    parser.add_argument("--workers", type=int, default=_default_workers())
    parser.add_argument("--limit", type=int, default=None, help="Optional frame limit for debugging; omit for all 248 clean frames.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.pixfracs is None:
        args.pixfracs = list(DEFAULT_PIXFRACS_4X if int(args.scale) == 4 else DEFAULT_PIXFRACS)
    summary = run_sweep(args)
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
