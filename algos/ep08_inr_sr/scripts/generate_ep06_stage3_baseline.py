#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent
EP08_ROOT = SCRIPTS_DIR.parent
PROJECT_ROOT = EP08_ROOT.parent.parent
SRC = EP08_ROOT / "src"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ep08.metrics import artifact_score, holdout_residual, p95_gradient, raw_control_agreement, split_half_nrmse
from ep08.splits import build_train_val_split
from ep08.stage1 import _gradient_magnitude, _upscale_raw_control, load_dataset_for_config, parse_patch_shape, save_image_figure
from ep08.utils import save_json, set_seed
from generate_ep06_patch_baseline import _import_ep06_map_tv, _jsonable, _run_map_tv, _split_train_indices, _write_metrics_csv
from thermal_core.plotting import savefig_academic, setup_academic_style

PROJECT_OUTPUT = PROJECT_ROOT / "output" / "ep08_inr_sr"
STAGE3_OUTPUT = PROJECT_OUTPUT / "stage3"
DEFAULT_N_FRAMES = 64
DEFAULT_COORD_ASPECT_MODE = "preserve"
DEFAULT_MAX_ITER = 20
DEFAULT_SPLIT_HALF_MAX_ITER = 10


def patch_shape_label(patch_shape: int | tuple[int, int] | None) -> str:
    if patch_shape is None:
        return "full"
    if isinstance(patch_shape, tuple):
        return f"{int(patch_shape[0])}x{int(patch_shape[1])}"
    return str(int(patch_shape))


def default_output_dir(
    *,
    n_frames: int,
    patch_shape: int | tuple[int, int] | None,
    coord_aspect_mode: str,
) -> Path:
    return STAGE3_OUTPUT / f"ep06_map_tv_{int(n_frames):03d}_{patch_shape_label(patch_shape)}_{coord_aspect_mode}"


def _cfg(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "data": {
            "scale": 2,
            "default_n_frames": int(args.n_frames),
            "patch_shape": args.patch_shape,
            "default_patch_size_lr_px": None if args.patch_shape is None else args.patch_shape,
            "val_ratio": float(args.val_ratio),
        },
        "coordinates": {"aspect_mode": str(args.coord_aspect_mode)},
        "forward": {"psf_sigma_lr_px": float(args.psf_sigma)},
        "preprocess": {"highpass_sigma_bg_lr_px": float(args.highpass_sigma), "highpass_mode": "nearest"},
        "runtime": {
            "data_mode": str(args.data_mode),
            "device": str(args.device),
            "workers": int(args.workers),
            "alignment_method": str(args.alignment_method),
            "seed": int(args.seed),
            "output_dir": str(args.output_dir),
            "data_dir": args.data_dir,
            "frame_audit_path": args.frame_audit_path,
        },
        "metrics": {"noise_sigma": float(args.noise_sigma)},
    }


def _write_convergence_csv(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({key for row in records for key in row}) if records else ["iteration"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def _save_convergence_curve(records: list[dict[str, Any]], path: Path) -> bool:
    if not records:
        return False
    setup_academic_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    iterations = np.array([float(row.get("iteration", idx + 1)) for idx, row in enumerate(records)], dtype=float)
    plotted = False
    for key, label, color in (
        ("objective", "objective", "#4C72B0"),
        ("residual_mse", "residual MSE", "#C44E52"),
        ("relative_update", "relative update", "#55A868"),
    ):
        values = np.array([float(row[key]) for row in records if key in row and np.isfinite(float(row[key]))], dtype=float)
        value_steps = np.array([float(row.get("iteration", idx + 1)) for idx, row in enumerate(records) if key in row and np.isfinite(float(row[key]))], dtype=float)
        keep = values > 0
        if values.size and np.any(keep):
            ax.plot(value_steps[keep], values[keep], label=label, color=color)
            plotted = True
    if not plotted:
        plt.close(fig)
        return False
    ax.set_yscale("log")
    ax.set_title("EP06 MAP-TV Stage 3 Convergence")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Value")
    ax.legend()
    if iterations.size:
        ax.set_xlim(float(np.min(iterations)), float(np.max(iterations)))
    savefig_academic(fig, path)
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate an EP06 MAP-TV baseline using the EP08 Stage 3 full-frame protocol.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--data-mode", choices=["synthetic", "real"], default="real")
    parser.add_argument("--n-frames", type=int, default=DEFAULT_N_FRAMES)
    parser.add_argument("--patch-shape", type=parse_patch_shape, default=argparse.SUPPRESS, help="LR patch shape: H,W, a single int, or full/None")
    parser.add_argument("--patch-size", type=int, default=None, help="Backward-compatible square LR patch size in pixels")
    parser.add_argument("--coord-aspect-mode", "--coordinate-aspect-mode", choices=["preserve", "stretch"], default=DEFAULT_COORD_ASPECT_MODE)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alignment-method", default="contour_refined")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--frame-audit-path", type=Path, default=None)
    parser.add_argument("--highpass-sigma", type=float, default=8.0)
    parser.add_argument("--psf-sigma", type=float, default=1.0)
    parser.add_argument("--noise-sigma", type=float, default=0.0724)
    parser.add_argument("--lambda-tv", type=float, default=1.0e-3)
    parser.add_argument("--max-iter", type=int, default=DEFAULT_MAX_ITER)
    parser.add_argument("--split-half-max-iter", type=int, default=DEFAULT_SPLIT_HALF_MAX_ITER)
    parser.add_argument("--skip-split-half", action="store_true")
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1.0e-4)
    parser.add_argument("--tv-inner-iter", type=int, default=30)
    return parser


def finalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if args.n_frames is None:
        args.n_frames = DEFAULT_N_FRAMES
    if args.data_mode is None:
        args.data_mode = "real"
    if args.device is None:
        args.device = "cpu"
    if args.workers is None:
        args.workers = 1
    if args.alignment_method is None:
        args.alignment_method = "contour_refined"
    if args.val_ratio is None:
        args.val_ratio = 0.2
    if args.seed is None:
        args.seed = 42
    if args.coord_aspect_mode is None:
        args.coord_aspect_mode = DEFAULT_COORD_ASPECT_MODE
    if args.highpass_sigma is None:
        args.highpass_sigma = 8.0
    if args.psf_sigma is None:
        args.psf_sigma = 1.0
    if args.noise_sigma is None:
        args.noise_sigma = 0.0724
    if args.lambda_tv is None:
        args.lambda_tv = 1.0e-3
    if args.max_iter is None:
        args.max_iter = DEFAULT_MAX_ITER
    if args.split_half_max_iter is None:
        args.split_half_max_iter = DEFAULT_SPLIT_HALF_MAX_ITER
    if args.step_size is None:
        args.step_size = 1.0
    if args.tol is None:
        args.tol = 1.0e-4
    if args.tv_inner_iter is None:
        args.tv_inner_iter = 30
    if not hasattr(args, "patch_shape"):
        args.patch_shape = int(args.patch_size) if args.patch_size is not None else None
    if args.output_dir is None:
        args.output_dir = default_output_dir(
            n_frames=int(args.n_frames),
            patch_shape=args.patch_shape,
            coord_aspect_mode=str(args.coord_aspect_mode),
        )
    return args


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    return finalize_args(build_parser().parse_args(argv))


def run_map_tv_stage3_baseline(args: argparse.Namespace) -> None:
    _import_ep06_map_tv()
    args = finalize_args(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg = _cfg(args)
    device = torch.device(str(args.device))
    set_seed(int(args.seed))

    observations, shifts, raw_control, forward_operator, metadata = load_dataset_for_config("siren", cfg, device)
    forward_operator = forward_operator.to(device)
    frame_ids = np.arange(int(observations.shape[0]))
    train_indices, val_indices, val_mask = build_train_val_split(
        frame_ids,
        shifts.detach().cpu().numpy(),
        val_ratio=float(args.val_ratio),
        seed=int(args.seed),
    )
    if val_indices.size == 0 and frame_ids.size > 1:
        val_indices = frame_ids[-max(1, frame_ids.size // 5) :]
        train_indices = np.setdiff1d(frame_ids, val_indices)
        val_mask = np.zeros(frame_ids.size, dtype=bool)
        val_mask[val_indices] = True

    observations_np = observations.detach().cpu().numpy().astype(np.float32)
    shifts_np = shifts.detach().cpu().numpy().astype(np.float32)
    start = time.perf_counter()
    hr_image, convergence = _run_map_tv(
        observations_np[train_indices],
        shifts_np[train_indices],
        lambda_tv=float(args.lambda_tv),
        max_iter=int(args.max_iter),
        step_size=float(args.step_size),
        psf_sigma=float(args.psf_sigma),
        workers=int(args.workers),
        tol=float(args.tol),
        tv_inner_iter=int(args.tv_inner_iter),
    )
    elapsed = float(time.perf_counter() - start)
    hr_shape = tuple(int(v) for v in hr_image.shape)
    raw_control_hr = _upscale_raw_control(raw_control, train_indices, hr_shape).astype(np.float32)

    split_score: float | None = None
    split_a_image: np.ndarray | None = None
    split_b_image: np.ndarray | None = None
    if not bool(args.skip_split_half):
        half_a, half_b = _split_train_indices(train_indices, int(args.seed))
        split_a_image, _ = _run_map_tv(
            observations_np[half_a],
            shifts_np[half_a],
            lambda_tv=float(args.lambda_tv),
            max_iter=int(args.split_half_max_iter),
            step_size=float(args.step_size),
            psf_sigma=float(args.psf_sigma),
            workers=int(args.workers),
            tol=float(args.tol),
            tv_inner_iter=int(args.tv_inner_iter),
        )
        split_b_image, _ = _run_map_tv(
            observations_np[half_b],
            shifts_np[half_b],
            lambda_tv=float(args.lambda_tv),
            max_iter=int(args.split_half_max_iter),
            step_size=float(args.step_size),
            psf_sigma=float(args.psf_sigma),
            workers=int(args.workers),
            tol=float(args.tol),
            tv_inner_iter=int(args.tv_inner_iter),
        )
        split_score = split_half_nrmse(split_a_image, split_b_image)
        np.save(args.output_dir / "split_half_a.npy", split_a_image.astype(np.float32))
        np.save(args.output_dir / "split_half_b.npy", split_b_image.astype(np.float32))

    metrics = {
        "method": "ep06_map_tv",
        "method_label": "EP06 MAP-TV",
        "family": "classic_opt",
        "data_mode": str(args.data_mode),
        "protocol": f"ep08_stage3_{int(args.n_frames):03d}_{patch_shape_label(args.patch_shape)}_{args.coord_aspect_mode}_seed{int(args.seed)}",
        "n_frames": int(observations.shape[0]),
        "train_frame_count": int(train_indices.size),
        "val_frame_count": int(val_indices.size),
        "lr_shape": [int(v) for v in observations.shape[-2:]],
        "hr_shape": [int(v) for v in hr_image.shape],
        "holdout_residual": holdout_residual(
            hr_image,
            observations,
            forward_operator,
            indices=val_indices,
            noise_sigma=float(args.noise_sigma),
        ),
        "split_half_nrmse": split_score,
        "artifact_score": artifact_score(hr_image, pin_mask=None),
        "raw_control_agreement": raw_control_agreement(_gradient_magnitude(hr_image), _gradient_magnitude(raw_control_hr)),
        "p95_gradient": p95_gradient(hr_image),
        "lambda_tv": float(args.lambda_tv),
        "best_step": int(len(convergence)),
        "final_step": int(len(convergence)),
        "elapsed_sec": elapsed,
        "coord_aspect_mode": str(args.coord_aspect_mode),
        "stage_gate": "complete" if np.isfinite(hr_image).all() else "failed_nonfinite_image",
        "source": "algos/ep06_sr_poc/src/map_tv.reconstruct_map_tv via EP08 loader/split",
    }

    np.save(args.output_dir / "hr_image.npy", hr_image.astype(np.float32))
    np.save(args.output_dir / "hr_raw_control.npy", raw_control_hr.astype(np.float32))
    save_json(args.output_dir / "metrics.json", metrics)
    _write_metrics_csv(args.output_dir / "metrics.csv", metrics)
    save_json(
        args.output_dir / "split_indices.json",
        {
            "seed": int(args.seed),
            "train_indices": train_indices.tolist(),
            "val_indices": val_indices.tolist(),
            "val_mask": val_mask.tolist(),
        },
    )
    save_json(args.output_dir / "config_used.json", {"config": _jsonable(cfg), "metadata": _jsonable(metadata)})
    _write_convergence_csv(args.output_dir / "convergence.csv", convergence)
    _save_convergence_curve(convergence, args.output_dir / "training_curve.png")
    save_image_figure(hr_image, args.output_dir / "hr_highpass.png", title="EP06 MAP-TV Stage 3 HR highpass")
    save_image_figure(raw_control_hr, args.output_dir / "hr_raw_control.png", title="EP06 MAP-TV Stage 3 raw-control bicubic reference")
    if split_a_image is not None and split_b_image is not None:
        save_image_figure(split_a_image - split_b_image, args.output_dir / "split_half_difference.png", title="EP06 MAP-TV Stage 3 split-half difference")
    print(f"saved EP06 MAP-TV Stage 3 baseline to {args.output_dir}")


def main(argv: Sequence[str] | None = None) -> None:
    run_map_tv_stage3_baseline(parse_args(argv))


if __name__ == "__main__":
    main()


__all__ = [
    "DEFAULT_N_FRAMES",
    "default_output_dir",
    "finalize_args",
    "parse_args",
    "patch_shape_label",
    "run_map_tv_stage3_baseline",
]
