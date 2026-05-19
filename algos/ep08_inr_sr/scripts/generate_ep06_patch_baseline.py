#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EP08_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EP08_ROOT.parent.parent
CORE_SRC = PROJECT_ROOT / "core" / "src"
EP06_SRC = PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src"
for path in (CORE_SRC, EP06_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ep08.metrics import artifact_score, holdout_residual, p95_gradient, raw_control_agreement, split_half_nrmse
from ep08.splits import build_train_val_split
from ep08.stage1 import _gradient_magnitude, _upscale_raw_control, load_dataset_for_config, save_image_figure
from ep08.utils import save_json, set_seed


PROJECT_OUTPUT = PROJECT_ROOT / "output" / "ep08_inr_sr"


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if torch.is_tensor(value):
        return _jsonable(value.detach().cpu().numpy())
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_metrics_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(_jsonable(row))


def _import_ep06_map_tv():
    try:
        from map_tv import reconstruct_map_tv
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "EP06 MAP-TV import failed. Run this from the repository with EP06 dependencies installed "
            "(for example: cd algos/ep06_sr_poc && uv sync), or verify algos/ep06_sr_poc/src is present. "
            f"Original error: {exc!r}"
        ) from exc
    return reconstruct_map_tv


def _cfg(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "data": {
            "scale": 2,
            "default_n_frames": int(args.n_frames),
            "patch_shape": (int(args.patch_size), int(args.patch_size)),
            "default_patch_size_lr_px": int(args.patch_size),
            "val_ratio": float(args.val_ratio),
        },
        "forward": {"psf_sigma_lr_px": float(args.psf_sigma)},
        "preprocess": {"highpass_sigma_bg_lr_px": float(args.highpass_sigma), "highpass_mode": "nearest"},
        "runtime": {
            "data_mode": "real",
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


def _run_map_tv(
    frames: np.ndarray,
    shifts: np.ndarray,
    *,
    lambda_tv: float,
    max_iter: int,
    step_size: float,
    psf_sigma: float,
    workers: int,
    tol: float,
    tv_inner_iter: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    reconstruct_map_tv = _import_ep06_map_tv()
    image, records = reconstruct_map_tv(
        frames,
        shifts,
        initial="saa",
        lambda_tv=float(lambda_tv),
        max_iter=int(max_iter),
        step_size=float(step_size),
        psf_sigma=float(psf_sigma),
        scale=2,
        workers=int(workers),
        tol=float(tol),
        tv_inner_iter=int(tv_inner_iter),
        use_fista=True,
    )
    if hasattr(records, "to_dict"):
        records = records.to_dict(orient="records")
    return np.asarray(image, dtype=np.float32), list(records)


def _split_train_indices(train_indices: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if train_indices.size < 2:
        return train_indices.copy(), train_indices.copy()
    rng = np.random.default_rng(seed + 202)
    shuffled = train_indices.copy()
    rng.shuffle(shuffled)
    mid = max(1, len(shuffled) // 2)
    return np.sort(shuffled[:mid]), np.sort(shuffled[mid:])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate EP06 MAP-TV baseline on the EP08 32-frame 256x256 patch protocol.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_OUTPUT / "ep06_patch_baseline")
    parser.add_argument("--n-frames", type=int, default=32)
    parser.add_argument("--patch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--alignment-method", default="contour_refined")
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--frame-audit-path", type=Path, default=None)
    parser.add_argument("--highpass-sigma", type=float, default=5.0)
    parser.add_argument("--psf-sigma", type=float, default=1.0)
    parser.add_argument("--noise-sigma", type=float, default=0.0724)
    parser.add_argument("--lambda-tv", type=float, default=1.0e-3)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--split-half-max-iter", type=int, default=60)
    parser.add_argument("--skip-split-half", action="store_true")
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1.0e-4)
    parser.add_argument("--tv-inner-iter", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
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
        np.save(args.output_dir / "split_half_a.npy", split_a_image)
        np.save(args.output_dir / "split_half_b.npy", split_b_image)

    metrics = {
        "method": "ep06_map_tv",
        "method_label": "EP06 MAP-TV",
        "family": "classic_opt",
        "data_mode": "real",
        "protocol": f"ep08_{int(args.n_frames)}_frame_{int(args.patch_size)}_patch_seed{int(args.seed)}_split",
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
        "stage_gate": "complete" if np.isfinite(hr_image).all() else "failed_nonfinite_image",
        "source": "algos/ep06_sr_poc/src/map_tv.reconstruct_map_tv via EP08 loader/split",
    }

    np.save(args.output_dir / "hr_image.npy", hr_image.astype(np.float32))
    np.save(args.output_dir / "hr_raw_control.npy", raw_control_hr.astype(np.float32))
    save_json(args.output_dir / "metrics.json", metrics)
    _write_metrics_csv(args.output_dir / "metrics.csv", metrics)
    save_json(args.output_dir / "split_indices.json", {"seed": int(args.seed), "train_indices": train_indices.tolist(), "val_indices": val_indices.tolist(), "val_mask": val_mask.tolist()})
    save_json(args.output_dir / "config_used.json", {"config": _jsonable(cfg), "metadata": _jsonable(metadata)})
    with (args.output_dir / "convergence.csv").open("w", encoding="utf-8", newline="") as f:
        fieldnames = sorted({key for row in convergence for key in row}) if convergence else ["iteration"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(convergence)
    save_image_figure(hr_image, args.output_dir / "hr_highpass.png", title="EP06 MAP-TV patch HR highpass")
    save_image_figure(raw_control_hr, args.output_dir / "hr_raw_control.png", title="EP06 MAP-TV raw-control bicubic reference")
    if split_a_image is not None and split_b_image is not None:
        save_image_figure(split_a_image - split_b_image, args.output_dir / "split_half_difference.png", title="EP06 MAP-TV split-half difference")
    print(f"saved EP06 patch baseline to {args.output_dir}")


if __name__ == "__main__":
    main()
