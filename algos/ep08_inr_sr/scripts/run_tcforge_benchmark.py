#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import deepinv as dinv
import matplotlib.pyplot as plt
import numpy as np
import torch

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

EP08_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EP08_ROOT.parent.parent
CORE_SRC = PROJECT_ROOT / "core" / "src"
TCFORGE_SRC = PROJECT_ROOT / "tcforge" / "src"
for path in (CORE_SRC, TCFORGE_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ep08.forward import ForwardOperator
from ep08.models import DeepDecoder, Siren, Wire
from ep08.trainer import INRTrainer, TrainConfig
from ep08.utils import save_json, set_seed
from thermal_core.plotting import savefig_academic, setup_academic_style
from tcforge import (
    add_noise,
    build_scene_mask_with_metadata,
    edge_map,
    generate_lr_burst,
    highpass_preprocess,
    ideal_phase_grid,
    render_temperature_field,
)


METHODS = ("siren", "wire", "deep_decoder", "deepinv_dip")
METHOD_LABELS = {
    "siren": "SIREN",
    "wire": "WIRE",
    "deep_decoder": "Deep Decoder",
    "deepinv_dip": "DeepInverse-DIP",
}


def _global_ssim(a: np.ndarray, b: np.ndarray) -> float:
    x = np.asarray(a, dtype=np.float64)
    y = np.asarray(b, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {x.shape} vs {y.shape}")
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    if x.size == 0:
        return float("nan")
    data_range = float(max(np.max(x), np.max(y)) - min(np.min(x), np.min(y)))
    c1 = (0.01 * data_range) ** 2 + np.finfo(np.float64).eps
    c2 = (0.03 * data_range) ** 2 + np.finfo(np.float64).eps
    mux, muy = float(np.mean(x)), float(np.mean(y))
    vx = float(np.mean((x - mux) ** 2))
    vy = float(np.mean((y - muy) ** 2))
    cov = float(np.mean((x - mux) * (y - muy)))
    return float(((2 * mux * muy + c1) * (2 * cov + c2)) / ((mux * mux + muy * muy + c1) * (vx + vy + c2)))


def _psnr(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    err = float(np.mean((pred - target) ** 2))
    if err <= 0:
        return float("inf")
    data_range = float(np.max(target) - np.min(target))
    if data_range <= 0:
        return float("nan")
    return float(20.0 * np.log10(data_range) - 10.0 * np.log10(err))


def _nrmse(pred: np.ndarray, target: np.ndarray) -> float:
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    rmse = float(np.sqrt(np.mean((pred - target) ** 2)))
    denom = float(np.std(target))
    return rmse / max(denom, np.finfo(np.float64).eps)


def _mae(pred: np.ndarray, target: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(pred, dtype=np.float64) - np.asarray(target, dtype=np.float64))))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


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


def _make_scene(
    scene_dir: Path,
    *,
    lr_shape: tuple[int, int],
    n_frames: int,
    seed: int,
    scale: int,
    workers: int,
) -> dict[str, np.ndarray]:
    hr_shape = (int(lr_shape[0]) * scale, int(lr_shape[1]) * scale)
    mask, geometry = build_scene_mask_with_metadata(
        "easy",
        seed,
        canvas_shape=hr_shape,
        pixel_size_um=20.0,
        scale=scale,
        rotation_deg_center=47.6,
        rotation_jitter_deg=0.2,
    )
    hr_temperature = render_temperature_field(
        mask,
        t_bg_c=0.0,
        delta_t_c=1.0,
        low_freq_amplitude_c=0.05,
        low_freq_sigma_px=24.0,
        seed=seed + 1,
    ).astype(np.float32)
    shifts = ideal_phase_grid(n_frames=n_frames, scale=scale, phase_steps=4, jitter_std_px=0.01, seed=seed + 2)
    lr_raw = generate_lr_burst(
        hr_temperature,
        shifts,
        forward_mode="exact_ep06_point",
        psf_sigma_lr_px=1.0,
        scale=scale,
        workers=workers,
    )
    lr_raw = add_noise(lr_raw, noise_sigma_c=0.003, seed=seed + 3).astype(np.float32)
    lr_highpass = highpass_preprocess(lr_raw, sigma_bg=5.0, workers=workers, mode="nearest").astype(np.float32)
    hr_highpass = highpass_preprocess(hr_temperature[np.newaxis], sigma_bg=5.0, mode="nearest")[0].astype(np.float32)

    scene_dir.mkdir(parents=True, exist_ok=True)
    np.save(scene_dir / "hr_temperature_2x.npy", hr_temperature)
    np.save(scene_dir / "hr_highpass_2x.npy", hr_highpass)
    np.save(scene_dir / "hr_mask_2x.npy", mask.astype(np.uint8))
    np.save(scene_dir / "hr_edge_map_2x.npy", edge_map(mask).astype(np.float32))
    np.save(scene_dir / "lr_burst_raw.npy", lr_raw)
    np.save(scene_dir / "lr_burst_highpass.npy", lr_highpass)
    np.save(scene_dir / "shifts.npy", shifts.astype(np.float32))
    save_json(
        scene_dir / "metadata.json",
        {
            "scene_id": "ep08_tcforge_benchmark",
            "scale": int(scale),
            "lr_shape": list(lr_shape),
            "hr_shape": list(hr_shape),
            "n_frames": int(n_frames),
            "seed": int(seed),
            "geometry": geometry,
            "physics": {
                "psf_sigma_lr_px": 1.0,
                "noise_sigma_c": 0.003,
                "highpass_sigma_px_for_hr_and_lr": 5.0,
                "comparison_domain": "highpass_preprocess(hr_temperature_2x[np.newaxis], sigma_bg=5.0, mode='nearest')[0]",
            },
        },
    )
    return {
        "hr_temperature": hr_temperature,
        "hr_highpass": hr_highpass,
        "lr_raw": lr_raw,
        "lr_highpass": lr_highpass,
        "shifts": shifts.astype(np.float32),
    }


def _native_latent_spatial(hr_shape: tuple[int, int], n_upsamples: int) -> tuple[int, int]:
    factor = 2 ** int(n_upsamples)
    if hr_shape[0] % factor or hr_shape[1] % factor:
        raise ValueError(f"HR shape {hr_shape} must be divisible by 2**{n_upsamples}={factor} for native Deep Decoder output")
    return (hr_shape[0] // factor, hr_shape[1] // factor)


def _train_coord_model(
    method: str,
    observations: torch.Tensor,
    shifts: torch.Tensor,
    forward_operator: ForwardOperator,
    *,
    hr_shape: tuple[int, int],
    iterations: int,
    lr: float,
    batch_k: int,
    seed: int,
    device: torch.device,
    hidden_features: int,
    hidden_layers: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    set_seed(seed)
    if method == "siren":
        model = Siren(hidden_features=hidden_features, hidden_layers=hidden_layers, first_omega_0=30.0, hidden_omega_0=30.0)
    elif method == "wire":
        model = Wire(
            hidden_features=hidden_features,
            hidden_layers=hidden_layers,
            first_omega_0=20.0,
            hidden_omega_0=20.0,
            first_sigma_0=10.0,
            hidden_sigma_0=10.0,
        )
    else:
        raise ValueError(f"unexpected coordinate model: {method}")
    trainer = INRTrainer(
        model,
        observations,
        shifts,
        hr_shape=hr_shape,
        forward_operator=forward_operator,
        device=device,
        config=TrainConfig(
            max_iter=iterations,
            lr=lr,
            warmup_steps=min(50, iterations),
            batch_k=batch_k,
            val_interval=0,
            early_stop_patience=0,
            seed=seed,
        ),
        train_indices=np.arange(int(observations.shape[0])),
        val_indices=[],
    )
    result = trainer.fit()
    return result.image.numpy().astype(np.float32), {"best_loss": float(result.best_loss), "best_step": int(result.best_step)}


def _train_deep_decoder(
    observations: torch.Tensor,
    shifts: torch.Tensor,
    forward_operator: ForwardOperator,
    *,
    hr_shape: tuple[int, int],
    iterations: int,
    lr: float,
    batch_k: int,
    seed: int,
    device: torch.device,
    latent_channels: int,
    hidden_channels: tuple[int, ...],
) -> tuple[np.ndarray, dict[str, Any]]:
    set_seed(seed)
    latent_spatial = _native_latent_spatial(hr_shape, len(hidden_channels))
    model = DeepDecoder(
        latent_channels=latent_channels,
        hidden_channels=hidden_channels,
        latent_spatial=latent_spatial,
        seed=seed,
    )
    trainer = INRTrainer(
        model,
        observations,
        shifts,
        hr_shape=hr_shape,
        forward_operator=forward_operator,
        device=device,
        config=TrainConfig(
            max_iter=iterations,
            lr=lr,
            warmup_steps=min(50, iterations),
            batch_k=batch_k,
            val_interval=0,
            early_stop_patience=0,
            seed=seed,
        ),
        train_indices=np.arange(int(observations.shape[0])),
        val_indices=[],
    )
    result = trainer.fit()
    return result.image.numpy().astype(np.float32), {
        "best_loss": float(result.best_loss),
        "best_step": int(result.best_step),
        "latent_spatial": list(latent_spatial),
        "native_output_shape": list(hr_shape),
    }


def _render_convdecoder(backbone: torch.nn.Module, z: torch.Tensor, hr_shape: tuple[int, int]) -> torch.Tensor:
    pred = backbone(z)
    expected = (1, 1, int(hr_shape[0]), int(hr_shape[1]))
    if tuple(pred.shape) != expected:
        raise ValueError(f"ConvDecoder output shape mismatch: got {tuple(pred.shape)}, expected {expected}")
    return pred[0, 0]


def _predict_batch(forward_operator: ForwardOperator, x_hr: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    return torch.stack([forward_operator(x_hr, int(idx)) for idx in indices.detach().cpu().tolist()], dim=0)


def _train_deepinv_dip(
    observations: torch.Tensor,
    forward_operator: ForwardOperator,
    *,
    hr_shape: tuple[int, int],
    iterations: int,
    lr: float,
    batch_k: int,
    seed: int,
    device: torch.device,
    channels: int,
    layers: int,
    in_spatial: tuple[int, int],
    early_stop_patience: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    set_seed(seed)
    backbone = dinv.models.ConvDecoder(
        img_size=(1, int(hr_shape[0]), int(hr_shape[1])),
        in_size=tuple(int(v) for v in in_spatial),
        layers=int(layers),
        channels=int(channels),
    ).to(device)
    z = torch.randn((1, int(channels), int(in_spatial[0]), int(in_spatial[1])), dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(backbone.parameters(), lr=float(lr))
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    best_loss = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float]] = []
    stale = 0
    n_frames = int(observations.shape[0])
    start = time.perf_counter()
    for step in range(1, int(iterations) + 1):
        order = torch.randperm(n_frames, generator=generator, device=device)[: min(batch_k, n_frames)]
        optimizer.zero_grad(set_to_none=True)
        x_hr = _render_convdecoder(backbone, z, hr_shape)
        pred = _predict_batch(forward_operator, x_hr, order)
        target = observations.index_select(0, order)
        loss = torch.mean((pred - target).square())
        loss.backward()
        optimizer.step()
        loss_value = float(loss.detach().cpu())
        history.append({"step": float(step), "train_loss": loss_value})
        if loss_value < best_loss - 1.0e-8:
            best_loss = loss_value
            best_step = int(step)
            best_state = {key: value.detach().cpu().clone() for key, value in backbone.state_dict().items()}
            stale = 0
        else:
            stale += 1
        if early_stop_patience > 0 and stale >= early_stop_patience:
            break
    if best_state is not None:
        backbone.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    backbone.eval()
    with torch.no_grad():
        image = _render_convdecoder(backbone, z, hr_shape).detach().cpu().numpy().astype(np.float32)
    return image, {
        "best_loss": float(best_loss),
        "best_step": int(best_step),
        "final_step": int(history[-1]["step"]) if history else 0,
        "elapsed_sec": float(time.perf_counter() - start),
        "deepinv_version": getattr(dinv, "__version__", "unknown"),
        "dip_loop": "custom_bounded_adam",
    }


def _metric_row(method: str, image: np.ndarray, target: np.ndarray, extra: dict[str, Any], scene_dir: Path) -> dict[str, Any]:
    return {
        "method": METHOD_LABELS[method],
        "method_key": method,
        "domain": "hr_highpass",
        "psnr_db": _psnr(image, target),
        "global_ssim": _global_ssim(image, target),
        "nrmse": _nrmse(image, target),
        "mae_c": _mae(image, target),
        "best_loss": extra.get("best_loss"),
        "best_step": extra.get("best_step"),
        "final_step": extra.get("final_step", extra.get("best_step")),
        "scene_dir": str(scene_dir),
        **{key: value for key, value in extra.items() if key not in {"best_loss", "best_step", "final_step"}},
    }


def _symmetric_limits(images: list[np.ndarray], percentile: float = 99.0) -> tuple[float, float]:
    values = np.concatenate([np.ravel(np.asarray(image)[np.isfinite(image)]) for image in images])
    limit = float(np.percentile(np.abs(values), percentile)) if values.size else 1.0
    if not math.isfinite(limit) or limit <= 0:
        limit = 1.0
    return -limit, limit


def _save_side_by_side(rows: list[dict[str, Any]], images: dict[str, np.ndarray], path: Path) -> None:
    setup_academic_style()
    vmin, vmax = _symmetric_limits(list(images.values()))
    fig, axes = plt.subplots(1, len(images), figsize=(2.3 * len(images), 2.8), constrained_layout=True)
    axes_arr = np.atleast_1d(axes)
    for ax, (name, image) in zip(axes_arr, images.items(), strict=True):
        im = ax.imshow(image, cmap="RdBu_r", vmin=vmin, vmax=vmax, origin="upper")
        ax.set_title(name, fontsize=8)
        ax.set_axis_off()
    fig.colorbar(im, ax=axes_arr.tolist(), fraction=0.025, pad=0.01)
    caption = ", ".join(f"{row['method']}: {row['psnr_db']:.2f} dB" for row in rows)
    fig.suptitle(f"TCForge synthetic HR highpass benchmark. {caption}", fontsize=8)
    savefig_academic(fig, path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark EP08 INR/DIP methods on synthetic TCForge HR-GT highpass data.")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "ep08_inr_sr" / "tcforge_benchmark")
    parser.add_argument("--scene-dir", type=Path, default=PROJECT_ROOT / "data" / "synthetic" / "ep08_tcforge_benchmark" / "scenes" / "easy_seed42")
    parser.add_argument("--lr-shape", nargs=2, type=int, default=[256, 256], help="LR H W. Default gives 512x512 HR at scale=2.")
    parser.add_argument("--n-frames", type=int, default=32)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--iterations", type=int, default=800)
    parser.add_argument("--siren-iterations", type=int, default=None)
    parser.add_argument("--wire-iterations", type=int, default=None)
    parser.add_argument("--deep-decoder-iterations", type=int, default=None)
    parser.add_argument("--deepinv-iterations", type=int, default=None)
    parser.add_argument("--lr", type=float, default=5.0e-4)
    parser.add_argument("--batch-k", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--hidden-features", type=int, default=64)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--deep-decoder-latent-channels", type=int, default=32)
    parser.add_argument("--deep-decoder-hidden-channels", default="64,64,32")
    parser.add_argument("--deepinv-channels", type=int, default=32)
    parser.add_argument("--deepinv-layers", type=int, default=4)
    parser.add_argument("--deepinv-in-spatial", default="32,32")
    parser.add_argument("--deepinv-early-stop-patience", type=int, default=200)
    return parser.parse_args()


def _iterations_for(args: argparse.Namespace, method: str) -> int:
    value = {
        "siren": args.siren_iterations,
        "wire": args.wire_iterations,
        "deep_decoder": args.deep_decoder_iterations,
        "deepinv_dip": args.deepinv_iterations,
    }[method]
    return int(args.iterations if value is None else value)


def _parse_hw(text: str) -> tuple[int, int]:
    values = [int(v) for v in str(text).split(",") if v.strip()]
    if len(values) != 2:
        raise ValueError("expected H,W")
    return (values[0], values[1])


def main() -> None:
    args = parse_args()
    if int(args.scale) != 2:
        raise ValueError("EP08 benchmark is defined for scale=2")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    arrays = _make_scene(
        args.scene_dir,
        lr_shape=(int(args.lr_shape[0]), int(args.lr_shape[1])),
        n_frames=int(args.n_frames),
        seed=int(args.seed),
        scale=int(args.scale),
        workers=int(args.workers),
    )
    device = torch.device(str(args.device))
    observations = torch.as_tensor(arrays["lr_highpass"], dtype=torch.float32, device=device)
    shifts = torch.as_tensor(arrays["shifts"], dtype=torch.float32, device=device)
    target = arrays["hr_highpass"]
    hr_shape = tuple(int(v) for v in target.shape)
    op = ForwardOperator(
        hr_shape=hr_shape,
        lr_shape=tuple(int(v) for v in observations.shape[-2:]),
        shifts=shifts,
        psf_sigma=1.0,
        scale=int(args.scale),
    ).to(device)

    rows: list[dict[str, Any]] = []
    images: dict[str, np.ndarray] = {"Synthetic HR highpass GT": target}
    hidden_channels = tuple(int(v) for v in str(args.deep_decoder_hidden_channels).split(",") if v.strip())
    deepinv_in_spatial = _parse_hw(args.deepinv_in_spatial)

    for offset, method in enumerate(args.methods):
        method_seed = int(args.seed) + 100 * (offset + 1)
        iterations = _iterations_for(args, method)
        start = time.perf_counter()
        if method in {"siren", "wire"}:
            image, extra = _train_coord_model(
                method,
                observations,
                shifts,
                op,
                hr_shape=hr_shape,
                iterations=iterations,
                lr=float(args.lr),
                batch_k=int(args.batch_k),
                seed=method_seed,
                device=device,
                hidden_features=int(args.hidden_features),
                hidden_layers=int(args.hidden_layers),
            )
        elif method == "deep_decoder":
            image, extra = _train_deep_decoder(
                observations,
                shifts,
                op,
                hr_shape=hr_shape,
                iterations=iterations,
                lr=float(args.lr),
                batch_k=int(args.batch_k),
                seed=method_seed,
                device=device,
                latent_channels=int(args.deep_decoder_latent_channels),
                hidden_channels=hidden_channels,
            )
        elif method == "deepinv_dip":
            image, extra = _train_deepinv_dip(
                observations,
                op,
                hr_shape=hr_shape,
                iterations=iterations,
                lr=float(args.lr),
                batch_k=int(args.batch_k),
                seed=method_seed,
                device=device,
                channels=int(args.deepinv_channels),
                layers=int(args.deepinv_layers),
                in_spatial=deepinv_in_spatial,
                early_stop_patience=int(args.deepinv_early_stop_patience),
            )
        else:
            raise AssertionError(f"unhandled method: {method}")
        extra.setdefault("elapsed_sec", float(time.perf_counter() - start))
        np.save(args.output_dir / f"{method}_hr_highpass.npy", image)
        rows.append(_metric_row(method, image, target, extra, args.scene_dir))
        images[METHOD_LABELS[method]] = image
        if device.type == "cuda":
            torch.cuda.empty_cache()

    _write_csv(args.output_dir / "metrics.csv", rows)
    save_json(
        args.output_dir / "metrics.json",
        {
            "rows": _jsonable(rows),
            "config": _jsonable(vars(args)),
            "target": "hr_highpass_2x.npy",
            "target_definition": "highpass_preprocess(hr_temperature_2x[np.newaxis], sigma_bg=5.0, mode='nearest')[0]",
        },
    )
    _save_side_by_side(rows, images, args.output_dir / "tcforge_benchmark_highpass.png")
    print(f"saved TCForge benchmark results to {args.output_dir}")


if __name__ == "__main__":
    main()
