#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

try:
    import deepinv as dinv
except ModuleNotFoundError:  # pragma: no cover - exercised only in lean test envs
    dinv = None

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ep08.metrics import artifact_score, holdout_residual, p95_gradient, raw_control_agreement, split_half_nrmse
from ep08.splits import build_train_val_split
from ep08.stage1 import (
    _gradient_magnitude,
    _split_train_indices,
    _upscale_raw_control,
    load_dataset_for_config,
    parse_patch_shape,
    save_image_figure,
    save_training_curve,
)
from ep08.utils import save_json, set_seed

EP08_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = EP08_ROOT.parent.parent
PROJECT_OUTPUT = PROJECT_ROOT / "output" / "ep08_inr_sr"


@dataclass(slots=True)
class ConvDecoderDipResult:
    image: torch.Tensor
    history: list[dict[str, float]]
    best_loss: float
    best_step: int
    final_step: int
    elapsed_sec: float
    latent: torch.Tensor


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _default_config() -> dict[str, Any]:
    return {
        "deepinv": {"channels": 32, "layers": 4, "in_spatial": [32, 32], "verbose": True},
        "data": {"scale": 2, "default_n_frames": 32, "default_patch_size_lr_px": 256, "val_ratio": 0.2},
        "forward": {"psf_sigma_lr_px": 1.0},
        "preprocess": {"highpass_sigma_bg_lr_px": 5.0, "highpass_mode": "nearest"},
        "train": {
            "lr": 5.0e-4,
            "max_iter": 10000,
            "val_interval": 100,
            "early_stop_patience": 1000,
            "early_stop_min_delta": 1.0e-6,
            "split_half_max_iter": 3000,
            "split_half_enabled": True,
        },
        "runtime": {
            "data_mode": "real",
            "device": "cuda:0",
            "workers": 1,
            "alignment_method": "contour_refined",
            "seed": 42,
            "output_dir": "output/ep08_inr_sr/deepinv_dip_stage2",
            "data_dir": None,
            "frame_audit_path": None,
        },
        "metrics": {"noise_sigma": 0.0724},
    }


def _load_config(path: Path | None) -> dict[str, Any]:
    cfg = _default_config()
    if path is not None and path.exists():
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"config must contain a mapping: {path}")
        _deep_update(cfg, payload)
    return cfg


def _resolve_output_dir(path_value: str | Path | None) -> Path:
    if path_value is None:
        return PROJECT_OUTPUT / "deepinv_dip_stage2"
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "output":
        return PROJECT_ROOT / path
    return Path.cwd() / path


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
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _write_metrics_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writeheader()
        writer.writerow(_jsonable(row))


def _write_history_outputs(out_dir: Path, history: list[dict[str, float]]) -> None:
    save_json(out_dir / "training_history.json", {"history": history})
    fieldnames: list[str] = []
    for row in history:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        return
    with (out_dir / "training_history.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def _coerce_positive_int(value: Any, *, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    if isinstance(value, str):
        text = value.strip()
        if not text or not text.lstrip("+-").isdigit():
            raise ValueError(f"{label} must be a positive integer")
        parsed = int(text)
    else:
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} must be a positive integer") from exc
        if str(parsed) != str(value) and not isinstance(value, (int, np.integer)):
            raise ValueError(f"{label} must be a positive integer")
    if parsed <= 0:
        raise ValueError(f"{label} must be positive")
    return parsed


def parse_spatial_shape(value: Any) -> tuple[int, int]:
    """Parse H,W latent spatial sizes without assuming a square grid."""
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            raise ValueError("spatial shape must not be empty")
        separator = "," if "," in text else "x" if "x" in text else None
        if separator is None:
            size = _coerce_positive_int(text, label="spatial shape")
            return size, size
        parts = [part.strip() for part in text.split(separator)]
        if len(parts) != 2 or any(not part for part in parts):
            raise ValueError("spatial shape must be H,W or HxW")
        height = _coerce_positive_int(parts[0], label="spatial height")
        width = _coerce_positive_int(parts[1], label="spatial width")
        return height, width
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError("spatial shape list/tuple must be [height, width]")
        height = _coerce_positive_int(value[0], label="spatial height")
        width = _coerce_positive_int(value[1], label="spatial width")
        return height, width
    size = _coerce_positive_int(value, label="spatial shape")
    return size, size


def _build_convdecoder(
    cfg: dict[str, Any],
    hr_shape: tuple[int, int],
    device: torch.device,
) -> tuple[torch.nn.Module, torch.Tensor]:
    if dinv is None:
        raise ModuleNotFoundError("deepinv is required to build DeepInverse ConvDecoder-DIP")
    dip_cfg = cfg["deepinv"]
    channels = int(dip_cfg["channels"])
    in_spatial = parse_spatial_shape(dip_cfg["in_spatial"])
    generator_img_size = (1, int(hr_shape[0]), int(hr_shape[1]))
    backbone = dinv.models.ConvDecoder(
        img_size=generator_img_size,
        in_size=in_spatial,
        layers=int(dip_cfg["layers"]),
        channels=channels,
    ).to(device)
    z = torch.randn((1, channels, *in_spatial), dtype=torch.float32, device=device)
    return backbone, z


def _render_backbone(backbone: torch.nn.Module, z: torch.Tensor, hr_shape: tuple[int, int]) -> torch.Tensor:
    pred = backbone(z)
    expected = (1, 1, int(hr_shape[0]), int(hr_shape[1]))
    if tuple(pred.shape) != expected:
        raise ValueError(f"ConvDecoder output shape mismatch: got {tuple(pred.shape)}, expected {expected}")
    return pred[0, 0]


def _predict_frames(forward_operator: torch.nn.Module, x_hr: torch.Tensor, indices: np.ndarray | torch.Tensor) -> torch.Tensor:
    if torch.is_tensor(indices):
        index_list = [int(v) for v in indices.detach().cpu().tolist()]
    else:
        index_list = [int(v) for v in np.asarray(indices).reshape(-1).tolist()]
    if not index_list:
        raise ValueError("frame indices must not be empty")
    return torch.stack([forward_operator(x_hr, index) for index in index_list], dim=0)


@torch.no_grad()
def _evaluate_loss(
    backbone: torch.nn.Module,
    z: torch.Tensor,
    hr_shape: tuple[int, int],
    observations: torch.Tensor,
    forward_operator: torch.nn.Module,
    indices: np.ndarray,
) -> float:
    if indices.size == 0:
        return float("nan")
    backbone.eval()
    x_hr = _render_backbone(backbone, z, hr_shape)
    pred = _predict_frames(forward_operator, x_hr, indices)
    target = observations.index_select(0, torch.as_tensor(indices, dtype=torch.long, device=observations.device))
    return float(torch.mean((pred - target).square()).detach().cpu())


def _state_dict_cpu(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _train_dip_once(
    cfg: dict[str, Any],
    observations: torch.Tensor,
    forward_operator: torch.nn.Module,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    device: torch.device,
    *,
    max_iter: int | None = None,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.nn.Module, ConvDecoderDipResult]:
    local_cfg = deepcopy(cfg)
    if max_iter is not None:
        local_cfg["train"]["max_iter"] = int(max_iter)
    set_seed(int(local_cfg["runtime"]["seed"] if seed is None else seed))
    hr_shape = (int(observations.shape[-2]) * int(local_cfg["data"]["scale"]), int(observations.shape[-1]) * int(local_cfg["data"]["scale"]))
    backbone, z = _build_convdecoder(local_cfg, hr_shape, device)
    train_indices = np.asarray(train_indices, dtype=np.int64).reshape(-1)
    val_indices = np.asarray(val_indices, dtype=np.int64).reshape(-1)
    if train_indices.size == 0:
        raise ValueError("train_indices must not be empty")
    train_cfg = local_cfg["train"]
    total_steps = int(train_cfg["max_iter"])
    val_interval = int(train_cfg.get("val_interval", 100))
    patience = int(train_cfg.get("early_stop_patience", 1000))
    min_delta = float(train_cfg.get("early_stop_min_delta", 1.0e-6))
    batch_k_value = train_cfg.get("batch_k")
    batch_k = train_indices.size if batch_k_value is None else int(batch_k_value)
    if batch_k <= 0:
        raise ValueError("train.batch_k must be positive when provided")
    batch_k = min(batch_k, train_indices.size)
    check_indices = train_indices[:batch_k]
    check_target = observations.index_select(0, torch.as_tensor(check_indices, dtype=torch.long, device=device))
    with torch.no_grad():
        dry_run = _predict_frames(forward_operator, torch.zeros(hr_shape, dtype=observations.dtype, device=device), check_indices)
    if tuple(dry_run.shape) != tuple(check_target.shape):
        raise ValueError(f"forward shape mismatch: A(x)={tuple(dry_run.shape)} but y={tuple(check_target.shape)}")
    batch_generator = torch.Generator(device="cpu")
    batch_generator.manual_seed(int(local_cfg["runtime"]["seed"] if seed is None else seed))
    optimizer = torch.optim.Adam(backbone.parameters(), lr=float(train_cfg["lr"]))
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None

    start = time.perf_counter()
    for step in range(1, total_steps + 1):
        backbone.train()
        optimizer.zero_grad(set_to_none=True)
        if batch_k == train_indices.size:
            batch_indices = train_indices
        else:
            batch_order = torch.randperm(train_indices.size, generator=batch_generator)[:batch_k].numpy()
            batch_indices = train_indices[batch_order]
        batch_index_t = torch.as_tensor(batch_indices, dtype=torch.long, device=device)
        y_batch = observations.index_select(0, batch_index_t)
        x_hr = _render_backbone(backbone, z, hr_shape)
        pred = _predict_frames(forward_operator, x_hr, batch_indices)
        loss = torch.mean((pred - y_batch).square())
        loss.backward()
        optimizer.step()

        record: dict[str, float] = {
            "step": float(step),
            "train_loss": float(loss.detach().cpu()),
            "batch_k": float(batch_indices.size),
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        should_validate = val_interval <= 0 or step == 1 or step % val_interval == 0
        if should_validate:
            if val_indices.size:
                record["holdout_loss"] = _evaluate_loss(backbone, z, hr_shape, observations, forward_operator, val_indices)
            record["train_set_loss"] = _evaluate_loss(backbone, z, hr_shape, observations, forward_operator, train_indices)
            holdout_loss = record.get("holdout_loss")
            monitor = holdout_loss if holdout_loss is not None and np.isfinite(holdout_loss) else record["train_set_loss"]
            if monitor + min_delta < best_loss:
                best_loss = float(monitor)
                best_step = int(step)
                best_state = _state_dict_cpu(backbone)
        history.append(record)

        if best_step > 0 and patience > 0 and should_validate and step - best_step >= patience:
            break

    elapsed = float(time.perf_counter() - start)
    if best_state is not None:
        backbone.load_state_dict({key: value.to(device) for key, value in best_state.items()})
    backbone.eval()
    with torch.no_grad():
        hr_result = _render_backbone(backbone, z, hr_shape).detach().cpu()
    final_step = int(history[-1]["step"]) if history else 0
    result = ConvDecoderDipResult(
        image=hr_result,
        history=history,
        best_loss=float(best_loss),
        best_step=int(best_step),
        final_step=final_step,
        elapsed_sec=elapsed,
        latent=z.detach().cpu().clone(),
    )
    return hr_result, backbone, result


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train DeepInverse ConvDecoder-DIP on EP08 data.")
    parser.add_argument("--config", type=Path, default=EP08_ROOT / "configs" / "deepinv_dip.yaml")
    parser.add_argument("--data-mode", choices=["synthetic", "real"], default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--n-frames", type=int, default=None)
    parser.add_argument("--patch-shape", type=parse_patch_shape, default=argparse.SUPPRESS, help="LR patch shape: H,W, a single int, or full/None")
    parser.add_argument("--patch-size", type=int, default=None, help="Backward-compatible square LR patch size in pixels")
    parser.add_argument("--batch-k", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--val-interval", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None, dest="early_stop_patience")
    parser.add_argument("--early-stopping-min-delta", type=float, default=None, dest="early_stop_min_delta")
    parser.add_argument("--split-half-max-iter", type=int, default=None)
    parser.add_argument("--skip-split-half", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--alignment-method", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--frame-audit-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--layers", type=int, default=None)
    parser.add_argument("--in-spatial", type=parse_spatial_shape, default=None, help="HxW latent noise size, e.g. 30,40 or 30x40")
    parser.add_argument("--quiet", action="store_true")
    return parser.parse_args()


parse_args = _parse_args


def _apply_cli(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    cfg = deepcopy(cfg)
    for cli_key, section, cfg_key in (
        ("data_mode", "runtime", "data_mode"),
        ("device", "runtime", "device"),
        ("workers", "runtime", "workers"),
        ("alignment_method", "runtime", "alignment_method"),
        ("seed", "runtime", "seed"),
        ("output_dir", "runtime", "output_dir"),
        ("max_iter", "train", "max_iter"),
        ("batch_k", "train", "batch_k"),
        ("lr", "train", "lr"),
        ("val_interval", "train", "val_interval"),
        ("early_stop_patience", "train", "early_stop_patience"),
        ("early_stop_min_delta", "train", "early_stop_min_delta"),
        ("split_half_max_iter", "train", "split_half_max_iter"),
        ("val_ratio", "data", "val_ratio"),
        ("channels", "deepinv", "channels"),
        ("layers", "deepinv", "layers"),
    ):
        value = getattr(args, cli_key, None)
        if value is not None:
            cfg.setdefault(section, {})[cfg_key] = value
    if args.n_frames is not None:
        cfg["data"]["default_n_frames"] = int(args.n_frames)
    if args.patch_size is not None:
        cfg["data"]["default_patch_size_lr_px"] = int(args.patch_size)
        if not hasattr(args, "patch_shape"):
            cfg["data"]["patch_shape"] = int(args.patch_size)
    if hasattr(args, "patch_shape"):
        cfg["data"]["patch_shape"] = args.patch_shape
    if args.frame_audit_path is not None:
        cfg["runtime"]["frame_audit_path"] = args.frame_audit_path
    if args.data_dir is not None:
        cfg["runtime"]["data_dir"] = args.data_dir
    if args.in_spatial is not None:
        cfg["deepinv"]["in_spatial"] = list(parse_spatial_shape(args.in_spatial))
    if args.skip_split_half:
        cfg["train"]["split_half_enabled"] = False
    if args.quiet:
        cfg["deepinv"]["verbose"] = False
    return cfg


def run_deepinv_dip_training(args: argparse.Namespace) -> None:
    cfg = _apply_cli(_load_config(args.config), args)
    out_dir = _resolve_output_dir(cfg["runtime"].get("output_dir"))
    cfg["runtime"]["output_dir"] = str(out_dir)
    cfg["runtime"]["config_path"] = str(args.config)
    cfg["runtime"]["project_root"] = str(PROJECT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(str(cfg["runtime"]["device"]))
    set_seed(int(cfg["runtime"]["seed"]))
    observations, shifts, raw_control, forward_operator, metadata = load_dataset_for_config("deep_decoder", cfg, device)
    forward_operator = forward_operator.to(device)
    frame_ids = np.arange(int(observations.shape[0]))
    train_indices, val_indices, val_mask = build_train_val_split(
        frame_ids,
        shifts.detach().cpu().numpy(),
        val_ratio=float(cfg["data"].get("val_ratio", 0.2)),
        seed=int(cfg["runtime"]["seed"]),
    )
    if val_indices.size == 0 and frame_ids.size > 1:
        val_indices = frame_ids[-max(1, frame_ids.size // 5) :]
        train_indices = np.setdiff1d(frame_ids, val_indices)
        val_mask = np.zeros(frame_ids.size, dtype=bool)
        val_mask[val_indices] = True

    save_json(
        out_dir / "split_indices.json",
        {
            "seed": int(cfg["runtime"]["seed"]),
            "val_ratio": float(cfg["data"].get("val_ratio", 0.2)),
            "train_indices": train_indices.tolist(),
            "val_indices": val_indices.tolist(),
            "val_mask": val_mask.tolist(),
        },
    )

    hr_tensor, backbone, train_result = _train_dip_once(
        cfg,
        observations,
        forward_operator,
        train_indices,
        val_indices,
        device,
    )
    hr_image = hr_tensor.numpy().astype(np.float32)
    hr_shape = tuple(int(v) for v in hr_image.shape)
    raw_control_hr = _upscale_raw_control(raw_control, train_indices, hr_shape).astype(np.float32)

    split_score = float("nan")
    split_a_image: np.ndarray | None = None
    split_b_image: np.ndarray | None = None
    if bool(cfg["train"].get("split_half_enabled", True)):
        half_a, half_b = _split_train_indices(train_indices, int(cfg["runtime"]["seed"]))
        split_max_iter = int(cfg["train"].get("split_half_max_iter", 3000))
        split_seed = int(cfg["runtime"]["seed"]) + 101
        # For DIP, split-half should isolate data-subset stability. Different
        # ConvDecoder initializations dominate the score before they reveal
        # useful data sensitivity, so both halves share the same fixed latent
        # and initialization seed.
        split_a_tensor, _, _ = _train_dip_once(
            cfg,
            observations,
            forward_operator,
            half_a,
            val_indices,
            device,
            max_iter=split_max_iter,
            seed=split_seed,
        )
        split_a_image = split_a_tensor.numpy().astype(np.float32)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        split_b_tensor, _, _ = _train_dip_once(
            cfg,
            observations,
            forward_operator,
            half_b,
            val_indices,
            device,
            max_iter=split_max_iter,
            seed=split_seed,
        )
        split_b_image = split_b_tensor.numpy().astype(np.float32)
        split_score = split_half_nrmse(split_a_image, split_b_image)
        np.save(out_dir / "split_half_a.npy", split_a_image)
        np.save(out_dir / "split_half_b.npy", split_b_image)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw_edge_agreement = raw_control_agreement(_gradient_magnitude(hr_image), _gradient_magnitude(raw_control_hr))
    train_residual = holdout_residual(
        hr_image,
        observations,
        forward_operator,
        indices=train_indices,
        noise_sigma=float(cfg["metrics"].get("noise_sigma", 0.0724)),
    )
    metrics = {
        "method": "deepinv_dip",
        "data_mode": cfg["runtime"]["data_mode"],
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
            noise_sigma=float(cfg["metrics"].get("noise_sigma", 0.0724)),
        ),
        "split_half_nrmse": split_score,
        "artifact_score": artifact_score(hr_image, pin_mask=None),
        "raw_control_agreement": raw_edge_agreement,
        "p95_gradient": p95_gradient(hr_image),
        "best_loss": float(train_result.best_loss),
        "train_residual": float(train_residual),
        "best_step": int(train_result.best_step),
        "final_step": int(train_result.final_step),
        "early_stopped_before_500": bool(train_result.final_step < 500),
        "elapsed_sec": float(train_result.elapsed_sec),
        "raw_control_agreement_domain": "gradient magnitude: HR highpass vs bicubic LR raw-control reference",
        "stage1_gate": "complete" if np.isfinite(hr_image).all() else "failed_nonfinite_image",
        "stage_gate": "complete" if np.isfinite(hr_image).all() else "failed_nonfinite_image",
        "deepinv_version": getattr(dinv, "__version__", "unknown"),
    }

    np.save(out_dir / "hr_image.npy", hr_image)
    np.save(out_dir / "hr_raw_control.npy", raw_control_hr)
    _write_history_outputs(out_dir, train_result.history)
    save_json(out_dir / "metrics.json", metrics)
    _write_metrics_csv(out_dir / "metrics.csv", metrics)
    save_json(out_dir / "config_used.json", {"config": cfg, "metadata": metadata})
    torch.save(
        {
            "model_name": "deepinv_dip",
            "model_state_dict": {key: value.detach().cpu().clone() for key, value in backbone.state_dict().items()},
            "latent": train_result.latent,
            "history": train_result.history,
            "best_loss": train_result.best_loss,
            "best_step": train_result.best_step,
            "config": _jsonable(cfg),
            "metadata": _jsonable(metadata),
            "metrics": _jsonable(metrics),
        },
        out_dir / "checkpoint.pt",
    )
    save_training_curve(train_result.history, out_dir / "training_curve.png", title="DeepInverse-DIP Training")
    save_image_figure(hr_image, out_dir / "hr_highpass.png", title="DeepInverse-DIP HR highpass")
    save_image_figure(raw_control_hr, out_dir / "hr_raw_control.png", title="DeepInverse-DIP raw-control bicubic reference")
    if split_a_image is not None and split_b_image is not None:
        save_image_figure(split_a_image - split_b_image, out_dir / "split_half_difference.png", title="DeepInverse-DIP split-half difference")
    print(f"saved DeepInverse-DIP outputs to {out_dir}")


def main() -> None:
    run_deepinv_dip_training(_parse_args())


if __name__ == "__main__":
    main()
