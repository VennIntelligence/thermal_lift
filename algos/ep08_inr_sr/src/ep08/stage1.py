from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch import nn

from ep08.data import RealDataBundle, load_real_dataset
from ep08.forward import ForwardOperator
from ep08.metrics import artifact_score, holdout_residual, p95_gradient, raw_control_agreement, split_half_nrmse
from ep08.models import DeepDecoder, Siren, Wire
from ep08.splits import build_train_val_split
from ep08.trainer import INRTrainer, TrainConfig, TrainResult
from ep08.utils import save_json, set_seed
from thermal_core.plotting import savefig_academic, setup_academic_style

EP08_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = EP08_ROOT.parent.parent
PROJECT_OUTPUT = PROJECT_ROOT / "output" / "ep08_inr_sr"


class SyntheticForwardOperator(nn.Module):
    def forward(self, x_hr: torch.Tensor, index: int) -> torch.Tensor:
        del index
        return F.avg_pool2d(x_hr[None, None], kernel_size=2, stride=2)[0, 0]


def default_config_path(model_name: str) -> Path:
    return EP08_ROOT / "configs" / f"{model_name}.yaml"


def default_output_dir(model_name: str) -> Path:
    suffix = "deep_decoder_stage1" if model_name == "deep_decoder" else f"{model_name}_stage1"
    return PROJECT_OUTPUT / suffix


def _base_defaults(model_name: str) -> dict[str, Any]:
    model_defaults: dict[str, Any]
    if model_name == "deep_decoder":
        model_defaults = {
            "name": "deep_decoder",
            "latent_channels": 16,
            "hidden_channels": [32, 32, 16],
            "latent_spatial": [8, 8],
            "seed": 0,
        }
    else:
        model_defaults = {
            "name": model_name,
            "hidden_features": 64,
            "hidden_layers": 2,
            "first_omega_0": 30.0 if model_name == "siren" else 20.0,
            "hidden_omega_0": 30.0 if model_name == "siren" else 20.0,
        }
        if model_name == "wire":
            model_defaults.update({"first_sigma_0": 10.0, "hidden_sigma_0": 10.0})

    return {
        "model": model_defaults,
        "data": {
            "scale": 2,
            "default_n_frames": 32,
            "default_patch_size_lr_px": 256,
            "val_ratio": 0.2,
        },
        "forward": {"psf_sigma_lr_px": 1.0},
        "preprocess": {
            "highpass_sigma_bg_lr_px": 5.0,
            "highpass_mode": "nearest",
        },
        "train": {
            "lr": 1.0e-4,
            "max_iter": 200,
            "batch_k": 8,
            "warmup_steps": 20,
            "min_lr_factor": 0.05,
            "grad_clip_norm": 1.0,
            "val_interval": 25,
            "early_stop_patience": 50,
            "early_stop_min_delta": 1.0e-6,
            "early_stop_min_steps": 0,
            "split_half_max_iter": 3000,
            "split_half_enabled": True,
        },
        "runtime": {
            "data_mode": "synthetic",
            "device": "cpu",
            "workers": 1,
            "alignment_method": "contour_refined",
            "seed": 42,
            "output_dir": str(default_output_dir(model_name)),
            "data_dir": None,
            "frame_audit_path": None,
        },
        "metrics": {"noise_sigma": 0.0724},
    }


def _deep_update(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def _load_config(path: Path | None, model_name: str) -> dict[str, Any]:
    cfg = _base_defaults(model_name)
    if path is not None and path.exists():
        with path.open("r", encoding="utf-8") as f:
            payload = yaml.safe_load(f) or {}
        if not isinstance(payload, dict):
            raise ValueError(f"config must contain a mapping: {path}")
        _deep_update(cfg, payload)
    train_alias = cfg.get("training")
    if isinstance(train_alias, dict):
        _deep_update(cfg["train"], train_alias)
    return cfg


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


def _resolve_output_dir(path_value: str | Path | None, model_name: str) -> Path:
    if path_value is None:
        return default_output_dir(model_name)
    path = Path(path_value)
    if path.is_absolute():
        return path
    if path.parts and path.parts[0] == "output":
        return PROJECT_ROOT / path
    return Path.cwd() / path


def _add_common_args(parser: argparse.ArgumentParser, model_name: str) -> None:
    parser.add_argument("--config", type=Path, default=default_config_path(model_name))
    parser.add_argument("--data-mode", choices=["synthetic", "real"], default=None)
    parser.add_argument("--device", default=None, help="cuda:0, cuda:1, or cpu")
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--n-frames", type=int, default=None)
    parser.add_argument("--patch-size", type=int, default=None, help="LR patch size in pixels")
    parser.add_argument("--batch-k", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--warmup-steps", type=int, default=None)
    parser.add_argument("--min-lr-factor", type=float, default=None)
    parser.add_argument("--grad-clip", type=float, default=None, dest="grad_clip_norm")
    parser.add_argument("--val-interval", type=int, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None, dest="early_stop_patience")
    parser.add_argument("--early-stopping-min-delta", type=float, default=None, dest="early_stop_min_delta")
    parser.add_argument("--early-stopping-min-steps", type=int, default=None, dest="early_stop_min_steps")
    parser.add_argument("--split-half-max-iter", type=int, default=None)
    parser.add_argument("--skip-split-half", action="store_true")
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--alignment-method", default=None)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--frame-audit-path", type=Path, default=None)
    parser.add_argument("--data-dir", type=Path, default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--hidden-features", type=int, default=None)
    parser.add_argument("--hidden-layers", type=int, default=None)
    parser.add_argument("--omega-0", type=float, default=None)
    parser.add_argument("--first-omega-0", type=float, default=None)
    parser.add_argument("--hidden-omega-0", type=float, default=None)
    parser.add_argument("--sigma-0", type=float, default=None)
    parser.add_argument("--first-sigma-0", type=float, default=None)
    parser.add_argument("--hidden-sigma-0", type=float, default=None)
    parser.add_argument("--latent-channels", type=int, default=None)
    parser.add_argument("--hidden-channels", default=None, help="Comma-separated Deep Decoder channels, e.g. 64,64,32")
    parser.add_argument("--latent-spatial", default=None, help="HxW Deep Decoder latent size, e.g. 8,8")


def parse_training_args(model_name: str, description: str) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    _add_common_args(parser, model_name)
    return parser.parse_args()


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
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
        ("warmup_steps", "train", "warmup_steps"),
        ("min_lr_factor", "train", "min_lr_factor"),
        ("grad_clip_norm", "train", "grad_clip_norm"),
        ("val_interval", "train", "val_interval"),
        ("early_stop_patience", "train", "early_stop_patience"),
        ("early_stop_min_delta", "train", "early_stop_min_delta"),
        ("early_stop_min_steps", "train", "early_stop_min_steps"),
        ("split_half_max_iter", "train", "split_half_max_iter"),
        ("val_ratio", "data", "val_ratio"),
        ("hidden_features", "model", "hidden_features"),
        ("hidden_layers", "model", "hidden_layers"),
        ("first_omega_0", "model", "first_omega_0"),
        ("hidden_omega_0", "model", "hidden_omega_0"),
        ("first_sigma_0", "model", "first_sigma_0"),
        ("hidden_sigma_0", "model", "hidden_sigma_0"),
        ("latent_channels", "model", "latent_channels"),
    ):
        value = getattr(args, cli_key, None)
        if value is not None:
            cfg.setdefault(section, {})[cfg_key] = value

    if args.n_frames is not None:
        cfg.setdefault("data", {})["default_n_frames"] = int(args.n_frames)
    if args.patch_size is not None:
        cfg.setdefault("data", {})["default_patch_size_lr_px"] = int(args.patch_size)
    if args.frame_audit_path is not None:
        cfg.setdefault("runtime", {})["frame_audit_path"] = args.frame_audit_path
    if args.data_dir is not None:
        cfg.setdefault("runtime", {})["data_dir"] = args.data_dir
    if args.omega_0 is not None:
        cfg.setdefault("model", {})["first_omega_0"] = float(args.omega_0)
        cfg.setdefault("model", {})["hidden_omega_0"] = float(args.omega_0)
    if args.sigma_0 is not None:
        cfg.setdefault("model", {})["first_sigma_0"] = float(args.sigma_0)
        cfg.setdefault("model", {})["hidden_sigma_0"] = float(args.sigma_0)
    if args.hidden_channels:
        cfg.setdefault("model", {})["hidden_channels"] = [int(v) for v in str(args.hidden_channels).split(",") if v.strip()]
    if args.latent_spatial:
        values = [int(v) for v in str(args.latent_spatial).split(",") if v.strip()]
        if len(values) != 2:
            raise ValueError("--latent-spatial must contain two comma-separated integers")
        cfg.setdefault("model", {})["latent_spatial"] = values
    if args.skip_split_half:
        cfg.setdefault("train", {})["split_half_enabled"] = False
    return cfg


def build_synthetic_observations(
    model_name: str,
    n_frames: int,
    patch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hr_size = patch_size * 2
    coords = torch.linspace(-1.0, 1.0, hr_size, device=device)
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    if model_name == "wire":
        hr = torch.sin(10.0 * xx + 2.0 * yy) * torch.exp(-10.0 * (yy + 0.1).square())
        hr = hr + 0.4 * ((xx > 0.0) & (yy < 0.35)).float()
    elif model_name == "deep_decoder":
        hr = 0.7 * torch.exp(-20.0 * ((xx + 0.25).square() + (yy - 0.1).square()))
        hr = hr - 0.4 * torch.exp(-18.0 * ((xx - 0.28).square() + (yy + 0.2).square()))
    else:
        hr = torch.sin(8.0 * xx) * torch.exp(-12.0 * (yy - 0.15).square())
        hr = hr + 0.5 * ((xx > -0.2) & (yy > -0.1)).float()
    lr = F.avg_pool2d(hr[None, None], kernel_size=2, stride=2)[0, 0]
    frames = lr.unsqueeze(0).repeat(n_frames, 1, 1)
    frames = frames + 0.002 * torch.randn_like(frames)
    shifts = torch.zeros(n_frames, 2, device=device)
    return frames, shifts, frames.clone()


def build_model(model_name: str, cfg: dict[str, Any]) -> nn.Module:
    model_cfg = cfg["model"]
    if model_name == "siren":
        return Siren(
            hidden_features=int(model_cfg["hidden_features"]),
            hidden_layers=int(model_cfg["hidden_layers"]),
            first_omega_0=float(model_cfg["first_omega_0"]),
            hidden_omega_0=float(model_cfg["hidden_omega_0"]),
        )
    if model_name == "wire":
        return Wire(
            hidden_features=int(model_cfg["hidden_features"]),
            hidden_layers=int(model_cfg["hidden_layers"]),
            first_omega_0=float(model_cfg["first_omega_0"]),
            hidden_omega_0=float(model_cfg["hidden_omega_0"]),
            first_sigma_0=float(model_cfg["first_sigma_0"]),
            hidden_sigma_0=float(model_cfg["hidden_sigma_0"]),
        )
    if model_name == "deep_decoder":
        return DeepDecoder(
            latent_channels=int(model_cfg["latent_channels"]),
            hidden_channels=tuple(int(v) for v in model_cfg["hidden_channels"]),
            latent_spatial=tuple(int(v) for v in model_cfg["latent_spatial"]),
            seed=int(model_cfg.get("seed", 0)),
        )
    raise ValueError(f"unknown model_name: {model_name}")


def make_train_config(cfg: dict[str, Any], *, max_iter: int | None = None, seed: int | None = None) -> TrainConfig:
    train_cfg = cfg["train"]
    resolved_max_iter = int(train_cfg["max_iter"] if max_iter is None else max_iter)
    return TrainConfig(
        max_iter=resolved_max_iter,
        lr=float(train_cfg["lr"]),
        warmup_steps=min(int(train_cfg["warmup_steps"]), max(resolved_max_iter, 1)),
        min_lr_factor=float(train_cfg["min_lr_factor"]),
        batch_k=int(train_cfg["batch_k"]),
        grad_clip_norm=None if train_cfg.get("grad_clip_norm") is None else float(train_cfg["grad_clip_norm"]),
        val_interval=int(train_cfg.get("val_interval", 25)),
        early_stop_patience=int(train_cfg["early_stop_patience"]),
        early_stop_min_delta=float(train_cfg.get("early_stop_min_delta", 1.0e-6)),
        early_stop_min_steps=min(int(train_cfg.get("early_stop_min_steps", 0)), resolved_max_iter),
        seed=int(cfg["runtime"]["seed"] if seed is None else seed),
    )


def load_dataset_for_config(
    model_name: str,
    cfg: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, ForwardOperator | SyntheticForwardOperator, dict[str, Any]]:
    data_cfg = cfg["data"]
    runtime = cfg["runtime"]
    preprocess = cfg["preprocess"]
    n_frames = int(data_cfg["default_n_frames"])
    patch_size = int(data_cfg["default_patch_size_lr_px"])
    data_mode = str(runtime["data_mode"])

    if data_mode == "synthetic":
        observations, shifts, raw_control = build_synthetic_observations(model_name, n_frames, patch_size, device)
        forward_operator: ForwardOperator | SyntheticForwardOperator = SyntheticForwardOperator()
        metadata = {
            "data_mode": "synthetic",
            "n_frames": int(n_frames),
            "lr_shape": tuple(int(v) for v in observations.shape[-2:]),
            "raw_control_source": "synthetic LR observations",
        }
    elif data_mode == "real":
        bundle: RealDataBundle = load_real_dataset(
            n_frames=n_frames,
            patch_size=patch_size,
            workers=int(runtime.get("workers", 1)),
            alignment_method=str(runtime.get("alignment_method", "contour_refined")),
            data_dir=runtime.get("data_dir"),
            frame_audit_path=runtime.get("frame_audit_path"),
            highpass_sigma=float(preprocess["highpass_sigma_bg_lr_px"]),
            highpass_mode=str(preprocess.get("highpass_mode", "nearest")),
            track="highpass",
            device=device,
        )
        observations, shifts, raw_control = bundle.observations, bundle.shifts, bundle.raw_control
        forward_operator = ForwardOperator(
            hr_shape=(int(observations.shape[-2]) * 2, int(observations.shape[-1]) * 2),
            lr_shape=tuple(int(v) for v in observations.shape[-2:]),
            shifts=shifts,
            psf_sigma=float(cfg["forward"]["psf_sigma_lr_px"]),
            scale=int(data_cfg["scale"]),
        )
        metadata = bundle.metadata
    else:
        raise ValueError("data_mode must be synthetic or real")

    return observations, shifts, raw_control, forward_operator, metadata


def _state_dict_cpu(model: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def _split_train_indices(train_indices: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    if train_indices.size < 2:
        return train_indices.copy(), train_indices.copy()
    rng = np.random.default_rng(seed + 202)
    shuffled = train_indices.copy()
    rng.shuffle(shuffled)
    mid = max(1, len(shuffled) // 2)
    return np.sort(shuffled[:mid]), np.sort(shuffled[mid:])


def _train_once(
    model_name: str,
    cfg: dict[str, Any],
    observations: torch.Tensor,
    shifts: torch.Tensor,
    forward_operator: nn.Module,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    device: torch.device,
    *,
    max_iter: int | None = None,
    seed: int | None = None,
) -> tuple[nn.Module, TrainResult]:
    set_seed(int(cfg["runtime"]["seed"] if seed is None else seed))
    model = build_model(model_name, cfg)
    lr_shape = tuple(int(v) for v in observations.shape[-2:])
    hr_shape = (lr_shape[0] * int(cfg["data"]["scale"]), lr_shape[1] * int(cfg["data"]["scale"]))
    trainer = INRTrainer(
        model,
        observations,
        shifts,
        hr_shape=hr_shape,
        forward_operator=forward_operator,
        psf_sigma_lr_px=float(cfg["forward"]["psf_sigma_lr_px"]),
        highpass_sigma_bg_lr_px=float(cfg["preprocess"]["highpass_sigma_bg_lr_px"]),
        scale=int(cfg["data"]["scale"]),
        device=device,
        config=make_train_config(cfg, max_iter=max_iter, seed=seed),
        train_indices=train_indices,
        val_indices=val_indices,
    )
    result = trainer.fit()
    return model, result


def _gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gy, gx = np.gradient(np.asarray(image, dtype=np.float64))
    return np.sqrt(gx * gx + gy * gy)


def _upscale_raw_control(raw_control: torch.Tensor, train_indices: np.ndarray, hr_shape: tuple[int, int]) -> np.ndarray:
    indices = torch.as_tensor(train_indices, dtype=torch.long, device=raw_control.device)
    reference = raw_control.index_select(0, indices).mean(dim=0)
    upscaled = F.interpolate(reference[None, None], size=hr_shape, mode="bilinear", align_corners=False)[0, 0]
    return upscaled.detach().cpu().numpy()


def _symmetric_limits(image: np.ndarray, percentile: float = 99.0) -> tuple[float, float]:
    finite = np.asarray(image)[np.isfinite(image)]
    if finite.size == 0:
        return -1.0, 1.0
    limit = float(np.percentile(np.abs(finite), percentile))
    if limit <= 0 or not math.isfinite(limit):
        limit = float(np.max(np.abs(finite))) if finite.size else 1.0
    if limit <= 0 or not math.isfinite(limit):
        limit = 1.0
    return -limit, limit


def save_image_figure(image: np.ndarray, path: Path, *, title: str, cmap: str = "RdBu_r") -> None:
    setup_academic_style()
    fig, ax = plt.subplots(figsize=(5.0, 4.2))
    vmin, vmax = _symmetric_limits(image)
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title)
    ax.set_xlabel("HR x pixel")
    ax.set_ylabel("HR y pixel")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    savefig_academic(fig, path)


def save_training_curve(history: list[dict[str, float]], path: Path, *, title: str) -> None:
    setup_academic_style()
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    steps = np.array([row["step"] for row in history], dtype=float)
    train = np.array([row["train_loss"] for row in history], dtype=float)
    ax.plot(steps, train, label="batch train loss", color="#4C72B0")
    holdout_steps = np.array([row["step"] for row in history if "holdout_loss" in row], dtype=float)
    if holdout_steps.size:
        holdout = np.array([row["holdout_loss"] for row in history if "holdout_loss" in row], dtype=float)
        ax.plot(holdout_steps, holdout, label="hold-out loss", color="#C44E52")
    train_set_steps = np.array([row["step"] for row in history if "train_set_loss" in row], dtype=float)
    if train_set_steps.size:
        train_set = np.array([row["train_set_loss"] for row in history if "train_set_loss" in row], dtype=float)
        ax.plot(train_set_steps, train_set, label="train-set loss", color="#55A868", alpha=0.8)
    ax.set_yscale("log")
    ax.set_title(title)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("MSE loss")
    ax.legend()
    savefig_academic(fig, path)


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


def update_stage1_comparison() -> None:
    rows: list[dict[str, Any]] = []
    for model_name, dirname in (("siren", "siren_stage1"), ("wire", "wire_stage1")):
        metrics_path = PROJECT_OUTPUT / dirname / "metrics.json"
        if not metrics_path.exists():
            continue
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "method": model_name.upper() if model_name == "wire" else "SIREN",
                "family": "inr_gabor" if model_name == "wire" else "inr_sine",
                "holdout_residual": payload.get("holdout_residual"),
                "split_half_nrmse": payload.get("split_half_nrmse"),
                "artifact_score": payload.get("artifact_score"),
                "raw_control_agreement": payload.get("raw_control_agreement"),
                "p95_gradient": payload.get("p95_gradient"),
                "best_step": payload.get("best_step"),
                "stage1_gate": payload.get("stage1_gate"),
                "source": str(metrics_path.relative_to(PROJECT_ROOT)),
            }
        )
    if not rows:
        return
    out = PROJECT_OUTPUT / "stage1_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    save_stage1_comparison_figure(rows, PROJECT_OUTPUT / "stage1_comparison.png")


def save_stage1_comparison_figure(rows: list[dict[str, Any]], path: Path) -> None:
    setup_academic_style()
    metrics = [
        ("holdout_residual", "Hold-out residual"),
        ("split_half_nrmse", "Split-half NRMSE"),
        ("artifact_score", "Artifact score"),
        ("raw_control_agreement", "Raw-control agreement"),
        ("p95_gradient", "P95 gradient"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4))
    methods = [row["method"] for row in rows]
    colors = ["#4C72B0", "#C44E52"]
    for ax, (key, label) in zip(axes.ravel(), metrics):
        values = [float(row[key]) for row in rows]
        ax.bar(methods, values, color=colors[: len(rows)])
        ax.set_title(label)
        ax.tick_params(axis="x", rotation=20)
        if key in {"holdout_residual", "artifact_score", "p95_gradient"}:
            ax.set_yscale("log")
    axes.ravel()[-1].axis("off")
    savefig_academic(fig, path)


def run_stage1_training(model_name: str, args: argparse.Namespace) -> None:
    cfg = apply_cli_overrides(_load_config(args.config, model_name), args)
    out_dir = _resolve_output_dir(cfg["runtime"].get("output_dir"), model_name)
    cfg["runtime"]["output_dir"] = str(out_dir)
    cfg["runtime"]["config_path"] = str(args.config)
    cfg["runtime"]["project_root"] = str(PROJECT_ROOT)
    out_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(str(cfg["runtime"]["device"]))
    set_seed(int(cfg["runtime"]["seed"]))
    observations, shifts, raw_control, forward_operator, metadata = load_dataset_for_config(model_name, cfg, device)
    forward_operator = forward_operator.to(device)
    shifts_np = shifts.detach().cpu().numpy()
    frame_ids = np.arange(int(observations.shape[0]))
    train_indices, val_indices, val_mask = build_train_val_split(
        frame_ids,
        shifts_np,
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

    model, result = _train_once(
        model_name,
        cfg,
        observations,
        shifts,
        forward_operator,
        train_indices,
        val_indices,
        device,
    )
    hr_image = result.image.detach().cpu().numpy().astype(np.float32)
    hr_shape = tuple(int(v) for v in hr_image.shape)
    raw_control_hr = _upscale_raw_control(raw_control, train_indices, hr_shape).astype(np.float32)

    split_score = float("nan")
    split_a_image: np.ndarray | None = None
    split_b_image: np.ndarray | None = None
    if bool(cfg["train"].get("split_half_enabled", True)):
        half_a, half_b = _split_train_indices(train_indices, int(cfg["runtime"]["seed"]))
        split_max_iter = int(cfg["train"].get("split_half_max_iter", 3000))
        _, split_a = _train_once(
            model_name,
            cfg,
            observations,
            shifts,
            forward_operator,
            half_a,
            val_indices,
            device,
            max_iter=split_max_iter,
            seed=int(cfg["runtime"]["seed"]) + 101,
        )
        split_a_image = split_a.image.detach().cpu().numpy().astype(np.float32)
        if device.type == "cuda":
            torch.cuda.empty_cache()
        _, split_b = _train_once(
            model_name,
            cfg,
            observations,
            shifts,
            forward_operator,
            half_b,
            val_indices,
            device,
            max_iter=split_max_iter,
            seed=int(cfg["runtime"]["seed"]) + 202,
        )
        split_b_image = split_b.image.detach().cpu().numpy().astype(np.float32)
        split_score = split_half_nrmse(split_a_image, split_b_image)
        np.save(out_dir / "split_half_a.npy", split_a_image)
        np.save(out_dir / "split_half_b.npy", split_b_image)
        if device.type == "cuda":
            torch.cuda.empty_cache()

    raw_edge_agreement = raw_control_agreement(_gradient_magnitude(hr_image), _gradient_magnitude(raw_control_hr))
    metrics = {
        "method": model_name,
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
        "best_loss": float(result.best_loss),
        "best_step": int(result.best_step),
        "final_step": int(result.history[-1]["step"]) if result.history else 0,
        "early_stopped_before_500": bool(result.history and result.history[-1]["step"] < 500),
        "raw_control_agreement_domain": "gradient magnitude: HR highpass vs bicubic LR raw-control reference",
        "stage1_gate": "complete" if np.isfinite(hr_image).all() else "failed_nonfinite_image",
    }

    np.save(out_dir / "hr_image.npy", hr_image)
    np.save(out_dir / "hr_raw_control.npy", raw_control_hr)
    _write_history_outputs(out_dir, result.history)
    save_json(out_dir / "metrics.json", metrics)
    _write_metrics_csv(out_dir / "metrics.csv", metrics)
    save_json(out_dir / "config_used.json", {"config": cfg, "metadata": metadata})
    torch.save(
        {
            "model_name": model_name,
            "model_state_dict": _state_dict_cpu(model),
            "history": result.history,
            "best_loss": result.best_loss,
            "best_step": result.best_step,
            "config": _jsonable(cfg),
            "metadata": _jsonable(metadata),
        },
        out_dir / "checkpoint.pt",
    )
    save_training_curve(result.history, out_dir / "training_curve.png", title=f"{model_name.upper()} Stage 1 Training")
    save_image_figure(hr_image, out_dir / "hr_highpass.png", title=f"{model_name.upper()} HR highpass")
    save_image_figure(raw_control_hr, out_dir / "hr_raw_control.png", title=f"{model_name.upper()} raw-control bicubic reference")
    if split_a_image is not None and split_b_image is not None:
        save_image_figure(split_a_image - split_b_image, out_dir / "split_half_difference.png", title=f"{model_name.upper()} split-half difference")

    if model_name in {"siren", "wire"}:
        update_stage1_comparison()
    print(f"saved Stage 1 outputs to {out_dir}")


__all__ = ["parse_training_args", "run_stage1_training"]
