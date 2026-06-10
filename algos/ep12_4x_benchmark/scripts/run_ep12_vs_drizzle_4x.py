#!/usr/bin/env python3
"""Run EP12 4x UNet@2000 vs bare drizzle center-zoom benchmark."""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy import ndimage


ALGO_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
EP06_SRC = PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src"
for path in (ALGO_ROOT / "src", EP06_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.alignment import load_alignment_shifts  # noqa: E402
from common.data_loader import (  # noqa: E402
    bicubic_upsample,
    highpass_preprocess,
    load_main_session_frames,
    load_main_session_metadata,
)
from common.metrics import artifact_score, split_half_consistency  # noqa: E402
from sr4x.inference import bare_drizzle_temperature, infer_from_burst  # noqa: E402
from sr4x.model import ThermalSR4xUNet  # noqa: E402
from thermal_core.ep10_cache import center_fraction_crop  # noqa: E402
from thermal_core.plotting import COLORMAPS, savefig_academic, setup_academic_style  # noqa: E402


DEFAULT_CHECKPOINT = (
    PROJECT_ROOT / "algos" / "ep12_4x_sr" / "outputs" / "ep12_large_bucketv2" / "checkpoint_step_002000.pt"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "ep12_4x_benchmark"
DEFAULT_EXPECTED_SHAPE = (1920, 2560)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _pearson_finite(a: np.ndarray, b: np.ndarray) -> float:
    lhs = np.asarray(a, dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(lhs) & np.isfinite(rhs)
    if int(valid.sum()) < 2:
        return float("nan")
    return float(np.corrcoef(lhs[valid].ravel(), rhs[valid].ravel())[0, 1])


def _resolve_device(requested: str, *, allow_cuda0: bool = False) -> str:
    request = str(requested).strip().lower()
    if not request.startswith("cuda"):
        return request
    if not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        return "cpu"

    count = torch.cuda.device_count()
    if request == "cuda":
        if count >= 2:
            return "cuda:1"
        print("Only cuda:0 is visible; falling back to CPU to avoid the busy training GPU.")
        return "cpu"

    try:
        index = int(request.split(":", 1)[1])
    except (IndexError, ValueError) as exc:
        raise ValueError(f"Invalid CUDA device string: {requested}") from exc
    if index < 0 or index >= count:
        print(f"Requested {requested}, but only {count} CUDA device(s) are visible; falling back to CPU.")
        return "cpu"
    if index == 0 and not allow_cuda0:
        print("cuda:0 requested, but cuda:0 is protected for the existing training run; falling back to CPU.")
        return "cpu"
    return f"cuda:{index}"


def _load_checkpoint_config(checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint_path}")

    cfg = dict(ckpt.get("config") or {})
    config_path = checkpoint_path.parent / "config.json"
    if config_path.exists():
        disk_cfg = json.loads(config_path.read_text(encoding="utf-8"))
        merged = dict(disk_cfg)
        merged.update(cfg)
        cfg = merged
    defaults = {
        "scale": 4,
        "in_channels": 8,
        "base_channels": 48,
        "unet_depth": 4,
        "dilated_bottleneck": True,
        "predict_log_variance": True,
        "patch_size": 256,
        "drizzle_kernel": "bilinear",
    }
    for key, value in defaults.items():
        if key not in cfg:
            cfg[key] = value
    return cfg, ckpt["model_state_dict"]


def _build_model(cfg: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> ThermalSR4xUNet:
    model = ThermalSR4xUNet(
        in_channels=int(cfg.get("in_channels", 8)),
        out_channels=int(cfg.get("out_channels", 1)),
        base_channels=int(cfg.get("base_channels", 48)),
        scale=1,
        depth=int(cfg.get("unet_depth", 4)),
        dilated_bottleneck=bool(cfg.get("dilated_bottleneck", True)),
        predict_log_variance=bool(cfg.get("predict_log_variance", True)),
        min_log_variance=float(cfg.get("min_log_variance", -8.0)),
        max_log_variance=float(cfg.get("max_log_variance", 4.0)),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _run_ep12_temperature(
    model: ThermalSR4xUNet,
    raw_frames: np.ndarray,
    shifts: np.ndarray,
    *,
    device: str,
    scale: int,
    patch_size: int,
    overlap: int,
    sigma_bg: float,
    drizzle_kernel: str,
) -> np.ndarray:
    return infer_from_burst(
        model,
        raw_frames,
        shifts,
        scale=scale,
        patch_size=patch_size,
        overlap=overlap,
        device=device,
        sigma_bg=sigma_bg,
        drizzle_kernel=drizzle_kernel,
    ).astype(np.float32, copy=False)


def _shared_vmax(crops: list[np.ndarray]) -> float:
    finite = [crop[np.isfinite(crop)].ravel() for crop in crops if np.isfinite(crop).any()]
    if not finite:
        return 1.0
    vmax = float(np.percentile(np.abs(np.concatenate(finite)), 99.0))
    return max(vmax, 1e-6)


def _zoom_center(image: np.ndarray, *, center_fraction: float, zoom: float) -> np.ndarray:
    crop = center_fraction_crop(np.asarray(image), fraction=float(center_fraction))
    return ndimage.zoom(crop, zoom=float(zoom), order=1).astype(np.float32, copy=False)


def _step_labels(checkpoint_path: Path) -> tuple[str, str]:
    match = re.search(r"step_0*(\d+)", checkpoint_path.stem)
    if match:
        step_num = match.group(1)
        return f"step {step_num}", f"step{step_num}"
    return "step 2000", "step2000"


def save_highpass_comparison(
    ep12_hp: np.ndarray,
    drizzle_hp: np.ndarray,
    output_dir: Path,
    *,
    zoom: float,
    center_fraction: float,
    step_label: str,
) -> Path:
    setup_academic_style()
    panels = [
        (f"EP12 4x @ {step_label}", _zoom_center(ep12_hp, center_fraction=center_fraction, zoom=zoom)),
        ("Bare drizzle 4x", _zoom_center(drizzle_hp, center_fraction=center_fraction, zoom=zoom)),
    ]
    vmax = _shared_vmax([panel for _, panel in panels])
    fig, axes = plt.subplots(1, len(panels), figsize=(min(7.2, 2.55 * len(panels)), 2.9), squeeze=False)
    for ax, (title, image) in zip(axes.ravel(), panels, strict=True):
        im = ax.imshow(
            image,
            cmap=COLORMAPS["residual_diff"],
            vmin=-vmax,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.035, pad=0.03).set_label("Highpass response [deg C]")
    path = output_dir / "ep12_vs_drizzle_4x_center_zoom3x_highpass.png"
    savefig_academic(fig, path)
    return path


def save_temperature_comparison(
    ep12_temp: np.ndarray,
    drizzle_temp: np.ndarray,
    output_dir: Path,
    *,
    zoom: float,
    center_fraction: float,
    step_label: str,
) -> Path:
    setup_academic_style()
    panels = [
        (f"EP12 4x @ {step_label}", _zoom_center(ep12_temp, center_fraction=center_fraction, zoom=zoom)),
        ("Bare drizzle 4x", _zoom_center(drizzle_temp, center_fraction=center_fraction, zoom=zoom)),
    ]
    values = np.concatenate(
        [image[np.isfinite(image)].ravel() for _, image in panels if np.isfinite(image).any()]
    )
    if values.size:
        vmin, vmax = float(np.percentile(values, 1.0)), float(np.percentile(values, 99.0))
    else:
        vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(1, len(panels), figsize=(min(7.2, 2.55 * len(panels)), 2.9), squeeze=False)
    for ax, (title, image) in zip(axes.ravel(), panels, strict=True):
        im = ax.imshow(image, cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.035, pad=0.03).set_label("Temperature [deg C]")
    path = output_dir / "ep12_vs_drizzle_4x_center_zoom3x_temperature.png"
    savefig_academic(fig, path)
    return path


def _safe_split_half(
    raw_frames: np.ndarray,
    shifts: np.ndarray,
    model_cfg: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    *,
    device: str,
    scale: int,
    patch_size: int,
    overlap: int,
    sigma_bg: float,
    drizzle_kernel: str,
    n_splits: int,
    random_state: int,
) -> float:
    if n_splits <= 0:
        return float("nan")

    def method(frames_subset: np.ndarray, shifts_subset: np.ndarray, **_: Any) -> np.ndarray:
        local_model = _build_model(model_cfg, state_dict)
        temp = _run_ep12_temperature(
            local_model,
            frames_subset,
            shifts_subset,
            device=device,
            scale=scale,
            patch_size=patch_size,
            overlap=overlap,
            sigma_bg=sigma_bg,
            drizzle_kernel=drizzle_kernel,
        )
        return highpass_preprocess(temp, sigma_bg=sigma_bg)

    split_df = split_half_consistency(
        raw_frames,
        shifts,
        method,
        n_splits=int(n_splits),
        random_state=int(random_state),
    )
    return float(split_df["nrmse"].median())


def _write_notes(
    path: Path,
    *,
    ep12_name: str,
    ep12_metrics: dict[str, float],
    drizzle_metrics: dict[str, float],
    ep12_hp: np.ndarray,
    drizzle_hp: np.ndarray,
    center_fraction: float,
    zoom: float,
    n_frames: int,
    device: str,
    checkpoint_step: int,
) -> None:
    ep12_crop = center_fraction_crop(ep12_hp, center_fraction)
    drizzle_crop = center_fraction_crop(drizzle_hp, center_fraction)
    ep12_art = float(ep12_metrics.get("artifact_score", np.nan))
    drizzle_art = float(drizzle_metrics.get("artifact_score", np.nan))
    ep12_split = float(ep12_metrics.get("split_half_nrmse", np.nan))
    drizzle_split = float(drizzle_metrics.get("split_half_nrmse", np.nan))
    ep12_std = float(np.nanstd(ep12_crop))
    drizzle_std = float(np.nanstd(drizzle_crop))

    if np.isfinite(ep12_art) and np.isfinite(drizzle_art) and np.isfinite(ep12_split) and np.isfinite(drizzle_split):
        if ep12_art < drizzle_art and ep12_split <= 1.25 * drizzle_split:
            preference = (
                f"{ep12_name} shows a plausible contour gain over bare drizzle at step {checkpoint_step}, "
                "but only after visual inspection confirms the added edges are not synthetic-domain artifacts."
            )
        else:
            preference = (
                "Bare drizzle remains the safer near-term 4x baseline; EP12 continuation should be gated on "
                "whether the center crop shows real inner-contour gains without extra ringing."
            )
    else:
        preference = "Use the center-zoom figures as the primary evidence; proxy metrics alone are not decisive."

    lines = [
        f"# EP12 {ep12_name} vs Bare Drizzle 4x Notes",
        "",
        f"- Input: {n_frames} clean main-session frames; device used: `{device}`; center crop fraction={center_fraction:g}, display zoom={zoom:g}.",
        "- Highpass figure: same center ROI, same 3x display zoom, same residual_diff colormap, shared 99th-percentile symmetric limits.",
        "- Temperature figure: same ROI/zoom for sanity context on absolute thermal structure, not metrology.",
        "- Bare drizzle baseline: tcforge scatter-add drizzle mean channel at 4x, not STScI pixfrac drizzle.",
        f"- Center-crop highpass contrast std: EP12={ep12_std:.6g} deg C, drizzle={drizzle_std:.6g} deg C.",
        f"- Proxy metrics: EP12 split-half NRMSE={ep12_split:.6g}, artifact score={ep12_art:.6g}, raw-control corr={ep12_metrics.get('raw_control_corr', float('nan')):.6g}; "
        f"drizzle split-half NRMSE={drizzle_split:.6g}, artifact score={drizzle_art:.6g}, raw-control corr={drizzle_metrics.get('raw_control_corr', float('nan')):.6g}.",
        f"- Working recommendation: {preference}",
        f"- Boundary: checkpoint is synthetic-pretrained at step {checkpoint_step}; real-data gains carry domain-gap risk.",
        "- Boundary: 3x is display zoom only; reconstruction grid is true 4x (1920x2560).",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device, allow_cuda0=args.allow_cuda0)
    step_label, step_suffix = _step_labels(args.checkpoint)
    ep12_method_name = f"EP12@{step_suffix.replace('step', '')}"

    print(f"Using device: {device}")
    print("Loading 248 clean-frame raw input and contour-refined shifts")
    raw_frames, metadata = load_main_session_frames(workers=args.workers, dtype=np.float32, limit=args.limit)
    if args.limit is None:
        shifts = load_alignment_shifts(args.alignment_method, metadata=metadata).astype(np.float32, copy=False)
    else:
        full_metadata = load_main_session_metadata()
        full_shifts = load_alignment_shifts(args.alignment_method, metadata=full_metadata).astype(np.float32, copy=False)
        shifts = full_shifts[: len(metadata)]
    print(f"Loaded frames={raw_frames.shape}, shifts={shifts.shape}")

    cfg, state_dict = _load_checkpoint_config(args.checkpoint)
    scale = int(args.scale or cfg.get("scale", 4))
    patch_size = int(args.patch_size or cfg.get("patch_size", 256))
    drizzle_kernel = str(args.drizzle_kernel or cfg.get("drizzle_kernel", "bilinear"))
    model = _build_model(cfg, state_dict)

    print("Building bare drizzle 4x baseline")
    drizzle_temp = bare_drizzle_temperature(
        raw_frames,
        shifts,
        scale=scale,
        drizzle_kernel=drizzle_kernel,
    )
    drizzle_hp = highpass_preprocess(drizzle_temp, sigma_bg=float(args.highpass_sigma))

    print("Running EP12 4x full-frame inference")
    ep12_temp = _run_ep12_temperature(
        model,
        raw_frames,
        shifts,
        device=device,
        scale=scale,
        patch_size=patch_size,
        overlap=int(args.overlap),
        sigma_bg=float(args.highpass_sigma),
        drizzle_kernel=drizzle_kernel,
    )
    ep12_hp = highpass_preprocess(ep12_temp, sigma_bg=float(args.highpass_sigma))

    expected_shape = (raw_frames.shape[1] * scale, raw_frames.shape[2] * scale)
    for name, arr in (("EP12", ep12_hp), ("drizzle", drizzle_hp)):
        if arr.shape != expected_shape:
            raise ValueError(f"{name} highpass shape {arr.shape} != expected {expected_shape}")
    if args.limit is None and ep12_hp.shape != DEFAULT_EXPECTED_SHAPE:
        raise ValueError(f"Full EP12 4x shape {ep12_hp.shape} != expected {DEFAULT_EXPECTED_SHAPE}")
    if not np.isfinite(ep12_hp).any():
        raise ValueError("EP12 highpass contains no finite values")

    ep12_temp_path = output_dir / f"ep12_{step_suffix}_hr_temp.npy"
    ep12_hp_path = output_dir / f"ep12_{step_suffix}_hr_highpass.npy"
    drizzle_temp_path = output_dir / "drizzle_bare_4x_hr_temp.npy"
    drizzle_hp_path = output_dir / "drizzle_bare_4x_hr_highpass.npy"
    np.save(ep12_temp_path, ep12_temp.astype(np.float32, copy=False))
    np.save(ep12_hp_path, ep12_hp.astype(np.float32, copy=False))
    np.save(drizzle_temp_path, drizzle_temp.astype(np.float32, copy=False))
    np.save(drizzle_hp_path, drizzle_hp.astype(np.float32, copy=False))

    print(f"Writing center-zoom figures using {step_label}")
    highpass_fig = save_highpass_comparison(
        ep12_hp,
        drizzle_hp,
        output_dir,
        zoom=float(args.zoom),
        center_fraction=float(args.center_fraction),
        step_label=step_label,
    )
    temp_fig = save_temperature_comparison(
        ep12_temp,
        drizzle_temp,
        output_dir,
        zoom=float(args.zoom),
        center_fraction=float(args.center_fraction),
        step_label=step_label,
    )

    raw_control_temp = bicubic_upsample(np.nanmean(raw_frames, axis=0), scale=scale)
    raw_control_hp = highpass_preprocess(raw_control_temp, sigma_bg=float(args.highpass_sigma))
    np.save(output_dir / "raw_mean_control_4x_hr_temp.npy", raw_control_temp.astype(np.float32, copy=False))
    np.save(output_dir / "raw_mean_control_4x_hr_highpass.npy", raw_control_hp.astype(np.float32, copy=False))

    print("Computing proxy metrics")
    split_nrmse = _safe_split_half(
        raw_frames,
        shifts,
        cfg,
        state_dict,
        device=device,
        scale=scale,
        patch_size=patch_size,
        overlap=int(args.overlap),
        sigma_bg=float(args.highpass_sigma),
        drizzle_kernel=drizzle_kernel,
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
    )
    ep12_metrics = {
        "split_half_nrmse": split_nrmse,
        "artifact_score": float(artifact_score(ep12_hp, scale=scale)),
        "raw_control_corr": _pearson_finite(ep12_hp, raw_control_hp),
    }
    drizzle_metrics = {
        "split_half_nrmse": float("nan"),
        "artifact_score": float(artifact_score(drizzle_hp, scale=scale)),
        "raw_control_corr": _pearson_finite(drizzle_hp, raw_control_hp),
    }

    rows = [
        {
            "method": ep12_method_name,
            **ep12_metrics,
            "notes": f"Computed in EP12 4x benchmark on {len(raw_frames)} frame(s); split_half n_splits={int(args.n_splits)}.",
        },
        {
            "method": "Bare drizzle 4x",
            **drizzle_metrics,
            "notes": "tcforge scatter-add drizzle mean; split-half not recomputed here.",
        },
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "comparison_summary.csv", index=False)

    checkpoint_step = int(re.search(r"(\d+)", step_suffix).group(1)) if re.search(r"(\d+)", step_suffix) else 2000
    _write_notes(
        output_dir / "comparison_notes.md",
        ep12_name=ep12_method_name,
        ep12_metrics=ep12_metrics,
        drizzle_metrics=drizzle_metrics,
        ep12_hp=ep12_hp,
        drizzle_hp=drizzle_hp,
        center_fraction=float(args.center_fraction),
        zoom=float(args.zoom),
        n_frames=len(raw_frames),
        device=device,
        checkpoint_step=checkpoint_step,
    )

    _write_json(
        output_dir / "run_manifest.json",
        {
            "checkpoint": _relative(args.checkpoint),
            "output_dir": _relative(output_dir),
            "frames_shape": list(raw_frames.shape),
            "shifts_shape": list(shifts.shape),
            "ep12_temp_shape": list(ep12_temp.shape),
            "ep12_highpass_shape": list(ep12_hp.shape),
            "drizzle_shape": list(drizzle_temp.shape),
            "device_requested": args.device,
            "device_used": device,
            "scale": scale,
            "highpass_sigma": float(args.highpass_sigma),
            "patch_size": patch_size,
            "overlap": int(args.overlap),
            "drizzle_kernel": drizzle_kernel,
            "center_fraction": float(args.center_fraction),
            "zoom": float(args.zoom),
            "zoom_role": "display zoom only; reconstruction grid remains 4x",
            "n_splits": int(args.n_splits),
            "figures": [_relative(highpass_fig), _relative(temp_fig)],
        },
    )

    print(summary.to_string(index=False))
    print(f"Wrote {_relative(ep12_hp_path)}")
    print(f"Wrote {_relative(highpass_fig)}")
    print(f"Wrote {_relative(temp_fig)}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--alignment-method", default="contour_refined")
    parser.add_argument("--highpass-sigma", type=float, default=5.0)
    parser.add_argument("--patch-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--drizzle-kernel", default=None)
    parser.add_argument("--zoom", type=float, default=3.0)
    parser.add_argument("--center-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--allow-cuda0", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=1)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (0.0 < float(args.center_fraction) <= 1.0):
        raise ValueError("--center-fraction must be in (0, 1]")
    if not math.isfinite(float(args.zoom)) or float(args.zoom) <= 0:
        raise ValueError("--zoom must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
