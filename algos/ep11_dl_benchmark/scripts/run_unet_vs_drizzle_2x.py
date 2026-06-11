#!/usr/bin/env python3
"""Run EP11 UNet@4000 vs TGV 2x center-zoom benchmark."""

from __future__ import annotations

import argparse
import json
import math
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
from thermal_core.ep10_cache import center_fraction_crop  # noqa: E402
from thermal_core.plotting import COLORMAPS, savefig_academic, setup_academic_style  # noqa: E402
from unet_sr import ThermalSRUNet  # noqa: E402
from unet_sr.inference import infer_from_burst  # noqa: E402


DEFAULT_CHECKPOINT = PROJECT_ROOT / "algos" / "ep07_unet_sr" / "outputs" / "ep07_large_bucket" / "checkpoint_step_006000.pt"
DEFAULT_BASELINE_NAME = "TGV best 2x"
DEFAULT_BASELINE_HR = PROJECT_ROOT / "output" / "ep10_tgv_sr" / "best_hr_highpass.npy"
DEFAULT_BASELINE_SWEEP = PROJECT_ROOT / "output" / "ep10_tgv_sr" / "sweep_results.csv"
DEFAULT_BASELINE_SUMMARY = PROJECT_ROOT / "output" / "ep10_tgv_sr" / "run_summary.json"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "ep11_dl_benchmark"
DEFAULT_EXPECTED_SHAPE = (960, 1280)
EP10_SRC = PROJECT_ROOT / "algos" / "ep10_tgv_sr" / "src"
DEFAULT_TGV_LAMBDA = 0.003
DEFAULT_TGV_PSF_SIGMA = 0.50
DEFAULT_TGV_ANISO_RATIO_Y = 1.5


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
    """Resolve CUDA requests while keeping cuda:0 free by default."""

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


def _load_checkpoint_config(checkpoint_path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor], int]:
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
    for key, value in {"scale": 2, "residual": True, "in_channels": 6, "base_channels": 48}.items():
        if key not in cfg:
            cfg[key] = value
    step = int(ckpt.get("step", 0))
    return cfg, ckpt["model_state_dict"], step


def _checkpoint_labels(checkpoint_path: Path, step: int) -> tuple[str, str, str]:
    import re

    match = re.search(r"step_0*(\d+)", checkpoint_path.stem)
    if match:
        step_num = match.group(1)
    elif step > 0:
        step_num = str(step)
    elif checkpoint_path.stem == "model_final":
        step_num = "final"
    else:
        step_num = "0"
    step_label = f"step {step_num}" if step_num != "final" else "final"
    step_suffix = f"step{step_num}" if step_num != "final" else "final"
    return step_num, step_label, step_suffix


def _build_model(cfg: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> ThermalSRUNet:
    residual = bool(cfg.get("residual", True))
    model_scale = 1 if residual else int(cfg.get("scale", 2))
    model = ThermalSRUNet(
        in_channels=int(cfg.get("in_channels", 6)),
        out_channels=int(cfg.get("out_channels", 1)),
        scale=model_scale,
        base_channels=int(cfg.get("base_channels", 48)),
        hr_upsampler=str(cfg.get("hr_upsampler", "bilinear")),
        hr_res_blocks=int(cfg.get("hr_res_blocks", 0)),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _run_unet_temperature(
    model: ThermalSRUNet,
    raw_frames: np.ndarray,
    shifts: np.ndarray,
    *,
    device: str,
    scale: int,
    patch_size_hr: int,
    overlap: int,
    sigma_bg: float,
    residual: bool,
) -> np.ndarray:
    return infer_from_burst(
        model,
        raw_frames,
        shifts,
        scale=scale,
        patch_size_hr=patch_size_hr,
        overlap=overlap,
        device=device,
        residual=residual,
        sigma_bg=sigma_bg,
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


def _zoom_tag(zoom: float) -> str:
    zoom_value = float(zoom)
    if math.isclose(zoom_value, round(zoom_value), rel_tol=0.0, abs_tol=1e-6):
        return f"zoom{int(round(zoom_value))}x"
    return f"zoom{zoom_value:g}x"


def _temperature_limits_shared(crops: list[np.ndarray]) -> tuple[float, float]:
    finite = [crop[np.isfinite(crop)].ravel() for crop in crops if np.isfinite(crop).any()]
    if not finite:
        return 0.0, 1.0
    values = np.concatenate(finite)
    return float(np.percentile(values, 1.0)), float(np.percentile(values, 99.0))


def tgv_highpass_to_temperature(
    tgv_hp: np.ndarray,
    raw_frames: np.ndarray,
    *,
    scale: int,
    sigma_bg: float,
) -> np.ndarray:
    """Restore a Celsius HR temperature map from TGV highpass output."""

    ref_idx = len(raw_frames) // 2
    ref_temp_hr = bicubic_upsample(raw_frames[ref_idx], scale=scale).astype(np.float32, copy=False)
    ref_hp_hr = highpass_preprocess(ref_temp_hr, sigma_bg=float(sigma_bg) * 2.0)
    lowfreq = ref_temp_hr - ref_hp_hr
    return (lowfreq + np.asarray(tgv_hp, dtype=np.float32)).astype(np.float32, copy=False)


def reconstruct_tgv_highpass(
    raw_frames: np.ndarray,
    shifts: np.ndarray,
    *,
    sigma_bg: float,
    lambda_tv: float,
    psf_sigma: float,
    aniso_ratio_y: float,
    coverage_weighted: bool,
    workers: int,
) -> np.ndarray:
    if str(EP10_SRC) not in sys.path:
        sys.path.insert(0, str(EP10_SRC))
    from ep10_tgv_sr import reconstruct_map_tgv  # noqa: WPS433

    hp_frames = highpass_preprocess(raw_frames, sigma_bg=float(sigma_bg), workers=workers)
    tgv_hp, _records = reconstruct_map_tgv(
        hp_frames,
        shifts,
        lambda_tv=float(lambda_tv),
        alpha_ratio=2.0,
        psf_sigma=float(psf_sigma),
        max_iter=100,
        step_size=1.0,
        use_fista=True,
        workers=workers,
        tgv_inner_iter=80,
        tgv_device="cpu",
        aniso_ratio_y=float(aniso_ratio_y),
        coverage_weighted=bool(coverage_weighted),
    )
    return np.asarray(tgv_hp, dtype=np.float32)


def save_highpass_comparison(
    unet_hp: np.ndarray,
    baseline_hp: np.ndarray,
    output_dir: Path,
    *,
    baseline_name: str,
    zoom: float,
    center_fraction: float,
    step_label: str = "step 4000",
) -> Path:
    setup_academic_style()
    panels = [
        (f"UNet 2x @ EP07 {step_label}", _zoom_center(unet_hp, center_fraction=center_fraction, zoom=zoom)),
        (baseline_name, _zoom_center(baseline_hp, center_fraction=center_fraction, zoom=zoom)),
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
    path = output_dir / f"unet_vs_tgv_2x_center_{_zoom_tag(zoom)}_highpass.png"
    savefig_academic(fig, path)
    return path


def save_temperature_comparison(
    unet_temp: np.ndarray,
    baseline_temp: np.ndarray,
    output_dir: Path,
    *,
    baseline_name: str,
    zoom: float,
    center_fraction: float,
    step_label: str = "step 4000",
) -> Path:
    setup_academic_style()
    panels = [
        (f"UNet 2x @ EP07 {step_label}", _zoom_center(unet_temp, center_fraction=center_fraction, zoom=zoom)),
        (baseline_name, _zoom_center(baseline_temp, center_fraction=center_fraction, zoom=zoom)),
    ]
    vmin, vmax = _temperature_limits_shared([panel for _, panel in panels])

    fig, axes = plt.subplots(1, len(panels), figsize=(min(7.4, 2.6 * len(panels)), 3.0), squeeze=False)
    for ax, (title, image) in zip(axes.ravel(), panels, strict=True):
        im = ax.imshow(
            image,
            cmap=COLORMAPS["temperature"],
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.035, pad=0.03).set_label("Temperature [deg C]")
    path = output_dir / f"unet_vs_tgv_2x_center_{_zoom_tag(zoom)}_temperature.png"
    savefig_academic(fig, path)
    return path


def save_unet_temperature_view(
    unet_temp: np.ndarray,
    output_dir: Path,
    *,
    zoom: float,
    center_fraction: float,
    step_label: str = "step 4000",
) -> Path:
    setup_academic_style()
    image = _zoom_center(unet_temp, center_fraction=center_fraction, zoom=zoom)
    if np.isfinite(image).any():
        values = image[np.isfinite(image)].ravel()
        vmin, vmax = float(np.percentile(values, 1.0)), float(np.percentile(values, 99.0))
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(1, 1, figsize=(4.1, 3.0), squeeze=True)
    im = ax.imshow(image, cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(f"UNet 2x @ EP07 {step_label} (temperature)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Temperature [deg C]")
    path = output_dir / f"unet_{step_label.replace(' ', '')}_center_{_zoom_tag(zoom)}_temperature.png"
    savefig_academic(fig, path)
    return path


def _read_baseline_metrics(sweep_path: Path, summary_path: Path, baseline_name: str) -> dict[str, float | str]:
    if not sweep_path.exists():
        return {
            "split_half_nrmse": float("nan"),
            "artifact_score": float("nan"),
            "raw_control_corr": float("nan"),
            "notes": f"{baseline_name} sweep_results.csv missing; image loaded only.",
        }
    sweep = pd.read_csv(sweep_path)
    if "label" in sweep.columns and summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        label = summary.get("best_tgv_label") or summary.get("best_label")
        rows = sweep[sweep["label"].astype(str) == str(label)]
        if rows.empty:
            rows = sweep.head(1)
    elif "pixfrac" in sweep.columns:
        rows = sweep[np.isclose(pd.to_numeric(sweep["pixfrac"], errors="coerce"), 1.0)]
        if rows.empty:
            rows = sweep.sort_values("split_half_nrmse", na_position="last").head(1)
    else:
        rows = sweep.head(1)
    row = rows.iloc[0]
    split_key = "split_half_nrmse_median" if "split_half_nrmse_median" in row.index else "split_half_nrmse"
    label = str(row.get("label", baseline_name))
    return {
        "split_half_nrmse": float(row.get(split_key, np.nan)),
        "artifact_score": float(row.get("artifact_score", np.nan)),
        "raw_control_corr": float(row.get("raw_control_corr", np.nan)),
        "notes": f"Loaded from {sweep_path.name}; selected {label}.",
    }


def _safe_split_half(
    raw_frames: np.ndarray,
    shifts: np.ndarray,
    model_cfg: dict[str, Any],
    state_dict: dict[str, torch.Tensor],
    *,
    device: str,
    scale: int,
    patch_size_hr: int,
    overlap: int,
    sigma_bg: float,
    residual: bool,
    n_splits: int,
    random_state: int,
) -> float:
    if n_splits <= 0:
        return float("nan")

    def method(frames_subset: np.ndarray, shifts_subset: np.ndarray, **_: Any) -> np.ndarray:
        local_model = _build_model(model_cfg, state_dict)
        temp = _run_unet_temperature(
            local_model,
            frames_subset,
            shifts_subset,
            device=device,
            scale=scale,
            patch_size_hr=patch_size_hr,
            overlap=overlap,
            sigma_bg=sigma_bg,
            residual=residual,
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
    unet_name: str,
    unet_metrics: dict[str, float],
    baseline_metrics: dict[str, float | str],
    unet_hp: np.ndarray,
    baseline_hp: np.ndarray,
    baseline_name: str,
    center_fraction: float,
    zoom: float,
    n_frames: int,
    device: str,
) -> None:
    unet_crop = center_fraction_crop(unet_hp, center_fraction)
    baseline_crop = center_fraction_crop(baseline_hp, center_fraction)
    unet_art = float(unet_metrics.get("artifact_score", np.nan))
    baseline_art = float(baseline_metrics.get("artifact_score", np.nan))
    unet_split = float(unet_metrics.get("split_half_nrmse", np.nan))
    baseline_split = float(baseline_metrics.get("split_half_nrmse", np.nan))
    unet_std = float(np.nanstd(unet_crop))
    baseline_std = float(np.nanstd(baseline_crop))

    if np.isfinite(unet_art) and np.isfinite(baseline_art) and np.isfinite(unet_split) and np.isfinite(baseline_split):
        if unet_art < baseline_art and unet_split <= 1.25 * baseline_split:
            preference = f"{unet_name} is worth the next focused continuation run, but only after visual inspection confirms the apparent edges are not synthetic-domain artifacts."
        else:
            preference = f"{baseline_name} remains the safer near-term baseline; UNet continuation should be gated by whether the center crop shows real inner-contour gains."
    else:
        preference = "Preference cannot be decided from proxies alone; use the two center-zoom figures as the primary EP11 evidence."

    lines = [
        f"# EP11 {unet_name} vs {baseline_name} Notes",
        "",
        f"- Input: {n_frames} clean main-session frames; device used: `{device}`; center crop fraction={center_fraction:g}, display zoom={zoom:g}.",
        "- The highpass figure is the fair visual comparison: both panels use the same center ROI, same 3x display zoom, same residual_diff colormap, and shared 99th-percentile symmetric limits.",
        "- The temperature figure is UNet-only sanity context. The raw-mean control is not shown as an algorithm comparison; it is used only for raw-control correlation.",
        f"- Center-crop highpass contrast std: UNet={unet_std:.6g} deg C, {baseline_name}={baseline_std:.6g} deg C. Higher contrast can mean clearer edges, but can also mean ringing or hallucinated texture.",
        f"- Proxy metrics: UNet split-half NRMSE={unet_split:.6g}, artifact score={unet_art:.6g}, raw-control corr={unet_metrics.get('raw_control_corr', float('nan')):.6g}; {baseline_name} split-half NRMSE={baseline_split:.6g}, artifact score={baseline_art:.6g}, raw-control corr={float(baseline_metrics.get('raw_control_corr', np.nan)):.6g}.",
        f"- Working recommendation: {preference}",
        "- Boundary: this is a contour-level visual benchmark, not a 5 um metrology or temperature-accuracy claim. Tenengrad/sharpness alone is not used to declare a winner.",
        "- Boundary: 3x is display zoom on the center ROI; the reconstruction grid is still the 2x EP07 output because this checkpoint was trained with scale=2.",
        f"- Boundary: {unet_name} is a synthetic-pretrained checkpoint, so any real-data edge improvement must be treated as domain-gap-sensitive until split-half and raw-control behavior are stable.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device, allow_cuda0=args.allow_cuda0)

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

    if args.reconstruct_tgv:
        print(
            "Running MAP-TGV reconstruction on CPU "
            f"(lambda={args.tgv_lambda:g}, sigma={args.tgv_psf_sigma:g})"
        )
        baseline = reconstruct_tgv_highpass(
            raw_frames,
            shifts,
            sigma_bg=float(args.highpass_sigma),
            lambda_tv=float(args.tgv_lambda),
            psf_sigma=float(args.tgv_psf_sigma),
            aniso_ratio_y=float(args.tgv_aniso_ratio_y),
            coverage_weighted=bool(args.tgv_coverage_weighted),
            workers=int(args.workers),
        )
        np.save(output_dir / "tgv_best_hr_highpass.npy", baseline.astype(np.float32, copy=False))
        baseline_temp = tgv_highpass_to_temperature(
            baseline,
            raw_frames,
            scale=int(args.scale),
            sigma_bg=float(args.highpass_sigma),
        )
        np.save(output_dir / "tgv_best_hr_temperature.npy", baseline_temp.astype(np.float32, copy=False))
        print(f"Reconstructed TGV highpass shape={baseline.shape}")
    else:
        baseline = np.load(args.baseline_hr).astype(np.float32, copy=False)
        print(f"Loaded baseline highpass ({args.baseline_name}): {args.baseline_hr} {baseline.shape}")
        baseline_temp_path = args.baseline_temperature
        if baseline_temp_path is not None and baseline_temp_path.exists():
            baseline_temp = np.load(baseline_temp_path).astype(np.float32, copy=False)
            print(f"Loaded baseline temperature: {baseline_temp_path} {baseline_temp.shape}")
        else:
            baseline_temp = tgv_highpass_to_temperature(
                baseline,
                raw_frames,
                scale=int(args.scale),
                sigma_bg=float(args.highpass_sigma),
            )
            np.save(output_dir / "tgv_best_hr_temperature.npy", baseline_temp.astype(np.float32, copy=False))

    cfg, state_dict, checkpoint_step = _load_checkpoint_config(args.checkpoint)
    residual = bool(cfg.get("residual", True))
    scale = int(args.scale or cfg.get("scale", 2))
    patch_size_hr = int(args.patch_size_hr or cfg.get("patch_size_hr", 256))
    model = _build_model(cfg, state_dict)

    print("Running UNet full-frame inference")
    unet_temp = _run_unet_temperature(
        model,
        raw_frames,
        shifts,
        device=device,
        scale=scale,
        patch_size_hr=patch_size_hr,
        overlap=int(args.overlap),
        sigma_bg=float(args.highpass_sigma),
        residual=residual,
    )
    unet_hp = highpass_preprocess(unet_temp, sigma_bg=float(args.highpass_sigma))

    expected_shape = (raw_frames.shape[1] * scale, raw_frames.shape[2] * scale)
    if unet_hp.shape != expected_shape:
        raise ValueError(f"UNet highpass shape {unet_hp.shape} != expected {expected_shape}")
    if baseline.shape != unet_hp.shape:
        raise ValueError(f"Baseline shape {baseline.shape} != UNet shape {unet_hp.shape}")
    if baseline_temp.shape != unet_temp.shape:
        raise ValueError(f"Baseline temperature shape {baseline_temp.shape} != UNet shape {unet_temp.shape}")
    if args.limit is None and unet_hp.shape != DEFAULT_EXPECTED_SHAPE:
        raise ValueError(f"Full EP11 shape {unet_hp.shape} != expected {DEFAULT_EXPECTED_SHAPE}")
    if not np.isfinite(unet_hp).any():
        raise ValueError("UNet highpass contains no finite values")

    step_num, step_label, step_suffix = _checkpoint_labels(args.checkpoint, checkpoint_step)
    unet_method_name = f"UNet@{step_num}"

    unet_temp_path = output_dir / f"unet_{step_suffix}_hr_temp.npy"
    unet_hp_path = output_dir / f"unet_{step_suffix}_hr_highpass.npy"
    np.save(unet_temp_path, unet_temp.astype(np.float32, copy=False))
    np.save(unet_hp_path, unet_hp.astype(np.float32, copy=False))

    raw_control_temp = bicubic_upsample(np.nanmean(raw_frames, axis=0), scale=scale)
    raw_control_hp = highpass_preprocess(raw_control_temp, sigma_bg=float(args.highpass_sigma))
    np.save(output_dir / "raw_mean_control_2x_hr_temp.npy", raw_control_temp.astype(np.float32, copy=False))
    np.save(output_dir / "raw_mean_control_2x_hr_highpass.npy", raw_control_hp.astype(np.float32, copy=False))

    print(f"Writing center-zoom figures using {step_label}")
    highpass_fig = save_highpass_comparison(
        unet_hp,
        baseline,
        output_dir,
        baseline_name=str(args.baseline_name),
        zoom=float(args.zoom),
        center_fraction=float(args.center_fraction),
        step_label=step_label,
    )
    raw_fig = save_unet_temperature_view(
        unet_temp,
        output_dir,
        zoom=float(args.zoom),
        center_fraction=float(args.center_fraction),
        step_label=step_label,
    )
    temp_fig = save_temperature_comparison(
        unet_temp,
        baseline_temp,
        output_dir,
        baseline_name=str(args.baseline_name),
        zoom=float(args.zoom),
        center_fraction=float(args.center_fraction),
        step_label=step_label,
    )

    print("Computing proxy metrics")
    split_nrmse = _safe_split_half(
        raw_frames,
        shifts,
        cfg,
        state_dict,
        device=device,
        scale=scale,
        patch_size_hr=patch_size_hr,
        overlap=int(args.overlap),
        sigma_bg=float(args.highpass_sigma),
        residual=residual,
        n_splits=int(args.n_splits),
        random_state=int(args.random_state),
    )
    unet_metrics = {
        "split_half_nrmse": split_nrmse,
        "artifact_score": float(artifact_score(unet_hp, scale=scale)),
        "raw_control_corr": _pearson_finite(unet_hp, raw_control_hp),
    }
    baseline_metrics = _read_baseline_metrics(args.baseline_sweep, args.baseline_summary, str(args.baseline_name))

    rows = [
        {
            "method": unet_method_name,
            **unet_metrics,
            "notes": f"Computed in EP11 on {len(raw_frames)} frame(s); split_half n_splits={int(args.n_splits)}.",
        },
        {"method": str(args.baseline_name), **baseline_metrics},
    ]
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "comparison_summary.csv", index=False)

    _write_notes(
        output_dir / "comparison_notes.md",
        unet_name=unet_method_name,
        unet_metrics=unet_metrics,
        baseline_metrics=baseline_metrics,
        unet_hp=unet_hp,
        baseline_hp=baseline,
        baseline_name=str(args.baseline_name),
        center_fraction=float(args.center_fraction),
        zoom=float(args.zoom),
        n_frames=len(raw_frames),
        device=device,
    )

    _write_json(
        output_dir / "run_manifest.json",
        {
            "checkpoint": _relative(args.checkpoint),
            "baseline_name": str(args.baseline_name),
            "baseline_hr": _relative(args.baseline_hr),
            "baseline_sweep": _relative(args.baseline_sweep),
            "baseline_summary": _relative(args.baseline_summary),
            "output_dir": _relative(output_dir),
            "frames_shape": list(raw_frames.shape),
            "shifts_shape": list(shifts.shape),
            "unet_temp_shape": list(unet_temp.shape),
            "unet_highpass_shape": list(unet_hp.shape),
            "baseline_shape": list(baseline.shape),
            "device_requested": args.device,
            "device_used": device,
            "scale": scale,
            "residual": residual,
            "highpass_sigma": float(args.highpass_sigma),
            "patch_size_hr": patch_size_hr,
            "overlap": int(args.overlap),
            "center_fraction": float(args.center_fraction),
            "zoom": float(args.zoom),
            "zoom_role": "display zoom only; reconstruction scale remains 2x",
            "n_splits": int(args.n_splits),
            "figures": [_relative(highpass_fig), _relative(temp_fig), _relative(raw_fig)],
            "tgv_reconstructed": bool(args.reconstruct_tgv),
        },
    )

    print(summary.to_string(index=False))
    print(f"Wrote {_relative(unet_hp_path)}")
    print(f"Wrote {_relative(highpass_fig)}")
    print(f"Wrote {_relative(temp_fig)}")
    print(f"Wrote {_relative(raw_fig)}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--baseline-hr", "--drizzle-hr", dest="baseline_hr", type=Path, default=DEFAULT_BASELINE_HR)
    parser.add_argument("--baseline-sweep", "--drizzle-sweep", dest="baseline_sweep", type=Path, default=DEFAULT_BASELINE_SWEEP)
    parser.add_argument("--baseline-summary", type=Path, default=DEFAULT_BASELINE_SUMMARY)
    parser.add_argument("--baseline-name", default=DEFAULT_BASELINE_NAME)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--alignment-method", default="contour_refined")
    parser.add_argument("--highpass-sigma", type=float, default=5.0)
    parser.add_argument("--patch-size-hr", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=32)
    parser.add_argument("--zoom", type=float, default=3.0)
    parser.add_argument("--center-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--baseline-temperature", type=Path, default=None)
    parser.add_argument("--reconstruct-tgv", action="store_true")
    parser.add_argument("--tgv-lambda", type=float, default=DEFAULT_TGV_LAMBDA)
    parser.add_argument("--tgv-psf-sigma", type=float, default=DEFAULT_TGV_PSF_SIGMA)
    parser.add_argument("--tgv-aniso-ratio-y", type=float, default=DEFAULT_TGV_ANISO_RATIO_Y)
    parser.add_argument("--tgv-coverage-weighted", action=argparse.BooleanOptionalAction, default=True)
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
