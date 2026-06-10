#!/usr/bin/env python3
"""Compare EP07 2x output upsampled to 4x against EP12 native 4x."""

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
EP07_SRC = PROJECT_ROOT / "algos" / "ep07_unet_sr" / "src"
EP12_SRC = PROJECT_ROOT / "algos" / "ep12_4x_sr" / "src"
for path in (ALGO_ROOT / "src", EP12_SRC, EP07_SRC, EP06_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from common.alignment import load_alignment_shifts  # noqa: E402
from common.data_loader import (  # noqa: E402
    bicubic_upsample,
    highpass_preprocess,
    load_main_session_frames,
    load_main_session_metadata,
)
from sr4x.inference import bare_drizzle_temperature, infer_from_burst as infer_ep12_from_burst  # noqa: E402
from sr4x.model import ThermalSR4xUNet  # noqa: E402
from thermal_core.ep10_cache import center_fraction_crop  # noqa: E402
from thermal_core.plotting import COLORMAPS, savefig_academic, setup_academic_style  # noqa: E402
from unet_sr.inference import infer_from_burst as infer_ep07_from_burst  # noqa: E402
from unet_sr.model import ThermalSRUNet  # noqa: E402


DEFAULT_EP07_CHECKPOINT = (
    PROJECT_ROOT / "algos" / "ep07_unet_sr" / "outputs" / "ep07_v6_physics" / "model_final.pt"
)
DEFAULT_EP12_CHECKPOINT = (
    PROJECT_ROOT / "algos" / "ep12_4x_sr" / "outputs" / "ep12_hybrid_v1" / "checkpoint_step_048000.pt"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "ep12_4x_benchmark" / "ep07x2up_vs_ep12"
DEFAULT_EXPECTED_SHAPE = (1920, 2560)


def _relative(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor], int]:
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"Unsupported checkpoint format: {path}")
    cfg = dict(ckpt.get("config") or {})
    config_path = path.parent / "config.json"
    if config_path.exists():
        disk_cfg = json.loads(config_path.read_text(encoding="utf-8"))
        cfg = {**disk_cfg, **cfg}
    return cfg, ckpt["model_state_dict"], int(ckpt.get("step", 0))


def _step_suffix(path: Path, step: int) -> str:
    match = re.search(r"step_0*(\d+)", path.stem)
    if match:
        return f"step{match.group(1)}"
    if step > 0:
        return f"step{step}"
    return path.stem


def _build_ep07_model(cfg: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> ThermalSRUNet:
    residual = bool(cfg.get("residual", False))
    scale = int(cfg.get("scale", 2))
    model = ThermalSRUNet(
        in_channels=int(cfg.get("in_channels", 6 if residual else 5)),
        out_channels=int(cfg.get("out_channels", 1)),
        base_channels=int(cfg.get("base_channels", 64)),
        scale=1 if residual else scale,
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _build_ep12_model(cfg: dict[str, Any], state_dict: dict[str, torch.Tensor]) -> ThermalSR4xUNet:
    defaults = {
        "in_channels": 8,
        "out_channels": 1,
        "base_channels": 48,
        "unet_depth": 4,
        "dilated_bottleneck": True,
        "predict_log_variance": True,
        "min_log_variance": -8.0,
        "max_log_variance": 4.0,
    }
    cfg = {**defaults, **cfg}
    model_scale = int(cfg.get("model_scale", int(cfg.get("scale", 4)) // int(cfg.get("drizzle_scale", 2))))
    model = ThermalSR4xUNet(
        in_channels=int(cfg["in_channels"]),
        out_channels=int(cfg["out_channels"]),
        base_channels=int(cfg["base_channels"]),
        scale=model_scale,
        depth=int(cfg["unet_depth"]),
        dilated_bottleneck=bool(cfg["dilated_bottleneck"]),
        predict_log_variance=bool(cfg["predict_log_variance"]),
        min_log_variance=float(cfg["min_log_variance"]),
        max_log_variance=float(cfg["max_log_variance"]),
    )
    model.load_state_dict(state_dict)
    model.eval()
    return model


def _pearson_finite(a: np.ndarray, b: np.ndarray) -> float:
    lhs = np.asarray(a, dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(lhs) & np.isfinite(rhs)
    if int(valid.sum()) < 2:
        return float("nan")
    return float(np.corrcoef(lhs[valid].ravel(), rhs[valid].ravel())[0, 1])


def _artifact_score(image: np.ndarray) -> float:
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim != 2 or not np.isfinite(arr).any():
        return float("inf")
    high_freq = arr - ndimage.gaussian_filter(arr, sigma=1.0, mode="nearest")
    lap = ndimage.laplace(arr, mode="nearest")
    base = float(np.nanstd(arr))
    if base <= 1e-12:
        return 0.0
    return float((np.nanstd(high_freq) + 0.25 * np.nanstd(lap)) / base)


def _p95_gradient(image: np.ndarray) -> float:
    arr = np.asarray(image, dtype=np.float32)
    gy, gx = np.gradient(arr)
    mag = np.hypot(gx, gy)
    values = mag[np.isfinite(mag)]
    return float(np.percentile(values, 95.0)) if values.size else float("nan")


def _shared_vmax(images: list[np.ndarray], percentile: float = 99.0) -> float:
    finite = [np.abs(image[np.isfinite(image)]).ravel() for image in images if np.isfinite(image).any()]
    if not finite:
        return 1.0
    return max(float(np.percentile(np.concatenate(finite), percentile)), 1e-6)


def _temperature_limits(images: list[np.ndarray]) -> tuple[float, float]:
    finite = [image[np.isfinite(image)].ravel() for image in images if np.isfinite(image).any()]
    if not finite:
        return 0.0, 1.0
    values = np.concatenate(finite)
    return float(np.percentile(values, 1.0)), float(np.percentile(values, 99.0))


def _crop_at_fraction(image: np.ndarray, *, fraction: float, y_frac: float = 0.5, x_frac: float = 0.5) -> np.ndarray:
    rows, cols = image.shape
    crop_rows = max(1, int(round(rows * fraction)))
    crop_cols = max(1, int(round(cols * fraction)))
    cy = int(round(rows * y_frac))
    cx = int(round(cols * x_frac))
    y0 = min(max(0, cy - crop_rows // 2), rows - crop_rows)
    x0 = min(max(0, cx - crop_cols // 2), cols - crop_cols)
    return image[y0 : y0 + crop_rows, x0 : x0 + crop_cols]


def _display_crop(
    image: np.ndarray,
    *,
    center_fraction: float,
    zoom: float,
    roi_fraction: float | None = None,
    roi_y_frac: float = 0.5,
    roi_x_frac: float = 0.5,
) -> np.ndarray:
    if roi_fraction is None:
        crop = center_fraction_crop(np.asarray(image), fraction=float(center_fraction))
    else:
        crop = _crop_at_fraction(
            np.asarray(image),
            fraction=float(roi_fraction),
            y_frac=float(roi_y_frac),
            x_frac=float(roi_x_frac),
        )
    return ndimage.zoom(crop, zoom=float(zoom), order=1).astype(np.float32, copy=False)


def _save_panel_figure(
    arrays: dict[str, np.ndarray],
    output_path: Path,
    *,
    mode: str,
    zoom: float,
    center_fraction: float,
    roi_fraction: float | None = None,
    roi_y_frac: float = 0.5,
    roi_x_frac: float = 0.5,
) -> Path:
    setup_academic_style()
    panels = [
        (
            name,
            _display_crop(
                image,
                center_fraction=center_fraction,
                zoom=zoom,
                roi_fraction=roi_fraction,
                roi_y_frac=roi_y_frac,
                roi_x_frac=roi_x_frac,
            ),
        )
        for name, image in arrays.items()
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(min(13.2, 3.0 * len(panels)), 3.2), squeeze=False)
    if mode == "temperature":
        vmin, vmax = _temperature_limits(list(arrays.values()))
        cmap = COLORMAPS["temperature"]
        label = "Temperature [deg C]"
    elif mode == "highpass":
        vmax = _shared_vmax([image for _, image in panels])
        vmin = -vmax
        cmap = COLORMAPS["residual_diff"]
        label = "Highpass response [deg C]"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    for ax, (title, image) in zip(axes.ravel(), panels, strict=True):
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02).set_label(label)
    savefig_academic(fig, output_path)
    return output_path


def _write_notes(
    path: Path,
    *,
    metrics: pd.DataFrame,
    ep07_hp: np.ndarray,
    ep12_hp: np.ndarray,
    center_fraction: float,
) -> None:
    ep07_std = float(np.nanstd(center_fraction_crop(ep07_hp, center_fraction)))
    ep12_std = float(np.nanstd(center_fraction_crop(ep12_hp, center_fraction)))
    by_method = metrics.set_index("method")
    ep07_corr = float(by_method.loc["EP07 2x x2up", "raw_control_highpass_pearson"])
    ep12_corr = float(by_method.loc["EP12 4x", "raw_control_highpass_pearson"])
    ep07_art = float(by_method.loc["EP07 2x x2up", "artifact_score"])
    ep12_art = float(by_method.loc["EP12 4x", "artifact_score"])
    if ep12_std > 1.05 * ep07_std and ep12_corr >= ep07_corr - 0.02 and ep12_art <= 1.25 * ep07_art:
        conclusion = "EP12 may have visible contour gain over EP07x2up, but center-ROI inspection must rule out synthetic artifacts."
    else:
        conclusion = "EP12 4x does not show a clear low-risk gain over EP07x2up by the proxy gate; visual ROI inspection should decide whether 4x training remains justified."

    lines = [
        "# EP07 2x x2up vs EP12 4x Notes",
        "",
        "- Input: EP06 clean main-session 248 frames with contour_refined shifts.",
        "- Arm A: EP07 v6 physics final checkpoint, native 2x inference, scipy cubic x2 display-grid upsample to 4x.",
        "- Arm B: EP12 hybrid v1 step 48000 native 4x inference.",
        "- Arm C: bare drizzle mean at 4x.",
        "- Arm D: bicubic LR mean at 4x.",
        "- Temperature figures use shared full-frame 1-99 percentile limits across all four arms.",
        "- Highpass figures use sigma=5 highpass and shared symmetric 99th-percentile limits.",
        f"- Center-crop highpass std: EP07x2up={ep07_std:.6g} deg C, EP12={ep12_std:.6g} deg C.",
        f"- Raw-control highpass Pearson: EP07x2up={ep07_corr:.6g}, EP12={ep12_corr:.6g}.",
        f"- Artifact score: EP07x2up={ep07_art:.6g}, EP12={ep12_art:.6g}.",
        f"- Working conclusion: {conclusion}",
        "- Boundary: P95 gradient is auxiliary only; it can increase from ringing/noise and is not SR proof.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args: argparse.Namespace) -> pd.DataFrame:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    device = _resolve_device(args.device, allow_cuda0=args.allow_cuda0)
    print(f"Using device: {device}")

    print("Loading clean main-session frames and contour-refined shifts")
    raw_frames, metadata = load_main_session_frames(workers=args.workers, dtype=np.float32, limit=args.limit)
    if args.limit is None:
        shifts = load_alignment_shifts(args.alignment_method, metadata=metadata).astype(np.float32, copy=False)
    else:
        full_metadata = load_main_session_metadata()
        full_shifts = load_alignment_shifts(args.alignment_method, metadata=full_metadata).astype(np.float32, copy=False)
        shifts = full_shifts[: len(metadata)]
    print(f"Loaded frames={raw_frames.shape}, shifts={shifts.shape}")

    print("Loading EP07 and EP12 checkpoints")
    ep07_cfg, ep07_state, ep07_step = _load_checkpoint(args.ep07_checkpoint)
    ep12_cfg, ep12_state, ep12_step = _load_checkpoint(args.ep12_checkpoint)
    ep07_model = _build_ep07_model(ep07_cfg, ep07_state)
    ep12_model = _build_ep12_model(ep12_cfg, ep12_state)

    ep07_scale = int(ep07_cfg.get("scale", 2))
    ep12_scale = int(args.scale or ep12_cfg.get("scale", 4))
    if ep07_scale != 2 or ep12_scale != 4:
        raise ValueError(f"Expected EP07 scale=2 and EP12 scale=4; got {ep07_scale=} {ep12_scale=}")
    highpass_sigma = float(args.highpass_sigma)

    print("Running EP07 native 2x inference, then cubic x2 upsample to 4x")
    ep07_2x_temp = infer_ep07_from_burst(
        ep07_model,
        raw_frames,
        shifts,
        scale=2,
        patch_size_hr=int(args.ep07_patch_size_hr or ep07_cfg.get("patch_size_hr", 256)),
        overlap=int(args.ep07_overlap),
        device=device,
        sigma_bg=highpass_sigma,
        residual=bool(ep07_cfg.get("residual", False)),
    ).astype(np.float32, copy=False)
    ep07_4x_temp = ndimage.zoom(ep07_2x_temp, zoom=2.0, order=3, mode="nearest").astype(np.float32, copy=False)

    print("Running EP12 native 4x inference")
    ep12_temp = infer_ep12_from_burst(
        ep12_model,
        raw_frames,
        shifts,
        scale=4,
        drizzle_scale=int(ep12_cfg.get("drizzle_scale", 2)),
        patch_size=int(args.ep12_patch_size or ep12_cfg.get("patch_size", 256)),
        overlap=int(args.ep12_overlap),
        device=device,
        sigma_bg=highpass_sigma,
        drizzle_kernel=str(args.drizzle_kernel or ep12_cfg.get("drizzle_kernel", "bilinear")),
    ).astype(np.float32, copy=False)

    print("Building bare drizzle and bicubic raw-mean controls")
    drizzle_temp = bare_drizzle_temperature(
        raw_frames,
        shifts,
        scale=4,
        drizzle_scale=int(ep12_cfg.get("drizzle_scale", 2)),
        drizzle_kernel=str(args.drizzle_kernel or ep12_cfg.get("drizzle_kernel", "bilinear")),
    ).astype(np.float32, copy=False)
    bicubic_temp = bicubic_upsample(np.nanmean(raw_frames, axis=0), scale=4).astype(np.float32, copy=False)

    temps = {
        "EP07 2x x2up": ep07_4x_temp,
        "EP12 4x": ep12_temp,
        "Bare drizzle 4x": drizzle_temp,
        "Bicubic LR mean 4x": bicubic_temp,
    }
    expected = (raw_frames.shape[1] * 4, raw_frames.shape[2] * 4)
    for name, image in temps.items():
        if image.shape != expected:
            raise ValueError(f"{name} shape {image.shape} != expected {expected}")
    if args.limit is None and expected != DEFAULT_EXPECTED_SHAPE:
        raise ValueError(f"Full 4x shape {expected} != expected {DEFAULT_EXPECTED_SHAPE}")

    highpasses = {name: highpass_preprocess(image, sigma_bg=highpass_sigma) for name, image in temps.items()}
    raw_control_hp = highpasses["Bicubic LR mean 4x"]

    for stem, image in temps.items():
        np.save(output_dir / f"{stem.lower().replace(' ', '_')}_temp.npy", image.astype(np.float32, copy=False))
    for stem, image in highpasses.items():
        np.save(output_dir / f"{stem.lower().replace(' ', '_')}_highpass.npy", image.astype(np.float32, copy=False))

    print("Writing four-arm figures")
    temp_fig = _save_panel_figure(
        temps,
        output_dir / "ep07x2up_vs_ep12_center_zoom3x_temperature.png",
        mode="temperature",
        zoom=float(args.zoom),
        center_fraction=float(args.center_fraction),
    )
    hp_fig = _save_panel_figure(
        highpasses,
        output_dir / "ep07x2up_vs_ep12_center_zoom3x_highpass.png",
        mode="highpass",
        zoom=float(args.zoom),
        center_fraction=float(args.center_fraction),
    )
    roi_temp_fig = _save_panel_figure(
        temps,
        output_dir / "ep07x2up_vs_ep12_zigzag_roi_temperature.png",
        mode="temperature",
        zoom=float(args.roi_zoom),
        center_fraction=float(args.center_fraction),
        roi_fraction=float(args.roi_fraction),
        roi_y_frac=float(args.roi_y_frac),
        roi_x_frac=float(args.roi_x_frac),
    )
    roi_hp_fig = _save_panel_figure(
        highpasses,
        output_dir / "ep07x2up_vs_ep12_zigzag_roi_highpass.png",
        mode="highpass",
        zoom=float(args.roi_zoom),
        center_fraction=float(args.center_fraction),
        roi_fraction=float(args.roi_fraction),
        roi_y_frac=float(args.roi_y_frac),
        roi_x_frac=float(args.roi_x_frac),
    )

    rows = []
    for name, hp in highpasses.items():
        rows.append(
            {
                "method": name,
                "artifact_score": _artifact_score(hp),
                "raw_control_highpass_pearson": _pearson_finite(hp, raw_control_hp),
                "p95_gradient_highpass": _p95_gradient(hp),
                "temperature_shape": "x".join(map(str, temps[name].shape)),
            }
        )
    metrics = pd.DataFrame(rows)
    metrics.to_csv(output_dir / "metrics_summary.csv", index=False)
    _write_notes(
        output_dir / "comparison_notes.md",
        metrics=metrics,
        ep07_hp=highpasses["EP07 2x x2up"],
        ep12_hp=highpasses["EP12 4x"],
        center_fraction=float(args.center_fraction),
    )

    _write_json(
        output_dir / "run_manifest.json",
        {
            "ep07_checkpoint": _relative(args.ep07_checkpoint),
            "ep07_step": ep07_step,
            "ep12_checkpoint": _relative(args.ep12_checkpoint),
            "ep12_step": ep12_step,
            "output_dir": _relative(output_dir),
            "frames_shape": list(raw_frames.shape),
            "shifts_shape": list(shifts.shape),
            "alignment_method": args.alignment_method,
            "device_requested": args.device,
            "device_used": device,
            "highpass_sigma": highpass_sigma,
            "center_fraction": float(args.center_fraction),
            "zoom": float(args.zoom),
            "roi_fraction": float(args.roi_fraction),
            "roi_center_fraction_yx": [float(args.roi_y_frac), float(args.roi_x_frac)],
            "roi_zoom": float(args.roi_zoom),
            "ep07_suffix": _step_suffix(args.ep07_checkpoint, ep07_step),
            "ep12_suffix": _step_suffix(args.ep12_checkpoint, ep12_step),
            "figures": [_relative(p) for p in (temp_fig, hp_fig, roi_temp_fig, roi_hp_fig)],
            "metrics_csv": _relative(output_dir / "metrics_summary.csv"),
        },
    )

    print(metrics.round(6).to_string(index=False))
    print(f"Wrote {_relative(output_dir)}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep07-checkpoint", type=Path, default=DEFAULT_EP07_CHECKPOINT)
    parser.add_argument("--ep12-checkpoint", type=Path, default=DEFAULT_EP12_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scale", type=int, default=4)
    parser.add_argument("--alignment-method", default="contour_refined")
    parser.add_argument("--highpass-sigma", type=float, default=5.0)
    parser.add_argument("--ep07-patch-size-hr", type=int, default=256)
    parser.add_argument("--ep07-overlap", type=int, default=128)
    parser.add_argument("--ep12-patch-size", type=int, default=None)
    parser.add_argument("--ep12-overlap", type=int, default=64)
    parser.add_argument("--drizzle-kernel", default=None)
    parser.add_argument("--zoom", type=float, default=3.0)
    parser.add_argument("--center-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--roi-fraction", type=float, default=1.0 / 6.0)
    parser.add_argument("--roi-y-frac", type=float, default=0.5)
    parser.add_argument("--roi-x-frac", type=float, default=0.5)
    parser.add_argument("--roi-zoom", type=float, default=5.0)
    parser.add_argument("--device", default="cuda:1")
    parser.add_argument("--allow-cuda0", action="store_true")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not (0.0 < float(args.center_fraction) <= 1.0):
        raise ValueError("--center-fraction must be in (0, 1]")
    if not (0.0 < float(args.roi_fraction) <= 1.0):
        raise ValueError("--roi-fraction must be in (0, 1]")
    if not math.isfinite(float(args.zoom)) or float(args.zoom) <= 0:
        raise ValueError("--zoom must be positive")
    if not math.isfinite(float(args.roi_zoom)) or float(args.roi_zoom) <= 0:
        raise ValueError("--roi-zoom must be positive")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
