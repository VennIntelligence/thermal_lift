"""EP14 Loss Atlas — 4x Drizzle-informed SR training-input + ThermalSR4xLoss visual guide."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle, FancyArrowPatch

from thermal_core.plotting import COLORMAPS, FIGURE_SIZES, METHOD_COLOR_LIST, savefig_academic, setup_academic_style


@dataclass(frozen=True)
class LossRecipe4x:
    lf_weight: float = 1.0
    hf_weight: float = 0.3
    edge_weight: float = 0.1
    forward_weight: float = 0.2
    nll_weight: float = 0.05
    hf_detail_weight: float = 0.3
    
    sigma_lf: float = 8.0
    psf_sigma_lr_px: float = 0.25
    coverage_gain: float = 4.0
    hf_detail_gain: float = 4.0
    edge_mask_boost: float = 2.0
    edge_coarse_weight: float = 0.25
    min_log_variance: float = -8.0
    max_log_variance: float = 4.0


OBS_CHANNEL_LABELS_4X = [
    "ch0 drizzle mean (4x)",
    "ch1 drizzle coverage (4x)",
    "ch2 drizzle variance (4x)",
    "ch3 mean upsampled",
    "ch4 median upsampled",
    "ch5 coverage upsampled",
    "ch6 variance upsampled",
    "ch7 highpass upsampled",
]


def load_training_demo_bundle(output_dir: Path) -> dict[str, Any]:
    npz_path = output_dir / "training_demo_bundle.npz"
    meta_path = output_dir / "training_demo_meta.json"
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing TCForge 4x demo bundle under {output_dir}. Run scripts/build_ep14_cache.py first."
        )
    with np.load(npz_path) as data:
        bundle = {k: data[k] for k in data.files}
    bundle["meta"] = json.loads(meta_path.read_text(encoding="utf-8"))
    return bundle


def _gaussian_kernel_1d(sigma: float, size: int | None = None) -> np.ndarray:
    if sigma <= 0:
        return np.array([1.0], dtype=np.float64)
    if size is None:
        size = int(4 * sigma + 0.5) * 2 + 1
    coords = np.arange(size, dtype=np.float64) - (size - 1) / 2.0
    g = np.exp(-(coords**2) / (2.0 * sigma * sigma))
    return g / g.sum()


def gaussian_blur_2d(image: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return image.astype(np.float32, copy=True)
    g = _gaussian_kernel_1d(sigma)
    pad = len(g) // 2
    work = np.pad(image.astype(np.float64), ((pad, pad), (pad, pad)), mode="reflect")
    tmp = np.apply_along_axis(lambda row: np.convolve(row, g, mode="valid"), 1, work)
    out = np.apply_along_axis(lambda col: np.convolve(col, g, mode="valid"), 0, tmp)
    return out.astype(np.float32)


def sobel_edges(image: np.ndarray) -> np.ndarray:
    work = np.pad(image.astype(np.float64), 1, mode="reflect")
    gx = (
        work[:-2, 2:] + 2 * work[1:-1, 2:] + work[2:, 2:]
        - work[:-2, :-2] - 2 * work[1:-1, :-2] - work[2:, :-2]
    )
    gy = (
        work[2:, :-2] + 2 * work[2:, 1:-1] + work[2:, 2:]
        - work[:-2, :-2] - 2 * work[:-2, 1:-1] - work[:-2, 2:]
    )
    return np.sqrt(gx * gx + gy * gy).astype(np.float32)


def crop_center(arr: np.ndarray, patch: int) -> tuple[np.ndarray, tuple[int, int]]:
    h, w = arr.shape[-2], arr.shape[-1]
    y0 = max(0, h // 2 - patch // 2)
    x0 = max(0, w // 2 - patch // 2)
    y1 = min(h, y0 + patch)
    x1 = min(w, x0 + patch)
    if arr.ndim == 2:
        return arr[y0:y1, x0:x1], (y0, x0)
    if arr.ndim == 3:
        return arr[:, y0:y1, x0:x1], (y0, x0)
    raise ValueError("crop_center supports 2D or 3D arrays")


def make_loss_scene_from_bundle(bundle: dict[str, Any], *, patch: int = 128) -> dict[str, np.ndarray]:
    target, (y0, x0) = crop_center(bundle["hr_temperature"], patch)
    mask, _ = crop_center(bundle["hr_mask"], patch)
    obs, _ = crop_center(bundle["obs_features"], patch)
    
    # Simulate a prediction with noise + ringing near edges to demonstrate the losses
    yy, xx = np.mgrid[0:target.shape[0], 0:target.shape[1]]
    edge = sobel_edges(mask)
    edge /= max(float(edge.max()), 1e-6)
    ringing = 0.35 * np.sin(xx * 0.9) * np.sin(yy * 0.9) * (edge > 0.1)
    pred = target + ringing + np.random.default_rng(42).normal(0, 0.05, size=target.shape)
    
    # Simulate heteroscedastic uncertainty log_var
    # In reality, the model predicts higher uncertainty (larger log_var) near edges and low coverage areas
    edge_blur = gaussian_blur_2d(edge, 2.0)
    coverage_4x = obs[1] # drizzle coverage
    cov_norm = np.clip(coverage_4x / max(float(coverage_4x.max()), 1e-6), 0.0, 1.0)
    
    # Higher variance (uncert) near edges or low coverage zones
    var_sim = 0.005 + 0.1 * edge_blur + 0.08 * (1.0 - cov_norm)
    log_var = np.log(var_sim)
    
    return {
        "target": target.astype(np.float32),
        "pred": pred.astype(np.float32),
        "log_var": log_var.astype(np.float32),
        "mask": (mask > 0.5).astype(np.float32),
        "obs": obs.astype(np.float32),
        "crop_origin": np.array([y0, x0], dtype=np.int32),
    }


def _coverage_weight(coverage: np.ndarray, gain: float) -> np.ndarray:
    cov = np.clip(coverage, 0.0, None)
    max_val = max(float(cov.max()), 1e-6)
    cov_norm = np.clip(cov / max_val, 0.0, 1.0)
    return 1.0 + float(gain) * np.sqrt(cov_norm)


def _coverage_inverse_weight(coverage: np.ndarray, gain: float) -> np.ndarray:
    cov = np.clip(coverage, 0.0, None)
    max_val = max(float(cov.max()), 1e-6)
    cov_norm = np.clip(cov / max_val, 0.0, 1.0)
    return 1.0 + float(gain) * (1.0 - np.sqrt(cov_norm))


def _weighted_mean(value: np.ndarray, weight: np.ndarray | None = None) -> float:
    if weight is None:
        return float(value.mean())
    return float((value * weight).sum() / max(weight.sum(), 1e-6))


def compute_loss_breakdown_4x(scene: dict[str, np.ndarray], recipe: LossRecipe4x | None = None) -> dict[str, Any]:
    recipe = recipe or LossRecipe4x()
    pred, target = scene["pred"], scene["target"]
    log_var = scene["log_var"]
    obs = scene["obs"]
    coverage_4x = obs[1] # channel 1 is drizzle coverage 4x
    drizzle_mean_4x = obs[0] # channel 0 is drizzle mean 4x
    scale = 4

    cov_weight = _coverage_weight(coverage_4x, recipe.coverage_gain)

    # 1. Low Frequency Loss
    pred_lf = gaussian_blur_2d(pred, recipe.sigma_lf)
    target_lf = gaussian_blur_2d(target, recipe.sigma_lf)
    lf = float(np.abs(pred_lf - target_lf).mean())

    # 2. High Frequency Loss (weighted by coverage)
    pred_hf = pred - pred_lf
    target_hf = target - target_lf
    hf = _weighted_mean(np.abs(pred_hf - target_hf), cov_weight)

    # 3. Sobel Edge Loss
    pred_edges = sobel_edges(pred)
    target_edges = sobel_edges(target)
    edge_error = np.abs(pred_edges - target_edges)
    edge_mask = sobel_edges(scene["mask"])
    edge_mask /= max(float(edge_mask.max()), 1e-6)
    edge_w = 1.0 + recipe.edge_mask_boost * (edge_mask > 0.1)
    edge_fine = _weighted_mean(edge_error, edge_w)

    # Coarse edge
    pred_2x = pred.reshape(pred.shape[0] // 2, 2, pred.shape[1] // 2, 2).mean(axis=(1, 3))
    target_2x = target.reshape(target.shape[0] // 2, 2, target.shape[1] // 2, 2).mean(axis=(1, 3))
    edge_coarse = float(np.abs(sobel_edges(pred_2x) - sobel_edges(target_2x)).mean())
    edge = edge_fine + recipe.edge_coarse_weight * edge_coarse

    # 4. Forward Consistency Loss
    sigma_hr = max(recipe.psf_sigma_lr_px * scale, 1e-6)
    blurred_pred = gaussian_blur_2d(pred, sigma_hr)
    
    # Downsample HR blurred_pred to LR (4x avg pool)
    lr_h, lr_w = pred.shape[0] // scale, pred.shape[1] // scale
    lr_pred = blurred_pred.reshape(lr_h, scale, lr_w, scale).mean(axis=(1, 3))
    
    # Downsample observed drizzle mean and coverage
    lr_cov = coverage_4x.reshape(lr_h, scale, lr_w, scale).mean(axis=(1, 3))
    lr_num = (drizzle_mean_4x * coverage_4x).reshape(lr_h, scale, lr_w, scale).mean(axis=(1, 3))
    
    # Avoid zero coverage division
    safe_lr_cov = np.maximum(lr_cov, 1e-6)
    lr_obs = lr_num / safe_lr_cov
    
    forward_error = np.abs(lr_pred - lr_obs)
    forward = _weighted_mean(forward_error, lr_cov)

    # 5. Heteroscedastic NLL Loss
    lv = np.clip(log_var, recipe.min_log_variance, recipe.max_log_variance)
    nll_map = 0.5 * np.exp(-lv) * (pred - target)**2 + 0.5 * lv
    nll = _weighted_mean(nll_map, cov_weight)

    # 6. High Frequency Detail Loss (inverse coverage weight)
    inv_weight = _coverage_inverse_weight(coverage_4x, recipe.hf_detail_gain)
    hf_detail = _weighted_mean(np.abs(pred_hf - target_hf), inv_weight)

    total = (
        recipe.lf_weight * lf
        + recipe.hf_weight * hf
        + recipe.edge_weight * edge
        + recipe.forward_weight * forward
        + recipe.nll_weight * nll
        + recipe.hf_detail_weight * hf_detail
    )

    return {
        "recipe": asdict(recipe),
        "lf": lf,
        "hf": hf,
        "edge": edge,
        "forward": forward,
        "nll": nll,
        "hf_detail": hf_detail,
        "total": total,
        "maps": {
            "pred_lf": pred_lf,
            "target_lf": target_lf,
            "pred_hf": pred_hf,
            "target_hf": target_hf,
            "cov_weight": cov_weight,
            "inv_weight": inv_weight,
            "pred_edges": pred_edges,
            "target_edges": target_edges,
            "edge_error": edge_error,
            "edge_weight_map": edge_w,
            "lr_pred": lr_pred,
            "lr_obs": lr_obs,
            "lr_cov": lr_cov,
            "forward_error_lr": forward_error,
            "nll_map": nll_map,
            "uncertainty_sigma": np.sqrt(np.exp(lv)),
        },
    }


def _panel(ax, image, title, *, cmap, vmin=None, vmax=None, cbar_label=""):
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, origin="upper")
    ax.set_title(title, fontsize=8)
    ax.set_xticks([])
    ax.set_yticks([])
    if cbar_label:
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label, fontsize=7)
    return im


def _save(fig, output_dir: Path, name: str) -> str:
    path = output_dir / name
    savefig_academic(fig, path)
    return name


def _plot_training_pipeline_schematic_4x(output_dir: Path, meta: dict[str, Any]) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    n = int(meta.get("n_frames_per_scene", 248))
    boxes = [
        (0.2, 4.8, 2.0, 1.0, "TCForge\nHR mask + temp", "#E8EEF7"),
        (2.5, 4.8, 2.0, 1.0, f"Forward x{n}\nLR frames", "#FBE7C6"),
        (4.8, 5.4, 2.2, 0.7, "Drizzle 4x\n(3ch HR)", "#D8EAD3"),
        (4.8, 4.3, 2.2, 0.7, "Align & Fuse 1x\n(5ch LR)", "#D8EAD3"),
        (7.3, 4.8, 2.0, 1.0, "Concatenate\n8ch HR input", "#EFE3F2"),
        (9.6, 4.8, 1.8, 1.0, "Patch\nsampler", "#E8EEF7"),
        (11.8, 4.8, 1.8, 1.0, "UNet 4x", "#D8EAD3"),
        (11.8, 2.6, 1.8, 1.2, "pred 4x HR\n+ log_var", "#FBE7C6"),
        (9.0, 2.7, 2.0, 1.0, "GT temp\n(from mask)", "#FBE7C6"),
        (6.0, 1.0, 5.5, 1.0, "ThermalSR4xLoss (6 terms)", "#F3F3F3"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=0.8))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7)
    arrows = [
        ((2.2, 5.3), (2.5, 5.3)),
        ((4.5, 5.5), (4.8, 5.7)),
        ((4.5, 5.1), (4.8, 4.7)),
        ((7.0, 5.75), (7.3, 5.4)),
        ((7.0, 4.65), (7.3, 5.2)),
        ((9.3, 5.3), (9.6, 5.3)),
        ((11.4, 5.3), (11.8, 5.3)),
        ((12.7, 4.8), (12.7, 3.8)),
        ((11.8, 3.2), (11.0, 3.2)),
        ((10.0, 2.7), (10.0, 2.0)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=0.9))
    ax.text(
        0.2,
        0.2,
        f"Demo shows {meta.get('n_frames_demo', 16)} LR frames; offline fusion/drizzle uses {n} frames/scene.",
        fontsize=8,
    )
    ax.set_title("EP12 4x training data flow (8ch fusion -> dataloader -> 4x UNet -> Loss)", fontsize=10)
    return _save(fig, output_dir, "00_training_pipeline_schematic_4x.png")


def _plot_hr_mask_temperature_4x(bundle: dict[str, Any], output_dir: Path) -> str:
    mask, temp = bundle["hr_mask"], bundle["hr_temperature"]
    meta = bundle["meta"]
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["double_col"])
    _panel(axes[0], mask, f"HR mask 4x (theta={meta['rotation_deg']:.1f} deg)", cmap="gray", vmin=0, vmax=1)
    vmin, vmax = float(temp.min()), float(temp.max())
    _panel(axes[1], temp, "HR temperature 4x (TCForge render)", cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax, cbar_label="deg C")
    fig.suptitle("Step 1-2: 4x geometry + 4x temperature target on detector grid", fontsize=10, y=1.02)
    return _save(fig, output_dir, "02_hr_mask_and_temperature_4x.png")


def _plot_lr_burst_samples_4x(bundle: dict[str, Any], output_dir: Path) -> str:
    burst = bundle["lr_burst"]
    n_show = min(6, burst.shape[0])
    idx = np.linspace(0, burst.shape[0] - 1, n_show, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    vmin, vmax = float(burst.min()), float(burst.max())
    for ax, frame_idx in zip(axes.ravel(), idx, strict=True):
        _panel(ax, burst[frame_idx], f"LR frame {frame_idx} (1/4 size)", cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax)
    meta = bundle["meta"]
    total = int(meta.get("n_frames_per_scene", 248))
    physics = meta.get("physics_meta", {})
    fig.suptitle(
        f"Step 3: forward scale=4 + noise + drift ({n_show} of {burst.shape[0]} demo / {total} total)",
        fontsize=10,
        y=1.02,
    )
    return _save(fig, output_dir, "03_lr_burst_samples_4x.png")


def _plot_obs_channels_4x(bundle: dict[str, Any], output_dir: Path) -> str:
    obs = bundle["obs_features"]
    fig, axes = plt.subplots(2, 4, figsize=(9.6, 5.0))
    cmaps = [
        COLORMAPS["temperature"], COLORMAPS["coverage"], COLORMAPS["residual_pos"], # Drizzle 4x
        COLORMAPS["temperature"], COLORMAPS["temperature"], COLORMAPS["coverage"], COLORMAPS["residual_pos"], COLORMAPS["residual_diff"] # 1x upsampled
    ]
    for ax, ch, label, cmap in zip(axes.ravel(), range(8), OBS_CHANNEL_LABELS_4X, cmaps, strict=True):
        img = obs[ch]
        vmin, vmax = (0, 1) if "coverage" in label else (float(img.min()), float(img.max()))
        _panel(ax, img, label, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.suptitle("Step 4: concatenate 4x Drizzle (ch0-2) + 1x features upsampled (ch3-7) -> 8ch input", fontsize=10, y=1.02)
    return _save(fig, output_dir, "05_obs_feature_channels_4x.png")


def _plot_compact_storage_4x(output_dir: Path, meta: dict[str, Any]) -> str:
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["double_col"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    files = [
        (0.5, 3.8, "hr_mask_4x.png"),
        (0.5, 2.6, "hr_edge_4x.png"),
        (0.5, 1.4, "legacy obs_features_4x.npz (3ch drizzle)"),
        (0.5, 0.2, "obs_features_1x.npz + metadata.json"),
    ]
    for x, y, label in files:
        ax.add_patch(Rectangle((x, y), 4.0, 0.9, facecolor="#E8EEF7", edgecolor="black", linewidth=0.8))
        ax.text(x + 0.15, y + 0.45, label, fontsize=8, va="center")
    ax.add_patch(Rectangle((5.2, 1.0), 4.2, 3.5, fill=False, edgecolor="#C44E52", linewidth=1.2, linestyle="--"))
    ax.text(5.4, 4.2, "NOT stored in compact pool:", fontsize=8, color="#C44E52")
    n_frames = int(meta.get("n_frames_per_scene", 248))
    ax.text(5.4, 3.5, f"- full LR burst ({n_frames} frames)", fontsize=8)
    ax.text(5.4, 2.8, "- HR temperature .npy (GT)", fontsize=8)
    ax.text(5.4, 2.1, "GT reconstructed at train time", fontsize=8)
    ax.text(5.4, 1.4, "via reconstruct_hr_temperature()", fontsize=8)
    ax.set_title("Step 5: compact 4x training_pool scene on disk", fontsize=10)
    return _save(fig, output_dir, "06_compact_storage_schematic_4x.png")


def _plot_forward_consistency_flow(scene: dict[str, np.ndarray], maps: dict[str, Any], output_dir: Path) -> str:
    pred = scene["pred"]
    lr_pred = maps["lr_pred"]
    lr_obs = maps["lr_obs"]
    lr_cov = maps["lr_cov"]
    forward_error = maps["forward_error_lr"]
    
    fig, axes = plt.subplots(1, 5, figsize=(9.6, 2.5))
    _panel(axes[0], pred, "Pred 4x HR", cmap=COLORMAPS["temperature"])
    _panel(axes[1], lr_pred, "Pred LR (PSF+Pool4x)", cmap=COLORMAPS["temperature"])
    _panel(axes[2], lr_obs, "Obs LR (drizzle mean)", cmap=COLORMAPS["temperature"])
    _panel(axes[3], lr_cov, "LR coverage weight", cmap=COLORMAPS["coverage"], vmin=0)
    _panel(axes[4], forward_error, "Forward L1 error (LR)", cmap=COLORMAPS["residual_pos"])
    fig.suptitle("Loss 4 - Forward Consistency Flow (HR pred -> physical LR projection -> data fidelity)", fontsize=10, y=1.02)
    return _save(fig, output_dir, "11_forward_consistency_flow.png")


def _plot_heteroscedastic_nll_flow(scene: dict[str, np.ndarray], maps: dict[str, Any], output_dir: Path) -> str:
    pred = scene["pred"]
    target = scene["target"]
    sigma = maps["uncertainty_sigma"]
    nll_map = maps["nll_map"]
    
    fig, axes = plt.subplots(1, 4, figsize=(8.8, 2.5))
    vmin, vmax = float(target.min()), float(target.max())
    _panel(axes[0], np.abs(pred - target), "|pred - target|", cmap=COLORMAPS["residual_pos"])
    _panel(axes[1], sigma, "Predicted uncertainty sigma", cmap=COLORMAPS["residual_diff"])
    _panel(axes[2], nll_map, "Heteroscedastic NLL map", cmap=COLORMAPS["residual_pos"])
    _panel(axes[3], maps["cov_weight"], "Coverage weight", cmap=COLORMAPS["coverage"])
    fig.suptitle("Loss 5 - Heteroscedastic NLL Flow (confidence-aware learning)", fontsize=10, y=1.02)
    return _save(fig, output_dir, "12_heteroscedastic_nll_flow.png")


def _plot_hf_coverage_weighting(scene: dict[str, np.ndarray], maps: dict[str, Any], output_dir: Path) -> str:
    target_hf = maps["target_hf"]
    pred_hf = maps["pred_hf"]
    hf_error = np.abs(pred_hf - target_hf)
    cov_w = maps["cov_weight"]
    inv_w = maps["inv_weight"]
    
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    _panel(axes[0, 0], hf_error, "HF L1 error", cmap=COLORMAPS["residual_pos"])
    _panel(axes[0, 1], cov_w, "ch1 coverage weight", cmap=COLORMAPS["coverage"])
    _panel(axes[0, 2], hf_error * cov_w, "weighted HF loss (L1)", cmap=COLORMAPS["residual_pos"])
    
    _panel(axes[1, 0], hf_error, "HF L1 error", cmap=COLORMAPS["residual_pos"])
    _panel(axes[1, 1], inv_w, "inverse coverage weight", cmap=COLORMAPS["coverage"])
    _panel(axes[1, 2], hf_error * inv_w, "weighted HF detail loss", cmap=COLORMAPS["residual_pos"])
    
    fig.suptitle("Loss 2 (HF, coverage-weighted) vs Loss 6 (HF detail, inverse-coverage-weighted)", fontsize=10, y=1.02)
    return _save(fig, output_dir, "13_hf_coverage_weighting.png")


def _plot_edge_loss_4x(maps: dict[str, Any], output_dir: Path) -> str:
    edge_error = maps["edge_error"]
    edge_w = maps["edge_weight_map"]
    
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["double_col"])
    _panel(axes[0], edge_error, "Sobel Fine Error", cmap=COLORMAPS["residual_pos"])
    _panel(axes[1], edge_w, "Sobel Weight (edge mask boost)", cmap=COLORMAPS["coverage"])
    _panel(axes[2], edge_error * edge_w, "Weighted Edge Loss", cmap=COLORMAPS["residual_pos"])
    fig.suptitle("Loss 3 - Sobel Edge Loss (fine detail with boundary boost)", fontsize=10, y=1.02)
    return _save(fig, output_dir, "14_edge_loss_4x.png")


def build_loss_atlas_figures_4x(output_dir: Path, *, recipe: LossRecipe4x | None = None) -> dict[str, Any]:
    setup_academic_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    recipe = recipe or LossRecipe4x()
    bundle = load_training_demo_bundle(output_dir)
    meta = bundle["meta"]
    scene = make_loss_scene_from_bundle(bundle)
    breakdown = compute_loss_breakdown_4x(scene, recipe)
    maps = breakdown["maps"]
    pred, target = scene["pred"], scene["target"]
    figures: list[str] = []

    # 1. Pipeline schematics and training input overview
    figures.append(_plot_training_pipeline_schematic_4x(output_dir, meta))
    figures.append(_plot_hr_mask_temperature_4x(bundle, output_dir))
    figures.append(_plot_lr_burst_samples_4x(bundle, output_dir))
    figures.append(_plot_obs_channels_4x(bundle, output_dir))
    figures.append(_plot_compact_storage_4x(output_dir, meta))

    # 2. General Target vs Pred vs error
    vmin, vmax = float(min(target.min(), pred.min())), float(max(target.max(), pred.max()))
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["double_col"])
    _panel(axes[0], target, "Target temperature 4x", cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax, cbar_label="deg C")
    _panel(axes[1], pred, "Pred 4x (demo ringing/noise)", cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax, cbar_label="deg C")
    _panel(axes[2], np.abs(pred - target), "|pred-target| 4x", cmap=COLORMAPS["residual_pos"], cbar_label="deg C")
    fig.suptitle("Loss stage: 4x target and simulated prediction", fontsize=10, y=1.02)
    figures.append(_save(fig, output_dir, "08_temperature_pair_4x.png"))

    # 3. Loss 1 - Low Frequency Loss
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["double_col"])
    _panel(axes[0], maps["target_lf"], "LF(Target)", cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax)
    _panel(axes[1], maps["pred_lf"], "LF(Pred)", cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax)
    _panel(axes[2], np.abs(maps["pred_lf"] - maps["target_lf"]), "LF L1 Error", cmap=COLORMAPS["residual_pos"])
    fig.suptitle("Loss 1 - Low Frequency Loss (Gaussian blur sigma=8.0, L1)", fontsize=10, y=1.02)
    figures.append(_save(fig, output_dir, "09_lf_loss_4x.png"))

    # 4. Detailed Flows for Forward Consistency, Heteroscedastic NLL, and HF weighting
    figures.append(_plot_forward_consistency_flow(scene, maps, output_dir))
    figures.append(_plot_heteroscedastic_nll_flow(scene, maps, output_dir))
    figures.append(_plot_hf_coverage_weighting(scene, maps, output_dir))
    figures.append(_plot_edge_loss_4x(maps, output_dir))

    # 5. Bar plot summarizing recipe
    names = ["lf x1.0", "hf x0.3", "edge x0.1", "forward x0.2", "nll x0.05", "hf_detail x0.3"]
    raw = [breakdown["lf"], breakdown["hf"], breakdown["edge"], breakdown["forward"], breakdown["nll"], breakdown["hf_detail"]]
    weighted = [
        recipe.lf_weight * breakdown["lf"],
        recipe.hf_weight * breakdown["hf"],
        recipe.edge_weight * breakdown["edge"],
        recipe.forward_weight * breakdown["forward"],
        recipe.nll_weight * breakdown["nll"],
        recipe.hf_detail_weight * breakdown["hf_detail"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["double_col"])
    axes[0].bar(names, raw, color=METHOD_COLOR_LIST[: len(names)])
    axes[0].set_title("raw loss values")
    axes[0].tick_params(axis="x", rotation=25)
    axes[1].bar(names, weighted, color=METHOD_COLOR_LIST[: len(names)])
    axes[1].set_title(f"weighted contribution (total={breakdown['total']:.4f})")
    axes[1].tick_params(axis="x", rotation=25)
    fig.suptitle("EP12 4x SR balance recipe", fontsize=10, y=1.02)
    figures.append(_save(fig, output_dir, "16_total_loss_recipe_4x.png"))

    manifest = {
        "episode": "ep14_4x_loss_atlas",
        "data_source": "tcforge_4x",
        "rotation_deg": meta["rotation_deg"],
        "n_frames_demo": meta["n_frames_demo"],
        "n_frames_train_ref": meta["n_frames_train_ref"],
        "figures": figures,
        "loss_breakdown": {k: breakdown[k] for k in ("lf", "hf", "edge", "forward", "nll", "hf_detail", "total")},
        "recipe": breakdown["recipe"],
        "obs_channel_names": meta["obs_channel_names"],
    }
    (output_dir / "loss_breakdown.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
