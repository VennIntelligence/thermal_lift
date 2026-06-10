"""EP13 Loss Atlas — TCForge training-input + ContourSRLoss visual guide (Refactored to match current code)."""

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
class LossRecipe:
    mse_weight: float = 0.2
    highpass_weight: float = 1.0
    edge_weight: float = 0.05
    ssim_weight: float = 0.15
    highpass_sigma: float = 5.0
    structure_boost: float = 4.0
    edge_coarse_weight: float = 0.25


OBS_CHANNEL_LABELS = [
    "ch0 aligned mean",
    "ch1 aligned median",
    "ch2 coverage",
    "ch3 variance",
    "ch4 highpass fused",
]


def load_training_demo_bundle(output_dir: Path) -> dict[str, Any]:
    npz_path = output_dir / "training_demo_bundle.npz"
    meta_path = output_dir / "training_demo_meta.json"
    if not npz_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing TCForge demo bundle under {output_dir}. Run scripts/build_ep13_cache.py first."
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


def highpass(image: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    return image - gaussian_blur_2d(image, sigma)


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
    yy, xx = np.mgrid[0:target.shape[0], 0:target.shape[1]]
    edge = sobel_edges(mask)
    edge /= max(float(edge.max()), 1e-6)
    ringing = 0.28 * np.sin(xx * 1.1) * np.sin(yy * 1.1) * (edge > 0.12)
    pred = target + ringing
    return {
        "target": target.astype(np.float32),
        "pred": pred.astype(np.float32),
        "mask": (mask > 0.5).astype(np.float32),
        "crop_origin": np.array([y0, x0], dtype=np.int32),
    }


def _conv2d_same(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2
    work = np.pad(image.astype(np.float64), ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")
    out = np.zeros_like(image, dtype=np.float64)
    for y in range(out.shape[0]):
        for x in range(out.shape[1]):
            out[y, x] = float((work[y : y + kh, x : x + kw] * kernel).sum())
    return out


def _ssim_map(pred: np.ndarray, target: np.ndarray, window: int = 11, sigma: float = 1.5) -> tuple[float, np.ndarray]:
    g = _gaussian_kernel_1d(sigma, window)
    g2d = np.outer(g, g)
    g2d /= g2d.sum()
    mu_p = _conv2d_same(pred, g2d)
    mu_t = _conv2d_same(target, g2d)
    mu_p2 = _conv2d_same(pred * pred, g2d)
    mu_t2 = _conv2d_same(target * target, g2d)
    mu_pt = _conv2d_same(pred * target, g2d)
    sigma_p2 = np.clip(mu_p2 - mu_p * mu_p, 0.0, None)
    sigma_t2 = np.clip(mu_t2 - mu_t * mu_t, 0.0, None)
    sigma_pt = mu_pt - mu_p * mu_t
    data_range = float(target.max() - target.min()) + 1e-8
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    ssim_local = ((2 * mu_p * mu_t + c1) * (2 * sigma_pt + c2)) / ((mu_p * mu_p + mu_t * mu_t + c1) * (sigma_p2 + sigma_t2 + c2))
    return float(ssim_local.mean()), ssim_local.astype(np.float32)


def compute_loss_breakdown(scene: dict[str, np.ndarray], recipe: LossRecipe | None = None) -> dict[str, Any]:
    recipe = recipe or LossRecipe()
    pred, target = scene["pred"], scene["target"]

    mse_map = (pred - target) ** 2
    mse = float(mse_map.mean())

    pred_hp = highpass(pred, recipe.highpass_sigma)
    target_hp = highpass(target, recipe.highpass_sigma)
    hp_error = np.abs(pred_hp - target_hp)
    
    # Gradient-based structure weight (from target)
    target_edges = sobel_edges(target)
    edge_max = max(float(target_edges.max()), 1e-8)
    edge_norm = target_edges / edge_max
    weight_map = 1.0 + recipe.structure_boost * edge_norm
    hp_loss = float((hp_error * weight_map).mean())

    pred_edges = sobel_edges(pred)
    edge_fine = float(np.abs(pred_edges - target_edges).mean())
    pred_2x = pred.reshape(pred.shape[0] // 2, 2, pred.shape[1] // 2, 2).mean(axis=(1, 3))
    target_2x = target.reshape(target.shape[0] // 2, 2, target.shape[1] // 2, 2).mean(axis=(1, 3))
    edge = edge_fine + recipe.edge_coarse_weight * float(np.abs(sobel_edges(pred_2x) - sobel_edges(target_2x)).mean())

    ssim_val, ssim_local = _ssim_map(pred, target)
    ssim_loss = 1.0 - ssim_val
    
    total = (
        recipe.mse_weight * mse
        + recipe.highpass_weight * hp_loss
        + recipe.edge_weight * edge
        + recipe.ssim_weight * ssim_loss
    )
    return {
        "recipe": asdict(recipe),
        "mse": mse,
        "highpass": hp_loss,
        "edge": edge,
        "ssim": ssim_loss,
        "total": total,
        "maps": {
            "mse": mse_map,
            "pred_hp": pred_hp,
            "target_hp": target_hp,
            "hp_error": hp_error,
            "weight_map": weight_map,
            "pred_edges": pred_edges,
            "target_edges": target_edges,
            "edge_error": np.abs(pred_edges - target_edges),
            "ssim_local": ssim_local,
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


def _plot_training_pipeline_schematic(output_dir: Path, meta: dict[str, Any]) -> str:
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")
    n = int(meta.get("n_frames_per_scene", meta.get("n_frames_train_ref", 248)))
    boxes = [
        (0.2, 4.8, 2.0, 1.0, "TCForge\nHR mask + temp", "#E8EEF7"),
        (2.6, 4.8, 2.0, 1.0, f"Forward x{n}\nLR frames", "#FBE7C6"),
        (5.0, 4.8, 2.2, 1.0, "Align +\nFuse 5ch", "#D8EAD3"),
        (7.6, 4.8, 2.0, 1.0, "Compact\nscene on disk", "#EFE3F2"),
        (10.0, 4.8, 1.8, 1.0, "Patch\nsampler", "#E8EEF7"),
        (12.0, 4.8, 1.6, 1.0, "UNet", "#D8EAD3"),
        (12.0, 2.8, 1.6, 1.0, "pred HR", "#FBE7C6"),
        (9.8, 2.8, 2.0, 1.0, "GT temp\nfrom mask", "#FBE7C6"),
        (6.8, 1.0, 5.0, 1.0, "ContourSRLoss total", "#F3F3F3"),
    ]
    for x, y, w, h, label, color in boxes:
        ax.add_patch(Rectangle((x, y), w, h, facecolor=color, edgecolor="black", linewidth=0.8))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=7)
    arrows = [
        ((2.2, 5.3), (2.6, 5.3)),
        ((4.6, 5.3), (5.0, 5.3)),
        ((7.2, 5.3), (7.6, 5.3)),
        ((9.6, 5.3), (10.0, 5.3)),
        ((11.8, 5.3), (12.0, 5.3)),
        ((12.8, 4.8), (12.8, 3.8)),
        ((11.6, 3.3), (9.8, 3.3)),
        ((10.8, 2.8), (10.8, 2.0)),
    ]
    for start, end in arrows:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10, linewidth=0.9))
    ax.text(
        0.2,
        0.2,
        f"Demo shows {meta.get('n_frames_demo', 16)} LR frames; offline fusion uses {n} frames/scene "
        "(config: n_frames_per_scene).",
        fontsize=8,
    )
    ax.set_title("EP07 training data flow (offline generation -> dataloader -> loss)", fontsize=10)
    return _save(fig, output_dir, "00_training_pipeline_schematic.png")


def _plot_hr_mask_temperature(bundle: dict[str, Any], output_dir: Path) -> str:
    mask, temp = bundle["hr_mask"], bundle["hr_temperature"]
    meta = bundle["meta"]
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["double_col"])
    _panel(axes[0], mask, f"HR mask (theta={meta['rotation_deg']:.1f} deg)", cmap="gray", vmin=0, vmax=1)
    vmin, vmax = float(temp.min()), float(temp.max())
    _panel(axes[1], temp, "HR temperature (TCForge render)", cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax, cbar_label="deg C")
    fig.suptitle("Step 1-2: geometry + temperature target on detector grid", fontsize=10, y=1.02)
    return _save(fig, output_dir, "02_hr_mask_and_temperature.png")


def _plot_lr_burst_samples(bundle: dict[str, Any], output_dir: Path) -> str:
    burst = bundle["lr_burst"]
    n_show = min(6, burst.shape[0])
    idx = np.linspace(0, burst.shape[0] - 1, n_show, dtype=int)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    vmin, vmax = float(burst.min()), float(burst.max())
    for ax, frame_idx in zip(axes.ravel(), idx, strict=True):
        _panel(ax, burst[frame_idx], f"LR frame {frame_idx}", cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax)
    meta = bundle["meta"]
    total = int(meta.get("n_frames_per_scene", meta.get("n_frames_train_ref", 248)))
    physics = meta.get("physics_meta", {})
    fig.suptitle(
        f"Step 3: forward + noise + drift ({n_show} of {burst.shape[0]} demo / {total} fused)",
        fontsize=10,
        y=1.02,
    )
    return _save(fig, output_dir, "03_lr_burst_samples.png")


def _plot_fusion_schematic(bundle: dict[str, Any], output_dir: Path) -> str:
    burst = bundle["lr_burst"]
    shifts = bundle["shifts_demo"]
    frame_a, frame_b = burst[0], burst[len(burst) // 2]
    shift = shifts[len(shifts) // 2]
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["double_col"])
    vmin, vmax = float(burst.min()), float(burst.max())
    _panel(axes[0], frame_a, "frame A (native LR grid)", cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax)
    _panel(axes[1], frame_b, f"frame B shift=({shift[0]:+.2f},{shift[1]:+.2f}) px", cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax)
    _panel(axes[2], bundle["obs_features"][0], "aligned mean (reference grid)", cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax)
    fig.suptitle("Step 4: per-frame shifts align observations before fusion", fontsize=10, y=1.02)
    return _save(fig, output_dir, "04_alignment_fusion_schematic.png")


def _plot_obs_channels(bundle: dict[str, Any], output_dir: Path) -> str:
    obs = bundle["obs_features"]
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.8))
    for ax in axes[1, 2:]:
        ax.axis("off")
    cmaps = [COLORMAPS["temperature"], COLORMAPS["temperature"], COLORMAPS["coverage"], COLORMAPS["residual_pos"], COLORMAPS["residual_diff"]]
    for ax, ch, label, cmap in zip(axes.ravel()[:5], range(5), OBS_CHANNEL_LABELS, cmaps, strict=True):
        img = obs[ch]
        vmin, vmax = (0, 1) if ch == 2 else (float(img.min()), float(img.max()))
        _panel(ax, img, label, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.suptitle("Step 5: fuse_burst_to_features -> 5-channel obs_features (1x LR)", fontsize=10, y=1.02)
    return _save(fig, output_dir, "05_obs_feature_channels.png")


def _plot_compact_storage(output_dir: Path, meta: dict[str, Any]) -> str:
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["double_col"])
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    ax.axis("off")
    files = [
        (0.5, 3.8, "hr_mask_2x.png"),
        (0.5, 2.6, "hr_edge_2x.png"),
        (0.5, 1.4, "obs_features_1x.npz"),
        (0.5, 0.2, "metadata.json + shifts.npy"),
    ]
    for x, y, label in files:
        ax.add_patch(Rectangle((x, y), 4.0, 0.9, facecolor="#E8EEF7", edgecolor="black", linewidth=0.8))
        ax.text(x + 0.15, y + 0.45, label, fontsize=8, va="center")
    ax.add_patch(Rectangle((5.2, 1.0), 4.2, 3.5, fill=False, edgecolor="#C44E52", linewidth=1.2, linestyle="--"))
    ax.text(5.4, 4.2, "NOT stored in compact pool:", fontsize=8, color="#C44E52")
    n_frames = int(meta.get("n_frames_per_scene", meta.get("n_frames_train_ref", 248)))
    ax.text(5.4, 3.5, f"- full LR burst ({n_frames} frames)", fontsize=8)
    ax.text(5.4, 2.8, "- HR temperature .npy", fontsize=8)
    ax.text(5.4, 2.1, "GT temp rebuilt at train time", fontsize=8)
    ax.text(5.4, 1.4, "via reconstruct_hr_temperature()", fontsize=8)
    ax.set_title("Step 6: compact training_pool scene on disk", fontsize=10)
    return _save(fig, output_dir, "06_compact_storage_schematic.png")


def _plot_patch_and_unet(bundle: dict[str, Any], scene: dict[str, np.ndarray], output_dir: Path) -> str:
    obs = bundle["obs_features"]
    scale = int(bundle["meta"]["scale"])
    ph, pw = scene["target"].shape
    obs_patch, _ = crop_center(obs, max(ph // scale, 32))
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["double_col"])
    _panel(axes[0], obs[0], "obs ch0 (full scene)", cmap=COLORMAPS["temperature"])
    _panel(axes[1], obs_patch[0], f"LR patch fed to UNet ({obs_patch.shape[1]}x{obs_patch.shape[2]})", cmap=COLORMAPS["temperature"])
    _panel(axes[2], scene["target"], f"HR target patch ({ph}x{pw})", cmap=COLORMAPS["temperature"])
    fig.suptitle(f"Step 7-8: dataloader crop + UNet maps {obs.shape[0]}ch@1x -> 1ch@{scale}x HR", fontsize=10, y=1.02)
    return _save(fig, output_dir, "07_patch_and_unet_io.png")


def build_loss_atlas_figures(output_dir: Path, *, recipe: LossRecipe | None = None) -> dict[str, Any]:
    setup_academic_style()
    output_dir.mkdir(parents=True, exist_ok=True)
    recipe = recipe or LossRecipe()
    bundle = load_training_demo_bundle(output_dir)
    meta = bundle["meta"]
    scene = make_loss_scene_from_bundle(bundle)
    breakdown = compute_loss_breakdown(scene, recipe)
    maps = breakdown["maps"]
    pred, target = scene["pred"], scene["target"]
    figures: list[str] = []

    figures.append(_plot_training_pipeline_schematic(output_dir, meta))
    figures.append(_plot_hr_mask_temperature(bundle, output_dir))
    figures.append(_plot_lr_burst_samples(bundle, output_dir))
    figures.append(_plot_fusion_schematic(bundle, output_dir))
    figures.append(_plot_obs_channels(bundle, output_dir))
    figures.append(_plot_compact_storage(output_dir, meta))
    figures.append(_plot_patch_and_unet(bundle, scene, output_dir))

    vmin, vmax = float(min(target.min(), pred.min())), float(max(target.max(), pred.max()))
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["double_col"])
    _panel(axes[0], target, "Target temperature", cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax, cbar_label="deg C")
    _panel(axes[1], pred, "Pred (demo ringing)", cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax, cbar_label="deg C")
    _panel(axes[2], np.abs(pred - target), "|pred-target|", cmap=COLORMAPS["residual_pos"], cbar_label="deg C")
    fig.suptitle("Loss stage: TCForge center patch", fontsize=10, y=1.02)
    figures.append(_save(fig, output_dir, "08_temperature_pair.png"))

    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["double_col"])
    _panel(axes[0], target, "target", cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax)
    _panel(axes[1], pred, "pred", cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax)
    _panel(axes[2], maps["mse"], "MSE map", cmap=COLORMAPS["residual_pos"])
    fig.suptitle(r"Loss 1 - MSE, weight=0.2", fontsize=10, y=1.02)
    figures.append(_save(fig, output_dir, "09_mse_loss.png"))

    row = pred.shape[0] // 2
    blur = gaussian_blur_2d(target, recipe.highpass_sigma)
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["double_col"])
    x = np.arange(target.shape[1])
    ax.plot(x, target[row], label="original", color="#4C72B0", linewidth=1.2)
    ax.plot(x, blur[row], label=f"Gaussian blur sigma={recipe.highpass_sigma:g}", color="#55A868", linewidth=1.2)
    ax.plot(x, highpass(target, recipe.highpass_sigma)[row], label="highpass", color="#C44E52", linewidth=1.2)
    ax.set_xlabel("Pixel x")
    ax.set_ylabel("Temperature [deg C]")
    ax.set_title(f"Highpass construction (row y={row})")
    ax.legend(fontsize=8, frameon=False)
    figures.append(_save(fig, output_dir, "10_highpass_1d_profile.png"))

    hp_vmax = max(float(np.abs(maps["pred_hp"]).max()), float(np.abs(maps["target_hp"]).max()), 1e-6)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 5.0))
    _panel(axes[0, 0], target, "target temp", cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax)
    _panel(axes[0, 1], gaussian_blur_2d(target, recipe.highpass_sigma), "blur(target)", cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax)
    _panel(axes[0, 2], maps["target_hp"], "target_hp", cmap=COLORMAPS["residual_diff"], vmin=-hp_vmax, vmax=hp_vmax)
    _panel(axes[1, 0], pred, "pred temp", cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax)
    _panel(axes[1, 1], gaussian_blur_2d(pred, recipe.highpass_sigma), "blur(pred)", cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax)
    _panel(axes[1, 2], maps["pred_hp"], "pred_hp", cmap=COLORMAPS["residual_diff"], vmin=-hp_vmax, vmax=hp_vmax)
    fig.suptitle(r"Loss 2 - Highpass", fontsize=10, y=1.02)
    figures.append(_save(fig, output_dir, "11_highpass_maps.png"))

    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["double_col"])
    _panel(axes[0], maps["hp_error"], "|hp_pred-hp_target|", cmap=COLORMAPS["residual_pos"])
    _panel(axes[1], maps["weight_map"], "gradient structure weight map", cmap=COLORMAPS["coverage"])
    _panel(axes[2], maps["hp_error"] * maps["weight_map"], "weighted hp error", cmap=COLORMAPS["residual_pos"])
    fig.suptitle("Highpass weighted by continuous gradient structure map", fontsize=10, y=1.02)
    figures.append(_save(fig, output_dir, "12_highpass_weighted_error.png"))

    edge_vmax = max(float(maps["pred_edges"].max()), float(maps["target_edges"].max()), 1e-6)
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["double_col"])
    _panel(axes[0], maps["pred_edges"], "Sobel(pred)", cmap=COLORMAPS["coverage"], vmin=0, vmax=edge_vmax)
    _panel(axes[1], maps["target_edges"], "Sobel(target)", cmap=COLORMAPS["coverage"], vmin=0, vmax=edge_vmax)
    _panel(axes[2], maps["edge_error"], "edge error", cmap=COLORMAPS["residual_pos"])
    fig.suptitle("Loss 3 - Edge (Sobel fine + coarse)", fontsize=10, y=1.02)
    figures.append(_save(fig, output_dir, "13_edge_loss.png"))

    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["double_col"])
    _panel(axes[0], target, "target", cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax)
    _panel(axes[1], pred, "pred", cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax)
    _panel(axes[2], maps["ssim_local"], "local SSIM", cmap=COLORMAPS["coverage"], vmin=0, vmax=1)
    fig.suptitle("Loss 4 - SSIM", fontsize=10, y=1.02)
    figures.append(_save(fig, output_dir, "14_ssim_loss.png"))

    names = ["mse x0.2", "hp x1.0", "edge x0.05", "ssim x0.15"]
    raw = [breakdown["mse"], breakdown["highpass"], breakdown["edge"], breakdown["ssim"]]
    weighted = [
        recipe.mse_weight * breakdown["mse"],
        recipe.highpass_weight * breakdown["highpass"],
        recipe.edge_weight * breakdown["edge"],
        recipe.ssim_weight * breakdown["ssim"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["double_col"])
    axes[0].bar(names, raw, color=METHOD_COLOR_LIST[: len(names)])
    axes[0].set_title("raw loss values")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].bar(names, weighted, color=METHOD_COLOR_LIST[: len(names)])
    axes[1].set_title(f"weighted contribution (total={breakdown['total']:.4f})")
    axes[1].tick_params(axis="x", rotation=20)
    fig.suptitle("EP07 ContourSRLoss recipe", fontsize=10, y=1.02)
    figures.append(_save(fig, output_dir, "16_total_loss_recipe.png"))

    manifest = {
        "episode": "ep13_loss_atlas",
        "data_source": "tcforge",
        "rotation_deg": meta["rotation_deg"],
        "n_frames_demo": meta["n_frames_demo"],
        "n_frames_train_ref": meta["n_frames_train_ref"],
        "figures": figures,
        "loss_breakdown": {k: breakdown[k] for k in ("mse", "highpass", "edge", "ssim", "total")},
        "recipe": breakdown["recipe"],
        "obs_channel_names": meta["obs_channel_names"],
    }
    (output_dir / "loss_breakdown.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
