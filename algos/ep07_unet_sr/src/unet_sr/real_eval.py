"""Real-data EP11-style evaluation for TensorBoard during EP07 training."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch
from matplotlib import colormaps
from scipy import ndimage
from scipy.ndimage import zoom as scipy_zoom
from scipy.ndimage import gaussian_filter, laplace
from tcforge.classical_sr import phase_bin_drizzle
from tcforge.fusion import fuse_burst_to_features
from tcforge.highpass import highpass_preprocess
from torch.utils.tensorboard import SummaryWriter

from .dataset import HYBRID_DRIZZLE_MEAN_CHANNEL
from .forward_torch import ScenePSF
from .inference import infer_from_burst
from .inference import _positions, _window_2d
from .metrics import out_of_band_ratio


@dataclass(frozen=True)
class RealEvalConfig:
    enabled: bool = True
    every: int = 0
    frame_limit: int = 248
    alignment_method: str = "contour_refined"
    baseline_hr: str = ""
    center_fraction: float = 1.0 / 3.0
    zoom: float = 3.0
    overlap: int = 128
    highpass_sigma: float = 5.0
    workers: int = 2
    output_dir: str = ""
    save_png: bool = True


def _project_root() -> Path:
    return Path(__file__).resolve().parents[4]


def _default_baseline_hr() -> Path:
    return _project_root() / "output" / "ep10_tgv_sr" / "best_hr_highpass.npy"


@lru_cache(maxsize=1)
def _import_ep06_common() -> tuple[Any, Any, Any]:
    ep06_src = _project_root() / "algos" / "ep06_sr_poc" / "src"
    ep06_src_str = str(ep06_src)
    if ep06_src_str not in sys.path:
        sys.path.insert(0, ep06_src_str)
    from common.alignment import load_alignment_shifts  # noqa: PLC0415
    from common.data_loader import bicubic_upsample, load_main_session_frames  # noqa: PLC0415

    return load_main_session_frames, load_alignment_shifts, bicubic_upsample


def center_fraction_crop(image: np.ndarray, fraction: float = 1.0 / 3.0) -> np.ndarray:
    rows, cols = image.shape
    crop_rows = max(1, int(round(rows * fraction)))
    crop_cols = max(1, int(round(cols * fraction)))
    y0 = max(0, rows // 2 - crop_rows // 2)
    x0 = max(0, cols // 2 - crop_cols // 2)
    return image[y0 : y0 + crop_rows, x0 : x0 + crop_cols]


def zoom_center(image: np.ndarray, *, center_fraction: float, zoom: float) -> np.ndarray:
    crop = center_fraction_crop(np.asarray(image), fraction=center_fraction)
    return ndimage.zoom(crop, zoom=float(zoom), order=1).astype(np.float32, copy=False)


def _temperature_limits(image: np.ndarray) -> tuple[float, float]:
    values = image[np.isfinite(image)].ravel()
    if values.size == 0:
        return 0.0, 1.0
    return float(np.percentile(values, 1.0)), float(np.percentile(values, 99.0))


def _temperature_rgb(
    image: np.ndarray,
    *,
    cmap_name: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
) -> np.ndarray:
    """EP11-style temperature panel for TensorBoard (inferno, 1–99 percentile).

    When *vmin*/*vmax* are supplied (e.g. from a full-frame image), use them
    instead of computing percentiles from *image* itself.  This keeps the
    background purple (matching real-data views) even after center-ROI cropping.
    """
    if vmin is None or vmax is None:
        vmin, vmax = _temperature_limits(image)
    if vmax - vmin < 1e-6:
        norm = np.zeros_like(image, dtype=np.float32)
    else:
        norm = np.clip((image - vmin) / (vmax - vmin), 0.0, 1.0).astype(np.float32, copy=False)
    rgba = colormaps.get_cmap(cmap_name)(norm)
    return rgba[..., :3].transpose(2, 0, 1).astype(np.float32, copy=False)


def save_ep11_temperature_figure(
    unet_temp: np.ndarray,
    output_path: Path,
    *,
    zoom: float,
    center_fraction: float,
    step: int,
    scale: int,
    method_label: str = "UNet",
    vmin: float | None = None,
    vmax: float | None = None,
) -> Path:
    """Save the same center-zoom temperature sanity figure as EP11 benchmark.

    When *vmin*/*vmax* are provided (from full-frame percentiles), the colormap
    range stays consistent with the full image so the background renders as
    purple instead of black.
    """

    import matplotlib.pyplot as plt

    image = zoom_center(unet_temp, center_fraction=center_fraction, zoom=zoom)
    if vmin is None or vmax is None:
        vmin, vmax = _temperature_limits(image)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(4.1, 3.0), squeeze=True)
    im = ax.imshow(image, cmap="inferno", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(f"{method_label} {scale}x @ EP07 step {step} (temperature)")
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Temperature [deg C]")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def artifact_score(sr_img: np.ndarray, *, scale: int = 2) -> float:
    arr = np.asarray(sr_img, dtype=np.float32)
    if arr.ndim != 2 or not np.isfinite(arr).any():
        return float("inf")
    high_freq = arr - gaussian_filter(arr, sigma=1.0, mode="nearest")
    lap = laplace(arr, mode="nearest")
    base = float(np.nanstd(arr))
    if base <= 1e-12:
        return 0.0
    return float((np.nanstd(high_freq) + 0.25 * np.nanstd(lap)) / base)


def pearson_finite(a: np.ndarray, b: np.ndarray) -> float:
    lhs = np.asarray(a, dtype=np.float64)
    rhs = np.asarray(b, dtype=np.float64)
    valid = np.isfinite(lhs) & np.isfinite(rhs)
    if int(valid.sum()) < 2:
        return float("nan")
    return float(np.corrcoef(lhs[valid].ravel(), rhs[valid].ravel())[0, 1])


def _shared_vmax(crops: list[np.ndarray]) -> float:
    finite = [crop[np.isfinite(crop)].ravel() for crop in crops if np.isfinite(crop).any()]
    if not finite:
        return 1.0
    vmax = float(np.percentile(np.abs(np.concatenate(finite)), 99.0))
    return max(vmax, 1e-6)


def _diverging_rgb(image: np.ndarray, vmax: float) -> np.ndarray:
    x = np.asarray(image, dtype=np.float32)
    t = np.clip(0.5 + 0.5 * x / max(vmax, 1e-6), 0.0, 1.0)
    red = np.clip(2.0 * t - 0.5, 0.0, 1.0)
    blue = np.clip(1.5 - 2.0 * t, 0.0, 1.0)
    green = 1.0 - np.abs(2.0 * t - 1.0)
    return np.stack([red, green, blue], axis=0)


def _stack_horizontal(*panels: np.ndarray) -> np.ndarray:
    return np.concatenate(list(panels), axis=-1)


def _select_solver_eval_frames(
    lr_burst: np.ndarray,
    shifts: np.ndarray,
    m_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Deterministically subsample a real burst for solver real-eval DC steps."""

    n_frames = int(lr_burst.shape[0])
    if n_frames <= 0:
        raise ValueError("lr_burst must contain at least one frame")
    m = min(max(1, int(m_frames)), n_frames)
    if m == n_frames:
        indices = np.arange(n_frames)
    else:
        indices = np.unique(np.linspace(0, n_frames - 1, m, dtype=np.int64))
        if indices.size < m:
            indices = np.arange(m, dtype=np.int64)
    return (
        np.asarray(lr_burst[indices], dtype=np.float32),
        np.asarray(shifts[indices], dtype=np.float32),
    )


def _solver_real_psf(training_config: Any, batch_size: int, device: torch.device) -> ScenePSF:
    sigma = float(getattr(training_config, "forward_model_psf_sigma", 0.5))
    return ScenePSF(
        sigma_lr_px=torch.full((batch_size,), sigma, dtype=torch.float32, device=device),
        shape=["gaussian"] * batch_size,
        sigma_y_lr_px=[sigma] * batch_size,
        angle_deg=torch.zeros(batch_size, dtype=torch.float32, device=device),
    )


def _lr_edge_mask(h: int, w: int, rim: int, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(1, 1, h, w, dtype=torch.float32, device=device)
    if rim > 0:
        mask[..., rim:-rim, rim:-rim] = 1.0
    else:
        mask[...] = 1.0
    return mask


@torch.no_grad()
def infer_solver_from_burst(
    solver: torch.nn.Module,
    lr_burst: np.ndarray,
    shifts: np.ndarray,
    *,
    training_config: Any,
    patch_size_hr: int,
    overlap: int,
    device: torch.device | str,
) -> np.ndarray:
    """Run tiled real-data inference for the unrolled solver.

    Plain UNet/V10 inference consumes only fused observation features.  The
    solver also needs a burst, shifts, a PSF spec, and the same LR rim mask used
    during training, so it gets a dedicated adapter.
    """

    scale = int(training_config.scale)
    if scale <= 0:
        raise ValueError("scale must be positive")
    if patch_size_hr <= 0 or patch_size_hr % scale != 0:
        raise ValueError("patch_size_hr must be positive and divisible by scale")
    if overlap < 0 or overlap >= patch_size_hr:
        raise ValueError("overlap must satisfy 0 <= overlap < patch_size_hr")

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")

    burst = np.asarray(lr_burst, dtype=np.float32)
    shift_arr = np.asarray(shifts, dtype=np.float32)
    if burst.ndim != 3:
        raise ValueError(f"lr_burst must be (N,H,W), got {burst.shape}")
    if shift_arr.shape != (burst.shape[0], 2):
        raise ValueError(f"shifts must be ({burst.shape[0]}, 2), got {shift_arr.shape}")

    features_1x = fuse_burst_to_features(
        burst,
        shift_arr,
        sigma_bg=float(getattr(training_config, "highpass_sigma", 5.0)),
    )
    features_up = scipy_zoom(features_1x, (1, scale, scale), order=1).astype(np.float32)
    if bool(getattr(training_config, "solver_no_drizzle", False)):
        cond = features_up
        mean_ch = 0
    else:
        n_bins = int(getattr(training_config, "phase_bin_channels", 4))
        drz = phase_bin_drizzle(burst, shift_arr, scale=scale, n_bins=n_bins)
        cond = np.concatenate([features_up, drz], axis=0).astype(np.float32, copy=False)
        mean_ch = HYBRID_DRIZZLE_MEAN_CHANNEL

    burst_sub, shifts_sub = _select_solver_eval_frames(
        burst,
        shift_arr,
        int(getattr(training_config, "solver_m_frames", 16)),
    )
    _, h_lr, w_lr = burst.shape
    patch_lr = patch_size_hr // scale
    overlap_lr = int(overlap // scale)
    step_lr = max(1, patch_lr - overlap_lr)
    ys = _positions(h_lr, patch_lr, step_lr)
    xs = _positions(w_lr, patch_lr, step_lr)
    out_hr = np.zeros((h_lr * scale, w_lr * scale), dtype=np.float32)
    weight_hr = np.zeros_like(out_hr)

    solver = solver.to(requested_device)
    was_training = solver.training
    solver.eval()
    psf = _solver_real_psf(training_config, 1, requested_device)
    frame_mask = _lr_edge_mask(
        patch_lr,
        patch_lr,
        int(getattr(training_config, "solver_dc_rim_lr_px", 0)),
        requested_device,
    )

    for y_lr in ys:
        for x_lr in xs:
            y_hr = y_lr * scale
            x_hr = x_lr * scale
            cond_patch = cond[:, y_hr : y_hr + patch_size_hr, x_hr : x_hr + patch_size_hr]
            if cond_patch.shape[1:] != (patch_size_hr, patch_size_hr):
                raise RuntimeError(f"condition patch shape mismatch: {cond_patch.shape}")
            burst_patch = burst_sub[:, y_lr : y_lr + patch_lr, x_lr : x_lr + patch_lr]
            if burst_patch.shape != (burst_sub.shape[0], patch_lr, patch_lr):
                raise RuntimeError(f"burst patch shape mismatch: {burst_patch.shape}")
            obs_t = torch.from_numpy(cond_patch[None]).to(requested_device)
            x0 = obs_t[:, mean_ch : mean_ch + 1]
            pred_t = solver(
                x0,
                torch.from_numpy(burst_patch[None]).to(requested_device),
                torch.from_numpy(shifts_sub[None]).to(requested_device),
                psf,
                obs_t,
                frame_mask=frame_mask,
            )
            pred = pred_t.detach().float().cpu().numpy()[0, 0]
            window = _window_2d(*pred.shape)
            out_hr[y_hr : y_hr + pred.shape[0], x_hr : x_hr + pred.shape[1]] += pred * window
            weight_hr[y_hr : y_hr + pred.shape[0], x_hr : x_hr + pred.shape[1]] += window

    if was_training:
        solver.train()
    return out_hr / np.maximum(weight_hr, 1e-6)


@lru_cache(maxsize=1)
def _load_real_eval_cache(
    frame_limit: int,
    alignment_method: str,
) -> tuple[np.ndarray, np.ndarray]:
    load_main_session_frames, load_alignment_shifts, _ = _import_ep06_common()
    raw_frames, metadata = load_main_session_frames(
        workers=2,
        dtype=np.float32,
        limit=frame_limit if frame_limit > 0 else None,
    )
    full_metadata = load_main_session_frames(workers=2, dtype=np.float32, limit=None)[1]
    full_shifts = load_alignment_shifts(alignment_method, metadata=full_metadata).astype(np.float32, copy=False)
    shifts = full_shifts[: len(metadata)]
    return raw_frames, shifts


def maybe_log_real_eval(
    writer: SummaryWriter | None,
    *,
    model: torch.nn.Module,
    config: RealEvalConfig,
    training_config: Any,
    step: int,
    device: torch.device,
) -> dict[str, float] | None:
    """Run EP11-style real-data inference and log center-zoom3x panels to TensorBoard."""

    if writer is None or not config.enabled:
        return None
    every = config.every if config.every > 0 else int(training_config.save_every)
    if step % every != 0:
        return None

    raw_frames, shifts = _load_real_eval_cache(config.frame_limit, config.alignment_method)
    scale = int(training_config.scale)
    patch_size_hr = int(training_config.patch_size_hr)
    residual = bool(training_config.residual)
    residual_mode = str(getattr(training_config, "residual_mode", "none"))
    residual_channel = HYBRID_DRIZZLE_MEAN_CHANNEL if residual_mode == "drizzle2x" else None
    input_mode = str(getattr(training_config, "input_mode", "lr"))
    sigma_bg = float(config.highpass_sigma or training_config.highpass_sigma)

    model_was_training = model.training
    model.eval()
    with torch.no_grad():
        unet_temp = infer_from_burst(
            model,
            raw_frames,
            shifts,
            scale=scale,
            patch_size_hr=patch_size_hr,
            overlap=int(config.overlap),
            device=str(device),
            residual=residual,
            residual_channel=residual_channel,
            sigma_bg=sigma_bg,
            input_mode=input_mode,
        ).astype(np.float32, copy=False)
    if model_was_training:
        model.train()

    unet_hp = highpass_preprocess(unet_temp, sigma_bg=sigma_bg)

    baseline_path = Path(config.baseline_hr) if config.baseline_hr else _default_baseline_hr()
    baseline_hp = np.load(baseline_path).astype(np.float32, copy=False) if baseline_path.exists() else None

    # Compute vmin/vmax from full-frame BEFORE cropping, so the background
    # maps to inferno-purple (matching real-data views) instead of black.
    temp_vmin, temp_vmax = _temperature_limits(unet_temp)

    unet_hp_zoom = zoom_center(unet_hp, center_fraction=config.center_fraction, zoom=config.zoom)
    unet_temp_zoom = zoom_center(unet_temp, center_fraction=config.center_fraction, zoom=config.zoom)
    panels = [unet_hp_zoom]
    if baseline_hp is not None:
        panels.append(zoom_center(baseline_hp, center_fraction=config.center_fraction, zoom=config.zoom))
    vmax = _shared_vmax(panels)
    highpass_rgb = _diverging_rgb(_stack_horizontal(*panels), vmax)
    temp_rgb = _temperature_rgb(unet_temp_zoom, vmin=temp_vmin, vmax=temp_vmax)

    writer.add_image("eval_real/highpass_center_zoom", highpass_rgb, step)
    writer.add_image("eval_real/temperature_center_zoom", temp_rgb, step)
    # out_of_band_ratio (GT-free, PSF-free) is the headline artifact/hallucination
    # monitor on real data — it replaces raw_control_corr, which correlated the
    # clean output against a bicubic blur and so rewarded *not* restoring.
    oob = out_of_band_ratio(unet_temp, scale=scale)
    writer.add_scalar("eval_real/out_of_band_ratio", oob, step)
    # artifact_score retained only as a secondary FM-1 cliff monitor: a relative
    # jump across checkpoints flags beading onset (it cannot tell SR from artifact).
    writer.add_scalar("eval_real/artifact_score", artifact_score(unet_hp, scale=scale), step)
    writer.add_scalar("eval_real/frame_limit", float(config.frame_limit), step)
    writer.flush()

    if config.save_png and config.output_dir:
        eval_dir = Path(config.output_dir) / "eval_real"
        temp_png = save_ep11_temperature_figure(
            unet_temp,
            eval_dir / f"unet_step{step}_center_zoom{int(config.zoom)}x_temperature.png",
            zoom=config.zoom,
            center_fraction=config.center_fraction,
            step=step,
            scale=scale,
            method_label="UNet",
            vmin=temp_vmin,
            vmax=temp_vmax,
        )
        print(f"Saved EP11-style temperature figure: {temp_png}")

    return {
        "out_of_band_ratio": oob,
        "artifact_score": artifact_score(unet_hp, scale=scale),
    }


def maybe_log_solver_real_eval(
    writer: SummaryWriter | None,
    *,
    solver: torch.nn.Module,
    config: RealEvalConfig,
    training_config: Any,
    step: int,
    device: torch.device,
) -> dict[str, float] | None:
    """Run EP11-style real-data inference/logging for the unrolled solver."""

    if writer is None or not config.enabled:
        return None
    every = config.every if config.every > 0 else int(training_config.save_every)
    if every <= 0 or step % every != 0:
        return None

    raw_frames, shifts = _load_real_eval_cache(config.frame_limit, config.alignment_method)
    scale = int(training_config.scale)
    solver_temp = infer_solver_from_burst(
        solver,
        raw_frames,
        shifts,
        training_config=training_config,
        patch_size_hr=int(training_config.patch_size_hr),
        overlap=int(config.overlap),
        device=device,
    ).astype(np.float32, copy=False)
    solver_hp = highpass_preprocess(
        solver_temp,
        sigma_bg=float(config.highpass_sigma or training_config.highpass_sigma),
    )

    baseline_path = Path(config.baseline_hr) if config.baseline_hr else _default_baseline_hr()
    baseline_hp = np.load(baseline_path).astype(np.float32, copy=False) if baseline_path.exists() else None

    temp_vmin, temp_vmax = _temperature_limits(solver_temp)
    solver_hp_zoom = zoom_center(solver_hp, center_fraction=config.center_fraction, zoom=config.zoom)
    solver_temp_zoom = zoom_center(solver_temp, center_fraction=config.center_fraction, zoom=config.zoom)
    panels = [solver_hp_zoom]
    if baseline_hp is not None:
        panels.append(zoom_center(baseline_hp, center_fraction=config.center_fraction, zoom=config.zoom))
    vmax = _shared_vmax(panels)
    highpass_rgb = _diverging_rgb(_stack_horizontal(*panels), vmax)
    temp_rgb = _temperature_rgb(solver_temp_zoom, vmin=temp_vmin, vmax=temp_vmax)

    writer.add_image("eval_real/highpass_center_zoom", highpass_rgb, step)
    writer.add_image("eval_real/temperature_center_zoom", temp_rgb, step)
    oob = out_of_band_ratio(solver_temp, scale=scale)
    score = artifact_score(solver_hp, scale=scale)
    writer.add_scalar("eval_real/out_of_band_ratio", oob, step)
    writer.add_scalar("eval_real/artifact_score", score, step)
    writer.add_scalar("eval_real/frame_limit", float(config.frame_limit), step)
    writer.flush()

    if config.save_png and config.output_dir:
        eval_dir = Path(config.output_dir) / "eval_real"
        temp_png = save_ep11_temperature_figure(
            solver_temp,
            eval_dir / f"solver_step{step}_center_zoom{int(config.zoom)}x_temperature.png",
            zoom=config.zoom,
            center_fraction=config.center_fraction,
            step=step,
            scale=scale,
            method_label="Solver",
            vmin=temp_vmin,
            vmax=temp_vmax,
        )
        print(f"Saved solver EP11-style temperature figure: {temp_png}")

    return {
        "out_of_band_ratio": oob,
        "artifact_score": score,
    }


def clear_real_eval_cache() -> None:
    _load_real_eval_cache.cache_clear()
