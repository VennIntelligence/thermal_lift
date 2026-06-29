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
    tile_batch: int = 16
    solver_mode: str = "tiled"
    solver_halo_hr: int = 0
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


def _select_holdout_eval_frames(
    lr_burst: np.ndarray,
    shifts: np.ndarray,
    m_dc: int,
    m_eval: int = 16,
) -> tuple[np.ndarray, np.ndarray]:
    """Frames NOT used in the solver's DC subset, for an honest real data-consistency check.

    The solver's DC step consumes the `m_dc` frames picked by `_select_solver_eval_frames`; if we
    measured ||A x - y|| on those same frames the residual would be partly self-fit. We instead
    sample up to `m_eval` frames from the complement (falling back to the full burst when there are
    too few frames to hold any out)."""

    n_frames = int(lr_burst.shape[0])
    m_dc = min(max(1, int(m_dc)), n_frames)
    if m_dc >= n_frames:
        dc_idx: set[int] = set(range(n_frames))
    else:
        dc_idx = set(np.unique(np.linspace(0, n_frames - 1, m_dc, dtype=np.int64)).tolist())
    complement = np.array([i for i in range(n_frames) if i not in dc_idx], dtype=np.int64)
    if complement.size == 0:
        complement = np.arange(n_frames, dtype=np.int64)
    m = min(max(1, int(m_eval)), complement.size)
    sel = complement[np.unique(np.linspace(0, complement.size - 1, m, dtype=np.int64))]
    return (
        np.asarray(lr_burst[sel], dtype=np.float32),
        np.asarray(shifts[sel], dtype=np.float32),
    )


@torch.no_grad()
def _solver_real_dc_residual(
    solver: torch.nn.Module,
    solver_temp: np.ndarray,
    raw_frames: np.ndarray,
    shifts: np.ndarray,
    *,
    training_config: Any,
    device: torch.device,
) -> tuple[float, float]:
    """RMS of (A x - y) on HELD-OUT real frames — the only GT-free, physics-grounded real metric.

    Runs the full-frame reconstruction `solver_temp` back through the certified forward operator
    against frames the DC step never saw. Returns (highpass-band RMS, full-band RMS). The real PSF
    is the misspecified single Gaussian (sigma=forward_model_psf_sigma); treat the number as a
    RELATIVE comparison across configs/checkpoints, not an absolute physics certificate (ACL-032)."""

    scale = int(training_config.scale)
    y, sh = _select_holdout_eval_frames(
        raw_frames, shifts, int(getattr(training_config, "solver_m_frames", 16))
    )
    x = torch.from_numpy(np.ascontiguousarray(solver_temp[None, None])).to(device)
    y_t = torch.from_numpy(y[None]).to(device)
    sh_t = torch.from_numpy(sh[None]).to(device)
    psf = _solver_real_psf(training_config, 1, device)
    band_rms = float(solver.dc_residual_rms(x, y_t, sh_t, psf, band=True))
    full_rms = float(solver.dc_residual_rms(x, y_t, sh_t, psf, band=False))
    return band_rms, full_rms


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


def _solver_conditioning_from_burst(
    lr_burst: np.ndarray,
    shifts: np.ndarray,
    *,
    training_config: Any,
    scale: int,
) -> tuple[np.ndarray, int]:
    """Build solver conditioning channels and return the warm-start channel index."""

    features_1x = fuse_burst_to_features(
        lr_burst,
        shifts,
        sigma_bg=float(getattr(training_config, "highpass_sigma", 5.0)),
    )
    features_up = scipy_zoom(features_1x, (1, scale, scale), order=1).astype(np.float32)
    if bool(getattr(training_config, "solver_no_drizzle", False)):
        return features_up, 0

    n_bins = int(getattr(training_config, "phase_bin_channels", 4))
    drz = phase_bin_drizzle(lr_burst, shifts, scale=scale, n_bins=n_bins)
    cond = np.concatenate([features_up, drz], axis=0).astype(np.float32, copy=False)
    # Warm-start source must match training (ACL-032): aligned_mean (ch0, de-waffled) vs the
    # first phase-bin drizzle channel (ch5). cond stays 9ch either way.
    mean_ch = 0 if str(getattr(training_config, "solver_warmstart", "phasebin")) == "aligned_mean" \
        else HYBRID_DRIZZLE_MEAN_CHANNEL
    return cond, mean_ch


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
    tile_batch_size: int = 16,
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

    cond, mean_ch = _solver_conditioning_from_burst(
        burst,
        shift_arr,
        training_config=training_config,
        scale=scale,
    )

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
    tile_batch_size = max(1, int(tile_batch_size))
    out_hr = np.zeros((h_lr * scale, w_lr * scale), dtype=np.float32)
    weight_hr = np.zeros_like(out_hr)

    solver = solver.to(requested_device)
    was_training = solver.training
    solver.eval()
    frame_mask = _lr_edge_mask(
        patch_lr,
        patch_lr,
        int(getattr(training_config, "solver_dc_rim_lr_px", 0)),
        requested_device,
    )

    tiles = [(y_lr, x_lr) for y_lr in ys for x_lr in xs]
    for start in range(0, len(tiles), tile_batch_size):
        chunk = tiles[start:start + tile_batch_size]
        cond_patches: list[np.ndarray] = []
        burst_patches: list[np.ndarray] = []
        for y_lr, x_lr in chunk:
            y_hr = y_lr * scale
            x_hr = x_lr * scale
            cond_patch = cond[:, y_hr : y_hr + patch_size_hr, x_hr : x_hr + patch_size_hr]
            if cond_patch.shape[1:] != (patch_size_hr, patch_size_hr):
                raise RuntimeError(f"condition patch shape mismatch: {cond_patch.shape}")
            burst_patch = burst_sub[:, y_lr : y_lr + patch_lr, x_lr : x_lr + patch_lr]
            if burst_patch.shape != (burst_sub.shape[0], patch_lr, patch_lr):
                raise RuntimeError(f"burst patch shape mismatch: {burst_patch.shape}")
            cond_patches.append(cond_patch)
            burst_patches.append(burst_patch)
        bsz = len(chunk)
        obs_t = torch.from_numpy(np.stack(cond_patches, axis=0)).to(requested_device)
        burst_t = torch.from_numpy(np.stack(burst_patches, axis=0)).to(requested_device)
        shifts_t = torch.from_numpy(np.broadcast_to(shifts_sub, (bsz, *shifts_sub.shape)).copy()).to(requested_device)
        psf = _solver_real_psf(training_config, bsz, requested_device)
        x0 = obs_t[:, mean_ch : mean_ch + 1]
        pred_t = solver(
            x0,
            burst_t,
            shifts_t,
            psf,
            obs_t,
            frame_mask=frame_mask,
        )
        pred_np = pred_t.detach().float().cpu().numpy()[:, 0]
        for (y_lr, x_lr), pred in zip(chunk, pred_np, strict=True):
            y_hr = y_lr * scale
            x_hr = x_lr * scale
            window = _window_2d(*pred.shape)
            out_hr[y_hr : y_hr + pred.shape[0], x_hr : x_hr + pred.shape[1]] += pred * window
            weight_hr[y_hr : y_hr + pred.shape[0], x_hr : x_hr + pred.shape[1]] += window

    if was_training:
        solver.train()
    return out_hr / np.maximum(weight_hr, 1e-6)


@torch.no_grad()
def infer_solver_from_burst_full_halo(
    solver: torch.nn.Module,
    lr_burst: np.ndarray,
    shifts: np.ndarray,
    *,
    training_config: Any,
    halo_hr: int,
    device: torch.device | str,
) -> np.ndarray:
    """Run full-frame solver inference with an outer reflect halo, then crop to the original FOV.

    This eval path avoids patch-local prox boundaries inside the visible image.  The halo is applied
    on the LR burst before feature fusion; after solving on the enlarged field, the HR output is
    cropped back to the original detector FOV.
    """

    scale = int(training_config.scale)
    if scale <= 0:
        raise ValueError("scale must be positive")
    if int(halo_hr) < 0 or int(halo_hr) % scale != 0:
        raise ValueError("halo_hr must be non-negative and divisible by scale")

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")

    burst = np.asarray(lr_burst, dtype=np.float32)
    shift_arr = np.asarray(shifts, dtype=np.float32)
    if burst.ndim != 3:
        raise ValueError(f"lr_burst must be (N,H,W), got {burst.shape}")
    if shift_arr.shape != (burst.shape[0], 2):
        raise ValueError(f"shifts must be ({burst.shape[0]}, 2), got {shift_arr.shape}")

    halo_lr = int(halo_hr) // scale
    if halo_lr > 0:
        burst_solve = np.pad(
            burst,
            ((0, 0), (halo_lr, halo_lr), (halo_lr, halo_lr)),
            mode="reflect",
        ).astype(np.float32, copy=False)
    else:
        burst_solve = burst

    cond, mean_ch = _solver_conditioning_from_burst(
        burst_solve,
        shift_arr,
        training_config=training_config,
        scale=scale,
    )
    burst_sub, shifts_sub = _select_solver_eval_frames(
        burst_solve,
        shift_arr,
        int(getattr(training_config, "solver_m_frames", 16)),
    )

    solver = solver.to(requested_device)
    was_training = solver.training
    solver.eval()

    obs_t = torch.from_numpy(np.ascontiguousarray(cond[None])).to(requested_device)
    burst_t = torch.from_numpy(np.ascontiguousarray(burst_sub[None])).to(requested_device)
    shifts_t = torch.from_numpy(np.ascontiguousarray(shifts_sub[None])).to(requested_device)
    psf = _solver_real_psf(training_config, 1, requested_device)
    frame_mask = _lr_edge_mask(
        burst_solve.shape[-2],
        burst_solve.shape[-1],
        int(getattr(training_config, "solver_dc_rim_lr_px", 0)),
        requested_device,
    )
    x0 = obs_t[:, mean_ch : mean_ch + 1]
    pred_t = solver(x0, burst_t, shifts_t, psf, obs_t, frame_mask=frame_mask)
    pred = pred_t.detach().float().cpu().numpy()[0, 0]
    if was_training:
        solver.train()
    if int(halo_hr) > 0:
        h_hr = int(burst.shape[-2]) * scale
        w_hr = int(burst.shape[-1]) * scale
        pred = pred[int(halo_hr) : int(halo_hr) + h_hr, int(halo_hr) : int(halo_hr) + w_hr]
    return pred.astype(np.float32, copy=False)


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
    solver_mode = str(getattr(config, "solver_mode", "tiled"))
    if solver_mode == "full_halo":
        solver_temp = infer_solver_from_burst_full_halo(
            solver,
            raw_frames,
            shifts,
            training_config=training_config,
            halo_hr=int(getattr(config, "solver_halo_hr", 0)),
            device=device,
        ).astype(np.float32, copy=False)
    elif solver_mode == "tiled":
        solver_temp = infer_solver_from_burst(
            solver,
            raw_frames,
            shifts,
            training_config=training_config,
            patch_size_hr=int(training_config.patch_size_hr),
            overlap=int(config.overlap),
            tile_batch_size=int(config.tile_batch),
            device=device,
        ).astype(np.float32, copy=False)
    else:
        raise ValueError(f"unknown solver real-eval mode: {solver_mode!r}")
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
    writer.add_scalar("eval_real/solver_halo_hr", float(getattr(config, "solver_halo_hr", 0)), step)
    # Physics-grounded real metric (ACL-032): ||A x - y|| on HELD-OUT real frames. Unlike the
    # synthetic PSNR/boundary_f1 (which reward fitting the GT generator and are anti-correlated
    # with real quality), this asks whether the reconstruction actually explains the observed
    # photons. Lower is better; compare relatively across configs (real PSF is misspecified).
    dc_band = dc_full = float("nan")
    try:
        dc_band, dc_full = _solver_real_dc_residual(
            solver, solver_temp, raw_frames, shifts, training_config=training_config, device=device
        )
        writer.add_scalar("eval_real/dc_resid_band", dc_band, step)
        writer.add_scalar("eval_real/dc_resid_full", dc_full, step)
    except Exception as exc:  # never let a monitor crash an overnight run
        print(f"[real_eval] dc_residual skipped ({type(exc).__name__}: {exc})")
    writer.flush()

    if config.save_png and config.output_dir:
        eval_dir = Path(config.output_dir) / "eval_real"
        temp_png = save_ep11_temperature_figure(
            solver_temp,
            eval_dir / (
                f"solver_step{step}_{solver_mode}"
                f"{int(getattr(config, 'solver_halo_hr', 0)) if solver_mode == 'full_halo' else ''}"
                f"_center_zoom{int(config.zoom)}x_temperature.png"
            ),
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
        "dc_resid_band": dc_band,
        "dc_resid_full": dc_full,
    }


def clear_real_eval_cache() -> None:
    _load_real_eval_cache.cache_clear()
