"""Full-frame inference for EP12 hybrid 2x-drizzle → 4x SR."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from tcforge.classical_sr import DRIZZLE_CH_MEAN, drizzle_features
from tcforge.fusion import fuse_burst_to_features


def _positions(length: int, patch: int, step: int) -> list[int]:
    if patch <= 0 or step <= 0:
        raise ValueError("patch and step must be positive")
    if patch >= length:
        return [0]
    positions = list(range(0, length - patch + 1, step))
    if positions[-1] != length - patch:
        positions.append(length - patch)
    return positions


def _window_2d(rows: int, cols: int) -> np.ndarray:
    if rows <= 1 or cols <= 1:
        return np.ones((rows, cols), dtype=np.float32)
    wy = np.hanning(rows).astype(np.float32)
    wx = np.hanning(cols).astype(np.float32)
    window = np.outer(wy, wx).astype(np.float32)
    return np.maximum(window, 1e-3)


def _split_model_output(output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, tuple):
        return output[0]
    return output


def build_obs_features(
    lr_burst: np.ndarray,
    shifts: np.ndarray,
    *,
    scale: int = 4,
    drizzle_scale: int = 2,
    sigma_bg: float = 5.0,
    drizzle_kernel: str = "bilinear",
    output_shape: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build drizzle features at drizzle_scale and 1x fused context.

    Returns
    -------
    obs_drz : (3, H_lr*drizzle_scale, W_lr*drizzle_scale)
    obs_1x  : (5, H_lr, W_lr)
    """

    drz_output_shape = None
    if output_shape is not None:
        h_hr, w_hr = output_shape
        model_up = scale // drizzle_scale
        drz_output_shape = (h_hr // model_up, w_hr // model_up)

    obs_drz = drizzle_features(
        lr_burst,
        shifts,
        scale=drizzle_scale,
        output_shape=drz_output_shape,
        kernel=drizzle_kernel,
    ).astype(np.float32, copy=False)
    obs_1x = fuse_burst_to_features(lr_burst, shifts, sigma_bg=sigma_bg).astype(np.float32, copy=False)
    return obs_drz, obs_1x


@torch.no_grad()
def infer_full_frame(
    model: torch.nn.Module,
    obs_drz: np.ndarray,
    obs_1x: np.ndarray,
    *,
    scale: int = 4,
    drizzle_scale: int = 2,
    patch_size: int = 256,
    overlap: int = 32,
    device: str = "cuda",
) -> np.ndarray:
    """Run tiled inference: input at drizzle grid, output at 4x HR grid.

    Parameters
    ----------
    obs_drz : (3, H_drz, W_drz) — drizzle features at drizzle_scale resolution
    obs_1x  : (5, H_lr, W_lr) — 1x fused context features
    scale : overall SR factor (LR→HR)
    drizzle_scale : drizzle accumulation scale
    patch_size : output patch size at HR grid (4x)
    overlap : overlap at HR grid
    """

    drizzle = np.asarray(obs_drz, dtype=np.float32)
    context = np.asarray(obs_1x, dtype=np.float32)
    if drizzle.ndim != 3 or drizzle.shape[0] < 3:
        raise ValueError("obs_drz must have shape (>=3, H, W)")
    if context.ndim != 3:
        raise ValueError("obs_1x must have shape (C, H_lr, W_lr)")
    if scale <= 0:
        raise ValueError("scale must be positive")
    model_up = scale // drizzle_scale  # 2 for default

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    model = model.to(requested_device)
    was_training = model.training
    model.eval()

    _, h_drz, w_drz = drizzle.shape
    h_hr = h_drz * model_up
    w_hr = w_drz * model_up

    # Tile at drizzle grid resolution
    input_patch = patch_size // model_up  # model input patch at drizzle grid
    input_overlap = overlap // model_up
    step = max(1, input_patch - input_overlap)
    ys = _positions(h_drz, input_patch, step)
    xs = _positions(w_drz, input_patch, step)

    out = np.zeros((h_hr, w_hr), dtype=np.float32)
    weight = np.zeros_like(out)

    for y in ys:
        for x in xs:
            drizzle_patch = drizzle[:, y : y + input_patch, x : x + input_patch]

            # Crop 1x context and upsample to drizzle grid
            y_lr = y // drizzle_scale
            x_lr = x // drizzle_scale
            p_lr = input_patch // drizzle_scale
            y_lr_end = min(y_lr + p_lr, context.shape[1])
            x_lr_end = min(x_lr + p_lr, context.shape[2])
            context_crop = context[:, y_lr:y_lr_end, x_lr:x_lr_end]

            context_tensor = torch.from_numpy(context_crop[None]).to(requested_device)
            context_up = F.interpolate(
                context_tensor,
                size=(input_patch, input_patch),
                mode="bilinear",
                align_corners=False,
            )
            drizzle_tensor = torch.from_numpy(drizzle_patch[None]).to(requested_device)
            obs = torch.cat([drizzle_tensor, context_up], dim=1)

            with torch.amp.autocast(
                device_type=requested_device.type,
                dtype=torch.float16,
                enabled=requested_device.type == "cuda",
            ):
                pred = _split_model_output(model(obs)).detach().cpu().numpy()[0, 0].astype(np.float32, copy=False)

            # Output coordinates at HR grid
            y_hr = y * model_up
            x_hr = x * model_up
            out_h, out_w = pred.shape
            window = _window_2d(out_h, out_w)
            out[y_hr : y_hr + out_h, x_hr : x_hr + out_w] += pred * window
            weight[y_hr : y_hr + out_h, x_hr : x_hr + out_w] += window

    if was_training:
        model.train()
    return out / np.maximum(weight, 1e-6)


@torch.no_grad()
def infer_from_burst(
    model: torch.nn.Module,
    lr_burst: np.ndarray,
    shifts: np.ndarray,
    *,
    scale: int = 4,
    drizzle_scale: int = 2,
    patch_size: int = 256,
    overlap: int = 32,
    device: str = "cuda",
    sigma_bg: float = 5.0,
    drizzle_kernel: str = "bilinear",
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Fuse burst observations and run tiled EP12 inference."""

    obs_drz, obs_1x = build_obs_features(
        lr_burst,
        shifts,
        scale=scale,
        drizzle_scale=drizzle_scale,
        sigma_bg=sigma_bg,
        drizzle_kernel=drizzle_kernel,
        output_shape=output_shape,
    )
    return infer_full_frame(
        model,
        obs_drz,
        obs_1x,
        scale=scale,
        drizzle_scale=drizzle_scale,
        patch_size=patch_size,
        overlap=overlap,
        device=device,
    )


def bare_drizzle_temperature(
    lr_burst: np.ndarray,
    shifts: np.ndarray,
    *,
    scale: int = 4,
    drizzle_scale: int = 2,
    drizzle_kernel: str = "bilinear",
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Return the classical scatter-add drizzle mean at HR resolution.

    Drizzle is computed at drizzle_scale, then upsampled to full scale if needed.
    """

    drz_output_shape = None
    if output_shape is not None:
        h_hr, w_hr = output_shape
        model_up = scale // drizzle_scale
        drz_output_shape = (h_hr // model_up, w_hr // model_up)

    obs_drz = drizzle_features(
        lr_burst,
        shifts,
        scale=drizzle_scale,
        output_shape=drz_output_shape,
        kernel=drizzle_kernel,
    )
    mean_drz = np.asarray(obs_drz[DRIZZLE_CH_MEAN], dtype=np.float32)

    # Upsample to full HR scale if needed
    if scale != drizzle_scale:
        from scipy.ndimage import zoom
        model_up = scale // drizzle_scale
        mean_hr = zoom(mean_drz, model_up, order=1).astype(np.float32, copy=False)
        return mean_hr
    return mean_drz
