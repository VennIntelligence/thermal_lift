"""Inference helpers for EP07v2 UNet SR."""

from __future__ import annotations

import math

import numpy as np
import torch

from tcforge.classical_sr import drizzle_features
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


@torch.no_grad()
def infer_full_frame(
    model: torch.nn.Module,
    obs_features: np.ndarray,
    *,
    scale: int = 4,
    patch_size_hr: int = 256,
    overlap: int = 32,
    device: str = "cuda",
    residual: bool = False,
) -> np.ndarray:
    """Run tiled full-frame inference from observation features.

    When *residual* is True, the model operates at the target resolution
    (scale=1 internally) and its output is added to the last channel of
    *obs_features* (the classical SR baseline).
    """

    features = np.asarray(obs_features, dtype=np.float32)
    if features.ndim != 3:
        raise ValueError("obs_features must have shape (C, H_lr, W_lr)")
    if scale <= 0:
        raise ValueError("scale must be positive")
    if patch_size_hr <= 0 or patch_size_hr % scale != 0:
        raise ValueError("patch_size_hr must be positive and divisible by scale")
    if overlap < 0 or overlap >= patch_size_hr:
        raise ValueError("overlap must satisfy 0 <= overlap < patch_size_hr")

    # In residual mode, model uses scale=1 (same input/output resolution)
    model_scale = 1 if residual else scale

    requested_device = torch.device(device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        requested_device = torch.device("cpu")
    model = model.to(requested_device)
    was_training = model.training
    model.eval()

    _, h_lr, w_lr = features.shape
    patch_lr = max(1, patch_size_hr // model_scale)
    overlap_lr = int(math.floor(overlap / model_scale))
    step_lr = max(1, patch_lr - overlap_lr)
    ys = _positions(h_lr, patch_lr, step_lr)
    xs = _positions(w_lr, patch_lr, step_lr)

    out_hr = np.zeros((h_lr * model_scale, w_lr * model_scale), dtype=np.float32)
    weight_hr = np.zeros_like(out_hr)

    for y_lr in ys:
        for x_lr in xs:
            patch = features[:, y_lr : y_lr + patch_lr, x_lr : x_lr + patch_lr]
            tensor = torch.from_numpy(patch[None, :, :, :]).to(requested_device)
            with torch.amp.autocast(
                device_type=requested_device.type,
                dtype=torch.float16,
                enabled=requested_device.type == "cuda",
            ):
                pred = model(tensor).detach().cpu().numpy()[0, 0].astype(np.float32, copy=False)
            y_hr = y_lr * model_scale
            x_hr = x_lr * model_scale
            rows, cols = pred.shape
            window = _window_2d(rows, cols)
            out_hr[y_hr : y_hr + rows, x_hr : x_hr + cols] += pred * window
            weight_hr[y_hr : y_hr + rows, x_hr : x_hr + cols] += window

    if was_training:
        model.train()
    result = out_hr / np.maximum(weight_hr, 1e-6)

    if residual:
        # Add classical SR baseline (last channel of input features)
        classical_sr = features[-1]
        # classical_sr is at the same resolution as result
        result = result + classical_sr

    return result


def infer_from_burst(
    model: torch.nn.Module,
    lr_burst: np.ndarray,
    shifts: np.ndarray,
    *,
    scale: int = 4,
    patch_size_hr: int = 256,
    overlap: int = 32,
    device: str = "cuda",
    sigma_bg: float = 5.0,
    residual: bool = False,
    classical_sr: np.ndarray | None = None,
    input_mode: str = "lr",
) -> np.ndarray:
    """Fuse an LR burst and run tiled UNet inference.

    When *residual* is True, *classical_sr* must be provided (or it is
    computed via shift-and-add internally) and the model output is added
    to it.

    When *input_mode* is ``"hybrid_drizzle2x"``, 5ch fused features are
    upsampled to 2x and concatenated with 3ch scatter drizzle at 2x,
    yielding an 8ch input at 2x grid.  The model operates at scale=1
    (direct prediction, no residual add).
    """

    if input_mode == "hybrid_drizzle2x":
        from scipy.ndimage import zoom
        features_1x = fuse_burst_to_features(lr_burst, shifts, sigma_bg=sigma_bg)
        features_up = zoom(features_1x, (1, scale, scale), order=1).astype(np.float32)
        drz = drizzle_features(lr_burst, shifts, scale=scale, kernel="bilinear")
        features = np.concatenate([features_up, drz], axis=0)
        return infer_full_frame(
            model,
            features,
            scale=1,
            patch_size_hr=patch_size_hr,
            overlap=overlap,
            device=device,
            residual=False,
        )

    features = fuse_burst_to_features(lr_burst, shifts, sigma_bg=sigma_bg)

    if residual:
        from scipy.ndimage import zoom
        features_hr = zoom(features, (1, scale, scale), order=1).astype(np.float32)
        if classical_sr is None:
            from tcforge.classical_sr import shift_and_add
            classical_sr = shift_and_add(lr_burst, shifts, scale=scale)
        features = np.concatenate([features_hr, classical_sr[None, :, :]], axis=0)

    return infer_full_frame(
        model,
        features,
        scale=scale,
        patch_size_hr=patch_size_hr,
        overlap=overlap,
        device=device,
        residual=residual,
    )
