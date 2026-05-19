from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(device: str | torch.device) -> torch.device:
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested {resolved}, but CUDA is not available")
    if resolved.type == "cuda" and resolved.index is not None:
        if resolved.index >= torch.cuda.device_count():
            raise RuntimeError(
                f"requested {resolved}, but only {torch.cuda.device_count()} CUDA device(s) are visible"
            )
    return resolved


def coordinate_grid(
    height: int,
    width: int,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype = torch.float32,
    flatten: bool = True,
) -> torch.Tensor:
    """Create an x/y coordinate grid in [-1, 1]."""

    y = torch.linspace(-1.0, 1.0, height, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, width, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    grid = torch.stack((xx, yy), dim=-1)
    return grid.reshape(-1, 2) if flatten else grid


def render_model_image(
    model: torch.nn.Module,
    hr_shape: tuple[int, int],
    *,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Render either a coordinate INR or an image-generator model to HxW."""

    if bool(getattr(model, "expects_coords", True)):
        coords = coordinate_grid(*hr_shape, device=device, dtype=dtype, flatten=True)
        pred = model(coords).reshape(hr_shape[0], hr_shape[1], -1)
        if pred.shape[-1] != 1:
            raise ValueError(f"expected one output channel, got {pred.shape[-1]}")
        return pred[..., 0]

    pred = model(output_shape=hr_shape)
    if pred.ndim == 3:
        if pred.shape[0] != 1:
            raise ValueError(f"expected one output channel, got {pred.shape[0]}")
        return pred[0]
    if pred.ndim == 2:
        return pred
    raise ValueError(f"expected HxW or 1xHxW model output, got shape {tuple(pred.shape)}")


def warmup_cosine_lr_lambda(
    step: int,
    *,
    warmup_steps: int,
    total_steps: int,
    min_factor: float = 0.05,
) -> float:
    if total_steps <= 0:
        return 1.0
    if warmup_steps > 0 and step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    denom = max(total_steps - warmup_steps, 1)
    progress = min(max((step - warmup_steps) / denom, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_factor + (1.0 - min_factor) * cosine


def sample_frame_indices(
    num_frames: int,
    batch_k: int,
    *,
    device: torch.device,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    if num_frames <= 0:
        raise ValueError("num_frames must be positive")
    k = min(batch_k, num_frames)
    return torch.randperm(num_frames, generator=generator, device=device)[:k]


def prepare_highpass_observations(
    frames: torch.Tensor,
    *,
    sigma_bg_lr_px: float = 5.0,
) -> torch.Tensor:
    """Use EP08 highpass preprocessing when available.

    Task B does not own ``highpass.py``. This hook keeps the trainer wired to
    that module without providing a conflicting implementation here.
    """

    try:
        from ep08.highpass import highpass_preprocess
    except ModuleNotFoundError:
        return frames
    return highpass_preprocess(frames, sigma_bg=sigma_bg_lr_px)


def as_frame_tensor(frames: torch.Tensor | np.ndarray, *, device: torch.device) -> torch.Tensor:
    tensor = torch.as_tensor(frames, dtype=torch.float32, device=device)
    if tensor.ndim == 4 and tensor.shape[1] == 1:
        tensor = tensor[:, 0]
    if tensor.ndim != 3:
        raise ValueError(f"expected frames as NxHxW or Nx1xHxW, got {tuple(tensor.shape)}")
    return tensor


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
