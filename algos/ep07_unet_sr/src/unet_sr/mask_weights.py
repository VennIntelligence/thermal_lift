"""Per-patch thin-structure and narrow-gap loss weight maps for ContourSRLoss."""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage


def _narrow_gap_mask(binary: np.ndarray, *, max_width_px: int) -> np.ndarray:
    """Detect background pixels with structure on opposing sides nearby."""

    bg = ~binary
    if not np.any(bg):
        return np.zeros_like(binary, dtype=bool)
    bg_dist = ndimage.distance_transform_edt(bg)
    width_est = 2.0 * bg_dist - 1.0
    narrow = bg & (width_est <= float(max_width_px))
    radius = max(1, int(max_width_px) + 1)

    left = np.zeros_like(binary, dtype=bool)
    right = np.zeros_like(binary, dtype=bool)
    up = np.zeros_like(binary, dtype=bool)
    down = np.zeros_like(binary, dtype=bool)
    for offset in range(1, radius + 1):
        left[:, offset:] |= binary[:, :-offset]
        right[:, :-offset] |= binary[:, offset:]
        up[offset:, :] |= binary[:-offset, :]
        down[:-offset, :] |= binary[offset:, :]
    between_structures = (left & right) | (up & down)
    return narrow & between_structures


def _as_mask_2d(hr_mask: np.ndarray) -> np.ndarray:
    arr = np.asarray(hr_mask, dtype=np.float32)
    if arr.ndim == 4:
        if arr.shape[0] != 1 or arr.shape[1] != 1:
            raise ValueError(f"batched mask must have shape (1, 1, H, W), got {arr.shape}")
        arr = arr[0, 0]
    elif arr.ndim == 3:
        if arr.shape[0] != 1:
            raise ValueError(f"mask must have shape (1, H, W) or (H, W), got {arr.shape}")
        arr = arr[0]
    elif arr.ndim != 2:
        raise ValueError(f"mask must be 2D or (1,H,W), got {arr.shape}")
    return arr


def compute_mask_loss_weights_np(
    hr_mask: np.ndarray,
    *,
    thin_boost: float,
    gap_boost: float,
    max_width_px: int = 3,
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Build optional thin-structure and narrow-gap loss multipliers for one patch.

    Operates on a single HR patch mask. Distance transforms use the thresholded
    support so AA edge pixels do not become their own morphology targets.
    """

    need_thin = float(thin_boost) > 1.0
    need_gap = float(gap_boost) > 1.0
    if not need_thin and not need_gap:
        return None, None

    mask_2d = _as_mask_2d(hr_mask)
    binary = mask_2d >= 0.5
    thin = np.ones((1, *mask_2d.shape), dtype=np.float32) if need_thin else None
    gap = np.ones((1, *mask_2d.shape), dtype=np.float32) if need_gap else None

    if need_thin:
        struct_dist = ndimage.distance_transform_edt(binary)
        width_est = 2.0 * struct_dist - 1.0
        thin_roi = binary & (width_est <= float(max_width_px))
        thin[0][thin_roi] = float(thin_boost)

    if need_gap:
        gap_roi = _narrow_gap_mask(binary, max_width_px=max_width_px)
        gap[0][gap_roi] = float(gap_boost)

    return thin, gap


def compute_mask_loss_weights(
    hr_mask: torch.Tensor,
    *,
    thin_boost: float,
    gap_boost: float,
    max_width_px: int = 3,
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """Batch wrapper kept for tests; prefer dataset-side per-patch computation."""

    need_thin = float(thin_boost) > 1.0
    need_gap = float(gap_boost) > 1.0
    if not need_thin and not need_gap:
        return None, None
    if hr_mask.ndim != 4:
        raise ValueError("hr_mask must have shape (B, C, H, W)")

    mask_np = hr_mask.detach().cpu().numpy()
    thin = np.ones(mask_np.shape, dtype=np.float32) if need_thin else None
    gap = np.ones(mask_np.shape, dtype=np.float32) if need_gap else None
    for index in np.ndindex(mask_np.shape[:2]):
        binary = mask_np[index] >= 0.5
        if need_thin:
            struct_dist = ndimage.distance_transform_edt(binary)
            width_est = 2.0 * struct_dist - 1.0
            thin_roi = binary & (width_est <= float(max_width_px))
            thin[index][thin_roi] = float(thin_boost)  # type: ignore[index]
        if need_gap:
            gap_roi = _narrow_gap_mask(binary, max_width_px=max_width_px)
            gap[index][gap_roi] = float(gap_boost)  # type: ignore[index]

    thin_tensor = None if thin is None else torch.from_numpy(thin)
    gap_tensor = None if gap is None else torch.from_numpy(gap)
    return thin_tensor, gap_tensor
