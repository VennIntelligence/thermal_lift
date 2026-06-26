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


def _support_boundary(binary: np.ndarray) -> np.ndarray:
    """4-neighbour morphological boundary of a binary support mask.

    A pixel is on the boundary if the support label differs across any of its
    four edges.  This fires identically on an outer contour, a hole rim, a
    crack wall, or a notch edge — no assumption about line-like geometry.
    """

    edge = np.zeros_like(binary, dtype=bool)
    diff_v = binary[:-1, :] != binary[1:, :]
    diff_h = binary[:, :-1] != binary[:, 1:]
    edge[:-1, :] |= diff_v
    edge[1:, :] |= diff_v
    edge[:, :-1] |= diff_h
    edge[:, 1:] |= diff_h
    return edge


def compute_boundary_weight_np(
    hr_mask: np.ndarray,
    *,
    boundary_boost: float,
    tau_px: float = 2.5,
) -> np.ndarray | None:
    """Geometry-agnostic boundary-emphasis loss weight for one HR patch.

    Replaces the thin-structure / narrow-gap masks (which encoded a
    perfect-line / perfect-rectangle prior).  Builds a single continuous
    multiplier from the *distance to the nearest support boundary*::

        w = 1 + boundary_boost * exp(-(dist / tau_px) ** 2)

    so that *every* structural edge — outer chip contour, hole rim, crack
    wall, notch — is emphasised uniformly, and the weight decays to 1 in flat
    interiors and background.  The old special cases fall out for free:

    - **Thin structures**: every pixel sits within ``tau_px`` of a boundary,
      so the whole sliver is boosted (no width threshold, no line detector).
    - **Narrow gaps / cracks**: background pixels flanked by support are close
      to two boundaries, so they are boosted too.

    Unlike a gradient-of-signal weight, this is **contrast-independent**: it
    boosts a faint low-ΔT defect edge exactly as much as a high-contrast one,
    which matters for the hard/stress difficulty tiers.

    Returns ``(1, H, W)`` float32, or ``None`` when ``boundary_boost <= 0``
    (caller treats a missing weight as uniform).
    """

    if float(boundary_boost) <= 0.0:
        return None
    mask_2d = _as_mask_2d(hr_mask)
    binary = mask_2d >= 0.5
    # A patch fully inside one body (all support) or fully background has no
    # boundary; emphasise nothing rather than dividing by an empty distance map.
    if not binary.any() or binary.all():
        return np.ones((1, *mask_2d.shape), dtype=np.float32)
    boundary = _support_boundary(binary)
    dist = ndimage.distance_transform_edt(~boundary)
    prox = np.exp(-((dist / float(tau_px)) ** 2))
    weight = (1.0 + float(boundary_boost) * prox)[None]
    return weight.astype(np.float32, copy=False)


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
