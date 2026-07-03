"""Per-frame evidence fusion for the unrolled solver (Stage 2a E1, design draft
research_log/stage2a_perframe_fusion_design_draft.md).

The E3 mainline prox conditions on 5 aggregated statistics whose construction
collapses sub-pixel phase — the diagnosed root cause #1 of the neural-vs-TGV
extraction gap (draft §1). This module lifts each burst frame onto the solver's
HR grid, encodes it together with its sub-pixel phase, and pools across frames
with permutation-invariant, M-invariant statistics, so the prox finally sees
per-frame phase evidence.

Grid convention (ACL-049 red line, draft §2/§5 test ③): the lift MUST land on
the solver x's corner grid — the same ``scale*(i+d)+{0..scale-1}`` sampling the
DC term uses. We guarantee this BY CONSTRUCTION: the splat is the exact autograd
transpose of :func:`forward_torch.block_average_shifted_batched`, i.e. the very
code the DC forward runs, so the two can never drift apart.

VRAM budget (draft §4): with feat_channels=16 and frame_chunk=8 at batch 4,
p384 HR patches, the transient per-chunk tensors are the 4-ch encoder input
(B*c,4,384,384 ≈ 75 MB fp32) plus three E-ch running stats (B,16,384,384 ≈
38 MB each) — well under the <4 GB envelope claimed in the draft even before
activation checkpointing.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .forward_torch import block_average_shifted_batched


def corner_grid_splat(
    y_frames: torch.Tensor,
    shifts: torch.Tensor,
    scale: int,
    hr_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Lift LR frames onto the solver's corner-convention HR grid.

    Args:
        y_frames: (B, C, h, w) LR frames (a chunk of the burst).
        shifts:   (B, C, 2) per-frame [dx, dy] in LR px.
        scale:    SR factor.
        hr_shape: (H, W) target HR grid (must equal (h*scale, w*scale)).

    Returns:
        (value, coverage): both (B, C, H, W). ``coverage = A_geomT 1`` is the
        scatter weight mass; ``value = (A_geomT y) / max(coverage, eps)`` is the
        coverage-normalized lift (SAA-style), where ``A_geom`` is the PSF-free
        block-average forward. Computed via autograd through
        ``block_average_shifted_batched`` so the sample positions are exactly
        the DC term's ``scale*(i+d)+{0..scale-1}`` (block center +0.5 HR px in
        both axes) — the ACL-049 convention, enforced by reuse rather than by a
        parallel implementation.
    """

    if y_frames.ndim != 4:
        raise ValueError(f"y_frames must be (B,C,h,w), got {tuple(y_frames.shape)}")
    if shifts.shape[:2] != y_frames.shape[:2] or shifts.shape[-1] != 2:
        raise ValueError(f"shifts must be (B,C,2) matching y_frames, got {tuple(shifts.shape)}")
    b, c, h, w = y_frames.shape
    height, width = int(hr_shape[0]), int(hr_shape[1])
    if (h * int(scale), w * int(scale)) != (height, width):
        raise ValueError(f"hr_shape {hr_shape} does not match LR {(h, w)} at scale {scale}")

    flat_shifts = shifts.reshape(b * c, 1, 2)
    # Two RHS per frame (the observed frame and an all-ones frame for coverage),
    # resolved with a single autograd transpose by stacking along batch.
    rhs = torch.cat(
        [
            y_frames.reshape(b * c, 1, h, w),
            torch.ones(b * c, 1, h, w, device=y_frames.device, dtype=y_frames.dtype),
        ],
        dim=0,
    )
    rhs_shifts = torch.cat([flat_shifts, flat_shifts], dim=0)
    with torch.enable_grad():
        x_dummy = torch.zeros(
            2 * b * c, height, width, device=y_frames.device, dtype=y_frames.dtype, requires_grad=True
        )
        proj = block_average_shifted_batched(x_dummy, rhs_shifts, int(scale))
        (scatter,) = torch.autograd.grad((proj * rhs).sum(), x_dummy)
    scatter = scatter.detach()
    raw = scatter[: b * c].reshape(b, c, height, width)
    coverage = scatter[b * c :].reshape(b, c, height, width)
    value = raw / coverage.clamp_min(1e-6)
    return value, coverage


def phase_maps(shifts: torch.Tensor, hr_shape: tuple[int, int]) -> torch.Tensor:
    """Constant per-frame sub-pixel phase maps: frac(dx), frac(dy) in [0,1).

    Returns (B, C, 2, H, W). The fractional part of the LR shift is exactly the
    information the 5-channel aggregate statistics destroy (draft §1)."""

    frac = shifts - torch.floor(shifts)  # [0,1) per axis
    b, c, _ = frac.shape
    height, width = int(hr_shape[0]), int(hr_shape[1])
    return frac.reshape(b, c, 2, 1, 1).expand(b, c, 2, height, width)


class PerFrameFusion(nn.Module):
    """Shared per-frame encoder + permutation/M-invariant streaming pooling.

    forward() returns (B, 3*feat_channels, H, W) = [mean ⊕ max ⊕ std] over the
    burst frames of a shared 3-layer encoding of [value, coverage, phase_x,
    phase_y]. No normalization layers (E3 extent discipline). Statistics are
    weighted by per-frame validity so padded frames neither contribute nor
    shift the denominators; duplicated frames (the constant-M top-up of
    dataset._select_m_indices) scale all weights uniformly and leave every
    statistic unchanged — the aggregation is M-invariant by construction
    (unlike the DC term's frame SUM, forward_torch.py:715).
    """

    def __init__(self, feat_channels: int = 16, frame_chunk: int = 8) -> None:
        super().__init__()
        if feat_channels < 1:
            raise ValueError("feat_channels must be >= 1")
        if frame_chunk < 1:
            raise ValueError("frame_chunk must be >= 1")
        self.feat_channels = int(feat_channels)
        self.frame_chunk = int(frame_chunk)
        e = self.feat_channels
        self.enc = nn.Sequential(
            nn.Conv2d(4, e, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(e, e, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(e, e, 3, padding=1),
        )

    @property
    def out_channels(self) -> int:
        return 3 * self.feat_channels

    @staticmethod
    def _frame_weights(frame_mask: torch.Tensor | None, b: int, m: int, device, dtype) -> torch.Tensor:
        """Reduce an optional DC-style frame_mask to per-frame scalar weights (B, M).

        The solver's frame_mask is either a spatial rim mask (broadcast over
        frames — every frame stays valid) or a (B, M, 1, 1) validity mask;
        a frame counts as invalid only when its mask is identically zero."""

        if frame_mask is None:
            return torch.ones(b, m, device=device, dtype=dtype)
        mask = frame_mask.to(dtype)
        if mask.ndim == 4 and mask.shape[:2] == (b, m):
            per_frame = mask.flatten(2).amax(dim=2)
            return (per_frame > 0).to(dtype)
        # Spatial-only mask (e.g. the LR edge rim): applies inside the DC term,
        # not to frame validity.
        return torch.ones(b, m, device=device, dtype=dtype)

    def forward(
        self,
        y_burst: torch.Tensor,
        shifts: torch.Tensor,
        scale: int,
        hr_shape: tuple[int, int],
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if y_burst.ndim != 4:
            raise ValueError(f"y_burst must be (B,M,h,w), got {tuple(y_burst.shape)}")
        b, m = int(y_burst.shape[0]), int(y_burst.shape[1])
        height, width = int(hr_shape[0]), int(hr_shape[1])
        e = self.feat_channels
        weights = self._frame_weights(frame_mask, b, m, y_burst.device, y_burst.dtype)
        if not bool((weights.sum(dim=1) > 0).all()):
            raise ValueError("PerFrameFusion needs at least one valid frame per sample")

        # Streaming weighted Welford (merge form) in float64 so the pooled
        # statistics are permutation-stable to ~1e-6 (draft §6 numerical risk).
        stat_dtype = torch.float64
        w_total = torch.zeros(b, 1, 1, 1, device=y_burst.device, dtype=stat_dtype)
        mean = torch.zeros(b, e, height, width, device=y_burst.device, dtype=stat_dtype)
        m2 = torch.zeros(b, e, height, width, device=y_burst.device, dtype=stat_dtype)
        running_max = torch.full(
            (b, e, height, width), float("-inf"), device=y_burst.device, dtype=stat_dtype
        )

        for start in range(0, m, self.frame_chunk):
            stop = min(start + self.frame_chunk, m)
            c = stop - start
            w_chunk = weights[:, start:stop]  # (B, c)
            value, coverage = corner_grid_splat(
                y_burst[:, start:stop], shifts[:, start:stop], scale, (height, width)
            )
            phase = phase_maps(shifts[:, start:stop], (height, width))
            feats_in = torch.cat([value.unsqueeze(2), coverage.unsqueeze(2), phase], dim=2)
            encoded = self.enc(feats_in.reshape(b * c, 4, height, width)).reshape(b, c, e, height, width)

            enc64 = encoded.to(stat_dtype)
            wc = w_chunk.to(stat_dtype).reshape(b, c, 1, 1, 1)
            chunk_w = wc.sum(dim=1)  # (B,1,1,1)
            safe_w = chunk_w.clamp_min(1e-12)
            chunk_mean = (enc64 * wc).sum(dim=1) / safe_w
            chunk_m2 = (wc * (enc64 - chunk_mean.unsqueeze(1)) ** 2).sum(dim=1)

            new_total = w_total + chunk_w
            has_chunk = chunk_w > 0
            delta = chunk_mean - mean
            safe_total = new_total.clamp_min(1e-12)
            mean = torch.where(has_chunk, mean + delta * (chunk_w / safe_total), mean)
            m2 = torch.where(has_chunk, m2 + chunk_m2 + delta**2 * (w_total * chunk_w / safe_total), m2)
            w_total = new_total

            masked = torch.where(
                (wc > 0).expand_as(enc64), enc64, torch.full_like(enc64, float("-inf"))
            )
            running_max = torch.maximum(running_max, masked.amax(dim=1))

        var = (m2 / w_total.clamp_min(1e-12)).clamp_min(0.0)
        pooled = torch.cat([mean, running_max, torch.sqrt(var)], dim=1)
        return pooled.to(y_burst.dtype)
