"""Certified differentiable forward operator A for the physics-constrained unrolled SR solver.

Replicates tcforge ``physical_block_average_forward`` EXACTLY — the same operator that
generated the v3 training bursts.  Validated to ~3e-6 against the data-gen pipeline on real
scenes (see ``tests/test_forward_torch.py`` Gate A; the numpy parity proof is in the project
scratchpad ``forward_match_numpy.py`` / ``data_integrity_local.py``).

Key design decision (do NOT replace with a hand-written adjoint):
    The adjoint ``A^T`` is obtained from autograd — the vjp of a linear map IS its exact
    transpose, boundary handling included.  The existing numpy/ep08/ep15 hand-written
    adjoints are only APPROXIMATE backprojections (they disagree with A^T at the boundary by
    ~0.5 in a dense-matrix test), which would inject a systematic per-iteration error into the
    DC gradient step.  ``data_consistency_grad`` below computes ``A^T(Ax - y)`` via
    ``torch.autograd.grad`` so the unrolled solver gets the exact gradient and is end-to-end
    trainable.

Convention (matches tcforge.forward and forward self-check T1):
    * shifts ``(...,2)`` are ``[dx, dy]`` in **LR pixels**; HR is sampled at
      ``scale*(i_lr + d) + block_offset`` with ``block_offset in {0..scale-1}``.  The constant
      +0.5 HR-px block-center offset (self-check T1) is implicit in this offset grid and is
      reproduced by autograd in A^T automatically.
    * PSF sigma in HR px = ``psf_sigma_lr_px * scale``.
    * 65% of v3 scenes use elliptical/Airy PSFs — the DC operator MUST use the per-scene PSF
      from metadata (``psf_shape``/``psf_sigma_y_lr_px``/``psf_angle_deg``), else it is
      misspecified for those scenes.  Pass them through ``ScenePSF``.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F

# Vectorized batched forward (ACL-033): replaces the per-sample `for b in range(B)` Python loop
# (which serialized the physics and starved the GPU) with grouped convolutions that process the
# whole batch's heterogeneous per-scene PSFs in a single launch. Numerically equivalent to the
# certified per-sample path (proven to fp64 in tests/test_forward_torch.py). Set
# TL_SOLVER_FAST_FORWARD=0 (or pass forward_burst(..., fast=False)) to fall back to the reference
# loop if anything misbehaves on a given box.
_FAST_FORWARD_DEFAULT = os.environ.get("TL_SOLVER_FAST_FORWARD", "1") != "0"


# ---------------------------------------------------------------------------
# PSF kernels — mirror tcforge.physics.make_psf_kernel / apply_psf_blur
# ---------------------------------------------------------------------------


def gaussian_kernel1d(sigma_hr: float, *, device, dtype, truncate: float = 4.0) -> torch.Tensor | None:
    """Separable Gaussian 1D kernel matching scipy.gaussian_filter (radius int(4σ+0.5))."""
    if sigma_hr <= 0:
        return None
    radius = int(truncate * float(sigma_hr) + 0.5)
    t = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    k = torch.exp(-0.5 * (t / float(sigma_hr)) ** 2)
    return k / k.sum()


def make_psf_kernel2d(
    *,
    psf_sigma_lr_px: float,
    scale: int,
    psf_shape: str = "gaussian",
    psf_sigma_y_lr_px: float | None = None,
    psf_angle_deg: float = 0.0,
    device,
    dtype,
) -> torch.Tensor | None:
    """2D PSF kernel on the HR grid — torch port of tcforge.physics.make_psf_kernel."""
    sx = float(psf_sigma_lr_px) * scale
    sy = (float(psf_sigma_y_lr_px) if psf_sigma_y_lr_px is not None else float(psf_sigma_lr_px)) * scale
    if sx == 0 and sy == 0:
        return None
    radius = max(int(math.ceil(max(sx, sy, 1.0) * 4.0)), 1)
    ax = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    yy, xx = torch.meshgrid(ax, ax, indexing="ij")
    theta = math.radians(float(psf_angle_deg))
    cos_t, sin_t = math.cos(theta), math.sin(theta)
    xr = xx * cos_t + yy * sin_t
    yr = -xx * sin_t + yy * cos_t
    sxe, sye = max(sx, 1e-6), max(sy, 1e-6)
    if psf_shape in ("gaussian", "elliptical_gaussian"):
        kernel = torch.exp(-0.5 * ((xr / sxe) ** 2 + (yr / sye) ** 2))
    elif psf_shape == "airy_disk":
        r = torch.sqrt((xr / sxe) ** 2 + (yr / sye) ** 2)
        z = math.pi * torch.clamp(r, min=1e-8)
        kernel = (2.0 * torch.special.bessel_j1(z) / z) ** 2
        kernel = torch.where(r < 1e-8, torch.ones_like(kernel), kernel)
    else:
        raise ValueError(f"psf_shape must be gaussian/elliptical_gaussian/airy_disk, got {psf_shape!r}")
    kernel = torch.clamp(kernel, min=0.0)
    total = kernel.sum()
    if total <= 0:
        raise ValueError("PSF kernel has zero mass")
    return kernel / total


def _blur_gauss_separable(img: torch.Tensor, sigma_hr: float) -> torch.Tensor:
    """Zero-padded separable Gaussian blur of a 2D image — matches gaussian_filter(mode='constant')."""
    k = gaussian_kernel1d(sigma_hr, device=img.device, dtype=img.dtype)
    if k is None:
        return img
    r = k.numel() // 2
    x = img[None, None]
    x = F.conv2d(F.pad(x, (0, 0, r, r)), k.view(1, 1, -1, 1))
    x = F.conv2d(F.pad(x, (r, r, 0, 0)), k.view(1, 1, 1, -1))
    return x[0, 0]


def _blur_kernel2d(img: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    """Zero-padded 2D convolution — matches ndimage.convolve(mode='constant').

    The PSF kernels are 180°-symmetric so convolution == correlation; we flip anyway to match
    ndimage.convolve semantics exactly for any future asymmetric kernel.
    """
    r = kernel.shape[0] // 2
    kk = torch.flip(kernel, (0, 1)).view(1, 1, *kernel.shape)
    return F.conv2d(F.pad(img[None, None], (r, r, r, r)), kk)[0, 0]


# ---------------------------------------------------------------------------
# Shifted detector block-average — exact port of tcforge.forward._block_average_from_blurred
# (the gather form; validated to 1.4e-7 vs the data-gen forward in forward_match_numpy.py)
# ---------------------------------------------------------------------------


def block_average_shifted(blurred: torch.Tensor, shifts: torch.Tensor, scale: int) -> torch.Tensor:
    """Block-average a pre-blurred HR image into N shifted LR frames.

    Args:
        blurred: ``(H, W)`` PSF-blurred HR image (differentiable).
        shifts:  ``(N, 2)`` ``[dx, dy]`` in LR pixels.
        scale:   integer SR factor.
    Returns: ``(N, h, w)`` LR frames, ``h = H // scale``.
    """
    H, W = blurred.shape
    s = int(scale)
    h, w = H // s, W // s
    N = shifts.shape[0]
    dev, dt = blurred.device, blurred.dtype
    off = torch.arange(s, device=dev, dtype=dt)
    dy = shifts[:, 1].view(N, 1, 1)
    dx = shifts[:, 0].view(N, 1, 1)
    # sample positions, ravel order [lr0_off0, lr0_off1, lr1_off0, ...] == reshape (h, s)
    yy = (s * (torch.arange(h, device=dev, dtype=dt).view(1, h, 1) + dy) + off.view(1, 1, s)).reshape(N, s * h)
    xx = (s * (torch.arange(w, device=dev, dtype=dt).view(1, w, 1) + dx) + off.view(1, 1, s)).reshape(N, s * w)
    y0 = torch.floor(yy)
    x0 = torch.floor(xx)
    fy = (yy - y0).unsqueeze(-1)            # (N, s*h, 1)
    fx = (xx - x0).unsqueeze(1)            # (N, 1, s*w)
    vy = ((yy >= 0) & (yy <= H - 1)).to(dt).unsqueeze(-1)
    vx = ((xx >= 0) & (xx <= W - 1)).to(dt).unsqueeze(1)
    y0i = y0.long(); y1i = y0i + 1
    x0i = x0.long(); x1i = x0i + 1
    y0c = y0i.clamp(0, H - 1); y1c = y1i.clamp(0, H - 1)
    x0c = x0i.clamp(0, W - 1); x1c = x1i.clamp(0, W - 1)
    # row gather: (N, s*h, W)
    rows = (
        blurred.index_select(0, y0c.reshape(-1)).view(N, s * h, W) * (1.0 - fy)
        + blurred.index_select(0, y1c.reshape(-1)).view(N, s * h, W) * fy
    ) * vy
    # col gather: (N, s*h, s*w)
    idx0 = x0c.unsqueeze(1).expand(N, s * h, s * w)
    idx1 = x1c.unsqueeze(1).expand(N, s * h, s * w)
    cols = (rows.gather(2, idx0) * (1.0 - fx) + rows.gather(2, idx1) * fx) * vx
    return cols.reshape(N, h, s, w, s).mean(dim=(2, 4))


def block_average_shifted_batched(blurred: torch.Tensor, shifts: torch.Tensor, scale: int) -> torch.Tensor:
    """Batched (over B) form of :func:`block_average_shifted` — identical math, one launch.

    Args:
        blurred: ``(B, H, W)`` per-sample PSF-blurred HR images.
        shifts:  ``(B, N, 2)`` ``[dx, dy]`` in LR pixels.
    Returns: ``(B, N, h, w)`` LR frames.
    """
    B, H, W = blurred.shape
    s = int(scale)
    h, w = H // s, W // s
    N = shifts.shape[1]
    dev, dt = blurred.device, blurred.dtype
    off = torch.arange(s, device=dev, dtype=dt)
    dy = shifts[:, :, 1].view(B, N, 1, 1)
    dx = shifts[:, :, 0].view(B, N, 1, 1)
    ar_h = torch.arange(h, device=dev, dtype=dt).view(1, 1, h, 1)
    ar_w = torch.arange(w, device=dev, dtype=dt).view(1, 1, w, 1)
    yy = (s * (ar_h + dy) + off.view(1, 1, 1, s)).reshape(B, N, s * h)
    xx = (s * (ar_w + dx) + off.view(1, 1, 1, s)).reshape(B, N, s * w)
    y0 = torch.floor(yy)
    x0 = torch.floor(xx)
    fy = (yy - y0).unsqueeze(-1)            # (B, N, s*h, 1)
    fx = (xx - x0).unsqueeze(2)            # (B, N, 1, s*w)
    vy = ((yy >= 0) & (yy <= H - 1)).to(dt).unsqueeze(-1)
    vx = ((xx >= 0) & (xx <= W - 1)).to(dt).unsqueeze(2)
    y0i = y0.long(); y1i = y0i + 1
    x0i = x0.long(); x1i = x0i + 1
    y0c = y0i.clamp(0, H - 1); y1c = y1i.clamp(0, H - 1)
    x0c = x0i.clamp(0, W - 1); x1c = x1i.clamp(0, W - 1)
    # row gather along H: (B, N, s*h, W)
    be = blurred.unsqueeze(1).expand(B, N, H, W)
    y0e = y0c.unsqueeze(-1).expand(B, N, s * h, W)
    y1e = y1c.unsqueeze(-1).expand(B, N, s * h, W)
    rows = (be.gather(2, y0e) * (1.0 - fy) + be.gather(2, y1e) * fy) * vy
    # col gather along W: (B, N, s*h, s*w)
    idx0 = x0c.unsqueeze(2).expand(B, N, s * h, s * w)
    idx1 = x1c.unsqueeze(2).expand(B, N, s * h, s * w)
    cols = (rows.gather(3, idx0) * (1.0 - fx) + rows.gather(3, idx1) * fx) * vx
    return cols.reshape(B, N, h, s, w, s).mean(dim=(3, 5))


# ---------------------------------------------------------------------------
# Per-scene PSF spec + batched forward + autograd data-consistency gradient
# ---------------------------------------------------------------------------


@dataclass
class ScenePSF:
    """Per-sample PSF parameters (from scene metadata).  Lists are length-B (one per sample)."""
    sigma_lr_px: torch.Tensor          # (B,)
    shape: list[str]                   # len B: 'gaussian'|'elliptical_gaussian'|'airy_disk'
    sigma_y_lr_px: list[float | None]  # len B
    angle_deg: torch.Tensor            # (B,)


@dataclass(frozen=True)
class _BlurPlan:
    sep_index: torch.Tensor | None
    sep_weight_v: torch.Tensor | None
    sep_weight_h: torch.Tensor | None
    ker_index: torch.Tensor | None
    ker_weight: torch.Tensor | None
    restore_perm: torch.Tensor


@dataclass(frozen=True)
class _BlockAveragePlan:
    scale: int
    hr_shape: tuple[int, int]
    out_shape: tuple[int, int]
    y0c: torch.Tensor
    y1c: torch.Tensor
    x0c: torch.Tensor
    x1c: torch.Tensor
    fy: torch.Tensor
    fx: torch.Tensor
    vy: torch.Tensor
    vx: torch.Tensor


@dataclass(frozen=True)
class ForwardBurstPlan:
    """Precomputed per-batch constants for repeated calls to ``forward_burst``.

    A solver step calls the same physical forward operator many times: once in each K-step DC
    update and once again during the custom VJP backward. PSF kernels and shifted detector gather
    indices depend on the batch metadata, not on the current estimate x, so they can be built once
    and reused without changing the math.
    """

    scale: int
    dtype: torch.dtype
    device: torch.device
    blur: _BlurPlan
    block: _BlockAveragePlan


def _blur_one(img: torch.Tensor, psf: ScenePSF, b: int, scale: int) -> torch.Tensor:
    shape = psf.shape[b]
    sigma = float(psf.sigma_lr_px[b])
    if shape == "gaussian" and psf.sigma_y_lr_px[b] is None and float(psf.angle_deg[b]) == 0.0:
        return _blur_gauss_separable(img, sigma * scale)  # fast separable path
    kernel = make_psf_kernel2d(
        psf_sigma_lr_px=sigma, scale=scale, psf_shape=shape,
        psf_sigma_y_lr_px=psf.sigma_y_lr_px[b], psf_angle_deg=float(psf.angle_deg[b]),
        device=img.device, dtype=img.dtype,
    )
    return img if kernel is None else _blur_kernel2d(img, kernel)


def _compute_dtype(dtype: torch.dtype) -> torch.dtype:
    """Upcast half precision to fp32 for AMP safety (ACL-011 fp16 Gaussian NaNs), but PRESERVE
    fp64 so the operator can be certified at double precision (Gate A linearity/adjoint)."""
    return torch.float32 if dtype in (torch.float16, torch.bfloat16) else dtype


def _build_blur_plan(psf: ScenePSF, scale: int, *, device: torch.device, dtype: torch.dtype) -> _BlurPlan:
    sep_idx: list[int] = []
    sep_k: list[torch.Tensor] = []
    ker_idx: list[int] = []
    ker_k: list[torch.Tensor] = []
    for b in range(len(psf.shape)):
        shape = psf.shape[b]
        sigma = float(psf.sigma_lr_px[b])
        sy = psf.sigma_y_lr_px[b]
        ang = float(psf.angle_deg[b])
        if shape == "gaussian" and sy is None and ang == 0.0:
            k1 = gaussian_kernel1d(sigma * scale, device=device, dtype=dtype)
            if k1 is None:
                k1 = torch.ones(1, device=device, dtype=dtype)
            sep_idx.append(b)
            sep_k.append(k1)
        else:
            k2 = make_psf_kernel2d(
                psf_sigma_lr_px=sigma, scale=scale, psf_shape=shape,
                psf_sigma_y_lr_px=sy, psf_angle_deg=ang, device=device, dtype=dtype,
            )
            if k2 is None:
                k2 = torch.ones(1, 1, device=device, dtype=dtype)
            ker_idx.append(b)
            ker_k.append(torch.flip(k2, (0, 1)))

    order: list[int] = []
    sep_index = sep_weight_v = sep_weight_h = None
    if sep_idx:
        order += sep_idx
        sep_index = torch.tensor(sep_idx, device=device, dtype=torch.long)
        radius = max(k.numel() // 2 for k in sep_k)
        sep_weight_v = torch.zeros(len(sep_k), 1, 2 * radius + 1, 1, device=device, dtype=dtype)
        for i, k in enumerate(sep_k):
            r = k.numel() // 2
            sep_weight_v[i, 0, radius - r:radius + r + 1, 0] = k
        sep_weight_h = sep_weight_v.view(len(sep_k), 1, 1, 2 * radius + 1)

    ker_index = ker_weight = None
    if ker_idx:
        order += ker_idx
        ker_index = torch.tensor(ker_idx, device=device, dtype=torch.long)
        radius = max(k.shape[0] // 2 for k in ker_k)
        ker_weight = torch.zeros(len(ker_k), 1, 2 * radius + 1, 2 * radius + 1, device=device, dtype=dtype)
        for i, k in enumerate(ker_k):
            r = k.shape[0] // 2
            ker_weight[i, 0, radius - r:radius + r + 1, radius - r:radius + r + 1] = k

    restore_perm = torch.argsort(torch.tensor(order, device=device, dtype=torch.long))
    return _BlurPlan(sep_index, sep_weight_v, sep_weight_h, ker_index, ker_weight, restore_perm)


def _build_block_average_plan(
    shifts: torch.Tensor,
    scale: int,
    hr_shape: tuple[int, int],
    *,
    dtype: torch.dtype,
) -> _BlockAveragePlan:
    H, W = int(hr_shape[0]), int(hr_shape[1])
    s = int(scale)
    h, w = H // s, W // s
    B, N = shifts.shape[:2]
    dev = shifts.device
    st = shifts.to(dtype)
    off = torch.arange(s, device=dev, dtype=dtype)
    dy = st[:, :, 1].view(B, N, 1, 1)
    dx = st[:, :, 0].view(B, N, 1, 1)
    ar_h = torch.arange(h, device=dev, dtype=dtype).view(1, 1, h, 1)
    ar_w = torch.arange(w, device=dev, dtype=dtype).view(1, 1, w, 1)
    yy = (s * (ar_h + dy) + off.view(1, 1, 1, s)).reshape(B, N, s * h)
    xx = (s * (ar_w + dx) + off.view(1, 1, 1, s)).reshape(B, N, s * w)
    y0 = torch.floor(yy)
    x0 = torch.floor(xx)
    fy = (yy - y0).unsqueeze(-1)
    fx = (xx - x0).unsqueeze(2)
    vy = ((yy >= 0) & (yy <= H - 1)).to(dtype).unsqueeze(-1)
    vx = ((xx >= 0) & (xx <= W - 1)).to(dtype).unsqueeze(2)
    y0i = y0.long(); y1i = y0i + 1
    x0i = x0.long(); x1i = x0i + 1
    return _BlockAveragePlan(
        scale=s,
        hr_shape=(H, W),
        out_shape=(h, w),
        y0c=y0i.clamp(0, H - 1),
        y1c=y1i.clamp(0, H - 1),
        x0c=x0i.clamp(0, W - 1),
        x1c=x1i.clamp(0, W - 1),
        fy=fy,
        fx=fx,
        vy=vy,
        vx=vx,
    )


def prepare_forward_burst_plan(
    shifts: torch.Tensor,
    psf: ScenePSF,
    scale: int,
    hr_shape: tuple[int, int],
    *,
    dtype: torch.dtype,
) -> ForwardBurstPlan:
    """Build reusable per-batch constants for repeated ``forward_burst`` calls."""

    if shifts.ndim != 3 or shifts.shape[-1] != 2:
        raise ValueError(f"shifts must be (B,N,2); got {tuple(shifts.shape)}")
    cdt = _compute_dtype(dtype)
    device = shifts.device
    return ForwardBurstPlan(
        scale=int(scale),
        dtype=cdt,
        device=device,
        blur=_build_blur_plan(psf, int(scale), device=device, dtype=cdt),
        block=_build_block_average_plan(shifts, int(scale), hr_shape, dtype=cdt),
    )


def _grouped_separable_blur(x: torch.Tensor, kernels: list[torch.Tensor]) -> torch.Tensor:
    """Apply a DIFFERENT separable Gaussian per sample in ONE pair of grouped conv1d launches.

    x: (n,1,H,W); kernels: list of n odd-length 1D tensors (the same 1D kernel is used for both
    axes, matching :func:`_blur_gauss_separable`). Smaller kernels are zero-padded to the group's
    common radius — exact, because the appended weights and the extra input padding are both zero.
    """
    n, _, H, W = x.shape
    dev, dt = x.device, x.dtype
    R = max(k.numel() // 2 for k in kernels)
    w1 = torch.zeros(n, 1, 2 * R + 1, 1, device=dev, dtype=dt)
    for i, k in enumerate(kernels):
        r = k.numel() // 2
        w1[i, 0, R - r:R + r + 1, 0] = k
    xv = x.reshape(1, n, H, W)
    xv = F.conv2d(F.pad(xv, (0, 0, R, R)), w1, groups=n)
    xv = F.conv2d(F.pad(xv, (R, R, 0, 0)), w1.view(n, 1, 1, 2 * R + 1), groups=n)
    return xv.reshape(n, 1, H, W)


def _grouped_separable_blur_planned(x: torch.Tensor, weight_v: torch.Tensor, weight_h: torch.Tensor) -> torch.Tensor:
    n, _, H, W = x.shape
    R = weight_v.shape[2] // 2
    xv = x.reshape(1, n, H, W)
    xv = F.conv2d(F.pad(xv, (0, 0, R, R)), weight_v, groups=n)
    xv = F.conv2d(F.pad(xv, (R, R, 0, 0)), weight_h, groups=n)
    return xv.reshape(n, 1, H, W)


def _grouped_kernel2d_blur(x: torch.Tensor, kernels: list[torch.Tensor]) -> torch.Tensor:
    """Apply a DIFFERENT 2D PSF per sample in ONE grouped conv2d launch.

    x: (n,1,H,W); kernels: list of n already-flipped square 2D tensors (flip matches
    :func:`_blur_kernel2d` / ndimage.convolve). Smaller kernels are zero-padded to the common
    radius (exact, same reasoning as the separable case)."""
    n, _, H, W = x.shape
    dev, dt = x.device, x.dtype
    R = max(k.shape[0] // 2 for k in kernels)
    wk = torch.zeros(n, 1, 2 * R + 1, 2 * R + 1, device=dev, dtype=dt)
    for i, k in enumerate(kernels):
        r = k.shape[0] // 2
        wk[i, 0, R - r:R + r + 1, R - r:R + r + 1] = k
    xv = x.reshape(1, n, H, W)
    xv = F.conv2d(F.pad(xv, (R, R, R, R)), wk, groups=n)
    return xv.reshape(n, 1, H, W)


def _grouped_kernel2d_blur_planned(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    n, _, H, W = x.shape
    R = weight.shape[2] // 2
    xv = x.reshape(1, n, H, W)
    xv = F.conv2d(F.pad(xv, (R, R, R, R)), weight, groups=n)
    return xv.reshape(n, 1, H, W)


def _blur_batch(xc: torch.Tensor, psf: ScenePSF, scale: int) -> torch.Tensor:
    """Per-sample PSF blur of (B,1,H,W) without a Python loop over the big-image convolution.

    Partitions the batch into the separable-Gaussian fast path and the general 2D-kernel path
    (each sample keeps EXACTLY the path :func:`_blur_one` would take, so numerics are unchanged),
    vectorizes each group with a grouped conv, then scatters back to the original order. Kernel
    construction stays a cheap per-sample loop; only the expensive convolution is batched."""
    B, _, H, W = xc.shape
    dev, dt = xc.device, xc.dtype
    sep_idx: list[int] = []; sep_k: list[torch.Tensor] = []
    ker_idx: list[int] = []; ker_k: list[torch.Tensor] = []
    for b in range(B):
        shape = psf.shape[b]
        sigma = float(psf.sigma_lr_px[b])
        sy = psf.sigma_y_lr_px[b]
        ang = float(psf.angle_deg[b])
        if shape == "gaussian" and sy is None and ang == 0.0:
            k1 = gaussian_kernel1d(sigma * scale, device=dev, dtype=dt)
            if k1 is None:                       # sigma<=0 -> identity (delta)
                k1 = torch.ones(1, device=dev, dtype=dt)
            sep_idx.append(b); sep_k.append(k1)
        else:
            k2 = make_psf_kernel2d(
                psf_sigma_lr_px=sigma, scale=scale, psf_shape=shape,
                psf_sigma_y_lr_px=sy, psf_angle_deg=ang, device=dev, dtype=dt,
            )
            if k2 is None:                       # zero-sigma -> identity (delta)
                k2 = torch.ones(1, 1, device=dev, dtype=dt)
            ker_idx.append(b); ker_k.append(torch.flip(k2, (0, 1)))
    parts: list[torch.Tensor] = []
    order: list[int] = []
    if sep_idx:
        parts.append(_grouped_separable_blur(xc[sep_idx], sep_k)); order += sep_idx
    if ker_idx:
        parts.append(_grouped_kernel2d_blur(xc[ker_idx], ker_k)); order += ker_idx
    blurred = torch.cat(parts, 0)
    perm = torch.argsort(torch.tensor(order, device=dev))
    return blurred.index_select(0, perm)         # back to original batch order (differentiable)


def _blur_batch_planned(xc: torch.Tensor, plan: _BlurPlan) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    if plan.sep_index is not None:
        assert plan.sep_weight_v is not None and plan.sep_weight_h is not None
        parts.append(
            _grouped_separable_blur_planned(
                xc.index_select(0, plan.sep_index),
                plan.sep_weight_v,
                plan.sep_weight_h,
            )
        )
    if plan.ker_index is not None:
        assert plan.ker_weight is not None
        parts.append(_grouped_kernel2d_blur_planned(xc.index_select(0, plan.ker_index), plan.ker_weight))
    blurred = torch.cat(parts, 0)
    return blurred.index_select(0, plan.restore_perm)


def _forward_burst_loop(x_hr: torch.Tensor, shifts: torch.Tensor, psf: ScenePSF, scale: int) -> torch.Tensor:
    """Reference per-sample forward (the certified path). Kept for fallback + equivalence tests."""
    B = x_hr.shape[0]
    in_dtype = x_hr.dtype
    cdt = _compute_dtype(in_dtype)
    xc = x_hr.to(cdt)
    outs = []
    for b in range(B):
        blurred = _blur_one(xc[b, 0], psf, b, scale)
        outs.append(block_average_shifted(blurred, shifts[b].to(cdt), scale))
    return torch.stack(outs, 0).to(in_dtype)


def _forward_burst_fast(x_hr: torch.Tensor, shifts: torch.Tensor, psf: ScenePSF, scale: int) -> torch.Tensor:
    """Vectorized forward: grouped-conv blur + batched block-average. ~Equivalent to the loop."""
    in_dtype = x_hr.dtype
    cdt = _compute_dtype(in_dtype)
    xc = x_hr.to(cdt)
    blurred = _blur_batch(xc, psf, scale)                       # (B,1,H,W)
    out = block_average_shifted_batched(blurred[:, 0], shifts.to(cdt), scale)
    return out.to(in_dtype)


def block_average_shifted_batched_planned(blurred: torch.Tensor, plan: _BlockAveragePlan) -> torch.Tensor:
    """Batched block-average using precomputed shift interpolation indices/weights."""
    B, H, W = blurred.shape
    if (H, W) != plan.hr_shape:
        raise ValueError(f"planned HR shape {plan.hr_shape} does not match blurred shape {(H, W)}")
    h, w = plan.out_shape
    N = plan.y0c.shape[1]
    be = blurred.unsqueeze(1).expand(B, N, H, W)
    y0e = plan.y0c.unsqueeze(-1).expand(B, N, plan.y0c.shape[2], W)
    y1e = plan.y1c.unsqueeze(-1).expand(B, N, plan.y1c.shape[2], W)
    rows = (be.gather(2, y0e) * (1.0 - plan.fy) + be.gather(2, y1e) * plan.fy) * plan.vy
    idx0 = plan.x0c.unsqueeze(2).expand(B, N, plan.y0c.shape[2], plan.x0c.shape[2])
    idx1 = plan.x1c.unsqueeze(2).expand(B, N, plan.y0c.shape[2], plan.x1c.shape[2])
    cols = (rows.gather(3, idx0) * (1.0 - plan.fx) + rows.gather(3, idx1) * plan.fx) * plan.vx
    s = plan.scale
    return cols.reshape(B, N, h, s, w, s).mean(dim=(3, 5))


def _forward_burst_planned(x_hr: torch.Tensor, plan: ForwardBurstPlan) -> torch.Tensor:
    in_dtype = x_hr.dtype
    cdt = _compute_dtype(in_dtype)
    if cdt != plan.dtype:
        raise ValueError(f"forward plan dtype {plan.dtype} does not match compute dtype {cdt}")
    if x_hr.device != plan.device:
        raise ValueError(f"forward plan device {plan.device} does not match x device {x_hr.device}")
    if tuple(x_hr.shape[-2:]) != plan.block.hr_shape:
        raise ValueError(f"forward plan HR shape {plan.block.hr_shape} does not match x shape {tuple(x_hr.shape[-2:])}")
    xc = x_hr.to(cdt)
    blurred = _blur_batch_planned(xc, plan.blur)
    out = block_average_shifted_batched_planned(blurred[:, 0], plan.block)
    return out.to(in_dtype)


def forward_burst(
    x_hr: torch.Tensor,
    shifts: torch.Tensor,
    psf: ScenePSF,
    scale: int,
    *,
    fast: bool | None = None,
    plan: ForwardBurstPlan | None = None,
) -> torch.Tensor:
    """A x:  (B,1,H,W) HR  ->  (B,N,h,w) LR burst, replicating physical_block_average per scene.

    Computes in fp32 for half-precision inputs (AMP safety) and in the input dtype otherwise
    (so fp64 inputs stay fp64 for tight certification). ``fast`` selects the vectorized path
    (default from TL_SOLVER_FAST_FORWARD); ``fast=False`` forces the reference per-sample loop.
    """
    if x_hr.ndim != 4 or x_hr.shape[1] != 1:
        raise ValueError(f"x_hr must be (B,1,H,W); got {tuple(x_hr.shape)}")
    if plan is not None:
        if int(scale) != plan.scale:
            raise ValueError(f"forward plan scale {plan.scale} does not match requested scale {scale}")
        return _forward_burst_planned(x_hr, plan)
    use_fast = _FAST_FORWARD_DEFAULT if fast is None else bool(fast)
    impl = _forward_burst_fast if use_fast else _forward_burst_loop
    return impl(x_hr, shifts, psf, scale)


def _highpass(t: torch.Tensor, sigma_lr_px: float) -> torch.Tensor:
    """Subtract a Gaussian-blurred version (per LR frame) to restrict the DC term to the SR band
    and reject smooth drift.  t: (B,N,h,w)."""
    if sigma_lr_px <= 0:
        return t
    B, N, h, w = t.shape
    in_dtype = t.dtype
    cdt = _compute_dtype(in_dtype)
    k = gaussian_kernel1d(sigma_lr_px, device=t.device, dtype=cdt)
    r = k.numel() // 2
    x = t.reshape(B * N, 1, h, w).to(cdt)
    lo = F.conv2d(F.pad(x, (0, 0, r, r), mode="reflect"), k.view(1, 1, -1, 1))
    lo = F.conv2d(F.pad(lo, (r, r, 0, 0), mode="reflect"), k.view(1, 1, 1, -1))
    return (x - lo).reshape(B, N, h, w).to(in_dtype)


_FAST_DCGRAD_DEFAULT = os.environ.get("TL_SOLVER_FAST_DCGRAD", "1") != "0"


def _dc_residual(x, y_burst, shifts, psf, scale, frame_mask, band, plan=None):
    """The masked, band-limited DC residual r = M ∘ H(Ax - y)."""
    r = forward_burst(x, shifts, psf, scale, plan=plan) - y_burst
    if band > 0:
        r = _highpass(r, band)
    if frame_mask is not None:
        r = r * frame_mask
    return r


class _DCGradLinearVJP(torch.autograd.Function):
    """g = A^T H^T M^2 H (A x - y) with an O(1)-order backward.

    The DC objective ½‖M H(Ax−y)‖² is QUADRATIC in x, so its gradient g is affine and the
    Jacobian J = dg/dx = A^T H^T M^2 H A is a fixed self-adjoint linear operator. Autograd's
    create_graph=True double-backward rebuilds a second-order graph through the whole forward to
    apply J — ~7x the cost of a single forward+adjoint (measured). Instead we provide the backward
    explicitly: J·v = ∇_v(½‖M H A v‖²), one FIRST-order autograd pass. Exact (no Huber). The forward
    still gets A^T from autograd (the certified adjoint); only the wasteful 2nd-order graph is gone.
    """

    @staticmethod
    def forward(ctx, x, y_burst, shifts, frame_mask, psf, scale, band, plan):
        with torch.enable_grad():
            xv = x.detach().requires_grad_(True)
            r = _dc_residual(xv, y_burst, shifts, psf, scale, frame_mask, band, plan)
            (g,) = torch.autograd.grad(0.5 * (r * r).sum(), xv)
        ctx.psf = psf; ctx.scale = int(scale); ctx.band = float(band)
        ctx.shifts = shifts; ctx.frame_mask = frame_mask; ctx.plan = plan
        return g.detach(), r.detach()

    @staticmethod
    def backward(ctx, grad_g, grad_r):
        # dL/dx = J^T grad_g = J grad_g (J self-adjoint) = ∇_v(½‖M H A v‖²) at v = grad_g.
        # grad_r is ignored: the residual output is detached (matches the eager path's r.detach()).
        with torch.enable_grad():
            v = grad_g.detach().requires_grad_(True)
            Av = forward_burst(v, ctx.shifts, ctx.psf, ctx.scale, plan=ctx.plan)
            if ctx.band > 0:
                Av = _highpass(Av, ctx.band)
            if ctx.frame_mask is not None:
                Av = Av * ctx.frame_mask
            (jv,) = torch.autograd.grad(0.5 * (Av * Av).sum(), v)
        return jv, None, None, None, None, None, None, None


def data_consistency_grad(
    x_hr: torch.Tensor,
    y_burst: torch.Tensor,
    shifts: torch.Tensor,
    psf: ScenePSF,
    scale: int,
    *,
    frame_mask: torch.Tensor | None = None,
    band_highpass_sigma_lr_px: float = 0.0,
    huber_delta: float | None = None,
    create_graph: bool = True,
    fast_vjp: bool | None = None,
    plan: ForwardBurstPlan | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(g, residual)`` where ``g = A^T(Ax - y)`` (exact, via autograd).

    ``g`` is the gradient of the (optionally Huber, optionally highpass-band) data-consistency
    objective and is suitable for an unrolled gradient/proximal step ``x <- x - eta*g``.
    Set ``create_graph=True`` during training so backprop flows through the unroll.

    Args:
        x_hr:     (B,1,H,W) current HR estimate.
        y_burst:  (B,N,h,w) observed LR burst (the SAVED burst — uses the SAVED shifts).
        frame_mask: optional (B,N,1,1) or (B,N,h,w) weights (e.g. 0 for padded frames / defects).
        band_highpass_sigma_lr_px: >0 restricts DC to the high-frequency band (rejects drift).
        huber_delta: if set, use a Huber data term (robust to hot/cold defects, stripe noise).
    """
    # Fast path: when we need a differentiable g (create_graph=True, i.e. training) and the
    # objective is the plain quadratic (no Huber), use the explicit self-adjoint VJP — same g, but
    # the backward is one first-order pass instead of a second-order graph (ACL-033). Env override:
    # TL_SOLVER_FAST_DCGRAD=0, or fast_vjp=False.
    use_fast_vjp = _FAST_DCGRAD_DEFAULT if fast_vjp is None else bool(fast_vjp)
    linear = huber_delta is None or huber_delta <= 0
    if create_graph and use_fast_vjp and linear:
        return _DCGradLinearVJP.apply(
            x_hr, y_burst, shifts, frame_mask, psf, scale, float(band_highpass_sigma_lr_px), plan
        )

    # Reference path (also used inside eval/inference wrappers decorated with torch.no_grad():
    # A^T(Ax-y) still needs a local graph w.r.t. x even when the caller wants no param gradients).
    with torch.enable_grad():
        x = x_hr if x_hr.requires_grad else x_hr.requires_grad_(True)
        Ax = forward_burst(x, shifts, psf, scale, plan=plan)
        r = Ax - y_burst
        if band_highpass_sigma_lr_px > 0:
            r = _highpass(r, band_highpass_sigma_lr_px)
        if frame_mask is not None:
            r = r * frame_mask
        if huber_delta is None or huber_delta <= 0:
            loss = 0.5 * (r * r).sum()
        else:
            a = r.abs()
            loss = torch.where(a <= huber_delta, 0.5 * r * r, huber_delta * (a - 0.5 * huber_delta)).sum()
        (g,) = torch.autograd.grad(loss, x, create_graph=create_graph)
    return g, r.detach()
