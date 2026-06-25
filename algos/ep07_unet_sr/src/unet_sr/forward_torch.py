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
from dataclasses import dataclass

import torch
import torch.nn.functional as F


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


def forward_burst(x_hr: torch.Tensor, shifts: torch.Tensor, psf: ScenePSF, scale: int) -> torch.Tensor:
    """A x:  (B,1,H,W) HR  ->  (B,N,h,w) LR burst, replicating physical_block_average per scene.

    Computes in fp32 for half-precision inputs (AMP safety) and in the input dtype otherwise
    (so fp64 inputs stay fp64 for tight certification).
    """
    if x_hr.ndim != 4 or x_hr.shape[1] != 1:
        raise ValueError(f"x_hr must be (B,1,H,W); got {tuple(x_hr.shape)}")
    B = x_hr.shape[0]
    in_dtype = x_hr.dtype
    cdt = _compute_dtype(in_dtype)
    xc = x_hr.to(cdt)
    outs = []
    for b in range(B):
        blurred = _blur_one(xc[b, 0], psf, b, scale)
        outs.append(block_average_shifted(blurred, shifts[b].to(cdt), scale))
    return torch.stack(outs, 0).to(in_dtype)


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
    x = x_hr if x_hr.requires_grad else x_hr.requires_grad_(True)
    Ax = forward_burst(x, shifts, psf, scale)
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
