"""Loss functions for thermal SR regression."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sobel edge gradient
# ---------------------------------------------------------------------------

def sobel_edges_xy(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Return Sobel gradient components (gx, gy) for a BCHW tensor.

    Returns
    -------
    gx, gy : each has shape (B, C, H, W)
    """

    if x.ndim != 4:
        raise ValueError("x must have shape (B, C, H, W)")
    kernel_x = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    kernel_y = x.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    channels = x.shape[1]
    gx = F.conv2d(x, kernel_x.repeat(channels, 1, 1, 1), padding=1, groups=channels)
    gy = F.conv2d(x, kernel_y.repeat(channels, 1, 1, 1), padding=1, groups=channels)
    return gx, gy


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Return Sobel gradient magnitude for a BCHW tensor."""
    gx, gy = sobel_edges_xy(x)
    return torch.sqrt(gx.square() + gy.square() + 1e-12)


def laplacian(x: torch.Tensor) -> torch.Tensor:
    """Return a 4-neighbor Laplacian response for a BCHW tensor."""

    if x.ndim != 4:
        raise ValueError("x must have shape (B, C, H, W)")
    kernel = x.new_tensor([[0.0, 1.0, 0.0], [1.0, -4.0, 1.0], [0.0, 1.0, 0.0]]).view(1, 1, 3, 3)
    channels = x.shape[1]
    return F.conv2d(x, kernel.repeat(channels, 1, 1, 1), padding=1, groups=channels)


def asymmetric_laplacian_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Penalize places where pred is less Laplacian-sharp than target."""

    pred_lap = torch.abs(laplacian(pred))
    target_lap = torch.abs(laplacian(target))
    return F.relu(target_lap - pred_lap).mean()


def grad_vector_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    structure_boost: float = 4.0,
    boundary_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gradient vector matching loss.

    Compares full Sobel gradient vectors (gx, gy) between *pred* and *target*,
    not just their magnitudes.  This is strictly more sensitive than
    magnitude-only comparison:

    - **Thickening**: edge shifts outward → gradient direction changes →
      vector difference is large even when magnitudes stay the same.
    - **Merging**: gradients between structures disappear → vector difference
      captures the missing gradient.
    - **Disconnection**: gradients at break points disappear → same as merging.
    - **Hallucination**: spurious gradients appear in pred → vector difference
      penalises them.

    The loss is weighted by the target gradient magnitude so that
    structure-rich regions dominate the loss signal.
    """

    gx_pred, gy_pred = sobel_edges_xy(pred)
    gx_target, gy_target = sobel_edges_xy(target)

    # Vector L1 difference (more stable than L2 for sparse edges)
    vec_error = torch.abs(gx_pred - gx_target) + torch.abs(gy_pred - gy_target)

    # Weight by target gradient magnitude — focuses on structure regions
    target_mag = torch.sqrt(gx_target.square() + gy_target.square() + 1e-12)
    mag_max = target_mag.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
    mag_norm = target_mag / mag_max  # [0, 1]
    weight_map = 1.0 + structure_boost * mag_norm
    if boundary_weight is not None:
        if boundary_weight.shape != target.shape:
            raise ValueError(f"boundary_weight shape mismatch: {boundary_weight.shape} vs {target.shape}")
        weight_map = weight_map * boundary_weight.to(dtype=weight_map.dtype)

    return (vec_error * weight_map).mean()


# ---------------------------------------------------------------------------
# Differentiable SSIM
# ---------------------------------------------------------------------------

def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """1D Gaussian kernel normalised to sum=1."""
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    g = torch.exp(-coords.square() / (2.0 * sigma * sigma))
    return g / g.sum()


def _gaussian_kernel_2d(size: int, sigma: float, channels: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """2D separable Gaussian kernel with shape (channels, 1, size, size)."""
    k1d = _gaussian_kernel_1d(size, sigma, device, dtype)
    k2d = k1d.unsqueeze(1) * k1d.unsqueeze(0)  # outer product
    return k2d.expand(channels, 1, size, size).contiguous()


def _ssim_float32(
    pred_f: torch.Tensor,
    target_f: torch.Tensor,
    *,
    window_size: int,
    sigma: float,
    data_range: float | None,
    eps: float,
) -> torch.Tensor:
    """Gaussian-window SSIM core in float32 (AMP-safe statistics)."""

    channels = pred_f.shape[1]
    kernel = _gaussian_kernel_2d(window_size, sigma, channels, pred_f.device, pred_f.dtype)
    if data_range is None:
        data_range = float(target_f.max() - target_f.min()) + eps

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    pad = window_size // 2
    mu_p = F.conv2d(pred_f, kernel, padding=pad, groups=channels)
    mu_t = F.conv2d(target_f, kernel, padding=pad, groups=channels)

    mu_p_sq = mu_p * mu_p
    mu_t_sq = mu_t * mu_t
    mu_pt = mu_p * mu_t

    sigma_p_sq = F.conv2d(pred_f * pred_f, kernel, padding=pad, groups=channels) - mu_p_sq
    sigma_t_sq = F.conv2d(target_f * target_f, kernel, padding=pad, groups=channels) - mu_t_sq
    sigma_pt = F.conv2d(pred_f * target_f, kernel, padding=pad, groups=channels) - mu_pt

    sigma_p_sq = sigma_p_sq.clamp(min=0.0)
    sigma_t_sq = sigma_t_sq.clamp(min=0.0)

    numerator = (2.0 * mu_pt + c1) * (2.0 * sigma_pt + c2)
    denominator = (mu_p_sq + mu_t_sq + c1) * (sigma_p_sq + sigma_t_sq + c2)
    return (numerator / (denominator + eps)).mean()


@torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
def _ssim_cuda(
    pred: torch.Tensor,
    target: torch.Tensor,
    window_size: int,
    sigma: float,
    data_range: float,
    eps: float,
) -> torch.Tensor:
    """CUDA SSIM forward: inputs cast once to fp32 without nested autocast(False)."""

    dr = None if data_range < 0 else data_range
    return _ssim_float32(
        pred,
        target,
        window_size=window_size,
        sigma=sigma,
        data_range=dr,
        eps=eps,
    )


def ssim(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    window_size: int = 11,
    sigma: float = 1.5,
    data_range: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Compute mean SSIM between pred and target (BCHW tensors).

    Returns a scalar in [0, 1] where 1 = identical.
    """

    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {pred.shape} vs {target.shape}")
    if pred.ndim != 4:
        raise ValueError("input must be (B, C, H, W)")

    dr_flag = -1.0 if data_range is None else float(data_range)
    if pred.is_cuda:
        return _ssim_cuda(pred, target, window_size, sigma, dr_flag, eps)

    return _ssim_float32(
        pred.float(),
        target.float(),
        window_size=window_size,
        sigma=sigma,
        data_range=data_range,
        eps=eps,
    )


# ---------------------------------------------------------------------------
# Combined loss
# ---------------------------------------------------------------------------

class ThermalSRLoss(nn.Module):
    def __init__(
        self,
        edge_weight: float = 0.1,
        edge_mask_boost: float = 2.0,
        ssim_weight: float = 0.1,
    ) -> None:
        super().__init__()
        if edge_weight < 0:
            raise ValueError("edge_weight must be >= 0")
        if edge_mask_boost < 0:
            raise ValueError("edge_mask_boost must be >= 0")
        if ssim_weight < 0:
            raise ValueError("ssim_weight must be >= 0")
        self.edge_weight = float(edge_weight)
        self.edge_mask_boost = float(edge_mask_boost)
        self.ssim_weight = float(ssim_weight)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        edge_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if pred.shape != target.shape:
            raise ValueError(f"pred and target shape mismatch: {pred.shape} vs {target.shape}")

        mse = F.mse_loss(pred, target)

        # Sobel edge loss
        pred_edges = sobel_edges(pred)
        target_edges = sobel_edges(target)
        edge_error = torch.abs(pred_edges - target_edges)
        if edge_mask is not None:
            if edge_mask.shape != target.shape:
                raise ValueError(f"edge_mask shape mismatch: {edge_mask.shape} vs {target.shape}")
            weights = 1.0 + self.edge_mask_boost * edge_mask.to(dtype=edge_error.dtype)
            edge = (edge_error * weights).mean()
        else:
            edge = edge_error.mean()

        # SSIM loss (1 - SSIM so that lower is better)
        ssim_val = ssim(pred, target)
        ssim_loss = 1.0 - ssim_val

        total = mse + self.edge_weight * edge + self.ssim_weight * ssim_loss
        return {"total": total, "mse": mse, "edge": edge, "ssim": ssim_loss}


# ---------------------------------------------------------------------------
# Gaussian blur for highpass computation
# ---------------------------------------------------------------------------

def _gaussian_blur_2d_float32(x_f: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur core in float32."""

    ks = int(4 * sigma + 0.5) * 2 + 1
    coords = torch.arange(ks, device=x_f.device, dtype=torch.float32) - (ks - 1) / 2.0
    g = torch.exp(-coords.square() / (2.0 * sigma * sigma))
    g = g / g.sum()
    c = x_f.shape[1]
    pad = ks // 2
    k_h = g.view(1, 1, 1, -1).expand(c, 1, 1, ks).contiguous()
    k_v = g.view(1, 1, -1, 1).expand(c, 1, ks, 1).contiguous()
    out = F.conv2d(F.pad(x_f, (pad, pad, 0, 0), mode="reflect"), k_h, groups=c)
    return F.conv2d(F.pad(out, (0, 0, pad, pad), mode="reflect"), k_v, groups=c)


@torch.amp.custom_fwd(cast_inputs=torch.float32, device_type="cuda")
def _gaussian_blur_2d_cuda(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """CUDA blur forward: single fp32 cast, no nested autocast(False) bubble."""

    return _gaussian_blur_2d_float32(x, sigma)


def gaussian_blur_2d(x: torch.Tensor, sigma: float) -> torch.Tensor:
    """Separable Gaussian blur for BCHW tensors. AMP-safe via custom_fwd fp32 ops."""

    if sigma <= 0:
        return x
    if x.is_cuda:
        return _gaussian_blur_2d_cuda(x, sigma)
    return _gaussian_blur_2d_float32(x.float(), sigma)


def forward_model_loss(
    pred: torch.Tensor,
    lr_observation: torch.Tensor,
    *,
    scale: int = 2,
    psf_sigma_lr_px: float = 0.5,
    band: str = "full",
    band_sigma_lr_px: float = 5.0,
) -> torch.Tensor:
    """Blur HR prediction by the PSF and block-average to the LR observation.

    When *band* is ``"highpass"``, both the downsampled prediction and the
    observation are highpass-filtered (subtract Gaussian blur at
    *band_sigma_lr_px*) before computing MSE.  This restricts the consistency
    constraint to the high-frequency band where edge brightness and width live,
    avoiding the low-frequency gradient conflict that caused oscillation
    plateaus with full-band forward model (ACL-005).
    """

    if pred.ndim != 4 or lr_observation.ndim != 4:
        raise ValueError("pred and lr_observation must have shape (B, C, H, W)")
    s = int(scale)
    if s <= 0:
        raise ValueError("scale must be > 0")
    if pred.shape[-2] % s != 0 or pred.shape[-1] % s != 0:
        raise ValueError("pred spatial shape must be divisible by scale")
    if band not in ("full", "highpass"):
        raise ValueError(f"band must be 'full' or 'highpass', got {band!r}")
    blurred = gaussian_blur_2d(pred, float(psf_sigma_lr_px) * s)
    down = F.avg_pool2d(blurred, kernel_size=s, stride=s)
    if down.shape != lr_observation.shape:
        raise ValueError(f"forward-model shape mismatch: {down.shape} vs {lr_observation.shape}")
    if band == "highpass":
        sigma = float(band_sigma_lr_px)
        down = down - gaussian_blur_2d(down, sigma)
        lr_observation = lr_observation - gaussian_blur_2d(lr_observation, sigma)
    return F.mse_loss(down, lr_observation)


# ---------------------------------------------------------------------------
# Fourier band filter for band-gated supervision (Stage 2a E3 / roadmap Step 3)
# ---------------------------------------------------------------------------

def fourier_band_filter(
    x: torch.Tensor,
    *,
    period_lo_px: float,
    period_hi_px: float,
    edge_softness_cyc: float = 0.05,
) -> torch.Tensor:
    """Radial band-pass a BCHW tensor, keeping spatial periods in [lo, hi] px.

    Implemented as an exact Fourier radial mask with raised-cosine edges
    (width *edge_softness_cyc* cycles/px) rather than a DoG: the target band
    (25-40 um on the 10 um/sample HR grid = 2.5-4.0 px periods) sits close to
    Nyquist (2 px), where DoG sigmas become sub-pixel and the response is hard
    to control; the FFT mask is exact, differentiable, and cheap at patch size.
    The mask depends only on shape/device/dtype, so it is rebuilt per call from
    cached fftfreq grids — negligible next to the model's conv stack.
    """

    if x.ndim != 4:
        raise ValueError("x must have shape (B, C, H, W)")
    lo = float(period_lo_px)
    hi = float(period_hi_px)
    if not (0 < lo < hi):
        raise ValueError(f"need 0 < period_lo_px < period_hi_px, got {lo}, {hi}")
    f_hi = 1.0 / lo  # upper frequency edge (short-period side)
    f_lo = 1.0 / hi  # lower frequency edge (long-period side)
    soft = float(edge_softness_cyc)

    h, w = x.shape[-2], x.shape[-1]
    fy = torch.fft.fftfreq(h, d=1.0, device=x.device, dtype=torch.float32)
    fx = torch.fft.rfftfreq(w, d=1.0, device=x.device, dtype=torch.float32)
    radius = torch.sqrt(fy[:, None] ** 2 + fx[None, :] ** 2)

    def _rise(f: torch.Tensor, edge: float) -> torch.Tensor:
        # 0 below edge-soft, 1 above edge, raised-cosine in between.
        t = ((f - (edge - soft)) / max(soft, 1e-9)).clamp(0.0, 1.0)
        return 0.5 - 0.5 * torch.cos(torch.pi * t)

    weight = _rise(radius, f_lo) * (1.0 - _rise(radius, f_hi + soft))
    spec = torch.fft.rfft2(x.float())
    out = torch.fft.irfft2(spec * weight, s=(h, w))
    return out.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# Contour-focused loss for structure/edge reconstruction
# ---------------------------------------------------------------------------

class ContourSRLoss(nn.Module):
    """Loss for contour-level SR: focuses on structure/edge reconstruction.

    Strips DC background via highpass and weights structure pixels by the
    product of two complementary maps: a **signal-gradient weight** from the
    target's Sobel magnitude (where thermal contrast lives) and an optional
    **geometry boundary weight** passed in from the mask (where physical edges
    live, contrast-independently — see ``mask_weights.compute_boundary_weight_np``).
    The boundary weight replaces the old thin-structure / narrow-gap masks,
    which baked in a perfect-line / perfect-rectangle prior unsuited to the v4
    defect data (holes, cracks, broken edges).

    Components
    ----------
    mse            : MSE on raw pred vs target — global DC / temperature anchor.
    highpass L1    : L1 on highpass(pred) vs highpass(target), structure×boundary
                     weighted.
    Sobel edge mag : L1 on gradient magnitude (multi-scale: 1× + 2×-downsampled).
    SSIM           : structural similarity (secondary).
    grad_vector    : Full Sobel gradient *vector* (gx, gy) matching, weighted by
                     target gradient magnitude.  Catches thickening (direction
                     shifts), merging (inter-structure gradients vanish), and
                     disconnection (intra-structure gradients vanish).
    flatness       : Penalises |grad(pred)| where the GT is flat (isothermal
                     interiors + background); encodes the near-isothermal prior.
                     Default off (weight 0); enable for v4 data.
    laplacian /
    forward_model  : Optional hybrid terms kept disabled by default and enabled
                     explicitly for v6/v8-style experiments.
    """

    def __init__(
        self,
        highpass_weight: float = 1.0,
        highpass_sigma: float = 5.0,
        edge_weight: float = 0.05,
        ssim_weight: float = 0.15,
        mse_weight: float = 0.2,
        structure_boost: float = 4.0,
        edge_coarse_weight: float = 0.25,
        grad_vector_weight: float = 0.3,
        flatness_weight: float = 0.0,
        flatness_tau: float = 0.25,
        laplacian_weight: float = 0.0,
        forward_model_weight: float = 0.0,
        forward_model_psf_sigma: float = 1.0,
        forward_model_scale: int = 2,
        forward_model_band: str = "full",
        forward_model_band_sigma: float = 5.0,
        band_loss_weight: float = 0.0,
        band_period_lo_px: float = 2.5,
        band_period_hi_px: float = 4.0,
    ) -> None:
        super().__init__()
        for name, val in [
            ("highpass_weight", highpass_weight),
            ("highpass_sigma", highpass_sigma),
            ("edge_weight", edge_weight),
            ("ssim_weight", ssim_weight),
            ("mse_weight", mse_weight),
            ("structure_boost", structure_boost),
            ("edge_coarse_weight", edge_coarse_weight),
            ("grad_vector_weight", grad_vector_weight),
            ("flatness_weight", flatness_weight),
            ("laplacian_weight", laplacian_weight),
            ("forward_model_weight", forward_model_weight),
            ("forward_model_psf_sigma", forward_model_psf_sigma),
            ("forward_model_band_sigma", forward_model_band_sigma),
            ("band_loss_weight", band_loss_weight),
        ]:
            if val < 0:
                raise ValueError(f"{name} must be >= 0")
        if not (0 < float(band_period_lo_px) < float(band_period_hi_px)):
            raise ValueError(
                f"need 0 < band_period_lo_px < band_period_hi_px, got {band_period_lo_px}, {band_period_hi_px}"
            )
        if float(flatness_tau) <= 0:
            raise ValueError("flatness_tau must be > 0")
        if int(forward_model_scale) <= 0:
            raise ValueError("forward_model_scale must be > 0")
        if forward_model_band not in ("full", "highpass"):
            raise ValueError(f"forward_model_band must be 'full' or 'highpass', got {forward_model_band!r}")
        self.highpass_weight = float(highpass_weight)
        self.highpass_sigma = float(highpass_sigma)
        self.edge_weight = float(edge_weight)
        self.ssim_weight = float(ssim_weight)
        self.mse_weight = float(mse_weight)
        self.structure_boost = float(structure_boost)
        self.edge_coarse_weight = float(edge_coarse_weight)
        self.grad_vector_weight = float(grad_vector_weight)
        self.flatness_weight = float(flatness_weight)
        self.flatness_tau = float(flatness_tau)
        self.laplacian_weight = float(laplacian_weight)
        self.forward_model_weight = float(forward_model_weight)
        self.forward_model_psf_sigma = float(forward_model_psf_sigma)
        self.forward_model_scale = int(forward_model_scale)
        self.forward_model_band = str(forward_model_band)
        self.forward_model_band_sigma = float(forward_model_band_sigma)
        self.band_loss_weight = float(band_loss_weight)
        self.band_period_lo_px = float(band_period_lo_px)
        self.band_period_hi_px = float(band_period_hi_px)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        lr_observation: torch.Tensor | None = None,
        lr_obs: torch.Tensor | None = None,
        boundary_weight: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if pred.shape != target.shape:
            raise ValueError(f"pred/target shape mismatch: {pred.shape} vs {target.shape}")
        if boundary_weight is not None and boundary_weight.shape != target.shape:
            raise ValueError(f"boundary_weight shape mismatch: {boundary_weight.shape} vs {target.shape}")

        # ---- MSE on raw prediction — global DC / temperature-level anchor ----
        mse_loss = (pred - target).square().mean()

        # ---- Highpass: strip smooth DC background ----
        pred_hp = pred - gaussian_blur_2d(pred, self.highpass_sigma)
        target_hp = target - gaussian_blur_2d(target, self.highpass_sigma)

        # ---- Structure weight: signal-gradient (target) × geometry boundary ----
        hp_error = torch.abs(pred_hp - target_hp)
        target_edges = sobel_edges(target)
        edge_max = target_edges.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-8)
        edge_norm = target_edges / edge_max
        weight_map = 1.0 + self.structure_boost * edge_norm
        if boundary_weight is not None:
            weight_map = weight_map * boundary_weight.to(dtype=weight_map.dtype)
        hp_loss = (hp_error * weight_map).mean()

        # ---- Multi-scale Sobel edge magnitude loss ----
        pred_edges = sobel_edges(pred)
        edge_loss_fine = torch.abs(pred_edges - target_edges).mean()

        pred_2x = F.avg_pool2d(pred, kernel_size=2, stride=2)
        target_2x = F.avg_pool2d(target, kernel_size=2, stride=2)
        pred_edges_2x = sobel_edges(pred_2x)
        target_edges_2x = sobel_edges(target_2x)
        edge_loss_coarse = torch.abs(pred_edges_2x - target_edges_2x).mean()

        edge_loss = edge_loss_fine + self.edge_coarse_weight * edge_loss_coarse

        # ---- SSIM (structural similarity) ----
        ssim_val = ssim(pred, target)
        ssim_loss = 1.0 - ssim_val

        # ---- Gradient vector matching ----
        gv_loss = pred.new_tensor(0.0)
        if self.grad_vector_weight > 0:
            gv_loss = grad_vector_loss(pred, target, self.structure_boost, boundary_weight=boundary_weight)

        # ---- Isothermal flatness: suppress predicted texture where GT is flat ----
        # Penalise |grad(pred)| weighted by a soft "is-flat" mask from the TARGET
        # gradient (~1 in isothermal interiors & background, ->0 on real edges).
        # edge_norm is contrast-normalised, so this never fights a true boundary
        # even at tiny ΔT; it counters the structure-weighting's neglect of flats.
        flat_loss = pred.new_tensor(0.0)
        if self.flatness_weight > 0:
            flat_mask = torch.exp(-((edge_norm / self.flatness_tau) ** 2))
            flat_loss = (sobel_edges(pred) * flat_mask).mean()

        # ---- Optional hybrid terms from v6_physics ----
        lap_loss = pred.new_tensor(0.0)
        if self.laplacian_weight > 0:
            lap_loss = asymmetric_laplacian_loss(pred, target)

        fm_loss = pred.new_tensor(0.0)
        if self.forward_model_weight > 0:
            observation = lr_obs if lr_obs is not None else lr_observation
            if observation is None:
                raise ValueError("lr_observation or lr_obs is required when forward_model_weight > 0")
            forward_scale = 2 if lr_obs is not None else self.forward_model_scale
            fm_loss = forward_model_loss(
                pred,
                observation,
                scale=forward_scale,
                psf_sigma_lr_px=self.forward_model_psf_sigma,
                band=self.forward_model_band,
                band_sigma_lr_px=self.forward_model_band_sigma,
            )

        # ---- Band-gated supervision (Stage 2a E3): L1 on the residual restricted
        # to the measured recoverable band (25-40 um periods = 2.5-4.0 HR px at
        # scale 2, EP15 M2 authoritative cutoff 25.45 um, ACL-048). Additive on
        # top of the existing terms — spends extra gradient budget exactly where
        # real information exists, without touching the full-band anchors.
        band_loss = pred.new_tensor(0.0)
        if self.band_loss_weight > 0:
            band_residual = fourier_band_filter(
                pred - target,
                period_lo_px=self.band_period_lo_px,
                period_hi_px=self.band_period_hi_px,
            )
            band_loss = torch.abs(band_residual).mean()

        total = (
            self.mse_weight * mse_loss
            + self.highpass_weight * hp_loss
            + self.edge_weight * edge_loss
            + self.ssim_weight * ssim_loss
            + self.grad_vector_weight * gv_loss
            + self.flatness_weight * flat_loss
            + self.laplacian_weight * lap_loss
            + self.forward_model_weight * fm_loss
            + self.band_loss_weight * band_loss
        )
        out = {
            "total": total,
            "mse": mse_loss,
            "highpass": hp_loss,
            "edge": edge_loss,
            "ssim": ssim_loss,
            "grad_vector": gv_loss,
        }
        if self.flatness_weight > 0:
            out["flatness"] = flat_loss
        if self.laplacian_weight > 0:
            out["laplacian"] = lap_loss
        if self.forward_model_weight > 0:
            out["forward_model"] = fm_loss
        if self.band_loss_weight > 0:
            out["band"] = band_loss
        return out
