"""Losses for EP12 drizzle-informed 4x thermal restoration."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sobel_edges(x: torch.Tensor) -> torch.Tensor:
    """Return Sobel gradient magnitude for a BCHW tensor."""

    if x.ndim != 4:
        raise ValueError("x must have shape (B, C, H, W)")
    kernel_x = x.new_tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]).view(1, 1, 3, 3)
    kernel_y = x.new_tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]).view(1, 1, 3, 3)
    channels = x.shape[1]
    gx = F.conv2d(x, kernel_x.repeat(channels, 1, 1, 1), padding=1, groups=channels)
    gy = F.conv2d(x, kernel_y.repeat(channels, 1, 1, 1), padding=1, groups=channels)
    return torch.sqrt(gx.square() + gy.square() + 1e-12)


def _gaussian_kernel_1d(size: int, sigma: float, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2.0
    kernel = torch.exp(-coords.square() / (2.0 * sigma * sigma))
    return kernel / kernel.sum().clamp_min(1e-12)


def _kernel_size_for_sigma(sigma: float) -> int:
    size = max(3, int(math.ceil(float(sigma) * 6.0)) | 1)
    return size


def gaussian_blur(x: torch.Tensor, sigma: float, kernel_size: int | None = None) -> torch.Tensor:
    """Differentiable Gaussian blur for BCHW tensors."""

    if x.ndim != 4:
        raise ValueError("x must have shape (B, C, H, W)")
    if sigma <= 0:
        return x
    size = _kernel_size_for_sigma(sigma) if kernel_size is None else int(kernel_size)
    if size % 2 == 0:
        size += 1
    pad = size // 2
    channels = x.shape[1]
    k1 = _gaussian_kernel_1d(size, sigma, x.device, x.dtype)
    ky = k1.view(1, 1, size, 1).repeat(channels, 1, 1, 1)
    kx = k1.view(1, 1, 1, size).repeat(channels, 1, 1, 1)
    out = F.conv2d(x, ky, padding=(pad, 0), groups=channels)
    return F.conv2d(out, kx, padding=(0, pad), groups=channels)


def _extract_prediction(
    pred: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(pred, tuple):
        if len(pred) != 2:
            raise ValueError("prediction tuple must be (temperature, log_var)")
        return pred[0], pred[1]
    if isinstance(pred, dict):
        value = pred.get("pred", pred.get("temperature"))
        if value is None:
            raise ValueError("prediction dict must contain pred or temperature")
        return value, pred.get("log_var")
    return pred, None


def _as_bchw(x: torch.Tensor, *, name: str) -> torch.Tensor:
    if x.ndim == 3:
        return x[:, None, :, :]
    if x.ndim != 4:
        raise ValueError(f"{name} must have shape (B,H,W) or (B,C,H,W)")
    return x


def _coverage_weight(coverage: torch.Tensor, gain: float) -> torch.Tensor:
    cov = _as_bchw(coverage, name="coverage").to(dtype=torch.float32).clamp_min(0.0)
    flat = cov.flatten(1)
    max_per_item = flat.amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    cov_norm = (cov / max_per_item).clamp(0.0, 1.0)
    return 1.0 + float(gain) * torch.sqrt(cov_norm)


def _coverage_inverse_weight(coverage: torch.Tensor, gain: float) -> torch.Tensor:
    """Inverse coverage weighting: low-coverage pixels get higher weight."""
    cov = _as_bchw(coverage, name="coverage").to(dtype=torch.float32).clamp_min(0.0)
    flat = cov.flatten(1)
    max_per_item = flat.amax(dim=1).view(-1, 1, 1, 1).clamp_min(1e-6)
    cov_norm = (cov / max_per_item).clamp(0.0, 1.0)
    return 1.0 + float(gain) * (1.0 - torch.sqrt(cov_norm))


def _weighted_mean(value: torch.Tensor, weight: torch.Tensor | None = None) -> torch.Tensor:
    if weight is None:
        return value.mean()
    w = weight.to(device=value.device, dtype=value.dtype)
    return (value * w).sum() / w.sum().clamp_min(1e-6)


class ForwardConsistencyLoss(nn.Module):
    """PSF-aware LR data-fidelity term.

    A valid 4x prediction should explain the observed drizzle mean after
    applying the calibrated PSF and averaging back to the LR detector grid.
    """

    def __init__(self, *, scale: int = 4, psf_sigma_lr_px: float = 0.25, eps: float = 1e-6) -> None:
        super().__init__()
        if scale <= 0:
            raise ValueError("scale must be positive")
        if psf_sigma_lr_px < 0:
            raise ValueError("psf_sigma_lr_px must be >= 0")
        self.scale = int(scale)
        self.psf_sigma_lr_px = float(psf_sigma_lr_px)
        self.eps = float(eps)
        sigma_hr = max(self.psf_sigma_lr_px * self.scale, 1e-6)
        size = _kernel_size_for_sigma(sigma_hr)
        coords = torch.arange(size, dtype=torch.float32) - (size - 1) / 2.0
        kernel_1d = torch.exp(-coords.square() / (2.0 * sigma_hr * sigma_hr))
        kernel_1d = kernel_1d / kernel_1d.sum().clamp_min(1e-12)
        kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
        self.register_buffer("psf_kernel", kernel_2d.view(1, 1, size, size), persistent=False)

    def forward(self, pred: torch.Tensor, drizzle_mean: torch.Tensor, coverage: torch.Tensor) -> torch.Tensor:
        if pred.ndim != 4:
            raise ValueError("pred must have shape (B,C,H,W)")
        drizzle = _as_bchw(drizzle_mean, name="drizzle_mean").to(device=pred.device, dtype=pred.dtype)
        cov = _as_bchw(coverage, name="coverage").to(device=pred.device, dtype=pred.dtype).clamp_min(0.0)
        if drizzle.shape != pred.shape or cov.shape != pred.shape:
            raise ValueError(f"forward consistency shape mismatch: pred={pred.shape}, drizzle={drizzle.shape}, cov={cov.shape}")
        if pred.shape[-2] % self.scale != 0 or pred.shape[-1] % self.scale != 0:
            raise ValueError("pred spatial shape must be divisible by scale")

        channels = pred.shape[1]
        kernel = self.psf_kernel.to(device=pred.device, dtype=pred.dtype).repeat(channels, 1, 1, 1)
        pad = kernel.shape[-1] // 2
        blurred = F.conv2d(pred, kernel, padding=pad, groups=channels)
        lr_pred = F.avg_pool2d(blurred, kernel_size=self.scale, stride=self.scale)

        lr_cov = F.avg_pool2d(cov, kernel_size=self.scale, stride=self.scale)
        lr_num = F.avg_pool2d(drizzle * cov, kernel_size=self.scale, stride=self.scale)
        lr_obs = lr_num / lr_cov.clamp_min(self.eps)

        error = torch.abs(lr_pred - lr_obs)
        return _weighted_mean(error, lr_cov)


class HeteroscedasticNLLLoss(nn.Module):
    """Gaussian NLL for confidence-aware temperature prediction."""

    def __init__(self, *, min_log_variance: float = -8.0, max_log_variance: float = 4.0) -> None:
        super().__init__()
        if min_log_variance >= max_log_variance:
            raise ValueError("min_log_variance must be < max_log_variance")
        self.min_log_variance = float(min_log_variance)
        self.max_log_variance = float(max_log_variance)

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        log_var: torch.Tensor,
        weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if pred.shape != target.shape or pred.shape != log_var.shape:
            raise ValueError(f"NLL shape mismatch: pred={pred.shape}, target={target.shape}, log_var={log_var.shape}")
        lv = log_var.clamp(self.min_log_variance, self.max_log_variance)
        nll = 0.5 * torch.exp(-lv) * (pred - target).square() + 0.5 * lv
        return _weighted_mean(nll, weight)


class ThermalSR4xLoss(nn.Module):
    """Frequency, coverage, forward-consistency, and confidence loss."""

    def __init__(
        self,
        *,
        sigma_lf: float = 8.0,
        lf_weight: float = 1.0,
        hf_weight: float = 0.3,
        edge_weight: float = 0.1,
        forward_weight: float = 0.2,
        nll_weight: float = 0.05,
        coverage_gain: float = 4.0,
        edge_mask_boost: float = 2.0,
        edge_coarse_weight: float = 0.25,
        hf_detail_weight: float = 0.3,
        hf_detail_gain: float = 4.0,
        scale: int = 4,
        psf_sigma_lr_px: float = 0.25,
        min_log_variance: float = -8.0,
        max_log_variance: float = 4.0,
    ) -> None:
        super().__init__()
        for name, value in {
            "lf_weight": lf_weight,
            "hf_weight": hf_weight,
            "edge_weight": edge_weight,
            "forward_weight": forward_weight,
            "nll_weight": nll_weight,
            "coverage_gain": coverage_gain,
            "edge_mask_boost": edge_mask_boost,
            "edge_coarse_weight": edge_coarse_weight,
            "hf_detail_weight": hf_detail_weight,
            "hf_detail_gain": hf_detail_gain,
        }.items():
            if value < 0:
                raise ValueError(f"{name} must be >= 0")
        if sigma_lf <= 0:
            raise ValueError("sigma_lf must be positive")
        self.sigma_lf = float(sigma_lf)
        self.lf_weight = float(lf_weight)
        self.hf_weight = float(hf_weight)
        self.edge_weight = float(edge_weight)
        self.forward_weight = float(forward_weight)
        self.nll_weight = float(nll_weight)
        self.coverage_gain = float(coverage_gain)
        self.edge_mask_boost = float(edge_mask_boost)
        self.edge_coarse_weight = float(edge_coarse_weight)
        self.hf_detail_weight = float(hf_detail_weight)
        self.hf_detail_gain = float(hf_detail_gain)
        self.forward_consistency = ForwardConsistencyLoss(scale=scale, psf_sigma_lr_px=psf_sigma_lr_px)
        self.nll = HeteroscedasticNLLLoss(
            min_log_variance=min_log_variance,
            max_log_variance=max_log_variance,
        )

    def forward(
        self,
        pred: torch.Tensor | tuple[torch.Tensor, torch.Tensor] | dict[str, torch.Tensor],
        target: torch.Tensor,
        *,
        edge_mask: torch.Tensor | None = None,
        coverage_4x: torch.Tensor | None = None,
        drizzle_mean_4x: torch.Tensor | None = None,
        log_var: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        pred_tensor, pred_log_var = _extract_prediction(pred)
        if log_var is None:
            log_var = pred_log_var
        if pred_tensor.shape != target.shape:
            raise ValueError(f"pred and target shape mismatch: {pred_tensor.shape} vs {target.shape}")

        cov_weight = _coverage_weight(coverage_4x, self.coverage_gain).to(
            device=pred_tensor.device,
            dtype=pred_tensor.dtype,
        ) if coverage_4x is not None else None

        pred_lf = gaussian_blur(pred_tensor, self.sigma_lf)
        target_lf = gaussian_blur(target, self.sigma_lf)
        lf = F.l1_loss(pred_lf, target_lf)

        pred_hf = pred_tensor - pred_lf
        target_hf = target - target_lf
        hf = _weighted_mean(torch.abs(pred_hf - target_hf), cov_weight)

        pred_edges = sobel_edges(pred_tensor)
        target_edges = sobel_edges(target)
        edge_error = torch.abs(pred_edges - target_edges)
        if edge_mask is not None:
            mask = _as_bchw(edge_mask, name="edge_mask").to(device=pred_tensor.device, dtype=pred_tensor.dtype)
            if mask.shape != edge_error.shape:
                raise ValueError(f"edge_mask shape mismatch: {mask.shape} vs {edge_error.shape}")
            edge_w = 1.0 + self.edge_mask_boost * mask
        else:
            edge_w = None
        edge_fine = _weighted_mean(edge_error, edge_w)

        pred_2x = F.avg_pool2d(pred_tensor, kernel_size=2, stride=2)
        target_2x = F.avg_pool2d(target, kernel_size=2, stride=2)
        edge_coarse = torch.abs(sobel_edges(pred_2x) - sobel_edges(target_2x)).mean()
        edge = edge_fine + self.edge_coarse_weight * edge_coarse

        zero = pred_tensor.new_zeros(())

        hf_detail = zero
        if self.hf_detail_weight > 0 and coverage_4x is not None:
            inv_weight = _coverage_inverse_weight(coverage_4x, self.hf_detail_gain).to(
                device=pred_tensor.device, dtype=pred_tensor.dtype,
            )
            hf_detail = _weighted_mean(torch.abs(pred_hf - target_hf), inv_weight)

        forward = zero
        if self.forward_weight > 0:
            if coverage_4x is None or drizzle_mean_4x is None:
                raise ValueError("forward consistency requires coverage_4x and drizzle_mean_4x")
            forward = self.forward_consistency(pred_tensor, drizzle_mean_4x, coverage_4x)

        nll = zero
        if self.nll_weight > 0:
            if log_var is None:
                raise ValueError("nll_weight > 0 requires log_var from the model")
            nll = self.nll(pred_tensor, target, log_var, weight=cov_weight)

        total = (
            self.lf_weight * lf
            + self.hf_weight * hf
            + self.edge_weight * edge
            + self.forward_weight * forward
            + self.nll_weight * nll
            + self.hf_detail_weight * hf_detail
        )
        return {
            "total": total,
            "lf": lf,
            "hf": hf,
            "edge": edge,
            "forward": forward,
            "nll": nll,
            "hf_detail": hf_detail,
        }
