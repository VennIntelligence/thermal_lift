"""PyTorch EP08 forward/adjoint operator matching EP06 numerics."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _validate_scale(scale: int) -> int:
    scale = int(scale)
    if scale != 2:
        raise ValueError("EP08 forward model mirrors EP06 and is defined for scale=2 only")
    return scale


def _sigma_hr(psf_sigma: float, scale: int) -> float:
    return max(0.0, float(psf_sigma) * float(scale))


def _as_tensor(value: torch.Tensor | object, *, like: torch.Tensor | None = None) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    dtype = torch.float64 if like is None else like.dtype
    device = None if like is None else like.device
    return torch.as_tensor(value, dtype=dtype, device=device)


def _gaussian_kernel1d(sigma: float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    radius = int(4.0 * float(sigma) + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    phi = torch.exp(-0.5 / (float(sigma) * float(sigma)) * x.square())
    return phi / phi.sum()


def _pad_for_mode(x: torch.Tensor, padding: tuple[int, int, int, int], mode: str) -> torch.Tensor:
    if mode == "constant":
        return F.pad(x, padding, mode="constant", value=0.0)
    if mode == "nearest":
        return F.pad(x, padding, mode="replicate")
    raise ValueError("mode must be 'constant' or 'nearest'")


def gaussian_filter2d(image: torch.Tensor, sigma: float, *, mode: str = "constant") -> torch.Tensor:
    """Separable 2D Gaussian filter using SciPy's default truncate=4.0."""

    if sigma <= 0:
        return image
    if image.ndim != 2:
        raise ValueError("image must be 2D")
    kernel = _gaussian_kernel1d(float(sigma), dtype=image.dtype, device=image.device)
    radius = kernel.numel() // 2
    x = image[None, None]
    ky = kernel.view(1, 1, -1, 1)
    kx = kernel.view(1, 1, 1, -1)
    x = F.conv2d(_pad_for_mode(x, (0, 0, radius, radius), mode), ky)
    x = F.conv2d(_pad_for_mode(x, (radius, radius, 0, 0), mode), kx)
    return x[0, 0]


def _sample_reference_to_lr(image_hr: torch.Tensor, shift: torch.Tensor, *, scale: int) -> torch.Tensor:
    h_hr, w_hr = image_hr.shape
    h_lr, w_lr = h_hr // scale, w_hr // scale
    dx, dy = shift.to(dtype=image_hr.dtype, device=image_hr.device)

    y = scale * (torch.arange(h_lr, dtype=image_hr.dtype, device=image_hr.device) + dy)
    x = scale * (torch.arange(w_lr, dtype=image_hr.dtype, device=image_hr.device) + dx)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    y0 = torch.floor(yy).to(torch.long)
    x0 = torch.floor(xx).to(torch.long)
    fy = yy - y0.to(image_hr.dtype)
    fx = xx - x0.to(image_hr.dtype)

    inside = (yy >= 0) & (yy <= h_hr - 1) & (xx >= 0) & (xx <= w_hr - 1)
    out = torch.zeros((h_lr, w_lr), dtype=image_hr.dtype, device=image_hr.device)
    for oy in (0, 1):
        yi = y0 + oy
        wy = (1.0 - fy) if oy == 0 else fy
        valid_y = (yi >= 0) & (yi < h_hr)
        yi_safe = yi.clamp(0, h_hr - 1)
        for ox in (0, 1):
            xi = x0 + ox
            wx = (1.0 - fx) if ox == 0 else fx
            valid = inside & valid_y & (xi >= 0) & (xi < w_hr)
            xi_safe = xi.clamp(0, w_hr - 1)
            out = out + torch.where(valid, image_hr[yi_safe, xi_safe] * wy * wx, torch.zeros_like(out))
    return out


def _scatter_lr_to_reference(
    image_lr: torch.Tensor,
    shift: torch.Tensor,
    *,
    hr_shape: tuple[int, int],
    scale: int,
) -> torch.Tensor:
    h_hr, w_hr = hr_shape
    h_lr, w_lr = image_lr.shape
    dx, dy = shift.to(dtype=image_lr.dtype, device=image_lr.device)

    y = scale * (torch.arange(h_lr, dtype=image_lr.dtype, device=image_lr.device)[:, None] + dy)
    x = scale * (torch.arange(w_lr, dtype=image_lr.dtype, device=image_lr.device)[None, :] + dx)
    y0 = torch.floor(y).to(torch.long)
    x0 = torch.floor(x).to(torch.long)
    fy = y - y0.to(image_lr.dtype)
    fx = x - x0.to(image_lr.dtype)

    finite = torch.isfinite(image_lr)
    clean = torch.where(finite, image_lr, torch.zeros_like(image_lr))
    out = torch.zeros(hr_shape, dtype=image_lr.dtype, device=image_lr.device)
    out_flat = out.reshape(-1)

    for oy in (0, 1):
        yy = y0 + oy
        wy = (1.0 - fy) if oy == 0 else fy
        valid_y = (yy >= 0) & (yy < h_hr)
        for ox in (0, 1):
            xx = x0 + ox
            wx = (1.0 - fx) if ox == 0 else fx
            valid = valid_y & (xx >= 0) & (xx < w_hr) & finite
            if bool(valid.any()):
                idx = (yy * w_hr + xx).expand(h_lr, w_lr)
                values = clean * wy * wx
                out_flat.index_add_(0, idx[valid], values[valid])
    return out


def forward(
    x_hr: torch.Tensor | object,
    shift: tuple[float, float] | torch.Tensor,
    psf_sigma: float = 1.0,
    *,
    scale: int = 2,
    mode: str = "constant",
) -> torch.Tensor:
    """Predict one raw LR observation from a reference HR image."""

    scale = _validate_scale(scale)
    x = _as_tensor(x_hr)
    if x.ndim != 2:
        raise ValueError("x_hr must be 2D")
    shift_t = _as_tensor(shift, like=x)
    sigma = _sigma_hr(psf_sigma, scale)
    blurred = gaussian_filter2d(x, sigma, mode=mode) if sigma > 0 else x
    return _sample_reference_to_lr(blurred, shift_t, scale=scale)


def adjoint(
    y_residual: torch.Tensor | object,
    shift: tuple[float, float] | torch.Tensor,
    psf_sigma: float = 1.0,
    *,
    hr_shape: tuple[int, int] | None = None,
    scale: int = 2,
    mode: str = "constant",
) -> torch.Tensor:
    """Backproject one LR residual into the reference HR grid."""

    scale = _validate_scale(scale)
    y = _as_tensor(y_residual)
    if y.ndim != 2:
        raise ValueError("y_residual must be 2D")
    if hr_shape is None:
        hr_shape = (int(y.shape[0]) * scale, int(y.shape[1]) * scale)
    shift_t = _as_tensor(shift, like=y)
    scattered = _scatter_lr_to_reference(y, shift_t, hr_shape=tuple(map(int, hr_shape)), scale=scale)
    sigma = _sigma_hr(psf_sigma, scale)
    return gaussian_filter2d(scattered, sigma, mode=mode) if sigma > 0 else scattered


class ForwardOperator(nn.Module):
    """Differentiable EP06-compatible observation operator."""

    def __init__(
        self,
        hr_shape: tuple[int, int],
        lr_shape: tuple[int, int],
        shifts: torch.Tensor | object,
        psf_sigma: float = 1.0,
        scale: int = 2,
        mode: str = "constant",
    ) -> None:
        super().__init__()
        scale = _validate_scale(scale)
        self.hr_shape = tuple(map(int, hr_shape))
        self.lr_shape = tuple(map(int, lr_shape))
        self.psf_sigma = float(psf_sigma)
        self.scale = scale
        self.mode = str(mode)
        expected_lr = (self.hr_shape[0] // scale, self.hr_shape[1] // scale)
        if self.lr_shape != expected_lr:
            raise ValueError("lr_shape is inconsistent with hr_shape and scale")
        shift_tensor = _as_tensor(shifts)
        if shift_tensor.ndim != 2 or shift_tensor.shape[1] != 2:
            raise ValueError("shifts must have shape (N, 2)")
        self.register_buffer("shifts", shift_tensor)

    def forward(self, x_hr: torch.Tensor, index: int) -> torch.Tensor:
        return forward(x_hr, self.shifts[int(index)], self.psf_sigma, scale=self.scale, mode=self.mode)

    def adjoint(self, y_residual: torch.Tensor, index: int) -> torch.Tensor:
        return adjoint(
            y_residual,
            self.shifts[int(index)],
            self.psf_sigma,
            hr_shape=self.hr_shape,
            scale=self.scale,
            mode=self.mode,
        )

    def forward_all(self, x_hr: torch.Tensor) -> torch.Tensor:
        return torch.stack([self.forward(x_hr, idx) for idx in range(len(self.shifts))], dim=0)

    def adjoint_sum(self, residuals: torch.Tensor, *, average: bool = False) -> torch.Tensor:
        if residuals.ndim != 3 or len(residuals) != len(self.shifts):
            raise ValueError("residuals must have shape (N, H, W) matching shifts")
        total = torch.zeros(self.hr_shape, dtype=residuals.dtype, device=residuals.device)
        for idx, residual in enumerate(residuals):
            total = total + self.adjoint(residual, idx)
        if average and len(residuals):
            total = total / float(len(residuals))
        return total
