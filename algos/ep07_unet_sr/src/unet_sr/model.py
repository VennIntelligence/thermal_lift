"""Regression UNet with channel attention for compact thermal SR features."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel attention.

    Learns per-channel importance weights via global average pooling →
    bottleneck FC → sigmoid gating.  Adds < 0.1% parameters but helps
    the network automatically weight input channels (e.g., highpass vs
    mean vs coverage).
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 4)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(channels, mid),
            nn.SiLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x).view(x.size(0), -1, 1, 1)


class ConvBlock(nn.Module):
    """Two 3×3 conv layers with GroupNorm + SiLU, followed by SE attention."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.se = SEBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.net(x))


def _icnr_init(conv: nn.Conv2d, *, scale: int) -> None:
    """Initialize sub-pixel conv weights to start near nearest-neighbor upsample."""

    if scale <= 0:
        raise ValueError("scale must be positive")
    out_channels, in_channels, kh, kw = conv.weight.shape
    groups = scale * scale
    if out_channels % groups != 0:
        raise ValueError("conv out_channels must be divisible by scale^2 for ICNR")
    subkernel = conv.weight.new_empty(out_channels // groups, in_channels, kh, kw)
    nn.init.kaiming_normal_(subkernel, nonlinearity="linear")
    with torch.no_grad():
        conv.weight.copy_(subkernel.repeat_interleave(groups, dim=0))
        if conv.bias is not None:
            conv.bias.zero_()


class HRResBlock(nn.Module):
    """Lightweight HR-space residual refinement without normalization."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ThermalSRUNet(nn.Module):
    """Compact UNet with SE attention for thermal super-resolution.

    Maps fused observation features to an HR temperature field.
    When scale=1 (residual mode), no upsampling is performed and the
    network outputs a same-resolution residual correction.
    """

    def __init__(
        self,
        in_channels: int = 5,
        out_channels: int = 1,
        base_channels: int = 48,
        scale: int = 4,
        hr_upsampler: str = "bilinear",
        hr_res_blocks: int = 0,
    ) -> None:
        super().__init__()
        if scale <= 0:
            raise ValueError("scale must be positive")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if hr_upsampler not in ("bilinear", "pixelshuffle"):
            raise ValueError("hr_upsampler must be 'bilinear' or 'pixelshuffle'")
        if hr_res_blocks < 0:
            raise ValueError("hr_res_blocks must be >= 0")
        self.scale = int(scale)
        self.hr_upsampler = str(hr_upsampler)

        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8

        self.enc1 = ConvBlock(in_channels, c1)
        self.enc2 = ConvBlock(c1, c2)
        self.enc3 = ConvBlock(c2, c3)
        self.bottleneck = ConvBlock(c3, c4)
        self.pool = nn.MaxPool2d(kernel_size=2)

        self.up3 = nn.Conv2d(c4, c3, kernel_size=1)
        self.dec3 = ConvBlock(c3 + c3, c3)
        self.up2 = nn.Conv2d(c3, c2, kernel_size=1)
        self.dec2 = ConvBlock(c2 + c2, c2)
        self.up1 = nn.Conv2d(c2, c1, kernel_size=1)
        self.dec1 = ConvBlock(c1 + c1, c1)

        if self.hr_upsampler == "bilinear":
            self.hr_refine = nn.Sequential(
                nn.Conv2d(c1, c1, kernel_size=3, padding=1),
                nn.GroupNorm(_group_count(c1), c1),
                nn.SiLU(inplace=True),
                nn.Conv2d(c1, out_channels, kernel_size=3, padding=1),
            )
        else:
            self.pixel_shuffle_conv = nn.Conv2d(c1, c1 * self.scale * self.scale, kernel_size=3, padding=1)
            _icnr_init(self.pixel_shuffle_conv, scale=self.scale)
            self.pixel_shuffle = nn.PixelShuffle(self.scale)
            refine_layers: list[nn.Module] = [HRResBlock(c1) for _ in range(int(hr_res_blocks))]
            refine_layers.append(nn.Conv2d(c1, out_channels, kernel_size=3, padding=1))
            self.hr_refine = nn.Sequential(*refine_layers)

    @staticmethod
    def _upsample_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4:
            raise ValueError("input must have shape (B, C, H, W)")
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(self._upsample_to(b, e3))
        d3 = self.dec3(torch.cat([d3, e3], dim=1))
        d2 = self.up2(self._upsample_to(d3, e2))
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = self.up1(self._upsample_to(d2, e1))
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        if self.hr_upsampler == "bilinear":
            hr = F.interpolate(d1, scale_factor=self.scale, mode="bilinear", align_corners=False)
        else:
            hr = self.pixel_shuffle(self.pixel_shuffle_conv(d1))
        return self.hr_refine(hr)
