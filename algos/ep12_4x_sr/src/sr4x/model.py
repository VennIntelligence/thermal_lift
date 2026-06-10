"""UNet architecture for EP12 same-grid 4x restoration."""

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
    """Squeeze-and-Excitation channel attention."""

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
    """Two convolution layers with GroupNorm, SiLU, and SE attention."""

    def __init__(self, in_channels: int, out_channels: int, *, dilation: int = 1) -> None:
        super().__init__()
        padding = int(dilation)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )
        self.se = SEBlock(out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.se(self.net(x))


class DilatedBottleneck(nn.Module):
    """Residual dilated bottleneck to extend context without another pool."""

    def __init__(self, channels: int, dilations: tuple[int, ...] = (2, 4, 8)) -> None:
        super().__init__()
        blocks: list[nn.Module] = []
        for dilation in dilations:
            blocks.extend(
                [
                    nn.Conv2d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
                    nn.GroupNorm(_group_count(channels), channels),
                    nn.SiLU(inplace=True),
                ]
            )
        self.net = nn.Sequential(*blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ThermalSR4xUNet(nn.Module):
    """Same-resolution UNet with optional confidence head.

    The default depth is four encoder levels, which increases the context
    available for 4x gap filling compared with the EP07 three-level model.
    When ``predict_log_variance`` is true, forward returns ``(temp, log_var)``.
    """

    def __init__(
        self,
        in_channels: int = 8,
        out_channels: int = 1,
        base_channels: int = 48,
        *,
        scale: int = 1,
        depth: int = 4,
        dilated_bottleneck: bool = True,
        predict_log_variance: bool = True,
        min_log_variance: float = -8.0,
        max_log_variance: float = 4.0,
    ) -> None:
        super().__init__()
        if scale <= 0:
            raise ValueError("scale must be positive")
        if depth < 3 or depth > 5:
            raise ValueError("depth must be between 3 and 5")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if min_log_variance >= max_log_variance:
            raise ValueError("min_log_variance must be < max_log_variance")

        self.scale = int(scale)
        self.predict_log_variance = bool(predict_log_variance)
        self.min_log_variance = float(min_log_variance)
        self.max_log_variance = float(max_log_variance)

        channels = [int(base_channels) * (2**level) for level in range(depth)]
        encoders: list[nn.Module] = []
        prev_channels = int(in_channels)
        for ch in channels:
            encoders.append(ConvBlock(prev_channels, ch))
            prev_channels = ch
        self.encoders = nn.ModuleList(encoders)
        self.pool = nn.MaxPool2d(kernel_size=2)

        bottleneck_channels = channels[-1] * 2
        bottleneck: list[nn.Module] = [ConvBlock(channels[-1], bottleneck_channels)]
        if dilated_bottleneck:
            bottleneck.append(DilatedBottleneck(bottleneck_channels))
        self.bottleneck = nn.Sequential(*bottleneck)

        up_projs: list[nn.Module] = []
        decoders: list[nn.Module] = []
        current_channels = bottleneck_channels
        for skip_channels in reversed(channels):
            up_projs.append(nn.Conv2d(current_channels, skip_channels, kernel_size=1))
            decoders.append(ConvBlock(skip_channels * 2, skip_channels))
            current_channels = skip_channels
        self.up_projs = nn.ModuleList(up_projs)
        self.decoders = nn.ModuleList(decoders)

        self.refine = nn.Sequential(
            nn.Conv2d(channels[0], channels[0], kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(channels[0]), channels[0]),
            nn.SiLU(inplace=True),
        )
        if self.scale > 1:
            self.upsample = nn.Sequential(
                nn.Conv2d(channels[0], channels[0] * self.scale ** 2, kernel_size=3, padding=1),
                nn.PixelShuffle(self.scale),
                nn.GroupNorm(_group_count(channels[0]), channels[0]),
                nn.SiLU(inplace=True),
            )
        else:
            self.upsample = None
        self.temp_head = nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)
        self.log_var_head = (
            nn.Conv2d(channels[0], out_channels, kernel_size=3, padding=1)
            if self.predict_log_variance
            else None
        )

    @staticmethod
    def _upsample_to(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=ref.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 4:
            raise ValueError("input must have shape (B, C, H, W)")

        skips: list[torch.Tensor] = []
        h = x
        for encoder in self.encoders:
            h = encoder(h)
            skips.append(h)
            h = self.pool(h)

        h = self.bottleneck(h)
        for up_proj, decoder, skip in zip(self.up_projs, self.decoders, reversed(skips), strict=True):
            h = up_proj(self._upsample_to(h, skip))
            h = decoder(torch.cat([h, skip], dim=1))

        h = self.refine(h)
        if self.upsample is not None:
            h = self.upsample(h)
        temp = self.temp_head(h)

        if self.log_var_head is None:
            return temp
        log_var = self.log_var_head(h).clamp(self.min_log_variance, self.max_log_variance)
        return temp, log_var
