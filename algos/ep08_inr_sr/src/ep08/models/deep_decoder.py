from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class DeepDecoder(nn.Module):
    """Low-parameter CNN decoder with a fixed latent input.

    The model emits one signed HR highpass image. It deliberately avoids a
    sigmoid or tanh output clamp because EP08 optimizes signed highpass maps,
    not display-normalized intensities.

    Architecture:
        fixed z -> [Upsample2x -> Conv1x1 -> BatchNorm -> ReLU] x N -> Conv1x1
    """

    expects_coords = False

    def __init__(
        self,
        output_channels: int = 1,
        latent_channels: int = 32,
        hidden_channels: tuple[int, ...] = (64, 64, 32),
        latent_spatial: tuple[int, int] = (8, 8),
        *,
        seed: int = 0,
        upsample_mode: str = "bilinear",
    ) -> None:
        super().__init__()
        self.output_channels = output_channels
        self.upsample_mode = upsample_mode
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        latent = torch.randn(1, latent_channels, *latent_spatial, generator=generator)
        self.register_buffer("latent", latent)

        layers: list[nn.Module] = []
        in_channels = latent_channels
        for channels in hidden_channels:
            layers.append(self._make_upsample())
            layers.append(nn.Conv2d(in_channels, channels, kernel_size=1))
            layers.append(nn.BatchNorm2d(channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = channels
        self.blocks = nn.Sequential(*layers)
        self.output = nn.Conv2d(in_channels, output_channels, kernel_size=1)
        self.reset_parameters()

    def _make_upsample(self) -> nn.Upsample:
        align_corners = (
            False
            if self.upsample_mode in {"linear", "bilinear", "bicubic", "trilinear"}
            else None
        )
        return nn.Upsample(scale_factor=2, mode=self.upsample_mode, align_corners=align_corners)

    def reset_parameters(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, output_shape: tuple[int, int] | None = None) -> torch.Tensor:
        x = self.latent
        x = self.blocks(x)
        x = self.output(x)
        if output_shape is not None:
            x = self._match_output_shape(x, output_shape)
        return x.squeeze(0)

    def _match_output_shape(self, x: torch.Tensor, output_shape: tuple[int, int]) -> torch.Tensor:
        target_h, target_w = output_shape
        current_h, current_w = int(x.shape[-2]), int(x.shape[-1])
        if (current_h, current_w) == (target_h, target_w):
            return x

        if current_h >= target_h and current_w >= target_w:
            top = (current_h - target_h) // 2
            left = (current_w - target_w) // 2
            return x[..., top : top + target_h, left : left + target_w]

        return F.interpolate(
            x,
            size=output_shape,
            mode=self.upsample_mode,
            align_corners=False if self.upsample_mode in {"linear", "bilinear", "bicubic", "trilinear"} else None,
        )
