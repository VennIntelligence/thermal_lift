from __future__ import annotations

import math

import torch
from torch import nn


class WireLayer(nn.Module):
    """Real-valued WIRE-style Gabor layer.

    The activation is a sinusoid gated by a Gaussian envelope. It keeps the
    SIREN MLP interface while biasing the representation toward localized
    high-frequency edge responses in the signed highpass field.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 20.0,
        sigma_0: float = 10.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.sigma_0 = sigma_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.linear_scale = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / max(self.omega_0, 1e-6)
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)
            self.linear_scale.weight.uniform_(-bound, bound)
            if self.linear_scale.bias is not None:
                self.linear_scale.bias.uniform_(-bound, bound)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        carrier_input = self.linear(coords)
        scale_input = self.linear_scale(coords)
        carrier = torch.sin(self.omega_0 * carrier_input)
        envelope = torch.exp(-0.5 * (self.sigma_0 * scale_input).square())
        return carrier * envelope


class Wire(nn.Module):
    """WIRE coordinate MLP for signed highpass HR fields."""

    expects_coords = True

    def __init__(
        self,
        in_features: int = 2,
        hidden_features: int = 256,
        hidden_layers: int = 4,
        out_features: int = 1,
        *,
        first_omega_0: float = 20.0,
        hidden_omega_0: float = 20.0,
        first_sigma_0: float = 10.0,
        hidden_sigma_0: float = 10.0,
        outermost_linear: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            WireLayer(
                in_features,
                hidden_features,
                is_first=True,
                omega_0=first_omega_0,
                sigma_0=first_sigma_0,
            )
        ]
        for _ in range(hidden_layers):
            layers.append(
                WireLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                    sigma_0=hidden_sigma_0,
                )
            )
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = math.sqrt(6.0 / hidden_features) / max(hidden_omega_0, 1e-6)
                final_linear.weight.uniform_(-bound, bound)
                if final_linear.bias is not None:
                    final_linear.bias.uniform_(-bound, bound)
            layers.append(final_linear)
        else:
            layers.append(
                WireLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                    sigma_0=hidden_sigma_0,
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)
