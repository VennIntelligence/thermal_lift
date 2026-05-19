from __future__ import annotations

import math

import torch
from torch import nn


class SirenLayer(nn.Module):
    """Linear layer followed by SIREN sine activation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        bias: bool = True,
        is_first: bool = False,
        omega_0: float = 30.0,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.is_first = is_first
        self.omega_0 = omega_0
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            if self.is_first:
                bound = 1.0 / self.in_features
            else:
                bound = math.sqrt(6.0 / self.in_features) / self.omega_0
            self.linear.weight.uniform_(-bound, bound)
            if self.linear.bias is not None:
                self.linear.bias.uniform_(-bound, bound)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(coords))


class Siren(nn.Module):
    """Coordinate MLP for signed highpass HR fields."""

    expects_coords = True

    def __init__(
        self,
        in_features: int = 2,
        hidden_features: int = 256,
        hidden_layers: int = 4,
        out_features: int = 1,
        *,
        first_omega_0: float = 30.0,
        hidden_omega_0: float = 30.0,
        outermost_linear: bool = True,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            SirenLayer(
                in_features,
                hidden_features,
                is_first=True,
                omega_0=first_omega_0,
            )
        ]
        for _ in range(hidden_layers):
            layers.append(
                SirenLayer(
                    hidden_features,
                    hidden_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )
        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                bound = math.sqrt(6.0 / hidden_features) / hidden_omega_0
                final_linear.weight.uniform_(-bound, bound)
                if final_linear.bias is not None:
                    final_linear.bias.uniform_(-bound, bound)
            layers.append(final_linear)
        else:
            layers.append(
                SirenLayer(
                    hidden_features,
                    out_features,
                    is_first=False,
                    omega_0=hidden_omega_0,
                )
            )
        self.net = nn.Sequential(*layers)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        return self.net(coords)
