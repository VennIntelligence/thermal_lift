from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ep08.models import Wire, WireLayer
from ep08.utils import coordinate_grid


def test_wire_shape_and_backward() -> None:
    model = Wire(hidden_features=16, hidden_layers=1)
    coords = coordinate_grid(8, 10)
    out = model(coords)
    loss = out.square().mean()
    loss.backward()

    assert out.shape == (80, 1)
    assert torch.isfinite(out).all()
    assert all(param.grad is not None for param in model.parameters() if param.requires_grad)


def test_wire_layer_initialization_is_finite() -> None:
    layer = WireLayer(2, 16, is_first=True)
    out = layer(torch.zeros(4, 2))
    assert out.shape == (4, 16)
    assert torch.isfinite(out).all()
    assert layer.linear.weight.shape == layer.linear_scale.weight.shape


def test_wire_layer_uses_independent_carrier_and_envelope_projections() -> None:
    layer = WireLayer(2, 8, is_first=True)
    coords = torch.randn(6, 2)
    loss = layer(coords).square().mean()
    loss.backward()

    assert layer.linear.weight.grad is not None
    assert layer.linear_scale.weight.grad is not None
    assert torch.isfinite(layer.linear.weight.grad).all()
    assert torch.isfinite(layer.linear_scale.weight.grad).all()
