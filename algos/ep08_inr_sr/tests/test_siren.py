from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ep08.models import Siren, SirenLayer
from ep08.utils import coordinate_grid


def test_siren_shape_and_backward() -> None:
    model = Siren(hidden_features=16, hidden_layers=1)
    coords = coordinate_grid(8, 10)
    out = model(coords)
    loss = out.square().mean()
    loss.backward()

    assert out.shape == (80, 1)
    assert all(param.grad is not None for param in model.parameters() if param.requires_grad)


def test_siren_layer_initialization_is_finite() -> None:
    layer = SirenLayer(2, 16, is_first=True)
    assert torch.isfinite(layer.linear.weight).all()
    assert float(layer.linear.weight.detach().abs().max()) <= 0.5
