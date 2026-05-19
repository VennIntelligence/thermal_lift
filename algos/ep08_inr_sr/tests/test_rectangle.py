from __future__ import annotations

import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ep08.forward import ForwardOperator
from ep08.models import Siren
from ep08.stage1 import apply_cli_overrides, parse_training_args
from ep08.trainer import INRTrainer, TrainConfig
from ep08.utils import coordinate_grid


def test_coordinate_grid_preserve_and_stretch_rectangles() -> None:
    preserve = coordinate_grid(6, 10, flatten=False, aspect_mode="preserve")
    default = coordinate_grid(6, 10, flatten=False)
    stretch = coordinate_grid(6, 10, flatten=False, aspect_mode="stretch")

    assert torch.allclose(default, preserve)
    assert preserve.shape == (6, 10, 2)
    assert torch.isclose(preserve[..., 0].amin(), torch.tensor(-1.0))
    assert torch.isclose(preserve[..., 0].amax(), torch.tensor(1.0))
    assert torch.isclose(preserve[..., 1].amin(), torch.tensor(-0.6))
    assert torch.isclose(preserve[..., 1].amax(), torch.tensor(0.6))

    assert torch.isclose(stretch[..., 0].amin(), torch.tensor(-1.0))
    assert torch.isclose(stretch[..., 0].amax(), torch.tensor(1.0))
    assert torch.isclose(stretch[..., 1].amin(), torch.tensor(-1.0))
    assert torch.isclose(stretch[..., 1].amax(), torch.tensor(1.0))

    square = coordinate_grid(4, 4, flatten=False, aspect_mode="preserve")
    assert torch.isclose(square[..., 0].amin(), torch.tensor(-1.0))
    assert torch.isclose(square[..., 0].amax(), torch.tensor(1.0))
    assert torch.isclose(square[..., 1].amin(), torch.tensor(-1.0))
    assert torch.isclose(square[..., 1].amax(), torch.tensor(1.0))

    with pytest.raises(ValueError, match="aspect_mode"):
        coordinate_grid(6, 10, aspect_mode="invalid")


def test_forward_operator_rectangular_shape_smoke() -> None:
    shifts = torch.tensor([[0.0, 0.0], [0.25, -0.25]])
    op = ForwardOperator(hr_shape=(6, 10), lr_shape=(3, 5), shifts=shifts, psf_sigma=0.0)
    x_hr = torch.arange(60, dtype=torch.float32).reshape(6, 10)

    pred = op(x_hr, 0)
    all_preds = op.forward_all(x_hr)
    backprojected = op.adjoint(pred, 0)

    assert pred.shape == (3, 5)
    assert all_preds.shape == (2, 3, 5)
    assert backprojected.shape == (6, 10)
    assert torch.isfinite(all_preds).all()


def test_inr_trainer_rectangular_siren_preserve_smoke() -> None:
    observations = torch.zeros(3, 3, 5)
    shifts = torch.zeros(3, 2)
    model = Siren(hidden_features=8, hidden_layers=1)
    forward_operator = ForwardOperator(
        hr_shape=(6, 10),
        lr_shape=(3, 5),
        shifts=shifts,
        psf_sigma=0.0,
    )
    trainer = INRTrainer(
        model,
        observations,
        shifts,
        hr_shape=(6, 10),
        forward_operator=forward_operator,
        config=TrainConfig(max_iter=1, batch_k=2, warmup_steps=1, val_interval=1),
        coord_aspect_mode="preserve",
    )

    result = trainer.fit()

    assert trainer.coord_aspect_mode == "preserve"
    assert result.image.shape == (6, 10)
    assert len(result.history) == 1
    assert torch.isfinite(result.image).all()


@pytest.mark.parametrize(
    ("flag", "mode"),
    [
        ("--coord-aspect-mode", "stretch"),
        ("--coordinate-aspect-mode", "preserve"),
    ],
)
def test_cli_coord_aspect_mode_override(monkeypatch: pytest.MonkeyPatch, flag: str, mode: str) -> None:
    monkeypatch.setattr(sys, "argv", ["train_siren.py", flag, mode])

    args = parse_training_args("siren", "test parser")
    cfg = apply_cli_overrides({"coordinates": {"aspect_mode": "preserve"}}, args)

    assert args.coord_aspect_mode == mode
    assert cfg["coordinates"]["aspect_mode"] == mode
