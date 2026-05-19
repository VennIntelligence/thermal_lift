from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")
from torch import nn

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ep08.models import DeepDecoder


def test_deep_decoder_shape_and_backward() -> None:
    model = DeepDecoder(latent_channels=4, hidden_channels=(8, 8), latent_spatial=(4, 4))
    out = model(output_shape=(16, 20))
    loss = out.square().mean()
    loss.backward()

    assert out.shape == (1, 16, 20)
    assert torch.isfinite(out).all()
    assert all(param.grad is not None for param in model.parameters() if param.requires_grad)


def test_deep_decoder_is_deterministic_for_same_seed() -> None:
    model_a = DeepDecoder(latent_channels=4, hidden_channels=(8,), latent_spatial=(4, 4), seed=12)
    model_b = DeepDecoder(latent_channels=4, hidden_channels=(8,), latent_spatial=(4, 4), seed=12)
    assert torch.allclose(model_a.latent, model_b.latent)


def test_deep_decoder_natural_and_cropped_shapes() -> None:
    model = DeepDecoder(latent_channels=4, hidden_channels=(8, 6), latent_spatial=(3, 5))

    natural = model()
    cropped = model(output_shape=(10, 18))

    assert natural.shape == (1, 12, 20)
    assert cropped.shape == (1, 10, 18)


def test_deep_decoder_production_config_has_exact_native_shape() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "deep_decoder.yaml"
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg = cfg["model"]

    assert model_cfg["hidden_channels"] == [64, 64, 32, 32]
    assert model_cfg["latent_spatial"] == [32, 32]

    model = DeepDecoder(
        latent_channels=model_cfg["latent_channels"],
        hidden_channels=tuple(model_cfg["hidden_channels"]),
        latent_spatial=tuple(model_cfg["latent_spatial"]),
    )

    natural = model()
    matched = model(output_shape=(512, 512))

    assert natural.shape == (1, 512, 512)
    assert matched.shape == (1, 512, 512)


def test_deep_decoder_allows_small_output_shape_correction() -> None:
    model = DeepDecoder(latent_channels=4, hidden_channels=(8, 8), latent_spatial=(4, 4))

    corrected = model(output_shape=(16, 20))

    assert corrected.shape == (1, 16, 20)


def test_deep_decoder_rejects_broad_interpolation_by_gap() -> None:
    model = DeepDecoder(latent_channels=4, hidden_channels=(8, 8), latent_spatial=(4, 4))

    with pytest.raises(ValueError, match="refusing broad interpolation"):
        model(output_shape=(16, 21))


def test_deep_decoder_rejects_broad_interpolation_by_scale() -> None:
    model = DeepDecoder(latent_channels=4, hidden_channels=(8,), latent_spatial=(1, 1))

    with pytest.raises(ValueError, match="refusing broad interpolation"):
        model(output_shape=(5, 2))


def test_deep_decoder_parameter_count_is_limited() -> None:
    model = DeepDecoder(latent_channels=32, hidden_channels=(64, 64, 32, 32), latent_spatial=(32, 32))
    trainable_params = sum(param.numel() for param in model.parameters() if param.requires_grad)

    assert trainable_params < 10_000


def test_deep_decoder_upsamples_before_each_conv() -> None:
    model = DeepDecoder(latent_channels=4, hidden_channels=(8, 6), latent_spatial=(4, 4))
    layers = list(model.blocks)

    assert len(layers) == 8
    for block_start in range(0, len(layers), 4):
        assert isinstance(layers[block_start], nn.Upsample)
        assert isinstance(layers[block_start + 1], nn.Conv2d)
        assert isinstance(layers[block_start + 2], nn.BatchNorm2d)
        assert isinstance(layers[block_start + 3], nn.ReLU)
