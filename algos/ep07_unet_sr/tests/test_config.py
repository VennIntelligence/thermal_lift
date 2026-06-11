from __future__ import annotations

import pytest

from unet_sr.config import config_from_args


def test_config_parses_pixelshuffle_head_args() -> None:
    cfg = config_from_args(
        [
            "--training-pool-dir",
            "dummy_pool",
            "--hr-upsampler",
            "pixelshuffle",
            "--hr-res-blocks",
            "1",
            "--thin-boost",
            "3.0",
            "--gap-boost",
            "2.0",
        ]
    )

    assert cfg.hr_upsampler == "pixelshuffle"
    assert cfg.hr_res_blocks == 1
    assert cfg.thin_boost == 3.0
    assert cfg.gap_boost == 2.0


def test_config_parses_forward_model_band_args() -> None:
    """V9B: forward-model-band CLI args are parsed correctly."""
    cfg = config_from_args([
        "--training-pool-dir", "dummy_pool",
        "--forward-model-band", "highpass",
        "--forward-model-band-sigma", "3.0",
        "--forward-model-weight", "0.1",
        "--thin-boost", "3.0", "--gap-boost", "2.0",
    ])
    assert cfg.forward_model_band == "highpass"
    assert cfg.forward_model_band_sigma == 3.0
    assert cfg.forward_model_weight == 0.1


def test_config_forward_model_band_default_is_full() -> None:
    """V9B: default band is 'full' for backward compatibility."""
    cfg = config_from_args([
        "--training-pool-dir", "dummy_pool",
        "--thin-boost", "3.0", "--gap-boost", "2.0",
    ])
    assert cfg.forward_model_band == "full"
    assert cfg.forward_model_band_sigma == 5.0


def test_config_parses_hybrid_drizzle2x_input_mode() -> None:
    """V9A: input_mode='hybrid_drizzle2x' auto-sets in_channels=8."""
    cfg = config_from_args([
        "--training-pool-dir", "dummy_pool",
        "--input-mode", "hybrid_drizzle2x",
        "--thin-boost", "3.0", "--gap-boost", "2.0",
    ])
    assert cfg.input_mode == "hybrid_drizzle2x"
    assert cfg.in_channels == 8


def test_config_input_mode_default_is_lr() -> None:
    """V9A: default input_mode is 'lr' for backward compatibility."""
    cfg = config_from_args([
        "--training-pool-dir", "dummy_pool",
        "--thin-boost", "3.0", "--gap-boost", "2.0",
    ])
    assert cfg.input_mode == "lr"
    assert cfg.in_channels == 5


def test_config_allows_hybrid_with_forward_model_at_scale_2() -> None:
    """V9C: hybrid + forward_model_weight is legal when scale=2 provides lr_obs."""
    cfg = config_from_args([
        "--training-pool-dir", "dummy_pool",
        "--input-mode", "hybrid_drizzle2x",
        "--scale", "2",
        "--forward-model-weight", "0.1",
        "--forward-model-band", "highpass",
        "--thin-boost", "3.0", "--gap-boost", "2.0",
    ])

    assert cfg.input_mode == "hybrid_drizzle2x"
    assert cfg.in_channels == 8
    assert cfg.forward_model_weight == 0.1


def test_config_rejects_hybrid_forward_model_without_2x_geometry() -> None:
    """V9C: legal hybrid forward anchor requires explicit 2x→1x geometry."""
    with pytest.raises(ValueError, match="requires --scale 2"):
        config_from_args([
            "--training-pool-dir", "dummy_pool",
            "--input-mode", "hybrid_drizzle2x",
            "--forward-model-weight", "0.1",
            "--thin-boost", "3.0", "--gap-boost", "2.0",
        ])
