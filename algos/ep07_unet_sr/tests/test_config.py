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
            "--boundary-boost",
            "3.0",
            "--flatness-weight",
            "0.05",
        ]
    )

    assert cfg.hr_upsampler == "pixelshuffle"
    assert cfg.hr_res_blocks == 1
    assert cfg.boundary_boost == 3.0
    assert cfg.flatness_weight == 0.05


def test_config_parses_forward_model_band_args() -> None:
    """V9B: forward-model-band CLI args are parsed correctly."""
    cfg = config_from_args([
        "--training-pool-dir", "dummy_pool",
        "--forward-model-band", "highpass",
        "--forward-model-band-sigma", "3.0",
        "--forward-model-weight", "0.1",
        "--boundary-boost", "3.0",
    ])
    assert cfg.forward_model_band == "highpass"
    assert cfg.forward_model_band_sigma == 3.0
    assert cfg.forward_model_weight == 0.1


def test_config_forward_model_band_default_is_full() -> None:
    """V9B: default band is 'full' for backward compatibility."""
    cfg = config_from_args([
        "--training-pool-dir", "dummy_pool",
        "--boundary-boost", "3.0",
    ])
    assert cfg.forward_model_band == "full"
    assert cfg.forward_model_band_sigma == 5.0


def test_config_parses_hybrid_drizzle2x_input_mode() -> None:
    """V9A: input_mode='hybrid_drizzle2x' auto-sets in_channels=9 (5 fused + 4 phase-bin)."""
    cfg = config_from_args([
        "--training-pool-dir", "dummy_pool",
        "--input-mode", "hybrid_drizzle2x",
        "--boundary-boost", "3.0",
    ])
    assert cfg.input_mode == "hybrid_drizzle2x"
    assert cfg.in_channels == 9


def test_config_parses_solver_full_halo_real_eval_args() -> None:
    cfg = config_from_args([
        "--training-pool-dir", "dummy_pool",
        "--scale", "2",
        "--real-eval-solver-mode", "full_halo",
        "--real-eval-solver-halo-hr", "64",
        "--boundary-boost", "3.0",
    ])

    assert cfg.real_eval_solver_mode == "full_halo"
    assert cfg.real_eval_solver_halo_hr == 64


def test_config_rejects_solver_real_eval_halo_not_divisible_by_scale() -> None:
    with pytest.raises(ValueError, match="divisible by --scale"):
        config_from_args([
            "--training-pool-dir", "dummy_pool",
            "--scale", "2",
            "--real-eval-solver-mode", "full_halo",
            "--real-eval-solver-halo-hr", "63",
            "--boundary-boost", "3.0",
        ])


def test_config_input_mode_default_is_lr() -> None:
    """V9A: default input_mode is 'lr' for backward compatibility."""
    cfg = config_from_args([
        "--training-pool-dir", "dummy_pool",
        "--boundary-boost", "3.0",
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
        "--boundary-boost", "3.0",
    ])

    assert cfg.input_mode == "hybrid_drizzle2x"
    assert cfg.in_channels == 9
    assert cfg.forward_model_weight == 0.1


def test_config_rejects_hybrid_forward_model_without_2x_geometry() -> None:
    """V9C: legal hybrid forward anchor requires explicit 2x→1x geometry."""
    with pytest.raises(ValueError, match="requires --scale 2"):
        config_from_args([
            "--training-pool-dir", "dummy_pool",
            "--input-mode", "hybrid_drizzle2x",
            "--forward-model-weight", "0.1",
            "--boundary-boost", "3.0",
        ])


def test_config_allows_drizzle2x_residual_mode() -> None:
    cfg = config_from_args([
        "--training-pool-dir", "dummy_pool",
        "--input-mode", "hybrid_drizzle2x",
        "--scale", "2",
        "--residual-mode", "drizzle2x",
        "--residual-penalty-weight", "0.25",
        "--boundary-boost", "3.0",
    ])

    assert cfg.residual_mode == "drizzle2x"
    assert cfg.residual_penalty_weight == 0.25
    assert cfg.input_mode == "hybrid_drizzle2x"
    assert cfg.in_channels == 9


def test_config_rejects_drizzle2x_residual_without_hybrid_input() -> None:
    with pytest.raises(ValueError, match="requires input_mode='hybrid_drizzle2x'"):
        config_from_args([
            "--training-pool-dir", "dummy_pool",
            "--scale", "2",
            "--residual-mode", "drizzle2x",
            "--boundary-boost", "3.0",
        ])


def test_config_rejects_drizzle2x_residual_without_scale_2() -> None:
    with pytest.raises(ValueError, match="requires --scale 2"):
        config_from_args([
            "--training-pool-dir", "dummy_pool",
            "--input-mode", "hybrid_drizzle2x",
            "--residual-mode", "drizzle2x",
            "--boundary-boost", "3.0",
        ])


def test_config_rejects_drizzle2x_residual_with_forward_model() -> None:
    with pytest.raises(ValueError, match="mutually exclusive with forward_model_weight"):
        config_from_args([
            "--training-pool-dir", "dummy_pool",
            "--input-mode", "hybrid_drizzle2x",
            "--scale", "2",
            "--residual-mode", "drizzle2x",
            "--forward-model-weight", "0.1",
            "--boundary-boost", "3.0",
        ])


def test_config_rejects_drizzle2x_residual_with_legacy_residual() -> None:
    with pytest.raises(ValueError, match="cannot be combined with --residual"):
        config_from_args([
            "--training-pool-dir", "dummy_pool",
            "--input-mode", "hybrid_drizzle2x",
            "--scale", "2",
            "--residual-mode", "drizzle2x",
            "--residual",
            "--boundary-boost", "3.0",
        ])
