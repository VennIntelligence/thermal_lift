from __future__ import annotations

from sr4x.config import config_from_args


def test_config_parses_ep12_loss_options() -> None:
    cfg = config_from_args(
        [
            "--training-pool-dir",
            "/tmp/pool",
            "--device",
            "cpu",
            "--total-steps",
            "2",
            "--forward-loss-weight",
            "0.4",
            "--nll-loss-weight",
            "0.07",
            "--coverage-loss-gain",
            "5.0",
            "--unet-depth",
            "4",
        ]
    )

    assert cfg.forward_loss_weight == 0.4
    assert cfg.nll_loss_weight == 0.07
    assert cfg.coverage_loss_gain == 5.0
    assert cfg.unet_depth == 4
    assert cfg.burst_augment is True
    assert cfg.amp is False


def test_config_can_disable_burst_augmentation_for_legacy_pools() -> None:
    cfg = config_from_args(
        [
            "--training-pool-dir",
            "/tmp/pool",
            "--device",
            "cpu",
            "--total-steps",
            "2",
            "--no-burst-augment",
        ]
    )

    assert cfg.burst_augment is False
