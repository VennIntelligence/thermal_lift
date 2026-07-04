from __future__ import annotations

import numpy as np
import pytest
import torch

from unet_sr.config import config_from_args
from unet_sr.losses import ContourSRLoss, fourier_band_filter


def _sinusoid(period_px: float, size: int = 128) -> torch.Tensor:
    xx = torch.arange(size, dtype=torch.float32)
    grid = xx[None, :].expand(size, size)
    return torch.sin(2.0 * np.pi * grid / period_px)[None, None]


def test_fourier_band_filter_passes_in_band_and_rejects_out_of_band() -> None:
    in_band = _sinusoid(3.0)      # 30 um period at 10 um/sample — inside 2.5-4.0 px
    low_freq = _sinusoid(8.0)     # 80 um period — below the band
    near_nyq = _sinusoid(2.1)     # 21 um period — above the band (aperture-zero region)

    def _gain(x: torch.Tensor) -> float:
        y = fourier_band_filter(x, period_lo_px=2.5, period_hi_px=4.0)
        return float(y.square().mean().sqrt() / x.square().mean().sqrt())

    assert _gain(in_band) > 0.7
    assert _gain(low_freq) < 0.15
    assert _gain(near_nyq) < 0.35


def test_band_loss_default_off_is_byte_identical() -> None:
    torch.manual_seed(0)
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randn(2, 1, 64, 64)
    legacy = ContourSRLoss()
    explicit_off = ContourSRLoss(band_loss_weight=0.0)
    out_legacy = legacy(pred, target)
    out_off = explicit_off(pred, target)
    assert "band" not in out_legacy and "band" not in out_off
    for key in out_legacy:
        assert torch.equal(out_legacy[key], out_off[key]), key


def test_band_loss_gate_on_weights_in_band_error_heavier() -> None:
    torch.manual_seed(1)
    target = torch.zeros(1, 1, 128, 128)
    err_in = _sinusoid(3.0)
    err_out = _sinusoid(8.0)
    # Equalize L2 magnitude of the two error maps.
    err_out = err_out * (err_in.square().mean().sqrt() / err_out.square().mean().sqrt())

    crit = ContourSRLoss(band_loss_weight=0.5)
    out_in = crit(err_in.clone().requires_grad_(True), target)
    out_out = crit(err_out, target)
    assert torch.isfinite(out_in["total"])
    assert float(out_in["band"]) > 3.0 * float(out_out["band"])
    out_in["total"].backward()  # gradient must flow through the FFT gate

    with pytest.raises(ValueError):
        ContourSRLoss(band_loss_weight=0.5, band_period_lo_px=5.0, band_period_hi_px=2.0)


def test_config_band_loss_flags_parse_and_validate() -> None:
    cfg = config_from_args(
        [
            "--training-pool-dir", "dummy_pool",
            "--input-mode", "hybrid_drizzle2x",
            "--scale", "2",
            "--unroll-steps", "2",
            "--solver-band-loss", "gate25_40",
            "--solver-band-loss-weight", "0.7",
        ]
    )
    assert cfg.solver_band_loss == "gate25_40"
    assert cfg.solver_band_loss_weight == 0.7

    with pytest.raises(ValueError, match="solver_band_loss_weight"):
        config_from_args(
            [
                "--training-pool-dir", "dummy_pool",
                "--input-mode", "hybrid_drizzle2x",
                "--scale", "2",
                "--unroll-steps", "2",
                "--solver-band-loss", "gate25_40",
                "--solver-band-loss-weight", "0",
            ]
        )
