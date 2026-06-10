from __future__ import annotations

import torch
import torch.nn as nn

from sr4x.losses import ForwardConsistencyLoss, ThermalSR4xLoss
from sr4x.model import ThermalSR4xUNet


def test_unet_outputs_temperature_and_log_variance() -> None:
    model = ThermalSR4xUNet(
        in_channels=8,
        out_channels=1,
        base_channels=8,
        depth=4,
        scale=1,
        predict_log_variance=True,
    )
    x = torch.randn(2, 8, 64, 64)

    pred, log_var = model(x)

    assert pred.shape == (2, 1, 64, 64)
    assert log_var.shape == pred.shape
    assert not any(isinstance(module, nn.BatchNorm2d) for module in model.modules())


def test_thermal_sr4x_loss_is_finite_and_backpropagates() -> None:
    criterion = ThermalSR4xLoss(
        sigma_lf=2.0,
        lf_weight=1.0,
        hf_weight=0.3,
        edge_weight=0.1,
        forward_weight=0.2,
        nll_weight=0.05,
        scale=4,
        psf_sigma_lr_px=0.25,
    )
    pred = torch.randn(2, 1, 32, 32, requires_grad=True)
    log_var = torch.zeros_like(pred, requires_grad=True)
    target = torch.randn(2, 1, 32, 32)
    coverage = torch.rand(2, 1, 32, 32)
    drizzle = target + 0.05 * torch.randn_like(target)
    edge = torch.zeros_like(target)
    edge[:, :, 8:16, 8:16] = 1.0

    losses = criterion(
        pred,
        target,
        edge_mask=edge,
        coverage_4x=coverage,
        drizzle_mean_4x=drizzle,
        log_var=log_var,
    )
    losses["total"].backward()

    assert set(losses) == {"total", "lf", "hf", "edge", "forward", "nll"}
    assert all(torch.isfinite(value) for value in losses.values())
    assert pred.grad is not None
    assert log_var.grad is not None


def test_forward_consistency_accepts_zero_coverage() -> None:
    loss = ForwardConsistencyLoss(scale=4, psf_sigma_lr_px=0.25)
    pred = torch.randn(1, 1, 32, 32)
    drizzle = torch.randn(1, 1, 32, 32)
    coverage = torch.zeros(1, 1, 32, 32)

    value = loss(pred, drizzle, coverage)

    assert torch.isfinite(value)
