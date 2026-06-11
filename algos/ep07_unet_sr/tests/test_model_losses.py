from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from unet_sr.losses import ContourSRLoss, ThermalSRLoss, sobel_edges
from unet_sr.model import ThermalSRUNet
from unet_sr.mask_weights import compute_mask_loss_weights


def test_unet_outputs_scaled_patch_without_batchnorm() -> None:
    model = ThermalSRUNet(in_channels=5, out_channels=1, base_channels=8, scale=4)
    x = torch.randn(2, 5, 16, 16)

    y = model(x)

    assert y.shape == (2, 1, 64, 64)
    assert not any(isinstance(module, nn.BatchNorm2d) for module in model.modules())


def test_unet_pixelshuffle_head_outputs_scaled_patch() -> None:
    model = ThermalSRUNet(
        in_channels=5,
        out_channels=1,
        base_channels=8,
        scale=2,
        hr_upsampler="pixelshuffle",
        hr_res_blocks=1,
    )
    x = torch.randn(2, 5, 16, 16)

    y = model(x)

    assert y.shape == (2, 1, 32, 32)
    assert not any(isinstance(module, nn.BatchNorm2d) for module in model.modules())


def test_thermal_sr_loss_is_finite() -> None:
    loss = ThermalSRLoss(edge_weight=0.1)
    pred = torch.randn(2, 1, 32, 32, requires_grad=True)
    target = torch.randn(2, 1, 32, 32)
    edge_mask = torch.zeros_like(target)
    edge_mask[:, :, 8:16, 8:16] = 1.0

    values = loss(pred, target, edge_mask=edge_mask)
    values["total"].backward()

    assert set(values) == {"total", "mse", "edge", "ssim"}
    assert all(torch.isfinite(value) for value in values.values())
    assert torch.isfinite(sobel_edges(target)).all()
    assert pred.grad is not None


def test_contour_sr_loss_accepts_optional_thin_and_gap_weights() -> None:
    loss = ContourSRLoss()
    pred = torch.randn(2, 1, 32, 32, requires_grad=True)
    target = torch.randn(2, 1, 32, 32)
    thin = torch.ones_like(target)
    gap = torch.ones_like(target)
    thin[:, :, 12:14, 12:20] = 6.0
    gap[:, :, 20:22, 8:24] = 4.0

    values = loss(pred, target, thin_weight=thin, gap_weight=gap)
    values["total"].backward()

    assert set(values) == {"total", "mse", "highpass", "edge", "ssim", "grad_vector"}
    assert all(torch.isfinite(value) for value in values.values())
    assert pred.grad is not None


def test_contour_sr_loss_all_one_weights_match_default_behavior() -> None:
    loss = ContourSRLoss()
    pred = torch.randn(1, 1, 24, 24)
    target = torch.randn(1, 1, 24, 24)
    ones = torch.ones_like(target)

    base = loss(pred, target)
    weighted = loss(pred, target, thin_weight=ones, gap_weight=ones)

    for key in base:
        assert torch.allclose(base[key], weighted[key])


def test_contour_sr_loss_weight_shape_mismatch_raises() -> None:
    loss = ContourSRLoss()
    pred = torch.randn(1, 1, 16, 16)
    target = torch.randn(1, 1, 16, 16)
    bad = torch.ones(1, 1, 8, 8)

    with pytest.raises(ValueError, match="thin_weight shape mismatch"):
        loss(pred, target, thin_weight=bad)


def test_mask_loss_weights_boost_thin_structures_and_narrow_gaps() -> None:
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, 4:12, 4:6] = 1.0
    mask[:, :, 4:12, 9:11] = 1.0
    mask[:, :, 7:8, 4:11] = 1.0

    thin_weight, gap_weight = compute_mask_loss_weights(mask, thin_boost=6.0, gap_boost=4.0)

    assert thin_weight is not None
    assert gap_weight is not None
    assert float(thin_weight.max()) == 6.0
    assert float(gap_weight.max()) == 4.0
    assert float(thin_weight[0, 0, 7, 5]) == 6.0
    assert float(gap_weight[0, 0, 6, 7]) == 4.0
    assert float(gap_weight[0, 0, 0, 0]) == 1.0


def test_contour_sr_loss_finite_under_cuda_amp() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("CUDA required for AMP SSIM regression test")

    loss = ContourSRLoss(
        grad_vector_weight=0.3,
        laplacian_weight=0.1,
        forward_model_weight=0.1,
    )
    model = ThermalSRUNet(in_channels=5, out_channels=1, base_channels=16, scale=2).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler(enabled=True)
    obs = torch.randn(2, 5, 32, 32, device=device)
    target = torch.randn(2, 1, 64, 64, device=device)

    with autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        pred = model(obs)
        values = loss(pred, target, lr_observation=obs[:, 0:1])

    assert all(torch.isfinite(value) for value in values.values())
    scaler.scale(values["total"]).backward()
    scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


def test_forward_model_loss_band_full_matches_legacy() -> None:
    """band='full' must produce identical results to the pre-V9B code path."""
    from unet_sr.losses import forward_model_loss

    pred = torch.randn(2, 1, 64, 64)
    obs = torch.randn(2, 1, 32, 32)
    loss_full = forward_model_loss(pred, obs, scale=2, psf_sigma_lr_px=0.5, band="full")
    loss_default = forward_model_loss(pred, obs, scale=2, psf_sigma_lr_px=0.5)
    assert torch.allclose(loss_full, loss_default)


def test_forward_model_loss_highpass_ignores_dc_offset() -> None:
    """Highpass band: pure DC offset should yield near-zero loss."""
    from unet_sr.losses import forward_model_loss

    pred = torch.ones(1, 1, 128, 128) * 5.0
    obs = torch.ones(1, 1, 64, 64) * 5.0
    loss = forward_model_loss(pred, obs, scale=2, psf_sigma_lr_px=0.5,
                              band="highpass", band_sigma_lr_px=5.0)
    assert loss.item() < 1e-6


def test_forward_model_loss_highpass_detects_hf_perturbation() -> None:
    """Highpass band: high-frequency perturbation should produce significant loss."""
    from unet_sr.losses import forward_model_loss

    pred = torch.randn(1, 1, 128, 128)
    obs = torch.zeros(1, 1, 64, 64)
    loss = forward_model_loss(pred, obs, scale=2, psf_sigma_lr_px=0.5,
                              band="highpass", band_sigma_lr_px=5.0)
    assert loss.item() > 1e-4


def test_forward_model_loss_highpass_invariant_to_dc_shift() -> None:
    """Highpass band: adding DC offset to both pred and obs doesn't change loss."""
    from unet_sr.losses import forward_model_loss

    pred = torch.randn(1, 1, 128, 128)
    obs = torch.randn(1, 1, 64, 64)
    loss_base = forward_model_loss(pred, obs, scale=2, psf_sigma_lr_px=0.5,
                                   band="highpass", band_sigma_lr_px=5.0)
    loss_shifted = forward_model_loss(pred + 10.0, obs + 10.0, scale=2, psf_sigma_lr_px=0.5,
                                      band="highpass", band_sigma_lr_px=5.0)
    assert torch.allclose(loss_base, loss_shifted, atol=1e-5)


def test_contour_sr_loss_with_highpass_band_forward_model() -> None:
    """V9B: ContourSRLoss with band='highpass' and forward_model_weight > 0 produces finite loss."""
    loss = ContourSRLoss(
        forward_model_weight=0.1,
        forward_model_band="highpass",
        forward_model_band_sigma=5.0,
    )
    pred = torch.randn(2, 1, 128, 128, requires_grad=True)
    target = torch.randn(2, 1, 128, 128)
    obs = torch.randn(2, 1, 64, 64)

    values = loss(pred, target, lr_observation=obs)
    values["total"].backward()

    assert "forward_model" in values
    assert all(torch.isfinite(v) for v in values.values())
    assert pred.grad is not None


def test_forward_model_loss_highpass_cuda_amp() -> None:
    """V9B: highpass band forward model loss is finite under CUDA AMP."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("CUDA required for AMP test")

    from unet_sr.losses import forward_model_loss

    pred = torch.randn(2, 1, 128, 128, device=device)
    obs = torch.randn(2, 1, 64, 64, device=device)
    with autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        loss = forward_model_loss(pred, obs, scale=2, psf_sigma_lr_px=0.5,
                                  band="highpass", band_sigma_lr_px=5.0)
    assert torch.isfinite(loss)


def test_amp_forward_backward() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = device.type == "cuda"
    model = ThermalSRUNet(in_channels=5, out_channels=1, base_channels=16, scale=4).to(device)
    loss = ThermalSRLoss(edge_weight=0.1, ssim_weight=0.1)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler(enabled=use_amp)
    obs = torch.randn(2, 5, 16, 16, device=device)
    target = torch.randn(2, 1, 64, 64, device=device)
    edge_mask = torch.zeros_like(target)

    with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
        pred = model(obs)
        values = loss(pred, target, edge_mask=edge_mask)

    assert pred.shape == target.shape
    assert all(torch.isfinite(value) for value in values.values())

    scaler.scale(values["total"]).backward()
    scaler.unscale_(optimizer)
    # GradScaler handles NaN grads by skipping the step — this is expected AMP behaviour.
    # Verify the step completes without raising:
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
