from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast

from unet_sr.losses import ContourSRLoss, ThermalSRLoss, forward_model_loss, sobel_edges
from unet_sr.model import ThermalSRUNet
from unet_sr.mask_weights import compute_boundary_weight_np
from unet_sr.train import _delta_l1_penalty


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


def test_contour_sr_loss_accepts_optional_boundary_weight() -> None:
    loss = ContourSRLoss()
    pred = torch.randn(2, 1, 32, 32, requires_grad=True)
    target = torch.randn(2, 1, 32, 32)
    boundary = torch.ones_like(target)
    boundary[:, :, 12:14, 12:20] = 5.0

    values = loss(pred, target, boundary_weight=boundary)
    values["total"].backward()

    assert set(values) == {"total", "mse", "highpass", "edge", "ssim", "grad_vector"}
    assert all(torch.isfinite(value) for value in values.values())
    assert pred.grad is not None


def test_contour_sr_loss_all_one_weight_matches_default_behavior() -> None:
    loss = ContourSRLoss()
    pred = torch.randn(1, 1, 24, 24)
    target = torch.randn(1, 1, 24, 24)
    ones = torch.ones_like(target)

    base = loss(pred, target)
    weighted = loss(pred, target, boundary_weight=ones)

    for key in base:
        assert torch.allclose(base[key], weighted[key])


def test_contour_sr_loss_weight_shape_mismatch_raises() -> None:
    loss = ContourSRLoss()
    pred = torch.randn(1, 1, 16, 16)
    target = torch.randn(1, 1, 16, 16)
    bad = torch.ones(1, 1, 8, 8)

    with pytest.raises(ValueError, match="boundary_weight shape mismatch"):
        loss(pred, target, boundary_weight=bad)


def test_residual_penalty_increases_with_delta_magnitude() -> None:
    base_total = torch.tensor(1.0)
    weight = 0.5
    small_delta = torch.ones(2, 1, 8, 8) * 0.1
    large_delta = torch.ones(2, 1, 8, 8) * 0.4

    small_penalty, _, _ = _delta_l1_penalty(small_delta)
    large_penalty, _, _ = _delta_l1_penalty(large_delta)
    small_total = base_total + weight * small_penalty
    large_total = base_total + weight * large_penalty

    assert large_penalty > small_penalty
    assert large_total > small_total


def test_boundary_weight_emphasises_every_edge_type() -> None:
    mask = np.zeros((24, 24), dtype=np.float32)
    mask[4:20, 4:20] = 1.0                              # a solid block
    yy, xx = np.mgrid[0:24, 0:24]
    mask[(yy - 12) ** 2 + (xx - 12) ** 2 <= 4] = 0.0    # a hole carved inside it

    w = compute_boundary_weight_np(mask, boundary_boost=4.0, tau_px=2.5)

    assert w is not None
    assert w.shape == (1, 24, 24)
    assert abs(float(w.max()) - 5.0) < 1e-4             # on a boundary -> 1 + boost
    assert 1.0 <= float(w.min()) < 1.1                  # decays to ~1 away from edges
    assert float(w[0, 4, 12]) > 4.0                     # outer block edge boosted
    assert float(w[0, 12, 9]) > 3.0                     # hole rim boosted
    # disabled boost -> None; degenerate (all-support / all-bg) patch -> uniform 1
    assert compute_boundary_weight_np(mask, boundary_boost=0.0) is None
    assert np.allclose(compute_boundary_weight_np(np.ones((8, 8), np.float32), boundary_boost=4.0), 1.0)


def test_contour_sr_loss_flatness_term_activates_and_is_finite() -> None:
    loss = ContourSRLoss(flatness_weight=0.1)
    pred = torch.randn(1, 1, 24, 24, requires_grad=True)
    target = torch.zeros(1, 1, 24, 24)        # mostly-flat GT -> flatness fully active
    target[:, :, 8:16, 8:16] = 1.0            # one block so a real edge exists too

    values = loss(pred, target)
    values["total"].backward()

    assert "flatness" in values
    assert torch.isfinite(values["flatness"])
    assert float(values["flatness"]) > 0.0    # random pred has gradient in flat regions
    assert pred.grad is not None


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


def test_contour_sr_loss_lr_obs_uses_explicit_2x_downsample() -> None:
    """V9C: hybrid model scale=1 still anchors pred to a 1x lr_obs via 2x downsample."""
    loss = ContourSRLoss(
        forward_model_weight=0.1,
        forward_model_scale=1,
        forward_model_band="highpass",
    )
    pred = torch.randn(2, 1, 64, 64, requires_grad=True)
    target = torch.randn(2, 1, 64, 64)
    lr_obs = torch.randn(2, 1, 32, 32)

    values = loss(pred, target, lr_obs=lr_obs)
    values["total"].backward()
    expected_fm = forward_model_loss(
        pred.detach(),
        lr_obs,
        scale=2,
        psf_sigma_lr_px=loss.forward_model_psf_sigma,
        band="highpass",
        band_sigma_lr_px=loss.forward_model_band_sigma,
    )

    assert "forward_model" in values
    assert torch.allclose(values["forward_model"].detach(), expected_fm)
    assert all(torch.isfinite(v) for v in values.values())
    assert pred.grad is not None


def test_contour_sr_loss_rejects_2x_hybrid_channel_as_lr_observation() -> None:
    """V9C: the upsampled hybrid channel is not a legal 1x forward-model observation."""
    loss = ContourSRLoss(forward_model_weight=0.1, forward_model_scale=2)
    pred = torch.randn(2, 1, 64, 64)
    target = torch.randn(2, 1, 64, 64)
    hybrid_ch0 = torch.randn(2, 1, 64, 64)

    with pytest.raises(ValueError, match="forward-model shape mismatch"):
        loss(pred, target, lr_observation=hybrid_ch0)


def test_hybrid_contour_sr_loss_finite_under_cuda_amp() -> None:
    """V9C: 9ch hybrid input + legal lr_obs forward anchor is finite under CUDA AMP."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("CUDA required for hybrid AMP regression test")

    model = ThermalSRUNet(in_channels=9, out_channels=1, base_channels=16, scale=1).to(device)
    loss = ContourSRLoss(
        grad_vector_weight=0.15,
        forward_model_weight=0.1,
        forward_model_scale=2,
        forward_model_band="highpass",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scaler = GradScaler(enabled=True)
    obs = torch.randn(2, 9, 64, 64, device=device)
    target = torch.randn(2, 1, 64, 64, device=device)
    lr_obs = torch.randn(2, 1, 32, 32, device=device)

    with autocast(device_type="cuda", dtype=torch.float16, enabled=True):
        pred = model(obs)
        values = loss(pred, target, lr_obs=lr_obs)

    assert all(torch.isfinite(value) for value in values.values())
    scaler.scale(values["total"]).backward()
    scaler.unscale_(optimizer)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)


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
