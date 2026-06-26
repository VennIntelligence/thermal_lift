from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from unet_sr.inference import infer_from_burst, infer_full_frame


class UpsampleFirstChannel(torch.nn.Module):
    def __init__(self, scale: int = 4) -> None:
        super().__init__()
        self.scale = scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x[:, :1], scale_factor=self.scale, mode="bilinear", align_corners=False)


def test_infer_full_frame_tiled_shape() -> None:
    model = UpsampleFirstChannel(scale=4)
    obs = np.zeros((5, 8, 10), dtype=np.float32)
    obs[0] = np.arange(80, dtype=np.float32).reshape(8, 10)

    out = infer_full_frame(model, obs, scale=4, patch_size_hr=16, overlap=4, device="cpu")

    assert out.shape == (32, 40)
    assert np.isfinite(out).all()


def test_infer_from_burst_fuses_before_inference() -> None:
    model = UpsampleFirstChannel(scale=4)
    burst = np.ones((3, 8, 10), dtype=np.float32) * 21.0
    shifts = np.zeros((3, 2), dtype=np.float32)

    out = infer_from_burst(model, burst, shifts, scale=4, patch_size_hr=16, overlap=4, device="cpu")

    assert out.shape == (32, 40)
    assert np.allclose(out, 21.0, atol=1e-5)


class IdentityRefine(torch.nn.Module):
    """Cch → 1ch identity: returns first channel."""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :1]


class RequireChannels(torch.nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.channels = channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.channels:
            raise RuntimeError(f"expected {self.channels} channels, got {x.shape[1]}")
        return x[:, :1]


class ZeroRefine(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.zeros_like(x[:, :1])


def test_infer_from_burst_hybrid_drizzle2x_shape() -> None:
    """V9A: hybrid_drizzle2x inference produces correct HR output shape."""
    model = RequireChannels(9)
    burst = np.ones((40, 8, 10), dtype=np.float32) * 21.0
    shifts = np.random.default_rng(42).uniform(-0.3, 0.3, (40, 2)).astype(np.float32)

    out = infer_from_burst(
        model, burst, shifts,
        scale=2, patch_size_hr=8, overlap=2,
        device="cpu", input_mode="hybrid_drizzle2x",
    )

    assert out.shape == (16, 20)
    assert np.isfinite(out).all()


def test_infer_full_frame_residual_channel_adds_input_channel() -> None:
    model = ZeroRefine()
    obs = np.zeros((8, 7, 9), dtype=np.float32)
    anchor = np.arange(63, dtype=np.float32).reshape(7, 9) / 10.0
    obs[5] = anchor

    out = infer_full_frame(
        model,
        obs,
        scale=1,
        patch_size_hr=4,
        overlap=1,
        device="cpu",
        residual_channel=5,
    )

    assert out.shape == anchor.shape
    assert np.allclose(out, anchor, atol=1e-6)


def test_infer_full_frame_without_residual_channel_preserves_old_direct_path() -> None:
    model = ZeroRefine()
    obs = np.zeros((8, 7, 9), dtype=np.float32)
    obs[5] = 3.0

    out = infer_full_frame(model, obs, scale=1, patch_size_hr=4, overlap=1, device="cpu")

    assert out.shape == (7, 9)
    assert np.allclose(out, 0.0, atol=1e-6)
