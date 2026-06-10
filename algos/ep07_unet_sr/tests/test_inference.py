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
