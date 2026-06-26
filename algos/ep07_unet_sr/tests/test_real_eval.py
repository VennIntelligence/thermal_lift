"""Tests for real-data TensorBoard eval helpers."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from unet_sr.real_eval import (
    _diverging_rgb,
    _select_solver_eval_frames,
    _temperature_rgb,
    center_fraction_crop,
    infer_solver_from_burst,
    zoom_center,
)


def test_center_fraction_crop_and_zoom() -> None:
    image = np.arange(90, dtype=np.float32).reshape(9, 10)
    crop = center_fraction_crop(image, fraction=1.0 / 3.0)
    assert crop.shape == (3, 3)
    zoomed = zoom_center(image, center_fraction=1.0 / 3.0, zoom=2.0)
    assert zoomed.shape == (6, 6)


def test_diverging_rgb_shape() -> None:
    panel = np.zeros((32, 64), dtype=np.float32)
    rgb = _diverging_rgb(panel, vmax=1.0)
    assert rgb.shape == (3, 32, 64)


def test_temperature_rgb_uses_inferno() -> None:
    image = np.linspace(20.0, 25.0, 100, dtype=np.float32).reshape(10, 10)
    rgb = _temperature_rgb(image)
    assert rgb.shape == (3, 10, 10)
    assert rgb.max() <= 1.0 + 1e-6
    assert rgb.min() >= -1e-6


def test_select_solver_eval_frames_is_deterministic_subset() -> None:
    burst = np.arange(7 * 3 * 4, dtype=np.float32).reshape(7, 3, 4)
    shifts = np.arange(14, dtype=np.float32).reshape(7, 2)

    sub_burst, sub_shifts = _select_solver_eval_frames(burst, shifts, m_frames=3)

    assert sub_burst.shape == (3, 3, 4)
    assert sub_shifts.shape == (3, 2)
    np.testing.assert_array_equal(sub_shifts, shifts[[0, 3, 6]])


class RecordingSolver(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.cond_channels: list[int] = []
        self.burst_frames: list[int] = []

    def forward(
        self,
        x0: torch.Tensor,
        y_burst: torch.Tensor,
        shifts: torch.Tensor,
        psf,
        cond: torch.Tensor,
        *,
        frame_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.cond_channels.append(int(cond.shape[1]))
        self.burst_frames.append(int(y_burst.shape[1]))
        assert shifts.shape[1:] == (y_burst.shape[1], 2)
        assert frame_mask is not None
        return x0


def _solver_eval_config(*, solver_no_drizzle: bool) -> SimpleNamespace:
    return SimpleNamespace(
        scale=2,
        highpass_sigma=1.0,
        solver_no_drizzle=solver_no_drizzle,
        solver_m_frames=3,
        solver_dc_rim_lr_px=1,
        forward_model_psf_sigma=0.5,
        phase_bin_channels=4,
    )


def test_infer_solver_from_burst_no_drizzle_uses_5ch_contract() -> None:
    solver = RecordingSolver()
    burst = np.ones((7, 8, 10), dtype=np.float32) * 21.0
    shifts = np.linspace(-0.2, 0.2, 14, dtype=np.float32).reshape(7, 2)

    out = infer_solver_from_burst(
        solver,
        burst,
        shifts,
        training_config=_solver_eval_config(solver_no_drizzle=True),
        patch_size_hr=8,
        overlap=2,
        device="cpu",
    )

    assert out.shape == (16, 20)
    assert np.isfinite(out).all()
    assert set(solver.cond_channels) == {5}
    assert set(solver.burst_frames) == {3}


def test_infer_solver_from_burst_hybrid_uses_9ch_contract() -> None:
    solver = RecordingSolver()
    burst = np.ones((7, 8, 10), dtype=np.float32) * 21.0
    shifts = np.linspace(-0.2, 0.2, 14, dtype=np.float32).reshape(7, 2)

    out = infer_solver_from_burst(
        solver,
        burst,
        shifts,
        training_config=_solver_eval_config(solver_no_drizzle=False),
        patch_size_hr=8,
        overlap=2,
        device="cpu",
    )

    assert out.shape == (16, 20)
    assert np.isfinite(out).all()
    assert set(solver.cond_channels) == {9}
    assert set(solver.burst_frames) == {3}
