from __future__ import annotations

import sys
import importlib
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest
import tcforge.forward as tc_forward

forward_ref = importlib.import_module("tcforge._ep06_reference.forward")


def test_forward_uses_alignment_shift_to_predict_raw_observation() -> None:
    hr = np.zeros((8, 8), dtype=float)
    hr[2, 3] = 1.0

    lr = forward_ref.forward(hr, shift=(0.5, 0.0), psf_sigma=0.0)

    assert lr.shape == (4, 4)
    assert lr[1, 1] == 1.0
    assert np.count_nonzero(lr) == 1


def test_adjoint_backprojects_with_positive_alignment_shift() -> None:
    residual = np.zeros((4, 4), dtype=float)
    residual[1, 1] = 2.0

    backprojected = forward_ref.adjoint(residual, shift=(0.5, 0.0), psf_sigma=0.0, hr_shape=(8, 8))

    assert backprojected.shape == (8, 8)
    assert backprojected[2, 3] == 2.0
    assert np.count_nonzero(backprojected) == 1


def test_forward_and_adjoint_are_transposes_for_point_model() -> None:
    rng = np.random.default_rng(42)
    x_hr = rng.normal(size=(20, 24))
    y_lr = rng.normal(size=(10, 12))
    shift = np.array([0.23, 0.37])

    lhs = float(np.vdot(forward_ref.forward(x_hr, shift, psf_sigma=0.0), y_lr))
    rhs = float(np.vdot(x_hr, forward_ref.adjoint(y_lr, shift, psf_sigma=0.0, hr_shape=x_hr.shape)))

    assert np.isclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_forward_model_with_psf_returns_finite_lr_frame() -> None:
    rng = np.random.default_rng(7)
    x_hr = rng.normal(size=(18, 22))

    lr = forward_ref.forward(x_hr, shift=(0.15, 0.4), psf_sigma=0.35)
    bp = forward_ref.adjoint(lr, shift=(0.15, 0.4), psf_sigma=0.35, hr_shape=x_hr.shape)

    assert lr.shape == (9, 11)
    assert bp.shape == x_hr.shape
    assert np.isfinite(lr).all()
    assert np.isfinite(bp).all()


def test_ep06_reference_block_average_and_adjoint_support_scale_4() -> None:
    hr = np.arange(8 * 12, dtype=float).reshape(8, 12)
    lr = np.arange(2 * 3, dtype=float).reshape(2, 3)

    down = forward_ref.downsample_block_average(hr, scale=4)
    up = forward_ref.upsample_block_adjoint(lr, scale=4)

    assert np.allclose(down, hr.reshape(2, 4, 3, 4).mean(axis=(1, 3)))
    assert up.shape == (8, 12)
    assert np.allclose(up[:4, :4], lr[0, 0] / 16.0)
    assert np.allclose(up[4:, 8:], lr[1, 2] / 16.0)


def test_ep06_reference_point_forward_and_adjoint_support_scale_4() -> None:
    rng = np.random.default_rng(123)
    x_hr = rng.normal(size=(16, 20))
    y_lr = rng.normal(size=(4, 5))
    shift = np.array([0.2, 0.35])

    sampled = forward_ref.forward(x_hr, (0.0, 0.0), psf_sigma=0.0, scale=4)
    lhs = float(np.vdot(forward_ref.forward(x_hr, shift, psf_sigma=0.0, scale=4), y_lr))
    rhs = float(np.vdot(x_hr, forward_ref.adjoint(y_lr, shift, psf_sigma=0.0, hr_shape=x_hr.shape, scale=4)))

    assert sampled.shape == (4, 5)
    assert np.allclose(sampled, x_hr[::4, ::4])
    assert np.isclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_observation_operator_validates_lr_shape_and_stack_count() -> None:
    shifts = np.asarray([[0.0, 0.0], [0.5, 0.25]], dtype=float)
    op = forward_ref.build_observation_operator((12, 14), shifts=shifts, psf_sigma=0.2)
    x = np.ones((12, 14), dtype=float)
    burst = op.forward_all(x)

    assert burst.shape == (2, 6, 7)
    assert np.isfinite(burst).all()
    with pytest.raises(ValueError, match="lr_shape"):
        forward_ref.build_observation_operator((12, 14), lr_shape=(7, 7), shifts=shifts)


def test_generate_lr_burst_honors_distinct_forward_modes() -> None:
    hr = np.zeros((8, 8), dtype=np.float32)
    hr[2:4, 2:4] = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    shifts = np.asarray([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32)

    point = tc_forward.generate_lr_burst(hr, shifts, forward_mode="exact_ep06_point", psf_sigma_lr_px=0.0)
    block = tc_forward.generate_lr_burst(hr, shifts, forward_mode="physical_block_average", psf_sigma_lr_px=0.0)

    assert point.shape == (2, 4, 4)
    assert block.shape == (2, 4, 4)
    assert point.dtype == np.float32
    assert block.dtype == np.float32
    assert np.isfinite(point).all()
    assert np.isfinite(block).all()
    assert not np.allclose(point, block)


def test_generate_lr_burst_exact_point_accepts_scale_4() -> None:
    hr = np.zeros((16, 20), dtype=np.float32)
    hr[4:8, 8:12] = 1.0
    shifts = np.asarray([[0.0, 0.0], [0.25, 0.5]], dtype=np.float32)

    burst = tc_forward.generate_lr_burst(
        hr,
        shifts,
        forward_mode="exact_ep06_point",
        psf_sigma_lr_px=0.0,
        scale=4,
    )

    assert burst.shape == (2, 4, 5)
    assert burst.dtype == np.float32
    assert np.isfinite(burst).all()


def test_physical_block_average_accepts_elliptical_psf_shape() -> None:
    hr = np.zeros((32, 40), dtype=np.float32)
    hr[12:20, 16:24] = 1.0
    shifts = np.asarray([[0.0, 0.0], [0.25, 0.5]], dtype=np.float32)

    burst = tc_forward.generate_lr_burst(
        hr,
        shifts,
        forward_mode="physical_block_average",
        psf_sigma_lr_px=0.25,
        psf_shape="elliptical_gaussian",
        psf_sigma_y_lr_px=0.45,
        psf_angle_deg=35.0,
        scale=4,
        workers=1,
    )

    assert burst.shape == (2, 8, 10)
    assert burst.dtype == np.float32
    assert np.isfinite(burst).all()
