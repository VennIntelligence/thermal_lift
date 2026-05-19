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
