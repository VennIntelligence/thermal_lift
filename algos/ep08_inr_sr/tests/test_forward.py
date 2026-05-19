from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

EP06_SRC = Path(__file__).resolve().parents[2] / "ep06_sr_poc" / "src"
if str(EP06_SRC) not in sys.path:
    sys.path.insert(0, str(EP06_SRC))

from common import forward_model as ep06_forward
from ep08.forward import ForwardOperator, adjoint, forward


def _tensor(arr: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(arr, dtype=torch.float64)


def test_forward_matches_ep06_reference_without_psf() -> None:
    rng = np.random.default_rng(123)
    x_hr = rng.normal(size=(18, 20))
    shift = np.array([0.23, 0.37])

    expected = ep06_forward.forward(x_hr, shift, psf_sigma=0.0)
    actual = forward(_tensor(x_hr), _tensor(shift), psf_sigma=0.0).detach().cpu().numpy()

    assert np.max(np.abs(actual - expected)) < 1e-5


def test_forward_matches_ep06_reference_with_constant_psf() -> None:
    rng = np.random.default_rng(9)
    x_hr = rng.normal(size=(24, 22))
    shift = np.array([0.15, 0.4])

    expected = ep06_forward.forward(x_hr, shift, psf_sigma=0.35)
    actual = forward(_tensor(x_hr), _tensor(shift), psf_sigma=0.35).detach().cpu().numpy()

    assert np.max(np.abs(actual - expected)) < 1e-5


def test_adjoint_matches_ep06_reference_with_constant_psf() -> None:
    rng = np.random.default_rng(77)
    y_lr = rng.normal(size=(9, 11))
    shift = np.array([0.31, 0.18])
    hr_shape = (18, 22)

    expected = ep06_forward.adjoint(y_lr, shift, psf_sigma=0.45, hr_shape=hr_shape)
    actual = adjoint(_tensor(y_lr), _tensor(shift), psf_sigma=0.45, hr_shape=hr_shape).detach().cpu().numpy()

    assert np.max(np.abs(actual - expected)) < 1e-5


def test_forward_operator_matches_ep06_observation_operator() -> None:
    rng = np.random.default_rng(14)
    x_hr = rng.normal(size=(20, 24))
    shifts = np.array([[0.0, 0.0], [0.2, 0.35], [0.6, 0.15]], dtype=np.float64)

    ep06_op = ep06_forward.build_observation_operator(x_hr.shape, shifts=shifts, psf_sigma=0.25)
    ep08_op = ForwardOperator(x_hr.shape, (10, 12), _tensor(shifts), psf_sigma=0.25)

    actual = ep08_op.forward_all(_tensor(x_hr)).detach().cpu().numpy()
    expected = ep06_op.forward_all(x_hr)

    assert np.max(np.abs(actual - expected)) < 1e-5
