from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibp import adjoint, forward


def test_forward_uses_alignment_shift_to_predict_raw_observation() -> None:
    """A frame pixel shifted by +shift lands at the reference HR coordinate."""

    hr = np.zeros((8, 8), dtype=float)
    hr[2, 3] = 1.0

    # shift=(+0.5, 0) moves the LR frame right by one HR pixel on the
    # reference grid. The raw observation at LR (1, 1) samples reference
    # coordinate (2, 3), equivalent to an internal -shift scene move.
    lr = forward(hr, shift=(0.5, 0.0), psf_sigma=0.0)

    assert lr.shape == (4, 4)
    assert lr[1, 1] == 1.0
    assert np.count_nonzero(lr) == 1


def test_adjoint_backprojects_with_positive_alignment_shift() -> None:
    residual = np.zeros((4, 4), dtype=float)
    residual[1, 1] = 2.0

    backprojected = adjoint(
        residual,
        shift=(0.5, 0.0),
        psf_sigma=0.0,
        hr_shape=(8, 8),
    )

    assert backprojected.shape == (8, 8)
    assert backprojected[2, 3] == 2.0
    assert np.count_nonzero(backprojected) == 1


def test_forward_and_adjoint_are_transposes_for_point_model() -> None:
    rng = np.random.default_rng(42)
    x_hr = rng.normal(size=(20, 24))
    y_lr = rng.normal(size=(10, 12))
    shift = np.array([0.23, 0.37])

    lhs = float(np.vdot(forward(x_hr, shift, psf_sigma=0.0), y_lr))
    rhs = float(
        np.vdot(
            x_hr,
            adjoint(y_lr, shift, psf_sigma=0.0, hr_shape=x_hr.shape),
        )
    )

    assert np.isclose(lhs, rhs, rtol=1e-12, atol=1e-12)


def test_forward_model_with_psf_returns_finite_lr_frame() -> None:
    rng = np.random.default_rng(7)
    x_hr = rng.normal(size=(18, 22))

    lr = forward(x_hr, shift=(0.15, 0.4), psf_sigma=0.35)
    bp = adjoint(lr, shift=(0.15, 0.4), psf_sigma=0.35, hr_shape=x_hr.shape)

    assert lr.shape == (9, 11)
    assert bp.shape == x_hr.shape
    assert np.isfinite(lr).all()
    assert np.isfinite(bp).all()


def test_adjoint_default_splat_sigma_preserves_bilinear_path_at_4x() -> None:
    rng = np.random.default_rng(12)
    residual = rng.normal(size=(5, 6))
    shift = np.array([0.21, 0.34])

    implicit = adjoint(residual, shift, psf_sigma=0.0, hr_shape=(20, 24), scale=4)
    explicit = adjoint(
        residual,
        shift,
        psf_sigma=0.0,
        hr_shape=(20, 24),
        scale=4,
        splat_sigma=None,
    )

    assert np.array_equal(implicit, explicit)


def test_gaussian_adjoint_splat_expands_4x_support() -> None:
    residual = np.ones((3, 3), dtype=float)

    bilinear = adjoint(residual, (0.0, 0.0), psf_sigma=0.0, hr_shape=(12, 12), scale=4)
    gaussian = adjoint(
        residual,
        (0.0, 0.0),
        psf_sigma=0.0,
        hr_shape=(12, 12),
        scale=4,
        splat_sigma=1.5,
    )

    assert np.count_nonzero(gaussian) > np.count_nonzero(bilinear) * 4
    assert np.isfinite(gaussian).all()
