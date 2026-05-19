from __future__ import annotations

import numpy as np
import torch

from ep08.forward import ForwardOperator
from ep08.metrics import (
    artifact_score,
    holdout_residual,
    p95_gradient,
    raw_control_agreement,
    split_half_nrmse,
    summarize_metrics,
)


def test_holdout_residual_is_finite_and_zero_for_matching_observations() -> None:
    x_hr = torch.arange(64, dtype=torch.float32).reshape(8, 8) / 10.0
    shifts = torch.tensor([[0.0, 0.0], [0.25, 0.5]], dtype=torch.float32)
    op = ForwardOperator(hr_shape=(8, 8), lr_shape=(4, 4), shifts=shifts, psf_sigma=0.0)
    observations = op.forward_all(x_hr)

    score = holdout_residual(x_hr, observations, op, indices=torch.tensor([0, 1]))

    assert np.isfinite(score)
    assert score == 0.0


def test_holdout_residual_increases_with_observation_error() -> None:
    x_hr = torch.ones((8, 8), dtype=torch.float32)
    shifts = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    op = ForwardOperator(hr_shape=(8, 8), lr_shape=(4, 4), shifts=shifts, psf_sigma=0.0)
    observations = op.forward_all(x_hr)

    clean = holdout_residual(x_hr, observations, op, noise_sigma=0.5)
    noisy = holdout_residual(x_hr, observations + 0.25, op, noise_sigma=0.5)

    assert clean < noisy
    assert np.isfinite(noisy)


def test_split_half_nrmse_directionality_and_no_nan() -> None:
    base = np.arange(16, dtype=np.float64).reshape(4, 4)
    small_error = base + 0.1
    large_error = base + np.linspace(0.0, 3.0, 16).reshape(4, 4)

    assert split_half_nrmse(base, base) == 0.0
    assert split_half_nrmse(base, small_error) < split_half_nrmse(base, large_error)
    assert np.isfinite(split_half_nrmse(torch.as_tensor(base), torch.as_tensor(small_error)))


def test_artifact_score_uses_laplacian_energy_and_mask() -> None:
    flat = np.ones((7, 7), dtype=np.float64)
    oscillatory = flat.copy()
    oscillatory[3, 3] = 3.0
    pin_mask = np.zeros_like(flat, dtype=bool)
    pin_mask[2:5, 2:5] = True

    assert artifact_score(flat, pin_mask) == 0.0
    assert artifact_score(oscillatory, pin_mask) > artifact_score(flat, pin_mask)
    assert np.isfinite(artifact_score(torch.as_tensor(oscillatory), None))


def test_raw_control_agreement_proxy_directionality() -> None:
    a = np.linspace(-1.0, 1.0, 25).reshape(5, 5)
    similar = a + 0.01
    inverted = -a

    same_score = raw_control_agreement(a, a)
    similar_score = raw_control_agreement(a, similar)
    inverted_score = raw_control_agreement(a, inverted)

    assert same_score == 1.0
    assert similar_score > inverted_score
    assert np.isfinite(similar_score)


def test_p95_gradient_and_summary_are_finite() -> None:
    image = np.zeros((6, 6), dtype=np.float64)
    image[:, 3:] = 2.0
    summary = summarize_metrics(split_a=image, split_b=image + 0.01, artifact_image=image, raw_a=image, raw_b=image)

    assert p95_gradient(image) > 0.0
    assert set(summary) == {"split_half_nrmse", "artifact_score", "raw_control_agreement"}
    assert all(np.isfinite(value) for value in summary.values())
