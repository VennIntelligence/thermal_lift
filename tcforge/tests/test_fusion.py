from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcforge.fusion import fuse_burst_to_features


def test_fuse_burst_to_features_returns_expected_zero_shift_statistics() -> None:
    rng = np.random.default_rng(7)
    burst = rng.normal(21.0, 0.2, size=(6, 12, 16)).astype(np.float32)
    shifts = np.zeros((6, 2), dtype=np.float32)

    features = fuse_burst_to_features(burst, shifts, sigma_bg=2.0)

    assert features.shape == (5, 12, 16)
    assert features.dtype == np.float32
    assert np.allclose(features[0], burst.mean(axis=0), atol=1e-6)
    assert np.allclose(features[1], np.median(burst, axis=0), atol=1e-6)
    assert np.allclose(features[2], 1.0)
    assert np.all(features[3] >= 0.0)
    assert np.allclose(features[3], burst.var(axis=0), atol=1e-6)
    assert np.isfinite(features).all()


def test_fuse_burst_to_features_coverage_and_constant_highpass_contract() -> None:
    burst = np.full((3, 8, 10), 22.0, dtype=np.float32)
    shifts = np.asarray([[0.0, 0.0], [0.5, 0.0], [-1.0, 1.0]], dtype=np.float32)

    features = fuse_burst_to_features(burst, shifts, sigma_bg=1.5)
    coverage = features[2]

    assert features.shape == (5, 8, 10)
    assert np.all((coverage >= 0.0) & (coverage <= 1.0))
    assert np.all(features[3] >= 0.0)
    assert np.allclose(features[4], 0.0, atol=1e-6)
    assert np.isfinite(features).all()
