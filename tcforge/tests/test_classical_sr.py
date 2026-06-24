from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcforge.classical_sr import (
    DRIZZLE_CH_COVERAGE,
    DRIZZLE_CH_MEAN,
    DRIZZLE_CH_VARIANCE,
    drizzle_features,
    phase_bin_drizzle,
)


def test_drizzle_features_nearest_sparse_coverage_and_variance() -> None:
    burst = np.stack(
        [
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        ],
        axis=0,
    )
    shifts = np.zeros((2, 2), dtype=np.float32)

    features = drizzle_features(burst, shifts, scale=2, kernel="nearest")

    assert features.shape == (3, 4, 4)
    assert np.isclose(features[DRIZZLE_CH_MEAN, 0, 0], 1.0)
    assert np.isclose(features[DRIZZLE_CH_MEAN, 0, 2], 2.0)
    assert np.isclose(features[DRIZZLE_CH_COVERAGE, 0, 0], 1.0)
    assert features[DRIZZLE_CH_COVERAGE, 0, 1] == 0.0
    assert np.allclose(features[DRIZZLE_CH_VARIANCE], 0.0)


def test_drizzle_features_bilinear_accumulates_fractional_shift() -> None:
    burst = np.ones((1, 2, 2), dtype=np.float32)
    shifts = np.asarray([[0.25, 0.25]], dtype=np.float32)

    features = drizzle_features(burst, shifts, scale=4, kernel="bilinear")

    assert features.shape == (3, 8, 8)
    assert float(features[DRIZZLE_CH_COVERAGE].sum()) > 0.0
    observed = features[DRIZZLE_CH_COVERAGE] > 0
    assert np.allclose(features[DRIZZLE_CH_MEAN][observed], 1.0)


def test_phase_bin_drizzle_shape_and_reproducible() -> None:
    rng = np.random.default_rng(0)
    burst = rng.normal(1.0, 0.3, size=(16, 6, 8)).astype(np.float32)
    shifts = rng.uniform(-1.0, 1.0, size=(16, 2)).astype(np.float32)

    a = phase_bin_drizzle(burst, shifts, scale=2, n_bins=4)
    b = phase_bin_drizzle(burst, shifts, scale=2, n_bins=4)

    assert a.shape == (4, 12, 16)
    assert a.dtype == np.float32
    assert np.array_equal(a, b)


def test_phase_bin_drizzle_routes_known_phases_to_expected_bins() -> None:
    # g = 2; phase (frac_dy, frac_dx) → bin = row*2 + col.
    # Frame 0: dx=0.1, dy=0.1 → (row=0, col=0) → bin 0 only.
    # Frame 1: dx=0.6, dy=0.6 → (row=1, col=1) → bin 3 only.
    burst = np.stack(
        [np.full((4, 4), 2.0, np.float32), np.full((4, 4), 9.0, np.float32)],
        axis=0,
    )
    shifts = np.asarray([[0.1, 0.1], [0.6, 0.6]], dtype=np.float32)

    out = phase_bin_drizzle(burst, shifts, scale=2, n_bins=4)

    assert out.shape == (4, 8, 8)
    global_mean = float(burst.mean())
    # Empty bins (1, 2) are filled with the global burst mean — proving the
    # value-2 and value-9 frames were NOT routed there.
    assert np.allclose(out[1], global_mean)
    assert np.allclose(out[2], global_mean)
    # Bin 0 only saw the value-2 frame: its observed values never exceed 2.0.
    obs0 = out[0][out[0] > 0]
    assert obs0.size > 0
    assert float(obs0.max()) <= 2.0 + 1e-4
    # Bin 3 only saw the value-9 frame: it reaches up to 9.0 (>2.0).
    assert float(out[3].max()) > 2.0 + 1e-4
    assert float(out[3].max()) <= 9.0 + 1e-4


def test_phase_bin_drizzle_rejects_non_square_n_bins() -> None:
    burst = np.ones((4, 4, 4), dtype=np.float32)
    shifts = np.zeros((4, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        phase_bin_drizzle(burst, shifts, scale=2, n_bins=3)
