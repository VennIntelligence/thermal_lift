from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcforge.classical_sr import DRIZZLE_CH_COVERAGE, DRIZZLE_CH_MEAN, DRIZZLE_CH_VARIANCE, drizzle_features


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
