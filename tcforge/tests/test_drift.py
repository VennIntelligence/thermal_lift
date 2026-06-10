from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tcforge.highpass as highpass
import tcforge.physics as physics


def test_scalar_drift_degrades_raw_more_than_highpass_after_background_subtraction() -> None:
    rng = np.random.default_rng(5)
    base = rng.normal(0.0, 0.1, size=(32, 24, 24)).astype(np.float32)
    drifted = physics.apply_drift(base, model="scalar_offset", amplitude_c=0.2, seed=5)

    raw_delta = float(np.std(drifted - base))
    hp_delta = float(
        np.std(
            highpass.highpass_preprocess(drifted, sigma_bg=3.0)
            - highpass.highpass_preprocess(base, sigma_bg=3.0)
        )
    )

    assert drifted.shape == base.shape
    assert raw_delta > 0.02
    assert hp_delta < raw_delta


def test_lowfreq_drift_is_seed_reproducible_and_finite() -> None:
    frames = np.zeros((8, 32, 32), dtype=np.float32)
    a = physics.apply_drift(frames, model="lowfreq", amplitude_c=0.15, lowfreq_sigma_px=6.0, seed=12)
    b = physics.apply_drift(frames, model="lowfreq", amplitude_c=0.15, lowfreq_sigma_px=6.0, seed=12)

    assert a.dtype == np.float32
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()
    assert float(np.std(a)) > 0.0


def test_temporal_trend_drift_is_reproducible_and_changes_frame_mean_over_time() -> None:
    frames = np.zeros((10, 16, 16), dtype=np.float32)
    a = physics.apply_drift(frames, model="temporal_trend", amplitude_c=0.25, seed=22)
    b = physics.apply_drift(frames, model="temporal_trend", amplitude_c=0.25, seed=22)
    frame_means = a.mean(axis=(1, 2))

    assert a.dtype == np.float32
    assert np.array_equal(a, b)
    assert np.isfinite(a).all()
    assert abs(float(frame_means[-1] - frame_means[0])) > 0.01
