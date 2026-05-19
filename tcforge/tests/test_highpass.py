from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import pytest
import tcforge.highpass as highpass


def test_highpass_constant_frame_is_zero_and_float32() -> None:
    constant = np.full((10, 10), 21.0, dtype=np.float32)
    hp = highpass.highpass_preprocess(constant, sigma_bg=5.0)

    assert hp.dtype == np.float32
    assert np.abs(hp).max() < 1e-5


def test_highpass_2d_and_3d_single_frame_match() -> None:
    frame_2d = np.random.default_rng(0).random((16, 18)).astype(np.float32)
    hp_2d = highpass.highpass_preprocess(frame_2d, sigma_bg=5.0)
    hp_3d = highpass.highpass_preprocess(frame_2d[np.newaxis, ...], sigma_bg=5.0)[0]

    assert hp_2d.dtype == np.float32
    assert hp_3d.dtype == np.float32
    assert np.allclose(hp_2d, hp_3d)


def test_highpass_matches_ep06_nearest_mode_reference() -> None:
    rng = np.random.default_rng(42)
    frames = rng.normal(size=(3, 14, 15)).astype(np.float32)
    expected = frames - gaussian_filter(frames, sigma=(0.0, 2.0, 2.0), mode="nearest")
    observed = highpass.highpass_preprocess(frames, sigma_bg=2.0, workers=1, mode="nearest")

    assert observed.dtype == np.float32
    assert np.allclose(observed, expected.astype(np.float32), atol=1e-6)


def test_highpass_rejects_non_image_rank() -> None:
    with pytest.raises(ValueError, match="2D or 3D"):
        highpass.highpass_preprocess(np.zeros((1, 2, 3, 4), dtype=np.float32))
