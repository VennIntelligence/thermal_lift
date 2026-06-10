from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcforge.physics import render_temperature_field
from tcforge.reconstruct import reconstruct_hr_temperature


def test_reconstruct_hr_temperature_matches_physics_renderer() -> None:
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[4:18, 7:24] = 1

    actual = reconstruct_hr_temperature(
        mask,
        T_bg_c=20.5,
        delta_T_c=1.7,
        low_freq_amplitude_c=0.25,
        low_freq_sigma_px=5.0,
        seed=42,
    )
    expected = render_temperature_field(
        mask,
        t_bg_c=20.5,
        delta_t_c=1.7,
        low_freq_amplitude_c=0.25,
        low_freq_sigma_px=5.0,
        seed=42,
    )
    repeat = reconstruct_hr_temperature(
        mask,
        T_bg_c=20.5,
        delta_T_c=1.7,
        low_freq_amplitude_c=0.25,
        low_freq_sigma_px=5.0,
        seed=42,
    )

    assert actual.shape == mask.shape
    assert actual.dtype == np.float32
    assert np.array_equal(actual, expected)
    assert np.array_equal(actual, repeat)
