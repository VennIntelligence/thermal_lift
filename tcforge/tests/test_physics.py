from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tcforge.physics as physics


def test_temperature_field_respects_mask_semantics_and_bounds() -> None:
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[6:18, 8:24] = 1
    temp = physics.render_temperature_field(mask, t_bg_c=21.0, delta_t_c=2.5, low_freq_amplitude_c=0.0, seed=7)

    assert temp.shape == mask.shape
    assert temp.dtype == np.float32
    assert np.isfinite(temp).all()
    assert np.allclose(temp[mask == 0], 21.0)
    assert np.allclose(temp[mask == 1], 23.5)


def test_edge_map_is_binary_like_and_localized_to_mask_boundary() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    edge = physics.edge_map(mask)

    assert edge.shape == mask.shape
    assert np.isfinite(edge).all()
    assert edge.sum() > 0
    assert edge[16, 16] == 0


def test_gaussian_noise_is_seed_reproducible_and_has_expected_scale() -> None:
    frame = np.full((128, 128), 21.0, dtype=np.float32)
    noisy_a = physics.add_noise(frame, noise_sigma_c=0.0724, seed=11)
    noisy_b = physics.add_noise(frame, noise_sigma_c=0.0724, seed=11)

    assert noisy_a.dtype == np.float32
    assert np.array_equal(noisy_a, noisy_b)
    assert 0.05 < float(np.std(noisy_a - frame)) < 0.095
