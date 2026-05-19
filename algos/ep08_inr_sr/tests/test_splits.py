from __future__ import annotations

import numpy as np

from ep08.splits import build_train_val_split


def test_build_train_val_split_is_bit_exact_for_seed() -> None:
    frames = np.zeros((20, 2, 2), dtype=np.float32)
    phases = np.array(
        [
            [0.05, 0.05],
            [0.10, 0.10],
            [0.15, 0.15],
            [0.20, 0.20],
            [0.30, 0.05],
            [0.32, 0.10],
            [0.34, 0.15],
            [0.36, 0.20],
            [0.05, 0.30],
            [0.10, 0.32],
            [0.15, 0.34],
            [0.20, 0.36],
            [0.30, 0.30],
            [0.32, 0.32],
            [0.34, 0.34],
            [0.36, 0.36],
            [0.60, 0.60],
            [0.62, 0.62],
            [0.64, 0.64],
            [0.90, 0.90],
        ],
        dtype=np.float64,
    )
    shifts = phases + np.array([2.0, -1.0])

    train, val, mask = build_train_val_split(frames, shifts, val_ratio=0.25, seed=42)

    np.testing.assert_array_equal(val, np.array([3, 7, 8, 14, 16], dtype=np.int64))
    np.testing.assert_array_equal(train, np.array([0, 1, 2, 4, 5, 6, 9, 10, 11, 12, 13, 15, 17, 18, 19]))
    expected_mask = np.zeros(20, dtype=bool)
    expected_mask[val] = True
    np.testing.assert_array_equal(mask, expected_mask)


def test_small_phase_bins_remain_in_train() -> None:
    frames = np.zeros((5, 2, 2), dtype=np.float32)
    shifts = np.array([[0.1, 0.1], [0.2, 0.2], [0.6, 0.6], [0.7, 0.7], [0.8, 0.8]])

    train, val, mask = build_train_val_split(frames, shifts, val_ratio=0.5, seed=1)

    np.testing.assert_array_equal(train, np.arange(5))
    assert len(val) == 0
    assert not mask.any()
