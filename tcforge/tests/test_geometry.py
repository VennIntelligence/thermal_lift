from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tcforge.geometry as geometry


def test_rectangle_mask_uses_um_units_and_uint8_binary_values() -> None:
    mask = geometry.make_rectangle(
        cx_um=100.0,
        cy_um=80.0,
        w_um=40.0,
        h_um=20.0,
        canvas_shape=(64, 80),
        pixel_size_um=10.0,
        scale=2,
    )
    assert mask.shape == (64, 80)
    assert mask.dtype == np.uint8
    assert set(np.unique(mask).tolist()) <= {0, 1}
    assert mask.sum() > 0


def test_composite_and_rotate_preserve_binary_mask_contract() -> None:
    a = geometry.make_rectangle(100, 100, 40, 40, canvas_shape=(64, 64), pixel_size_um=10, scale=2)
    b = geometry.make_cross(150, 120, 20, 60, canvas_shape=(64, 64), pixel_size_um=10, scale=2)
    combined = geometry.composite(a, b, canvas_shape=(64, 64))
    rotated = geometry.rotate_mask(combined, 47.6)

    assert combined.dtype == np.uint8
    assert rotated.dtype == np.uint8
    assert set(np.unique(rotated).tolist()) <= {0, 1}
    assert rotated.shape == combined.shape
    assert rotated.sum() > 0


def test_build_scene_mask_is_seed_reproducible_and_difficulty_sensitive() -> None:
    easy_a = geometry.build_scene_mask("easy", 123, canvas_shape=(96, 128), pixel_size_um=10.0, scale=2)
    easy_b = geometry.build_scene_mask("easy", 123, canvas_shape=(96, 128), pixel_size_um=10.0, scale=2)
    hard = geometry.build_scene_mask("hard", 124, canvas_shape=(96, 128), pixel_size_um=10.0, scale=2)

    assert easy_a.shape == (96, 128)
    assert easy_a.dtype == np.uint8
    assert np.array_equal(easy_a, easy_b)
    assert not np.array_equal(easy_a, hard)
