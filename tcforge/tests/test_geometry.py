from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tcforge.geometry as geometry


def test_geometry_defaults_remain_2x_and_exposes_4x_constants() -> None:
    assert geometry.DEFAULT_CANVAS_SHAPE == (960, 1280)
    assert geometry.DEFAULT_SCALE == 2
    assert geometry.CANVAS_SHAPE_4X == (1920, 2560)
    assert geometry.SCALE_4X == 4


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


def test_curved_pad_primitives_are_binary_and_seedless_deterministic() -> None:
    circle = geometry.make_circle_pad(120, 100, 40, canvas_shape=(64, 80), pixel_size_um=10, scale=2)
    ellipse = geometry.make_ellipse_pad(
        120,
        100,
        60,
        30,
        angle_deg=35,
        canvas_shape=(64, 80),
        pixel_size_um=10,
        scale=2,
    )
    vias = geometry.make_via_array(
        2,
        3,
        35,
        16,
        120,
        100,
        stagger=True,
        canvas_shape=(64, 80),
        pixel_size_um=10,
        scale=2,
    )

    for mask in (circle, ellipse, vias):
        assert mask.dtype == np.uint8
        assert mask.shape == (64, 80)
        assert set(np.unique(mask).tolist()) <= {0, 1}
        assert mask.sum() > 0


def test_build_scene_mask_is_seed_reproducible_and_difficulty_sensitive() -> None:
    easy_a = geometry.build_scene_mask(
        "easy", 123, canvas_shape=(96, 128), pixel_size_um=10.0, scale=2, antialias=False,
    )
    easy_b = geometry.build_scene_mask(
        "easy", 123, canvas_shape=(96, 128), pixel_size_um=10.0, scale=2, antialias=False,
    )
    hard = geometry.build_scene_mask(
        "hard", 124, canvas_shape=(96, 128), pixel_size_um=10.0, scale=2, antialias=False,
    )

    assert easy_a.shape == (96, 128)
    assert easy_a.dtype == np.uint8
    assert np.array_equal(easy_a, easy_b)
    assert not np.array_equal(easy_a, hard)


def test_build_scene_mask_defaults_to_soft_antialiased_coverage() -> None:
    mask_a, meta_a = geometry.build_scene_mask_with_metadata(
        "hard",
        123,
        canvas_shape=(96, 128),
        pixel_size_um=10.0,
        scale=2,
    )
    mask_b, _meta_b = geometry.build_scene_mask_with_metadata(
        "hard",
        123,
        canvas_shape=(96, 128),
        pixel_size_um=10.0,
        scale=2,
    )

    partial = (mask_a > 0.0) & (mask_a < 1.0)
    assert mask_a.shape == (96, 128)
    assert mask_a.dtype == np.float32
    assert 0.0 <= float(mask_a.min()) <= float(mask_a.max()) <= 1.0
    assert int(partial.sum()) > 0
    assert np.array_equal(mask_a, mask_b)
    assert meta_a["antialias"] is True
    assert meta_a["ssaa_factor"] == 4
    assert meta_a["mask_semantics"] == "coverage"


def test_downsample_coverage_keeps_diagonal_edge_monotonic_and_partial() -> None:
    super_mask = np.zeros((32, 32), dtype=np.float32)
    yy, xx = np.mgrid[:32, :32]
    super_mask[xx >= yy] = 1.0

    coverage = geometry._downsample_coverage(super_mask, 4)
    profile = coverage[4]

    assert coverage.shape == (8, 8)
    assert coverage.dtype == np.float32
    assert np.any((coverage > 0.0) & (coverage < 1.0))
    assert np.all(np.diff(profile) >= -1e-6)


def test_build_scene_mask_with_metadata_accepts_explicit_4x_canvas() -> None:
    mask_a, meta_a = geometry.build_scene_mask_with_metadata(
        "medium",
        42,
        canvas_shape=(128, 160),
        pixel_size_um=10.0,
        scale=4,
        antialias=False,
    )
    mask_b, _meta_b = geometry.build_scene_mask_with_metadata(
        "medium",
        42,
        canvas_shape=(128, 160),
        pixel_size_um=10.0,
        scale=4,
        antialias=False,
    )

    assert mask_a.shape == (128, 160)
    assert mask_a.dtype == np.uint8
    assert set(np.unique(mask_a).tolist()) <= {0, 1}
    assert np.array_equal(mask_a, mask_b)
    assert meta_a["difficulty"] == "medium"
    primitive_types = {str(item["type"]) for item in meta_a["primitives"]}  # type: ignore[index]
    assert primitive_types & {"circle_pad", "ellipse_pad", "via_array"}


def test_multi_temp_mask_and_edge_diffusion_keep_shape_and_ranges() -> None:
    labels, meta = geometry.build_multi_temp_mask_with_metadata(
        "hard",
        9,
        n_temp_levels=4,
        canvas_shape=(96, 128),
        pixel_size_um=10.0,
        scale=4,
    )
    soft = geometry.apply_edge_diffusion(labels > 0, sigma_um=5.0, pixel_size_um=10.0, scale=4)

    assert labels.shape == (96, 128)
    assert labels.dtype == np.uint8
    assert set(np.unique(labels).tolist()) <= {0, 1, 2, 3}
    assert int(meta["n_temp_levels"]) == 4
    assert soft.shape == labels.shape
    assert soft.dtype == np.float32
    assert 0.0 <= float(soft.min()) <= float(soft.max()) <= 1.0
