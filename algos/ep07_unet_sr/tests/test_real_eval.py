"""Tests for real-data TensorBoard eval helpers."""

from __future__ import annotations

import numpy as np

from unet_sr.real_eval import _diverging_rgb, _temperature_rgb, center_fraction_crop, zoom_center


def test_center_fraction_crop_and_zoom() -> None:
    image = np.arange(90, dtype=np.float32).reshape(9, 10)
    crop = center_fraction_crop(image, fraction=1.0 / 3.0)
    assert crop.shape == (3, 3)
    zoomed = zoom_center(image, center_fraction=1.0 / 3.0, zoom=2.0)
    assert zoomed.shape == (6, 6)


def test_diverging_rgb_shape() -> None:
    panel = np.zeros((32, 64), dtype=np.float32)
    rgb = _diverging_rgb(panel, vmax=1.0)
    assert rgb.shape == (3, 32, 64)


def test_temperature_rgb_uses_inferno() -> None:
    image = np.linspace(20.0, 25.0, 100, dtype=np.float32).reshape(10, 10)
    rgb = _temperature_rgb(image)
    assert rgb.shape == (3, 10, 10)
    assert rgb.max() <= 1.0 + 1e-6
    assert rgb.min() >= -1e-6
