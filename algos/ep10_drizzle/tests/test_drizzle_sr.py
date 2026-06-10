from __future__ import annotations

import numpy as np
from scipy import ndimage

from common.forward_model import forward
from ep10_drizzle.drizzle_sr import build_pixmap, drizzle_reconstruct, psnr


def _scene(shape: tuple[int, int] = (64, 72)) -> np.ndarray:
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    img = 0.20 * np.sin(2 * np.pi * x / shape[1])
    img += 0.18 * np.cos(2 * np.pi * y / shape[0])
    img += 0.55 * np.exp(-((x - 0.35 * shape[1]) ** 2 + (y - 0.35 * shape[0]) ** 2) / 95.0)
    img += 0.45 * ((x > 0.58 * shape[1]) & (y > 0.45 * shape[0]))
    img -= img.min()
    img /= img.max()
    return img.astype(np.float32)


def test_build_pixmap_uses_ep06_dx_dy_axis_order() -> None:
    pixmap = build_pixmap((0.5, 0.25), lr_shape=(4, 5), scale=2)

    assert pixmap.shape == (4, 5, 2)
    assert pixmap[1, 2, 0] == 2 * (2 + 0.5)
    assert pixmap[1, 2, 1] == 2 * (1 + 0.25)


def test_build_pixmap_supports_exploratory_4x_grid() -> None:
    pixmap = build_pixmap((0.5, 0.25), lr_shape=(4, 5), scale=4)

    assert pixmap.shape == (4, 5, 2)
    assert pixmap[1, 2, 0] == 4 * (2 + 0.5)
    assert pixmap[1, 2, 1] == 4 * (1 + 0.25)


def test_drizzle_coordinate_matches_forward_model_point() -> None:
    hr = np.zeros((8, 8), dtype=np.float32)
    hr[2, 3] = 1.0
    shift = np.array([[0.5, 0.0]], dtype=np.float32)
    frame = forward(hr, shift[0], psf_sigma=0.0).astype(np.float32)

    recon, coverage = drizzle_reconstruct(
        frame[np.newaxis, ...],
        shift,
        pixfrac=0.01,
        coverage_threshold=0.0,
    )

    assert recon.shape == hr.shape
    assert coverage.shape == hr.shape
    assert np.unravel_index(np.nanargmax(recon), recon.shape) == (2, 3)


def test_drizzle_recovers_quadrature_phase_synthetic_scene() -> None:
    truth = _scene()
    phases = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
        ],
        dtype=np.float32,
    )
    shifts = np.tile(phases, (8, 1))
    frames = np.stack([forward(truth, shift, psf_sigma=0.0) for shift in shifts]).astype(np.float32)
    recon, _ = drizzle_reconstruct(frames, shifts, pixfrac=0.5, coverage_threshold=1.0)
    baseline = ndimage.zoom(np.mean(frames, axis=0), zoom=(2, 2), order=3)

    assert np.isfinite(recon).all()
    assert psnr(truth, recon) > psnr(truth, baseline) + 3.0
    assert psnr(truth, recon) > 25.0


def test_drizzle_4x_recovers_quadrature_phase_synthetic_scene() -> None:
    truth = _scene((64, 72))
    phases = np.array(
        [
            [0.0, 0.0],
            [0.25, 0.0],
            [0.5, 0.0],
            [0.75, 0.0],
            [0.0, 0.25],
            [0.25, 0.25],
            [0.5, 0.25],
            [0.75, 0.25],
            [0.0, 0.5],
            [0.25, 0.5],
            [0.5, 0.5],
            [0.75, 0.5],
            [0.0, 0.75],
            [0.25, 0.75],
            [0.5, 0.75],
            [0.75, 0.75],
        ],
        dtype=np.float32,
    )
    shifts = np.tile(phases, (4, 1))
    frames = np.stack([forward(truth, shift, psf_sigma=0.0, scale=4) for shift in shifts]).astype(np.float32)
    recon, coverage = drizzle_reconstruct(frames, shifts, scale=4, pixfrac=0.4, coverage_threshold=0.0)
    baseline = ndimage.zoom(np.mean(frames, axis=0), zoom=(4, 4), order=3)

    assert recon.shape == truth.shape
    assert coverage.shape == truth.shape
    assert np.isfinite(recon).all()
    assert psnr(truth, recon) > psnr(truth, baseline) + 3.0
