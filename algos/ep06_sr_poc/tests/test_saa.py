from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibp import forward
from saa import reconstruct_saa, saa_uniform, saa_weighted


def _scene(shape: tuple[int, int] = (64, 72)) -> np.ndarray:
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    img = 0.22 * np.sin(2 * np.pi * x / shape[1])
    img += 0.18 * np.cos(2 * np.pi * y / shape[0])
    img += 0.55 * np.exp(-((x - 0.35 * shape[1]) ** 2 + (y - 0.35 * shape[0]) ** 2) / 95.0)
    img += 0.45 * ((x > 0.58 * shape[1]) & (y > 0.45 * shape[0]))
    img -= img.min()
    img /= img.max()
    return img


def _phase_shifts(n_frames: int, seed: int = 1) -> np.ndarray:
    rng = np.random.default_rng(seed)
    phases = np.array(
        [
            [0.0, 0.0],
            [0.5, 0.0],
            [0.0, 0.5],
            [0.5, 0.5],
        ],
        dtype=float,
    )
    shifts = phases[np.arange(n_frames) % len(phases)]
    shifts = shifts + rng.normal(scale=0.015, size=shifts.shape)
    return np.clip(shifts, 0.0, 0.5)


def _psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    mse = float(np.mean((reference - estimate) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-15)))


def test_saa_uniform_reconstructs_2x_reference_grid() -> None:
    hr = _scene()
    shifts = _phase_shifts(32)
    frames = np.stack([forward(hr, shift, psf_sigma=0.0) for shift in shifts])
    frames += np.random.default_rng(2).normal(scale=0.003, size=frames.shape)

    baseline = ndimage.zoom(np.mean(frames, axis=0), zoom=(2, 2), order=3)
    recon = saa_uniform(frames, shifts, workers=1)

    assert recon.shape == hr.shape
    assert np.isfinite(recon).all()
    assert _psnr(hr, recon) > _psnr(hr, baseline) + 4.0
    assert _psnr(hr, recon) > 25.0


def test_saa_weighted_downweights_noisy_frames() -> None:
    hr = _scene()
    shifts = _phase_shifts(40, seed=3)
    clean_frames = np.stack([forward(hr, shift, psf_sigma=0.0) for shift in shifts])

    rng = np.random.default_rng(4)
    noise_sigma = np.full(shifts.shape[0], 0.004)
    noise_sigma[::3] = 0.05
    frames = clean_frames + rng.normal(scale=noise_sigma[:, None, None], size=clean_frames.shape)
    weights = 1.0 / (noise_sigma * noise_sigma)
    weights /= weights.max()

    uniform = reconstruct_saa(frames, shifts, workers=1)
    weighted = saa_weighted(frames, shifts, weights=weights, n_jobs=2)

    assert weighted.shape == hr.shape
    assert np.isfinite(weighted).all()
    assert _psnr(hr, weighted) >= _psnr(hr, uniform) + 0.5
