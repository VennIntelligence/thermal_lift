from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibp import forward, reconstruct_ibp
from saa import reconstruct_saa


def _scene(shape: tuple[int, int] = (48, 56)) -> np.ndarray:
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    img = 0.2 * np.sin(2 * np.pi * x / shape[1])
    img += 0.3 * np.exp(-((x - 18.0) ** 2 + (y - 17.0) ** 2) / 70.0)
    img += 0.4 * ((x > 30) & (y > 22))
    img -= img.min()
    img /= img.max()
    return img


def _shifts(n_frames: int) -> np.ndarray:
    phases = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
    return phases[np.arange(n_frames) % 4]


def _psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    mse = float(np.mean((reference - estimate) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-15)))


def test_ibp_returns_image_and_convergence_dataframe() -> None:
    hr = _scene()
    shifts = _shifts(24)
    frames = np.stack([forward(hr, shift, psf_sigma=0.25) for shift in shifts])
    frames += np.random.default_rng(5).normal(scale=0.0015, size=frames.shape)

    initial = reconstruct_saa(frames, shifts)
    baseline = ndimage.zoom(np.mean(frames, axis=0), zoom=(2, 2), order=3)
    recon, convergence = reconstruct_ibp(
        frames,
        shifts,
        initial=initial,
        max_iter=12,
        beta=0.9,
        tol=0.0,
        psf_sigma=0.25,
        workers=2,
    )

    assert recon.shape == hr.shape
    assert np.isfinite(recon).all()
    assert isinstance(convergence, pd.DataFrame)
    assert {"iteration", "residual_mse", "relative_update"}.issubset(convergence.columns)
    assert len(convergence) == 12
    assert convergence["residual_mse"].iloc[-1] <= convergence["residual_mse"].iloc[0]
    assert _psnr(hr, recon) > _psnr(hr, baseline) + 1.0
    assert _psnr(hr, recon) >= _psnr(hr, initial) - 0.25


def test_ibp_can_return_records() -> None:
    hr = _scene((24, 28))
    shifts = _shifts(8)
    frames = np.stack([forward(hr, shift, psf_sigma=0.0) for shift in shifts])

    recon, records = reconstruct_ibp(
        frames,
        shifts,
        max_iter=2,
        beta=0.5,
        psf_sigma=0.0,
        return_records=True,
    )

    assert recon.shape == hr.shape
    assert isinstance(records, list)
    assert records[0]["iteration"] == 1
