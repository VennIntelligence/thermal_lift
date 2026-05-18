from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage

SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ibp import forward
from map_tv import reconstruct_map_tv, tv_denoise_chambolle
from map_tv.map_tv import tv_norm
from saa import reconstruct_saa


def _scene(shape: tuple[int, int] = (48, 56)) -> np.ndarray:
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    img = 0.18 * np.cos(2 * np.pi * y / shape[0])
    img += 0.4 * np.exp(-((x - 17.0) ** 2 + (y - 18.0) ** 2) / 60.0)
    img += 0.5 * ((x > 28) & (y > 20))
    img -= img.min()
    img /= img.max()
    return img


def _shifts(n_frames: int) -> np.ndarray:
    phases = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]])
    return phases[np.arange(n_frames) % 4]


def _psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    mse = float(np.mean((reference - estimate) ** 2))
    return float(10.0 * np.log10(1.0 / max(mse, 1e-15)))


def test_tv_prox_reduces_total_variation() -> None:
    rng = np.random.default_rng(8)
    noisy = _scene((40, 44)) + rng.normal(scale=0.08, size=(40, 44))

    denoised = tv_denoise_chambolle(noisy, weight=0.06, max_iter=60)

    assert denoised.shape == noisy.shape
    assert np.isfinite(denoised).all()
    assert tv_norm(denoised) < tv_norm(noisy)


def test_map_tv_returns_records_and_improves_over_lr_baseline() -> None:
    hr = _scene()
    shifts = _shifts(28)
    clean_frames = np.stack([forward(hr, shift, psf_sigma=0.18) for shift in shifts])
    frames = clean_frames + np.random.default_rng(9).normal(scale=0.03, size=clean_frames.shape)

    initial = reconstruct_saa(frames, shifts)
    baseline = ndimage.zoom(np.mean(frames, axis=0), zoom=(2, 2), order=3)
    recon, records = reconstruct_map_tv(
        frames,
        shifts,
        initial=initial,
        lambda_tv=0.002,
        max_iter=8,
        step_size=0.9,
        psf_sigma=0.18,
        workers=2,
        tol=0.0,
        tv_inner_iter=25,
    )

    assert recon.shape == hr.shape
    assert np.isfinite(recon).all()
    assert isinstance(records, list)
    assert len(records) == 8
    assert {"iteration", "objective", "relative_update"}.issubset(records[0])
    assert _psnr(hr, recon) > _psnr(hr, baseline) + 1.0
    assert _psnr(hr, recon) >= _psnr(hr, initial) + 1.0


def test_map_tv_can_return_dataframe() -> None:
    hr = _scene((24, 28))
    shifts = _shifts(8)
    frames = np.stack([forward(hr, shift, psf_sigma=0.0) for shift in shifts])

    _, convergence = reconstruct_map_tv(
        frames,
        shifts,
        lambda_tv=0.0,
        max_iter=2,
        psf_sigma=0.0,
        tol=0.0,
        return_dataframe=True,
    )

    assert isinstance(convergence, pd.DataFrame)
    assert len(convergence) == 2
