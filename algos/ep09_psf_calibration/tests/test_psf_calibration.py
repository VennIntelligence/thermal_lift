from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
for path in [ROOT / "algos" / "ep09_psf_calibration" / "src", ROOT / "algos" / "ep06_sr_poc" / "src", ROOT / "core" / "src"]:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from psf_calibration.esf_fitting import erf_model, fit_esf_profile
from psf_calibration.utils import parabolic_minimum


def test_parabolic_minimum_recovers_known_minimum() -> None:
    sigmas = np.linspace(0.1, 0.6, 26)
    values = (sigmas - 0.32) ** 2 + 0.01
    sigma, ok = parabolic_minimum(sigmas, values)
    assert ok
    assert abs(sigma - 0.32) < 1e-6


def test_esf_fit_recovers_synthetic_sigma() -> None:
    x = np.linspace(-8, 8, 129)
    truth = np.array([2.5, -0.35, 0.85, 22.0])
    y = erf_model(x, *truth)
    params, _, r2, rmse = fit_esf_profile(x, y)
    assert r2 > 0.999999
    assert rmse < 1e-6
    assert abs(abs(params[2]) - truth[2]) < 1e-4
