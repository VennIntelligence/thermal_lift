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
from psf_calibration.stage0a_mvp import (
    PsfCandidate,
    _anisotropic_kernel,
    _read_stage_config,
    _score_one_frame,
    build_psf_candidates,
    forward_block_average_shifted,
    shift_refine_offsets,
)
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


def test_stage0a_psf_candidates_deduplicate_isotropic_angles() -> None:
    candidates = build_psf_candidates(sigmas=[0.5], anisotropy_ratios=[1.0, 2.0], angles=[0.0, 45.0])
    labels = {candidate.label for candidate in candidates}
    assert "sx0.500_sy0.500_a0.0" in labels
    assert "sx0.500_sy1.000_a0.0" in labels
    assert "sx0.500_sy1.000_a45.0" in labels
    assert len(candidates) == 3


def test_stage0a_anisotropic_kernel_is_normalized() -> None:
    kernel = _anisotropic_kernel(PsfCandidate(0.3, 0.6, 30.0), scale=2)
    assert kernel is not None
    assert kernel.ndim == 2
    assert abs(float(kernel.sum()) - 1.0) < 1e-6


def test_stage0a_rejects_legacy_10um_stage_config(tmp_path: Path) -> None:
    config = tmp_path / "stage_10um.json"
    config.write_text(
        '{"pixel_size_um": 10.0, "current_spatial_resolution_um": 20.0, "theta_deg": 47.6}',
        encoding="utf-8",
    )
    try:
        _read_stage_config(config)
    except ValueError as exc:
        assert "20 um detector pitch" in str(exc)
    else:
        raise AssertionError("legacy 10um stage config was accepted")


def test_stage0a_shift_refine_recovers_synthetic_offset() -> None:
    hr = np.zeros((32, 32), dtype=np.float32)
    hr[8:24, 13:18] = 1.0
    hr[15:18, 6:26] += 0.5
    truth_shift = np.array([0.20, -0.10], dtype=np.float32)
    initial_shift = np.array([0.10, -0.05], dtype=np.float32)
    frame = forward_block_average_shifted(hr, truth_shift, scale=2)

    score = _score_one_frame(
        frame_idx=0,
        split="train",
        refine_mode="bounded",
        blurred_hr=hr,
        frame=frame,
        initial_shift=initial_shift,
        file_name="synthetic.txt",
        candidate=PsfCandidate(0.0, 0.0, 0.0),
        offsets=shift_refine_offsets(0.10, 0.05),
        scale=2,
        crop_margin=0,
        band_sigma_lr_px=0.0,
        huber_delta=0.0,
    )

    np.testing.assert_allclose(
        [score.refined_dx_px, score.refined_dy_px],
        truth_shift,
        atol=1e-6,
    )
