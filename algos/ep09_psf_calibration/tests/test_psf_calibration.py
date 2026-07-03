from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import ndimage


def ndimage_gaussian(image: np.ndarray, sigma: float) -> np.ndarray:
    return ndimage.gaussian_filter(image, sigma=sigma, mode="reflect").astype(np.float32)


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


def test_stage0a_forward_centered_matches_saa_grid_on_ramp() -> None:
    """Centered blocks must average to the SAA scatter position scale*(i+shift)."""

    h, w = 32, 40
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    hr = (2.0 * yy + 3.0 * xx).astype(np.float32)
    shift = np.array([0.15, -0.10], dtype=np.float32)  # (dx, dy)

    centered = forward_block_average_shifted(hr, shift, scale=2, block_convention="centered")
    legacy = forward_block_average_shifted(hr, shift, scale=2, block_convention="legacy_corner")

    i = np.arange(h // 2, dtype=np.float64)
    j = np.arange(w // 2, dtype=np.float64)
    expected = 2.0 * (2.0 * (i[:, None] + float(shift[1]))) + 3.0 * (2.0 * (j[None, :] + float(shift[0])))
    interior = (slice(2, -2), slice(2, -2))
    np.testing.assert_allclose(centered[interior], expected[interior], atol=1e-4)
    # Legacy corner convention is exactly +0.5 HR px off in both axes on a ramp.
    np.testing.assert_allclose((legacy - centered)[interior], 0.5 * (2.0 + 3.0), atol=1e-4)


def test_stage0a_centered_convention_removes_spurious_quarter_pixel_refinement() -> None:
    """Regression for ACL-046: scoring an SAA x̂ with the legacy corner forward
    makes every frame prefer a constant ~-0.25 LR px shift correction; the
    centered convention must prefer the true (zero) correction instead."""

    from saa.saa import reconstruct_saa

    rng = np.random.default_rng(3)
    hr_truth = ndimage_gaussian(rng.normal(size=(64, 64)).astype(np.float32), 2.0)
    shifts = np.column_stack(
        [rng.uniform(-0.9, 0.9, size=16), rng.uniform(-0.9, 0.9, size=16)]
    ).astype(np.float32)
    frames = np.stack(
        [forward_block_average_shifted(hr_truth, s, scale=2, block_convention="centered") for s in shifts]
    )
    xhat = reconstruct_saa(frames, shifts, scale=2, fill_missing=True).astype(np.float32)

    offsets = shift_refine_offsets(0.375, 0.125)

    def _best_delta(block_convention: str) -> np.ndarray:
        score = _score_one_frame(
            frame_idx=0,
            split="train",
            refine_mode="bounded",
            blurred_hr=xhat,
            frame=frames[0],
            initial_shift=shifts[0],
            file_name="synthetic.txt",
            candidate=PsfCandidate(0.0, 0.0, 0.0),
            offsets=offsets,
            scale=2,
            crop_margin=4,
            band_sigma_lr_px=0.0,
            huber_delta=0.0,
            block_convention=block_convention,
        )
        return np.array([score.delta_dx_px, score.delta_dy_px])

    delta_centered = _best_delta("centered")
    delta_legacy = _best_delta("legacy_corner")
    assert np.linalg.norm(delta_centered) <= 0.126, delta_centered
    assert np.linalg.norm(delta_legacy) >= 0.24, delta_legacy
    # The legacy bias points to negative deltas in both axes (compensating +0.5 HR px).
    assert delta_legacy[0] < 0 and delta_legacy[1] < 0


def test_stage0a_split_source_routes_frames_to_opposite_xhat() -> None:
    """Frames must be scored against the x̂ selected by frame_xhat_idx."""

    import pandas as pd

    from psf_calibration.stage0a_mvp import score_candidate

    rng = np.random.default_rng(11)
    hr_a = ndimage_gaussian(rng.normal(size=(32, 32)).astype(np.float32), 1.5)
    hr_b = ndimage_gaussian(rng.normal(size=(32, 32)).astype(np.float32), 1.5)
    shifts = np.zeros((4, 2), dtype=np.float32)
    # Frames 0/2 come from hr_a, frames 1/3 from hr_b.
    frames = np.stack(
        [
            forward_block_average_shifted(hr_a if k % 2 == 0 else hr_b, shifts[k], scale=2)
            for k in range(4)
        ]
    )
    metadata = pd.DataFrame({"file": [f"f{k}.txt" for k in range(4)]})
    summary_rows, frame_rows = score_candidate(
        PsfCandidate(0.0, 0.0, 0.0),
        refine_mode="none",
        hr_images=[hr_a, hr_b],
        frame_xhat_idx=np.array([0, 1, 0, 1]),
        frames_raw=frames,
        metadata=metadata,
        shifts=shifts,
        split_indices={"train": np.arange(4)},
        offsets=shift_refine_offsets(0.0, 0.1),
        scale=2,
        crop_margin=2,
        band_sigma_lr_px=0.0,
        huber_delta=0.0,
        workers=1,
    )
    assert all(row["band_mse"] < 1e-8 for row in frame_rows), frame_rows
    # Cross-routing the same frames must NOT score near zero.
    _, wrong_rows = score_candidate(
        PsfCandidate(0.0, 0.0, 0.0),
        refine_mode="none",
        hr_images=[hr_a, hr_b],
        frame_xhat_idx=np.array([1, 0, 1, 0]),
        frames_raw=frames,
        metadata=metadata,
        shifts=shifts,
        split_indices={"train": np.arange(4)},
        offsets=shift_refine_offsets(0.0, 0.1),
        scale=2,
        crop_margin=2,
        band_sigma_lr_px=0.0,
        huber_delta=0.0,
        workers=1,
    )
    assert all(row["band_mse"] > 1e-4 for row in wrong_rows)


def test_build_refined_alignment_merges_and_guards(tmp_path: Path) -> None:
    import pandas as pd

    from psf_calibration.refined_alignment import build_refined_alignment

    base = pd.DataFrame(
        {
            "file": ["a.txt", "b.txt", "c.txt"],
            "refined_align_dx_px": [1.00, -0.50, 0.25],
            "refined_align_dy_px": [-2.00, 0.75, 0.10],
            "success": [True, True, True],
        }
    )
    refinements = pd.DataFrame(
        {
            "file": ["a.txt", "b.txt", "c.txt"],
            "initial_dx_px": [1.00, -0.50, 0.25],
            "initial_dy_px": [-2.00, 0.75, 0.10],
            "delta_dx_px": [0.10, -0.20, 0.00],
            "delta_dy_px": [-0.05, 0.15, 0.30],
            "refined_dx_px": [1.10, -0.70, 0.25],
            "refined_dy_px": [-2.05, 0.90, 0.40],
        }
    )
    base_csv = tmp_path / "base.csv"
    ref_csv = tmp_path / "refinements.csv"
    base.to_csv(base_csv, index=False)
    refinements.to_csv(ref_csv, index=False)

    out = build_refined_alignment(
        refinements_csv=ref_csv,
        base_alignment_csv=base_csv,
        expected_frames=3,
    )
    np.testing.assert_allclose(out["refined_align_dx_px"], [1.10, -0.70, 0.25])
    np.testing.assert_allclose(out["refined_align_dy_px"], [-2.05, 0.90, 0.40])
    np.testing.assert_allclose(out["stage0a_delta_dx_px"], [0.10, -0.20, 0.00])
    assert "success" in out.columns

    # Refinements computed from a DIFFERENT alignment must be rejected.
    wrong = refinements.copy()
    wrong["initial_dx_px"] = wrong["initial_dx_px"] + 0.3
    wrong_csv = tmp_path / "wrong.csv"
    wrong.to_csv(wrong_csv, index=False)
    try:
        build_refined_alignment(
            refinements_csv=wrong_csv,
            base_alignment_csv=base_csv,
            expected_frames=3,
        )
    except ValueError as exc:
        assert "not computed from this base alignment" in str(exc)
    else:
        raise AssertionError("mismatched refinements were accepted")


def test_stage0a_bootstrap_ci_detects_real_and_null_improvement() -> None:
    import pandas as pd

    from psf_calibration.stage0a_mvp import _bootstrap_val_improvement_ci

    baseline = PsfCandidate(0.5, 0.5, 0.0)
    best = PsfCandidate(0.2, 0.2, 0.0)
    rng = np.random.default_rng(7)
    n = 40
    base_mse = rng.uniform(0.010, 0.014, size=n)

    def _rows(candidate: PsfCandidate, refine_mode: str, mse: np.ndarray) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "frame_index": np.arange(n),
                "split": ["val"] * n,
                "refine_mode": [refine_mode] * n,
                "sigma_x_lr_px": candidate.sigma_x_lr_px,
                "sigma_y_lr_px": candidate.sigma_y_lr_px,
                "angle_deg": candidate.angle_deg,
                "band_mse": mse,
            }
        )

    # Real paired improvement: every frame drops ~8%, plus small noise.
    improved = base_mse * rng.uniform(0.90, 0.94, size=n)
    frame_scores = pd.concat(
        [_rows(baseline, "none", base_mse), _rows(best, "bounded", improved)],
        ignore_index=True,
    )
    result = _bootstrap_val_improvement_ci(frame_scores, baseline, best, n_boot=500, seed=1)
    assert result["improvement_significant_at_95"]
    assert result["ci95_low_pct"] > 0
    assert 4.0 < result["point_pct"] < 12.0

    # Null case: pure noise around the baseline should not be significant.
    null = base_mse * rng.uniform(0.97, 1.03, size=n)
    frame_scores_null = pd.concat(
        [_rows(baseline, "none", base_mse), _rows(best, "bounded", null)],
        ignore_index=True,
    )
    result_null = _bootstrap_val_improvement_ci(frame_scores_null, baseline, best, n_boot=500, seed=1)
    assert not result_null["improvement_significant_at_95"]
