from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tcforge.physics as physics


def test_temperature_field_respects_mask_semantics_and_bounds() -> None:
    mask = np.zeros((24, 32), dtype=np.uint8)
    mask[6:18, 8:24] = 1
    temp = physics.render_temperature_field(mask, t_bg_c=21.0, delta_t_c=2.5, low_freq_amplitude_c=0.0, seed=7)

    assert temp.shape == mask.shape
    assert temp.dtype == np.float32
    assert np.isfinite(temp).all()
    assert np.allclose(temp[mask == 0], 21.0)
    assert np.allclose(temp[mask == 1], 23.5)


def test_temperature_field_accepts_soft_coverage_and_fractional_thin_lines() -> None:
    mask = np.zeros((8, 10), dtype=np.float32)
    mask[:, 4] = 0.25
    mask[:, 5] = 0.75

    temp = physics.render_temperature_field(mask, t_bg_c=21.0, delta_t_c=2.0, low_freq_amplitude_c=0.0)

    assert temp.dtype == np.float32
    assert np.allclose(temp[:, 4], 21.5)
    assert np.allclose(temp[:, 5], 22.5)
    assert float(temp[:, 4].max() - 21.0) < 2.0


def test_temperature_field_rejects_invalid_soft_coverage() -> None:
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[1, 1] = 1.2

    try:
        physics.render_temperature_field(mask, low_freq_amplitude_c=0.0)
    except ValueError as exc:
        assert "coverage" in str(exc)
    else:
        raise AssertionError("expected invalid coverage to raise ValueError")


def test_temperature_field_accepts_explicit_multilevel_offsets() -> None:
    labels = np.zeros((12, 16), dtype=np.uint8)
    labels[:, 4:8] = 1
    labels[:, 8:12] = 2
    temp = physics.render_temperature_field(
        labels,
        t_bg_c=20.0,
        temperature_offsets_c=[0.0, 0.75, 2.5],
        low_freq_amplitude_c=0.0,
    )

    assert np.allclose(temp[labels == 0], 20.0)
    assert np.allclose(temp[labels == 1], 20.75)
    assert np.allclose(temp[labels == 2], 22.5)


def test_edge_map_is_binary_like_and_localized_to_mask_boundary() -> None:
    mask = np.zeros((32, 32), dtype=np.uint8)
    mask[8:24, 8:24] = 1
    edge = physics.edge_map(mask)

    assert edge.shape == mask.shape
    assert np.isfinite(edge).all()
    assert edge.sum() > 0
    assert edge[16, 16] == 0


def test_gaussian_noise_is_seed_reproducible_and_has_expected_scale() -> None:
    frame = np.full((128, 128), 21.0, dtype=np.float32)
    noisy_a = physics.add_noise(frame, noise_sigma_c=0.0724, seed=11)
    noisy_b = physics.add_noise(frame, noise_sigma_c=0.0724, seed=11)

    assert noisy_a.dtype == np.float32
    assert np.array_equal(noisy_a, noisy_b)
    assert 0.05 < float(np.std(noisy_a - frame)) < 0.095


def test_mixed_noise_is_finite_reproducible_and_rms_anchored() -> None:
    frame = np.full((4, 128, 96), 21.0, dtype=np.float32)
    noisy_a = physics.add_noise(
        frame,
        noise_sigma_c=0.0724,
        seed=13,
        noise_model="mixed",
        fpn_sigma_px=5.0,
        stripe_sigma_c=0.015,
    )
    noisy_b = physics.add_noise(
        frame,
        noise_sigma_c=0.0724,
        seed=13,
        noise_model="mixed",
        fpn_sigma_px=5.0,
        stripe_sigma_c=0.015,
    )
    residual = noisy_a - frame

    assert noisy_a.dtype == np.float32
    assert np.isfinite(noisy_a).all()
    assert np.array_equal(noisy_a, noisy_b)
    assert abs(float(np.mean(residual))) < 1e-4
    assert abs(float(np.std(residual)) - 0.0724) < 0.003


def test_supported_noise_models_return_expected_shape() -> None:
    shape = (32, 48)
    for model in ("iid_gaussian", "fpn_lowfreq", "column_stripe", "spatial_correlated", "mixed", "detector_realistic"):
        residual = physics.make_noise(shape, noise_sigma_c=0.0724, seed=17, noise_model=model)
        assert residual.shape == shape
        assert residual.dtype == np.float32
        assert np.isfinite(residual).all()


def test_detector_defects_are_reproducible_fixed_spatial_pixels() -> None:
    burst = np.full((5, 20, 24), 21.0, dtype=np.float32)
    a = physics.add_detector_defects(burst, defect_rate=0.02, seed=21, hot_delta_c=1.0, cold_delta_c=-1.0)
    b = physics.add_detector_defects(burst, defect_rate=0.02, seed=21, hot_delta_c=1.0, cold_delta_c=-1.0)
    changed = np.any(a != burst, axis=0)

    assert np.array_equal(a, b)
    assert changed.sum() > 0
    assert np.all(a[:, changed] == a[0, changed][None, :])


def test_psf_kernel_sampling_and_blur_are_normalized_and_finite() -> None:
    params = physics.sample_psf_parameters(seed=5, elliptical_probability=1.0, airy_probability=0.0)
    kernel = physics.make_psf_kernel(scale=4, **params)  # type: ignore[arg-type]
    impulse = np.zeros((33, 33), dtype=np.float32)
    impulse[16, 16] = 1.0
    blurred = physics.apply_psf_blur(impulse, scale=4, **params)  # type: ignore[arg-type]

    assert kernel.dtype == np.float32
    assert kernel.ndim == 2
    assert np.isclose(float(kernel.sum()), 1.0, atol=1e-5)
    assert np.isfinite(blurred).all()
    assert np.isclose(float(blurred.sum()), 1.0, atol=1e-4)


def test_physics_parameter_sampling_covers_wide_range_specs() -> None:
    params = physics.sample_physics_parameters(
        seed=3,
        config={
            "delta_T_c_by_difficulty": {"hard": [0.5, 5.0]},
            "noise_sigma_c": {"dist": "lognormal", "mean": 0.0724, "sigma_factor": 0.2},
        },
        difficulty="hard",
    )

    assert 0.5 <= params["delta_T_c"] <= 5.0
    assert params["noise_sigma_c"] > 0.0
