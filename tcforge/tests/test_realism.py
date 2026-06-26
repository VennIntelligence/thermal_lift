"""Tests for the realism augmentations (defects / isothermal temp / field noise)."""
import numpy as np

from tcforge import realism


def test_apply_defects_only_removes_material():
    rng = np.random.default_rng(0)
    cov = np.zeros((220, 260), np.float32)
    cov[30:190, 30:230] = 1.0                      # one solid block
    defected, meta = realism.apply_defects(
        cov, rng, severity_range=(1.0, 1.0), max_holes=4, max_notches=4, max_cracks=3)
    assert defected.shape == cov.shape
    assert 0.0 <= float(defected.min()) and float(defected.max()) <= 1.0
    assert float(defected.sum()) < float(cov.sum())          # defects subtract material
    assert set(meta) >= {"holes", "notches", "cracks", "severity"}
    assert meta["holes"] + meta["notches"] + meta["cracks"] >= 1


def test_isothermal_is_uniform_within_a_connected_structure():
    rng = np.random.default_rng(1)
    mask = np.zeros((128, 128), np.float32)
    mask[30:90, 30:90] = 1.0                       # single connected block
    f = realism.render_isothermal_field(
        mask, rng, t_bg_c=21.0, delta_t_c=2.0, level_min=0.95, edge_sigma=0.0, low_freq_amplitude_c=0.0)
    assert float(f.min()) >= 21.0 - 1e-4           # never below background
    core = f[45:75, 45:75]                          # interior of the block
    assert float(core.std()) < 0.05                # ~isothermal within one structure


def test_field_noise_fpn_fixed_across_burst_grain_per_frame():
    rng = np.random.default_rng(2)
    burst = np.full((8, 64, 80), 20.0, np.float32)
    noisy = realism.field_noise_burst(burst, rng, vignette_c=0.2, stripe_c=0.05, grain_c=0.10)
    assert noisy.shape == burst.shape
    resid = noisy - burst
    fixed_est = resid.mean(axis=0)                  # vignette + stripe (fixed across frames)
    per_frame_std = resid.std(axis=0).mean()        # ~grain (fixed pattern cancels in std-over-frames)
    assert float(fixed_est.std()) > 0.05            # a real fixed pattern is present
    assert 0.04 < float(per_frame_std) < 0.20       # grain is per-frame (~grain_c)


def test_field_noise_accepts_single_frame():
    rng = np.random.default_rng(3)
    out = realism.field_noise_burst(np.full((48, 60), 19.0, np.float32), rng)
    assert out.shape == (48, 60)
