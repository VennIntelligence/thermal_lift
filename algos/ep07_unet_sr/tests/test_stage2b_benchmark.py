"""Tests for the Stage 2b synthetic benchmark harness (scripts/run_stage2b_synth_benchmark.py)."""

from __future__ import annotations

import importlib.util
import inspect
import json
import sys
from pathlib import Path

import numpy as np
import pytest

EP07_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = EP07_ROOT.parents[1]
SCRIPT = EP07_ROOT / "scripts" / "run_stage2b_synth_benchmark.py"


@pytest.fixture(scope="module")
def bench():
    spec = importlib.util.spec_from_file_location("run_stage2b_synth_benchmark", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["run_stage2b_synth_benchmark"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_bench48_config_is_verbatim_v6_except_allowed_keys():
    v6 = json.loads((REPO_ROOT / "configs" / "synthetic" / "pool_2x_v6_cpu.json").read_text())
    b48 = json.loads((REPO_ROOT / "configs" / "synthetic" / "pool_2x_v6_bench48.json").read_text())
    allowed = {"_comment", "dataset", "num_scenes", "seed", "output_dir", "n_frames_per_scene"}
    diff = {k for k in set(v6) | set(b48) if v6.get(k) != b48.get(k)}
    assert diff == allowed, f"bench48 config drifted from v6 beyond the allowed keys: {sorted(diff)}"
    assert b48["num_scenes"] == 48
    assert b48["n_frames_per_scene"] == 96
    assert b48["seed"] != v6["seed"], "bench pool must be seed-disjoint from the training pool"
    assert b48["output_dir"] == "data/synthetic/pool_2x_v6_bench48"


def test_select_prefix_frames(bench):
    burst = np.arange(5 * 4 * 6, dtype=np.float32).reshape(5, 4, 6)
    shifts = np.arange(10, dtype=np.float32).reshape(5, 2)
    frames, sh = bench.select_prefix_frames(burst, shifts, 3)
    assert frames.shape == (3, 4, 6) and sh.shape == (3, 2)
    np.testing.assert_array_equal(frames, burst[:3])
    np.testing.assert_array_equal(sh, shifts[:3])
    with pytest.raises(ValueError):
        bench.select_prefix_frames(burst, shifts, 6)
    with pytest.raises(ValueError):
        bench.select_prefix_frames(burst, shifts, 0)
    with pytest.raises(ValueError):
        bench.select_prefix_frames(burst, shifts[:4], 3)


def test_classify_gate(bench):
    assert bench.classify_gate(0.01, None) == "ok"
    assert bench.classify_gate(0.5, 0.05) == "corrected"
    assert bench.classify_gate(0.5, 0.4) == "abort"  # correction failed to land
    assert bench.classify_gate(0.5, None) == "abort"  # no correction measured
    assert bench.classify_gate(2.0, 0.01) == "abort"  # beyond structural-offset ceiling


def test_tertile_bins(bench):
    labels = bench.tertile_bins([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert labels == ["lo", "lo", "mid", "mid", "hi", "hi"]


def test_gate_offset_roundtrip(bench):
    from scipy.ndimage import gaussian_filter

    _, _, probe = bench._ep15_scripts()
    rng = np.random.default_rng(7)
    image = gaussian_filter(rng.standard_normal((160, 160)).astype(np.float32), 3.0)
    # ACL-049-class constant grid offset, then the gate's iterative measurement must recover it
    shifted = probe.fourier_shift(image, dx_px=0.6, dy_px=0.35)
    dx, dy, _ = bench.measure_offset_iterative(image, shifted, probe)
    norm = float(np.hypot(dx, dy))
    assert dx == pytest.approx(0.6, abs=0.06)
    assert dy == pytest.approx(0.35, abs=0.06)
    corrected = probe.fourier_shift(shifted, dx_px=-dx, dy_px=-dy)
    _, _, residual = bench.measure_offset_iterative(image, corrected, probe)
    assert residual < 0.05
    assert bench.classify_gate(norm, residual) == "corrected"
    # aligned pair -> 'ok', no correction
    dx0, dy0, _ = bench.measure_offset_iterative(image, image, probe)
    assert bench.classify_gate(float(np.hypot(dx0, dy0)), None) == "ok"


def test_band_metrics_sanity(bench):
    from scipy.ndimage import gaussian_filter

    rng = np.random.default_rng(11)
    gt = gaussian_filter(rng.standard_normal((160, 160)).astype(np.float32), 1.5)
    assert bench.band_rmse(gt, gt, scale=2, crop_lr_px=16) == pytest.approx(0.0, abs=1e-7)
    assert bench.psnr(gt, gt, scale=2, crop_lr_px=16) == float("inf")
    noisy = gt + 0.05 * rng.standard_normal(gt.shape).astype(np.float32)
    assert bench.band_rmse(noisy, gt, scale=2, crop_lr_px=16) > 0.0
    assert np.isfinite(bench.psnr(noisy, gt, scale=2, crop_lr_px=16))
    v2, _, _ = bench._ep15_scripts()
    curve = v2.frc_curve_v2(gt, gt, scale=2, pixel_size_um=20.0, crop_lr_px=16, tukey_alpha=0.25)
    band = curve[(curve["period_um"] >= 25.0) & (curve["period_um"] <= 40.0)]
    assert float(np.nanmean(band["frc"].to_numpy(dtype=float))) == pytest.approx(1.0, abs=1e-6)


def test_infer_full_halo_has_default_off_psf_override():
    sys.path.insert(0, str(EP07_ROOT / "src"))
    from unet_sr.real_eval import infer_solver_from_burst_full_halo

    sig = inspect.signature(infer_solver_from_burst_full_halo)
    param = sig.parameters.get("psf_override")
    assert param is not None, "psf_override parameter missing from infer_solver_from_burst_full_halo"
    assert param.default is None, "psf_override must default to None (byte-identical real-eval path)"


def test_perturbation_suffix(bench):
    # ACL-060 (A3 operator-error axis): arm-name suffix encodes active DC perturbations;
    # '' when both knobs are off (default byte-identical naming).
    assert bench.perturbation_suffix(None, 0.0) == ""
    assert bench.perturbation_suffix(0.25, 0.0) == "__dcsig0p25"
    assert bench.perturbation_suffix(None, 0.1) == "__jit0p1"
    assert bench.perturbation_suffix(0.4, 0.05) == "__dcsig0p4__jit0p05"


def test_jitter_shifts_deterministic_and_prefix_paired(bench):
    shifts = np.linspace(-0.4, 0.4, 96 * 2, dtype=np.float32).reshape(96, 2)
    a = bench.jitter_shifts(shifts, 0.1, 20260708, "scene_0007")
    b = bench.jitter_shifts(shifts, 0.1, 20260708, "scene_0007")
    # same (seed, scene) -> identical perturbation; this is exactly how the neural and
    # classical phases (separate processes) stay comparable
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, bench.jitter_shifts(shifts, 0.1, 20260708, "scene_0008"))
    assert not np.array_equal(a, bench.jitter_shifts(shifts, 0.1, 1, "scene_0007"))
    # jitter is zero-mean gaussian at the requested std
    noise = a - shifts
    assert abs(float(noise.std()) - 0.1) < 0.02
    assert abs(float(noise.mean())) < 0.02
    # frame i's jitter must not depend on the requested N: both phases jitter the FULL
    # array before prefix selection, so the prefix property holds by construction
    np.testing.assert_array_equal(a[:24], bench.jitter_shifts(shifts, 0.1, 20260708, "scene_0007")[:24])
    # default-off contract: std<=0 is an untouched passthrough (byte-identical path)
    assert bench.jitter_shifts(shifts, 0.0, 20260708, "scene_0007") is shifts


def test_resolve_dc_sigma(bench):
    assert bench.resolve_dc_sigma("tgv", "portable", 0.33, None) == 0.5
    assert bench.resolve_dc_sigma("maptv", "portable", 0.33, None) == 0.2
    assert bench.resolve_dc_sigma("tgv", "oracle", 0.33, None) == pytest.approx(0.33)
    # override wins over both conditions and both methods
    for method, cond in (("tgv", "portable"), ("tgv", "oracle"), ("maptv", "oracle")):
        assert bench.resolve_dc_sigma(method, cond, 0.33, 0.25) == 0.25


def test_resolve_neural_psf_params(bench):
    md = {
        "psf_sigma_lr_px": 0.31,
        "psf_shape": "elliptical_gaussian",
        "psf_sigma_y_lr_px": 0.5,
        "psf_angle_deg": 12.0,
    }
    base = bench.resolve_neural_psf_params(md, None)
    assert base["sigma_lr_px"] == pytest.approx(0.31)
    assert base["shape"] == "elliptical_gaussian"
    assert base["sigma_y_lr_px"] == pytest.approx(0.5)
    assert base["angle_deg"] == pytest.approx(12.0)
    # override -> fixed isotropic gaussian (the "wrong operator" semantics of A3)
    over = bench.resolve_neural_psf_params(md, 0.4)
    assert over == {"sigma_lr_px": 0.4, "shape": "gaussian", "sigma_y_lr_px": None, "angle_deg": 0.0}
    # metadata defaults reproduce the pre-ACL-060 ScenePSF construction exactly
    minimal = bench.resolve_neural_psf_params({"psf_sigma_lr_px": 0.2}, None)
    assert minimal == {"sigma_lr_px": pytest.approx(0.2), "shape": "gaussian", "sigma_y_lr_px": None, "angle_deg": 0.0}


def test_dc_knob_cli_defaults_off(bench):
    args = bench.parse_args(["--phase", "metrics", "--pool-dir", "x", "--output-dir", "y"])
    assert args.dc_sigma_override is None
    assert args.dc_shift_jitter_std_px == 0.0
    assert bench.perturbation_suffix(args.dc_sigma_override, args.dc_shift_jitter_std_px) == ""


def test_lowfreq_stability_columns(bench):
    # ACL-055 (verification-B lesson): catch v14-style low-frequency amplitude drift that
    # band-limited metrics are blind to. All three columns are relative to the scene's own GT.
    rng = np.random.default_rng(7)
    gt = rng.uniform(19.0, 23.0, size=(160, 160)).astype(np.float32)

    ident = bench.lowfreq_stability(gt, gt, scale=2, crop_lr_px=16)
    assert ident["fullband_rmse"] == pytest.approx(0.0, abs=1e-7)
    assert ident["mean_offset"] == pytest.approx(0.0, abs=1e-7)
    assert ident["range_excursion"] == pytest.approx(1.0, abs=1e-6)

    shifted = bench.lowfreq_stability(gt + 0.5, gt, scale=2, crop_lr_px=16)
    assert shifted["fullband_rmse"] == pytest.approx(0.5, abs=1e-5)
    assert shifted["mean_offset"] == pytest.approx(0.5, abs=1e-5)
    assert shifted["range_excursion"] == pytest.approx(1.0, abs=1e-6)

    # v14 signature: amplified dynamic range around the same mean → range_excursion ≈ gain,
    # mean_offset ≈ 0, fullband_rmse > 0 (band_rmse alone would under-report this).
    crop = bench._ep15_scripts()[0].crop_for_frc(gt, scale=2, crop_lr_px=16)
    mu = float(crop.mean())
    blown = (gt - mu) * 7.0 + mu
    drift = bench.lowfreq_stability(blown, gt, scale=2, crop_lr_px=16)
    assert drift["range_excursion"] == pytest.approx(7.0, rel=1e-3)
    assert abs(drift["mean_offset"]) < 0.05
    assert drift["fullband_rmse"] > 1.0
