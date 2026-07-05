from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[3]
for path in [
    ROOT / "algos" / "ep09_psf_calibration" / "src",
    ROOT / "algos" / "ep06_sr_poc" / "src",
    ROOT / "core" / "src",
    ROOT / "tcforge" / "src",
]:
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from psf_calibration.sigma_selfcal import (  # noqa: E402
    SelfCalConfig,
    autocorr_decay_length,
    evaluate_prereg,
    run_bench_validation,
    run_selfcal,
    spectral_flatness,
)
from tcforge._ep06_reference.forward import build_observation_operator  # noqa: E402


def _render_burst(
    rng: np.random.Generator,
    *,
    sigma_true: float,
    n_frames: int,
    lr_shape: tuple[int, int],
    scale: int = 2,
    noise: float = 0.02,
) -> tuple[np.ndarray, np.ndarray]:
    hr_shape = (lr_shape[0] * scale, lr_shape[1] * scale)
    scene = 20.0 + 2.0 * ndimage.gaussian_filter(rng.normal(size=hr_shape), 1.5)
    shifts = rng.uniform(-0.6, 0.6, size=(n_frames, 2))
    op = build_observation_operator(hr_shape, shifts=shifts, psf_sigma=sigma_true, scale=scale)
    burst = op.forward_all(scene) + noise * rng.normal(size=(n_frames, *lr_shape))
    return burst.astype(np.float64), shifts.astype(np.float64)


TINY_CFG = dict(rounds=3, holdout_frac=0.25, cg_iters=15, lam=1e-3, max_frames=None, seed=0, scale=2)


def test_e1_recovers_known_sigma(tmp_path: Path) -> None:
    # NOTE (ACL-056): at this tiny size the recovery works through border effects
    # (mode="constant" breaks the blur/shift commutation that makes E1 degenerate
    # at realistic sizes). This test guards plumbing + determinism, NOT metrological
    # validity of the estimator — see the module docstring limitation block.
    rng = np.random.default_rng(7)
    burst, shifts = _render_burst(rng, sigma_true=0.4, n_frames=12, lr_shape=(24, 32))
    cfg = SelfCalConfig(sigma_grid=(0.15, 0.25, 0.4, 0.6, 0.9), **TINY_CFG)
    summary = run_selfcal(burst, shifts, cfg, out_dir=tmp_path, label="tiny")
    assert abs(summary["sigma_hat_e1"] - 0.4) <= 0.12
    assert summary["ci_lo"] <= summary["sigma_hat_e1"] <= summary["ci_hi"]
    assert (tmp_path / "tiny_curves.csv").exists()
    assert (tmp_path / "tiny_summary.json").exists()
    assert (tmp_path / "tiny_curves.png").exists()


def test_e2_whiteness_separates_white_from_structured() -> None:
    rng = np.random.default_rng(3)
    white = rng.normal(size=(64, 64))
    structured = ndimage.gaussian_filter(white, 2.0)
    assert spectral_flatness(white) > spectral_flatness(structured) + 0.1
    assert autocorr_decay_length(white) < autocorr_decay_length(structured)


def _fake_rows(rel_errs: list[float], noises: list[float]) -> list[dict]:
    return [
        {"rel_err_signed": r, "noise_sigma_c": n, "psf_shape": "gaussian"}
        for r, n in zip(rel_errs, noises, strict=True)
    ]


def test_evaluate_prereg_verdict_logic() -> None:
    noises = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06] * 5
    rng = np.random.default_rng(0)
    balanced = list(rng.normal(0.0, 0.05, size=30))
    verdict = evaluate_prereg(_fake_rows(balanced, noises), n_boot=200)
    assert verdict["median_ok"] and not verdict["systematic_bias"] and verdict["prereg_pass"]

    too_big = list(rng.normal(0.0, 0.6, size=30))
    verdict = evaluate_prereg(_fake_rows(too_big, noises), n_boot=200)
    assert not verdict["median_ok"] and not verdict["prereg_pass"]

    biased = list(rng.normal(0.12, 0.01, size=30))  # within tol but same-sign in every tertile
    verdict = evaluate_prereg(_fake_rows(biased, noises), n_boot=200)
    assert verdict["median_ok"] and verdict["systematic_bias"] and not verdict["prereg_pass"]


def test_bench_mode_and_verdict_files(tmp_path: Path) -> None:
    pool = tmp_path / "pool"
    rng = np.random.default_rng(11)
    sigmas = [0.3, 0.5, 0.3, 0.5, 0.4, 0.4]
    noises = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06]
    for i, (sig, noi) in enumerate(zip(sigmas, noises, strict=True)):
        scene_dir = pool / f"scene_{i:04d}"
        scene_dir.mkdir(parents=True)
        burst, shifts = _render_burst(rng, sigma_true=sig, n_frames=8, lr_shape=(16, 24), noise=noi)
        np.save(scene_dir / "lr_burst.npy", burst.astype(np.float16))
        np.save(scene_dir / "shifts.npy", shifts.astype(np.float32))
        (scene_dir / "metadata.json").write_text(
            json.dumps({"psf_sigma_lr_px": sig, "psf_shape": "gaussian", "noise_sigma_c": noi, "delta_T_c": 2.0}),
            encoding="utf-8",
        )
    cfg = SelfCalConfig(sigma_grid=(0.2, 0.3, 0.45, 0.7), rounds=2, cg_iters=10, max_frames=None, crop_lr=14, seed=0)
    out = tmp_path / "out"
    result = run_bench_validation(pool, cfg, out, workers=1, scene_plots=False)
    assert len(result["rows"]) == 6
    assert all(np.isfinite(r["rel_err_signed"]) for r in result["rows"])
    assert result["verdict"]["median_abs_rel_err"] < 0.5
    assert (out / "bench_rows.csv").exists()
    assert (out / "bench_verdict.json").exists()
    assert (out / "bench_summary.png").exists()
    assert (out / "scenes" / "scene_0000_summary.json").exists()


def test_cli_generic_mode_smoke(tmp_path: Path, monkeypatch) -> None:
    rng = np.random.default_rng(5)
    burst, shifts = _render_burst(rng, sigma_true=0.4, n_frames=8, lr_shape=(16, 24))
    burst_path = tmp_path / "burst.npy"
    np.save(burst_path, burst)
    import pandas as pd

    shifts_path = tmp_path / "shifts.csv"
    pd.DataFrame({"dx_px": shifts[:, 0], "dy_px": shifts[:, 1]}).to_csv(shifts_path, index=False)

    script = ROOT / "algos" / "ep09_psf_calibration" / "scripts" / "sigma_selfcal.py"
    spec = importlib.util.spec_from_file_location("sigma_selfcal_cli", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sigma_selfcal.py",
            "--burst-npy", str(burst_path),
            "--shifts-csv", str(shifts_path),
            "--output-dir", str(tmp_path / "out"),
            "--sigma-grid", "0.2,0.3,0.45,0.7",
            "--rounds", "2",
            "--cg-iters", "10",
            "--max-frames", "-1",
            "--label", "smoke",
        ],
    )
    assert mod.main() == 0
    assert (tmp_path / "out" / "smoke_summary.json").exists()
