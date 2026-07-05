from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

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

from psf_calibration.esf_selfcal import (  # noqa: E402
    EsfSelfCalConfig,
    aperture_projection,
    combined_true_sigma_lr,
    discrete_gaussian_effective_sigma,
    edge_sigma_hr_to_effective_lr,
    esf_model,
    fit_edge_profile,
    resolve_aperture,
    reverdict_bench_rows,
    run_esf_selfcal,
    scene_edge_sigma_lr,
)
from psf_calibration.sigma_selfcal import safe_evaluate_prereg  # noqa: E402
from tcforge.forward import generate_lr_burst  # noqa: E402


def _raster_edge_hr(hr_shape: tuple[int, int], *, angle_deg: float, base: float, amp: float, ss: int = 8) -> np.ndarray:
    """Ideal straight step edge, box-antialiased per HR pixel (mimics SSAA rasterization).

    Independent of the fitted model: dense uniform box sampling of a sharp step,
    NOT the Gauss-Legendre/erf machinery used by the estimator.
    """

    h, w = hr_shape
    theta = np.deg2rad(angle_deg)
    nx, ny = np.cos(theta), np.sin(theta)
    rho = nx * (w / 2.0) + ny * (h / 2.0)
    sub = (np.arange(ss) + 0.5) / ss
    img = np.zeros(hr_shape, dtype=np.float64)
    yy = np.arange(h, dtype=np.float64)[:, None, None] + sub[None, :, None]  # (h, ss, 1)
    for x0 in range(w):
        xs = x0 + sub  # (ss,)
        d = nx * xs[None, None, :] + ny * yy - rho  # (h, ss, ss)
        img[:, x0] = base + amp * (d >= 0).mean(axis=(1, 2))
    return img


def _edge_burst(
    rng: np.random.Generator,
    *,
    sigma_true: float,
    angle_deg: float = 25.0,
    lr_shape: tuple[int, int] = (48, 64),
    n_frames: int = 16,
    scale: int = 2,
    noise: float = 0.05,
    amp: float = 4.0,
) -> tuple[np.ndarray, np.ndarray]:
    hr_shape = (lr_shape[0] * scale, lr_shape[1] * scale)
    hr = _raster_edge_hr(hr_shape, angle_deg=angle_deg, base=20.0, amp=amp)
    shifts = rng.uniform(-0.6, 0.6, size=(n_frames, 2))
    burst = generate_lr_burst(
        hr,
        shifts.astype(np.float32),
        forward_mode="physical_block_average",
        psf_sigma_lr_px=float(sigma_true),
        scale=scale,
        workers=1,
    ).astype(np.float64)
    burst += noise * rng.normal(size=burst.shape)
    return burst, shifts


FAST_CFG = dict(
    aperture="pool_block_average",
    max_points_per_edge=4000,
    bootstrap_rounds=60,
    frame_bootstrap_rounds=15,
    seed=0,
)


def test_aperture_projection_presets() -> None:
    for nx, ny in [(1.0, 0.0), (0.6, 0.8)]:
        # integer phase (f=0): bilinear collapses to the grid nodes -> scale^2 offsets
        offs, w, extra = aperture_projection("pool_block_average", 2, nx, ny, 8, frac_dx=0.0, frac_dy=0.0)
        assert len(offs) == 4 and np.isclose(w.sum(), 1.0) and np.isclose(extra, (1.0 / 12.0) / 4.0)
        # fractional phase: grid x bilinear corners -> up to scale^2 * 4 offsets
        offs_f, w_f, _ = aperture_projection("pool_block_average", 2, nx, ny, 8, frac_dx=0.5, frac_dy=0.25)
        assert len(offs_f) == 16 and np.isclose(w_f.sum(), 1.0)
        offs_b, w_b, extra_b = aperture_projection("detector_box", 2, nx, ny, 8)
        assert len(offs_b) == 64 and np.isclose(w_b.sum(), 1.0) and extra_b == 0.0
        assert np.isclose((offs_b * w_b).sum(), 0.0, atol=1e-12)  # symmetric
    offs_p, w_p, extra_p = aperture_projection("point", 2, 1.0, 0.0, 8)
    assert len(offs_p) == 1 and extra_p == 0.0
    assert resolve_aperture("auto", "physical_block_average") == "pool_block_average"
    assert resolve_aperture("auto", "exact_ep06_point") == "pool_point"
    assert resolve_aperture("auto", None) == "detector_box"
    assert resolve_aperture("detector_box", "physical_block_average") == "detector_box"


def test_discrete_gaussian_effective_sigma() -> None:
    # small nominal sigma: truncated discrete kernel carries much less variance
    assert abs(discrete_gaussian_effective_sigma(0.20, 2) - 0.142) <= 0.003
    assert discrete_gaussian_effective_sigma(0.15, 2) < 0.06
    # by sigma >= 0.35 the gap is ~1% or less
    for s in (0.35, 0.55, 1.0):
        eff = discrete_gaussian_effective_sigma(s, 2)
        assert abs(eff - s) / s <= 0.02
        assert eff <= s  # discrete/truncated never exceeds nominal
    assert discrete_gaussian_effective_sigma(0.0, 2) == 0.0


def test_fitter_math_recovers_sigma_point_aperture() -> None:
    rng = np.random.default_rng(1)
    cfg = EsfSelfCalConfig(aperture="point", seed=0)
    offs, w, extra = aperture_projection("point", 2, 1.0, 0.0, 8)
    d = rng.uniform(-5, 5, size=6000)
    v = esf_model(d, 3.0, 0.12, 0.42, 19.5, offs, w, extra) + 0.05 * rng.normal(size=d.shape)
    fit = fit_edge_profile(d, v, cfg, (offs, w), extra)
    assert fit["valid"]
    assert abs(fit["sigma_hat"] - 0.42) <= 0.02
    assert abs(fit["center_lr_px"] - 0.12) <= 0.02


def test_recover_sigma_oblique_edge_pool_render(tmp_path: Path) -> None:
    rng = np.random.default_rng(7)
    burst, shifts = _edge_burst(rng, sigma_true=0.40)
    cfg = EsfSelfCalConfig(**FAST_CFG)
    summary = run_esf_selfcal(burst, shifts, cfg, out_dir=tmp_path, label="edge")
    assert summary["status"] == "ok"
    assert abs(summary["sigma_hat"] - 0.40) / 0.40 <= 0.12
    assert summary["ci_lo"] <= summary["sigma_hat"] <= summary["ci_hi"]
    assert (tmp_path / "edge_esf_summary.json").exists()
    assert (tmp_path / "edge_esf_edges.csv").exists()
    assert (tmp_path / "edge_esf_profile.png").exists()


def test_box_and_render_extras_not_absorbed_into_sigma(tmp_path: Path) -> None:
    # sigma_true ~ 0: observed transition width is dominated by the KNOWN aperture
    # (discrete scale grid + raster box + bilinear tent, sigma_extra = 0.25 LR px).
    # A model that absorbed those into sigma would report >= ~0.26; the explicit
    # aperture model must keep sigma_hat near zero.
    rng = np.random.default_rng(11)
    burst, shifts = _edge_burst(rng, sigma_true=0.05)
    cfg = EsfSelfCalConfig(**{**FAST_CFG, "sigma_bounds": (0.005, 3.0)})
    summary = run_esf_selfcal(burst, shifts, cfg, out_dir=None, label="thin", make_plot=False)
    assert summary["status"] == "ok"
    assert summary["sigma_hat"] <= 0.15


def test_no_edge_rejection() -> None:
    rng = np.random.default_rng(3)
    burst = 20.0 + 0.05 * rng.normal(size=(12, 48, 64))
    shifts = rng.uniform(-0.6, 0.6, size=(12, 2))
    cfg = EsfSelfCalConfig(**FAST_CFG)
    summary = run_esf_selfcal(burst, shifts, cfg, out_dir=None, label="flat", make_plot=False)
    assert summary["status"] == "no_usable_edges"
    assert np.isnan(summary["sigma_hat"])


def test_scene_edge_sigma_truth_composition() -> None:
    # no config / non-isothermal pools -> no scene edge term, never guessed
    assert scene_edge_sigma_lr(None, 2) == (0.0, "none")
    value, source = scene_edge_sigma_lr({"temperature_model": "standard"}, 2)
    assert value == 0.0 and "isothermal" in source
    # bench48-style isothermal config: edge_sigma=0.6 HR px, same discrete shave as the PSF term
    iso = {"temperature_model": "isothermal", "temperature_isothermal": {"edge_sigma": 0.6}}
    value, source = scene_edge_sigma_lr(iso, 2)
    assert np.isclose(value, edge_sigma_hr_to_effective_lr(0.6, 2))
    assert np.isclose(value, discrete_gaussian_effective_sigma(0.3, 2))
    assert 0.2 < value < 0.3 and "edge_sigma" in source
    # isothermal WITHOUT the key -> generator default 1.4 HR px (mirrors generate_training_pool.py)
    value_default, _ = scene_edge_sigma_lr({"temperature_model": "isothermal"}, 2)
    assert np.isclose(value_default, edge_sigma_hr_to_effective_lr(1.4, 2))
    # quadrature composition
    assert np.isclose(combined_true_sigma_lr(0.3, 0.4), 0.5)
    assert combined_true_sigma_lr(0.3, 0.0) == 0.3


def _write_old_schema_rows_csv(path: Path, psf_only: list[float], edge_lr: float, jitter: list[float]) -> None:
    """bench_rows.csv in the pre-ACL-058 schema: `sigma_true` IS the psf-only truth."""

    import csv

    rows = []
    for i, (p, j) in enumerate(zip(psf_only, jitter)):
        hat = combined_true_sigma_lr(p, edge_lr) * (1.0 + j)  # estimator sees the TOTAL width
        rel = (hat - p) / p
        rows.append(
            {
                "scene_id": f"scene_{i:04d}",
                "kernel": "esf",
                "status": "ok",
                "sigma_true_nominal": p,
                "sigma_true": p,
                "sigma_hat_esf": hat,
                "sigma_hat_e1": hat,
                "ci_lo": hat - 0.02,
                "ci_hi": hat + 0.02,
                "rel_err_signed": rel,
                "abs_rel_err": abs(rel),
                "n_edges_valid": 4,
                "rel_spread": 0.05,
                "warnings": "",
                "psf_shape": "gaussian" if i % 2 == 0 else "elliptical_gaussian",
                "noise_sigma_c": 0.02 + 0.01 * i,
                "delta_T_c": 3.0,
                "aperture": "pool_block_average",
            }
        )
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def test_reverdict_corrects_old_schema_rows(tmp_path: Path) -> None:
    edge_lr = edge_sigma_hr_to_effective_lr(0.6, 2)
    psf_only = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55]
    jitter = [0.01, -0.01, 0.02, -0.02, 0.01, -0.01, 0.02, -0.02]
    rows_csv = tmp_path / "bench_rows.csv"
    _write_old_schema_rows_csv(rows_csv, psf_only, edge_lr, jitter)

    out = tmp_path / "reverdict"
    result = reverdict_bench_rows(rows_csv, out, scene_edge_sigma_lr_eff=edge_lr)
    verdict = result["verdict"]
    # uncorrected rows were biased high (total vs psf-only truth); corrected verdict must pass
    assert verdict["prereg_pass"] is True
    assert verdict["median_abs_rel_err"] <= 0.03
    assert not verdict["systematic_bias"]
    assert verdict["scene_edge_sigma_lr"] == edge_lr
    row0 = result["rows"][0]
    assert np.isclose(row0["sigma_true_psf_only"], 0.2)
    assert np.isclose(row0["sigma_true_total"], combined_true_sigma_lr(0.2, edge_lr))
    assert np.isclose(row0["sigma_true"], row0["sigma_true_total"])
    assert (out / "bench_rows.csv").exists()
    assert (out / "bench_verdict.json").exists()
    assert (out / "bench_summary.png").exists()
    # with a zero edge term the reverdict must reproduce the original (biased) reading
    result_zero = reverdict_bench_rows(rows_csv, tmp_path / "zero", scene_edge_sigma_lr_eff=0.0)
    assert result_zero["verdict"]["median_abs_rel_err"] > 0.25


def test_insufficient_scenes_verdict_skipped(tmp_path: Path) -> None:
    # direct guard
    verdict = safe_evaluate_prereg([], median_tol=0.15)
    assert verdict["verdict_skipped"] is True and verdict["prereg_pass"] is None
    # end-to-end via reverdict on a 3-row csv (the --scene-limit 3 crash scenario)
    edge_lr = edge_sigma_hr_to_effective_lr(0.6, 2)
    rows_csv = tmp_path / "rows3.csv"
    _write_old_schema_rows_csv(rows_csv, [0.2, 0.3, 0.4], edge_lr, [0.0, 0.0, 0.0])
    result = reverdict_bench_rows(rows_csv, tmp_path / "out3", scene_edge_sigma_lr_eff=edge_lr)
    assert result["verdict"]["verdict_skipped"] is True
    assert result["verdict"]["prereg_pass"] is None
    assert (tmp_path / "out3" / "bench_rows.csv").exists()


def test_cli_esf_generic_mode(tmp_path: Path, monkeypatch) -> None:
    rng = np.random.default_rng(5)
    burst, shifts = _edge_burst(rng, sigma_true=0.35, lr_shape=(40, 56), n_frames=12)
    burst_path = tmp_path / "burst.npy"
    np.save(burst_path, burst)
    shifts_path = tmp_path / "shifts.npy"  # CLI accepts (N,2) .npy; avoids a pandas dependency here
    np.save(shifts_path, shifts)

    script = ROOT / "algos" / "ep09_psf_calibration" / "scripts" / "sigma_selfcal.py"
    spec = importlib.util.spec_from_file_location("sigma_selfcal_cli_esf", script)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "sigma_selfcal.py",
            "--kernel", "esf",
            "--aperture", "pool_block_average",  # burst was pool-rendered above
            "--burst-npy", str(burst_path),
            "--shifts-csv", str(shifts_path),
            "--output-dir", str(tmp_path / "out"),
            "--esf-bootstrap", "40",
            "--label", "smoke",
        ],
    )
    assert mod.main() == 0
    assert (tmp_path / "out" / "smoke_esf_summary.json").exists()
