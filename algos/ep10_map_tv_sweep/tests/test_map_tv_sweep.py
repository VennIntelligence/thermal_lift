from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from common.forward_model import forward
from ep10_map_tv_sweep import (
    ParamSpec,
    cache_hr_name,
    evaluate_param,
    holdout_details_for_spec,
    init_worker,
    pareto_frontier,
    save_results_table,
    token,
)
from ep10_map_tv_sweep.map_tv_sweep import tv_denoise_chambolle, tv_norm


def _scene(shape: tuple[int, int] = (24, 28)) -> np.ndarray:
    y, x = np.mgrid[0 : shape[0], 0 : shape[1]]
    img = 0.2 * np.sin(2 * np.pi * x / shape[1])
    img += 0.3 * np.exp(-((x - 9.0) ** 2 + (y - 10.0) ** 2) / 40.0)
    img += 0.4 * ((x > 14) & (y > 11))
    img -= img.min()
    img /= max(float(img.max()), 1e-12)
    return img.astype(np.float32)


def _shifts(n_frames: int) -> np.ndarray:
    phases = np.array([[0.0, 0.0], [0.5, 0.0], [0.0, 0.5], [0.5, 0.5]], dtype=np.float32)
    return phases[np.arange(n_frames) % 4]


def _frames(n_frames: int = 6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truth = _scene()
    shifts = _shifts(n_frames)
    frames = np.stack([forward(truth, shift, psf_sigma=0.0) for shift in shifts]).astype(np.float32)
    return truth, frames, shifts


def test_token_and_cache_name_are_stable() -> None:
    spec = ParamSpec(lambda_tv=0.001, psf_sigma=0.5)

    assert token(0.001) == "0p001"
    assert cache_hr_name(spec) == "full_hr_lambda0p001_sigma0p5.npy"


def test_tv_prox_reduces_total_variation() -> None:
    rng = np.random.default_rng(12)
    noisy = _scene() + rng.normal(scale=0.05, size=(24, 28))

    denoised = tv_denoise_chambolle(noisy, weight=0.04, max_iter=30)

    assert denoised.shape == noisy.shape
    assert np.isfinite(denoised).all()
    assert tv_norm(denoised) < tv_norm(noisy)


def test_holdout_details_have_per_frame_rows() -> None:
    _truth, frames, shifts = _frames(6)
    spec = ParamSpec(lambda_tv=0.0001, psf_sigma=0.0)
    config = {
        "max_iter": 1,
        "step_size": 0.7,
        "use_fista": False,
        "map_tv_workers": 1,
        "holdout_mod": 3,
    }

    details = holdout_details_for_spec(frames, shifts, spec, config)

    assert list(details["frame_index"]) == [0, 3]
    assert {"mse", "rmse", "shift_dx_lr_px", "shift_dy_lr_px"}.issubset(details.columns)
    assert np.isfinite(details["mse"]).all()


def test_evaluate_param_writes_summary_and_detail_files(tmp_path: Path) -> None:
    truth, frames, shifts = _frames(6)
    paths = {
        "hp_frames": str(tmp_path / "frames.npy"),
        "shifts": str(tmp_path / "shifts.npy"),
        "ref_hp_hr": str(tmp_path / "ref.npy"),
    }
    np.save(paths["hp_frames"], frames)
    np.save(paths["shifts"], shifts)
    np.save(paths["ref_hp_hr"], truth)
    config = {
        "max_iter": 1,
        "step_size": 0.7,
        "use_fista": False,
        "map_tv_workers": 1,
        "split_half_splits": 1,
        "random_state": 0,
        "holdout_mod": 3,
        "cache_dir": str(tmp_path / "cache"),
        "detail_dir": str(tmp_path / "details"),
    }
    spec = ParamSpec(lambda_tv=0.0001, psf_sigma=0.0)
    init_worker(paths, config)

    row = evaluate_param(spec)

    assert row["lambda_tv"] == spec.lambda_tv
    assert np.isfinite(row["split_half_nrmse"])
    assert Path(row["hr_cache_file"]).exists()
    convergence_path = Path(row["convergence_file"])
    split_path = Path(row["split_half_detail_file"])
    holdout_path = Path(row["holdout_detail_file"])
    if not split_path.is_absolute():
        convergence_path = Path(__file__).resolve().parents[3] / convergence_path
        split_path = Path(__file__).resolve().parents[3] / split_path
        holdout_path = Path(__file__).resolve().parents[3] / holdout_path
    assert convergence_path.exists()
    assert split_path.exists()
    assert holdout_path.exists()


def test_pareto_frontier_and_result_columns(tmp_path: Path) -> None:
    rows = [
        {"lambda_tv": 0.001, "psf_sigma": 0.5, "split_half_nrmse": 0.2, "artifact_score": 1.4},
        {"lambda_tv": 0.01, "psf_sigma": 0.5, "split_half_nrmse": 0.1, "artifact_score": 1.0},
    ]

    table = save_results_table(tmp_path / "sweep_results.csv", rows)
    frontier = pareto_frontier(table)

    assert isinstance(table, pd.DataFrame)
    assert "split_half_corr" in table.columns
    assert len(frontier) == 1
    assert float(frontier.iloc[0]["lambda_tv"]) == 0.01
