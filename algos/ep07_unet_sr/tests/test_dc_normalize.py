"""ACL-055: DC sum→mean normalization option (decouple the DC step from the evidence budget M).

Legacy behavior (sum) must stay byte-identical by default; mean must equal the single-frame
gradient when the burst is M duplicates of one frame (sum = M·single, /M = single).
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
import torch

from unet_sr.config import TrainingConfig, config_from_args
from unet_sr.forward_torch import ScenePSF
from unet_sr.unroll import UnrolledSolver

SCALE = 2

SOLVER_ARGS = [
    "--training-pool-dir", "dummy_pool",
    "--input-mode", "hybrid_drizzle2x",
    "--scale", "2",
    "--unroll-steps", "2",
    "--boundary-boost", "3.0",
]


def _build(**kw) -> UnrolledSolver:
    torch.manual_seed(123)
    return UnrolledSolver(
        n_steps=2, cond_channels=5, base_channels=8, scale=SCALE,
        prox_use_se=False, prox_norm="none",
        band_highpass_sigma_lr_px=1.0,  # tiny test frames; default 5.0 over-pads 16x16
        **kw,
    ).eval()


def _inputs(m: int, seed: int = 3):
    rng = np.random.default_rng(seed)
    frame = torch.from_numpy(rng.normal(size=(1, 1, 16, 16)).astype(np.float32))
    shift = torch.from_numpy(rng.uniform(-0.9, 0.9, size=(1, 1, 2)).astype(np.float32))
    burst = frame.repeat(1, m, 1, 1)
    shifts = shift.repeat(1, m, 1)
    x0 = torch.zeros(1, 1, 32, 32)
    cond = torch.from_numpy(rng.normal(size=(1, 5, 32, 32)).astype(np.float32))
    psf = ScenePSF(
        sigma_lr_px=torch.tensor([0.4]), shape=["gaussian"], sigma_y_lr_px=[None],
        angle_deg=torch.tensor([0.0]),
    )
    return x0, burst, shifts, psf, cond


def test_default_is_sum_and_byte_identical() -> None:
    default = _build()
    explicit = _build(dc_normalize="sum")
    assert default.dc_normalize == "sum"
    x0, burst, shifts, psf, cond = _inputs(m=6)
    with torch.no_grad():
        out_default = default(x0, burst, shifts, psf, cond)
        out_explicit = explicit(x0, burst, shifts, psf, cond)
    torch.testing.assert_close(out_default, out_explicit, atol=0, rtol=0)


def test_mean_on_duplicated_burst_equals_sum_on_single_frame() -> None:
    mean_solver = _build(dc_normalize="mean")
    sum_solver = _build(dc_normalize="sum")
    x0, burst8, shifts8, psf, cond = _inputs(m=8)
    with torch.no_grad():
        out_mean8 = mean_solver(x0, burst8, shifts8, psf, cond)
        out_sum1 = sum_solver(x0, burst8[:, :1], shifts8[:, :1], psf, cond)
    # Σ over 8 identical frames / 8 == single frame (float assoc → allclose, not equal)
    torch.testing.assert_close(out_mean8, out_sum1, atol=1e-5, rtol=0)


def test_mean_is_m_invariant_under_duplication() -> None:
    mean_solver = _build(dc_normalize="mean")
    x0, burst4, shifts4, psf, cond = _inputs(m=4)
    burst8 = torch.cat([burst4, burst4], dim=1)
    shifts8 = torch.cat([shifts4, shifts4], dim=1)
    with torch.no_grad():
        out4 = mean_solver(x0, burst4, shifts4, psf, cond)
        out8 = mean_solver(x0, burst8, shifts8, psf, cond)
    torch.testing.assert_close(out4, out8, atol=1e-5, rtol=0)


def test_rejects_bad_value() -> None:
    with pytest.raises(ValueError, match="dc_normalize"):
        _build(dc_normalize="avg")


def test_config_cli_parse_and_default() -> None:
    cfg = config_from_args(SOLVER_ARGS)
    assert cfg.solver_dc_normalize == "sum"
    cfg_mean = config_from_args([*SOLVER_ARGS, "--solver-dc-normalize", "mean"])
    assert cfg_mean.solver_dc_normalize == "mean"


def test_config_validation_rejects_bad_value() -> None:
    cfg = dataclasses.replace(config_from_args(SOLVER_ARGS), solver_dc_normalize="avg")
    with pytest.raises(ValueError, match="solver_dc_normalize"):
        cfg.validate()  # validation lives in validate(), called by config_from_args


def test_old_checkpoint_config_rebuild_defaults_to_sum() -> None:
    # Stage 2b harness path: rebuild TrainingConfig from a pre-ACL-055 checkpoint's config dict
    # (vars(config) without the new key) via dataclass field filtering → must default to sum.
    cfg = config_from_args(SOLVER_ARGS)
    cfg_dict = {k: v for k, v in vars(cfg).items() if k != "solver_dc_normalize"}
    field_names = {f.name for f in dataclasses.fields(TrainingConfig)}
    rebuilt = TrainingConfig(**{k: v for k, v in cfg_dict.items() if k in field_names})
    assert rebuilt.solver_dc_normalize == "sum"


def test_build_solver_tolerates_config_object_missing_the_field() -> None:
    # Unpickled pre-ACL-055 TrainingConfig OBJECTS lack the attribute entirely; build_solver
    # must fall back to sum (getattr default) instead of raising.
    from unet_sr.solver_train import build_solver

    cfg = config_from_args([*SOLVER_ARGS, "--base-channels", "8"])
    delattr(cfg, "solver_dc_normalize")
    solver = build_solver(cfg, torch.device("cpu"), cond_channels=5)
    assert solver.dc_normalize == "sum"
