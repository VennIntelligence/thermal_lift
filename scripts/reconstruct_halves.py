#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "algos/ep15_info_limit/scripts"))
from run_m2_frc import stratified_split

sys.path.insert(0, str(PROJECT_ROOT / "algos/ep10_tgv_sr/src"))
from ep10_tgv_sr import reconstruct_map_tgv

sys.path.insert(0, str(PROJECT_ROOT / "algos/ep10_map_tv_sweep/src"))
from ep10_map_tv_sweep.map_tv_sweep import reconstruct_map_tv

sys.path.insert(0, str(PROJECT_ROOT / "algos/ep07_unet_sr/src"))
from unet_sr.real_eval import infer_solver_from_burst_full_halo, _load_real_eval_cache, _solver_conditioning_from_burst

from common.data_loader import highpass_preprocess

def load_data():
    raw_frames, shifts = _load_real_eval_cache(248, "contour_refined")
    metadata = pd.read_csv(PROJECT_ROOT / "output/ep01_data_processing/frame_audit.csv")
    metadata = metadata[metadata["is_sr_usable"] & metadata["is_main_session"]].iloc[:248]
    return raw_frames, shifts, metadata

def get_splits(metadata):
    indices = np.arange(len(metadata))
    phase_bins = np.zeros(len(metadata), dtype=int)  # dummy, stratified_split just uses it to group
    # Actually run_m2_frc uses `command_phase_bins(metadata, scale=2)`
    from run_m2_frc import command_phase_bins
    phase_bins = command_phase_bins(metadata, scale=2, theta_deg=47.6, pixel_size_um=20.0)
    a_idx, b_idx, _ = stratified_split(phase_bins, scale=2, seed=42)
    return a_idx, b_idx

def run_v11(frames, shifts, device="cpu"):
    from unet_sr.config import TrainingConfig
    from types import SimpleNamespace
    ckpt_path = PROJECT_ROOT / "algos/ep07_unet_sr/outputs/solver_v11_k2_p384_nogn_halo96_50k/solver_step_040000.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = dict(ckpt.get("config") or {})
    cfg.setdefault("scale", 2)
    cfg.setdefault("in_channels", 9)
    cfg.setdefault("base_channels", 64)
    cfg.setdefault("unroll_steps", 2)
    cfg.setdefault("solver_share_weights", True)
    cfg.setdefault("solver_band_sigma", 5.0)
    cfg.setdefault("solver_huber_delta", 0.0)
    cfg.setdefault("solver_eta_init", 0.5)
    cfg.setdefault("solver_learn_eta", False)
    cfg.setdefault("solver_prox_use_se", False)
    cfg.setdefault("solver_prox_norm", "none")
    cfg.setdefault("solver_prox_highpass_residual", False)
    cfg.setdefault("solver_prox_highpass_sigma_hr", 5.0)

    from unet_sr.unroll import UnrolledSolver
    cond_channels = 5 if cfg.get("solver_no_drizzle", False) else cfg["in_channels"]
    solver = UnrolledSolver(
        n_steps=cfg["unroll_steps"],
        cond_channels=cond_channels,
        base_channels=cfg["base_channels"],
        scale=cfg["scale"],
        share_weights=cfg["solver_share_weights"],
        band_highpass_sigma_lr_px=cfg["solver_band_sigma"],
        huber_delta=cfg["solver_huber_delta"],
        eta_init=cfg["solver_eta_init"],
        learn_eta=cfg["solver_learn_eta"],
        prox_use_se=cfg["solver_prox_use_se"],
        prox_norm=cfg["solver_prox_norm"],
        prox_highpass_residual=cfg["solver_prox_highpass_residual"],
        prox_highpass_sigma_hr=cfg["solver_prox_highpass_sigma_hr"],
    )
    solver.load_state_dict(ckpt["model_state_dict"])
    solver.eval().to(device)

    eval_cfg = SimpleNamespace(**cfg)
    return infer_solver_from_burst_full_halo(
        solver, frames, shifts, training_config=eval_cfg, halo_hr=96, device=torch.device(device)
    )

def run_tgv(frames, shifts):
    hp_frames = highpass_preprocess(frames, sigma_bg=5.0, workers=4)
    hr, _ = reconstruct_map_tgv(
        hp_frames, shifts, lambda_tv=0.003, alpha_ratio=2.0, psf_sigma=0.50,
        max_iter=100, step_size=1.0, use_fista=True, workers=4, tgv_inner_iter=80, tgv_device="cpu",
        aniso_ratio_y=1.5, coverage_weighted=True
    )
    return hr

def run_map_tv(frames, shifts):
    hp_frames = highpass_preprocess(frames, sigma_bg=5.0, workers=4)
    hr, _ = reconstruct_map_tv(
        hp_frames, shifts, lambda_tv=1e-3, psf_sigma=0.2,
        max_iter=150, step_size=0.1, use_fista=True, workers=4
    )
    return hr

def main():
    print("Loading data...")
    raw_frames, shifts, metadata = load_data()
    a_idx, b_idx = get_splits(metadata)

    out_dir = PROJECT_ROOT / "output/stage0c_frc_recons"
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, func in [("v11", lambda f, s: run_v11(f, s, "cpu")),
                       ("tgv", run_tgv),
                       ("maptv", run_map_tv)]:
        print(f"Running {name}...")
        try:
            a_img = func(raw_frames[a_idx], shifts[a_idx])
            np.save(out_dir / f"{name}_a.npy", a_img)
            b_img = func(raw_frames[b_idx], shifts[b_idx])
            np.save(out_dir / f"{name}_b.npy", b_img)
        except Exception as e:
            print(f"Error for {name}: {e}")

if __name__ == "__main__":
    main()
