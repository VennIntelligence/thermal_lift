#!/usr/bin/env python3
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "algos/ep15_info_limit/scripts"))
from run_m2_frc import stratified_split, command_phase_bins

sys.path.insert(0, str(PROJECT_ROOT / "algos/ep07_unet_sr/src"))
from unet_sr.real_eval import infer_solver_from_burst_full_halo, _load_real_eval_cache

def load_data():
    raw_frames, shifts = _load_real_eval_cache(248, "contour_refined")
    metadata = pd.read_csv(PROJECT_ROOT / "output/ep01_data_processing/frame_audit.csv")
    metadata = metadata[metadata["is_sr_usable"] & metadata["is_main_session"]].iloc[:248]
    return raw_frames, shifts, metadata

def get_splits(metadata):
    indices = np.arange(len(metadata))
    phase_bins = command_phase_bins(metadata, scale=2, theta_deg=47.6, pixel_size_um=20.0)
    a_idx, b_idx, _ = stratified_split(phase_bins, scale=2, seed=42)
    return a_idx, b_idx

def run_solver(frames, shifts, ckpt_path_str, device="cpu"):
    ckpt_path = PROJECT_ROOT / ckpt_path_str
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    cfg = dict(ckpt.get("config") or {})
    # Need to handle missing keys similarly to V11 script
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
    from types import SimpleNamespace
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

def main():
    print("Loading data...")
    raw_frames, shifts, metadata = load_data()
    a_idx, b_idx = get_splits(metadata)

    out_dir = PROJECT_ROOT / "output/stage0c_frc_recons"
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoints = [
        ("C_nodr", "algos/ep07_unet_sr/outputs/solver_v13_v6_nodr_ctrl/solver_final.pt"),
        ("D_dr01", "algos/ep07_unet_sr/outputs/solver_v13_v6_dr01/solver_final.pt")
    ]

    for name, ckpt_path in checkpoints:
        print(f"Running {name}...")
        try:
            a_img = run_solver(raw_frames[a_idx], shifts[a_idx], ckpt_path, "cpu")
            np.save(out_dir / f"{name}_a.npy", a_img)
            b_img = run_solver(raw_frames[b_idx], shifts[b_idx], ckpt_path, "cpu")
            np.save(out_dir / f"{name}_b.npy", b_img)
        except Exception as e:
            print(f"Error for {name}: {e}")

if __name__ == "__main__":
    main()
