#!/usr/bin/env python3
"""Stage 0h scratch renderer for refined-alignment split-half recons.

This is intentionally a tmp/ script: it should not become repo code.  It avoids
the ep07 real-eval LRU cache so the alignment CSV used for Stage 0h is explicit
and printed in the stdout log.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (
    PROJECT_ROOT / "core" / "src",
    PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src",
    PROJECT_ROOT / "algos" / "ep07_unet_sr" / "src",
    PROJECT_ROOT / "algos" / "ep10_tgv_sr" / "src",
    PROJECT_ROOT / "algos" / "ep10_map_tv_sweep" / "src",
    PROJECT_ROOT / "algos" / "ep15_info_limit" / "scripts",
):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from common.alignment import load_alignment_shifts  # noqa: E402
from common.data_loader import highpass_preprocess, load_main_session_frames  # noqa: E402
from ep10_map_tv_sweep.map_tv_sweep import reconstruct_map_tv  # noqa: E402
from ep10_tgv_sr import reconstruct_map_tgv  # noqa: E402
from run_m2_frc import command_phase_bins, stratified_split  # noqa: E402
from thermal_core.alignment_paths import default_contour_alignment_csv  # noqa: E402
from unet_sr.real_eval import infer_solver_from_burst_full_halo  # noqa: E402
from unet_sr.unroll import UnrolledSolver  # noqa: E402


CHECKPOINTS = {
    "v11": PROJECT_ROOT
    / "algos/ep07_unet_sr/outputs/solver_v11_k2_p384_nogn_halo96_50k/solver_step_040000.pt",
    "C_nodr": PROJECT_ROOT / "algos/ep07_unet_sr/outputs/solver_v13_v6_nodr_ctrl/solver_final.pt",
    "D_dr01": PROJECT_ROOT / "algos/ep07_unet_sr/outputs/solver_v13_v6_dr01/solver_final.pt",
}


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        try:
            return str(value.resolve().relative_to(PROJECT_ROOT))
        except ValueError:
            return str(value)
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def load_stage0h_inputs(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, Any]]:
    alignment_csv = default_contour_alignment_csv(project_root_path=PROJECT_ROOT).resolve()
    print(f"[stage0h] default alignment CSV: {alignment_csv}", flush=True)
    if alignment_csv.name != "stage0f_refined_alignment.csv":
        raise AssertionError(f"expected stage0f_refined_alignment.csv, got {alignment_csv}")

    raw_frames, metadata = load_main_session_frames(
        args.data_dir,
        args.frame_audit_csv,
        workers=int(args.io_workers),
        dtype=np.float32,
        limit=None,
    )
    shifts = load_alignment_shifts("contour_refined", metadata=metadata, alignment_csv=alignment_csv).astype(
        np.float32,
        copy=False,
    )
    if raw_frames.shape != (248, 480, 640):
        raise ValueError(f"expected (248, 480, 640) clean stack, got {raw_frames.shape}")
    if shifts.shape != (248, 2):
        raise ValueError(f"expected (248, 2) shifts, got {shifts.shape}")

    phase_bins = command_phase_bins(metadata, scale=2, theta_deg=47.6, pixel_size_um=20.0)
    a_idx, b_idx, balance = stratified_split(phase_bins, scale=2, seed=int(args.split_seed))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(
        {
            "frame_index": np.arange(len(metadata), dtype=int),
            "file": metadata["file"].astype(str).to_numpy(),
            "acquisition_order": metadata["acquisition_order"].to_numpy(),
            "phase_bin_2x": np.asarray(phase_bins, dtype=int),
            "split": np.where(np.isin(np.arange(len(metadata)), a_idx), "a", "b"),
            "shift_dx_lr_px": shifts[:, 0],
            "shift_dy_lr_px": shifts[:, 1],
        }
    ).to_csv(args.output_dir / "stage0h_split_indices.csv", index=False)
    balance.to_csv(args.output_dir / "stage0h_split_balance.csv", index=False)
    meta = {
        "alignment_csv": alignment_csv,
        "n_frames": int(raw_frames.shape[0]),
        "split_seed": int(args.split_seed),
        "n_a": int(len(a_idx)),
        "n_b": int(len(b_idx)),
        "shift_dx_mean": float(np.mean(shifts[:, 0])),
        "shift_dy_mean": float(np.mean(shifts[:, 1])),
        "shift_norm_mean": float(np.mean(np.linalg.norm(shifts, axis=1))),
    }
    return raw_frames, shifts, metadata, {"a_idx": a_idx, "b_idx": b_idx, "phase_bins": phase_bins, "meta": meta}


def solver_from_checkpoint(ckpt_path: Path, device: torch.device) -> tuple[UnrolledSolver, SimpleNamespace, dict[str, Any]]:
    print(f"[stage0h] loading checkpoint: {ckpt_path}", flush=True)
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
    cfg.setdefault("solver_no_drizzle", False)
    cfg.setdefault("phase_bin_channels", 4)
    cfg.setdefault("solver_warmstart", "phasebin")
    cfg.setdefault("solver_m_frames", 16)
    cfg.setdefault("solver_dc_rim_lr_px", 0)
    cfg.setdefault("highpass_sigma", 5.0)

    cond_channels = 5 if bool(cfg.get("solver_no_drizzle", False)) else int(cfg["in_channels"])
    solver = UnrolledSolver(
        n_steps=int(cfg["unroll_steps"]),
        cond_channels=cond_channels,
        base_channels=int(cfg["base_channels"]),
        scale=int(cfg["scale"]),
        share_weights=bool(cfg["solver_share_weights"]),
        band_highpass_sigma_lr_px=float(cfg["solver_band_sigma"]),
        huber_delta=float(cfg["solver_huber_delta"]),
        eta_init=float(cfg["solver_eta_init"]),
        learn_eta=bool(cfg["solver_learn_eta"]),
        prox_use_se=bool(cfg["solver_prox_use_se"]),
        prox_norm=str(cfg["solver_prox_norm"]),
        prox_highpass_residual=bool(cfg["solver_prox_highpass_residual"]),
        prox_highpass_sigma_hr=float(cfg["solver_prox_highpass_sigma_hr"]),
    )
    solver.load_state_dict(ckpt["model_state_dict"])
    solver.eval().to(device)
    return solver, SimpleNamespace(**cfg), cfg


def render_neural(
    name: str,
    raw_frames: np.ndarray,
    shifts: np.ndarray,
    split: dict[str, Any],
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> None:
    requested = torch.device(args.device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        print("[stage0h] CUDA requested but unavailable; falling back to CPU", flush=True)
        requested = torch.device("cpu")
    solver, cfg_ns, cfg = solver_from_checkpoint(CHECKPOINTS[name], requested)
    rows = []
    for tag, indices in (("a", split["a_idx"]), ("b", split["b_idx"])):
        started = time.perf_counter()
        print(f"[stage0h] rendering {name}_{tag}: n={len(indices)} device={requested}", flush=True)
        img = infer_solver_from_burst_full_halo(
            solver,
            raw_frames[indices],
            shifts[indices],
            training_config=cfg_ns,
            halo_hr=int(args.neural_halo_hr),
            device=requested,
        )
        out_path = args.output_dir / f"{name}_{tag}.npy"
        np.save(out_path, img.astype(np.float32, copy=False))
        elapsed = time.perf_counter() - started
        print(f"[stage0h] saved {out_path} shape={img.shape} elapsed_sec={elapsed:.2f}", flush=True)
        rows.append({"split": tag, "path": out_path, "elapsed_sec": elapsed})
        if requested.type == "cuda":
            torch.cuda.empty_cache()
    manifest["methods"][name] = {
        "kind": "neural",
        "checkpoint": CHECKPOINTS[name],
        "device": str(requested),
        "halo_hr": int(args.neural_halo_hr),
        "config_subset": {
            "scale": cfg.get("scale"),
            "solver_no_drizzle": cfg.get("solver_no_drizzle"),
            "phase_bin_channels": cfg.get("phase_bin_channels"),
            "solver_warmstart": cfg.get("solver_warmstart"),
            "solver_m_frames": cfg.get("solver_m_frames"),
            "solver_prox_highpass_residual": cfg.get("solver_prox_highpass_residual"),
            "solver_dc_shift_jitter_std_px": cfg.get("solver_dc_shift_jitter_std_px"),
        },
        "outputs": rows,
    }


def render_tgv(
    raw_frames: np.ndarray,
    shifts: np.ndarray,
    split: dict[str, Any],
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> None:
    hp_frames = highpass_preprocess(raw_frames, sigma_bg=5.0, workers=int(args.io_workers))
    kwargs = {
        "lambda_tv": 0.003,
        "alpha_ratio": 2.0,
        "psf_sigma": 0.50,
        "max_iter": int(args.tgv_max_iter),
        "step_size": 1.0,
        "use_fista": True,
        "workers": int(args.classic_workers),
        "tgv_inner_iter": int(args.tgv_inner_iter),
        "tgv_device": "cpu",
        "aniso_ratio_y": 1.5,
        "coverage_weighted": True,
    }
    rows = []
    for tag, indices in (("a", split["a_idx"]), ("b", split["b_idx"])):
        started = time.perf_counter()
        print(f"[stage0h] rendering tgv_{tag}: n={len(indices)} kwargs={kwargs}", flush=True)
        img, records = reconstruct_map_tgv(hp_frames[indices], shifts[indices], **kwargs)
        out_path = args.output_dir / f"tgv_{tag}.npy"
        conv_path = args.output_dir / f"tgv_{tag}_convergence.csv"
        np.save(out_path, img.astype(np.float32, copy=False))
        pd.DataFrame(records).to_csv(conv_path, index=False)
        elapsed = time.perf_counter() - started
        print(f"[stage0h] saved {out_path} shape={img.shape} elapsed_sec={elapsed:.2f}", flush=True)
        rows.append({"split": tag, "path": out_path, "convergence_csv": conv_path, "elapsed_sec": elapsed})
    manifest["methods"]["tgv"] = {"kind": "classic", "params": kwargs, "outputs": rows}


def render_maptv(
    raw_frames: np.ndarray,
    shifts: np.ndarray,
    split: dict[str, Any],
    args: argparse.Namespace,
    manifest: dict[str, Any],
) -> None:
    hp_frames = highpass_preprocess(raw_frames, sigma_bg=5.0, workers=int(args.io_workers))
    kwargs = {
        "lambda_tv": 1e-3,
        "psf_sigma": 0.2,
        "max_iter": int(args.maptv_max_iter),
        "step_size": 0.1,
        "use_fista": True,
        "workers": int(args.classic_workers),
    }
    rows = []
    for tag, indices in (("a", split["a_idx"]), ("b", split["b_idx"])):
        started = time.perf_counter()
        print(f"[stage0h] rendering maptv_{tag}: n={len(indices)} kwargs={kwargs}", flush=True)
        img, records = reconstruct_map_tv(hp_frames[indices], shifts[indices], **kwargs)
        out_path = args.output_dir / f"maptv_{tag}.npy"
        conv_path = args.output_dir / f"maptv_{tag}_convergence.csv"
        np.save(out_path, img.astype(np.float32, copy=False))
        pd.DataFrame(records).to_csv(conv_path, index=False)
        elapsed = time.perf_counter() - started
        print(f"[stage0h] saved {out_path} shape={img.shape} elapsed_sec={elapsed:.2f}", flush=True)
        rows.append({"split": tag, "path": out_path, "convergence_csv": conv_path, "elapsed_sec": elapsed})
    manifest["methods"]["maptv"] = {"kind": "classic", "params": kwargs, "outputs": rows}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output/stage0h_frc_recons")
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data/data_raw/infrared_avi")
    parser.add_argument("--frame-audit-csv", type=Path, default=PROJECT_ROOT / "output/ep01_data_processing/frame_audit.csv")
    parser.add_argument("--methods", default="v11,C_nodr,D_dr01,tgv,maptv")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--io-workers", type=int, default=8)
    parser.add_argument("--classic-workers", type=int, default=4)
    parser.add_argument("--neural-halo-hr", type=int, default=96)
    parser.add_argument("--tgv-max-iter", type=int, default=100)
    parser.add_argument("--tgv-inner-iter", type=int, default=80)
    parser.add_argument("--maptv-max-iter", type=int, default=150)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else PROJECT_ROOT / args.output_dir
    started = time.perf_counter()
    torch.set_num_threads(max(1, int(args.classic_workers)))
    thread_env = {
        name: os.environ.get(name, "")
        for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    }
    print(f"[stage0h] pid={os.getpid()} output_dir={args.output_dir}", flush=True)
    print(
        f"[stage0h] workers: io={args.io_workers} classic={args.classic_workers}; "
        f"device_request={args.device}; torch_threads={torch.get_num_threads()}; thread_env={thread_env}",
        flush=True,
    )
    raw_frames, shifts, _metadata, split = load_stage0h_inputs(args)
    methods = [item.strip() for item in str(args.methods).split(",") if item.strip()]
    manifest_path = args.output_dir / "render_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("task", "stage0h refined-alignment split-half renderer")
        manifest.setdefault("command_methods", [])
        manifest.setdefault("methods", {})
        manifest["command_methods"] = list(dict.fromkeys([*manifest["command_methods"], *methods]))
        manifest["inputs"] = split["meta"]
    else:
        manifest = {
            "task": "stage0h refined-alignment split-half renderer",
            "command_methods": methods,
            "inputs": split["meta"],
            "methods": {},
        }
    for method in methods:
        if method in CHECKPOINTS:
            render_neural(method, raw_frames, shifts, split, args, manifest)
        elif method == "tgv":
            render_tgv(raw_frames, shifts, split, args, manifest)
        elif method == "maptv":
            render_maptv(raw_frames, shifts, split, args, manifest)
        else:
            raise ValueError(f"unknown method: {method}")
        write_json(manifest_path, manifest)
    manifest["elapsed_sec"] = float(time.perf_counter() - started)
    write_json(manifest_path, manifest)
    print(f"[stage0h] done elapsed_sec={manifest['elapsed_sec']:.2f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
