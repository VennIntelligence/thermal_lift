#!/usr/bin/env python3
"""Compute actual TGV split-half and FRC metrics for the unified paper harness.

This is a CPU-only evidence-hardening runner for Task E1. It first reconstructs
the full 248-frame anisotropic coverage-weighted TGV highpass image through the
same child-process path used for the two half reconstructions, then compares it
to the submitted EP10 TGV anchor. Only if that self-check passes does the JSON
status become ``success``; otherwise downstream tables should retain the caveat.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLBACKEND", "Agg")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
EP06_SRC = PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src"
EP15_SCRIPTS = PROJECT_ROOT / "algos" / "ep15_info_limit" / "scripts"
CORE_SRC = PROJECT_ROOT / "core" / "src"
for _path in (EP06_SRC, EP15_SCRIPTS, CORE_SRC):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from common.alignment import load_alignment_shifts  # noqa: E402
from common.data_loader import highpass_preprocess, load_main_session_frames  # noqa: E402
from run_m2_frc import command_phase_bins, find_cutoff, frc_curve, stratified_split  # noqa: E402

EXPECTED_CLEAN_SR_FRAMES = 248
LR_SHAPE = (480, 640)
SCALE = 2
HIGHPASS_SIGMA = 5.0
FRC_SEED = 42
FRC_PERIODS = (20.0, 16.0, 14.0, 12.0, 10.0)


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fill_nan(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    fill = float(np.nanmedian(arr[finite])) if bool(finite.any()) else 0.0
    return np.where(finite, arr, fill).astype(np.float32, copy=False)


def nrmse_pair(a: np.ndarray, b: np.ndarray) -> float:
    lhs = np.asarray(a, dtype=np.float32)
    rhs = np.asarray(b, dtype=np.float32)
    valid = np.isfinite(lhs) & np.isfinite(rhs)
    if int(valid.sum()) < 2:
        return float("nan")
    denom = float(np.nanstd(lhs[valid]) + np.nanstd(rhs[valid]))
    return float(np.sqrt(np.nanmean((lhs[valid] - rhs[valid]) ** 2)) / max(denom, 1e-12))


def interpolate_frc(curve: pd.DataFrame, period_um: float) -> float:
    freq = curve["frequency_um_inv"].to_numpy(dtype=float)
    values = curve["frc"].to_numpy(dtype=float)
    valid = (freq > 0) & np.isfinite(freq) & np.isfinite(values)
    if not bool(valid.any()):
        return float("nan")
    order = np.argsort(freq[valid])
    return float(np.interp(1.0 / period_um, freq[valid][order], values[valid][order], left=np.nan, right=np.nan))


def load_stage_config(path: Path) -> dict[str, float]:
    data = read_json(path)
    return {"theta_deg": float(data["theta_deg"]), "pixel_size_um": float(data["pixel_size_um"])}


def prepare_inputs(args: argparse.Namespace) -> dict[str, Any]:
    cache_dir = args.output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    raw_path = cache_dir / "raw_frames.npy"
    hp_path = cache_dir / "hp_frames.npy"
    shifts_path = cache_dir / "contour_shifts.npy"
    bins_path = cache_dir / "phase_bins_2x.npy"
    metadata_path = cache_dir / "metadata.csv"

    if all(path.exists() for path in (raw_path, hp_path, shifts_path, bins_path, metadata_path)) and not args.force_inputs:
        raw_frames = np.load(raw_path, mmap_mode="r")
        hp_frames = np.load(hp_path, mmap_mode="r")
        shifts = np.load(shifts_path)
        phase_bins = np.load(bins_path)
        metadata = pd.read_csv(metadata_path)
    else:
        raw_frames, metadata = load_main_session_frames(
            args.data_dir,
            args.frame_audit_csv,
            workers=args.workers,
            dtype=np.float32,
        )
        if raw_frames.shape != (EXPECTED_CLEAN_SR_FRAMES, *LR_SHAPE):
            raise ValueError(f"Expected clean frame stack {(EXPECTED_CLEAN_SR_FRAMES, *LR_SHAPE)}, got {raw_frames.shape}")
        hp_frames = highpass_preprocess(raw_frames, sigma_bg=HIGHPASS_SIGMA, workers=args.workers)
        shifts = load_alignment_shifts("contour_refined", metadata=metadata).astype(np.float32, copy=False)
        stage = load_stage_config(args.stage_config)
        phase_bins = command_phase_bins(
            metadata,
            scale=SCALE,
            theta_deg=stage["theta_deg"],
            pixel_size_um=stage["pixel_size_um"],
        )
        np.save(raw_path, np.asarray(raw_frames, dtype=np.float32))
        np.save(hp_path, np.asarray(hp_frames, dtype=np.float32))
        np.save(shifts_path, np.asarray(shifts, dtype=np.float32))
        np.save(bins_path, np.asarray(phase_bins, dtype=np.int16))
        metadata.to_csv(metadata_path, index=False)

    return {
        "raw_frames": raw_frames,
        "hp_frames": hp_frames,
        "shifts": np.asarray(shifts, dtype=np.float32),
        "phase_bins": np.asarray(phase_bins, dtype=np.int16),
        "metadata": metadata,
        "raw_frames_npy": raw_path,
        "hp_frames_npy": hp_path,
    }


def write_child_spec(
    *,
    args: argparse.Namespace,
    run_id: str,
    indices: np.ndarray,
    hp_frames_npy: Path,
    shifts: np.ndarray,
) -> tuple[Path, Path, Path]:
    shifts_path = args.output_dir / "tgv_specs" / f"{run_id}_shifts.npy"
    spec_path = args.output_dir / "tgv_specs" / f"{run_id}.json"
    result_path = args.output_dir / "tgv_results" / f"{run_id}.json"
    hr_path = args.output_dir / "hr" / f"{run_id}_highpass.npy"
    convergence_path = args.output_dir / "convergence" / f"{run_id}_convergence.csv"
    shifts_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(shifts_path, np.asarray(shifts, dtype=np.float32))
    payload = {
        "run_id": run_id,
        "indices": np.asarray(indices, dtype=int).tolist(),
        "hp_frames_npy": str(hp_frames_npy.resolve()),
        "shifts_npy": str(shifts_path.resolve()),
        "hr_npy": str(hr_path.resolve()),
        "convergence_csv": str(convergence_path.resolve()),
        "result_json": str(result_path.resolve()),
        "lambda_tv": float(args.lambda_tv),
        "psf_sigma": float(args.psf_sigma),
        "alpha_ratio": float(args.alpha_ratio),
        "max_iter": int(args.max_iter),
        "tgv_inner_iter": int(args.tgv_inner_iter),
        "aniso_ratio_y": float(args.aniso_ratio_y),
        "coverage_weighted": bool(args.coverage_weighted),
        "workers": int(args.tgv_workers),
        "cuda_visible_devices": "",
    }
    spec_path.parent.mkdir(parents=True, exist_ok=True)
    spec_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return spec_path, result_path, hr_path


def run_tgv_child(args: argparse.Namespace, run_id: str, indices: np.ndarray, inputs: dict[str, Any]) -> dict[str, Any]:
    selected_shifts = np.asarray(inputs["shifts"], dtype=np.float32)[np.asarray(indices, dtype=int)]
    spec_path, result_path, hr_path = write_child_spec(
        args=args,
        run_id=run_id,
        indices=np.asarray(indices, dtype=int),
        hp_frames_npy=Path(inputs["hp_frames_npy"]),
        shifts=selected_shifts,
    )
    if result_path.exists() and hr_path.exists() and not args.force_tgv:
        cached = read_json(result_path)
        if cached.get("status") == "success":
            cached["cache_hit"] = True
            return cached

    cmd = [
        str(args.conda_exe),
        "run",
        "-p",
        str(args.tgv_env),
        "python",
        str(args.tgv_child),
        "--spec",
        str(spec_path),
    ]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = str(args.tgv_workers)
    env.setdefault("OPENBLAS_NUM_THREADS", "1")
    env.setdefault("MKL_NUM_THREADS", "1")
    started = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=PROJECT_ROOT,
        env=env,
        text=True,
        capture_output=True,
        timeout=float(args.tgv_timeout_sec),
        check=False,
    )
    child = read_json(result_path) if result_path.exists() else {
        "run_id": run_id,
        "status": "failed",
        "error": "child result JSON missing",
    }
    child["orchestrator_runtime_sec"] = float(time.perf_counter() - started)
    child["stdout_tail"] = proc.stdout[-4000:]
    child["stderr_tail"] = proc.stderr[-4000:]
    child["returncode"] = int(proc.returncode)
    child["cache_hit"] = False
    write_json(result_path, child)
    return child


def load_child_hr(child: dict[str, Any]) -> np.ndarray:
    path = Path(str(child["hr_npy"]))
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return np.load(path).astype(np.float32, copy=False)


def self_check(full_hp: np.ndarray, args: argparse.Namespace) -> dict[str, Any]:
    anchor = np.load(args.tgv_anchor_highpass).astype(np.float32, copy=False)
    diff = np.asarray(full_hp, dtype=np.float64) - np.asarray(anchor, dtype=np.float64)
    denom = max(float(np.linalg.norm(anchor[np.isfinite(anchor)])), 1e-12)
    valid = np.isfinite(diff) & np.isfinite(anchor)
    rel_l2 = float(np.linalg.norm(diff[valid]) / denom)
    return {
        "anchor_highpass": args.tgv_anchor_highpass,
        "relative_l2": rel_l2,
        "threshold": float(args.self_check_threshold),
        "passed": bool(rel_l2 < float(args.self_check_threshold)),
        "note": "same child-process TGV path, full 248-frame highpass domain",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "data_raw" / "infrared_avi")
    parser.add_argument("--frame-audit-csv", type=Path, default=PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--stage-config", type=Path, default=PROJECT_ROOT / "configs" / "stage_calibration.json")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "ep11_unified_harness" / "tgv_split_frc")
    parser.add_argument("--summary-json", type=Path, default=PROJECT_ROOT / "output" / "ep11_unified_harness" / "tgv_split_frc.json")
    parser.add_argument("--tgv-anchor-highpass", type=Path, default=PROJECT_ROOT / "output" / "ep10_tgv_sr" / "best_hr_highpass.npy")
    parser.add_argument("--conda-exe", type=Path, default=Path("/home/ujs/miniforge3/bin/conda"))
    parser.add_argument("--tgv-env", type=Path, default=PROJECT_ROOT / "algos" / "ep10_tgv_sr" / ".venv")
    parser.add_argument("--tgv-child", type=Path, default=PROJECT_ROOT / "algos" / "ep16_budget_robustness" / "scripts" / "run_tgv_child.py")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--tgv-workers", type=int, default=4)
    parser.add_argument("--lambda-tv", type=float, default=0.003)
    parser.add_argument("--psf-sigma", type=float, default=0.5)
    parser.add_argument("--alpha-ratio", type=float, default=2.0)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--tgv-inner-iter", type=int, default=80)
    parser.add_argument("--aniso-ratio-y", type=float, default=1.5)
    parser.add_argument("--coverage-weighted", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--self-check-threshold", type=float, default=1e-3)
    parser.add_argument("--tgv-timeout-sec", type=float, default=7200.0)
    parser.add_argument("--force-inputs", action="store_true")
    parser.add_argument("--force-tgv", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    payload: dict[str, Any] = {
        "task": "Task E1 actual TGV split-half/FRC",
        "created_or_updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "status": "started",
        "seed": FRC_SEED,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "params": {
            "lambda_tv": float(args.lambda_tv),
            "psf_sigma": float(args.psf_sigma),
            "alpha_ratio": float(args.alpha_ratio),
            "max_iter": int(args.max_iter),
            "tgv_inner_iter": int(args.tgv_inner_iter),
            "aniso_ratio_y": float(args.aniso_ratio_y),
            "coverage_weighted": bool(args.coverage_weighted),
            "tgv_device": "cpu",
            "scale": SCALE,
            "highpass_sigma": HIGHPASS_SIGMA,
        },
    }
    write_json(args.summary_json, payload)

    try:
        inputs = prepare_inputs(args)
        phase_bins = np.asarray(inputs["phase_bins"], dtype=np.int16)
        a_idx, b_idx, balance = stratified_split(phase_bins, scale=SCALE, seed=FRC_SEED)
        all_idx = np.arange(EXPECTED_CLEAN_SR_FRAMES, dtype=int)

        children: dict[str, dict[str, Any]] = {}
        for run_id, indices in (
            ("tgv_full_selfcheck", all_idx),
            ("tgv_split_seed42_a", a_idx),
            ("tgv_split_seed42_b", b_idx),
        ):
            child = run_tgv_child(args, run_id, indices, inputs)
            children[run_id] = child
            payload["children"] = children
            write_json(args.summary_json, payload)
            if child.get("status") != "success":
                raise RuntimeError(f"{run_id} failed: {child.get('error')}")
            if run_id == "tgv_full_selfcheck":
                check = self_check(load_child_hr(child), args)
                payload["self_check"] = check
                write_json(args.summary_json, payload)
                if not check["passed"]:
                    payload["status"] = "failed_self_check"
                    payload["runtime_sec"] = float(time.perf_counter() - started)
                    payload["failure_note"] = "Full-run TGV self-check did not match the submitted EP10 highpass anchor; split/FRC values are intentionally not trusted."
                    write_json(args.summary_json, payload)
                    return 2

        hp_a = load_child_hr(children["tgv_split_seed42_a"])
        hp_b = load_child_hr(children["tgv_split_seed42_b"])
        curve = frc_curve(fill_nan(hp_a), fill_nan(hp_b), scale=SCALE, crop_lr_px=16, tukey_alpha=0.25)
        cutoff = find_cutoff(curve, "threshold_1_7")
        curve_path = args.output_dir / "frc_curve_tgv_actual_seed42.csv"
        curve.to_csv(curve_path, index=False)
        metrics: dict[str, Any] = {
            "split_half_nrmse": nrmse_pair(hp_a, hp_b),
            "frc_cutoff_period_um_1_7": float(cutoff.period_um),
            "frc_cutoff_frequency_um_inv_1_7": float(cutoff.frequency_um_inv),
            "frc_cutoff_crossed_1_7": bool(cutoff.crossed),
            "frc_curve_csv": curve_path,
            "split_balance_max_abs_diff": int(balance["abs_diff"].max()),
            "split_n_a": int(a_idx.size),
            "split_n_b": int(b_idx.size),
        }
        for period in FRC_PERIODS:
            metrics[f"frc_{int(period)}um"] = interpolate_frc(curve, period)

        payload.update(
            {
                "status": "success",
                "runtime_sec": float(time.perf_counter() - started),
                "metrics": metrics,
                "split": {
                    "a_indices": a_idx.astype(int).tolist(),
                    "b_indices": b_idx.astype(int).tolist(),
                    "balance": balance.to_dict(orient="records"),
                },
                "outputs": {
                    "summary_json": args.summary_json,
                    "full_selfcheck_highpass": children["tgv_full_selfcheck"]["hr_npy"],
                    "split_a_highpass": children["tgv_split_seed42_a"]["hr_npy"],
                    "split_b_highpass": children["tgv_split_seed42_b"]["hr_npy"],
                    "frc_curve_csv": curve_path,
                },
            }
        )
        write_json(args.summary_json, payload)
        print(f"[tgv-split] wrote {rel(args.summary_json)}")
        print(f"[tgv-split] split_half_nrmse={metrics['split_half_nrmse']:.6f}, FRC16={metrics['frc_16um']:.6f}")
        return 0
    except Exception as exc:  # noqa: BLE001 - persist an honest failed manifest.
        payload["status"] = "failed"
        payload["runtime_sec"] = float(time.perf_counter() - started)
        payload["error"] = repr(exc)
        write_json(args.summary_json, payload)
        print(f"[tgv-split] failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
