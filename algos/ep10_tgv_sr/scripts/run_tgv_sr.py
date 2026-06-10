#!/usr/bin/env python3
"""Run EP10 MAP-TGV SR sweep on synthetic and 248 clean-frame real data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
ALGO_ROOT = SCRIPT_DIR.parents[0]
PROJECT_ROOT = SCRIPT_DIR.parents[2]
for path in (
    ALGO_ROOT / "src",
    PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src",
    PROJECT_ROOT / "core" / "src",
):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from common.alignment import load_alignment_shifts  # noqa: E402
from common.data_loader import (  # noqa: E402
    bicubic_upsample,
    highpass_preprocess,
    load_main_session_frames,
    offset_correction,
)
from common.forward_model import forward  # noqa: E402
from common.metrics import artifact_score, split_half_consistency  # noqa: E402
from ep10_tgv_sr import get_tgv_backend_provenance, reconstruct_map_tgv, tgv_denoise  # noqa: E402
from map_tv.map_tv import reconstruct_map_tv, tv_denoise_chambolle  # noqa: E402
from thermal_core.plotting import COLORMAPS, FIGURE_SIZES, savefig_academic, setup_academic_style  # noqa: E402


OUTPUT_DIR = PROJECT_ROOT / "output" / "ep10_tgv_sr"
DATA_DIR = PROJECT_ROOT / "data" / "data_raw" / "infrared_avi"
FRAME_AUDIT_CSV = PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv"
ALIGNMENT_CSV = PROJECT_ROOT / "output" / "ep05_contour_alignment" / "contour_alignment_results.csv"

RESULT_COLUMNS = [
    "method",
    "label",
    "lambda_tv",
    "alpha_ratio",
    "psf_sigma",
    "max_iter",
    "step_size",
    "n_splits",
    "tgv_inner_iter",
    "tv_inner_iter",
    "split_half_nrmse_median",
    "split_half_corr_median",
    "holdout_mse",
    "artifact_score",
    "raw_control_corr",
    "tgv_device",
    "tgv_backend",
    "tgv_backend_status",
    "tgv_backend_device",
    "tgv_backend_error",
    "full_session_metrics",
    "hr_cache_file",
    "convergence_file",
]

_HP_FRAMES: np.ndarray | None = None
_SHIFTS: np.ndarray | None = None
_REF_HP_HR: np.ndarray | None = None
_WORKER_CONFIG: dict[str, Any] | None = None


@dataclass(frozen=True)
class TgvSpec:
    lambda_tv: float
    psf_sigma: float
    device: str | int


def parse_float_grid(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("grid must contain at least one value")
    return values


def token(value: float) -> str:
    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def cache_hr_name(spec: TgvSpec) -> str:
    return f"full_hr_lambda{token(spec.lambda_tv)}_sigma{token(spec.psf_sigma)}.npy"


def parse_device_grid(text: str) -> list[str | int]:
    values: list[str | int] = []
    for item in text.split(","):
        part = item.strip().lower()
        if not part:
            continue
        values.append(int(part) if part.isdigit() else part)
    if not values:
        raise argparse.ArgumentTypeError("device grid must contain at least one value")
    return values


def available_gpu_indices() -> list[int]:
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception:
        return []
    devices: list[int] = []
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.isdigit():
            devices.append(int(line))
    return devices


def resolve_tgv_devices(device_arg: str, outer_workers: int | None) -> tuple[list[str | int], int]:
    key = device_arg.strip().lower()
    if key == "auto":
        devices: list[str | int] = available_gpu_indices()
        if not devices:
            devices = ["cpu"]
    else:
        devices = parse_device_grid(device_arg)

    if outer_workers is None:
        outer_workers = len(devices) if devices and devices != ["cpu"] else 1
    outer_workers = max(1, min(int(outer_workers), len(devices) if devices and devices != ["cpu"] else int(outer_workers)))
    return devices, outer_workers


def default_workers() -> int:
    """Conservative thread-level parallelism for one reconstruction."""
    return max(1, min(4, (os.cpu_count() or 2) // 4))


def center_crop(image: np.ndarray, fraction: float = 1.0 / 3.0) -> np.ndarray:
    h, w = image.shape
    ch = max(1, int(round(h * fraction)))
    cw = max(1, int(round(w * fraction)))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return image[y0 : y0 + ch, x0 : x0 + cw]


def robust_limits(images: list[np.ndarray], *, symmetric: bool = True) -> tuple[float, float]:
    values = np.concatenate([np.asarray(img, dtype=np.float32).ravel() for img in images])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    if symmetric:
        limit = float(np.nanpercentile(np.abs(values), 99.0))
        return -limit, limit
    return float(np.nanpercentile(values, 1.0)), float(np.nanpercentile(values, 99.0))


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(aa) & np.isfinite(bb)
    if int(valid.sum()) < 2:
        return float("nan")
    return float(np.corrcoef(aa[valid], bb[valid])[0, 1])


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def make_piecewise_linear(seed: int = 101) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:64, 0:64]
    clean = np.zeros((64, 64), dtype=np.float64)
    left = x < 32
    right = ~left
    clean[left] = 0.20 + 0.45 * x[left] / 31.0 + 0.12 * y[left] / 63.0
    clean[right] = 0.75 - 0.18 * (x[right] - 32) / 31.0 + 0.08 * y[right] / 63.0
    clean += 0.12 * ((x > 44) & (x < 55) & (y > 14) & (y < 45))
    noisy = clean + rng.normal(0.0, 0.06, clean.shape)
    return clean, noisy


def second_difference_score(image: np.ndarray) -> float:
    roi = np.asarray(image, dtype=np.float64)[8:56, 4:28]
    return float(np.mean(np.abs(np.diff(roi, n=2, axis=1))))


def save_synthetic_figure(
    output_path: Path,
    *,
    clean: np.ndarray,
    noisy: np.ndarray,
    tv: np.ndarray,
    tgv: np.ndarray,
    metrics: dict[str, Any],
) -> Path:
    setup_academic_style()
    images = [clean, noisy, tv, tgv]
    vmin, vmax = robust_limits(images, symmetric=False)
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.4), constrained_layout=True)
    panels = [
        (clean, "Clean piecewise linear"),
        (noisy, "Noisy input"),
        (tv, "TV denoise"),
        (tgv, "TGV denoise"),
    ]
    for ax, (image, title) in zip(axes.ravel()[:4], panels, strict=True):
        ax.imshow(image, cmap=COLORMAPS["temperature"], vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title, fontsize=8.5, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])

    row = 32
    axes[1, 1].plot(clean[row], label="clean", linewidth=1.0)
    axes[1, 1].plot(noisy[row], label="noisy", linewidth=0.7, alpha=0.65)
    axes[1, 1].plot(tv[row], label="TV", linewidth=1.0)
    axes[1, 1].plot(tgv[row], label="TGV", linewidth=1.0)
    axes[1, 1].set_title("Center row profile")
    axes[1, 1].set_xlabel("column")
    axes[1, 1].set_ylabel("intensity")
    axes[1, 1].legend(loc="best", frameon=False)

    names = ["Noisy", "TV", "TGV"]
    scores = [metrics["noisy_second_diff"], metrics["tv_second_diff"], metrics["tgv_second_diff"]]
    axes[1, 2].bar(names, scores, color=["0.65", "#4477AA", "#CC6677"])
    axes[1, 2].set_title("Ramp staircasing proxy")
    axes[1, 2].set_ylabel("mean abs second diff")

    fig.suptitle(
        "EP10 synthetic TGV gate: TGV should suppress noise while preserving a linear ramp",
        fontsize=10,
        fontweight="bold",
    )
    return savefig_academic(fig, output_path)


def run_synthetic_validation(
    output_dir: Path,
    *,
    weight: float,
    alpha_ratio: float,
    tgv_device: str | int | None = "auto",
) -> dict[str, Any]:
    clean, noisy = make_piecewise_linear()
    tv = tv_denoise_chambolle(noisy, weight=weight, max_iter=120)
    tgv = tgv_denoise(noisy, weight=weight, alpha_ratio=alpha_ratio, max_iter=120, device=tgv_device)
    metrics = {
        "weight": float(weight),
        "alpha_ratio": float(alpha_ratio),
        "tgv_device": str(tgv_device),
        "noisy_mse": float(np.mean((noisy - clean) ** 2)),
        "tv_mse": float(np.mean((tv - clean) ** 2)),
        "tgv_mse": float(np.mean((tgv - clean) ** 2)),
        "noisy_second_diff": second_difference_score(noisy),
        "tv_second_diff": second_difference_score(tv),
        "tgv_second_diff": second_difference_score(tgv),
    }
    metrics["passed"] = bool(
        metrics["tv_mse"] < metrics["noisy_mse"]
        and metrics["tgv_mse"] < metrics["noisy_mse"]
        and metrics["tgv_second_diff"] < 0.5 * metrics["tv_second_diff"]
    )
    save_synthetic_figure(
        output_dir / "synthetic_validation.png",
        clean=clean,
        noisy=noisy,
        tv=tv,
        tgv=tgv,
        metrics=metrics,
    )
    (output_dir / "synthetic_validation.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if not metrics["passed"]:
        raise RuntimeError(f"Synthetic TGV validation failed: {metrics}")
    return metrics


def reconstruct_holdout(
    hp_frames: np.ndarray,
    shifts: np.ndarray,
    sr_method: Callable[..., Any],
    *,
    psf_sigma: float,
    sr_kwargs: dict[str, Any],
) -> tuple[np.ndarray, float, pd.DataFrame]:
    indices = np.arange(len(hp_frames))
    holdout_idx = indices[indices % 5 == 0]
    train_idx = indices[indices % 5 != 0]
    result = sr_method(hp_frames[train_idx], shifts[train_idx], **sr_kwargs)
    if isinstance(result, tuple):
        hr, records = result
    else:
        hr, records = result, []
    hr = np.asarray(hr, dtype=np.float32)
    rows = []
    for idx in holdout_idx:
        pred = forward(hr, shifts[idx], psf_sigma=psf_sigma)
        residual = pred.astype(np.float32, copy=False) - hp_frames[idx]
        rows.append({"frame_index": int(idx), "mse": float(np.mean(residual * residual))})
    return hr, float(np.mean([row["mse"] for row in rows])), pd.DataFrame(rows)


def save_comparison_figure(
    output_path: Path,
    *,
    ref_hp_hr: np.ndarray,
    map_tv_hr: np.ndarray,
    tgv_hr: np.ndarray,
    title: str,
) -> Path:
    setup_academic_style()
    crops = [center_crop(img) for img in (ref_hp_hr, map_tv_hr, tgv_hr)]
    vmin, vmax = robust_limits(crops, symmetric=True)
    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["double_col"], constrained_layout=True)
    panels = [
        (crops[0], "Raw-control highpass"),
        (crops[1], "MAP-TV baseline"),
        (crops[2], "MAP-TGV best"),
    ]
    for ax, (img, label) in zip(axes, panels, strict=True):
        ax.imshow(img, cmap=COLORMAPS["residual_diff"], vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(label, fontsize=8.5, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(title, fontsize=10, fontweight="bold")
    return savefig_academic(fig, output_path)


def build_raw_control_reference(raw_frames: np.ndarray, *, sigma_bg: float) -> np.ndarray:
    ref_idx = len(raw_frames) // 2
    ref_raw_hr = bicubic_upsample(raw_frames[ref_idx], scale=2)
    return highpass_preprocess(ref_raw_hr, sigma_bg=float(sigma_bg) * 2.0)


def append_result(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_csv(output_dir / "sweep_results.csv", index=False)


def _path_signature(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    payload: dict[str, Any] = {"path": str(resolved), "exists": resolved.exists()}
    if resolved.exists() and resolved.is_file():
        stat = resolved.stat()
        payload.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return payload


def _input_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "data_dir": _path_signature(args.data_dir),
        "frame_audit_csv": _path_signature(args.frame_audit_csv),
        "alignment_csv": _path_signature(args.alignment_csv),
        "alignment_method": str(args.alignment_method),
        "highpass_sigma": float(args.highpass_sigma),
        "ref_hp_sigma_bg_hr_px": float(args.highpass_sigma) * 2.0,
    }


def _signature_digest(signature: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(signature, sort_keys=True).encode("utf-8")).hexdigest()


def prepare_real_inputs(args: argparse.Namespace, output_dir: Path) -> dict[str, str]:
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    hp_path = cache_dir / "hp_frames.npy"
    shifts_path = cache_dir / "shifts.npy"
    ref_hp_path = cache_dir / "ref_hp_hr.npy"
    meta_path = cache_dir / "metadata.csv"
    summary_path = cache_dir / "input_summary.json"
    signature = _input_signature(args)
    signature_digest = _signature_digest(signature)

    if args.reuse_cached_inputs and hp_path.exists() and shifts_path.exists() and ref_hp_path.exists():
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("input_signature_digest") == signature_digest:
                return {"hp_frames": str(hp_path), "shifts": str(shifts_path), "ref_hp_hr": str(ref_hp_path)}
        print("Cached inputs exist but input signature changed; rebuilding cache.")

    raw_input_frames, metadata = load_main_session_frames(
        args.data_dir,
        args.frame_audit_csv,
        workers=args.io_workers,
        dtype=np.float32,
    )
    shifts = load_alignment_shifts(
        method=args.alignment_method,
        metadata=metadata,
        alignment_csv=args.alignment_csv,
    )
    hp_frames = highpass_preprocess(raw_input_frames, sigma_bg=args.highpass_sigma, workers=args.io_workers)
    raw_frames = offset_correction(raw_input_frames)
    ref_hp_hr = build_raw_control_reference(raw_frames, sigma_bg=args.highpass_sigma)

    np.save(hp_path, hp_frames.astype(np.float32, copy=False))
    np.save(shifts_path, shifts.astype(np.float32, copy=False))
    np.save(ref_hp_path, ref_hp_hr.astype(np.float32, copy=False))
    metadata.to_csv(meta_path, index=False)
    write_json(
        summary_path,
        {
            "frames_shape": list(raw_input_frames.shape),
            "hp_frames": display_path(hp_path),
            "shifts_shape": list(shifts.shape),
            "alignment_method": args.alignment_method,
            "highpass_sigma": float(args.highpass_sigma),
            "ref_hp_sigma_bg_hr_px": float(args.highpass_sigma) * 2.0,
            "input_signature": signature,
            "input_signature_digest": signature_digest,
        },
    )
    return {"hp_frames": str(hp_path), "shifts": str(shifts_path), "ref_hp_hr": str(ref_hp_path)}


def _load_worker_arrays(paths: dict[str, str]) -> None:
    global _HP_FRAMES, _SHIFTS, _REF_HP_HR
    _HP_FRAMES = np.load(paths["hp_frames"], mmap_mode="r")
    _SHIFTS = np.load(paths["shifts"], mmap_mode="r")
    _REF_HP_HR = np.load(paths["ref_hp_hr"], mmap_mode="r")


def init_worker(paths: dict[str, str], config: dict[str, Any]) -> None:
    global _WORKER_CONFIG
    _load_worker_arrays(paths)
    _WORKER_CONFIG = dict(config)


def reconstruct_tgv_for_spec(
    frames: np.ndarray,
    shifts: np.ndarray,
    spec: TgvSpec,
    config: dict[str, Any],
    *,
    return_records: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict[str, object]]]:
    image, records = reconstruct_map_tgv(
        frames,
        shifts,
        lambda_tv=spec.lambda_tv,
        alpha_ratio=float(config["alpha_ratio"]),
        psf_sigma=spec.psf_sigma,
        max_iter=int(config["max_iter"]),
        step_size=float(config["step_size"]),
        use_fista=True,
        workers=int(config["gradient_workers"]),
        tgv_inner_iter=int(config["tgv_inner_iter"]),
        tgv_device=spec.device,
        aniso_ratio_y=float(config.get("aniso_ratio_y", 1.0)),
        coverage_weighted=bool(config.get("coverage_weighted", False)),
    )
    image = np.asarray(image, dtype=np.float32)
    if return_records:
        return image, list(records)
    return image


def evaluate_tgv_param(spec: TgvSpec) -> dict[str, Any]:
    if _HP_FRAMES is None or _SHIFTS is None or _REF_HP_HR is None or _WORKER_CONFIG is None:
        raise RuntimeError("worker arrays are not initialized")
    config = _WORKER_CONFIG
    label = f"tgv_lam{spec.lambda_tv:g}_sigma{spec.psf_sigma:g}"
    started = time.perf_counter()
    sr_kwargs = {
        "lambda_tv": float(spec.lambda_tv),
        "alpha_ratio": float(config["alpha_ratio"]),
        "psf_sigma": float(spec.psf_sigma),
        "max_iter": int(config["max_iter"]),
        "step_size": float(config["step_size"]),
        "use_fista": True,
        "workers": int(config["gradient_workers"]),
        "tgv_inner_iter": int(config["tgv_inner_iter"]),
        "tgv_device": spec.device,
        "aniso_ratio_y": float(config.get("aniso_ratio_y", 1.0)),
        "coverage_weighted": bool(config.get("coverage_weighted", False)),
    }

    split_df = split_half_consistency(
        _HP_FRAMES,
        _SHIFTS,
        reconstruct_map_tgv,
        n_splits=int(config["n_splits"]),
        random_state=int(config["random_state"]),
        **sr_kwargs,
    )
    split_df.insert(0, "method", "map_tgv")
    split_df.insert(1, "label", label)
    split_df.insert(2, "lambda_tv", float(spec.lambda_tv))
    split_df.insert(3, "alpha_ratio", float(config["alpha_ratio"]))
    split_df.insert(4, "psf_sigma", float(spec.psf_sigma))

    _holdout_hr, holdout_mse, holdout_df = reconstruct_holdout(
        _HP_FRAMES,
        _SHIFTS,
        reconstruct_map_tgv,
        psf_sigma=float(spec.psf_sigma),
        sr_kwargs=sr_kwargs,
    )
    holdout_df.insert(0, "method", "map_tgv")
    holdout_df.insert(1, "label", label)
    holdout_df.insert(2, "lambda_tv", float(spec.lambda_tv))
    holdout_df.insert(3, "alpha_ratio", float(config["alpha_ratio"]))
    holdout_df.insert(4, "psf_sigma", float(spec.psf_sigma))

    full_hr, convergence_records = reconstruct_tgv_for_spec(_HP_FRAMES, _SHIFTS, spec, config, return_records=True)
    backend = get_tgv_backend_provenance()
    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    hr_cache = cache_dir / cache_hr_name(spec)
    np.save(hr_cache, full_hr.astype(np.float32, copy=False))
    detail_dir = Path(config["detail_dir"])
    detail_dir.mkdir(parents=True, exist_ok=True)
    convergence_file = detail_dir / f"convergence_{label.replace('.', 'p')}.csv"
    convergence = pd.DataFrame.from_records(convergence_records)
    if not convergence.empty:
        if "method" not in convergence.columns:
            convergence.insert(0, "method", "map_tgv")
        if "label" not in convergence.columns:
            convergence.insert(1, "label", label)
        if "lambda_tv" not in convergence.columns:
            convergence.insert(2, "lambda_tv", float(spec.lambda_tv))
        if "alpha_ratio" not in convergence.columns:
            convergence.insert(3, "alpha_ratio", float(config["alpha_ratio"]))
        if "psf_sigma" not in convergence.columns:
            convergence.insert(4, "psf_sigma", float(spec.psf_sigma))
    convergence.to_csv(convergence_file, index=False)

    row = {
        "method": "map_tgv",
        "label": label,
        "n_input_frames": int(len(_HP_FRAMES)),
        "input_frame_count": int(len(_HP_FRAMES)),
        "lambda_tv": float(spec.lambda_tv),
        "alpha_ratio": float(config["alpha_ratio"]),
        "aniso_ratio_y": float(config.get("aniso_ratio_y", 1.0)),
        "coverage_weighted": bool(config.get("coverage_weighted", False)),
        "psf_sigma": float(spec.psf_sigma),
        "max_iter": int(config["max_iter"]),
        "step_size": float(config["step_size"]),
        "n_splits": int(config["n_splits"]),
        "tgv_inner_iter": int(config["tgv_inner_iter"]),
        "tv_inner_iter": np.nan,
        "split_half_nrmse_median": float(split_df["nrmse"].median()),
        "split_half_corr_median": float(split_df["corr"].median()),
        "holdout_mse": float(holdout_mse),
        "artifact_score": float(artifact_score(full_hr)),
        "raw_control_corr": pearson_corr(full_hr, _REF_HP_HR),
        "tgv_device": str(spec.device),
        "tgv_backend": str(backend.get("backend", "")),
        "tgv_backend_status": str(backend.get("status", "")),
        "tgv_backend_device": str(backend.get("selected_device", "")),
        "tgv_backend_error": "" if backend.get("error") is None else str(backend.get("error")),
        "full_session_metrics": True,
        "hr_cache_file": str(hr_cache),
        "convergence_file": str(convergence_file),
        "runtime_sec": float(time.perf_counter() - started),
    }
    return {
        "row": row,
        "split_rows": split_df.to_dict("records"),
        "holdout_rows": holdout_df.to_dict("records"),
    }


def existing_completed(results_path: Path) -> set[tuple[float, float]]:
    if not results_path.exists():
        return set()
    table = pd.read_csv(results_path)
    if not {"lambda_tv", "psf_sigma", "method"}.issubset(table.columns):
        return set()
    if "full_session_metrics" not in table.columns:
        return set()
    table = table[table["method"].eq("map_tgv")]
    table = table[table["full_session_metrics"].astype(str).str.lower().isin({"true", "1"})]
    return {(float(row.lambda_tv), float(row.psf_sigma)) for row in table.itertuples()}


def save_results_table(path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame(columns=RESULT_COLUMNS)
    for col in RESULT_COLUMNS:
        if col not in table.columns:
            table[col] = np.nan
    preferred = [col for col in RESULT_COLUMNS if col in table.columns]
    extra = [col for col in table.columns if col not in preferred]
    table = table[preferred + extra].copy()
    if {"method", "psf_sigma", "lambda_tv"}.issubset(table.columns):
        table = table.sort_values(["method", "psf_sigma", "lambda_tv"]).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(path, index=False)
    return table


def save_detail_tables(
    output_dir: Path,
    split_rows: list[dict[str, Any]],
    holdout_rows: list[dict[str, Any]],
) -> None:
    if split_rows:
        pd.DataFrame(split_rows).to_csv(output_dir / "split_half_details.csv", index=False)
    if holdout_rows:
        pd.DataFrame(holdout_rows).to_csv(output_dir / "holdout_details.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA_DIR)
    parser.add_argument("--frame-audit-csv", type=Path, default=FRAME_AUDIT_CSV)
    parser.add_argument("--alignment-csv", type=Path, default=ALIGNMENT_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--alignment-method", default="contour_refined")
    parser.add_argument("--lambda-grid", default="0.0003,0.001,0.003")
    parser.add_argument("--psf-grid", default="0.18,0.50")
    parser.add_argument("--alpha-ratio", type=float, default=2.0)
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--tgv-inner-iter", type=int, default=80)
    parser.add_argument("--tv-inner-iter", type=int, default=30)
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--highpass-sigma", type=float, default=5.0)
    parser.add_argument("--synthetic-weight", type=float, default=0.06)
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--aniso-ratio-y", type=float, default=1.5,
                        help="Y-direction TGV regularization scale (>1 suppresses horizontal stripes; default: 1.5)")
    parser.add_argument("--coverage-weighted", action=argparse.BooleanOptionalAction, default=True,
                        help="Normalize data gradient per-pixel by coverage map (default: enabled)")
    parser.add_argument("--workers", type=int, default=default_workers(),
                        help=f"thread workers for data-gradient computation (default: {default_workers()})")
    parser.add_argument("--outer-workers", type=int, default=None,
                        help="parameter-level worker processes (default: number of selected GPUs, or 1 on CPU)")
    parser.add_argument("--tgv-device", default="auto",
                        help="TGV backend: auto, cpu, gpu, a GPU index, or comma-separated GPU indices")
    parser.add_argument("--io-workers", type=int, default=min(8, max(1, os.cpu_count() or 1)))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reuse-cached-inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-params", type=int, default=None)
    return parser.parse_args()


def run_real_sweep(args: argparse.Namespace, output_dir: Path, started: float) -> None:
    lambdas = parse_float_grid(args.lambda_grid)
    psf_sigmas = parse_float_grid(args.psf_grid)
    devices, outer_workers = resolve_tgv_devices(args.tgv_device, args.outer_workers)
    print(
        f"TGV parallelism: outer_workers={outer_workers}, "
        f"gradient_workers={args.workers}, devices={devices}"
    )

    paths = prepare_real_inputs(args, output_dir)
    config = {
        "alpha_ratio": float(args.alpha_ratio),
        "max_iter": int(args.max_iter),
        "step_size": float(args.step_size),
        "gradient_workers": int(args.workers),
        "tgv_inner_iter": int(args.tgv_inner_iter),
        "n_splits": int(args.n_splits),
        "random_state": int(args.random_state),
        "cache_dir": str(output_dir / "cache"),
        "detail_dir": str(output_dir / "details"),
        "aniso_ratio_y": float(args.aniso_ratio_y),
        "coverage_weighted": bool(args.coverage_weighted),
    }

    specs = [
        TgvSpec(float(lam), float(sigma), devices[idx % len(devices)])
        for idx, (lam, sigma) in enumerate((lam, sigma) for lam in lambdas for sigma in psf_sigmas)
    ]
    if args.max_params is not None:
        specs = specs[: int(args.max_params)]

    results_path = output_dir / "sweep_results.csv"
    rows: list[dict[str, Any]] = []
    split_rows: list[dict[str, Any]] = []
    holdout_rows: list[dict[str, Any]] = []
    if args.resume and results_path.exists():
        existing = pd.read_csv(results_path)
        if "full_session_metrics" in existing.columns:
            existing = existing[existing["full_session_metrics"].astype(str).str.lower().isin({"true", "1"})]
        else:
            existing = existing.iloc[0:0]
        rows = existing.to_dict("records")
        done = existing_completed(results_path)
        specs = [spec for spec in specs if (float(spec.lambda_tv), float(spec.psf_sigma)) not in done]
        if rows and (output_dir / "split_half_details.csv").exists():
            split_rows = pd.read_csv(output_dir / "split_half_details.csv").to_dict("records")
        if rows and (output_dir / "holdout_details.csv").exists():
            holdout_rows = pd.read_csv(output_dir / "holdout_details.csv").to_dict("records")
    elif results_path.exists() and not args.force:
        raise FileExistsError(f"{results_path} exists; pass --force to overwrite or keep --resume")

    if args.force and results_path.exists() and not args.resume:
        results_path.unlink()

    if specs:
        if outer_workers == 1:
            _load_worker_arrays(paths)
            global _WORKER_CONFIG
            _WORKER_CONFIG = dict(config)
            iterator = tqdm(specs, desc="MAP-TGV sweep", unit="param")
            for spec in iterator:
                payload = evaluate_tgv_param(spec)
                rows.append(payload["row"])
                split_rows.extend(payload["split_rows"])
                holdout_rows.extend(payload["holdout_rows"])
                save_results_table(results_path, rows)
                save_detail_tables(output_dir, split_rows, holdout_rows)
        else:
            with ProcessPoolExecutor(
                max_workers=outer_workers,
                initializer=init_worker,
                initargs=(paths, config),
            ) as executor:
                futures = {executor.submit(evaluate_tgv_param, spec): spec for spec in specs}
                for future in tqdm(as_completed(futures), total=len(futures), desc="MAP-TGV sweep", unit="param"):
                    payload = future.result()
                    rows.append(payload["row"])
                    split_rows.extend(payload["split_rows"])
                    holdout_rows.extend(payload["holdout_rows"])
                    save_results_table(results_path, rows)
                    save_detail_tables(output_dir, split_rows, holdout_rows)

    table = save_results_table(results_path, rows)
    save_detail_tables(output_dir, split_rows, holdout_rows)

    _load_worker_arrays(paths)
    assert _HP_FRAMES is not None and _SHIFTS is not None and _REF_HP_HR is not None
    tgv_table = table[table["method"].eq("map_tgv")].copy()
    if tgv_table.empty:
        raise RuntimeError("No MAP-TGV reconstruction completed")
    tgv_table = tgv_table.sort_values(["split_half_nrmse_median", "holdout_mse", "artifact_score"]).reset_index(drop=True)
    best = tgv_table.iloc[0]
    best_spec = TgvSpec(float(best["lambda_tv"]), float(best["psf_sigma"]), str(best["tgv_device"]))
    best_cache = Path(str(best["hr_cache_file"]))
    if best_cache.exists():
        best_hr = np.load(best_cache).astype(np.float32, copy=False)
    else:
        best_hr = reconstruct_tgv_for_spec(_HP_FRAMES, _SHIFTS, best_spec, config)
        np.save(best_cache, best_hr.astype(np.float32, copy=False))
    np.save(output_dir / "best_hr_highpass.npy", best_hr.astype(np.float32, copy=False))

    baseline_kwargs = {
        "lambda_tv": 0.001,
        "psf_sigma": 0.50,
        "max_iter": int(args.max_iter),
        "step_size": float(args.step_size),
        "use_fista": True,
        "workers": int(args.workers),
        "tv_inner_iter": int(args.tv_inner_iter),
    }
    baseline_split = split_half_consistency(
        _HP_FRAMES,
        _SHIFTS,
        reconstruct_map_tv,
        n_splits=args.n_splits,
        random_state=args.random_state,
        **baseline_kwargs,
    )
    baseline_split.insert(0, "method", "map_tv_baseline")
    baseline_split.insert(1, "label", "map_tv_lambda0.001_sigma0.50")
    baseline_split.insert(2, "lambda_tv", 0.001)
    baseline_split.insert(3, "alpha_ratio", np.nan)
    baseline_split.insert(4, "psf_sigma", 0.50)
    split_rows.extend(baseline_split.to_dict("records"))

    _holdout_hr, baseline_holdout_mse, baseline_holdout = reconstruct_holdout(
        _HP_FRAMES,
        _SHIFTS,
        reconstruct_map_tv,
        psf_sigma=0.50,
        sr_kwargs=baseline_kwargs,
    )
    baseline_holdout.insert(0, "method", "map_tv_baseline")
    baseline_holdout.insert(1, "label", "map_tv_lambda0.001_sigma0.50")
    baseline_holdout.insert(2, "lambda_tv", 0.001)
    baseline_holdout.insert(3, "alpha_ratio", np.nan)
    baseline_holdout.insert(4, "psf_sigma", 0.50)
    holdout_rows.extend(baseline_holdout.to_dict("records"))

    map_tv_hr, _records = reconstruct_map_tv(_HP_FRAMES, _SHIFTS, **baseline_kwargs)
    np.save(output_dir / "map_tv_baseline_highpass.npy", map_tv_hr.astype(np.float32, copy=False))
    rows = [row for row in rows if row.get("method") != "map_tv_baseline"]
    rows.append(
        {
            "method": "map_tv_baseline",
            "label": "map_tv_lambda0.001_sigma0.50",
            "n_input_frames": int(len(_HP_FRAMES)),
            "input_frame_count": int(len(_HP_FRAMES)),
            "lambda_tv": 0.001,
            "alpha_ratio": np.nan,
            "psf_sigma": 0.50,
            "max_iter": int(args.max_iter),
            "step_size": float(args.step_size),
            "n_splits": int(args.n_splits),
            "tgv_inner_iter": np.nan,
            "tv_inner_iter": int(args.tv_inner_iter),
            "split_half_nrmse_median": float(baseline_split["nrmse"].median()),
            "split_half_corr_median": float(baseline_split["corr"].median()),
            "holdout_mse": float(baseline_holdout_mse),
            "artifact_score": float(artifact_score(map_tv_hr)),
            "raw_control_corr": pearson_corr(map_tv_hr, _REF_HP_HR),
            "tgv_device": np.nan,
            "tgv_backend": np.nan,
            "tgv_backend_status": np.nan,
            "tgv_backend_device": np.nan,
            "tgv_backend_error": np.nan,
            "full_session_metrics": True,
            "hr_cache_file": str(output_dir / "map_tv_baseline_highpass.npy"),
            "convergence_file": np.nan,
        }
    )
    save_results_table(results_path, rows)
    save_detail_tables(output_dir, split_rows, holdout_rows)

    save_comparison_figure(
        output_dir / "tgv_vs_tv_comparison.png",
        ref_hp_hr=_REF_HP_HR,
        map_tv_hr=map_tv_hr,
        tgv_hr=best_hr,
        title=f"EP10 center highpass crop: {best['label']} vs MAP-TV baseline",
    )
    write_json(
        output_dir / "run_summary.json",
        {
            "best_label": str(best["label"]),
            "best_label_scope": "map_tgv_only",
            "best_tgv_label": str(best["label"]),
            "elapsed_sec": float(time.perf_counter() - started),
            "n_frames": int(len(_HP_FRAMES)),
            "alignment_method": args.alignment_method,
            "highpass_sigma": float(args.highpass_sigma),
            "lambda_grid": lambdas,
            "psf_grid": psf_sigmas,
            "alpha_ratio": float(args.alpha_ratio),
            "aniso_ratio_y": float(args.aniso_ratio_y),
            "coverage_weighted": bool(args.coverage_weighted),
            "max_iter": int(args.max_iter),
            "step_size": float(args.step_size),
            "tgv_inner_iter": int(args.tgv_inner_iter),
            "gradient_workers": int(args.workers),
            "outer_workers": int(outer_workers),
            "tgv_devices": [str(device) for device in devices],
            "full_session_metrics": True,
        },
    )
    print(f"saved EP10 outputs to {display_path(output_dir)}")


def main() -> None:
    args = parse_args()
    start = time.perf_counter()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    synthetic_metrics = run_synthetic_validation(
        output_dir,
        weight=args.synthetic_weight,
        alpha_ratio=args.alpha_ratio,
        tgv_device="cpu",
    )
    print(f"synthetic validation passed: {synthetic_metrics}")
    if args.synthetic_only:
        return

    run_real_sweep(args, output_dir, start)


if __name__ == "__main__":
    main()
