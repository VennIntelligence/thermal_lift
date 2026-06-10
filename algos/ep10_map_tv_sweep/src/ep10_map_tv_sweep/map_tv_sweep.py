"""Reusable MAP-TV parameter-sweep machinery for EP10.

The long-running CLI in ``scripts/run_sweep.py`` is intentionally thin: this
module owns the reconstruction call, validation metrics, Pareto selection, and
diagnostic CSV writing so tests and notebooks can import the same logic.
"""

from __future__ import annotations

import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def find_project_root(start: Path) -> Path:
    current = start.resolve()
    for candidate in (current, *current.parents):
        if (candidate / "AGENTS.md").exists():
            return candidate
    raise RuntimeError(f"Could not find project root from {start}")


PROJECT_ROOT = find_project_root(Path(__file__))
EP06_SRC = PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src"
CORE_SRC = PROJECT_ROOT / "core" / "src"
for path in (EP06_SRC, CORE_SRC):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from common.forward_model import forward  # noqa: E402
from common.metrics import artifact_score, split_half_consistency  # noqa: E402
from map_tv.map_tv import (  # noqa: E402
    _data_gradient_and_loss,
    _data_gradient_and_loss_cached,
    reconstruct_map_tv,
    tv_denoise_chambolle,
    tv_norm,
)
from thermal_core.plotting import COLORMAPS, FIGURE_SIZES, savefig_academic, setup_academic_style  # noqa: E402


RESULT_COLUMNS = [
    "lambda_tv",
    "psf_sigma",
    "split_half_nrmse",
    "split_half_corr",
    "holdout_mse",
    "artifact_score",
    "raw_control_corr",
    "runtime_sec",
    "max_iter",
    "split_half_splits",
    "map_tv_workers",
    "hr_cache_file",
    "convergence_file",
    "split_half_detail_file",
    "holdout_detail_file",
]

_HP_FRAMES: np.ndarray | None = None
_SHIFTS: np.ndarray | None = None
_REF_HP_HR: np.ndarray | None = None
_WORKER_CONFIG: dict[str, Any] | None = None


@dataclass(frozen=True)
class ParamSpec:
    """One MAP-TV lambda/PSF combination."""

    lambda_tv: float
    psf_sigma: float


def token(value: float) -> str:
    """Filesystem-safe token for a numeric parameter value."""

    return f"{float(value):.6g}".replace("-", "m").replace(".", "p")


def spec_label(spec: ParamSpec) -> str:
    return f"lambda{token(spec.lambda_tv)}_sigma{token(spec.psf_sigma)}"


def cache_hr_name(spec: ParamSpec) -> str:
    return f"full_hr_{spec_label(spec)}.npy"


def top_hr_name(spec: ParamSpec) -> str:
    return f"hr_highpass_{token(spec.lambda_tv)}_{token(spec.psf_sigma)}.npy"


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def relative_to_project(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64)
    bb = np.asarray(b, dtype=np.float64)
    if aa.shape != bb.shape:
        raise ValueError(f"correlation shapes differ: {aa.shape} vs {bb.shape}")
    valid = np.isfinite(aa) & np.isfinite(bb)
    if int(valid.sum()) < 2:
        return float("nan")
    av = aa[valid].ravel()
    bv = bb[valid].ravel()
    a_std = float(np.std(av))
    b_std = float(np.std(bv))
    if a_std <= 1e-12 or b_std <= 1e-12:
        return float("nan")
    return float(np.corrcoef(av, bv)[0, 1])


def psnr(reference: np.ndarray, estimate: np.ndarray) -> float:
    ref = np.asarray(reference, dtype=np.float32)
    est = np.asarray(estimate, dtype=np.float32)
    mse = float(np.mean((ref - est) ** 2))
    if mse <= 0:
        return float("inf")
    data_range = float(np.max(ref) - np.min(ref))
    return float(20.0 * np.log10(max(data_range, 1e-6) / math.sqrt(mse)))


def reconstruct_for_spec(
    frames: np.ndarray,
    shifts: np.ndarray,
    spec: ParamSpec,
    config: dict[str, Any],
) -> np.ndarray:
    image, _records = reconstruct_map_tv(
        frames,
        shifts,
        lambda_tv=spec.lambda_tv,
        psf_sigma=spec.psf_sigma,
        max_iter=int(config["max_iter"]),
        step_size=float(config["step_size"]),
        use_fista=bool(config["use_fista"]),
        workers=int(config["map_tv_workers"]),
    )
    return np.asarray(image, dtype=np.float32)


def reconstruct_for_spec_with_records(
    frames: np.ndarray,
    shifts: np.ndarray,
    spec: ParamSpec,
    config: dict[str, Any],
) -> tuple[np.ndarray, list[dict[str, float | int | bool]]]:
    image, records = reconstruct_map_tv(
        frames,
        shifts,
        lambda_tv=spec.lambda_tv,
        psf_sigma=spec.psf_sigma,
        max_iter=int(config["max_iter"]),
        step_size=float(config["step_size"]),
        use_fista=bool(config["use_fista"]),
        workers=int(config["map_tv_workers"]),
    )
    return np.asarray(image, dtype=np.float32), list(records)


def holdout_details_for_spec(
    hp_frames: np.ndarray,
    shifts: np.ndarray,
    spec: ParamSpec,
    config: dict[str, Any],
) -> pd.DataFrame:
    frame_indices = np.arange(len(hp_frames), dtype=int)
    holdout = frame_indices % int(config["holdout_mod"]) == 0
    train = ~holdout
    if not np.any(train) or not np.any(holdout):
        raise ValueError("holdout split must contain both train and holdout frames")

    hr_holdout = reconstruct_for_spec(hp_frames[train], shifts[train], spec, config)
    rows: list[dict[str, Any]] = []
    for frame_index, frame, shift in zip(frame_indices[holdout], hp_frames[holdout], shifts[holdout], strict=True):
        pred = forward(hr_holdout, shift, psf_sigma=spec.psf_sigma)
        valid = np.isfinite(pred) & np.isfinite(frame)
        n_valid = int(valid.sum())
        if n_valid:
            residual = np.asarray(pred[valid], dtype=np.float64) - np.asarray(frame[valid], dtype=np.float64)
            mse = float(np.mean(residual * residual))
        else:
            mse = float("nan")
        rows.append(
            {
                "param_label": spec_label(spec),
                "lambda_tv": float(spec.lambda_tv),
                "psf_sigma": float(spec.psf_sigma),
                "frame_index": int(frame_index),
                "n_valid": n_valid,
                "mse": mse,
                "rmse": float(math.sqrt(mse)) if np.isfinite(mse) else float("nan"),
                "shift_dx_lr_px": float(shift[0]),
                "shift_dy_lr_px": float(shift[1]),
            }
        )
    return pd.DataFrame.from_records(rows)


def holdout_mse_for_spec(
    hp_frames: np.ndarray,
    shifts: np.ndarray,
    spec: ParamSpec,
    config: dict[str, Any],
) -> float:
    details = holdout_details_for_spec(hp_frames, shifts, spec, config)
    return float(details["mse"].mean()) if not details.empty else float("nan")


def _load_worker_arrays(paths: dict[str, str]) -> None:
    global _HP_FRAMES, _SHIFTS, _REF_HP_HR
    _HP_FRAMES = np.load(paths["hp_frames"], mmap_mode="r")
    _SHIFTS = np.load(paths["shifts"], mmap_mode="r")
    _REF_HP_HR = np.load(paths["ref_hp_hr"], mmap_mode="r")


def init_worker(paths: dict[str, str], config: dict[str, Any]) -> None:
    global _WORKER_CONFIG
    _load_worker_arrays(paths)
    _WORKER_CONFIG = dict(config)


def _detail_path(config: dict[str, Any], spec: ParamSpec, kind: str) -> Path:
    detail_dir = Path(config["detail_dir"])
    detail_dir.mkdir(parents=True, exist_ok=True)
    return detail_dir / f"{kind}_{spec_label(spec)}.csv"


def evaluate_param(spec: ParamSpec) -> dict[str, Any]:
    if _HP_FRAMES is None or _SHIFTS is None or _REF_HP_HR is None or _WORKER_CONFIG is None:
        raise RuntimeError("worker arrays are not initialized")
    config = _WORKER_CONFIG
    started = time.time()

    full_hr, convergence_records = reconstruct_for_spec_with_records(_HP_FRAMES, _SHIFTS, spec, config)
    cache_dir = Path(config["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_file = cache_dir / cache_hr_name(spec)
    np.save(cache_file, full_hr.astype(np.float32, copy=False))

    convergence = pd.DataFrame.from_records(convergence_records)
    if not convergence.empty:
        if "param_label" not in convergence.columns:
            convergence.insert(0, "param_label", spec_label(spec))
        if "lambda_tv" not in convergence.columns:
            convergence.insert(1, "lambda_tv", float(spec.lambda_tv))
        if "psf_sigma" not in convergence.columns:
            convergence.insert(2, "psf_sigma", float(spec.psf_sigma))
    convergence_path = _detail_path(config, spec, "convergence")
    convergence.to_csv(convergence_path, index=False)

    split_half = split_half_consistency(
        _HP_FRAMES,
        _SHIFTS,
        reconstruct_map_tv,
        n_splits=int(config["split_half_splits"]),
        random_state=int(config["random_state"]),
        lambda_tv=spec.lambda_tv,
        psf_sigma=spec.psf_sigma,
        max_iter=int(config["max_iter"]),
        step_size=float(config["step_size"]),
        use_fista=bool(config["use_fista"]),
        workers=int(config["map_tv_workers"]),
    )
    split_half.insert(0, "param_label", spec_label(spec))
    split_half.insert(1, "lambda_tv", float(spec.lambda_tv))
    split_half.insert(2, "psf_sigma", float(spec.psf_sigma))
    split_half_path = _detail_path(config, spec, "split_half")
    split_half.to_csv(split_half_path, index=False)

    holdout = holdout_details_for_spec(_HP_FRAMES, _SHIFTS, spec, config)
    holdout_path = _detail_path(config, spec, "holdout")
    holdout.to_csv(holdout_path, index=False)

    result = {
        "n_input_frames": int(len(_HP_FRAMES)),
        "input_frame_count": int(len(_HP_FRAMES)),
        "lambda_tv": float(spec.lambda_tv),
        "psf_sigma": float(spec.psf_sigma),
        "split_half_nrmse": float(split_half["nrmse"].median()),
        "split_half_corr": float(split_half["corr"].median()),
        "holdout_mse": float(holdout["mse"].mean()) if not holdout.empty else float("nan"),
        "artifact_score": float(artifact_score(full_hr)),
        "raw_control_corr": pearson_corr(full_hr, _REF_HP_HR),
        "runtime_sec": float(time.time() - started),
        "max_iter": int(config["max_iter"]),
        "split_half_splits": int(config["split_half_splits"]),
        "map_tv_workers": int(config["map_tv_workers"]),
        "hr_cache_file": relative_to_project(cache_file),
        "convergence_file": relative_to_project(convergence_path),
        "split_half_detail_file": relative_to_project(split_half_path),
        "holdout_detail_file": relative_to_project(holdout_path),
    }
    return result


def existing_completed(results_path: Path) -> set[tuple[float, float]]:
    if not results_path.exists():
        return set()
    table = pd.read_csv(results_path)
    if not {"lambda_tv", "psf_sigma"}.issubset(table.columns):
        return set()
    return {(float(row.lambda_tv), float(row.psf_sigma)) for row in table.itertuples()}


def save_results_table(results_path: Path, rows: list[dict[str, Any]]) -> pd.DataFrame:
    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame(columns=RESULT_COLUMNS)
    for col in RESULT_COLUMNS:
        if col not in table.columns:
            table[col] = np.nan
    table = table[RESULT_COLUMNS].copy()
    sort_cols = [col for col in ["psf_sigma", "lambda_tv"] if col in table.columns]
    if sort_cols:
        table = table.sort_values(sort_cols).reset_index(drop=True)
    results_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_path, index=False)
    return table


def is_dominated(row: pd.Series, table: pd.DataFrame) -> bool:
    better_or_equal = (
        (table["split_half_nrmse"] <= row["split_half_nrmse"])
        & (table["artifact_score"] <= row["artifact_score"])
    )
    strictly_better = (
        (table["split_half_nrmse"] < row["split_half_nrmse"])
        | (table["artifact_score"] < row["artifact_score"])
    )
    return bool((better_or_equal & strictly_better).any())


def pareto_frontier(table: pd.DataFrame) -> pd.DataFrame:
    required = {"split_half_nrmse", "artifact_score"}
    if table.empty or not required.issubset(table.columns):
        return pd.DataFrame()
    clean = table.dropna(subset=["split_half_nrmse", "artifact_score"]).copy()
    if clean.empty:
        return pd.DataFrame()
    mask = [not is_dominated(row, clean) for _, row in clean.iterrows()]
    frontier = clean.loc[mask].copy()
    return frontier.sort_values(["split_half_nrmse", "artifact_score"]).reset_index(drop=True)


def pareto_top3(table: pd.DataFrame) -> pd.DataFrame:
    return pareto_frontier(table).head(3).reset_index(drop=True)


def save_top3_outputs(table: pd.DataFrame, paths: dict[str, str], config: dict[str, Any], output_dir: Path) -> list[dict[str, Any]]:
    top = pareto_top3(table)
    if top.empty:
        return []
    _load_worker_arrays(paths)
    global _WORKER_CONFIG
    _WORKER_CONFIG = dict(config)
    saved: list[dict[str, Any]] = []
    for row in top.itertuples(index=False):
        spec = ParamSpec(float(row.lambda_tv), float(row.psf_sigma))
        cache_file = output_dir / "cache" / cache_hr_name(spec)
        if cache_file.exists():
            hr = np.load(cache_file).astype(np.float32, copy=False)
        else:
            assert _HP_FRAMES is not None and _SHIFTS is not None
            hr = reconstruct_for_spec(_HP_FRAMES, _SHIFTS, spec, config)
            np.save(cache_file, hr.astype(np.float32, copy=False))
        out_path = output_dir / top_hr_name(spec)
        np.save(out_path, hr.astype(np.float32, copy=False))
        item = {
            "n_input_frames": int(row.n_input_frames) if hasattr(row, "n_input_frames") else int(len(_HP_FRAMES)),
            "input_frame_count": int(row.input_frame_count) if hasattr(row, "input_frame_count") else int(len(_HP_FRAMES)),
            "lambda_tv": spec.lambda_tv,
            "psf_sigma": spec.psf_sigma,
            "split_half_nrmse": float(row.split_half_nrmse),
            "split_half_corr": float(getattr(row, "split_half_corr", float("nan"))),
            "artifact_score": float(row.artifact_score),
            "holdout_mse": float(row.holdout_mse),
            "raw_control_corr": float(row.raw_control_corr),
            "hr_cache_file": relative_to_project(cache_file),
            "hr_highpass_file": relative_to_project(out_path),
        }
        saved.append(item)
    return saved


def save_best_params(table: pd.DataFrame, paths: dict[str, str], config: dict[str, Any], output_dir: Path) -> None:
    top = save_top3_outputs(table, paths, config, output_dir)
    frontier = pareto_frontier(table)
    payload = {
        "pareto_definition": "minimize split_half_nrmse and artifact_score; top-3 sorted by split_half_nrmse",
        "top3": top,
        "frontier_count": int(len(frontier)),
        "frontier": [
            {
                "lambda_tv": float(row.lambda_tv),
                "psf_sigma": float(row.psf_sigma),
                "n_input_frames": int(row.n_input_frames) if hasattr(row, "n_input_frames") else None,
                "input_frame_count": int(row.input_frame_count) if hasattr(row, "input_frame_count") else None,
                "split_half_nrmse": float(row.split_half_nrmse),
                "split_half_corr": float(getattr(row, "split_half_corr", float("nan"))),
                "artifact_score": float(row.artifact_score),
                "holdout_mse": float(row.holdout_mse),
                "raw_control_corr": float(row.raw_control_corr),
                "hr_cache_file": str(getattr(row, "hr_cache_file", "")),
            }
            for row in frontier.itertuples(index=False)
        ],
        "generated_at_unix": float(time.time()),
    }
    write_json(output_dir / "best_params.json", payload)


def save_heatmap(table: pd.DataFrame, output_dir: Path) -> None:
    if table.empty:
        return
    lambdas = sorted(table["lambda_tv"].dropna().unique())
    sigmas = sorted(table["psf_sigma"].dropna().unique())
    if not lambdas or not sigmas:
        return

    def grid(metric: str) -> np.ndarray:
        pivot = table.pivot_table(index="psf_sigma", columns="lambda_tv", values=metric, aggfunc="median")
        return pivot.reindex(index=sigmas, columns=lambdas).to_numpy(dtype=float)

    setup_academic_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["double_col"], constrained_layout=True)
    metrics = [
        ("split_half_nrmse", "Split-half NRMSE"),
        ("artifact_score", "Artifact score"),
    ]
    for ax, (metric, title) in zip(axes, metrics, strict=True):
        image = ax.imshow(grid(metric), cmap=COLORMAPS["coverage"], aspect="auto", origin="lower")
        ax.set_title(title)
        ax.set_xlabel("lambda_TV")
        ax.set_ylabel("PSF sigma [LR px]")
        ax.set_xticks(np.arange(len(lambdas)), [f"{value:g}" for value in lambdas], rotation=45, ha="right")
        ax.set_yticks(np.arange(len(sigmas)), [f"{value:g}" for value in sigmas])
        cbar = fig.colorbar(image, ax=ax, shrink=0.82)
        cbar.ax.set_ylabel(metric)
    savefig_academic(fig, output_dir / "sweep_heatmap.png")


def combine_detail_tables(table: pd.DataFrame, output_dir: Path) -> None:
    for column, out_name in [
        ("convergence_file", "convergence_details.csv"),
        ("split_half_detail_file", "split_half_details.csv"),
        ("holdout_detail_file", "holdout_details.csv"),
    ]:
        if column not in table.columns:
            continue
        frames: list[pd.DataFrame] = []
        for item in table[column].dropna().astype(str):
            if not item:
                continue
            path = Path(item)
            if not path.is_absolute():
                path = PROJECT_ROOT / path
            if path.exists():
                frames.append(pd.read_csv(path))
        if frames:
            pd.concat(frames, ignore_index=True).to_csv(output_dir / out_name, index=False)


__all__ = [
    "ParamSpec",
    "RESULT_COLUMNS",
    "_data_gradient_and_loss",
    "_data_gradient_and_loss_cached",
    "cache_hr_name",
    "combine_detail_tables",
    "evaluate_param",
    "existing_completed",
    "holdout_details_for_spec",
    "holdout_mse_for_spec",
    "init_worker",
    "pareto_frontier",
    "pareto_top3",
    "pearson_corr",
    "psnr",
    "reconstruct_for_spec",
    "reconstruct_for_spec_with_records",
    "save_best_params",
    "save_heatmap",
    "save_results_table",
    "token",
    "tv_denoise_chambolle",
    "tv_norm",
]
