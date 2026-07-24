"""Reproducible EP05 alignment tuning study.

The study wraps the existing EP05 contour-alignment and SR-capacity utilities
into one reproducible run:

1. sweep edge percentile, Chamfer-refine radius, and refine step;
2. score every candidate with a common held-out edge definition;
3. compare the selected tuned refinement against the default refinement, NCC
   initialisation, and filename affine fit;
4. write CSV/JSON summaries and paper-style figures.

Stage commands are not used as alignment truth.  Filename affine is only a
coordinate-label baseline fitted from the tuned data-driven shifts.

用法（项目根目录）::

    uv run python scripts/run_ep05_alignment_tuning_study.py \
        [--mode quick|full] [--quick-limit 96] [--n-jobs N] \
        [--edge-percentiles 91 93 95] [--refine-radii-px 0.5 1.0] \
        [--refine-steps-px 0.125 0.25] [--skip-figures]

输入依赖: output/ep01_data_processing/frame_audit.csv、data/data_raw/infrared_avi/
    （--frame-audit-csv / --data-dir 可覆盖）
输出: output/ep05_alignment_tuning_study/（--output-dir 可覆盖）CSV/JSON 汇总与图表

关联: EP05
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.run_ep05_alignment_sr_capacity_check import (  # noqa: E402
    center_roi,
    chamfer_score,
    deterministic_subsample,
    edge_mask_from_frame,
    gradient_magnitude,
    image_correlation_after_shift,
    phase_occupancy,
)
from thermal_core.ep05 import affine_shift, fit_filename_affine  # noqa: E402
from scripts.run_ep05_contour_alignment_validation import (  # noqa: E402
    AlignmentTask,
    choose_reference,
    default_workers,
    run_alignment_task,
    select_main_session,
)
from thermal_core.io import load_frame  # noqa: E402
from thermal_core.plotting import (  # noqa: E402
    COLORMAPS,
    FIGURE_SIZES,
    METHOD_COLOR_LIST,
    savefig_academic,
    setup_academic_style,
)


COMPARISON_METHOD_ORDER = [
    "tuned_refined",
    "default_refined",
    "ncc_init",
    "filename_affine_fit",
]

COMPARISON_METHOD_LABELS = {
    "tuned_refined": "Tuned refined",
    "default_refined": "Default refined",
    "ncc_init": "NCC init",
    "filename_affine_fit": "Filename affine",
}


@dataclass(frozen=True)
class Candidate:
    roi_size: int
    edge_percentile: float
    refine_radius_px: float
    refine_step_px: float

    @property
    def name(self) -> str:
        return (
            f"r{self.roi_size}"
            f"_e{_float_token(self.edge_percentile, 10)}"
            f"_rad{_float_token(self.refine_radius_px, 100)}"
            f"_s{_float_token(self.refine_step_px, 1000)}"
        )


def _float_token(value: float, scale: int) -> str:
    scaled = int(round(float(value) * scale))
    width = 2 if scale == 10 else 3
    return f"{scaled:0{width}d}".rstrip("0") if scale == 10 else f"{scaled:0{width}d}"


def _json_ready(obj):
    if isinstance(obj, dict):
        return {str(k): _json_ready(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_ready(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if not isinstance(obj, (str, bytes)):
        try:
            if pd.isna(obj):
                return None
        except (TypeError, ValueError):
            pass
    return obj


def _unique_candidates(candidates: Iterable[Candidate]) -> list[Candidate]:
    seen: set[Candidate] = set()
    out: list[Candidate] = []
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        out.append(candidate)
    return out


def build_candidates(args: argparse.Namespace) -> tuple[list[Candidate], Candidate]:
    candidates = [
        Candidate(
            roi_size=int(args.roi_size),
            edge_percentile=float(edge),
            refine_radius_px=float(radius),
            refine_step_px=float(step),
        )
        for edge, radius, step in itertools.product(
            args.edge_percentiles,
            args.refine_radii_px,
            args.refine_steps_px,
        )
    ]
    default_candidate = Candidate(
        roi_size=int(args.roi_size),
        edge_percentile=float(args.default_edge_percentile),
        refine_radius_px=float(args.default_refine_radius_px),
        refine_step_px=float(args.default_refine_step_px),
    )
    candidates.append(default_candidate)
    return _unique_candidates(candidates), default_candidate


def run_tasks(tasks: list[AlignmentTask], n_jobs: int, desc: str) -> list[dict]:
    if n_jobs <= 1:
        return [run_alignment_task(task) for task in tqdm(tasks, desc=desc)]

    records: list[dict] = []
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        futures = [executor.submit(run_alignment_task, task) for task in tasks]
        for future in tqdm(as_completed(futures), total=len(futures), desc=desc):
            records.append(future.result())
    return records


def run_candidate_alignment(
    candidate: Candidate,
    main_df: pd.DataFrame,
    ref: pd.Series,
    args: argparse.Namespace,
    output_dir: Path,
) -> pd.DataFrame:
    tasks = [
        AlignmentTask(
            frame_index=int(i),
            file=str(row["file"]),
            acquisition_order=int(row["acquisition_order"]),
            x_um=float(row["X"]),
            y_um=float(row["Y"]),
            r=int(row["R"]),
            data_dir=str(args.data_dir),
            reference_file=str(ref["file"]),
            roi_size=int(candidate.roi_size),
            search_radius=int(args.search_radius),
            edge_percentile=float(candidate.edge_percentile),
            refine_radius_px=float(candidate.refine_radius_px),
            refine_step_px=float(candidate.refine_step_px),
            max_edge_points=int(args.max_edge_points),
        )
        for i, row in main_df.iterrows()
    ]

    candidate_dir = output_dir / "candidates" / candidate.name
    candidate_dir.mkdir(parents=True, exist_ok=True)
    records = run_tasks(tasks, int(args.n_jobs), desc=f"align {candidate.name}")
    results = pd.DataFrame(records).sort_values("acquisition_order").reset_index(drop=True)
    results.insert(0, "candidate", candidate.name)
    for key, value in asdict(candidate).items():
        results.insert(1, key, value)
    results.to_csv(candidate_dir / "contour_alignment_results.csv", index=False)
    return results


def summarize_candidate_self(results: pd.DataFrame, candidate: Candidate) -> dict:
    valid = results[results["success"].astype(str).str.lower().eq("true")].copy()
    summary = {
        "candidate": candidate.name,
        "roi_size": int(candidate.roi_size),
        "edge_percentile": float(candidate.edge_percentile),
        "refine_radius_px": float(candidate.refine_radius_px),
        "refine_step_px": float(candidate.refine_step_px),
        "n_frames": int(len(results)),
        "n_success": int(len(valid)),
        "self_before_holdout_median_px": np.nan,
        "self_init_holdout_median_px": np.nan,
        "self_refined_holdout_median_px": np.nan,
        "self_refined_holdout_p90_px": np.nan,
        "self_refined_gain_vs_init_pct": np.nan,
        "self_refined_gain_vs_noalign_pct": np.nan,
        "self_gradient_corr_median": np.nan,
        "self_gradient_corr_gain_median": np.nan,
        "shift_norm_median_px": np.nan,
        "shift_span_dx_px": np.nan,
        "shift_span_dy_px": np.nan,
        "worse_than_init_fraction": np.nan,
    }
    if valid.empty:
        return summary

    before = pd.to_numeric(valid["before_holdout_chamfer_px"], errors="coerce")
    init = pd.to_numeric(valid["init_holdout_chamfer_px"], errors="coerce")
    refined = pd.to_numeric(valid["refined_holdout_chamfer_px"], errors="coerce")
    summary.update(
        {
            "self_before_holdout_median_px": float(before.median()),
            "self_init_holdout_median_px": float(init.median()),
            "self_refined_holdout_median_px": float(refined.median()),
            "self_refined_holdout_p90_px": float(refined.quantile(0.9)),
            "self_refined_gain_vs_init_pct": _relative_gain_pct(init.median(), refined.median()),
            "self_refined_gain_vs_noalign_pct": _relative_gain_pct(before.median(), refined.median()),
            "self_gradient_corr_median": float(pd.to_numeric(valid["gradient_corr_refined"], errors="coerce").median()),
            "self_gradient_corr_gain_median": float(pd.to_numeric(valid["gradient_corr_gain"], errors="coerce").median()),
            "shift_norm_median_px": float(pd.to_numeric(valid["refined_shift_norm_px"], errors="coerce").median()),
            "shift_span_dx_px": float(valid["refined_align_dx_px"].max() - valid["refined_align_dx_px"].min()),
            "shift_span_dy_px": float(valid["refined_align_dy_px"].max() - valid["refined_align_dy_px"].min()),
            "worse_than_init_fraction": float((refined > init).mean()),
        }
    )
    return summary


def _relative_gain_pct(baseline: float, candidate: float) -> float:
    if not np.isfinite(baseline) or baseline <= 0 or not np.isfinite(candidate):
        return np.nan
    return float(100.0 * (baseline - candidate) / baseline)


def score_refined_alignment(
    main_df: pd.DataFrame,
    alignment: pd.DataFrame,
    *,
    data_dir: Path,
    roi_size: int,
    eval_edge_percentile: float,
    max_edge_points: int,
) -> pd.DataFrame:
    valid_alignment = alignment[alignment["success"].astype(str).str.lower().eq("true")].copy()
    if valid_alignment.empty:
        return pd.DataFrame()

    ref_file = str(valid_alignment["reference_file"].dropna().iloc[0])
    ref_full = load_frame(data_dir / ref_file).astype(np.float32, copy=False)
    roi = center_roi(ref_full.shape, int(roi_size))
    ref = ref_full[roi]
    ref_edges = edge_mask_from_frame(ref, float(eval_edge_percentile))
    ref_distance = distance_transform_edt(~ref_edges)
    ref_grad = gradient_magnitude(ref)

    align_lookup = valid_alignment.set_index("file").to_dict("index")
    records: list[dict] = []
    for _, row in main_df.iterrows():
        file_name = str(row["file"])
        if file_name not in align_lookup:
            continue
        mov = load_frame(data_dir / file_name).astype(np.float32, copy=False)[roi]
        mov_edges = edge_mask_from_frame(mov, float(eval_edge_percentile))
        mov_coords = deterministic_subsample(np.argwhere(mov_edges), int(max_edge_points))
        if mov_coords.shape[0] < 40:
            continue
        holdout_coords = mov_coords[1::2]
        if holdout_coords.shape[0] < 20:
            holdout_coords = mov_coords[::2]
        mov_grad = gradient_magnitude(mov)
        shifts = align_lookup[file_name]
        dx = float(shifts["refined_align_dx_px"])
        dy = float(shifts["refined_align_dy_px"])
        records.append(
            {
                "file": file_name,
                "acquisition_order": int(row["acquisition_order"]),
                "X": float(row["X"]),
                "Y": float(row["Y"]),
                "R": int(row["R"]),
                "align_dx_px": dx,
                "align_dy_px": dy,
                "align_norm_px": float(np.hypot(dx, dy)),
                "holdout_chamfer_px": chamfer_score(holdout_coords, ref_distance, dx, dy),
                "gradient_corr": image_correlation_after_shift(ref_grad, mov_grad, dx, dy),
            }
        )
    return pd.DataFrame(records)


def summarize_eval_scores(scores: pd.DataFrame, candidate: Candidate) -> dict:
    summary = {
        "candidate": candidate.name,
        "eval_refined_holdout_median_px": np.nan,
        "eval_refined_holdout_p90_px": np.nan,
        "eval_refined_gradient_corr_median": np.nan,
        "eval_refined_gradient_corr_p10": np.nan,
    }
    if scores.empty:
        return summary
    summary.update(
        {
            "eval_refined_holdout_median_px": float(scores["holdout_chamfer_px"].median()),
            "eval_refined_holdout_p90_px": float(scores["holdout_chamfer_px"].quantile(0.9)),
            "eval_refined_gradient_corr_median": float(scores["gradient_corr"].median()),
            "eval_refined_gradient_corr_p10": float(scores["gradient_corr"].quantile(0.1)),
        }
    )
    return summary


def score_candidate_comparison(
    main_df: pd.DataFrame,
    tuned_alignment: pd.DataFrame,
    default_alignment: pd.DataFrame,
    *,
    data_dir: Path,
    roi_size: int,
    eval_edge_percentile: float,
    max_edge_points: int,
) -> pd.DataFrame:
    tuned_valid = tuned_alignment[tuned_alignment["success"].astype(str).str.lower().eq("true")].copy()
    default_valid = default_alignment[default_alignment["success"].astype(str).str.lower().eq("true")].copy()
    if tuned_valid.empty or default_valid.empty:
        return pd.DataFrame()

    ref_file = str(tuned_valid["reference_file"].dropna().iloc[0])
    ref_full = load_frame(data_dir / ref_file).astype(np.float32, copy=False)
    roi = center_roi(ref_full.shape, int(roi_size))
    ref = ref_full[roi]
    ref_edges = edge_mask_from_frame(ref, float(eval_edge_percentile))
    ref_distance = distance_transform_edt(~ref_edges)
    ref_grad = gradient_magnitude(ref)

    tuned_lookup = tuned_valid.set_index("file").to_dict("index")
    default_lookup = default_valid.set_index("file").to_dict("index")
    affine_fit = fit_filename_affine(tuned_valid, robust=True)
    beta_dx, beta_dy = affine_fit.beta_dx, affine_fit.beta_dy

    records: list[dict] = []
    for _, row in main_df.iterrows():
        file_name = str(row["file"])
        if file_name not in tuned_lookup or file_name not in default_lookup:
            continue
        mov = load_frame(data_dir / file_name).astype(np.float32, copy=False)[roi]
        mov_edges = edge_mask_from_frame(mov, float(eval_edge_percentile))
        mov_coords = deterministic_subsample(np.argwhere(mov_edges), int(max_edge_points))
        if mov_coords.shape[0] < 40:
            continue
        holdout_coords = mov_coords[1::2]
        if holdout_coords.shape[0] < 20:
            holdout_coords = mov_coords[::2]
        mov_grad = gradient_magnitude(mov)
        tuned = tuned_lookup[file_name]
        default = default_lookup[file_name]
        method_shifts = {
            "tuned_refined": (
                float(tuned["refined_align_dx_px"]),
                float(tuned["refined_align_dy_px"]),
            ),
            "default_refined": (
                float(default["refined_align_dx_px"]),
                float(default["refined_align_dy_px"]),
            ),
            "ncc_init": (
                float(tuned["init_align_dx_px"]),
                float(tuned["init_align_dy_px"]),
            ),
            "filename_affine_fit": affine_shift(row, beta_dx, beta_dy),
        }
        for method, (dx, dy) in method_shifts.items():
            records.append(
                {
                    "file": file_name,
                    "acquisition_order": int(row["acquisition_order"]),
                    "X": float(row["X"]),
                    "Y": float(row["Y"]),
                    "R": int(row["R"]),
                    "method": method,
                    "method_label": COMPARISON_METHOD_LABELS[method],
                    "align_dx_px": float(dx),
                    "align_dy_px": float(dy),
                    "align_norm_px": float(np.hypot(dx, dy)),
                    "holdout_chamfer_px": chamfer_score(holdout_coords, ref_distance, dx, dy),
                    "gradient_corr": image_correlation_after_shift(ref_grad, mov_grad, dx, dy),
                }
            )
    return pd.DataFrame(records)


def summarize_method_scores(scores: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for method in COMPARISON_METHOD_ORDER:
        group = scores[scores["method"].eq(method)]
        if group.empty:
            continue
        rows.append(
            {
                "method": method,
                "method_label": COMPARISON_METHOD_LABELS[method],
                "n_frames": int(group["file"].nunique()),
                "holdout_chamfer_median_px": float(group["holdout_chamfer_px"].median()),
                "holdout_chamfer_p90_px": float(group["holdout_chamfer_px"].quantile(0.9)),
                "gradient_corr_median": float(group["gradient_corr"].median()),
                "gradient_corr_p10": float(group["gradient_corr"].quantile(0.1)),
                "shift_norm_median_px": float(group["align_norm_px"].median()),
                "shift_norm_p90_px": float(group["align_norm_px"].quantile(0.9)),
            }
        )
    return pd.DataFrame(rows)


def build_phase_coverage(scores: pd.DataFrame, scales: tuple[int, ...] = (2, 3, 4)) -> pd.DataFrame:
    rows: list[dict] = []
    for method in COMPARISON_METHOD_ORDER:
        group = scores[scores["method"].eq(method)]
        if group.empty:
            continue
        dx = group["align_dx_px"].to_numpy(dtype=float)
        dy = group["align_dy_px"].to_numpy(dtype=float)
        for scale in scales:
            occ = phase_occupancy(dx, dy, scale)
            rows.append(
                {
                    "method": method,
                    "method_label": COMPARISON_METHOD_LABELS[method],
                    "scale": int(scale),
                    "n_frames": int(group["file"].nunique()),
                    "occupied_bins": int(occ["occupied_bins"]),
                    "bad_bins": int(occ["bad_bins"]),
                    "total_bins": int(occ["total_bins"]),
                    "coverage_fraction": float(occ["occupied_bins"] / occ["total_bins"]),
                    "min_count": int(occ["min_count"]),
                    "max_count": int(occ["max_count"]),
                    "entropy_fraction": float(occ["entropy_fraction"]),
                    "counts": json.dumps(occ["counts"]),
                }
            )
    return pd.DataFrame(rows)


def plot_tuning_heatmap(summary: pd.DataFrame, output_path: Path, metric_col: str) -> None:
    setup_academic_style()
    steps = sorted(summary["refine_step_px"].dropna().unique())
    radii = sorted(summary["refine_radius_px"].dropna().unique())
    edges = sorted(summary["edge_percentile"].dropna().unique())
    ncols = max(1, len(steps))
    width = min(FIGURE_SIZES["double_col"][0], max(3.5, 2.75 * ncols))
    fig, axes = plt.subplots(1, ncols, figsize=(width, 3.0), squeeze=False)
    values = pd.to_numeric(summary[metric_col], errors="coerce")
    vmin = float(values.min())
    vmax = float(values.max())
    best_value = float(values.min())

    for ax, step in zip(axes.ravel(), steps):
        sub = summary[np.isclose(summary["refine_step_px"], step)]
        matrix = np.full((len(edges), len(radii)), np.nan, dtype=float)
        for row_i, edge in enumerate(edges):
            for col_i, radius in enumerate(radii):
                cell = sub[np.isclose(sub["edge_percentile"], edge) & np.isclose(sub["refine_radius_px"], radius)]
                if not cell.empty:
                    matrix[row_i, col_i] = float(cell[metric_col].iloc[0])
        im = ax.imshow(matrix, cmap=COLORMAPS["coverage"], vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_title(f"refine step = {step:g} px")
        ax.set_xticks(np.arange(len(radii)))
        ax.set_xticklabels([f"{value:g}" for value in radii])
        ax.set_yticks(np.arange(len(edges)))
        ax.set_yticklabels([f"{value:g}" for value in edges])
        ax.set_xlabel("Refine radius [px]")
        ax.set_ylabel("Edge percentile")
        for row_i in range(len(edges)):
            for col_i in range(len(radii)):
                value = matrix[row_i, col_i]
                if not np.isfinite(value):
                    continue
                suffix = "\n*" if np.isclose(value, best_value) else ""
                ax.text(col_i, row_i, f"{value:.3f}{suffix}", ha="center", va="center", fontsize=7, color="white")

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.046, pad=0.04)
    cbar.set_label("Held-out Chamfer [px], lower is better")
    savefig_academic(fig, output_path)


def plot_candidate_comparison(
    method_summary: pd.DataFrame,
    phase_coverage: pd.DataFrame,
    output_path: Path,
) -> None:
    setup_academic_style()
    ordered = (
        method_summary.set_index("method")
        .loc[[method for method in COMPARISON_METHOD_ORDER if method in set(method_summary["method"])]]
        .reset_index()
    )
    labels = ordered["method_label"].tolist()
    y = np.arange(len(ordered))
    colors = METHOD_COLOR_LIST

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.55), gridspec_kw={"width_ratios": [1.15, 1.05, 1.0]})

    axes[0].barh(y, ordered["holdout_chamfer_median_px"], color=colors[0], alpha=0.88, label="median")
    axes[0].scatter(ordered["holdout_chamfer_p90_px"], y, color="#333333", s=18, label="P90", zorder=3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Held-out Chamfer [px]")
    axes[0].set_title("Contour Error")
    axes[0].legend(loc="lower right")
    axes[0].grid(axis="x", alpha=0.25)

    axes[1].barh(y, ordered["gradient_corr_median"], color=colors[1], alpha=0.88, label="median")
    axes[1].scatter(ordered["gradient_corr_p10"], y, color="#333333", s=18, label="P10", zorder=3)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Gradient correlation")
    axes[1].set_title("Gradient Agreement")
    axes[1].set_xlim(0.0, 1.0)
    axes[1].legend(loc="lower right")
    axes[1].grid(axis="x", alpha=0.25)

    scales = [2, 3, 4]
    matrix = np.full((len(ordered), len(scales)), np.nan, dtype=float)
    text = np.empty((len(ordered), len(scales)), dtype=object)
    for row_i, method in enumerate(ordered["method"]):
        for col_i, scale in enumerate(scales):
            cell = phase_coverage[phase_coverage["method"].eq(method) & phase_coverage["scale"].eq(scale)]
            if cell.empty:
                text[row_i, col_i] = ""
                continue
            record = cell.iloc[0]
            matrix[row_i, col_i] = float(record["coverage_fraction"])
            text[row_i, col_i] = f"{int(record['occupied_bins'])}/{int(record['total_bins'])}"

    im = axes[2].imshow(matrix, cmap=COLORMAPS["coverage"], vmin=0.0, vmax=1.0, aspect="auto")
    axes[2].set_xticks(np.arange(len(scales)))
    axes[2].set_xticklabels([f"{scale}x" for scale in scales])
    axes[2].set_yticks(y)
    axes[2].set_yticklabels([])
    axes[2].set_title("Phase Coverage")
    for row_i in range(matrix.shape[0]):
        for col_i in range(matrix.shape[1]):
            value = matrix[row_i, col_i]
            if not np.isfinite(value):
                continue
            color = "black" if value > 0.62 else "white"
            axes[2].text(col_i, row_i, text[row_i, col_i], ha="center", va="center", fontsize=7, color=color)
    cbar = fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    cbar.set_label("Occupied fraction")

    savefig_academic(fig, output_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-audit-csv", type=Path, default=ROOT / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data" / "data_raw" / "infrared_avi")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "output" / "ep05_alignment_tuning_study")
    parser.add_argument("--mode", choices=["quick", "full"], default="quick")
    parser.add_argument("--quick-limit", type=int, default=96)
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--roi-size", type=int, default=360)
    parser.add_argument("--search-radius", type=int, default=18)
    parser.add_argument("--edge-percentiles", nargs="+", type=float, default=[91.0, 93.0, 95.0])
    parser.add_argument("--refine-radii-px", nargs="+", type=float, default=[0.5, 1.0])
    parser.add_argument("--refine-steps-px", nargs="+", type=float, default=[0.125, 0.25])
    parser.add_argument("--eval-edge-percentile", type=float, default=93.0)
    parser.add_argument("--default-edge-percentile", type=float, default=93.0)
    parser.add_argument("--default-refine-radius-px", type=float, default=1.0)
    parser.add_argument("--default-refine-step-px", type=float, default=0.25)
    parser.add_argument("--max-edge-points", type=int, default=8000)
    parser.add_argument("--n-jobs", type=int, default=default_workers())
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    start = time.time()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_academic_style()

    audit = pd.read_csv(args.frame_audit_csv)
    main_full = select_main_session(audit)
    ref = choose_reference(main_full)
    limit_frames = args.limit_frames
    if limit_frames is None and args.mode == "quick":
        limit_frames = int(args.quick_limit)
    main_df = main_full.iloc[:limit_frames].copy() if limit_frames is not None else main_full.copy()
    main_df = main_df.reset_index(drop=True)

    candidates, default_candidate = build_candidates(args)
    print("EP05 alignment tuning study")
    print(f"mode={args.mode}, frames={len(main_df)}/{len(main_full)}, candidates={len(candidates)}, n_jobs={args.n_jobs}")
    print(f"reference={ref['file']} (acq={int(ref['acquisition_order'])}), eval edge={args.eval_edge_percentile:g}")
    print(f"output={args.output_dir}")

    all_results: list[pd.DataFrame] = []
    self_summaries: list[dict] = []
    eval_summaries: list[dict] = []
    candidate_results: dict[str, pd.DataFrame] = {}
    candidate_eval_scores: dict[str, pd.DataFrame] = {}

    for candidate in candidates:
        results = run_candidate_alignment(candidate, main_df, ref, args, args.output_dir)
        candidate_results[candidate.name] = results
        all_results.append(results)
        self_summary = summarize_candidate_self(results, candidate)
        self_summaries.append(self_summary)
        eval_scores = score_refined_alignment(
            main_df,
            results,
            data_dir=args.data_dir,
            roi_size=int(args.roi_size),
            eval_edge_percentile=float(args.eval_edge_percentile),
            max_edge_points=int(args.max_edge_points),
        )
        candidate_eval_scores[candidate.name] = eval_scores
        eval_summary = summarize_eval_scores(eval_scores, candidate)
        eval_summaries.append(eval_summary)
        candidate_summary = {**self_summary, **eval_summary}
        with open(args.output_dir / "candidates" / candidate.name / "contour_alignment_summary.json", "w", encoding="utf-8") as f:
            json.dump(_json_ready(candidate_summary), f, ensure_ascii=False, indent=2)

    frame_results = pd.concat(all_results, ignore_index=True)
    frame_results.to_csv(args.output_dir / "tuning_frame_results.csv", index=False)
    self_df = pd.DataFrame(self_summaries)
    eval_df = pd.DataFrame(eval_summaries)
    tuning_summary = self_df.merge(eval_df, on="candidate", how="left")
    tuning_summary["is_default_candidate"] = tuning_summary["candidate"].eq(default_candidate.name)
    tuning_summary = tuning_summary.sort_values("eval_refined_holdout_median_px").reset_index(drop=True)
    tuning_summary.to_csv(args.output_dir / "tuning_summary.csv", index=False)

    best_row = tuning_summary.iloc[0]
    best_candidate = next(candidate for candidate in candidates if candidate.name == str(best_row["candidate"]))
    tuned_alignment = candidate_results[best_candidate.name]
    default_alignment = candidate_results[default_candidate.name]
    comparison_scores = score_candidate_comparison(
        main_df,
        tuned_alignment,
        default_alignment,
        data_dir=args.data_dir,
        roi_size=int(args.roi_size),
        eval_edge_percentile=float(args.eval_edge_percentile),
        max_edge_points=int(args.max_edge_points),
    )
    comparison_scores.to_csv(args.output_dir / "candidate_comparison_scores.csv", index=False)
    method_summary = summarize_method_scores(comparison_scores)
    method_summary.to_csv(args.output_dir / "candidate_comparison_summary.csv", index=False)
    phase_coverage = build_phase_coverage(comparison_scores)
    phase_coverage.to_csv(args.output_dir / "candidate_phase_coverage.csv", index=False)

    if not args.skip_figures:
        plot_tuning_heatmap(
            tuning_summary,
            args.output_dir / "tuning_heatmap_heldout_chamfer.png",
            metric_col="eval_refined_holdout_median_px",
        )
        plot_candidate_comparison(
            method_summary,
            phase_coverage,
            args.output_dir / "candidate_alignment_comparison.png",
        )

    tuned = method_summary[method_summary["method"].eq("tuned_refined")]
    default = method_summary[method_summary["method"].eq("default_refined")]
    ncc = method_summary[method_summary["method"].eq("ncc_init")]
    summary_json = {
        "mode": args.mode,
        "limit_frames": int(limit_frames) if limit_frames is not None else None,
        "n_main_frames_available": int(len(main_full)),
        "n_frames_scored": int(len(main_df)),
        "n_candidates": int(len(candidates)),
        "reference_file": str(ref["file"]),
        "reference_acquisition_order": int(ref["acquisition_order"]),
        "roi_size": int(args.roi_size),
        "search_radius": int(args.search_radius),
        "eval_edge_percentile": float(args.eval_edge_percentile),
        "default_candidate": asdict(default_candidate) | {"name": default_candidate.name},
        "best_candidate": asdict(best_candidate) | {"name": best_candidate.name},
        "selection_metric": "eval_refined_holdout_median_px",
        "tuning_summary_top": tuning_summary.head(10).to_dict(orient="records"),
        "candidate_comparison_summary": method_summary.to_dict(orient="records"),
        "candidate_phase_coverage": phase_coverage.to_dict(orient="records"),
        "tuned_vs_default_holdout_gain_pct": _relative_gain_pct(
            float(default["holdout_chamfer_median_px"].iloc[0]) if not default.empty else np.nan,
            float(tuned["holdout_chamfer_median_px"].iloc[0]) if not tuned.empty else np.nan,
        ),
        "tuned_vs_ncc_holdout_gain_pct": _relative_gain_pct(
            float(ncc["holdout_chamfer_median_px"].iloc[0]) if not ncc.empty else np.nan,
            float(tuned["holdout_chamfer_median_px"].iloc[0]) if not tuned.empty else np.nan,
        ),
        "runtime_sec": float(time.time() - start),
        "outputs": {
            "tuning_frame_results_csv": str(args.output_dir / "tuning_frame_results.csv"),
            "tuning_summary_csv": str(args.output_dir / "tuning_summary.csv"),
            "candidate_comparison_scores_csv": str(args.output_dir / "candidate_comparison_scores.csv"),
            "candidate_comparison_summary_csv": str(args.output_dir / "candidate_comparison_summary.csv"),
            "candidate_phase_coverage_csv": str(args.output_dir / "candidate_phase_coverage.csv"),
            "tuning_heatmap_png": str(args.output_dir / "tuning_heatmap_heldout_chamfer.png"),
            "candidate_comparison_png": str(args.output_dir / "candidate_alignment_comparison.png"),
        },
    }
    with open(args.output_dir / "tuning_study_summary.json", "w", encoding="utf-8") as f:
        json.dump(_json_ready(summary_json), f, ensure_ascii=False, indent=2)

    print("\nTop tuning candidates")
    print(
        tuning_summary[
            [
                "candidate",
                "edge_percentile",
                "refine_radius_px",
                "refine_step_px",
                "eval_refined_holdout_median_px",
                "eval_refined_holdout_p90_px",
                "eval_refined_gradient_corr_median",
            ]
        ]
        .head(8)
        .to_string(index=False)
    )
    print("\nCandidate comparison")
    print(method_summary.to_string(index=False))
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
