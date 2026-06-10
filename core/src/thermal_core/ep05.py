"""EP05 notebook helpers for SR capacity and alignment baseline outputs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_ORDER = [
    "no_alignment",
    "old_stage_model",
    "filename_affine_fit",
    "data_driven_ncc_init",
    "data_driven_contour_refined",
]

METHOD_LABELS = {
    "no_alignment": "No alignment",
    "old_stage_model": "Stage prior",
    "filename_affine_fit": "Filename affine",
    "data_driven_ncc_init": "NCC init",
    "data_driven_contour_refined": "Contour refined",
    "data_driven_contour": "Data-driven contour",
}

GROUP_ORDER = ["all_r0", "scanline_y10", "scanline_y20", "column_x10", "column_x20"]

GROUP_LABELS = {
    "all_r0": "All R=0",
    "scanline_y10": "Scanline Y=10",
    "scanline_y20": "Scanline Y=20",
    "column_x10": "Column X=10",
    "column_x20": "Column X=20",
}


@dataclass(frozen=True)
class FilenameAffineFit:
    """Robust filename-coordinate affine fit for EP05 alignment shifts."""

    beta_dx: np.ndarray
    beta_dy: np.ndarray
    baseline_beta_dx: np.ndarray
    baseline_beta_dy: np.ndarray
    excluded_files: tuple[str, ...]
    median_residual_px: float
    outlier_threshold_px: float
    fit_count: int
    clean_count: int


def _boolish(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def _filename_affine_fit_rows(alignment: pd.DataFrame, repeat: int = 0) -> pd.DataFrame:
    required = ["success", "R", "X", "Y", "refined_align_dx_px", "refined_align_dy_px"]
    missing = [col for col in required if col not in alignment]
    if missing:
        raise ValueError(f"Alignment table is missing required columns: {missing}")

    rows = alignment[_boolish(alignment["success"]) & alignment["R"].eq(repeat)].copy()
    numeric_cols = ["X", "Y", "refined_align_dx_px", "refined_align_dy_px"]
    for col in numeric_cols:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    finite = np.isfinite(rows[numeric_cols].to_numpy(dtype=float)).all(axis=1)
    return rows.loc[finite].copy()


def affine_design(frame_rows: pd.DataFrame) -> np.ndarray:
    """Return the [1, X, Y] design matrix used by filename affine fits."""

    return np.column_stack(
        [
            np.ones(len(frame_rows), dtype=float),
            pd.to_numeric(frame_rows["X"], errors="coerce").to_numpy(dtype=float),
            pd.to_numeric(frame_rows["Y"], errors="coerce").to_numpy(dtype=float),
        ]
    )


def affine_predict(frame_rows: pd.DataFrame, beta_dx: np.ndarray, beta_dy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Predict dx/dy shifts from frame X/Y columns and affine coefficients."""

    design = affine_design(frame_rows)
    return design @ np.asarray(beta_dx, dtype=float), design @ np.asarray(beta_dy, dtype=float)


def affine_shift(row: pd.Series, beta_dx: np.ndarray, beta_dy: np.ndarray) -> tuple[float, float]:
    """Predict one filename-affine shift for a metadata row."""

    coord = np.array([1.0, float(row["X"]), float(row["Y"])], dtype=float)
    return float(coord @ np.asarray(beta_dx, dtype=float)), float(coord @ np.asarray(beta_dy, dtype=float))


def fit_filename_affine(
    alignment: pd.DataFrame,
    *,
    robust: bool = True,
    outlier_threshold: float = 3.0,
    repeat: int = 0,
) -> FilenameAffineFit:
    """Fit X/Y filename coordinates to refined contour-alignment shifts.

    The baseline fit uses all successful ``R=repeat`` frames.  Robust mode
    removes frames whose baseline residual norm exceeds
    ``outlier_threshold * median_residual`` and refits the same three-parameter
    model.  The returned baseline coefficients are always the all-frame OLS
    result, which keeps diagnosis and coefficient-delta reporting reproducible.
    """

    valid = _filename_affine_fit_rows(alignment, repeat=repeat)
    if len(valid) < 3:
        raise ValueError(f"Need at least 3 successful R={repeat} rows for affine fit; got {len(valid)}")

    design = affine_design(valid)
    if np.linalg.matrix_rank(design) < 3:
        raise ValueError("Filename affine design matrix is rank deficient")

    target_dx = valid["refined_align_dx_px"].to_numpy(dtype=float)
    target_dy = valid["refined_align_dy_px"].to_numpy(dtype=float)
    baseline_beta_dx = np.linalg.lstsq(design, target_dx, rcond=None)[0]
    baseline_beta_dy = np.linalg.lstsq(design, target_dy, rcond=None)[0]

    res_dx = target_dx - design @ baseline_beta_dx
    res_dy = target_dy - design @ baseline_beta_dy
    res_norm = np.hypot(res_dx, res_dy)
    median_residual = float(np.median(res_norm))
    threshold_px = float(outlier_threshold * median_residual)

    excluded_mask = np.zeros(len(valid), dtype=bool)
    clean_mask = np.ones(len(valid), dtype=bool)
    beta_dx = baseline_beta_dx
    beta_dy = baseline_beta_dy
    if robust and np.isfinite(threshold_px) and threshold_px > 0.0:
        excluded_mask = res_norm > threshold_px
        clean_mask = ~excluded_mask
        if int(clean_mask.sum()) >= 3 and np.linalg.matrix_rank(design[clean_mask]) >= 3:
            beta_dx = np.linalg.lstsq(design[clean_mask], target_dx[clean_mask], rcond=None)[0]
            beta_dy = np.linalg.lstsq(design[clean_mask], target_dy[clean_mask], rcond=None)[0]
        else:
            excluded_mask = np.zeros(len(valid), dtype=bool)
            clean_mask = np.ones(len(valid), dtype=bool)

    excluded_files = tuple(valid.loc[excluded_mask, "file"].astype(str).tolist()) if "file" in valid else tuple()
    return FilenameAffineFit(
        beta_dx=np.asarray(beta_dx, dtype=float),
        beta_dy=np.asarray(beta_dy, dtype=float),
        baseline_beta_dx=np.asarray(baseline_beta_dx, dtype=float),
        baseline_beta_dy=np.asarray(baseline_beta_dy, dtype=float),
        excluded_files=excluded_files,
        median_residual_px=median_residual,
        outlier_threshold_px=threshold_px,
        fit_count=int(len(valid)),
        clean_count=int(clean_mask.sum()),
    )


def _percentile_rank(values: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(values, errors="coerce")
    ranks = numeric.rank(pct=True, method="average").to_numpy(dtype=float)
    return np.where(np.isfinite(ranks), ranks, 0.5)


def filename_affine_diagnostics(
    alignment: pd.DataFrame,
    fit: FilenameAffineFit | None = None,
    *,
    robust: bool = True,
    outlier_threshold: float = 3.0,
    repeat: int = 0,
) -> pd.DataFrame:
    """Return per-frame baseline residuals and multi-metric outlier scores."""

    affine_fit = fit or fit_filename_affine(
        alignment,
        robust=robust,
        outlier_threshold=outlier_threshold,
        repeat=repeat,
    )
    required = ["success", "X", "Y", "refined_align_dx_px", "refined_align_dy_px"]
    missing = [col for col in required if col not in alignment]
    if missing:
        raise ValueError(f"Alignment table is missing required columns: {missing}")

    rows = alignment[_boolish(alignment["success"])].copy()
    numeric_cols = ["X", "Y", "refined_align_dx_px", "refined_align_dy_px"]
    for col in numeric_cols:
        rows[col] = pd.to_numeric(rows[col], errors="coerce")
    finite = np.isfinite(rows[numeric_cols].to_numpy(dtype=float)).all(axis=1)
    rows = rows.loc[finite].copy()

    pred_dx, pred_dy = affine_predict(rows, affine_fit.baseline_beta_dx, affine_fit.baseline_beta_dy)
    robust_pred_dx, robust_pred_dy = affine_predict(rows, affine_fit.beta_dx, affine_fit.beta_dy)
    rows["pred_dx_px"] = pred_dx
    rows["pred_dy_px"] = pred_dy
    rows["res_dx_px"] = rows["refined_align_dx_px"].to_numpy(dtype=float) - pred_dx
    rows["res_dy_px"] = rows["refined_align_dy_px"].to_numpy(dtype=float) - pred_dy
    rows["res_norm_px"] = np.hypot(rows["res_dx_px"].to_numpy(dtype=float), rows["res_dy_px"].to_numpy(dtype=float))
    rows["robust_pred_dx_px"] = robust_pred_dx
    rows["robust_pred_dy_px"] = robust_pred_dy
    rows["robust_res_dx_px"] = rows["refined_align_dx_px"].to_numpy(dtype=float) - robust_pred_dx
    rows["robust_res_dy_px"] = rows["refined_align_dy_px"].to_numpy(dtype=float) - robust_pred_dy
    rows["robust_res_norm_px"] = np.hypot(
        rows["robust_res_dx_px"].to_numpy(dtype=float),
        rows["robust_res_dy_px"].to_numpy(dtype=float),
    )

    if "refined_shift_norm_px" not in rows:
        rows["refined_shift_norm_px"] = np.hypot(
            rows["refined_align_dx_px"].to_numpy(dtype=float),
            rows["refined_align_dy_px"].to_numpy(dtype=float),
        )
    if "refined_holdout_chamfer_px" not in rows:
        rows["refined_holdout_chamfer_px"] = np.nan
    if "gradient_corr_refined" not in rows:
        rows["gradient_corr_refined"] = np.nan

    rows["res_norm_rank"] = _percentile_rank(rows["res_norm_px"])
    rows["holdout_chamfer_rank"] = _percentile_rank(rows["refined_holdout_chamfer_px"])
    rows["gradient_corr_low_rank"] = _percentile_rank(-pd.to_numeric(rows["gradient_corr_refined"], errors="coerce"))
    rows["refined_shift_norm_rank"] = _percentile_rank(rows["refined_shift_norm_px"])
    rows["outlier_score"] = (
        0.45 * rows["res_norm_rank"]
        + 0.25 * rows["holdout_chamfer_rank"]
        + 0.20 * rows["gradient_corr_low_rank"]
        + 0.10 * rows["refined_shift_norm_rank"]
    )

    excluded = set(affine_fit.excluded_files)
    rows["used_for_affine_fit"] = rows["R"].eq(repeat) if "R" in rows else False
    rows["excluded_from_robust_fit"] = rows["file"].astype(str).isin(excluded) if "file" in rows else False
    rows["residual_gate_outlier"] = rows["res_norm_px"] > affine_fit.outlier_threshold_px
    rows["recommended_exclude_from_fit"] = rows["used_for_affine_fit"] & rows["residual_gate_outlier"]
    rows["recommended_exclude_from_affine_application"] = rows["residual_gate_outlier"]
    rows["suspicious_all_frames"] = rows["residual_gate_outlier"] | (rows["outlier_score"] >= 0.85)
    return rows


def load_capacity_outputs(output_dir: Path) -> dict:
    """Load all EP05 capacity-check outputs used by the notebook."""
    output_dir = Path(output_dir)
    with open(output_dir / "alignment_sr_capacity_summary.json", encoding="utf-8") as f:
        summary_json = json.load(f)
    return {
        "summary_json": summary_json,
        "method_summary": pd.read_csv(output_dir / "alignment_method_summary.csv"),
        "holdout_scores": pd.read_csv(output_dir / "alignment_method_holdout_scores.csv"),
        "phase_summary_2x": pd.read_csv(output_dir / "phase_bin_summary_2x.csv"),
        "phase_counts_2x": pd.read_csv(output_dir / "phase_bin_counts_2x.csv"),
        "overlay_density": pd.read_csv(output_dir / "alignment_overlay_density_metrics.csv"),
    }


def load_displacement_outputs(output_dir: Path) -> dict:
    """Load EP05 displacement-reassessment outputs."""
    output_dir = Path(output_dir)
    with open(output_dir / "displacement_reassessment_summary.json", encoding="utf-8") as f:
        summary_json = json.load(f)
    return {
        "summary_json": summary_json,
        "measurements": pd.read_csv(output_dir / "displacement_measurements.csv"),
        "summary_by_class": pd.read_csv(output_dir / "displacement_summary_by_class.csv"),
        "trajectory": pd.read_csv(output_dir / "main_session_cumulative_trajectory.csv"),
    }


def load_contour_alignment_outputs(output_dir: Path) -> dict:
    """Load data-driven contour-alignment validation outputs."""
    output_dir = Path(output_dir)
    with open(output_dir / "contour_alignment_summary.json", encoding="utf-8") as f:
        summary_json = json.load(f)
    return {
        "summary_json": summary_json,
        "results": pd.read_csv(output_dir / "contour_alignment_results.csv"),
    }


def load_overlay_alignment_outputs(output_dir: Path) -> dict:
    """Load TXT/BMP overlay alignment sanity-check outputs."""
    output_dir = Path(output_dir)
    return {
        "summary": pd.read_csv(output_dir / "overlay_alignment_summary.csv"),
    }


def _method_order_map() -> dict[str, int]:
    return {name: i for i, name in enumerate(METHOD_ORDER)}


def _with_method_labels(df: pd.DataFrame, method_col: str = "method") -> pd.DataFrame:
    out = df.copy()
    out["method_label"] = out[method_col].map(METHOD_LABELS).fillna(out[method_col])
    out["method_order"] = out[method_col].map(_method_order_map()).fillna(999).astype(int)
    return out


def _quantile(s: pd.Series, q: float) -> float:
    return float(pd.to_numeric(s, errors="coerce").dropna().quantile(q))


def _phase_counts_from_shift(df: pd.DataFrame, scale: int) -> np.ndarray:
    frac_x = np.mod(df["align_dx_px"].to_numpy(dtype=float), 1.0)
    frac_y = np.mod(df["align_dy_px"].to_numpy(dtype=float), 1.0)
    bin_x = np.clip(np.floor(frac_x * scale).astype(int), 0, scale - 1)
    bin_y = np.clip(np.floor(frac_y * scale).astype(int), 0, scale - 1)
    counts = np.zeros((scale, scale), dtype=int)
    np.add.at(counts, (bin_y, bin_x), 1)
    return counts


def _occupancy_from_counts(counts: np.ndarray) -> dict:
    flat = counts.ravel()
    n = int(flat.sum())
    occupied = int((flat > 0).sum())
    probs = flat[flat > 0] / n if n else np.array([], dtype=float)
    entropy = float(-(probs * np.log(probs)).sum() / np.log(flat.size)) if len(probs) else 0.0
    return {
        "n_frames": n,
        "occupied_bins": occupied,
        "bad_bins": int(flat.size - occupied),
        "total_bins": int(flat.size),
        "min_count": int(flat.min()) if len(flat) else 0,
        "max_count": int(flat.max()) if len(flat) else 0,
        "entropy_fraction": entropy,
        "expected_count": float(n / flat.size) if len(flat) else 0.0,
    }


def _phase_risk_note(method: str, scale: int, bad_bins: int) -> str:
    if scale == 2 and bad_bins == 0:
        return "2x phase capacity OK; alignment quality still needs separate checks"
    if scale >= 3 and method == "data_driven_contour_refined" and bad_bins > 0:
        return "High-magnification phase collapse after local contour refinement"
    if scale >= 3 and bad_bins == 0:
        return "Occupancy only; this is not evidence that 4x SR is feasible"
    return "Incomplete phase coverage; use only as a risk diagnostic"


def trajectory_capacity_table(summary_json: dict) -> pd.DataFrame:
    """Return main-session cumulative trajectory diagnostics."""
    traj = summary_json["trajectory"]
    rows = [
        {
            "diagnostic": "Main-session frames",
            "value": f"{traj['n_frames']}",
            "interpretation": "Default EP06 input scope; cross-session frames remain excluded",
        },
        {
            "diagnostic": "Coordinate coverage",
            "value": f"{traj['x_coordinate_count']} x {traj['y_coordinate_count']} coordinates",
            "interpretation": "Raster path covers a two-dimensional stage grid, not one isolated edge pair",
        },
        {
            "diagnostic": "Cumulative span",
            "value": f"{traj['span_dx_px']:.4f} x {traj['span_dy_px']:.4f} px",
            "interpretation": "Data-visible path span available for alignment and phase diagnostics",
        },
        {
            "diagnostic": "Span norm",
            "value": f"{traj['span_norm_px']:.4f} px",
            "interpretation": "Overall visible displacement envelope, not a stage-command truth label",
        },
        {
            "diagnostic": "Cumulative path length",
            "value": f"{traj['path_length_px']:.4f} px",
            "interpretation": "Total visible motion across raster acquisition order",
        },
    ]
    return pd.DataFrame(rows)


def visible_shift_key_table(measurements: pd.DataFrame, summary_by_class: pd.DataFrame) -> pd.DataFrame:
    """Return compact visible-shift diagnostics for the displacement preface."""
    rows: list[dict] = []
    main_x = measurements[
        (measurements["pair_set"] == "main_acq_adjacent")
        & (measurements["method"] == "highpass")
        & (measurements["pair_class"] == "same_y_x_motion")
    ].copy()
    for delta_um in [2.0, 4.0]:
        part = main_x[np.isclose(main_x["delta_X_um"], delta_um)]
        if part.empty:
            continue
        rows.append(
            {
                "diagnostic": f"Main X-adjacent {delta_um:g} um",
                "n_pairs": int(len(part)),
                "command_norm_px": _quantile(part["model_norm_px"], 0.5),
                "visible_norm_median_px": _quantile(part["measured_norm_px"], 0.5),
                "visible_norm_p90_px": _quantile(part["measured_norm_px"], 0.9),
                "projection_median_px": _quantile(part["measured_abs_projection_on_model_px"], 0.5),
                "peak_ncc_median": _quantile(part["peak_ncc"], 0.5),
                "use": "Short-time X-step visibility; useful for alignment initialization only",
            }
        )

    endpoint_specs = [
        ("same_y_scanline_endpoint", "same_y_x_motion", "R=0 X-scanline endpoint 40 um"),
        ("same_x_column_endpoint", "same_x_y_motion", "R=0 Y-column endpoint 40 um"),
        ("main_acq_adjacent", "raster_row_reset", "Main-session row reset"),
    ]
    for pair_set, pair_class, label in endpoint_specs:
        part = summary_by_class[
            (summary_by_class["pair_set"] == pair_set)
            & (summary_by_class["method"] == "highpass")
            & (summary_by_class["pair_class"] == pair_class)
        ]
        if part.empty:
            continue
        row = part.iloc[0]
        rows.append(
            {
                "diagnostic": label,
                "n_pairs": int(row["n"]),
                "command_norm_px": float(row["model_norm_median_px"]),
                "visible_norm_median_px": float(row["measured_norm_median_px"]),
                "visible_norm_p90_px": float(row["measured_norm_p90_px"]),
                "projection_median_px": float(row["abs_projection_median_px"]),
                "peak_ncc_median": float(row["peak_ncc_median"]),
                "use": "Coarse trajectory/coverage diagnostic; not an alignment ground truth",
            }
        )
    return pd.DataFrame(rows)


def ordered_method_table(method_summary: pd.DataFrame) -> pd.DataFrame:
    """Return the alignment comparison table in decision-order, not winner-order."""
    df = _with_method_labels(method_summary)
    cols = [
        "method_label",
        "n_frames",
        "holdout_chamfer_median_px",
        "holdout_chamfer_p90_px",
        "gradient_corr_median",
        "gradient_corr_p10",
        "shift_norm_median_px",
        "shift_norm_p90_px",
    ]
    return df.sort_values("method_order")[cols].reset_index(drop=True)


def phase_capacity_table(phase_summary: pd.DataFrame) -> pd.DataFrame:
    """Return the 2x phase-capacity table with readable method labels."""
    df = _with_method_labels(phase_summary)
    cols = [
        "method_label",
        "n_frames",
        "occupied_bins",
        "bad_bins",
        "total_bins",
        "entropy_fraction",
        "min_count",
        "max_count",
    ]
    return df.sort_values("method_order")[cols].reset_index(drop=True)


def multi_scale_phase_coverage_table(
    summary_json: dict,
    holdout_scores: pd.DataFrame | None = None,
    scales: tuple[int, ...] = (2, 3, 4),
) -> pd.DataFrame:
    """Return 2x/3x/4x phase occupancy as a risk diagnostic."""
    rows: list[dict] = []
    phase_capacity = summary_json.get("phase_capacity", {})
    for method in METHOD_ORDER:
        entries = {int(item["scale"]): item for item in phase_capacity.get(method, {}).get("phase_occupancy", [])}
        for scale in scales:
            item = entries.get(scale)
            if item is None:
                if holdout_scores is None:
                    continue
                method_scores = holdout_scores[holdout_scores["method"] == method]
                if method_scores.empty:
                    continue
                item = _occupancy_from_counts(_phase_counts_from_shift(method_scores, scale))
            rows.append(
                {
                    "method_label": METHOD_LABELS.get(method, method),
                    "scale": scale,
                    "n_frames": int(item.get("n_frames", summary_json.get("n_main_frames_scored", 0))),
                    "occupied_bins": int(item["occupied_bins"]),
                    "bad_bins": int(item["bad_bins"]),
                    "total_bins": int(item["total_bins"]),
                    "min_count": int(item["min_count"]),
                    "max_count": int(item["max_count"]),
                    "entropy_fraction": float(item["entropy_fraction"]),
                    "risk_note": _phase_risk_note(method, scale, int(item["bad_bins"])),
                    "method_order": _method_order_map().get(method, 999),
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["method_order", "scale"])
        .drop(columns=["method_order"])
        .reset_index(drop=True)
    )


def fractional_phase_distribution_table(holdout_scores: pd.DataFrame, scale: int = 4) -> pd.DataFrame:
    """Summarize fractional phase distribution using a count matrix table."""
    rows: list[dict] = []
    for method in METHOD_ORDER:
        df = holdout_scores[holdout_scores["method"] == method]
        if df.empty:
            continue
        counts = _phase_counts_from_shift(df, scale)
        occupancy = _occupancy_from_counts(counts)
        frac_x = np.mod(df["align_dx_px"].to_numpy(dtype=float), 1.0)
        frac_y = np.mod(df["align_dy_px"].to_numpy(dtype=float), 1.0)
        rows.append(
            {
                "method_label": METHOD_LABELS.get(method, method),
                "scale": scale,
                "occupied_bins": occupancy["occupied_bins"],
                "bad_bins": occupancy["bad_bins"],
                "frac_x_p10": float(np.quantile(frac_x, 0.1)),
                "frac_x_median": float(np.quantile(frac_x, 0.5)),
                "frac_x_p90": float(np.quantile(frac_x, 0.9)),
                "frac_y_p10": float(np.quantile(frac_y, 0.1)),
                "frac_y_median": float(np.quantile(frac_y, 0.5)),
                "frac_y_p90": float(np.quantile(frac_y, 0.9)),
                "phase_yx_count_matrix": " / ".join(
                    " ".join(f"{int(value):3d}" for value in row) for row in counts
                ),
                "method_order": _method_order_map().get(method, 999),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("method_order")
        .drop(columns=["method_order"])
        .reset_index(drop=True)
    )


def contour_alignment_tail_table(results: pd.DataFrame) -> pd.DataFrame:
    """Return absolute held-out Chamfer tail statistics for contour alignment."""
    specs = [
        ("before_holdout_chamfer_px", "No alignment"),
        ("init_holdout_chamfer_px", "NCC init"),
        ("refined_holdout_chamfer_px", "Contour refined"),
    ]
    rows = []
    for col, label in specs:
        values = pd.to_numeric(results[col], errors="coerce").dropna()
        rows.append(
            {
                "alignment_state": label,
                "n_frames": int(values.size),
                "chamfer_median_px": float(values.median()),
                "chamfer_p90_px": float(values.quantile(0.9)),
                "chamfer_max_px": float(values.max()),
                "frames_le_0p2_px": int((values <= 0.2).sum()),
            }
        )
    return pd.DataFrame(rows)


def worst_contour_frames_table(results: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Return worst refined-Chamfer frames for tail-risk inspection."""
    cols = [
        "file",
        "acquisition_order",
        "X",
        "Y",
        "R",
        "before_holdout_chamfer_px",
        "init_holdout_chamfer_px",
        "refined_holdout_chamfer_px",
        "ncc_peak",
        "gradient_corr_refined",
        "refined_shift_norm_px",
    ]
    return (
        results.sort_values("refined_holdout_chamfer_px", ascending=False)[cols]
        .head(n)
        .reset_index(drop=True)
    )


def data_driven_correction_table(holdout_scores: pd.DataFrame) -> pd.DataFrame:
    """Compare paired shift corrections between alignment methods."""
    index_cols = ["file", "acquisition_order", "X", "Y", "R"]
    wide = holdout_scores.pivot_table(
        index=index_cols,
        columns="method",
        values=["align_dx_px", "align_dy_px", "holdout_chamfer_px", "gradient_corr"],
        aggfunc="first",
    )
    wide.columns = ["__".join(col) for col in wide.columns]
    wide = wide.reset_index()
    comparisons = [
        ("NCC init - filename affine", "data_driven_ncc_init", "filename_affine_fit"),
        ("Contour refined - filename affine", "data_driven_contour_refined", "filename_affine_fit"),
        ("Contour refined - NCC init", "data_driven_contour_refined", "data_driven_ncc_init"),
        ("Filename affine - stage prior", "filename_affine_fit", "old_stage_model"),
    ]
    rows = []
    for label, method_a, method_b in comparisons:
        needed = [
            f"align_dx_px__{method_a}",
            f"align_dy_px__{method_a}",
            f"align_dx_px__{method_b}",
            f"align_dy_px__{method_b}",
        ]
        if any(col not in wide for col in needed):
            continue
        dx = wide[f"align_dx_px__{method_a}"] - wide[f"align_dx_px__{method_b}"]
        dy = wide[f"align_dy_px__{method_a}"] - wide[f"align_dy_px__{method_b}"]
        norm = np.hypot(dx, dy)
        chamfer_delta = wide[f"holdout_chamfer_px__{method_a}"] - wide[f"holdout_chamfer_px__{method_b}"]
        corr_delta = wide[f"gradient_corr__{method_a}"] - wide[f"gradient_corr__{method_b}"]
        rows.append(
            {
                "comparison": label,
                "n_frames": int(norm.notna().sum()),
                "delta_norm_median_px": float(norm.median()),
                "delta_norm_p90_px": float(norm.quantile(0.9)),
                "delta_norm_max_px": float(norm.max()),
                "delta_dx_span_px": float(dx.max() - dx.min()),
                "delta_dy_span_px": float(dy.max() - dy.min()),
                "paired_chamfer_delta_median_px": float(chamfer_delta.median()),
                "paired_gradient_corr_delta_median": float(corr_delta.median()),
            }
        )
    return pd.DataFrame(rows)


def overlay_density_table(overlay_density: pd.DataFrame) -> pd.DataFrame:
    """Return a compact contour-stack density table for the notebook."""
    df = _with_method_labels(overlay_density)
    cols = [
        "method_label",
        "sampled_frames",
        "density_peak",
        "density_p99",
        "near_reference_edge_mean",
        "off_reference_edge_mean",
        "near_off_ratio",
    ]
    return df[cols].reset_index(drop=True)


def overlay_group_summary_table(overlay_summary: pd.DataFrame) -> pd.DataFrame:
    """Return all overlay groups and methods in a stable reading order."""
    df = overlay_summary.copy()
    df["group_label"] = df["group"].map(GROUP_LABELS).fillna(df["group"])
    df["method_label"] = df["method"].map(METHOD_LABELS).fillna(df["method"])
    df["group_order"] = df["group"].map({name: i for i, name in enumerate(GROUP_ORDER)}).fillna(999).astype(int)
    df["method_order"] = df["method"].map(_method_order_map()).fillna(999).astype(int)
    best_idx = df.groupby("group")["median_chamfer_px"].idxmin()
    df["best_by_median"] = ""
    df.loc[best_idx, "best_by_median"] = "yes"
    cols = [
        "group_label",
        "method_label",
        "n_frames",
        "median_chamfer_px",
        "p90_chamfer_px",
        "mean_chamfer_px",
        "best_by_median",
    ]
    return df.sort_values(["group_order", "method_order"])[cols].reset_index(drop=True)


def overlay_group_winner_table(overlay_summary: pd.DataFrame) -> pd.DataFrame:
    """Return compact overlay winners with filename-vs-contour deltas."""
    rows = []
    for group in GROUP_ORDER:
        part = overlay_summary[overlay_summary["group"] == group]
        if part.empty:
            continue
        best = part.loc[part["median_chamfer_px"].idxmin()]
        filename = part[part["method"] == "filename_affine_fit"]
        contour = part[part["method"] == "data_driven_contour"]
        filename_median = float(filename["median_chamfer_px"].iloc[0]) if not filename.empty else np.nan
        contour_median = float(contour["median_chamfer_px"].iloc[0]) if not contour.empty else np.nan
        delta = contour_median - filename_median
        if np.isnan(delta):
            note = "Missing filename or contour comparison"
        elif delta < 0:
            note = "Data-driven contour is lower by median Chamfer"
        elif delta > 0:
            note = "Filename affine is lower by median Chamfer"
        else:
            note = "Median Chamfer tie"
        rows.append(
            {
                "group_label": GROUP_LABELS.get(group, group),
                "n_frames": int(best["n_frames"]),
                "best_method_by_median": METHOD_LABELS.get(best["method"], best["method"]),
                "best_median_chamfer_px": float(best["median_chamfer_px"]),
                "filename_median_chamfer_px": filename_median,
                "contour_median_chamfer_px": contour_median,
                "contour_minus_filename_px": delta,
                "note": note,
            }
        )
    return pd.DataFrame(rows)


def ep06_decision_table() -> pd.DataFrame:
    """Return the EP06 handoff decision table."""
    return pd.DataFrame(
        [
            {
                "item": "EP06 recommended alignment",
                "decision": "Use data-driven NCC init plus contour refinement gate",
                "reason": "Lowest held-out contour Chamfer while preserving a data-constrained local anchor.",
            },
            {
                "item": "Phase prior",
                "decision": "Keep filename affine and NCC init",
                "reason": "Both provide continuous 2x phase coverage and remain useful before local refinement.",
            },
            {
                "item": "Control groups",
                "decision": "No alignment, stage prior, filename affine, NCC init",
                "reason": "These separate display gain from true data-driven alignment gain.",
            },
            {
                "item": "High-magnification status",
                "decision": "Treat 3x/4x as risk diagnostics only",
                "reason": "Occupancy is not SR proof, and contour-refined high-scale phase collapse is a failure risk.",
            },
            {
                "item": "Overlay use",
                "decision": "Use overlay only as a visual sanity appendix",
                "reason": "Filename affine wins several groups and scanline_y20 favors contour; overlay is not an SR metric.",
            },
            {
                "item": "Primary failure risks",
                "decision": "Thermal drift, local contour ambiguity, PSF/SNR ceiling",
                "reason": "These can create visually plausible stacks without stable held-out contour improvement.",
            },
            {
                "item": "Acceptance metrics",
                "decision": "Split-half consistency, held-out contour Chamfer, phase-bin coverage, visual contour gain",
                "reason": "Back-projection residual or Tenengrad alone is not sufficient evidence for SR success.",
            },
        ]
    )


def _as_path_list(paths: Path | str | list[Path | str] | tuple[Path | str, ...]) -> list[Path]:
    if isinstance(paths, (str, Path)):
        return [Path(paths)]
    return [Path(path) for path in paths]


def _first_existing_path(base_dirs: list[Path], relative_path: str) -> Path | None:
    for base_dir in base_dirs:
        path = base_dir / relative_path
        if path.exists():
            return path
    return None


def load_alignment_tuning_outputs(base_dirs: Path | str | list[Path | str] | tuple[Path | str, ...]) -> dict:
    """Load optional EP05 alignment-tuning outputs without requiring every artifact."""
    dirs = _as_path_list(base_dirs)
    limit_path = _first_existing_path(dirs, "limit96_tuning_summary.csv")
    full_path = _first_existing_path(dirs, "full_candidate_eval93_summary.csv")

    capacity_dirs = []
    for base_dir in dirs:
        if base_dir.exists():
            capacity_dirs.extend(sorted(base_dir.glob("*_capacity_eval93")))

    capacity_rows = []
    for path in capacity_dirs:
        candidate = path.name.removesuffix("_capacity_eval93")
        capacity_rows.append(
            {
                "candidate": candidate,
                "directory": path,
                "method_summary_csv": path / "alignment_method_summary.csv",
                "comparison_png": path / "alignment_method_comparison.png",
                "phase_png": path / "phase_bin_coverage_2x.png",
                "overlay_png": path / "alignment_overlay_evidence.png",
                "has_method_summary": (path / "alignment_method_summary.csv").exists(),
                "has_comparison_png": (path / "alignment_method_comparison.png").exists(),
                "has_phase_png": (path / "phase_bin_coverage_2x.png").exists(),
                "has_overlay_png": (path / "alignment_overlay_evidence.png").exists(),
            }
        )

    return {
        "base_dirs": dirs,
        "limit_path": limit_path,
        "limit_summary": pd.read_csv(limit_path) if limit_path is not None else pd.DataFrame(),
        "full_path": full_path,
        "full_summary": pd.read_csv(full_path) if full_path is not None else pd.DataFrame(),
        "capacity_artifacts": pd.DataFrame(capacity_rows),
    }


def alignment_tuning_status_table(outputs: dict) -> pd.DataFrame:
    """Return a compact availability table for optional alignment-tuning artifacts."""
    rows = [
        {
            "artifact": "limit96_tuning_summary.csv",
            "status": "available" if outputs["limit_path"] is not None else "missing",
            "path": str(outputs["limit_path"]) if outputs["limit_path"] is not None else "",
            "use": "fast 96-frame parameter sweep",
        },
        {
            "artifact": "full_candidate_eval93_summary.csv",
            "status": "available" if outputs["full_path"] is not None else "missing",
            "path": str(outputs["full_path"]) if outputs["full_path"] is not None else "",
            "use": "clean-main finalist comparison re-scored at edge percentile 93",
        },
    ]
    artifacts = outputs.get("capacity_artifacts", pd.DataFrame())
    if not artifacts.empty:
        rows.append(
            {
                "artifact": "*_capacity_eval93 directories",
                "status": f"{len(artifacts)} found",
                "path": ", ".join(artifacts["candidate"].astype(str).tolist()),
                "use": "method-comparison and phase-coverage PNG/CSV appendices",
            }
        )
    else:
        rows.append(
            {
                "artifact": "*_capacity_eval93 directories",
                "status": "missing",
                "path": "",
                "use": "method-comparison and phase-coverage PNG/CSV appendices",
            }
        )
    return pd.DataFrame(rows)


def alignment_tuning_limit_table(limit_summary: pd.DataFrame) -> pd.DataFrame:
    """Return the 96-frame tuning sweep ranked by refined held-out Chamfer."""
    if limit_summary.empty:
        return pd.DataFrame()
    cols = [
        "rank",
        "name",
        "roi",
        "edge",
        "radius",
        "step",
        "n_success",
        "init_med",
        "refined_med",
        "refined_p90",
        "gain_vs_init_pct",
        "gain_vs_noalign_pct",
        "corr_gain_med",
        "shift_norm_med",
        "worse_than_init_frac",
    ]
    out = limit_summary.sort_values(["refined_med", "refined_p90", "worse_than_init_frac"]).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out[cols]


def alignment_tuning_full_candidate_table(full_summary: pd.DataFrame) -> pd.DataFrame:
    """Return the full-frame finalist comparison ranked by held-out Chamfer."""
    if full_summary.empty:
        return pd.DataFrame()
    cols = [
        "rank",
        "name",
        "roi",
        "edge",
        "radius",
        "step",
        "eval93_refined_med",
        "eval93_refined_p90",
        "eval93_refined_corr_med",
        "eval93_ncc_med",
        "eval93_ncc_corr_med",
        "eval93_filename_med",
        "eval93_filename_corr_med",
        "refined_gain_vs_ncc_pct",
        "refined_gain_vs_filename_pct",
        "phase2_min_count",
        "phase2_max_count",
        "phase2_entropy",
    ]
    out = full_summary.sort_values(["eval93_refined_med", "eval93_refined_p90"]).reset_index(drop=True)
    out.insert(0, "rank", np.arange(1, len(out) + 1))
    return out[cols]


def alignment_tuning_capacity_method_table(outputs: dict) -> pd.DataFrame:
    """Read candidate capacity method summaries when they are present."""
    artifacts = outputs.get("capacity_artifacts", pd.DataFrame())
    if artifacts.empty:
        return pd.DataFrame()
    rows = []
    for _, artifact in artifacts.iterrows():
        csv_path = artifact["method_summary_csv"]
        if not Path(csv_path).exists():
            continue
        df = pd.read_csv(csv_path)
        df = _with_method_labels(df)
        df["candidate"] = artifact["candidate"]
        rows.append(df)
    if not rows:
        return pd.DataFrame()
    combined = pd.concat(rows, ignore_index=True)
    cols = [
        "candidate",
        "method_label",
        "n_frames",
        "holdout_chamfer_median_px",
        "holdout_chamfer_p90_px",
        "gradient_corr_median",
        "gradient_corr_p10",
        "shift_norm_median_px",
        "shift_norm_p90_px",
    ]
    return combined.sort_values(["candidate", "method_order"])[cols].reset_index(drop=True)


def alignment_tuning_conclusion_table(
    limit_summary: pd.DataFrame,
    full_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Return a small evidence-boundary table for the tuning appendix."""
    rows = []
    if not limit_summary.empty:
        best_limit = alignment_tuning_limit_table(limit_summary).iloc[0]
        rows.append(
            {
                "question": "96-frame tuning winner",
                "answer": str(best_limit["name"]),
                "evidence": (
                    f"refined median={best_limit['refined_med']:.4f} px, "
                    f"P90={best_limit['refined_p90']:.4f} px"
                ),
                "boundary": "Fast sweep is a screening step; it does not replace full clean-input validation.",
            }
        )
    if not full_summary.empty:
        best_full = alignment_tuning_full_candidate_table(full_summary).iloc[0]
        rows.append(
            {
                "question": "Full-run recommended contour-refine setting",
                "answer": str(best_full["name"]),
                "evidence": (
                    f"eval93 refined median={best_full['eval93_refined_med']:.4f} px, "
                    f"P90={best_full['eval93_refined_p90']:.4f} px"
                ),
                "boundary": "This selects an alignment gate for EP06; it is not a standalone SR-success claim.",
            }
        )
        rows.append(
            {
                "question": "Gain over NCC init",
                "answer": f"{best_full['refined_gain_vs_ncc_pct']:.1f}% median Chamfer reduction",
                "evidence": (
                    f"NCC median={best_full['eval93_ncc_med']:.4f} px, "
                    f"refined median={best_full['eval93_refined_med']:.4f} px"
                ),
                "boundary": "NCC still keeps stronger gradient correlation and remains the continuous phase prior.",
            }
        )
        rows.append(
            {
                "question": "2x phase health",
                "answer": f"min/max bin count={int(best_full['phase2_min_count'])}/{int(best_full['phase2_max_count'])}",
                "evidence": f"2x entropy={best_full['phase2_entropy']:.3f}",
                "boundary": "2x phase coverage is capacity evidence only; PSF/SNR and split-half checks remain required.",
            }
        )
    return pd.DataFrame(rows)
