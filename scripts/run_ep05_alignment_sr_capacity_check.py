"""Compare alignment methods and 2x SR phase capacity on EP05 main frames.

This script answers two narrow questions:

1. Does the 255-frame main TXT sequence provide enough sub-pixel phase
   diversity for scale-2 contour SR?
2. Is the per-frame data-driven alignment measurably better than filename-based
   affine alignment on held-out contour points?

The comparison intentionally uses the same reference frame and edge extraction
as ``run_ep05_contour_alignment_validation.py``.  Filename affine shifts are
learned only as a global X/Y-label-to-shift model; data-driven shifts are the
per-frame NCC initialisation and the Chamfer-refined shifts already validated in
EP05.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates, shift as ndi_shift, sobel

from thermal_core.displacement import coordinate_to_shift
from thermal_core.io import load_frame
from thermal_core.plotting import COLORMAPS, FIGURE_SIZES, METHOD_COLOR_LIST, savefig_academic, setup_academic_style


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
}


def project_root() -> Path:
    root = Path.cwd()
    while not (root / "AGENTS.md").exists() and root != root.parent:
        root = root.parent
    return root


def boolish(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def center_roi(shape: tuple[int, int], size: int) -> tuple[slice, slice]:
    rows, cols = shape
    r0 = max(0, (rows - size) // 2)
    c0 = max(0, (cols - size) // 2)
    return slice(r0, r0 + min(size, rows)), slice(c0, c0 + min(size, cols))


def highpass(frame: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    data = np.asarray(frame, dtype=np.float32)
    return data - gaussian_filter(data, sigma=sigma, mode="nearest")


def gradient_magnitude(frame: np.ndarray) -> np.ndarray:
    hp = highpass(frame)
    gx = sobel(hp, axis=1, mode="nearest")
    gy = sobel(hp, axis=0, mode="nearest")
    return np.hypot(gx, gy).astype(np.float32, copy=False)


def z_norm(frame: np.ndarray) -> np.ndarray:
    data = np.asarray(frame, dtype=np.float32)
    centered = data - float(np.mean(data))
    scale = float(np.std(centered))
    if scale <= 0:
        return centered
    return centered / scale


def edge_mask_from_frame(frame: np.ndarray, percentile: float) -> np.ndarray:
    grad = gradient_magnitude(frame)
    threshold = float(np.percentile(grad, percentile))
    mask = grad >= threshold
    mask[:2, :] = False
    mask[-2:, :] = False
    mask[:, :2] = False
    mask[:, -2:] = False
    return mask


def deterministic_subsample(coords: np.ndarray, max_points: int) -> np.ndarray:
    if coords.shape[0] <= max_points:
        return coords
    idx = np.linspace(0, coords.shape[0] - 1, max_points, dtype=int)
    return coords[idx]


def chamfer_score(
    moving_coords_yx: np.ndarray,
    ref_distance: np.ndarray,
    align_dx: float,
    align_dy: float,
) -> float:
    if moving_coords_yx.size == 0:
        return np.nan
    coords = np.vstack([
        moving_coords_yx[:, 0] + align_dy,
        moving_coords_yx[:, 1] + align_dx,
    ])
    valid = (
        (coords[0] >= 0)
        & (coords[0] <= ref_distance.shape[0] - 1)
        & (coords[1] >= 0)
        & (coords[1] <= ref_distance.shape[1] - 1)
    )
    if valid.sum() < 20:
        return np.nan
    values = map_coordinates(ref_distance, coords[:, valid], order=1, mode="nearest")
    return float(np.mean(values))


def image_correlation_after_shift(ref_grad: np.ndarray, mov_grad: np.ndarray, align_dx: float, align_dy: float) -> float:
    shifted = ndi_shift(mov_grad, shift=(align_dy, align_dx), order=1, mode="nearest")
    return float(np.mean(z_norm(ref_grad) * z_norm(shifted)))


def select_main_session(audit: pd.DataFrame) -> pd.DataFrame:
    if "is_main_session" in audit:
        main = audit[boolish(audit["is_main_session"])].copy()
    else:
        main_session = audit.groupby("session")["file"].count().idxmax()
        main = audit[audit["session"].eq(main_session)].copy()
    return main.sort_values("acquisition_order").reset_index(drop=True)


def fit_affine_from_alignment(alignment: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    valid = alignment[alignment["success"].astype(str).str.lower().eq("true") & alignment["R"].eq(0)].copy()
    design = np.column_stack([np.ones(len(valid)), valid["X"].to_numpy(), valid["Y"].to_numpy()])
    beta_dx = np.linalg.lstsq(design, valid["refined_align_dx_px"].to_numpy(), rcond=None)[0]
    beta_dy = np.linalg.lstsq(design, valid["refined_align_dy_px"].to_numpy(), rcond=None)[0]
    return beta_dx, beta_dy


def affine_shift(row: pd.Series, beta_dx: np.ndarray, beta_dy: np.ndarray) -> tuple[float, float]:
    coord = np.array([1.0, float(row["X"]), float(row["Y"])])
    return float(coord @ beta_dx), float(coord @ beta_dy)


def stage_shift(row: pd.Series, ref_row: pd.Series, theta_deg: float, pixel_size_um: float) -> tuple[float, float]:
    dx, dy_math = coordinate_to_shift(
        float(row["X"]) - float(ref_row["X"]),
        float(row["Y"]) - float(ref_row["Y"]),
        theta_deg=theta_deg,
        pixel_size_um=pixel_size_um,
    )
    return float(-dx), float(dy_math)


def quantiles(values: pd.Series | np.ndarray) -> dict[str, float | int]:
    clean = pd.Series(values, dtype=float).replace([np.inf, -np.inf], np.nan).dropna()
    if clean.empty:
        return {"count": 0}
    q = clean.quantile([0.0, 0.1, 0.5, 0.9, 1.0])
    return {
        "count": int(clean.size),
        "min": float(q.loc[0.0]),
        "p10": float(q.loc[0.1]),
        "median": float(q.loc[0.5]),
        "p90": float(q.loc[0.9]),
        "max": float(q.loc[1.0]),
    }


def phase_occupancy(dx: np.ndarray, dy: np.ndarray, scale: int) -> dict:
    fx = np.mod(dx, 1.0)
    fy = np.mod(dy, 1.0)
    counts, _, _ = np.histogram2d(fy, fx, bins=scale, range=[[0.0, 1.0], [0.0, 1.0]])
    counts = counts.astype(int)
    probs = counts.ravel() / max(1, int(counts.sum()))
    nonzero = probs[probs > 0]
    entropy = float(-(nonzero * np.log2(nonzero)).sum() / np.log2(scale * scale)) if nonzero.size else 0.0
    return {
        "scale": int(scale),
        "counts": counts.tolist(),
        "occupied_bins": int(np.count_nonzero(counts)),
        "total_bins": int(scale * scale),
        "bad_bins": int(np.count_nonzero(counts == 0)),
        "min_count": int(counts.min()),
        "max_count": int(counts.max()),
        "entropy_fraction": entropy,
    }


def phase_tables(scores: pd.DataFrame, scale: int = 2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return method-level scale-2 phase summary and long-form bin counts."""
    summary_rows = []
    count_rows = []
    for method in METHOD_ORDER:
        group = scores[scores["method"].eq(method)].copy()
        if group.empty:
            continue
        dx = group["align_dx_px"].to_numpy(dtype=float)
        dy = group["align_dy_px"].to_numpy(dtype=float)
        occ = phase_occupancy(dx, dy, scale)
        counts = np.asarray(occ["counts"], dtype=int)
        expected = float(len(group) / (scale * scale))
        summary_rows.append(
            {
                "method": method,
                "method_label": METHOD_LABELS.get(method, method),
                "scale": int(scale),
                "n_frames": int(len(group)),
                "occupied_bins": int(occ["occupied_bins"]),
                "bad_bins": int(occ["bad_bins"]),
                "total_bins": int(occ["total_bins"]),
                "entropy_fraction": float(occ["entropy_fraction"]),
                "min_count": int(occ["min_count"]),
                "max_count": int(occ["max_count"]),
                "expected_count": expected,
            }
        )
        for y_bin in range(scale):
            for x_bin in range(scale):
                count_rows.append(
                    {
                        "method": method,
                        "method_label": METHOD_LABELS.get(method, method),
                        "scale": int(scale),
                        "phase_y_bin": int(y_bin),
                        "phase_x_bin": int(x_bin),
                        "count": int(counts[y_bin, x_bin]),
                        "bad_bin": bool(counts[y_bin, x_bin] == 0),
                        "expected_count": expected,
                    }
                )
    return pd.DataFrame(summary_rows), pd.DataFrame(count_rows)


def plot_phase_bin_coverage(phase_summary: pd.DataFrame, phase_counts: pd.DataFrame, output_path: Path) -> None:
    """Save a compact 2x phase-bin coverage chart for every alignment method."""
    scale = int(phase_summary["scale"].iloc[0])
    methods = [m for m in METHOD_ORDER if m in set(phase_summary["method"])]
    phase_order = [(0, 0), (0, 1), (1, 0), (1, 1)]
    phase_labels = [f"bin ({y},{x})" for y, x in phase_order]
    phase_colors = ["#4C72B0", "#DD8452", "#55A868", "#8172B2"]

    def text_color(hex_color: str) -> str:
        rgb = np.array([int(hex_color[i : i + 2], 16) for i in (1, 3, 5)], dtype=float) / 255.0
        luminance = float(np.dot(rgb, [0.2126, 0.7152, 0.0722]))
        return "black" if luminance > 0.55 else "white"

    fig, ax = plt.subplots(figsize=(FIGURE_SIZES["double_col"][0], 3.35))
    y_positions = np.arange(len(methods))
    expected = float(phase_summary["expected_count"].iloc[0])
    n_frames = int(phase_summary["n_frames"].max())

    for y_pos, method in zip(y_positions, methods):
        sub = phase_counts[phase_counts["method"].eq(method)].set_index(["phase_y_bin", "phase_x_bin"])
        left = 0.0
        for (phase_y, phase_x), color in zip(phase_order, phase_colors):
            count = int(sub.loc[(phase_y, phase_x), "count"])
            if count > 0:
                ax.barh(
                    y_pos,
                    count,
                    left=left,
                    height=0.62,
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                )
                if count >= 18:
                    ax.text(
                        left + count / 2.0,
                        y_pos,
                        str(count),
                        ha="center",
                        va="center",
                        fontsize=8,
                        color=text_color(color),
                    )
            left += count

        meta = phase_summary[phase_summary["method"].eq(method)].iloc[0]
        ax.text(
            n_frames + expected * 0.18,
            y_pos,
            f"occ {int(meta['occupied_bins'])}/{int(meta['total_bins'])}, "
            f"empty {int(meta['bad_bins'])}, H={float(meta['entropy_fraction']):.2f}",
            ha="left",
            va="center",
            fontsize=8,
            color="#333333",
        )

    for k in range(1, scale * scale):
        ax.axvline(expected * k, color="#777777", linestyle=":", linewidth=0.8, zorder=0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([METHOD_LABELS.get(m, m) for m in methods])
    ax.invert_yaxis()
    ax.set_xlim(0, n_frames + expected * 1.75)
    ax.set_xlabel("Frame count, stacked by 2x phase bin")
    ax.set_title("2x Phase-Bin Coverage")
    ax.grid(axis="x", alpha=0.2)
    handles = [plt.Rectangle((0, 0), 1, 1, color=color) for color in phase_colors]
    ax.legend(
        handles,
        phase_labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        ncol=1,
        fontsize=7,
        title="Phase bin",
        borderaxespad=0.0,
    )
    savefig_academic(fig, output_path)


def plot_alignment_method_comparison(summary: pd.DataFrame, output_path: Path) -> None:
    """Save held-out Chamfer and gradient-correlation comparison in fixed method order."""
    plot_df = summary.set_index("method").loc[[m for m in METHOD_ORDER if m in set(summary["method"])]].reset_index()
    labels = [METHOD_LABELS.get(m, m) for m in plot_df["method"]]
    y = np.arange(len(plot_df))

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["double_col"])
    color_a = METHOD_COLOR_LIST[0]
    color_b = METHOD_COLOR_LIST[1]

    axes[0].barh(y, plot_df["holdout_chamfer_median_px"], color=color_a, alpha=0.88, label="median")
    axes[0].scatter(plot_df["holdout_chamfer_p90_px"], y, color="#333333", s=18, label="P90", zorder=3)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_xlabel("Held-out Chamfer [px]")
    axes[0].set_title("Contour Holdout Error")
    axes[0].legend(loc="upper right", ncol=2)

    axes[1].barh(y, plot_df["gradient_corr_median"], color=color_b, alpha=0.88, label="median")
    axes[1].scatter(plot_df["gradient_corr_p10"], y, color="#333333", s=18, label="P10", zorder=3)
    axes[1].set_yticks(y)
    axes[1].set_yticklabels([])
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Gradient correlation")
    axes[1].set_xlim(0.45, 1.0)
    axes[1].set_title("Gradient Agreement")
    axes[1].legend(loc="upper right", ncol=2)

    savefig_academic(fig, output_path)


def _density_metrics(name: str, density: np.ndarray, ref_distance: np.ndarray) -> dict:
    near_ref = ref_distance <= 1.5
    off_ref = ~near_ref
    off_mean = float(np.mean(density[off_ref])) if np.any(off_ref) else np.nan
    near_mean = float(np.mean(density[near_ref])) if np.any(near_ref) else np.nan
    return {
        "method": name,
        "density_peak": float(np.max(density)),
        "density_p99": float(np.quantile(density, 0.99)),
        "near_reference_edge_mean": near_mean,
        "off_reference_edge_mean": off_mean,
        "near_off_ratio": float(near_mean / off_mean) if np.isfinite(off_mean) and off_mean > 0 else np.nan,
    }


def plot_overlay_density_evidence(
    main_df: pd.DataFrame,
    align_lookup: dict,
    data_dir: Path,
    roi: tuple[slice, slice],
    ref_edges: np.ndarray,
    ref_distance: np.ndarray,
    *,
    edge_percentile: float,
    max_frames: int,
    output_path: Path,
) -> pd.DataFrame:
    """Save readable edge-density evidence for no alignment vs contour refined alignment."""
    eligible = main_df[main_df["file"].astype(str).isin(align_lookup)].sort_values("acquisition_order").reset_index(drop=True)
    if len(eligible) > max_frames:
        sample_idx = np.linspace(0, len(eligible) - 1, max_frames, dtype=int)
        eligible = eligible.iloc[sample_idx].reset_index(drop=True)

    no_align = np.zeros_like(ref_distance, dtype=np.float32)
    refined = np.zeros_like(ref_distance, dtype=np.float32)

    for _, row in eligible.iterrows():
        file_name = str(row["file"])
        frame = load_frame(data_dir / file_name).astype(np.float32, copy=False)[roi]
        edges = edge_mask_from_frame(frame, edge_percentile).astype(np.float32)
        no_align += edges
        align = align_lookup[file_name]
        refined += ndi_shift(
            edges,
            shift=(float(align["refined_align_dy_px"]), float(align["refined_align_dx_px"])),
            order=1,
            mode="constant",
            cval=0.0,
        )

    n = max(1, len(eligible))
    no_align /= n
    refined /= n
    diff = refined - no_align
    vmax = max(float(np.quantile(no_align, 0.995)), float(np.quantile(refined, 0.995)), 1e-6)
    diff_abs = max(float(np.quantile(np.abs(diff), 0.995)), 1e-6)

    fig, axes = plt.subplots(1, 3, figsize=(7.8, 2.75))
    panels = [
        (no_align, "No alignment\nedge density", COLORMAPS["coverage"], 0.0, vmax),
        (refined, "Data-driven refined\nedge density", COLORMAPS["coverage"], 0.0, vmax),
        (diff, "Refined - no alignment", COLORMAPS["residual_diff"], -diff_abs, diff_abs),
    ]
    for ax, (image, title, cmap, vmin, vmax_i) in zip(axes, panels):
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax_i)
        ax.contour(ref_edges.astype(float), levels=[0.5], colors="white", linewidths=0.25, alpha=0.7)
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.suptitle(f"Contour stack evidence, sampled {n} frames")
    savefig_academic(fig, output_path)

    metrics = pd.DataFrame(
        [
            {"sampled_frames": int(n), **_density_metrics("no_alignment", no_align, ref_distance)},
            {"sampled_frames": int(n), **_density_metrics("data_driven_contour_refined", refined, ref_distance)},
        ]
    )
    return metrics


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-audit-csv", type=Path, default=root / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--alignment-csv", type=Path, default=root / "output" / "ep05_contour_alignment" / "contour_alignment_results.csv")
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "data_raw" / "infrared_avi")
    parser.add_argument("--stage-config", type=Path, default=root / "configs" / "stage_calibration.json")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "ep05_alignment_sr_capacity")
    parser.add_argument("--roi-size", type=int, default=360)
    parser.add_argument("--edge-percentile", type=float, default=93.0)
    parser.add_argument("--max-edge-points", type=int, default=8000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_academic_style()

    audit = pd.read_csv(args.frame_audit_csv)
    main_df = select_main_session(audit)
    alignment = pd.read_csv(args.alignment_csv)
    valid_alignment = alignment[alignment["success"].astype(str).str.lower().eq("true")].copy()

    with open(args.stage_config, encoding="utf-8") as f:
        stage = json.load(f)
    theta_deg = float(stage["theta_deg"])
    pixel_size_um = float(stage["pixel_size_um"])

    ref_file = str(valid_alignment["reference_file"].dropna().iloc[0])
    ref_row = main_df[main_df["file"].eq(ref_file)].iloc[0]
    beta_dx, beta_dy = fit_affine_from_alignment(valid_alignment)
    align_lookup = valid_alignment.set_index("file").to_dict("index")

    ref_full = load_frame(args.data_dir / ref_file).astype(np.float32, copy=False)
    roi = center_roi(ref_full.shape, args.roi_size)
    ref = ref_full[roi]
    ref_edges = edge_mask_from_frame(ref, args.edge_percentile)
    ref_distance = distance_transform_edt(~ref_edges)
    ref_grad = gradient_magnitude(ref)

    records = []
    for _, row in main_df.iterrows():
        file_name = str(row["file"])
        if file_name not in align_lookup:
            continue
        mov = load_frame(args.data_dir / file_name).astype(np.float32, copy=False)[roi]
        mov_edges = edge_mask_from_frame(mov, args.edge_percentile)
        mov_coords = deterministic_subsample(np.argwhere(mov_edges), args.max_edge_points)
        if mov_coords.shape[0] < 40:
            continue
        holdout_coords = mov_coords[1::2]
        if holdout_coords.shape[0] < 20:
            holdout_coords = mov_coords[::2]
        mov_grad = gradient_magnitude(mov)
        row_alignment = align_lookup[file_name]

        method_shifts = {
            "no_alignment": (0.0, 0.0),
            "old_stage_model": stage_shift(row, ref_row, theta_deg, pixel_size_um),
            "filename_affine_fit": affine_shift(row, beta_dx, beta_dy),
            "data_driven_ncc_init": (
                float(row_alignment["init_align_dx_px"]),
                float(row_alignment["init_align_dy_px"]),
            ),
            "data_driven_contour_refined": (
                float(row_alignment["refined_align_dx_px"]),
                float(row_alignment["refined_align_dy_px"]),
            ),
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
                    "align_dx_px": float(dx),
                    "align_dy_px": float(dy),
                    "align_norm_px": float(np.hypot(dx, dy)),
                    "holdout_chamfer_px": chamfer_score(holdout_coords, ref_distance, dx, dy),
                    "gradient_corr": image_correlation_after_shift(ref_grad, mov_grad, dx, dy),
                }
            )

    scores = pd.DataFrame(records)
    scores.to_csv(args.output_dir / "alignment_method_holdout_scores.csv", index=False)

    summary_rows = []
    for method, group in scores.groupby("method", sort=False):
        summary_rows.append(
            {
                "method": method,
                "n_frames": int(group["file"].nunique()),
                "holdout_chamfer_median_px": float(group["holdout_chamfer_px"].median()),
                "holdout_chamfer_p90_px": float(group["holdout_chamfer_px"].quantile(0.9)),
                "gradient_corr_median": float(group["gradient_corr"].median()),
                "gradient_corr_p10": float(group["gradient_corr"].quantile(0.1)),
                "shift_norm_median_px": float(group["align_norm_px"].median()),
                "shift_norm_p90_px": float(group["align_norm_px"].quantile(0.9)),
            }
        )
    summary = pd.DataFrame(summary_rows).sort_values("holdout_chamfer_median_px")
    summary.to_csv(args.output_dir / "alignment_method_summary.csv", index=False)

    phase_summary_2x, phase_counts_2x = phase_tables(scores, scale=2)
    phase_summary_2x.to_csv(args.output_dir / "phase_bin_summary_2x.csv", index=False)
    phase_counts_2x.to_csv(args.output_dir / "phase_bin_counts_2x.csv", index=False)
    plot_phase_bin_coverage(
        phase_summary_2x,
        phase_counts_2x,
        args.output_dir / "phase_bin_coverage_2x.png",
    )

    overlay_metrics = plot_overlay_density_evidence(
        main_df,
        align_lookup,
        args.data_dir,
        roi,
        ref_edges,
        ref_distance,
        edge_percentile=float(args.edge_percentile),
        max_frames=min(80, len(main_df)),
        output_path=args.output_dir / "alignment_overlay_evidence.png",
    )
    overlay_metrics.to_csv(args.output_dir / "alignment_overlay_density_metrics.csv", index=False)

    phase = {}
    for method, group in scores.groupby("method", sort=False):
        dx = group["align_dx_px"].to_numpy()
        dy = group["align_dy_px"].to_numpy()
        phase[method] = {
            "shift_span_dx_px": float(np.nanmax(dx) - np.nanmin(dx)),
            "shift_span_dy_px": float(np.nanmax(dy) - np.nanmin(dy)),
            "shift_norm": quantiles(np.hypot(dx, dy)),
            "phase_occupancy": [phase_occupancy(dx, dy, scale) for scale in (2, 3, 4)],
        }

    affine = scores[scores["method"].eq("filename_affine_fit")].set_index("file")
    refined = scores[scores["method"].eq("data_driven_contour_refined")].set_index("file")
    common = affine.index.intersection(refined.index)
    correction = np.hypot(
        refined.loc[common, "align_dx_px"].to_numpy() - affine.loc[common, "align_dx_px"].to_numpy(),
        refined.loc[common, "align_dy_px"].to_numpy() - affine.loc[common, "align_dy_px"].to_numpy(),
    )

    report = {
        "n_main_frames_scored": int(scores["file"].nunique()),
        "reference_file": ref_file,
        "roi_size": int(args.roi_size),
        "edge_percentile": float(args.edge_percentile),
        "alignment_method_summary": summary.to_dict(orient="records"),
        "phase_bin_summary_2x": phase_summary_2x.to_dict(orient="records"),
        "phase_capacity": phase,
        "data_driven_minus_filename_affine_norm_px": quantiles(correction),
        "overlay_density_metrics": overlay_metrics.to_dict(orient="records"),
    }
    with open(args.output_dir / "alignment_sr_capacity_summary.json", "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    plot_alignment_method_comparison(summary, args.output_dir / "alignment_method_comparison.png")

    print("Alignment method summary")
    print(summary.to_string(index=False))
    print("\nPhase occupancy")
    for method, item in phase.items():
        scale2 = item["phase_occupancy"][0]
        scale4 = item["phase_occupancy"][2]
        print(
            f"{method}: 2x SR min/max={scale2['min_count']}/{scale2['max_count']}, "
            f"2x SR occupied/bad={scale2['occupied_bins']}/{scale2['bad_bins']}, "
            f"2x SR entropy={scale2['entropy_fraction']:.3f}; "
            f"4x occupied={scale4['occupied_bins']}/{scale4['total_bins']}, "
            f"4x min/max={scale4['min_count']}/{scale4['max_count']}"
        )
    print(f"\nSaved to {args.output_dir}")


if __name__ == "__main__":
    main()
