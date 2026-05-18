"""Validate data-driven contour alignment without stage-direction assumptions.

The reference frame is only a coordinate system.  All shifts are estimated from
image evidence:

1. high-pass NCC gives a free 2D translation initialisation;
2. contour/edge Chamfer matching refines that translation;
3. held-out edge points validate whether contour alignment improves.

No stage angle, command direction, or nominal displacement magnitude is used.
"""

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates, shift as ndi_shift, sobel
from tqdm import tqdm

from thermal_core.displacement import subpixel_ncc
from thermal_core.io import load_frame
from thermal_core.plotting import COLORMAPS, FIGURE_SIZES, savefig_academic, setup_academic_style


@dataclass(frozen=True)
class AlignmentTask:
    frame_index: int
    file: str
    acquisition_order: int
    x_um: float
    y_um: float
    r: int
    data_dir: str
    reference_file: str
    roi_size: int
    search_radius: int
    edge_percentile: float
    refine_radius_px: float
    refine_step_px: float
    max_edge_points: int


def project_root() -> Path:
    root = Path.cwd()
    while not (root / "AGENTS.md").exists() and root != root.parent:
        root = root.parent
    return root


def default_workers() -> int:
    return max(1, min(os.cpu_count() or 1, 16))


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


def refine_translation_by_chamfer(
    moving_coords_fit: np.ndarray,
    ref_distance: np.ndarray,
    init_align_dx: float,
    init_align_dy: float,
    radius_px: float,
    step_px: float,
) -> tuple[float, float, float]:
    offsets = np.arange(-radius_px, radius_px + step_px * 0.5, step_px)
    best = (float(init_align_dx), float(init_align_dy), np.inf)
    for off_y in offsets:
        for off_x in offsets:
            dx = float(init_align_dx + off_x)
            dy = float(init_align_dy + off_y)
            score = chamfer_score(moving_coords_fit, ref_distance, dx, dy)
            if np.isfinite(score) and score < best[2]:
                best = (dx, dy, score)
    return best


def image_correlation_after_shift(ref_grad: np.ndarray, mov_grad: np.ndarray, align_dx: float, align_dy: float) -> float:
    shifted = ndi_shift(mov_grad, shift=(align_dy, align_dx), order=1, mode="nearest")
    a = z_norm(ref_grad)
    b = z_norm(shifted)
    return float(np.mean(a * b))


def run_alignment_task(task: AlignmentTask) -> dict:
    data_dir = Path(task.data_dir)
    ref_frame_full = load_frame(data_dir / task.reference_file).astype(np.float32, copy=False)
    mov_frame_full = load_frame(data_dir / task.file).astype(np.float32, copy=False)
    roi = center_roi(ref_frame_full.shape, task.roi_size)
    ref_frame = ref_frame_full[roi]
    mov_frame = mov_frame_full[roi]

    ncc = subpixel_ncc(
        ref_frame_full,
        mov_frame_full,
        roi=roi,
        search_radius=task.search_radius,
        fit_radius=2,
        preprocess="highpass",
        highpass_sigma=6.0,
    )
    init_align_dx = -float(ncc["dx_px"])
    init_align_dy = -float(ncc["dy_px"])

    ref_edges = edge_mask_from_frame(ref_frame, task.edge_percentile)
    mov_edges = edge_mask_from_frame(mov_frame, task.edge_percentile)
    ref_distance = distance_transform_edt(~ref_edges)
    mov_coords = np.argwhere(mov_edges)
    mov_coords = deterministic_subsample(mov_coords, task.max_edge_points)

    if mov_coords.shape[0] < 40 or ref_edges.sum() < 40:
        return {
            "frame_index": task.frame_index,
            "file": task.file,
            "acquisition_order": task.acquisition_order,
            "X": task.x_um,
            "Y": task.y_um,
            "R": task.r,
            "reference_file": task.reference_file,
            "success": False,
            "fail_reason": "too_few_edges",
            "n_ref_edges": int(ref_edges.sum()),
            "n_mov_edges": int(mov_edges.sum()),
        }

    fit_coords = mov_coords[::2]
    holdout_coords = mov_coords[1::2]
    if holdout_coords.shape[0] < 20:
        holdout_coords = fit_coords

    before_fit = chamfer_score(fit_coords, ref_distance, 0.0, 0.0)
    before_holdout = chamfer_score(holdout_coords, ref_distance, 0.0, 0.0)
    init_fit = chamfer_score(fit_coords, ref_distance, init_align_dx, init_align_dy)
    init_holdout = chamfer_score(holdout_coords, ref_distance, init_align_dx, init_align_dy)
    refined_dx, refined_dy, refined_fit = refine_translation_by_chamfer(
        fit_coords,
        ref_distance,
        init_align_dx,
        init_align_dy,
        task.refine_radius_px,
        task.refine_step_px,
    )
    refined_holdout = chamfer_score(holdout_coords, ref_distance, refined_dx, refined_dy)

    ref_grad = gradient_magnitude(ref_frame)
    mov_grad = gradient_magnitude(mov_frame)
    corr_before = image_correlation_after_shift(ref_grad, mov_grad, 0.0, 0.0)
    corr_refined = image_correlation_after_shift(ref_grad, mov_grad, refined_dx, refined_dy)

    improvement = np.nan
    if np.isfinite(before_holdout) and before_holdout > 0 and np.isfinite(refined_holdout):
        improvement = 100.0 * (before_holdout - refined_holdout) / before_holdout

    return {
        "frame_index": task.frame_index,
        "file": task.file,
        "acquisition_order": task.acquisition_order,
        "X": task.x_um,
        "Y": task.y_um,
        "R": task.r,
        "reference_file": task.reference_file,
        "success": True,
        "fail_reason": "ok",
        "n_ref_edges": int(ref_edges.sum()),
        "n_mov_edges": int(mov_edges.sum()),
        "n_used_edges": int(mov_coords.shape[0]),
        "ncc_dx_ref_to_frame_px": float(ncc["dx_px"]),
        "ncc_dy_ref_to_frame_px": float(ncc["dy_px"]),
        "ncc_peak": float(ncc["peak_ncc"]),
        "ncc_fit_ok": bool(ncc["fit_ok"]),
        "ncc_edge_peak": bool(ncc["edge_peak"]),
        "init_align_dx_px": init_align_dx,
        "init_align_dy_px": init_align_dy,
        "refined_align_dx_px": refined_dx,
        "refined_align_dy_px": refined_dy,
        "refined_shift_norm_px": float(np.hypot(refined_dx, refined_dy)),
        "before_fit_chamfer_px": before_fit,
        "before_holdout_chamfer_px": before_holdout,
        "init_fit_chamfer_px": init_fit,
        "init_holdout_chamfer_px": init_holdout,
        "refined_fit_chamfer_px": refined_fit,
        "refined_holdout_chamfer_px": refined_holdout,
        "holdout_improvement_pct": improvement,
        "gradient_corr_before": corr_before,
        "gradient_corr_refined": corr_refined,
        "gradient_corr_gain": corr_refined - corr_before,
    }


def select_main_session(audit: pd.DataFrame) -> pd.DataFrame:
    if "is_main_session" in audit:
        main = audit[boolish(audit["is_main_session"])].copy()
    else:
        main_session = audit.groupby("session")["file"].count().idxmax()
        main = audit[audit["session"].eq(main_session)].copy()
    return main.sort_values("acquisition_order").reset_index(drop=True)


def choose_reference(main: pd.DataFrame) -> pd.Series:
    middle = len(main) // 2
    return main.iloc[middle]


def quantiles(series: pd.Series) -> dict[str, float | int]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
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


def plot_alignment_summary(results: pd.DataFrame, output_dir: Path) -> None:
    valid = results[results["success"].astype(bool)].copy()
    if valid.empty:
        return
    setup_academic_style()

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["one_half_col"])
    ax.scatter(
        valid["refined_align_dx_px"],
        valid["refined_align_dy_px"],
        c=valid["acquisition_order"],
        cmap=COLORMAPS["coverage"],
        s=14,
        linewidths=0,
    )
    ax.axhline(0, color="#888888", linewidth=0.7)
    ax.axvline(0, color="#888888", linewidth=0.7)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Data-driven align dx [px]")
    ax.set_ylabel("Data-driven align dy [px]")
    ax.set_title("Contour-Refined Frame Alignment Shifts")
    cbar = fig.colorbar(ax.collections[0], ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Acquisition order")
    savefig_academic(fig, output_dir / "contour_refined_alignment_shifts.png")

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["one_half_col"])
    data = [
        valid["before_holdout_chamfer_px"].dropna().to_numpy(),
        valid["init_holdout_chamfer_px"].dropna().to_numpy(),
        valid["refined_holdout_chamfer_px"].dropna().to_numpy(),
    ]
    ax.boxplot(data, tick_labels=["before", "NCC init", "contour refined"], showfliers=False)
    ax.set_ylabel("Held-out contour Chamfer [px]")
    ax.set_title("Data-Driven Contour Alignment Validation")
    savefig_academic(fig, output_dir / "contour_alignment_chamfer_validation.png")

    fig, ax = plt.subplots(figsize=FIGURE_SIZES["one_half_col"])
    ax.scatter(
        valid["acquisition_order"],
        valid["holdout_improvement_pct"],
        s=12,
        color="#4C72B0",
        alpha=0.8,
        linewidths=0,
    )
    ax.axhline(0, color="#888888", linewidth=0.8)
    ax.set_xlabel("Acquisition order")
    ax.set_ylabel("Hold-out Chamfer improvement [%]")
    ax.set_title("Contour Alignment Improvement Over Time")
    savefig_academic(fig, output_dir / "contour_alignment_improvement_timeline.png")

    coord = (
        valid.groupby(["X", "Y"], as_index=False)
        .agg(
            dx=("refined_align_dx_px", "median"),
            dy=("refined_align_dy_px", "median"),
            norm=("refined_shift_norm_px", "median"),
        )
        .sort_values(["Y", "X"])
    )
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["one_half_col"])
    sc = ax.scatter(coord["X"], coord["Y"], c=coord["norm"], cmap=COLORMAPS["coverage"], s=24, zorder=2)
    ax.quiver(
        coord["X"],
        coord["Y"],
        coord["dx"],
        coord["dy"],
        angles="xy",
        scale_units="xy",
        scale=0.35,
        width=0.003,
        color="#333333",
        alpha=0.85,
        zorder=3,
    )
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Commanded X label [um]")
    ax.set_ylabel("Commanded Y label [um]")
    ax.set_title("Data-Driven Alignment Vector Field by Coordinate Label")
    cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label("Shift norm [px]")
    savefig_academic(fig, output_dir / "data_driven_coordinate_shift_field.png")


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-audit-csv", type=Path, default=root / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "data_raw" / "infrared_avi")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "ep05_contour_alignment")
    parser.add_argument("--roi-size", type=int, default=360)
    parser.add_argument("--search-radius", type=int, default=18)
    parser.add_argument("--edge-percentile", type=float, default=93.0)
    parser.add_argument("--refine-radius-px", type=float, default=1.0)
    parser.add_argument("--refine-step-px", type=float, default=0.25)
    parser.add_argument("--max-edge-points", type=int, default=8000)
    parser.add_argument("--n-jobs", type=int, default=default_workers())
    parser.add_argument("--limit-frames", type=int, default=None)
    parser.add_argument("--skip-figures", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    setup_academic_style()

    audit = pd.read_csv(args.frame_audit_csv)
    main_df = select_main_session(audit)
    ref = choose_reference(main_df)
    if args.limit_frames is not None:
        main_df = main_df.iloc[: args.limit_frames].copy()

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
            roi_size=int(args.roi_size),
            search_radius=int(args.search_radius),
            edge_percentile=float(args.edge_percentile),
            refine_radius_px=float(args.refine_radius_px),
            refine_step_px=float(args.refine_step_px),
            max_edge_points=int(args.max_edge_points),
        )
        for i, row in main_df.iterrows()
    ]

    print("EP05 contour alignment validation")
    print(f"frames: {len(tasks)}; reference coordinate frame: {ref['file']} (acq={int(ref['acquisition_order'])})")
    print(f"output: {args.output_dir}")
    print(f"n_jobs={args.n_jobs}, roi={args.roi_size}, search_radius={args.search_radius}")
    print("No stage angle or commanded direction is used.")

    records = []
    if args.n_jobs == 1:
        for task in tqdm(tasks, desc="align"):
            records.append(run_alignment_task(task))
    else:
        with ProcessPoolExecutor(max_workers=args.n_jobs) as executor:
            futures = [executor.submit(run_alignment_task, task) for task in tasks]
            for future in tqdm(as_completed(futures), total=len(futures), desc="align"):
                records.append(future.result())

    results = pd.DataFrame(records).sort_values("acquisition_order").reset_index(drop=True)
    results.to_csv(args.output_dir / "contour_alignment_results.csv", index=False)

    valid = results[results["success"].astype(bool)].copy()
    summary = {
        "n_frames": int(len(results)),
        "n_success": int(len(valid)),
        "reference_file": str(ref["file"]),
        "reference_acquisition_order": int(ref["acquisition_order"]),
        "roi_size": int(args.roi_size),
        "search_radius": int(args.search_radius),
        "edge_percentile": float(args.edge_percentile),
        "refine_radius_px": float(args.refine_radius_px),
        "refine_step_px": float(args.refine_step_px),
        "uses_stage_model": False,
        "before_holdout_chamfer_px": quantiles(valid["before_holdout_chamfer_px"]) if not valid.empty else {"count": 0},
        "init_holdout_chamfer_px": quantiles(valid["init_holdout_chamfer_px"]) if not valid.empty else {"count": 0},
        "refined_holdout_chamfer_px": quantiles(valid["refined_holdout_chamfer_px"]) if not valid.empty else {"count": 0},
        "holdout_improvement_pct": quantiles(valid["holdout_improvement_pct"]) if not valid.empty else {"count": 0},
        "refined_shift_norm_px": quantiles(valid["refined_shift_norm_px"]) if not valid.empty else {"count": 0},
        "gradient_corr_gain": quantiles(valid["gradient_corr_gain"]) if not valid.empty else {"count": 0},
        "shift_span_dx_px": float(valid["refined_align_dx_px"].max() - valid["refined_align_dx_px"].min()) if not valid.empty else np.nan,
        "shift_span_dy_px": float(valid["refined_align_dy_px"].max() - valid["refined_align_dy_px"].min()) if not valid.empty else np.nan,
    }
    with open(args.output_dir / "contour_alignment_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    if not args.skip_figures:
        plot_alignment_summary(results, args.output_dir)

    print("\nSummary")
    for key in [
        "before_holdout_chamfer_px",
        "init_holdout_chamfer_px",
        "refined_holdout_chamfer_px",
        "holdout_improvement_pct",
        "refined_shift_norm_px",
        "gradient_corr_gain",
    ]:
        print(f"{key}: {summary[key]}")
    print(f"shift_span_dx_px: {summary['shift_span_dx_px']:.4f}")
    print(f"shift_span_dy_px: {summary['shift_span_dy_px']:.4f}")


if __name__ == "__main__":
    main()
