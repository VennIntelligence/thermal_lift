"""Create overlay diagnostics for competing alignment assumptions.

The goal is visual and metric sanity checking: stack the same frame batch with
different shifts and see which assumption keeps contours sharp.
"""

from __future__ import annotations

from thermal_core.alignment_paths import default_contour_alignment_csv
import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter, map_coordinates, shift as ndi_shift, sobel

from thermal_core.displacement import coordinate_to_shift
from thermal_core.ep05 import affine_shift, filename_affine_diagnostics, fit_filename_affine
from thermal_core.io import load_frame
from thermal_core.plotting import COLORMAPS, FIGURE_SIZES, savefig_academic, setup_academic_style


def project_root() -> Path:
    root = Path.cwd()
    while not (root / "AGENTS.md").exists() and root != root.parent:
        root = root.parent
    return root


def boolish(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def highpass(frame: np.ndarray, sigma: float = 8.0) -> np.ndarray:
    data = np.asarray(frame, dtype=np.float32)
    return data - gaussian_filter(data, sigma=sigma, mode="nearest")


def gradient_magnitude(frame: np.ndarray) -> np.ndarray:
    hp = highpass(frame)
    gx = sobel(hp, axis=1, mode="nearest")
    gy = sobel(hp, axis=0, mode="nearest")
    return np.hypot(gx, gy).astype(np.float32, copy=False)


def edge_mask(frame: np.ndarray, percentile: float) -> np.ndarray:
    grad = gradient_magnitude(frame)
    threshold = float(np.percentile(grad, percentile))
    mask = grad >= threshold
    mask[:2, :] = False
    mask[-2:, :] = False
    mask[:, :2] = False
    mask[:, -2:] = False
    return mask


def crop_center(frame: np.ndarray, size: int) -> np.ndarray:
    rows, cols = frame.shape
    r0 = max(0, (rows - size) // 2)
    c0 = max(0, (cols - size) // 2)
    return frame[r0 : r0 + min(size, rows), c0 : c0 + min(size, cols)]


def robust_norm(frame: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(frame, [1, 99])
    return np.clip((frame - lo) / (hi - lo + 1e-9), 0, 1)


def load_main_session(audit_csv: Path) -> pd.DataFrame:
    audit = pd.read_csv(audit_csv)
    if "is_main_session" in audit:
        main = audit[boolish(audit["is_main_session"])].copy()
    else:
        main_session = audit.groupby("session")["file"].count().idxmax()
        main = audit[audit["session"].eq(main_session)].copy()
    return main.sort_values("acquisition_order").reset_index(drop=True)


def stage_shift(row: pd.Series, ref_row: pd.Series, theta_deg: float, pixel_size_um: float) -> tuple[float, float]:
    dx, dy_math = coordinate_to_shift(
        float(row["X"]) - float(ref_row["X"]),
        float(row["Y"]) - float(ref_row["Y"]),
        theta_deg=theta_deg,
        pixel_size_um=pixel_size_um,
    )
    # coordinate_to_shift returns math-y; scipy/image rows use image-y.
    return float(-dx), float(dy_math)


def data_driven_shift(row: pd.Series, alignment_lookup: dict[str, tuple[float, float]]) -> tuple[float, float]:
    return alignment_lookup[str(row["file"])]


def stack_group(
    group: pd.DataFrame,
    data_dir: Path,
    ref_file: str,
    shifts: dict[str, tuple[float, float]],
    crop_size: int,
    edge_percentile: float,
) -> dict:
    ref = crop_center(load_frame(data_dir / ref_file).astype(np.float32), crop_size)
    ref_edges = edge_mask(ref, edge_percentile)
    ref_dist = distance_transform_edt(~ref_edges)
    accum_img = []
    accum_edge = []
    chamfers = []
    edge_counts = []
    for _, row in group.iterrows():
        frame = crop_center(load_frame(data_dir / str(row["file"])).astype(np.float32), crop_size)
        dx, dy = shifts[str(row["file"])]
        shifted = ndi_shift(frame, shift=(dy, dx), order=1, mode="nearest")
        edges = edge_mask(shifted, edge_percentile)
        coords = np.argwhere(edges)
        if coords.shape[0] > 0:
            values = map_coordinates(ref_dist, np.vstack([coords[:, 0], coords[:, 1]]), order=1, mode="nearest")
            chamfers.append(float(np.mean(values)))
            edge_counts.append(int(coords.shape[0]))
        accum_img.append(robust_norm(shifted))
        accum_edge.append(edges.astype(np.float32))
    img_stack = np.stack(accum_img, axis=0)
    edge_stack = np.stack(accum_edge, axis=0)
    return {
        "median_image": np.median(img_stack, axis=0),
        "edge_persistence": np.mean(edge_stack, axis=0),
        "mean_chamfer_px": float(np.mean(chamfers)),
        "median_chamfer_px": float(np.median(chamfers)),
        "p90_chamfer_px": float(np.quantile(chamfers, 0.9)),
        "mean_edge_count": float(np.mean(edge_counts)),
        "n_frames": int(len(group)),
    }


def plot_method_grid(results: dict[str, dict], title: str, output_path: Path) -> None:
    methods = list(results)
    setup_academic_style()
    fig, axes = plt.subplots(2, len(methods), figsize=(max(7.2, 2.2 * len(methods)), 4.6))
    for col, method in enumerate(methods):
        image = results[method]["median_image"]
        edges = results[method]["edge_persistence"]
        axes[0, col].imshow(image, cmap=COLORMAPS["temperature"], interpolation="nearest")
        axes[0, col].set_title(method)
        axes[0, col].axis("off")
        im = axes[1, col].imshow(edges, cmap=COLORMAPS["coverage"], vmin=0, vmax=1, interpolation="nearest")
        axes[1, col].set_title(f"Chamfer med {results[method]['median_chamfer_px']:.3f}px")
        axes[1, col].axis("off")
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.01, label="Edge persistence")
    savefig_academic(fig, output_path)


def group_specs(main: pd.DataFrame) -> dict[str, pd.DataFrame]:
    r0 = main[main["R"].eq(0)].copy()
    groups = {
        "all_r0": r0,
        "scanline_y10": r0[r0["Y"].eq(10)].copy(),
        "scanline_y20": r0[r0["Y"].eq(20)].copy(),
        "column_x10": r0[r0["X"].eq(10)].copy(),
        "column_x20": r0[r0["X"].eq(20)].copy(),
    }
    return {k: v.sort_values("acquisition_order").reset_index(drop=True) for k, v in groups.items() if len(v) >= 8}


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frame-audit-csv", type=Path, default=root / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--alignment-csv", type=Path, default=default_contour_alignment_csv(project_root_path=root))
    parser.add_argument("--data-dir", type=Path, default=root / "data" / "data_raw" / "infrared_avi")
    parser.add_argument("--stage-config", type=Path, default=root / "configs" / "stage_calibration.json")
    parser.add_argument("--output-dir", type=Path, default=root / "output" / "ep05_overlay_alignment")
    parser.add_argument("--crop-size", type=int, default=360)
    parser.add_argument("--edge-percentile", type=float, default=93.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    with open(args.stage_config, encoding="utf-8") as f:
        stage = json.load(f)
    theta_deg = float(stage["theta_deg"])
    pixel_size_um = float(stage["pixel_size_um"])

    main_df = load_main_session(args.frame_audit_csv)
    alignment = pd.read_csv(args.alignment_csv)
    ref_file = str(alignment["reference_file"].dropna().iloc[0])
    ref_row = main_df[main_df["file"].eq(ref_file)].iloc[0]
    affine_fit = fit_filename_affine(alignment, robust=True)
    diagnosis = filename_affine_diagnostics(alignment, affine_fit)
    excluded_from_affine = set(
        diagnosis.loc[diagnosis["residual_gate_outlier"], "file"].astype(str).tolist()
    )
    print(
        "Filename affine robust fit: "
        f"fit_rows={affine_fit.fit_count}, clean_rows={affine_fit.clean_count}, "
        f"median_res={affine_fit.median_residual_px:.4f}px, "
        f"threshold={affine_fit.outlier_threshold_px:.4f}px, "
        f"fit_excluded={list(affine_fit.excluded_files)}, "
        f"affine_score_excluded={sorted(excluded_from_affine)}"
    )
    alignment_lookup = {
        str(row["file"]): (float(row["refined_align_dx_px"]), float(row["refined_align_dy_px"]))
        for _, row in alignment[alignment["success"].astype(str).str.lower().eq("true")].iterrows()
    }

    groups = group_specs(main_df)
    summary_rows = []
    for name, group in groups.items():
        shifts_by_method: dict[str, dict[str, tuple[float, float]]] = {}
        shifts_by_method["no_alignment"] = {str(row["file"]): (0.0, 0.0) for _, row in group.iterrows()}
        shifts_by_method["old_stage_model"] = {
            str(row["file"]): stage_shift(row, ref_row, theta_deg, pixel_size_um)
            for _, row in group.iterrows()
        }
        shifts_by_method["filename_affine_fit"] = {
            str(row["file"]): affine_shift(row, affine_fit.beta_dx, affine_fit.beta_dy)
            for _, row in group.iterrows()
            if str(row["file"]) not in excluded_from_affine
        }
        shifts_by_method["data_driven_contour"] = {
            str(row["file"]): data_driven_shift(row, alignment_lookup)
            for _, row in group.iterrows()
            if str(row["file"]) in alignment_lookup
        }

        results = {}
        for method, shifts in shifts_by_method.items():
            usable = group[group["file"].astype(str).isin(shifts)].copy()
            if usable.empty:
                continue
            result = stack_group(usable, args.data_dir, ref_file, shifts, args.crop_size, args.edge_percentile)
            results[method] = result
            summary_rows.append({
                "group": name,
                "method": method,
                **{k: v for k, v in result.items() if not isinstance(v, np.ndarray)},
            })
        plot_method_grid(results, f"{name}: overlay alignment check", args.output_dir / f"{name}_overlay_grid.png")

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(args.output_dir / "overlay_alignment_summary.csv", index=False)
    print(summary.sort_values(["group", "median_chamfer_px"]).to_string(index=False))


if __name__ == "__main__":
    main()
