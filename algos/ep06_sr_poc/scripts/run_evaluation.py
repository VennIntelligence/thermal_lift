#!/usr/bin/env python3
"""Evaluate EP06 SR outputs and generate direct comparison figures."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, gaussian_filter, laplace, sobel, zoom

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_saa import PROJECT_ROOT  # noqa: E402
from thermal_core.plotting import COLORMAPS, FIGURE_SIZES, METHOD_COLOR_LIST, savefig_academic, setup_academic_style  # noqa: E402


HIGH_METHODS = [
    ("lr_reference", "LR reference", "lr_reference.npy"),
    ("bicubic", "Bicubic reference", "bicubic_reference.npy"),
    ("saa_uniform", "SAA uniform", "saa_uniform_highpass.npy"),
    ("saa_weighted", "SAA weighted", "saa_weighted_highpass.npy"),
    ("ibp", "IBP", "ibp_highpass.npy"),
    ("map_tv", "MAP-TV", "map_tv_highpass.npy"),
]

RAW_METHODS = [
    ("saa_weighted", "SAA weighted raw", "saa_weighted_raw.npy"),
    ("ibp", "IBP raw", "ibp_raw.npy"),
    ("map_tv", "MAP-TV raw", "map_tv_raw.npy"),
]

RAW_REFERENCE_METHODS = [
    ("lr_raw", "LR raw reference", "lr_raw_reference.npy"),
    ("bicubic_raw", "Bicubic raw reference", "bicubic_raw_reference.npy"),
]


def load_array(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing required EP06 output: {path}")
    return np.load(path).astype(np.float32, copy=False)


def robust_limits(images: list[np.ndarray], *, symmetric: bool) -> tuple[float, float]:
    samples = []
    for image in images:
        arr = np.asarray(image, dtype=np.float32)
        step_y = max(1, arr.shape[0] // 512)
        step_x = max(1, arr.shape[1] // 512)
        samples.append(arr[::step_y, ::step_x].ravel())
    values = np.concatenate(samples)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return -1.0, 1.0
    if symmetric:
        limit = float(np.percentile(np.abs(values), 99.0))
        return -limit, limit
    return float(np.percentile(values, 1.0)), float(np.percentile(values, 99.0))


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gx = sobel(image, axis=1, mode="nearest")
    gy = sobel(image, axis=0, mode="nearest")
    return np.hypot(gx, gy).astype(np.float32)


def raw_control_structure(image: np.ndarray, *, sigma: float) -> np.ndarray:
    image = np.asarray(image, dtype=np.float32)
    return (image - gaussian_filter(image, sigma=sigma, mode="nearest")).astype(np.float32)


def artifact_score(image: np.ndarray) -> float:
    image = np.asarray(image, dtype=np.float32)
    if not np.isfinite(image).all():
        return float("inf")
    high_freq = image - gaussian_filter(image, sigma=1.0, mode="nearest")
    lap = laplace(image, mode="nearest")
    base = float(np.std(image))
    if base <= 1e-12:
        return 0.0
    return float((np.std(high_freq) + 0.25 * np.std(lap)) / base)


def nrmse_to(reference: np.ndarray, image: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float32)
    image = np.asarray(image, dtype=np.float32)
    if reference.shape != image.shape:
        reference = zoom(reference, (image.shape[0] / reference.shape[0], image.shape[1] / reference.shape[1]), order=1)
    denom = float(np.std(reference))
    return float(np.sqrt(np.mean((image - reference) ** 2)) / max(denom, 1e-6))


def corr_to(reference: np.ndarray, image: np.ndarray) -> float:
    reference = np.asarray(reference, dtype=np.float32)
    image = np.asarray(image, dtype=np.float32)
    if reference.shape != image.shape:
        reference = zoom(reference, (image.shape[0] / reference.shape[0], image.shape[1] / reference.shape[1]), order=1)
    a = reference - float(np.mean(reference))
    b = image - float(np.mean(image))
    denom = float(np.linalg.norm(a.ravel()) * np.linalg.norm(b.ravel()))
    return float(np.dot(a.ravel(), b.ravel()) / max(denom, 1e-12))


def load_segments(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    table = pd.read_csv(path)
    if {"x_px", "y_px"}.issubset(table.columns):
        if "pass_fail" in table.columns:
            passed = table[table["pass_fail"].astype(str).str.lower().isin({"true", "1", "yes"})].copy()
            if not passed.empty:
                table = passed
        return table.dropna(subset=["x_px", "y_px"]).copy()
    return pd.DataFrame()


def contour_chamfer_lr_px(image: np.ndarray, segments: pd.DataFrame, *, scale: int, edge_percentile: float) -> float:
    if segments.empty:
        return float("nan")
    grad = gradient_magnitude(image)
    threshold = float(np.percentile(grad, edge_percentile))
    edge_mask = grad >= threshold
    distance = distance_transform_edt(~edge_mask)
    xs = np.rint(segments["x_px"].to_numpy(dtype=float) * scale).astype(int)
    ys = np.rint(segments["y_px"].to_numpy(dtype=float) * scale).astype(int)
    valid = (ys >= 0) & (ys < distance.shape[0]) & (xs >= 0) & (xs < distance.shape[1])
    if valid.sum() == 0:
        return float("nan")
    return float(np.mean(distance[ys[valid], xs[valid]]) / scale)


def summarize_method(
    *,
    track: str,
    method: str,
    label: str,
    filename: str,
    image: np.ndarray,
    bicubic: np.ndarray,
    segments: pd.DataFrame,
    scale: int,
    edge_percentile: float,
) -> dict[str, object]:
    grad = gradient_magnitude(image)
    return {
        "track": track,
        "method": method,
        "label": label,
        "file": filename,
        "shape": f"{image.shape[0]}x{image.shape[1]}",
        "finite": bool(np.isfinite(image).all()),
        "min": float(np.nanmin(image)),
        "max": float(np.nanmax(image)),
        "mean": float(np.nanmean(image)),
        "std": float(np.nanstd(image)),
        "mean_gradient": float(np.nanmean(grad)),
        "p95_gradient": float(np.nanpercentile(grad, 95)),
        "edge_density_p95": float(np.mean(grad >= np.nanpercentile(grad, 95))),
        "artifact_score": artifact_score(image),
        "nrmse_to_bicubic": nrmse_to(bicubic, image),
        "corr_to_bicubic": corr_to(bicubic, image),
        "contour_chamfer_lr_px": contour_chamfer_lr_px(image, segments, scale=scale, edge_percentile=edge_percentile),
    }


def show_image(ax: plt.Axes, image: np.ndarray, title: str, *, cmap: str, vmin: float, vmax: float) -> None:
    ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title, fontsize=8.5, pad=2)
    ax.set_xticks([])
    ax.set_yticks([])


def upscale_lr_for_display(lr: np.ndarray, target_shape: tuple[int, int]) -> np.ndarray:
    return zoom(lr, (target_shape[0] / lr.shape[0], target_shape[1] / lr.shape[1]), order=0).astype(np.float32)


def get_structure_center(shape: tuple[int, int]) -> tuple[int, int]:
    scale = shape[0] // 480
    # The gradient centroid center at 2X scale was calculated at (455.08, 614.88)
    # relative to the math center (480, 640), representing a Y/X offset of -25 pixels.
    # We project this offset according to the current scale factor.
    offset = int(-25.0 * scale / 2.0)
    cy = shape[0] // 2 + offset
    cx = shape[1] // 2 + offset
    return cy, cx


def center_zoom_roi(shape: tuple[int, int], *, zoom_factor: float) -> tuple[slice, slice]:
    if zoom_factor <= 1.0:
        return slice(0, shape[0]), slice(0, shape[1])
    size = int(round(min(shape) / zoom_factor))
    return roi_slices(get_structure_center(shape), shape, max(32, size))


def save_fullview(output_dir: Path, high: dict[str, np.ndarray], *, zoom_factor: float) -> None:
    display_images = {
        key: (upscale_lr_for_display(image, high["bicubic"].shape) if key == "lr_reference" else image)
        for key, image in high.items()
    }
    roi = center_zoom_roi(high["bicubic"].shape, zoom_factor=zoom_factor)
    crops = [image[roi] for image in display_images.values()]
    vmin, vmax = robust_limits(crops, symmetric=True)
    setup_academic_style()
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.05), constrained_layout=False)
    for ax, (key, label, _) in zip(axes.ravel(), HIGH_METHODS, strict=True):
        show_image(ax, display_images[key][roi], label, cmap=COLORMAPS["residual_diff"], vmin=vmin, vmax=vmax)
    fig.suptitle(f"EP06 center highpass comparison, {zoom_factor:.1f}x visual zoom", fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.99, bottom=0.03, top=0.86, wspace=0.04, hspace=0.20)
    savefig_academic(fig, output_dir / "comparison_fullview.png")


def roi_slices(center_yx: tuple[int, int], shape: tuple[int, int], size: int) -> tuple[slice, slice]:
    cy, cx = center_yx
    half = size // 2
    y0 = min(max(0, cy - half), max(0, shape[0] - size))
    x0 = min(max(0, cx - half), max(0, shape[1] - size))
    return slice(y0, min(shape[0], y0 + size)), slice(x0, min(shape[1], x0 + size))


def parse_roi_sizes(value: str) -> list[int]:
    sizes = [int(part.strip()) for part in value.split(",") if part.strip()]
    if not sizes:
        raise ValueError("--center-roi-sizes must include at least one integer")
    return sizes


def save_roi_figures(output_dir: Path, high: dict[str, np.ndarray], *, roi_sizes: list[int]) -> None:
    target_shape = high["bicubic"].shape
    display_images = {
        key: (upscale_lr_for_display(image, target_shape) if key == "lr_reference" else image)
        for key, image in high.items()
    }
    vmin, vmax = robust_limits(list(display_images.values()), symmetric=True)
    center = get_structure_center(target_shape)
    for idx, size in enumerate(roi_sizes, start=1):
        if idx > 3:
            break
        roi = roi_slices(center, target_shape, int(size))
        fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.05), constrained_layout=False)
        for ax, (key, label, _) in zip(axes.ravel(), HIGH_METHODS, strict=True):
            show_image(
                ax,
                display_images[key][roi],
                label,
                cmap=COLORMAPS["residual_diff"],
                vmin=vmin,
                vmax=vmax,
            )
        fig.suptitle(f"Center ROI {idx}: direct highpass comparison, {int(size)} HR px crop", fontsize=10, fontweight="bold")
        fig.subplots_adjust(left=0.02, right=0.99, bottom=0.03, top=0.86, wspace=0.04, hspace=0.20)
        savefig_academic(fig, output_dir / f"comparison_roi_{idx}.png")


def save_control_track(
    output_dir: Path,
    high: dict[str, np.ndarray],
    raw: dict[str, np.ndarray],
    *,
    sigma: float,
    zoom_factor: float,
) -> None:
    raw_struct = {key: raw_control_structure(value, sigma=sigma) for key, value in raw.items()}
    columns = [
        ("saa_weighted", "SAA weighted"),
        ("ibp", "IBP"),
        ("map_tv", "MAP-TV"),
    ]
    roi = center_zoom_roi(high["bicubic"].shape, zoom_factor=zoom_factor)
    values = [high[key][roi] for key, _ in columns] + [raw_struct[key][roi] for key, _ in columns]
    vmin, vmax = robust_limits(values, symmetric=True)
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 3.7), constrained_layout=False)
    for col, (key, label) in enumerate(columns):
        show_image(axes[0, col], high[key][roi], label, cmap=COLORMAPS["residual_diff"], vmin=vmin, vmax=vmax)
        show_image(axes[1, col], raw_struct[key][roi], "", cmap=COLORMAPS["residual_diff"], vmin=vmin, vmax=vmax)
    fig.text(0.025, 0.63, "Highpass\ninput", ha="center", va="center", rotation=90, fontsize=8)
    fig.text(0.025, 0.25, "Raw\ncontrol", ha="center", va="center", rotation=90, fontsize=8)
    fig.suptitle(f"Main track vs raw-temperature control, {zoom_factor:.1f}x visual zoom", fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.03, top=0.86, wspace=0.04, hspace=0.02)
    savefig_academic(fig, output_dir / "comparison_control_track.png")


def save_center_raw_temperature(
    output_dir: Path,
    raw_refs: dict[str, np.ndarray],
    raw: dict[str, np.ndarray],
    *,
    center_size: int,
) -> None:
    """Save a center crop in ordinary raw-temperature visual space.

    The raw-control arrays are offset-corrected temperature reconstructions.
    This figure deliberately avoids highpass/residual styling so the central
    chip structures can be inspected as a conventional temperature image.
    """
    columns = [
        ("lr_raw", "LR raw\nreference"),
        ("bicubic_raw", "Bicubic raw\nreference"),
        ("saa_uniform", "SAA uniform"),
        ("saa_weighted", "SAA weighted"),
        ("ibp", "IBP"),
        ("map_tv", "MAP-TV"),
    ]
    reference = raw["saa_weighted"]
    target_shape = reference.shape
    display_images = {
        "lr_raw": upscale_lr_for_display(raw_refs["lr_raw"], target_shape),
        "bicubic_raw": raw_refs["bicubic_raw"],
        "saa_uniform": raw["saa_uniform"],
        "saa_weighted": raw["saa_weighted"],
        "ibp": raw["ibp"],
        "map_tv": raw["map_tv"],
    }
    roi = roi_slices(get_structure_center(reference.shape), reference.shape, center_size)
    crops = [display_images[key][roi] for key, _ in columns]
    vmin, vmax = robust_limits(crops, symmetric=False)
    fig, axes = plt.subplots(2, 3, figsize=(7.1, 4.35), constrained_layout=False)
    image_handle = None
    for ax, (key, label), crop in zip(axes.ravel(), columns, crops, strict=True):
        image_handle = ax.imshow(
            crop,
            cmap=COLORMAPS["temperature"],
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
        )
        ax.set_title(label, fontsize=8.5, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    if image_handle is not None:
        cax = fig.add_axes([0.92, 0.16, 0.018, 0.64])
        colorbar = fig.colorbar(image_handle, cax=cax)
        colorbar.set_label("offset-corrected temperature (C)")
    fig.suptitle("Center raw-temperature crop, ordinary visual check", fontsize=10, fontweight="bold")
    fig.subplots_adjust(left=0.02, right=0.90, bottom=0.03, top=0.84, wspace=0.04, hspace=0.22)
    savefig_academic(fig, output_dir / "comparison_center_raw_temperature.png")


def save_gradient_figure(output_dir: Path, summary: pd.DataFrame) -> None:
    high = summary[summary["track"].eq("highpass") & ~summary["method"].eq("lr_reference")].copy()
    x = np.arange(len(high))
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["double_col"], constrained_layout=True)
    axes[0].bar(x, high["mean_gradient"], color=METHOD_COLOR_LIST[: len(high)])
    axes[0].set_title("Mean Gradient")
    axes[0].set_xticks(x, high["label"], rotation=30, ha="right")
    axes[0].set_ylabel("gradient magnitude")
    axes[1].bar(x, high["p95_gradient"], color=METHOD_COLOR_LIST[: len(high)])
    axes[1].set_title("P95 Gradient")
    axes[1].set_xticks(x, high["label"], rotation=30, ha="right")
    axes[1].set_ylabel("gradient magnitude")
    savefig_academic(fig, output_dir / "gradient_magnitude_comparison.png")


def save_split_half_figure(output_dir: Path) -> None:
    path = output_dir / "map_tv_lambda_selection.csv"
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["one_half_col"], constrained_layout=True)
    if path.exists():
        table = pd.read_csv(path)
        for idx, (track, group) in enumerate(table.groupby("track")):
            color = METHOD_COLOR_LIST[idx % len(METHOD_COLOR_LIST)]
            group = group.sort_values("lambda_tv")
            ax.semilogx(group["lambda_tv"], group["split_half_nrmse"], marker="o", color=color, label=track)
            selected = group[group["selected"].astype(bool)]
            if not selected.empty:
                ax.scatter(selected["lambda_tv"], selected["split_half_nrmse"], s=70, facecolors="none", edgecolors=color)
        ax.set_xlabel("lambda TV")
        ax.set_ylabel("split-half NRMSE")
        ax.set_title("MAP-TV Lambda Selection")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "map_tv_lambda_selection.csv not found", ha="center", va="center")
        ax.set_axis_off()
    savefig_academic(fig, output_dir / "split_half_consistency.png")


def save_artifact_figure(output_dir: Path, summary: pd.DataFrame) -> None:
    high = summary[summary["track"].eq("highpass") & ~summary["method"].eq("lr_reference")].copy()
    x = np.arange(len(high))
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["double_col"], constrained_layout=True)
    axes[0].bar(x, high["artifact_score"], color=METHOD_COLOR_LIST[: len(high)])
    axes[0].set_title("Artifact Score")
    axes[0].set_xticks(x, high["label"], rotation=30, ha="right")
    axes[0].set_ylabel("relative high-frequency residual")
    axes[1].bar(x, high["contour_chamfer_lr_px"], color=METHOD_COLOR_LIST[: len(high)])
    axes[1].set_title("Contour Chamfer Proxy")
    axes[1].set_xticks(x, high["label"], rotation=30, ha="right")
    axes[1].set_ylabel("LR pixels")
    savefig_academic(fig, output_dir / "artifact_audit.png")


def load_json_metric(path: Path, key: str) -> float:
    if not path.exists():
        return float("nan")
    try:
        return float(json.loads(path.read_text(encoding="utf-8")).get(key, np.nan))
    except (json.JSONDecodeError, TypeError, ValueError):
        return float("nan")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "ep06_sr_poc")
    parser.add_argument("--segment-summary-csv", type=Path, default=PROJECT_ROOT / "output" / "ep04_global_validation" / "segment_summary.csv")
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--edge-percentile", type=float, default=95.0)
    parser.add_argument("--overview-zoom", type=float, default=6.0, help="Center crop zoom for main highpass/control figures")
    parser.add_argument("--center-roi-sizes", default="160,112,80", help="Comma-separated center highpass ROI sizes in HR pixels")
    parser.add_argument("--center-raw-size", type=int, default=160, help="Center raw-temperature crop size in HR pixels")
    parser.add_argument("--raw-visual-sigma", type=float, default=10.0, help="Gaussian sigma on HR raw-control images before visual subtraction")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir = args.output_dir.resolve()
    if args.scale not in (2, 4):
        raise ValueError("EP06 is a 2x/4x contour-level POC; keep --scale 2 or 4.")
    setup_academic_style()

    high = {key: load_array(args.output_dir / filename) for key, _, filename in HIGH_METHODS}
    raw = {key: load_array(args.output_dir / filename) for key, _, filename in RAW_METHODS}
    raw["saa_uniform"] = load_array(args.output_dir / "saa_uniform_raw.npy")
    raw_refs = {key: load_array(args.output_dir / filename) for key, _, filename in RAW_REFERENCE_METHODS}
    segments = load_segments(args.segment_summary_csv)
    bicubic = high["bicubic"]

    rows: list[dict[str, object]] = []
    for key, label, filename in HIGH_METHODS:
        image = high[key]
        if key == "lr_reference":
            image = upscale_lr_for_display(image, bicubic.shape)
        rows.append(
            summarize_method(
                track="highpass",
                method=key,
                label=label,
                filename=filename,
                image=image,
                bicubic=bicubic,
                segments=segments,
                scale=args.scale,
                edge_percentile=args.edge_percentile,
            )
        )
    for key, label, filename in RAW_METHODS:
        image = raw_control_structure(raw[key], sigma=args.raw_visual_sigma)
        rows.append(
            summarize_method(
                track="raw_control_highpass_visual",
                method=key,
                label=label,
                filename=filename,
                image=image,
                bicubic=bicubic,
                segments=segments,
                scale=args.scale,
                edge_percentile=args.edge_percentile,
            )
        )
    summary = pd.DataFrame(rows)
    summary["std_ratio_to_lr"] = np.nan
    for track, group in summary.groupby("track"):
        lr_ref = group[group["method"].eq("lr_reference")]
        if lr_ref.empty:
            continue
        ref_std = float(lr_ref["std"].iloc[0])
        if ref_std > 0:
            summary.loc[group.index, "std_ratio_to_lr"] = group["std"].astype(float) / ref_std
    summary["saa_uniform_synthetic_psnr_db"] = load_json_metric(args.output_dir / "saa_synthetic_validation.json", "uniform_psnr_db")
    summary["ibp_synthetic_psnr_db"] = load_json_metric(args.output_dir / "ibp_synthetic_validation.json", "ibp_psnr_db")
    summary["map_tv_synthetic_psnr_db"] = load_json_metric(args.output_dir / "map_tv_synthetic_validation.json", "map_tv_psnr_db")
    summary.to_csv(args.output_dir / "evaluation_summary.csv", index=False)

    save_fullview(args.output_dir, high, zoom_factor=args.overview_zoom)
    save_roi_figures(args.output_dir, high, roi_sizes=parse_roi_sizes(args.center_roi_sizes))
    save_control_track(args.output_dir, high, raw, sigma=args.raw_visual_sigma, zoom_factor=args.overview_zoom)
    save_center_raw_temperature(args.output_dir, raw_refs, raw, center_size=args.center_raw_size)
    save_gradient_figure(args.output_dir, summary)
    save_split_half_figure(args.output_dir)
    save_artifact_figure(args.output_dir, summary)

    print(f"Saved evaluation summary and figures to {args.output_dir.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
