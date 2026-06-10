"""EP08 notebook cache — Stage 3 matplotlib figures built offline."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter

from thermal_core.notebook_cache import (
    project_root,
    require_artifacts,
    write_manifest,
)
from thermal_core.plotting import METHOD_COLOR_LIST, savefig_academic, setup_academic_style

EP08_CACHE_VERSION = 1
REBUILD_COMMAND = "uv run python scripts/build_ep08_cache.py"

EP08_FIGURE_ARTIFACTS = (
    "stage3_progressive_metrics.png",
    "stage3_visual_comparison.png",
    "stage3_aspect_ablation.png",
)

STAGE3_METHOD_ORDER = ["ep06_map_tv", "siren", "wire", "deepinv_dip"]
STAGE3_METHOD_LABELS = {
    "ep06_map_tv": "EP06 MAP-TV\n(classic baseline)",
    "siren": "SIREN\n(sin activation INR)",
    "wire": "WIRE\n(Gabor activation INR)",
    "deepinv_dip": "DeepInverse-DIP\n(CNN decoder prior)",
}
CROP_H, CROP_W = 400, 520
TARGET_FRAMES = 255
ASPECT_METRIC_COLS = [
    "holdout_residual",
    "split_half_nrmse",
    "artifact_score",
    "raw_control_agreement",
    "p95_gradient",
]
TREND_METRICS = [
    ("holdout_residual", "Hold-out residual", "lower"),
    ("split_half_nrmse", "Split-half NRMSE", "lower"),
    ("artifact_score", "Artifact score", "lower"),
    ("raw_control_agreement", "Raw-control agreement", "higher"),
    ("p95_gradient", "P95 gradient", "proxy"),
]


@dataclass(frozen=True)
class Ep08Cache:
    output_dir: Path
    stage3_dir: Path
    manifest: dict

    def figure_path(self, name: str) -> Path:
        return self.output_dir / name


def _relative(root: Path, path: Path) -> str:
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _parse_run_name(run_dir: Path) -> dict:
    name = run_dir.name
    parts = name.split("_")
    aspect = parts[-1] if parts and parts[-1] in {"preserve", "stretch"} else None
    if aspect is not None:
        parts = parts[:-1]
    patch = parts[-1] if parts else None
    if patch is not None:
        parts = parts[:-1]
    n_frames = None
    if parts and parts[-1].isdigit():
        n_frames = int(parts[-1])
        parts = parts[:-1]
    method = "_".join(parts) if parts else None
    return {
        "run": name,
        "method_from_dir": method,
        "n_frames_from_dir": n_frames,
        "patch_shape_label": patch,
        "coord_aspect_mode_from_dir": aspect,
        "aspect_from_dir": aspect,
    }


def collect_stage3_metrics(stage3_dir: Path, *, aspect_only: bool = False) -> pd.DataFrame:
    rows = []
    for metrics_path in sorted(stage3_dir.glob("*/metrics.json")):
        parsed = _parse_run_name(metrics_path.parent)
        if aspect_only and parsed.get("aspect_from_dir") not in {"preserve", "stretch"}:
            continue
        payload = _read_json(metrics_path)
        if not payload:
            continue
        row = dict(payload)
        row.update(parsed)
        row["run_dir"] = metrics_path.parent
        row["method"] = row.get("method") or parsed["method_from_dir"]
        row["n_frames"] = row.get("n_frames") or parsed["n_frames_from_dir"]
        row["coord_aspect_mode"] = (
            row.get("coord_aspect_mode")
            or parsed.get("coord_aspect_mode_from_dir")
            or parsed.get("aspect_from_dir")
        )
        row["stage_gate"] = row.get("stage_gate") or row.get("stage1_gate")
        rows.append(row)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    for col in (
        "n_frames",
        "train_frame_count",
        "val_frame_count",
        "holdout_residual",
        "split_half_nrmse",
        "artifact_score",
        "raw_control_agreement",
        "p95_gradient",
        "best_step",
        "final_step",
        "elapsed_sec",
    ):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    sort_cols = ["method", "coord_aspect_mode", "n_frames", "run"]
    return df.sort_values(sort_cols, na_position="last")


def _load_npy(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.squeeze(np.asarray(np.load(path), dtype=np.float32))


def _find_chip_center(hp: np.ndarray, window: int = 100) -> tuple[int, int]:
    energy = np.nan_to_num(hp, nan=0.0) ** 2
    smoothed = uniform_filter(energy, size=window, mode="constant")
    peak = np.unravel_index(np.argmax(smoothed), smoothed.shape)
    return int(peak[0]), int(peak[1])


def _crop(img: np.ndarray, cy: int, cx: int, h: int = CROP_H, w: int = CROP_W) -> np.ndarray:
    r0 = max(0, min(cy - h // 2, img.shape[0] - h))
    c0 = max(0, min(cx - w // 2, img.shape[1] - w))
    return img[r0 : r0 + h, c0 : c0 + w]


def _sym_lim(images: list, pct: float = 99.0) -> float:
    vals = [np.abs(im.ravel()) for im in images if im is not None]
    if not vals:
        return 1.0
    cat = np.concatenate(vals)
    finite = cat[np.isfinite(cat)]
    lim = float(np.percentile(finite, pct)) if finite.size else 1.0
    return lim if lim > 0 else 1.0


def plot_stage3_progressive(stage3_metrics: pd.DataFrame, output_dir: Path) -> bool:
    if stage3_metrics.empty:
        return False
    available = [
        item
        for item in TREND_METRICS
        if item[0] in stage3_metrics.columns
        and pd.to_numeric(stage3_metrics[item[0]], errors="coerce").notna().any()
    ]
    plot_df = stage3_metrics.dropna(subset=["n_frames"]).copy()
    if not available or plot_df.empty:
        return False
    plot_df["plot_label"] = (
        plot_df["method"].fillna("unknown").astype(str)
        + " / "
        + plot_df["coord_aspect_mode"].fillna("?").astype(str)
    )
    ncols = 2
    nrows = int(math.ceil(len(available) / ncols))
    setup_academic_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, max(2.4, 2.2 * nrows)), squeeze=False)
    for ax in axes.ravel()[len(available) :]:
        ax.axis("off")
    for idx, (metric, label, direction) in enumerate(available):
        ax = axes.ravel()[idx]
        plotted = False
        for group_idx, (group_label, group) in enumerate(plot_df.groupby("plot_label", sort=True)):
            values = pd.to_numeric(group[metric], errors="coerce")
            keep = values.notna() & group["n_frames"].notna()
            if not keep.any():
                continue
            group_plot = group.loc[keep].assign(_value=values.loc[keep]).sort_values("n_frames")
            ax.plot(
                group_plot["n_frames"],
                group_plot["_value"],
                marker="o",
                color=METHOD_COLOR_LIST[group_idx % len(METHOD_COLOR_LIST)],
                label=group_label,
            )
            plotted = True
        ax.set_title(f"{label} ({direction})")
        ax.set_xlabel("Input frames")
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
        finite_values = pd.to_numeric(plot_df[metric], errors="coerce").dropna()
        if metric in {"holdout_residual", "split_half_nrmse", "artifact_score", "p95_gradient"} and (
            finite_values > 0
        ).all():
            ax.set_yscale("log")
        if plotted:
            ax.legend(fontsize=7)
    savefig_academic(fig, output_dir / "stage3_progressive_metrics.png")
    return True


def plot_stage3_visual_comparison(stage3_dir: Path, output_dir: Path) -> bool:
    runs: dict[str, dict] = {}
    for method in STAGE3_METHOD_ORDER:
        d = stage3_dir / f"{method}_{TARGET_FRAMES:03d}_full_preserve"
        if not (d / "metrics.json").exists():
            continue
        hr = _load_npy(d / "hr_image.npy")
        raw = _load_npy(d / "hr_raw_control.npy")
        sa = _load_npy(d / "split_half_a.npy")
        sb = _load_npy(d / "split_half_b.npy")
        runs[method] = {
            "metrics": _read_json(d / "metrics.json"),
            "hr": hr,
            "raw": raw,
            "split_diff": (sa - sb) if sa is not None and sb is not None else None,
        }
    if not runs:
        return False
    ref = runs.get("ep06_map_tv") or next(iter(runs.values()))
    cy, cx = _find_chip_center(ref["hr"])
    cropped = {}
    for method, data in runs.items():
        cropped[method] = {
            "hr": _crop(data["hr"], cy, cx) if data["hr"] is not None else None,
            "raw": _crop(data["raw"], cy, cx) if data["raw"] is not None else None,
            "split": _crop(data["split_diff"], cy, cx) if data["split_diff"] is not None else None,
        }
    hr_lim = _sym_lim([c["hr"] for c in cropped.values()])
    raw_lim = _sym_lim([c["raw"] for c in cropped.values()])
    split_lim = _sym_lim([c["split"] for c in cropped.values()])
    ordered = [m for m in STAGE3_METHOD_ORDER if m in runs]
    setup_academic_style()
    fig, axes = plt.subplots(len(ordered), 3, figsize=(11, 2.8 * len(ordered)), squeeze=False)
    col_titles = ["HR Highpass", "Raw-control (bicubic ref)", "Split-half A − B"]
    col_specs = [("hr", "RdBu_r", hr_lim), ("raw", "RdBu_r", raw_lim), ("split", "RdBu_r", split_lim)]
    for row, method in enumerate(ordered):
        c = cropped[method]
        m = runs[method]["metrics"]
        label = STAGE3_METHOD_LABELS.get(method, method)
        for col, (key, cmap, lim) in enumerate(col_specs):
            ax = axes[row, col]
            img = c[key]
            if img is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
            else:
                ax.imshow(img, cmap=cmap, vmin=-lim, vmax=lim, origin="upper", aspect="equal")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(col_titles[col], fontsize=11)
            if col == 0:
                ax.set_ylabel(label, fontsize=9, rotation=0, ha="right", va="center", labelpad=80)
        sh = m.get("split_half_nrmse", float("nan"))
        art = m.get("artifact_score", float("nan"))
        rc = m.get("raw_control_agreement", float("nan"))
        ho = m.get("holdout_residual", float("nan"))
        txt = (
            f"split-half  {sh:.3f}\n"
            f"artifact    {art:.3f}\n"
            f"raw-ctrl    {rc:.3f}\n"
            f"holdout     {ho:.3f}"
        )
        axes[row, 2].text(
            1.02,
            0.5,
            txt,
            transform=axes[row, 2].transAxes,
            fontsize=8,
            va="center",
            ha="left",
            family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85, ec="0.7"),
        )
    fig.tight_layout(rect=[0.18, 0, 0.88, 0.96])
    savefig_academic(fig, output_dir / "stage3_visual_comparison.png")
    return True


def _build_aspect_pairs(aspect_metrics: pd.DataFrame) -> pd.DataFrame:
    pairs = []
    for key, group in aspect_metrics.groupby(["method", "n_frames", "patch_shape_label"], dropna=False, sort=True):
        preserve = group[group["coord_aspect_mode"].eq("preserve")]
        stretch = group[group["coord_aspect_mode"].eq("stretch")]
        if preserve.empty or stretch.empty:
            continue
        preserve_row = preserve.iloc[-1]
        stretch_row = stretch.iloc[-1]
        row = {
            "method": key[0],
            "n_frames": key[1],
            "patch_shape_label": key[2],
            "preserve_run": preserve_row["run"],
            "stretch_run": stretch_row["run"],
        }
        for metric in ASPECT_METRIC_COLS:
            if metric not in aspect_metrics.columns:
                continue
            row[f"{metric}_preserve"] = preserve_row.get(metric)
            row[f"{metric}_stretch"] = stretch_row.get(metric)
            row[f"{metric}_preserve_minus_stretch"] = preserve_row.get(metric) - stretch_row.get(metric)
        pairs.append(row)
    return pd.DataFrame(pairs)


def plot_stage3_aspect_ablation(aspect_metrics: pd.DataFrame, output_dir: Path) -> bool:
    aspect_pairs = _build_aspect_pairs(aspect_metrics)
    if aspect_pairs.empty:
        return False
    plot_pairs = aspect_pairs.sort_values(["n_frames", "method"], ascending=[False, True]).head(6).copy()
    plot_pairs["label"] = (
        plot_pairs["method"].astype(str)
        + " "
        + plot_pairs["n_frames"].astype("Int64").astype(str)
        + "f"
    )
    available = [
        metric
        for metric in ASPECT_METRIC_COLS
        if f"{metric}_preserve" in plot_pairs.columns
        and pd.to_numeric(plot_pairs[f"{metric}_preserve"], errors="coerce").notna().any()
        and pd.to_numeric(plot_pairs[f"{metric}_stretch"], errors="coerce").notna().any()
    ]
    if not available:
        return False
    ncols = 2
    nrows = int(math.ceil(len(available) / ncols))
    setup_academic_style()
    fig, axes = plt.subplots(nrows, ncols, figsize=(7.2, max(2.4, 2.2 * nrows)), squeeze=False)
    x = np.arange(len(plot_pairs))
    width = 0.36
    for ax in axes.ravel()[len(available) :]:
        ax.axis("off")
    for idx, metric in enumerate(available):
        ax = axes.ravel()[idx]
        preserve_values = pd.to_numeric(plot_pairs[f"{metric}_preserve"], errors="coerce").to_numpy(dtype=float)
        stretch_values = pd.to_numeric(plot_pairs[f"{metric}_stretch"], errors="coerce").to_numpy(dtype=float)
        ax.bar(x - width / 2, preserve_values, width, label="preserve", color="#4C72B0")
        ax.bar(x + width / 2, stretch_values, width, label="stretch", color="#C44E52")
        ax.set_title(metric)
        ax.set_xticks(x)
        ax.set_xticklabels(plot_pairs["label"], rotation=25, ha="right")
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=7)
    savefig_academic(fig, output_dir / "stage3_aspect_ablation.png")
    return True


def _figures_cache_up_to_date(output_dir: Path) -> bool:
    manifest_path = output_dir / "cache_manifest.json"
    if not manifest_path.exists():
        return False
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    figures_built = manifest.get("figures_built", [])
    if not figures_built:
        return False
    return all((output_dir / name).exists() for name in figures_built)


def build_ep08_cache(
    *,
    project_root_path: Path | None = None,
    output_dir: Path | None = None,
    force: bool = False,
) -> Ep08Cache:
    root = project_root(project_root_path)
    output_dir = (output_dir or root / "output" / "ep08_inr_sr").resolve()
    stage3_dir = output_dir / "stage3"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not force and _figures_cache_up_to_date(output_dir):
        return load_ep08_cache(project_root_path=root, output_dir=output_dir)

    setup_academic_style()
    stage3_metrics = collect_stage3_metrics(stage3_dir)
    aspect_metrics = collect_stage3_metrics(stage3_dir, aspect_only=True)

    built: list[str] = []
    if plot_stage3_progressive(stage3_metrics, output_dir):
        built.append("stage3_progressive_metrics.png")
    if plot_stage3_visual_comparison(stage3_dir, output_dir):
        built.append("stage3_visual_comparison.png")
    if plot_stage3_aspect_ablation(aspect_metrics, output_dir):
        built.append("stage3_aspect_ablation.png")

    manifest = write_manifest(
        output_dir,
        version=EP08_CACHE_VERSION,
        artifacts=[*built, "cache_manifest.json"],
        rebuild_command=REBUILD_COMMAND,
        extra={
            "built_at_utc": datetime.now(timezone.utc).isoformat(),
            "stage3_runs": int(len(list(stage3_dir.glob("*/metrics.json")))) if stage3_dir.exists() else 0,
            "figures_built": built,
            "expected_figures": list(EP08_FIGURE_ARTIFACTS),
        },
    )
    return Ep08Cache(output_dir=output_dir, stage3_dir=stage3_dir, manifest=manifest)


def load_ep08_cache(
    *,
    project_root_path: Path | None = None,
    output_dir: Path | None = None,
    require_figures: bool = False,
) -> Ep08Cache:
    root = project_root(project_root_path)
    output_dir = (output_dir or root / "output" / "ep08_inr_sr").resolve()
    if require_figures:
        require_artifacts(output_dir, EP08_FIGURE_ARTIFACTS, rebuild_command=REBUILD_COMMAND)
    manifest_path = output_dir / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return Ep08Cache(output_dir=output_dir, stage3_dir=output_dir / "stage3", manifest=manifest)
