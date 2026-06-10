"""EP10 notebook cache — three-algorithm comparison figures built offline."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage

from thermal_core.notebook_cache import (
    cache_is_complete,
    project_root,
    require_artifacts,
    write_manifest,
)
from thermal_core.plotting import COLORMAPS, METHOD_COLOR_LIST, savefig_academic, setup_academic_style

EP10_CACHE_VERSION = 1
REBUILD_COMMAND = "uv run python scripts/build_ep10_cache.py"

EP10_FIGURE_ARTIFACTS = (
    "core_metric_comparison.png",
    "center_roi_highpass_comparison.png",
    "auxiliary_control_views.png",
    "intermediate_parameter_diagnostics.png",
    "split_holdout_distribution_diagnostics.png",
    "drizzle_4x_diagnostics.png",
    "drizzle_2x_vs_4x_center_third_crop.png",
)

COMMON_METRICS = ["split_half_nrmse", "holdout_mse", "artifact_score", "raw_control_corr"]
INPUT_FRAME_COLUMNS = ("input_frame_count", "n_input_frames", "n_frames")
LOWER_IS_BETTER = {"split_half_nrmse", "holdout_mse", "artifact_score"}
METHOD_ORDER = ["Drizzle", "MAP-TV", "TGV"]


@dataclass(frozen=True)
class Ep10Cache:
    project_root: Path
    output_dir: Path
    report_dir: Path
    ep10_dirs: dict[str, Path]
    artifacts: dict
    sweeps: dict[str, pd.DataFrame]
    best_rows: dict[str, pd.Series | None]
    summary_table: pd.DataFrame
    all_candidates: pd.DataFrame
    status_table: pd.DataFrame
    manifest: dict

    def figure_path(self, name: str) -> Path:
        return self.output_dir / name


def _bootstrap_ep06_metrics(root: Path) -> None:
    ep06_src = root / "algos" / "ep06_sr_poc" / "src"
    if str(ep06_src) not in sys.path:
        sys.path.insert(0, str(ep06_src))


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_npy(path: Path) -> np.ndarray | None:
    if not path.exists():
        return None
    return np.load(path).astype(np.float32, copy=False)


def _resolve_artifact_path(root: Path, value: object) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else root / path


def _infer_input_frame_count(ep10_dir: Path) -> int | None:
    input_summary = _read_json(ep10_dir / "cache" / "input_summary.json")
    frames_shape = input_summary.get("frames_shape")
    if isinstance(frames_shape, list) and frames_shape:
        return int(frames_shape[0])

    metadata_path = ep10_dir / "cache" / "metadata.csv"
    metadata = _read_csv(metadata_path)
    if not metadata.empty and "is_sr_usable" in metadata.columns:
        usable = metadata["is_sr_usable"].astype(str).str.lower().eq("true")
        return int(usable.sum())
    return None


def pf_token(value: float) -> str:
    return f"{value:.1f}"


def param_token(value: float) -> str:
    return f"{value:g}".replace(".", "p")


def center_roi(image: np.ndarray, size: int = 320) -> np.ndarray:
    rows, cols = image.shape
    size = min(size, rows, cols)
    y0 = max(0, rows // 2 - size // 2)
    x0 = max(0, cols // 2 - size // 2)
    return image[y0 : y0 + size, x0 : x0 + size]


def center_fraction_crop(image: np.ndarray, fraction: float = 1.0 / 3.0) -> np.ndarray:
    rows, cols = image.shape
    crop_rows = max(1, int(round(rows * fraction)))
    crop_cols = max(1, int(round(cols * fraction)))
    y0 = max(0, rows // 2 - crop_rows // 2)
    x0 = max(0, cols // 2 - crop_cols // 2)
    return image[y0 : y0 + crop_rows, x0 : x0 + crop_cols]


def metric_label(metric: str) -> str:
    labels = {
        "split_half_nrmse": "Split-half NRMSE",
        "holdout_mse": "Holdout MSE",
        "artifact_score": "Artifact score",
        "raw_control_corr": "Raw-control corr.",
    }
    return labels.get(metric, metric)


def normalize_sweep(method: str, df: pd.DataFrame, ep10_dirs: dict[str, Path], root: Path) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()
    _bootstrap_ep06_metrics(root)
    from common.metrics import artifact_score  # noqa: WPS433

    out = df.copy()
    if method == "Drizzle":
        out["method"] = method
        out["variant"] = out["pixfrac"].map(lambda x: f"pf={float(x):.1f}")
        out["parameter_summary"] = out["variant"]
        if "artifact_score_with_lr_overshoot" not in out.columns and "artifact_score" in out.columns:
            out["artifact_score_with_lr_overshoot"] = out["artifact_score"]
        unified_scores = []
        for pixfrac in out["pixfrac"]:
            path = ep10_dirs["drizzle"] / f"drizzle_pf{float(pixfrac):.1f}_hr.npy"
            if path.exists():
                unified_scores.append(float(artifact_score(np.load(path))))
            else:
                unified_scores.append(np.nan)
        if np.isfinite(unified_scores).any():
            out["artifact_score"] = unified_scores
            out["artifact_score_note"] = "recomputed_without_lr_overshoot"
    elif method == "MAP-TV":
        out["method"] = method
        out["variant"] = out.apply(lambda r: f"lambda={r['lambda_tv']:g}, sigma={r['psf_sigma']:g}", axis=1)
        out["parameter_summary"] = out["variant"]
        if not any(column in out.columns for column in INPUT_FRAME_COLUMNS):
            input_count = _infer_input_frame_count(ep10_dirs["map_tv"])
            if input_count is not None:
                out["input_frame_count"] = input_count
    elif method == "TGV":
        if "method" in out.columns:
            out["source_method"] = out["method"]
            out = out[out["source_method"].eq("map_tgv")].copy()
        out["method"] = method
        out["variant"] = out.get("label", pd.Series(index=out.index, dtype=object)).fillna("TGV")
        out["parameter_summary"] = out.apply(
            lambda r: f"lambda={r['lambda_tv']:g}, sigma={r['psf_sigma']:g}", axis=1
        )
        if "split_half_nrmse" not in out and "split_half_nrmse_median" in out:
            out["split_half_nrmse"] = out["split_half_nrmse_median"]
        if "split_half_corr" not in out and "split_half_corr_median" in out:
            out["split_half_corr"] = out["split_half_corr_median"]
    return out


def select_best(df: pd.DataFrame) -> pd.Series | None:
    if df.empty:
        return None
    sort_cols = [c for c in ["split_half_nrmse", "artifact_score", "holdout_mse"] if c in df.columns]
    if not sort_cols:
        return df.iloc[0]
    return df.sort_values(sort_cols, na_position="last").iloc[0]


def _series_to_summary(row: pd.Series | None) -> dict:
    if row is None:
        return {}
    payload = {
        "method": row.get("method"),
        "variant": row.get("variant"),
        "parameter_summary": row.get("parameter_summary"),
    }
    for key in INPUT_FRAME_COLUMNS:
        if key in row.index and pd.notna(row.get(key)):
            payload["input_frame_count"] = int(row.get(key))
            break
    for metric in COMMON_METRICS:
        payload[metric] = row.get(metric, np.nan)
    for key in ["pixfrac", "lambda_tv", "psf_sigma", "label", "hr_cache_file"]:
        if key in row.index:
            payload[key] = row.get(key)
    return payload


def rank_table(df: pd.DataFrame, n: int = 3) -> pd.DataFrame:
    if df.empty:
        return df
    sort_cols = [c for c in ["split_half_nrmse", "artifact_score", "holdout_mse"] if c in df.columns]
    cols = ["method", "variant", *COMMON_METRICS]
    ranked = df.sort_values(sort_cols, na_position="last").head(n) if sort_cols else df.head(n)
    return ranked[[c for c in cols if c in ranked.columns]]


def build_summary_table(best_rows: dict[str, pd.Series | None]) -> pd.DataFrame:
    rows = [_series_to_summary(best_rows.get(method)) for method in METHOD_ORDER]
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    display_cols = ["method", "variant", "input_frame_count", *COMMON_METRICS]
    return table[[c for c in display_cols if c in table.columns]]


def drizzle_image_path(row: pd.Series | None, ep10_dirs: dict[str, Path], suffix: str = "hr") -> Path | None:
    if row is None or pd.isna(row.get("pixfrac", np.nan)):
        return None
    return ep10_dirs["drizzle"] / f"drizzle_pf{pf_token(float(row['pixfrac']))}_{suffix}.npy"


def map_tv_image_path(row: pd.Series | None, ep10_dirs: dict[str, Path], root: Path) -> Path | None:
    if row is None:
        return None
    if "hr_highpass_file" in row.index:
        path = _resolve_artifact_path(root, row.get("hr_highpass_file"))
        if path is not None and path.exists():
            return path
    if "hr_cache_file" in row.index:
        path = _resolve_artifact_path(root, row.get("hr_cache_file"))
        if path is not None and path.exists():
            return path
    if pd.isna(row.get("lambda_tv", np.nan)) or pd.isna(row.get("psf_sigma", np.nan)):
        return None
    legacy_path = ep10_dirs["map_tv"] / f"hr_highpass_{param_token(float(row['lambda_tv']))}_{param_token(float(row['psf_sigma']))}.npy"
    return legacy_path if legacy_path.exists() else None


def tgv_image_path(row: pd.Series | None, ep10_dirs: dict[str, Path], artifacts: dict) -> Path | None:
    if row is None:
        return None
    best_path = ep10_dirs["tgv"] / "best_hr_highpass.npy"
    run_summary = artifacts["tgv"].get("run_summary", {})
    if run_summary.get("best_label") == row.get("label") and best_path.exists():
        return best_path
    cache_file = row.get("hr_cache_file") if "hr_cache_file" in row.index else None
    path = _resolve_artifact_path(ep10_dirs["tgv"].parents[1], cache_file)
    if path is not None and path.exists():
        return path
    return None


def load_candidate_images(
    best_rows: dict[str, pd.Series | None],
    *,
    ep10_dirs: dict[str, Path],
    root: Path,
    artifacts: dict,
) -> list[dict]:
    specs = [
        ("Drizzle", drizzle_image_path(best_rows.get("Drizzle"), ep10_dirs)),
        ("MAP-TV", map_tv_image_path(best_rows.get("MAP-TV"), ep10_dirs, root)),
        ("TGV", tgv_image_path(best_rows.get("TGV"), ep10_dirs, artifacts)),
    ]
    images = []
    for method, path in specs:
        if path is None:
            continue
        image = _load_npy(path)
        if image is None:
            continue
        row = best_rows.get(method)
        images.append(
            {
                "method": method,
                "path": path,
                "image": image,
                "label": row.get("variant", method) if row is not None else method,
            }
        )
    return images


def load_ep10_state(root: Path) -> Ep10Cache:
    ep10_dirs = {
        "drizzle": root / "output" / "ep10_drizzle",
        "drizzle_4x": root / "output" / "ep10_drizzle_4x",
        "map_tv": root / "output" / "ep10_map_tv_sweep",
        "tgv": root / "output" / "ep10_tgv_sr",
    }
    output_dir = root / "output" / "ep10_method_comparison"
    output_dir.mkdir(parents=True, exist_ok=True)

    artifacts = {
        "drizzle": {
            "sweep": _read_csv(ep10_dirs["drizzle"] / "sweep_results.csv"),
            "synthetic": _read_json(ep10_dirs["drizzle"] / "synthetic_validation.json"),
            "small_real": _read_json(ep10_dirs["drizzle"] / "small_real_check.json"),
        },
        "drizzle_4x": {
            "sweep": _read_csv(ep10_dirs["drizzle_4x"] / "sweep_results.csv"),
            "synthetic": _read_json(ep10_dirs["drizzle_4x"] / "synthetic_validation.json"),
            "small_real": _read_json(ep10_dirs["drizzle_4x"] / "small_real_check.json"),
        },
        "map_tv": {
            "sweep": _read_csv(ep10_dirs["map_tv"] / "sweep_results.csv"),
            "best": _read_json(ep10_dirs["map_tv"] / "best_params.json"),
            "synthetic": _read_json(ep10_dirs["map_tv"] / "synthetic_validation.json"),
        },
        "tgv": {
            "sweep": _read_csv(ep10_dirs["tgv"] / "sweep_results.csv"),
            "run_summary": _read_json(ep10_dirs["tgv"] / "run_summary.json"),
            "synthetic": _read_json(ep10_dirs["tgv"] / "synthetic_validation.json"),
        },
    }

    sweeps = {
        "Drizzle": normalize_sweep("Drizzle", artifacts["drizzle"]["sweep"], ep10_dirs, root),
        "MAP-TV": normalize_sweep("MAP-TV", artifacts["map_tv"]["sweep"], ep10_dirs, root),
        "TGV": normalize_sweep("TGV", artifacts["tgv"]["sweep"], ep10_dirs, root),
    }
    best_rows = {method: select_best(df) for method, df in sweeps.items()}
    map_tv_top = pd.DataFrame(artifacts["map_tv"].get("best", {}).get("top3", []))
    if not map_tv_top.empty:
        map_tv_top = normalize_sweep("MAP-TV", map_tv_top, ep10_dirs, root)
        best_rows["MAP-TV"] = map_tv_top.iloc[0]
    tgv_best_label = artifacts["tgv"].get("run_summary", {}).get("best_label")
    if tgv_best_label and not sweeps["TGV"].empty:
        tgv_match = sweeps["TGV"][sweeps["TGV"]["label"].eq(tgv_best_label)]
        if not tgv_match.empty:
            best_rows["TGV"] = tgv_match.iloc[0]

    summary_table = build_summary_table(best_rows)
    all_candidates = (
        pd.concat([df for df in sweeps.values() if not df.empty], ignore_index=True)
        if any(not df.empty for df in sweeps.values())
        else pd.DataFrame()
    )
    status_rows = []
    for key, path in ep10_dirs.items():
        status_rows.append(
            {
                "algorithm": key,
                "output_dir": str(path.relative_to(root)),
                "sweep_rows": len(artifacts[key].get("sweep", pd.DataFrame())),
                "synthetic_json": bool(artifacts[key].get("synthetic")),
            }
        )
    status_table = pd.DataFrame(status_rows)
    manifest_path = output_dir / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}

    return Ep10Cache(
        project_root=root,
        output_dir=output_dir,
        report_dir=root / "reports" / "ep10_method_comparison",
        ep10_dirs=ep10_dirs,
        artifacts=artifacts,
        sweeps=sweeps,
        best_rows=best_rows,
        summary_table=summary_table,
        all_candidates=all_candidates,
        status_table=status_table,
        manifest=manifest,
    )


def plot_core_metric_comparison(summary_table: pd.DataFrame, output_dir: Path) -> bool:
    plot_df = summary_table.dropna(subset=["method"]).copy() if not summary_table.empty else pd.DataFrame()
    if plot_df.empty:
        return False
    method_colors = dict(zip(METHOD_ORDER, METHOD_COLOR_LIST[: len(METHOD_ORDER)], strict=True))
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.6), squeeze=False)
    for ax, metric in zip(axes.ravel(), COMMON_METRICS, strict=True):
        values = pd.to_numeric(plot_df[metric], errors="coerce")
        methods = plot_df["method"].tolist()
        colors = [method_colors.get(method, "#777777") for method in methods]
        bars = ax.bar(methods, values, color=colors, width=0.62)
        ax.set_title(metric_label(metric))
        ax.set_ylabel("Value")
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        hint = "lower is better" if metric in LOWER_IS_BETTER else "higher is better"
        ax.text(0.02, 0.94, hint, transform=ax.transAxes, fontsize=7, va="top")
        for bar, value in zip(bars, values, strict=True):
            if np.isfinite(value):
                ax.annotate(
                    f"{value:.3g}",
                    (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha="center",
                    va="bottom",
                    fontsize=7,
                    xytext=(0, 2),
                    textcoords="offset points",
                )
    savefig_academic(fig, output_dir / "core_metric_comparison.png")
    return True


def plot_center_roi_highpass(cache: Ep10Cache) -> bool:
    candidate_images = load_candidate_images(
        cache.best_rows,
        ep10_dirs=cache.ep10_dirs,
        root=cache.project_root,
        artifacts=cache.artifacts,
    )
    if not candidate_images:
        return False
    crops = [center_roi(item["image"], size=320) for item in candidate_images]
    finite = np.concatenate([crop[np.isfinite(crop)].ravel() for crop in crops if np.isfinite(crop).any()])
    vmax = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    n = len(crops)
    fig, axes = plt.subplots(1, n, figsize=(min(7.2, 2.45 * n), 2.8), squeeze=False)
    for ax, item, crop in zip(axes.ravel(), candidate_images, crops, strict=True):
        im = ax.imshow(crop, cmap=COLORMAPS["residual_diff"], vmin=-vmax, vmax=vmax, interpolation="nearest")
        ax.set_title(item["method"])
        ax.set_xlabel(item["label"])
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.035, pad=0.03).set_label("Highpass response [deg C]")
    savefig_academic(fig, cache.output_dir / "center_roi_highpass_comparison.png")
    return True


def plot_auxiliary_control_views(cache: Ep10Cache) -> bool:
    coverage_path = drizzle_image_path(cache.best_rows.get("Drizzle"), cache.ep10_dirs, suffix="coverage")
    raw_control_path = drizzle_image_path(
        cache.best_rows.get("Drizzle"), cache.ep10_dirs, suffix="raw_control_highpass"
    )
    coverage = _load_npy(coverage_path) if coverage_path is not None else None
    raw_control = _load_npy(raw_control_path) if raw_control_path is not None else None
    if coverage is None and raw_control is None:
        return False
    panels = []
    if coverage is not None:
        panels.append(("Drizzle coverage", center_roi(coverage, size=320), COLORMAPS["coverage"], None))
    if raw_control is not None:
        panels.append(
            ("Drizzle raw-control highpass", center_roi(raw_control, size=320), COLORMAPS["residual_diff"], "symmetric")
        )
    fig, axes = plt.subplots(1, len(panels), figsize=(min(7.2, 3.4 * len(panels)), 3.0), squeeze=False)
    for ax, (title, crop, cmap, mode) in zip(axes.ravel(), panels, strict=True):
        if mode == "symmetric":
            finite = crop[np.isfinite(crop)]
            vmax = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0
            im = ax.imshow(crop, cmap=cmap, vmin=-vmax, vmax=vmax, interpolation="nearest")
        else:
            im = ax.imshow(crop, cmap=cmap, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    savefig_academic(fig, cache.output_dir / "auxiliary_control_views.png")
    return True


def plot_intermediate_parameter_diagnostics(cache: Ep10Cache) -> bool:
    drizzle_sweep = cache.sweeps.get("Drizzle", pd.DataFrame())
    map_tv_sweep = cache.sweeps.get("MAP-TV", pd.DataFrame())
    tgv_sweep = cache.sweeps.get("TGV", pd.DataFrame())
    if drizzle_sweep.empty and map_tv_sweep.empty and tgv_sweep.empty:
        return False
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 4.8), squeeze=False)
    ax = axes[0, 0]
    if not drizzle_sweep.empty:
        d = drizzle_sweep.sort_values("pixfrac")
        ax.plot(d["pixfrac"], d["split_half_nrmse"], marker="o", label="split-half NRMSE")
        ax.plot(d["pixfrac"], d["artifact_score"], marker="s", label="artifact score")
    ax.set_title("Drizzle pixfrac path")
    ax.set_xlabel("pixfrac")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(loc="best")

    ax = axes[0, 1]
    if not drizzle_sweep.empty and {"coverage_p05", "coverage_median", "coverage_p95"}.issubset(drizzle_sweep.columns):
        d = drizzle_sweep.sort_values("pixfrac")
        ax.plot(d["pixfrac"], d["coverage_p05"], marker="o", label="p05")
        ax.plot(d["pixfrac"], d["coverage_median"], marker="s", label="median")
        ax.plot(d["pixfrac"], d["coverage_p95"], marker="^", label="p95")
    ax.set_title("Drizzle coverage path")
    ax.set_xlabel("pixfrac")
    ax.set_ylabel("Coverage weight")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(loc="best")

    ax = axes[1, 0]
    if not map_tv_sweep.empty:
        for sigma, group in map_tv_sweep.sort_values("lambda_tv").groupby("psf_sigma"):
            ax.plot(group["lambda_tv"], group["split_half_nrmse"], marker="o", label=f"sigma={sigma:g}")
        ax.set_xscale("log")
    ax.set_title("MAP-TV sweep trajectory")
    ax.set_xlabel("lambda_TV")
    ax.set_ylabel("Split-half NRMSE")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(loc="best")

    ax = axes[1, 1]
    if not tgv_sweep.empty:
        labels = tgv_sweep["variant"].astype(str).tolist()
        xpos = np.arange(len(labels))
        ax.plot(xpos, tgv_sweep["split_half_nrmse"], marker="o", label="split-half NRMSE")
        ax.plot(xpos, tgv_sweep["artifact_score"], marker="s", label="artifact score")
        ax.set_xticks(xpos, labels, rotation=45, ha="right")
    ax.set_title("TGV candidate trajectory")
    ax.set_ylabel("Value")
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.legend(loc="best")

    savefig_academic(fig, cache.output_dir / "intermediate_parameter_diagnostics.png")
    return True


def plot_split_holdout_distribution_diagnostics(cache: Ep10Cache) -> bool:
    tgv_split = _read_csv(cache.ep10_dirs["tgv"] / "split_half_details.csv")
    tgv_holdout = _read_csv(cache.ep10_dirs["tgv"] / "holdout_details.csv")
    map_split = _read_csv(cache.ep10_dirs["map_tv"] / "split_half_details.csv")
    map_holdout = _read_csv(cache.ep10_dirs["map_tv"] / "holdout_details.csv")
    if tgv_split.empty and tgv_holdout.empty and map_split.empty and map_holdout.empty:
        return False
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2), squeeze=False)
    split_frames = []
    if not map_split.empty:
        split_frames.append(map_split.assign(source="MAP-TV"))
    if not tgv_split.empty:
        split_frames.append(tgv_split.assign(source="TGV"))
    if split_frames:
        split_df = pd.concat(split_frames, ignore_index=True)
        labels, values = [], []
        for label, group in split_df.groupby("source"):
            labels.append(label)
            values.append(pd.to_numeric(group["nrmse"], errors="coerce").dropna().to_numpy())
        axes[0, 0].boxplot(values, tick_labels=labels, showfliers=False)
    axes[0, 0].set_title("Split-half NRMSE details")
    axes[0, 0].set_ylabel("NRMSE")
    axes[0, 0].grid(axis="y", alpha=0.3, linewidth=0.5)

    holdout_frames = []
    if not map_holdout.empty:
        holdout_frames.append(map_holdout.assign(source="MAP-TV"))
    if not tgv_holdout.empty:
        holdout_frames.append(tgv_holdout.assign(source="TGV"))
    if holdout_frames:
        holdout_df = pd.concat(holdout_frames, ignore_index=True)
        labels, values = [], []
        for label, group in holdout_df.groupby("source"):
            labels.append(label)
            values.append(pd.to_numeric(group["mse"], errors="coerce").dropna().to_numpy())
        axes[0, 1].boxplot(values, tick_labels=labels, showfliers=False)
    axes[0, 1].set_title("Holdout MSE details")
    axes[0, 1].set_ylabel("MSE")
    axes[0, 1].grid(axis="y", alpha=0.3, linewidth=0.5)

    savefig_academic(fig, cache.output_dir / "split_holdout_distribution_diagnostics.png")
    return True


def plot_drizzle_4x_diagnostics(cache: Ep10Cache) -> bool:
    drizzle_4x = cache.artifacts["drizzle_4x"]["sweep"].copy()
    if drizzle_4x.empty:
        return False
    d = drizzle_4x.sort_values("pixfrac")
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.8), squeeze=False)
    axes[0, 0].plot(d["pixfrac"], d["split_half_nrmse"], marker="o", label="split-half")
    axes[0, 0].plot(d["pixfrac"], d["artifact_score"], marker="s", label="artifact")
    axes[0, 0].set_title("4x stability/artifact")
    axes[0, 0].set_xlabel("pixfrac")
    axes[0, 0].set_ylabel("Value")
    axes[0, 0].grid(axis="y", alpha=0.3, linewidth=0.5)
    axes[0, 0].legend(loc="best")

    axes[0, 1].plot(d["pixfrac"], d["coverage_p05"], marker="o", label="p05")
    axes[0, 1].plot(d["pixfrac"], d["coverage_median"], marker="s", label="median")
    axes[0, 1].plot(d["pixfrac"], d["coverage_p95"], marker="^", label="p95")
    axes[0, 1].set_title("4x coverage")
    axes[0, 1].set_xlabel("pixfrac")
    axes[0, 1].set_ylabel("Coverage weight")
    axes[0, 1].grid(axis="y", alpha=0.3, linewidth=0.5)
    axes[0, 1].legend(loc="best")

    axes[0, 2].plot(d["pixfrac"], d["coverage_lt1_fraction"], marker="o", color=METHOD_COLOR_LIST[2])
    axes[0, 2].set_title("Under-covered pixels")
    axes[0, 2].set_xlabel("pixfrac")
    axes[0, 2].set_ylabel("Fraction below 1")
    axes[0, 2].grid(axis="y", alpha=0.3, linewidth=0.5)

    savefig_academic(fig, cache.output_dir / "drizzle_4x_diagnostics.png")
    return True


def plot_drizzle_2x_vs_4x_center_third_crop(cache: Ep10Cache) -> bool:
    drizzle_4x = cache.artifacts["drizzle_4x"]["sweep"].copy()
    if drizzle_4x.empty or cache.best_rows.get("Drizzle") is None:
        return False
    sort_cols = [c for c in ["split_half_nrmse", "artifact_score", "holdout_mse"] if c in drizzle_4x.columns]
    best_4x = drizzle_4x.sort_values(sort_cols, na_position="last").iloc[0] if sort_cols else drizzle_4x.iloc[0]
    pf2 = float(cache.best_rows["Drizzle"]["pixfrac"])
    pf4 = float(best_4x["pixfrac"])
    img2 = _load_npy(cache.ep10_dirs["drizzle"] / f"drizzle_pf{pf2:.1f}_hr.npy")
    img4 = _load_npy(cache.ep10_dirs["drizzle_4x"] / f"drizzle_pf{pf4:.1f}_hr.npy")
    if img2 is None or img4 is None:
        return False
    crop2 = center_fraction_crop(img2, fraction=1.0 / 3.0)
    crop2_display = ndimage.zoom(crop2, zoom=2.0, order=1)
    crop4 = center_fraction_crop(img4, fraction=1.0 / 3.0)
    finite = np.concatenate(
        [crop2_display[np.isfinite(crop2_display)].ravel(), crop4[np.isfinite(crop4)].ravel()]
    )
    vmax = float(np.percentile(np.abs(finite), 99.0)) if finite.size else 1.0
    vmax = max(vmax, 1e-6)
    panels = [
        (f"2x Drizzle pf={pf2:.1f}", crop2_display, COLORMAPS["residual_diff"], -vmax, vmax),
        (f"4x Drizzle pf={pf4:.1f}", crop4, COLORMAPS["residual_diff"], -vmax, vmax),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(5.8, 3.0), squeeze=False)
    im = None
    for ax, (title, image, cmap, vmin, vmax_i) in zip(axes.ravel(), panels, strict=True):
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax_i, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.04).set_label("Highpass response [deg C]")
    savefig_academic(fig, cache.output_dir / "drizzle_2x_vs_4x_center_third_crop.png")
    return True


def build_ep10_cache(
    *,
    project_root_path: Path | None = None,
    output_dir: Path | None = None,
    force: bool = False,
) -> Ep10Cache:
    root = project_root(project_root_path)
    cache = load_ep10_state(root)
    output_dir = (output_dir or cache.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not force and cache_is_complete(output_dir, EP10_FIGURE_ARTIFACTS):
        return load_ep10_cache(project_root_path=root, output_dir=output_dir)

    setup_academic_style()
    built: list[str] = []
    if plot_core_metric_comparison(cache.summary_table, output_dir):
        built.append("core_metric_comparison.png")
    if plot_center_roi_highpass(cache):
        built.append("center_roi_highpass_comparison.png")
    if plot_auxiliary_control_views(cache):
        built.append("auxiliary_control_views.png")
    if plot_intermediate_parameter_diagnostics(cache):
        built.append("intermediate_parameter_diagnostics.png")
    if plot_split_holdout_distribution_diagnostics(cache):
        built.append("split_holdout_distribution_diagnostics.png")
    if plot_drizzle_4x_diagnostics(cache):
        built.append("drizzle_4x_diagnostics.png")
    if plot_drizzle_2x_vs_4x_center_third_crop(cache):
        built.append("drizzle_2x_vs_4x_center_third_crop.png")

    manifest = write_manifest(
        output_dir,
        version=EP10_CACHE_VERSION,
        artifacts=[*EP10_FIGURE_ARTIFACTS, "cache_manifest.json"],
        rebuild_command=REBUILD_COMMAND,
        extra={"figures_built": built},
    )
    loaded = load_ep10_cache(project_root_path=root, output_dir=output_dir)
    return Ep10Cache(
        project_root=loaded.project_root,
        output_dir=loaded.output_dir,
        report_dir=loaded.report_dir,
        ep10_dirs=loaded.ep10_dirs,
        artifacts=loaded.artifacts,
        sweeps=loaded.sweeps,
        best_rows=loaded.best_rows,
        summary_table=loaded.summary_table,
        all_candidates=loaded.all_candidates,
        status_table=loaded.status_table,
        manifest=manifest,
    )


def load_ep10_cache(
    *,
    project_root_path: Path | None = None,
    output_dir: Path | None = None,
    require_figures: bool = False,
) -> Ep10Cache:
    root = project_root(project_root_path)
    output_dir = (output_dir or root / "output" / "ep10_method_comparison").resolve()
    if require_figures:
        require_artifacts(output_dir, EP10_FIGURE_ARTIFACTS, rebuild_command=REBUILD_COMMAND)
    cache = load_ep10_state(root)
    manifest_path = output_dir / "cache_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.exists() else {}
    return Ep10Cache(
        project_root=cache.project_root,
        output_dir=output_dir,
        report_dir=cache.report_dir,
        ep10_dirs=cache.ep10_dirs,
        artifacts=cache.artifacts,
        sweeps=cache.sweeps,
        best_rows=cache.best_rows,
        summary_table=cache.summary_table,
        all_candidates=cache.all_candidates,
        status_table=cache.status_table,
        manifest=manifest,
    )
