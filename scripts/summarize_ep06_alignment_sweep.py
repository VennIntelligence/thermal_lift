#!/usr/bin/env python3
"""Summarize EP06 data-driven alignment sweep outputs.

The script only reads compact EP06 CSV/JSON summaries. It deliberately avoids
loading reconstruction ``.npy`` arrays so it can be rerun while long sweeps are
still producing image products in parallel.

用法（项目根目录）::

    uv run python scripts/summarize_ep06_alignment_sweep.py \
        [--sweep-root output/ep06_sr_poc_data_driven_align_sweep] \
        [--baseline-dir output/ep06_sr_poc] [--output-dir DIR]

输入依赖: sweep-root 下各实验子目录的 evaluation_summary.csv、
    map_tv_lambda_selection.csv、*_synthetic_validation.json，
    以及 baseline-dir 的 evaluation_summary.csv
输出: <sweep-root>/summary/（--output-dir 可覆盖）sweep_*.csv、
    sweep_summary.json 与汇总图表

关联: EP06
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def bootstrap_project_root() -> Path:
    root = Path(__file__).resolve()
    while root != root.parent:
        if (root / "AGENTS.md").exists():
            break
        root = root.parent
    if not (root / "AGENTS.md").exists():
        root = Path.cwd().resolve()
    core_src = root / "core" / "src"
    if str(core_src) not in sys.path:
        sys.path.insert(0, str(core_src))
    return root


PROJECT_ROOT = bootstrap_project_root()

from thermal_core.plotting import FIGURE_SIZES, METHOD_COLOR_LIST, savefig_academic, setup_academic_style  # noqa: E402


EVALUATION_FILE = "evaluation_summary.csv"
LAMBDA_FILE = "map_tv_lambda_selection.csv"
VALIDATION_FILES = {
    "saa": "saa_synthetic_validation.json",
    "ibp": "ibp_synthetic_validation.json",
    "map_tv": "map_tv_synthetic_validation.json",
}

METRIC_COLUMNS = [
    "std",
    "std_ratio_to_lr",
    "mean_gradient",
    "p95_gradient",
    "artifact_score",
    "contour_chamfer_lr_px",
    "nrmse_to_bicubic",
    "corr_to_bicubic",
]

METHOD_ORDER = [
    "lr_reference",
    "bicubic",
    "saa_uniform",
    "saa_weighted",
    "ibp",
    "map_tv",
]

METHOD_LABELS = {
    "lr_reference": "LR",
    "bicubic": "Bicubic",
    "saa_uniform": "SAA uniform",
    "saa_weighted": "SAA weighted",
    "ibp": "IBP",
    "map_tv": "MAP-TV",
}

METRIC_OUTPUT_COLUMNS = [
    "experiment",
    "track",
    "method",
    *METRIC_COLUMNS,
    "source_dir",
]


def relative_path(path: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def resolve_path(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def clean_json(value: Any) -> Any:
    if isinstance(value, Path):
        return relative_path(value)
    if isinstance(value, dict):
        return {str(key): clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, tuple):
        return [clean_json(item) for item in value]
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def read_json(path: Path) -> dict[str, Any] | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def boolish(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if pd.isna(value):
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def discover_experiments(sweep_root: Path, output_dir: Path) -> list[Path]:
    expected = {EVALUATION_FILE, LAMBDA_FILE, *VALIDATION_FILES.values()}
    if not sweep_root.exists():
        return []

    output_resolved = output_dir.resolve()

    def has_summary_file(path: Path) -> bool:
        return any((path / name).exists() for name in expected)

    if has_summary_file(sweep_root):
        return [sweep_root]

    experiments: list[Path] = []
    for child in sorted(path for path in sweep_root.iterdir() if path.is_dir()):
        if child.resolve() == output_resolved or child.name.startswith("."):
            continue
        if has_summary_file(child):
            experiments.append(child)
    return experiments


def add_missing(missing: list[dict[str, str]], experiment: str, path: Path, kind: str) -> None:
    missing.append(
        {
            "experiment": experiment,
            "kind": kind,
            "path": relative_path(path),
        }
    )


def read_evaluation_table(
    experiment_dir: Path,
    *,
    experiment: str,
    missing: list[dict[str, str]],
) -> pd.DataFrame:
    path = experiment_dir / EVALUATION_FILE
    if not path.exists():
        add_missing(missing, experiment, path, "evaluation")
        return pd.DataFrame(columns=METRIC_OUTPUT_COLUMNS)

    table = pd.read_csv(path)
    for column in ["track", "method", *METRIC_COLUMNS]:
        if column not in table.columns:
            table[column] = np.nan
    if "std_ratio_to_lr" not in table.columns or table["std_ratio_to_lr"].isna().all():
        table["std_ratio_to_lr"] = np.nan

    for track, group in table.groupby("track", dropna=False):
        if "std" not in group.columns:
            continue
        lr = group[group["method"].eq("lr_reference")]
        if lr.empty:
            continue
        ref_std = pd.to_numeric(lr["std"], errors="coerce").dropna()
        if ref_std.empty or float(ref_std.iloc[0]) <= 0:
            continue
        empty = table.index.isin(group.index) & table["std_ratio_to_lr"].isna()
        table.loc[empty, "std_ratio_to_lr"] = pd.to_numeric(table.loc[empty, "std"], errors="coerce") / float(ref_std.iloc[0])

    out = table[["track", "method", *METRIC_COLUMNS]].copy()
    out.insert(0, "experiment", experiment)
    out["source_dir"] = relative_path(experiment_dir)
    return out[METRIC_OUTPUT_COLUMNS]


def read_lambda_table(
    experiment_dir: Path,
    *,
    experiment: str,
    missing: list[dict[str, str]],
) -> pd.DataFrame:
    path = experiment_dir / LAMBDA_FILE
    if not path.exists():
        add_missing(missing, experiment, path, "map_tv_lambda")
        return pd.DataFrame()
    table = pd.read_csv(path)
    table.insert(0, "experiment", experiment)
    table["source_dir"] = relative_path(experiment_dir)
    return table


def validation_row(
    experiment_dir: Path,
    *,
    experiment: str,
    method: str,
    missing: list[dict[str, str]],
) -> dict[str, Any]:
    path = experiment_dir / VALIDATION_FILES[method]
    row: dict[str, Any] = {
        "experiment": experiment,
        "method": method,
        "status": "ok",
        "alignment_method": np.nan,
        "psnr_db": np.nan,
        "saa_init_psnr_db": np.nan,
        "pass_25db": np.nan,
        "beats_saa": np.nan,
        "lambda_tv": np.nan,
        "selected_lambda_highpass": np.nan,
        "selected_lambda_raw": np.nan,
        "psf_sigma": np.nan,
        "n_frames": np.nan,
        "real_n_frames": np.nan,
        "elapsed_sec": np.nan,
        "source_json": relative_path(path),
        "source_dir": relative_path(experiment_dir),
    }
    if not path.exists():
        row["status"] = "missing"
        add_missing(missing, experiment, path, f"{method}_validation")
        return row

    data = read_json(path)
    if data is None:
        row["status"] = "invalid_json"
        return row

    row["alignment_method"] = data.get("alignment_method", np.nan)
    row["saa_init_psnr_db"] = data.get("saa_init_psnr_db", np.nan)
    row["n_frames"] = data.get("n_frames", np.nan)
    row["real_n_frames"] = data.get("real_n_frames", np.nan)
    row["elapsed_sec"] = data.get("elapsed_sec", np.nan)
    row["psf_sigma"] = data.get("psf_sigma", data.get("psf_sigma_lr_px", np.nan))

    if method == "saa":
        row["psnr_db"] = data.get("weighted_psnr_db", data.get("uniform_psnr_db", np.nan))
        row["pass_25db"] = data.get("weighted_pass_25db", data.get("uniform_pass_25db", np.nan))
        row["uniform_psnr_db"] = data.get("uniform_psnr_db", np.nan)
        row["weighted_psnr_db"] = data.get("weighted_psnr_db", np.nan)
    elif method == "ibp":
        row["psnr_db"] = data.get("ibp_psnr_db", np.nan)
        row["beats_saa"] = data.get("ibp_beats_saa", np.nan)
        row["iterations"] = data.get("iterations", np.nan)
    elif method == "map_tv":
        row["psnr_db"] = data.get("map_tv_psnr_db", np.nan)
        row["beats_saa"] = data.get("map_tv_beats_saa", np.nan)
        row["lambda_tv"] = data.get("lambda_tv", np.nan)
        row["selected_lambda_highpass"] = data.get("selected_lambda_highpass", np.nan)
        row["selected_lambda_raw"] = data.get("selected_lambda_raw", np.nan)
        row["iterations"] = data.get("iterations", np.nan)
    return row


def read_validation_rows(
    experiment_dir: Path,
    *,
    experiment: str,
    missing: list[dict[str, str]],
) -> pd.DataFrame:
    rows = [
        validation_row(experiment_dir, experiment=experiment, method=method, missing=missing)
        for method in VALIDATION_FILES
    ]
    return pd.DataFrame(rows)


def load_baseline_metrics(baseline_dir: Path, missing: list[dict[str, str]]) -> pd.DataFrame:
    path = baseline_dir / EVALUATION_FILE
    if not path.exists():
        add_missing(missing, "baseline", path, "baseline_evaluation")
        return pd.DataFrame(columns=["track", "method", *METRIC_COLUMNS])
    table = pd.read_csv(path)
    for column in ["track", "method", *METRIC_COLUMNS]:
        if column not in table.columns:
            table[column] = np.nan
    return table[["track", "method", *METRIC_COLUMNS]].drop_duplicates(["track", "method"], keep="first")


def build_delta_table(metrics: pd.DataFrame, baseline: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty or baseline.empty:
        columns = [
            "experiment",
            "track",
            "method",
            "source_dir",
            *[f"{column}_current" for column in METRIC_COLUMNS],
            *[f"{column}_baseline" for column in METRIC_COLUMNS],
            *[f"{column}_delta_vs_baseline" for column in METRIC_COLUMNS],
        ]
        return pd.DataFrame(columns=columns)

    baseline_renamed = baseline.rename(columns={column: f"{column}_baseline" for column in METRIC_COLUMNS})
    current_renamed = metrics.rename(columns={column: f"{column}_current" for column in METRIC_COLUMNS})
    merged = current_renamed.merge(baseline_renamed, on=["track", "method"], how="left")
    for column in METRIC_COLUMNS:
        current = pd.to_numeric(merged[f"{column}_current"], errors="coerce")
        base = pd.to_numeric(merged[f"{column}_baseline"], errors="coerce")
        merged[f"{column}_delta_vs_baseline"] = current - base
    ordered = [
        "experiment",
        "track",
        "method",
        "source_dir",
        *[f"{column}_current" for column in METRIC_COLUMNS],
        *[f"{column}_baseline" for column in METRIC_COLUMNS],
        *[f"{column}_delta_vs_baseline" for column in METRIC_COLUMNS],
    ]
    return merged[ordered]


def ordered_methods(table: pd.DataFrame) -> list[str]:
    present = set(table["method"].dropna().astype(str))
    ordered = [method for method in METHOD_ORDER if method in present]
    ordered.extend(sorted(present.difference(ordered)))
    return ordered


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.replace("_", " "))


def plot_grouped_metric_bars(metrics: pd.DataFrame, output_dir: Path) -> Path | None:
    high = metrics[
        metrics["track"].eq("highpass")
        & metrics["method"].notna()
        & ~metrics["method"].eq("lr_reference")
    ].copy()
    if high.empty:
        return None

    setup_academic_style()
    methods = ordered_methods(high)
    experiments = sorted(high["experiment"].dropna().astype(str).unique())
    x = np.arange(len(methods), dtype=float)
    width = min(0.78 / max(1, len(experiments)), 0.28)
    metric_specs = [
        ("std_ratio_to_lr", "Std / LR std", "ratio"),
        ("p95_gradient", "P95 gradient", "gradient"),
        ("artifact_score", "Artifact score", "score"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["double_col"], constrained_layout=True)
    for ax, (column, title, ylabel) in zip(axes, metric_specs, strict=True):
        for idx, experiment in enumerate(experiments):
            values = []
            subset = high[high["experiment"].eq(experiment)]
            for method in methods:
                match = subset[subset["method"].eq(method)]
                values.append(float(pd.to_numeric(match[column], errors="coerce").iloc[0]) if not match.empty else np.nan)
            offset = (idx - (len(experiments) - 1) / 2) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                label=experiment,
                color=METHOD_COLOR_LIST[idx % len(METHOD_COLOR_LIST)],
            )
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([method_label(method) for method in methods], rotation=35, ha="right")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best")
    return savefig_academic(fig, output_dir / "sweep_metric_bars.png")


def plot_lambda_selection(lambda_table: pd.DataFrame, output_dir: Path) -> Path | None:
    if lambda_table.empty or "lambda_tv" not in lambda_table.columns:
        return None

    setup_academic_style()
    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZES["double_col"], constrained_layout=True)
    y_specs = [
        ("split_half_nrmse", "Split-half NRMSE", "NRMSE"),
        ("selection_proxy", "Selection proxy", "proxy score"),
    ]
    groups = list(lambda_table.groupby(["experiment", "track"], dropna=False))
    for ax, (column, title, ylabel) in zip(axes, y_specs, strict=True):
        if column not in lambda_table.columns:
            ax.text(0.5, 0.5, f"missing {column}", ha="center", va="center")
            ax.set_axis_off()
            continue
        for idx, ((experiment, track), group) in enumerate(groups):
            group = group.sort_values("lambda_tv")
            color = METHOD_COLOR_LIST[idx % len(METHOD_COLOR_LIST)]
            label = f"{experiment} / {track}"
            ax.semilogx(group["lambda_tv"], group[column], marker="o", color=color, label=label)
            if "selected" in group.columns:
                selected = group[group["selected"].map(boolish)]
                if not selected.empty:
                    ax.scatter(
                        selected["lambda_tv"],
                        selected[column],
                        s=60,
                        facecolors="none",
                        edgecolors=color,
                        linewidths=1.2,
                    )
        ax.set_title(title)
        ax.set_xlabel("lambda TV")
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best")
    return savefig_academic(fig, output_dir / "sweep_map_tv_lambda_selection.png")


def plot_delta_vs_baseline(delta: pd.DataFrame, output_dir: Path) -> Path | None:
    high = delta[
        delta["track"].eq("highpass")
        & delta["method"].notna()
        & ~delta["method"].eq("lr_reference")
    ].copy()
    if high.empty:
        return None

    setup_academic_style()
    methods = ordered_methods(high)
    experiments = sorted(high["experiment"].dropna().astype(str).unique())
    x = np.arange(len(methods), dtype=float)
    width = min(0.78 / max(1, len(experiments)), 0.28)
    metric_specs = [
        ("std_ratio_to_lr_delta_vs_baseline", "Delta std / LR std"),
        ("p95_gradient_delta_vs_baseline", "Delta P95 gradient"),
        ("artifact_score_delta_vs_baseline", "Delta artifact"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["double_col"], constrained_layout=True)
    for ax, (column, title) in zip(axes, metric_specs, strict=True):
        for idx, experiment in enumerate(experiments):
            values = []
            subset = high[high["experiment"].eq(experiment)]
            for method in methods:
                match = subset[subset["method"].eq(method)]
                values.append(float(pd.to_numeric(match[column], errors="coerce").iloc[0]) if not match.empty else np.nan)
            offset = (idx - (len(experiments) - 1) / 2) * width
            ax.bar(
                x + offset,
                values,
                width=width,
                label=experiment,
                color=METHOD_COLOR_LIST[idx % len(METHOD_COLOR_LIST)],
            )
        ax.axhline(0.0, color="#555555", linewidth=0.8)
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([method_label(method) for method in methods], rotation=35, ha="right")
        ax.set_ylabel("current - baseline")
        ax.grid(axis="y", alpha=0.25)
    axes[0].legend(loc="best")
    return savefig_academic(fig, output_dir / "sweep_delta_vs_baseline.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sweep-root",
        type=Path,
        default=Path("output/ep06_sr_poc_data_driven_align_sweep"),
        help="Directory containing one or more EP06 sweep subdirectories.",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=Path("output/ep06_sr_poc"),
        help="Old EP06 baseline directory with evaluation_summary.csv.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Summary output directory. Defaults to <sweep-root>/summary.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sweep_root = resolve_path(args.sweep_root).resolve()
    baseline_dir = resolve_path(args.baseline_dir).resolve()
    output_dir = resolve_path(args.output_dir).resolve() if args.output_dir else sweep_root / "summary"
    output_dir.mkdir(parents=True, exist_ok=True)

    missing: list[dict[str, str]] = []
    experiments = discover_experiments(sweep_root, output_dir)

    metric_tables: list[pd.DataFrame] = []
    lambda_tables: list[pd.DataFrame] = []
    validation_tables: list[pd.DataFrame] = []
    for experiment_dir in experiments:
        experiment = experiment_dir.name
        metric_tables.append(read_evaluation_table(experiment_dir, experiment=experiment, missing=missing))
        lambda_tables.append(read_lambda_table(experiment_dir, experiment=experiment, missing=missing))
        validation_tables.append(read_validation_rows(experiment_dir, experiment=experiment, missing=missing))

    metrics = pd.concat(metric_tables, ignore_index=True) if metric_tables else pd.DataFrame(columns=METRIC_OUTPUT_COLUMNS)
    lambdas = pd.concat(lambda_tables, ignore_index=True) if lambda_tables else pd.DataFrame()
    validations = pd.concat(validation_tables, ignore_index=True) if validation_tables else pd.DataFrame()
    baseline = load_baseline_metrics(baseline_dir, missing)
    delta = build_delta_table(metrics, baseline)

    metric_path = output_dir / "sweep_method_metrics.csv"
    lambda_path = output_dir / "sweep_map_tv_lambda.csv"
    validation_path = output_dir / "sweep_validation_summary.csv"
    delta_path = output_dir / "sweep_delta_vs_baseline.csv"

    metrics.to_csv(metric_path, index=False)
    lambdas.to_csv(lambda_path, index=False)
    validations.to_csv(validation_path, index=False)
    delta.to_csv(delta_path, index=False)

    figures = [
        plot_grouped_metric_bars(metrics, output_dir),
        plot_lambda_selection(lambdas, output_dir),
        plot_delta_vs_baseline(delta, output_dir),
    ]
    figures = [path for path in figures if path is not None]

    summary = {
        "sweep_root": relative_path(sweep_root),
        "baseline_dir": relative_path(baseline_dir),
        "output_dir": relative_path(output_dir),
        "experiments": [path.name for path in experiments],
        "counts": {
            "experiments": len(experiments),
            "metric_rows": int(len(metrics)),
            "lambda_rows": int(len(lambdas)),
            "validation_rows": int(len(validations)),
            "delta_rows": int(len(delta)),
            "missing_files": len(missing),
            "figures": len(figures),
        },
        "missing_files": missing,
        "outputs": {
            "sweep_method_metrics": metric_path,
            "sweep_map_tv_lambda": lambda_path,
            "sweep_validation_summary": validation_path,
            "sweep_delta_vs_baseline": delta_path,
            "figures": figures,
        },
    }
    summary_path = output_dir / "sweep_summary.json"
    summary_path.write_text(json.dumps(clean_json(summary), indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Experiments: {len(experiments)}")
    print(f"Metric rows: {len(metrics)}")
    print(f"Lambda rows: {len(lambdas)}")
    print(f"Validation rows: {len(validations)}")
    print(f"Missing files recorded: {len(missing)}")
    print(f"Saved summary to {relative_path(output_dir)}")


if __name__ == "__main__":
    main()
