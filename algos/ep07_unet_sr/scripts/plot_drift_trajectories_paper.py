#!/usr/bin/env python3
"""Generate paper Fig. 3 null-space drift trajectories from TensorBoard logs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from extract_checkpoint_metrics import (
    DEFAULT_FORWARD_OUTPUT_CSV,
    DEFAULT_OUTPUT_CSV,
    DEFAULT_FORWARD_ARMS,
    extract_all,
    extract_forward_loss,
)
from thermal_core.plotting import get_method_style, savefig_academic, setup_academic_style


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PAPER_OUTPUT_DIR = PROJECT_ROOT / "output" / "paper_figures"

MAIN_ARMS = ["v6", "v8.1a", "v8.1b", "v9b", "v9d"]
COMPANION_ARM = "v9a"
PLOT_ARMS = [*MAIN_ARMS, COMPANION_ARM]
TRAINING_ARMS = {"v9a", "v9d"}
TOTAL_STEPS = 60_000

ARM_LABELS = {
    "v6": "v6 hot + full fwd",
    "v8.1a": "v8.1a conservative",
    "v8.1b": "v8.1b PixelShuffle",
    "v9b": "v9b band fwd",
    "v9d": "v9d full fwd",
    "v9a": "v9a hybrid input",
}
CANONICAL_STEPS = {
    "v6": 8_000,
    "v8.1a": 15_000,
    "v8.1b": 5_000,
    "v9b": 11_000,
}
METRIC_COLUMNS = {
    "artifact_score": "artifact_score (lower is better)",
    "raw_control_corr": "raw_control_corr (higher is better)",
}


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _k_formatter(value: float, _pos: int) -> str:
    if value == 0:
        return "0"
    return f"{int(value / 1000)}K"


def _markevery(n_rows: int) -> int:
    return max(1, int(np.ceil(max(n_rows, 1) / 8)))


def _style_for_arm(arm: str) -> dict:
    return get_method_style(PLOT_ARMS.index(arm))


def _arm_label(arm: str, group: pd.DataFrame) -> str:
    label = ARM_LABELS[arm]
    if arm in TRAINING_ARMS:
        max_step = int(group["step"].max()) if not group.empty else 0
        if max_step < TOTAL_STEPS:
            label = f"{label} (training)"
    return label


def _read_metrics(input_csv: Path, *, refresh: bool) -> pd.DataFrame:
    required = {"arm", "step", "artifact_score", "raw_control_corr"}
    if refresh or not input_csv.exists():
        metrics = extract_all(None, arms=PLOT_ARMS)
    else:
        metrics = pd.read_csv(input_csv)
        missing_cols = required - set(metrics.columns)
        if missing_cols:
            raise KeyError(f"Missing metric CSV column(s): {sorted(missing_cols)}")

        present = set(metrics["arm"].astype(str).unique())
        missing_arms = [arm for arm in PLOT_ARMS if arm not in present]
        if missing_arms:
            supplement = extract_all(None, arms=missing_arms)
            metrics = pd.concat([metrics, supplement], ignore_index=True)

    metrics = metrics[["arm", "step", "artifact_score", "raw_control_corr"]].copy()
    metrics["arm"] = metrics["arm"].astype(str)
    metrics["step"] = metrics["step"].astype(int)
    for col in ("artifact_score", "raw_control_corr"):
        metrics[col] = metrics[col].astype(float)
    metrics["arm_order"] = pd.Categorical(metrics["arm"], categories=PLOT_ARMS, ordered=True)
    metrics = metrics.sort_values(["arm_order", "step"]).drop(columns=["arm_order"]).reset_index(drop=True)
    return metrics


def _read_forward_loss(forward_csv: Path, *, refresh: bool) -> pd.DataFrame:
    if refresh or not forward_csv.exists():
        forward = extract_forward_loss(forward_csv, arms=DEFAULT_FORWARD_ARMS, strict=False)
    else:
        forward = pd.read_csv(forward_csv)

    if forward.empty:
        return pd.DataFrame(columns=["arm", "step", "value"])

    required = {"arm", "step", "value"}
    missing_cols = required - set(forward.columns)
    if missing_cols:
        raise KeyError(f"Missing forward CSV column(s): {sorted(missing_cols)}")

    forward = forward[list(required)].copy()
    forward["arm"] = forward["arm"].astype(str)
    forward["step"] = forward["step"].astype(int)
    forward["value"] = forward["value"].astype(float)
    return forward.sort_values(["arm", "step"]).reset_index(drop=True)


def _plot_metric_series(ax: Axes, metrics: pd.DataFrame, metric: str, arms: list[str]) -> None:
    for arm in arms:
        group = metrics[metrics["arm"] == arm].sort_values("step")
        if group.empty:
            continue
        style = _style_for_arm(arm)
        ax.plot(
            group["step"],
            group[metric],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=_markevery(len(group)),
            label=_arm_label(arm, group),
            alpha=0.95,
        )

        canonical_step = CANONICAL_STEPS.get(arm)
        if canonical_step is not None:
            canonical = group[group["step"] == canonical_step]
            if not canonical.empty:
                ax.scatter(
                    canonical["step"],
                    canonical[metric],
                    s=58,
                    facecolors="white",
                    edgecolors=style["color"],
                    linewidths=1.3,
                    zorder=5,
                )

        terminal = group[group["step"] == TOTAL_STEPS]
        if not terminal.empty:
            ax.scatter(
                terminal["step"],
                terminal[metric],
                marker="x",
                s=48,
                color=style["color"],
                linewidths=1.3,
                zorder=6,
            )


def _add_forward_inset(ax: Axes, forward: pd.DataFrame) -> None:
    inset = inset_axes(ax, width="49%", height="43%", loc="lower right", borderpad=0.8)
    inset.axhspan(0.004, 0.009, color="#999999", alpha=0.18, linewidth=0)

    plotted = False
    for arm in DEFAULT_FORWARD_ARMS:
        group = forward[forward["arm"] == arm].sort_values("step")
        group = group[np.isfinite(group["value"]) & (group["value"] > 0)]
        if group.empty:
            continue
        style = _style_for_arm(arm)
        smoothed = group["value"].ewm(span=50, adjust=False).mean()
        inset.plot(
            group["step"],
            smoothed,
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.0,
            label=arm,
        )
        plotted = True

    inset.set_yscale("log")
    inset.set_title("Forward loss", fontsize=7)
    inset.set_xlabel("step", fontsize=6)
    inset.set_ylabel("loss", fontsize=6)
    inset.xaxis.set_major_formatter(FuncFormatter(_k_formatter))
    inset.tick_params(axis="both", labelsize=6, length=2.2, width=0.5)
    inset.grid(axis="y", alpha=0.25, linewidth=0.4)
    inset.text(
        0.03,
        0.08,
        "floor\n0.004-0.009",
        transform=inset.transAxes,
        fontsize=6,
        color="#444444",
        va="bottom",
    )
    if plotted:
        inset.legend(loc="upper right", fontsize=6, handlelength=1.2, borderpad=0.2)
    else:
        inset.text(
            0.5,
            0.5,
            "no forward-loss scalar",
            ha="center",
            va="center",
            transform=inset.transAxes,
            fontsize=6,
        )


def _finish_axes(axes: list[Axes] | np.ndarray) -> None:
    for ax in axes:
        ax.set_xlabel("checkpoint step")
        ax.xaxis.set_major_formatter(FuncFormatter(_k_formatter))
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax.set_xlim(0, TOTAL_STEPS * 1.02)


def _save_png_pdf(fig: plt.Figure, output_dir: Path, stem: str) -> tuple[Path, Path]:
    png = output_dir / f"{stem}.png"
    pdf = output_dir / f"{stem}.pdf"
    savefig_academic(fig, png, close=False)
    savefig_academic(fig, pdf, close=True)
    return png, pdf


def plot_main(metrics: pd.DataFrame, forward: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    setup_academic_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.85), sharex=True, constrained_layout=False)
    fig.subplots_adjust(left=0.075, right=0.99, top=0.92, bottom=0.34, wspace=0.22)

    _plot_metric_series(axes[0], metrics, "artifact_score", MAIN_ARMS)
    _plot_metric_series(axes[1], metrics, "raw_control_corr", MAIN_ARMS)
    _add_forward_inset(axes[0], forward)

    axes[0].set_title("(a) Artifact drift")
    axes[0].set_ylabel(METRIC_COLUMNS["artifact_score"])
    axes[1].set_title("(b) Raw-control agreement")
    axes[1].set_ylabel(METRIC_COLUMNS["raw_control_corr"])
    _finish_axes(axes)

    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.10), ncol=3)
    fig.text(
        0.01,
        0.01,
        "TensorBoard-scale real-eval proxies; hollow circles mark canonical checkpoints, x marks 60K endpoints.",
        fontsize=7,
    )
    return _save_png_pdf(fig, output_dir, "fig03_nullspace_drift")


def plot_v9a_companion(metrics: pd.DataFrame, output_dir: Path) -> tuple[Path, Path]:
    setup_academic_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.7), sharex=True)
    group = metrics[metrics["arm"] == COMPANION_ARM].sort_values("step")

    for ax, metric in zip(axes, ("artifact_score", "raw_control_corr")):
        if group.empty:
            ax.text(
                0.5,
                0.5,
                "V9A TensorBoard real-eval not available",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue
        style = _style_for_arm(COMPANION_ARM)
        ax.plot(
            group["step"],
            group[metric],
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            label=_arm_label(COMPANION_ARM, group),
        )
        if COMPANION_ARM in TRAINING_ARMS and int(group["step"].max()) < TOTAL_STEPS:
            last = group.iloc[-1]
            ax.scatter(
                [last["step"]],
                [last[metric]],
                marker="x",
                s=52,
                color=style["color"],
                linewidths=1.3,
                zorder=5,
            )

    axes[0].set_title("(a) Artifact proxy")
    axes[0].set_ylabel(METRIC_COLUMNS["artifact_score"])
    axes[1].set_title("(b) Raw-control agreement")
    axes[1].set_ylabel(METRIC_COLUMNS["raw_control_corr"])
    _finish_axes(axes)
    if not group.empty:
        axes[1].legend(loc="best")

    fig.text(
        0.01,
        -0.02,
        "Companion only: V9A uses hybrid drizzle input, so its proxy scale is not cross-mode comparable.",
        fontsize=7,
    )
    return _save_png_pdf(fig, output_dir, "fig03s_v9a_trajectory")


def _summarise(metrics: pd.DataFrame, forward: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, int | str | None]] = []
    for arm in PLOT_ARMS:
        metric_group = metrics[metrics["arm"] == arm]
        forward_group = forward[forward["arm"] == arm]
        rows.append(
            {
                "arm": arm,
                "eval_min": int(metric_group["step"].min()) if not metric_group.empty else None,
                "eval_max": int(metric_group["step"].max()) if not metric_group.empty else None,
                "eval_count": int(len(metric_group)),
                "forward_min": int(forward_group["step"].min()) if not forward_group.empty else None,
                "forward_max": int(forward_group["step"].max()) if not forward_group.empty else None,
                "forward_count": int(len(forward_group)),
                "status": (
                    "training"
                    if arm in TRAINING_ARMS
                    and (metric_group.empty or int(metric_group["step"].max()) < TOTAL_STEPS)
                    else "complete"
                ),
            }
        )
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, object]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = _read_metrics(args.input_csv, refresh=args.refresh)
    forward = _read_forward_loss(args.forward_csv, refresh=args.refresh)

    main_png, main_pdf = plot_main(metrics, forward, output_dir)
    v9a_png, v9a_pdf = plot_v9a_companion(metrics, output_dir)

    summary = _summarise(metrics, forward)
    print("Available TensorBoard steps:")
    print(summary.to_string(index=False))
    print(f"Wrote {_rel(main_png)}")
    print(f"Wrote {_rel(main_pdf)}")
    print(f"Wrote {_rel(v9a_png)}")
    print(f"Wrote {_rel(v9a_pdf)}")
    return {
        "summary": summary,
        "outputs": [main_png, main_pdf, v9a_png, v9a_pdf],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument("--forward-csv", type=Path, default=DEFAULT_FORWARD_OUTPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_PAPER_OUTPUT_DIR)
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Re-read TensorBoard logs, refresh forward_loss_curves.csv, and redraw figures.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
