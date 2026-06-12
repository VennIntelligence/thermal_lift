"""Run zero-training linear fusion baselines for V10.

The baseline is:

    fused(lambda) = (1 - lambda) * anchor + lambda * unet_pred

where anchor is either the real-data drizzle mean or EP10 TGV, and unet_pred is
an existing V9A checkpoint prediction cached as npy.  This script never runs
checkpoint inference; missing UNet caches are treated as an explicit error so
CPU/GPU use stays under operator control.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TGV_PATH,
    configure_matplotlib,
    ensure_output_dirs,
    highpass_fine,
    load_real_inputs,
    metric_row,
    project_path,
)

TGV_HP_CORR_INPUT = 0.960
TGV_SHARP_P95 = 0.96


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=project_path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=project_path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--tgv-path", type=project_path, default=DEFAULT_TGV_PATH)
    parser.add_argument("--frame-limit", type=int, default=248)
    parser.add_argument("--alignment-method", default="contour_refined")
    parser.add_argument(
        "--v9a20-path",
        type=project_path,
        default=DEFAULT_CACHE_DIR / "v9a_20k_step20000_temperature.npy",
        help="Cached V9A 20K full-frame temperature npy.",
    )
    parser.add_argument(
        "--v9a60-path",
        type=project_path,
        default=DEFAULT_CACHE_DIR / "v9a_60k_step60000_temperature.npy",
        help="Cached V9A 60K full-frame temperature npy.",
    )
    parser.add_argument(
        "--v9a-metrics-csv",
        type=project_path,
        default=DEFAULT_OUTPUT_DIR / "v9a_pareto_metrics.csv",
        help="Optional V9A trajectory CSV from run_pareto_sweep.py for plot overlay.",
    )
    parser.add_argument(
        "--lambdas",
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated lambda values.",
    )
    return parser.parse_args()


def parse_lambdas(text: str) -> list[float]:
    values = [float(item.strip()) for item in text.split(",") if item.strip()]
    if not values:
        raise ValueError("expected at least one lambda value")
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("lambda values must be in [0, 1]")
    return values


def load_required_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"required cached prediction is missing: {path}\n"
            "Build or copy the V9A npy cache first; this script intentionally does not run checkpoint inference."
        )
    return np.load(path).astype(np.float32, copy=False)


def compute_fusion_rows(
    *,
    anchors: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    lambdas: list[float],
    drizzle_hp: np.ndarray,
    tgv_hp: np.ndarray,
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str | None]] = []
    for anchor_name, anchor in anchors.items():
        for pred_name, pred in predictions.items():
            if anchor.shape != pred.shape:
                raise ValueError(f"shape mismatch for {anchor_name} {anchor.shape} vs {pred_name} {pred.shape}")
            for lam in lambdas:
                fused = (1.0 - lam) * anchor + lam * pred
                row = metric_row(
                    f"fusion_{anchor_name}_{pred_name}_lam{lam:.1f}",
                    fused,
                    drizzle_hp_fine=drizzle_hp,
                    tgv_hp_fine=tgv_hp,
                )
                row.update(
                    {
                        "kind": "fusion",
                        "anchor": anchor_name,
                        "unet_pred": pred_name,
                        "lambda": lam,
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def load_v9a_trajectory(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "name" not in df.columns:
        return pd.DataFrame()
    return df[df["name"].astype(str).str.startswith("v9a_")].copy()


def save_fusion_plot(
    path: Path,
    *,
    fusion_df: pd.DataFrame,
    refs_df: pd.DataFrame,
    v9a_df: pd.DataFrame,
) -> Path:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(8.0, 5.8))

    if not v9a_df.empty:
        v9a_df = v9a_df.dropna(subset=["step"]).sort_values("step")
        ax.plot(
            v9a_df["hp_corr_input"],
            v9a_df["sharp_p95"],
            color="0.35",
            marker="o",
            linewidth=1.2,
            markersize=4,
            label="V9A checkpoint trajectory",
            zorder=2,
        )
        for _, row in v9a_df.iterrows():
            ax.annotate(
                f"{int(row.step) // 1000}K",
                (row.hp_corr_input, row.sharp_p95),
                textcoords="offset points",
                xytext=(4, 3),
                fontsize=8,
                color="0.25",
            )

    style = {
        ("drizzle", "v9a20"): ("tab:blue", "o"),
        ("drizzle", "v9a60"): ("tab:cyan", "s"),
        ("tgv", "v9a20"): ("tab:orange", "^"),
        ("tgv", "v9a60"): ("tab:red", "D"),
    }
    for (anchor, pred), sub in fusion_df.groupby(["anchor", "unet_pred"]):
        sub = sub.sort_values("lambda")
        color, marker = style.get((anchor, pred), ("tab:purple", "o"))
        ax.plot(
            sub["hp_corr_input"],
            sub["sharp_p95"],
            color=color,
            marker=marker,
            linewidth=1.8,
            markersize=4,
            label=f"{anchor} -> {pred}",
            zorder=3,
        )
        for lam in (0.0, 0.5, 1.0):
            point = sub[np.isclose(sub["lambda"], lam)]
            if point.empty:
                continue
            row = point.iloc[0]
            ax.annotate(
                f"{lam:.1f}",
                (row.hp_corr_input, row.sharp_p95),
                textcoords="offset points",
                xytext=(5, -10),
                fontsize=8,
                color=color,
            )

    for _, row in refs_df.iterrows():
        marker = "*" if row["name"] == "input_drizzle" else "P"
        ax.scatter(row["hp_corr_input"], row["sharp_p95"], marker=marker, s=180, color="crimson", zorder=4)
        ax.annotate(
            str(row["name"]),
            (row["hp_corr_input"], row["sharp_p95"]),
            textcoords="offset points",
            xytext=(6, -12),
            fontsize=10,
            color="crimson",
        )

    ax.axvline(TGV_HP_CORR_INPUT, color="crimson", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.axhline(TGV_SHARP_P95, color="crimson", linestyle="--", linewidth=1.0, alpha=0.5)
    ax.set_xlabel("Fine-window highpass corr vs drizzle input (fidelity)")
    ax.set_ylabel("Fine-window P95 |gradient| (sharpness proxy)")
    ax.set_title("Zero-training fusion baseline vs V9A and TGV")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def dominance_summary(fusion_df: pd.DataFrame) -> tuple[bool, pd.DataFrame, str]:
    candidates = fusion_df[fusion_df["lambda"] > 0.0].copy()
    candidates["dominates_tgv_workpoint"] = (
        (candidates["hp_corr_input"] >= TGV_HP_CORR_INPUT)
        & (candidates["sharp_p95"] >= TGV_SHARP_P95)
        & (
            (candidates["hp_corr_input"] > TGV_HP_CORR_INPUT)
            | (candidates["sharp_p95"] > TGV_SHARP_P95)
        )
    )
    dominates = bool(candidates["dominates_tgv_workpoint"].any())
    frontier = candidates.sort_values(["dominates_tgv_workpoint", "hp_corr_input", "sharp_p95"], ascending=False)
    best = frontier.head(8)
    if dominates:
        conclusion = "YES: at least one post-hoc linear fusion point strictly dominates the TGV workpoint."
    else:
        conclusion = (
            "NO: no post-hoc linear fusion point with lambda > 0 strictly dominates "
            f"the TGV workpoint ({TGV_HP_CORR_INPUT:.3f}, {TGV_SHARP_P95:.2f})."
        )
    return dominates, best, conclusion


def dataframe_to_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values: list[str] = []
        for col in columns:
            value = row[col]
            if isinstance(value, float):
                values.append(f"{value:.4f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    ensure_output_dirs(args.output_dir, args.cache_dir)
    lambdas = parse_lambdas(args.lambdas)

    _, _, refs = load_real_inputs(
        frame_limit=args.frame_limit,
        alignment_method=args.alignment_method,
        tgv_path=args.tgv_path,
    )
    predictions = {
        "v9a20": load_required_npy(args.v9a20_path),
        "v9a60": load_required_npy(args.v9a60_path),
    }
    anchors = {
        "drizzle": refs.drizzle_mean,
        "tgv": refs.tgv,
    }
    drizzle_hp = highpass_fine(refs.drizzle_mean)
    tgv_hp = highpass_fine(refs.tgv)

    refs_rows = [
        metric_row("input_drizzle", refs.drizzle_mean, drizzle_hp_fine=drizzle_hp, tgv_hp_fine=tgv_hp),
        metric_row("tgv", refs.tgv, drizzle_hp_fine=drizzle_hp, tgv_hp_fine=tgv_hp),
    ]
    refs_df = pd.DataFrame(refs_rows)
    refs_df["kind"] = "reference"
    refs_df["anchor"] = ""
    refs_df["unet_pred"] = ""
    refs_df["lambda"] = np.nan

    fusion_df = compute_fusion_rows(
        anchors=anchors,
        predictions=predictions,
        lambdas=lambdas,
        drizzle_hp=drizzle_hp,
        tgv_hp=tgv_hp,
    )
    out_df = pd.concat([refs_df, fusion_df], ignore_index=True)
    out_csv = args.output_dir / "fusion_baseline_metrics.csv"
    out_df.to_csv(out_csv, index=False)

    v9a_df = load_v9a_trajectory(args.v9a_metrics_csv)
    out_png = args.output_dir / "fusion_pareto_overlay.png"
    save_fusion_plot(out_png, fusion_df=fusion_df, refs_df=refs_df, v9a_df=v9a_df)

    dominates, best, conclusion = dominance_summary(fusion_df)
    summary_path = args.output_dir / "fusion_baseline_summary.md"
    best_columns = [
        "anchor",
        "unet_pred",
        "lambda",
        "hp_corr_input",
        "hp_corr_tgv",
        "sharp_p95",
        "lattice_score",
        "dominates_tgv_workpoint",
    ]
    summary = [
        "# Fusion Baseline Summary",
        "",
        f"- Dominates TGV workpoint: {'yes' if dominates else 'no'}",
        f"- Conclusion: {conclusion}",
        f"- CSV: `{out_csv}`",
        f"- Plot: `{out_png}`",
        "",
        "## Top Candidates",
        "",
        dataframe_to_markdown(best, best_columns),
        "",
    ]
    summary_path.write_text("\n".join(summary), encoding="utf-8")

    print(out_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    print(conclusion)
    print(best.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nsaved -> {out_csv}")
    print(f"saved -> {out_png}")
    print(f"saved -> {summary_path}")


if __name__ == "__main__":
    main()
