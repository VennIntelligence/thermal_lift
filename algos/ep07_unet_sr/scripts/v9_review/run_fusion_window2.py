"""Validate the zero-training fusion baseline on an independent second ROI.

This is the E3 check for Supplement D.7.  It intentionally uses cached
full-frame predictions and CPU-only real-data references; it does not run
checkpoint inference.

The original D.7 fine window is rows 384:518, cols 478:674 on the 2x grid.
Window 2 keeps the same size and columns, then shifts down by one window
height plus a 24 px gap: rows 542:676, cols 478:674.  This creates a fixed,
non-overlapping validation ROI before any lambda re-selection.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
from tcforge.highpass import highpass_preprocess

from common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_TGV_PATH,
    HIGHPASS_SIGMA,
    ensure_output_dirs,
    lattice_score,
    load_real_inputs,
    pearson_finite,
    project_path,
    sharp_p95,
)
from run_fusion_baseline import parse_lambdas

ORIGINAL_WINDOW = (slice(384, 518), slice(478, 674))
WINDOW2 = (slice(542, 676), slice(478, 674))

V10_NAME = "v10hl_lam120_15k"
ORIGINAL_FUSION_CSV = DEFAULT_OUTPUT_DIR / "fusion_baseline_metrics.csv"
ORIGINAL_V10_CSV = DEFAULT_OUTPUT_DIR / "v10_highlam" / "v9a_pareto_metrics.csv"


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
        "--v10-path",
        type=project_path,
        default=DEFAULT_CACHE_DIR / "v10hl_lam120_15k_step15000_temperature.npy",
        help="Cached V10 high-lambda working-point full-frame temperature npy.",
    )
    parser.add_argument(
        "--original-fusion-csv",
        type=project_path,
        default=ORIGINAL_FUSION_CSV,
        help="Original D.7 fine-window fusion CSV for cross-window comparison.",
    )
    parser.add_argument(
        "--original-v10-csv",
        type=project_path,
        default=ORIGINAL_V10_CSV,
        help="Original D.7/V10 fine-window metrics CSV for cross-window comparison.",
    )
    parser.add_argument(
        "--lambdas",
        default="0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
        help="Comma-separated lambda values.",
    )
    return parser.parse_args()


def window_name(window: tuple[slice, slice]) -> str:
    rows, cols = window
    return f"rows{rows.start}:{rows.stop}_cols{cols.start}:{cols.stop}"


def window_view(image: np.ndarray, window: tuple[slice, slice]) -> np.ndarray:
    rows, cols = window
    return np.asarray(image)[rows, cols]


def highpass_window(
    image: np.ndarray,
    window: tuple[slice, slice],
    *,
    sigma_bg: float = HIGHPASS_SIGMA,
) -> np.ndarray:
    return window_view(highpass_preprocess(image, sigma_bg=sigma_bg), window)


def metric_row_window(
    name: str,
    temp: np.ndarray,
    *,
    window: tuple[slice, slice],
    drizzle_hp: np.ndarray,
    tgv_hp: np.ndarray,
    kind: str,
    step: int | None = None,
    anchor: str = "",
    unet_pred: str = "",
    lam: float | None = None,
) -> dict[str, float | int | str | None]:
    hp = highpass_window(temp, window)
    return {
        "name": name,
        "step": step,
        "window": window_name(window),
        "hp_corr_input": pearson_finite(hp, drizzle_hp),
        "hp_corr_tgv": pearson_finite(hp, tgv_hp),
        "sharp_p95": sharp_p95(window_view(temp, window)),
        "lattice_score": lattice_score(hp),
        "kind": kind,
        "anchor": anchor,
        "unet_pred": unet_pred,
        "lambda": lam,
    }


def load_required_npy(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"required cached full-frame npy is missing: {path}")
    return np.load(path).astype(np.float32, copy=False)


def compute_window2_rows(
    *,
    refs: dict[str, np.ndarray],
    predictions: dict[str, np.ndarray],
    v10: np.ndarray,
    lambdas: list[float],
    window: tuple[slice, slice],
) -> pd.DataFrame:
    drizzle_hp = highpass_window(refs["drizzle"], window)
    tgv_hp = highpass_window(refs["tgv"], window)
    rows: list[dict[str, float | int | str | None]] = [
        metric_row_window(
            "input_drizzle",
            refs["drizzle"],
            window=window,
            drizzle_hp=drizzle_hp,
            tgv_hp=tgv_hp,
            kind="reference",
        ),
        metric_row_window(
            "tgv",
            refs["tgv"],
            window=window,
            drizzle_hp=drizzle_hp,
            tgv_hp=tgv_hp,
            kind="reference",
        ),
        metric_row_window(
            V10_NAME,
            v10,
            window=window,
            drizzle_hp=drizzle_hp,
            tgv_hp=tgv_hp,
            kind="v10",
            step=15000,
        ),
    ]
    anchors = {"drizzle": refs["drizzle"], "tgv": refs["tgv"]}
    for anchor_name, anchor in anchors.items():
        for pred_name, pred in predictions.items():
            if anchor.shape != pred.shape:
                raise ValueError(f"shape mismatch: {anchor_name} {anchor.shape} vs {pred_name} {pred.shape}")
            for lam in lambdas:
                fused = (1.0 - lam) * anchor + lam * pred
                rows.append(
                    metric_row_window(
                        f"fusion_{anchor_name}_{pred_name}_lam{lam:.1f}",
                        fused,
                        window=window,
                        drizzle_hp=drizzle_hp,
                        tgv_hp=tgv_hp,
                        kind="fusion",
                        anchor=anchor_name,
                        unet_pred=pred_name,
                        lam=lam,
                    )
                )
    return pd.DataFrame(rows)


def add_tgv_relative_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    tgv_ref = out[(out["kind"] == "reference") & (out["name"] == "tgv")].iloc[0]
    out["delta_hp_corr_input_vs_tgv"] = out["hp_corr_input"] - float(tgv_ref["hp_corr_input"])
    out["delta_sharp_p95_vs_tgv"] = out["sharp_p95"] - float(tgv_ref["sharp_p95"])
    out["delta_lattice_vs_tgv"] = out["lattice_score"] - float(tgv_ref["lattice_score"])
    out["frontier_vs_tgv"] = (
        (out["delta_hp_corr_input_vs_tgv"] >= 0)
        & (out["delta_sharp_p95_vs_tgv"] >= 0)
        & (out["delta_lattice_vs_tgv"] <= 0)
    )
    return out


def select_tgv_v9a60_lambda(df: pd.DataFrame) -> tuple[pd.Series, pd.DataFrame]:
    """Re-select lambda using only the second-window proxy frontier.

    Primary gate: TGV anchor, V9A-60K prediction, lambda > 0, with fidelity and
    sharpness at least the local TGV reference and lattice no worse than TGV.
    Tie-breaker: highest hp_corr_input, then lowest lattice, then highest
    sharpness.  If the strict gate is empty, use the same sort on all lambda > 0
    TGV x V9A-60K candidates and report that it is not a gated frontier point.
    """

    candidates = df[
        (df["kind"] == "fusion")
        & (df["anchor"] == "tgv")
        & (df["unet_pred"] == "v9a60")
        & (df["lambda"] > 0)
    ].copy()
    pool = candidates[candidates["frontier_vs_tgv"]].copy()
    if pool.empty:
        pool = candidates.copy()
    pool = pool.sort_values(
        ["frontier_vs_tgv", "hp_corr_input", "lattice_score", "sharp_p95"],
        ascending=[False, False, True, False],
    )
    return pool.iloc[0], candidates


def original_selected_lambda(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    if "window" not in df.columns:
        df["window"] = window_name(ORIGINAL_WINDOW)
    df = add_tgv_relative_columns(df)
    candidates = df[
        (df["kind"] == "fusion")
        & (df["anchor"] == "tgv")
        & (df["unet_pred"] == "v9a60")
        & (df["lambda"] > 0)
    ].copy()
    pool = candidates[candidates["frontier_vs_tgv"]].copy()
    if pool.empty:
        pool = candidates.copy()
    pool = pool.sort_values(
        ["frontier_vs_tgv", "hp_corr_input", "lattice_score", "sharp_p95"],
        ascending=[False, False, True, False],
    )
    return pool.iloc[0]


def original_v10_row(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    row = df[df["name"] == V10_NAME]
    if row.empty:
        raise ValueError(f"{V10_NAME} not found in {path}")
    return row.iloc[0]


def fmt(value: float) -> str:
    return f"{float(value):.4f}"


def dataframe_to_markdown(df: pd.DataFrame, columns: list[str]) -> str:
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        vals: list[str] = []
        for col in columns:
            val = row[col]
            if isinstance(val, float):
                vals.append(fmt(val))
            else:
                vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def write_summary(
    path: Path,
    *,
    metrics_csv: Path,
    df: pd.DataFrame,
    selected: pd.Series,
    original_selected: pd.Series,
    original_v10: pd.Series,
) -> None:
    refs = df[df["kind"].isin(["reference", "v10"])].copy()
    tgv_ref = refs[refs["name"] == "tgv"].iloc[0]
    v10_row = refs[refs["name"] == V10_NAME].iloc[0]
    selected_table = pd.DataFrame([original_selected, selected])[
        ["name", "window", "lambda", "hp_corr_input", "hp_corr_tgv", "sharp_p95", "lattice_score", "frontier_vs_tgv"]
    ].copy()
    selected_table.iloc[0, selected_table.columns.get_loc("window")] = window_name(ORIGINAL_WINDOW)
    v10_compare = pd.DataFrame(
        [
            {
                "name": V10_NAME,
                "window": window_name(ORIGINAL_WINDOW),
                "hp_corr_input": original_v10["hp_corr_input"],
                "hp_corr_tgv": original_v10["hp_corr_tgv"],
                "sharp_p95": original_v10["sharp_p95"],
                "lattice_score": original_v10["lattice_score"],
            },
            {
                "name": V10_NAME,
                "window": window_name(WINDOW2),
                "hp_corr_input": v10_row["hp_corr_input"],
                "hp_corr_tgv": v10_row["hp_corr_tgv"],
                "sharp_p95": v10_row["sharp_p95"],
                "lattice_score": v10_row["lattice_score"],
            },
        ]
    )
    stable_lambda = np.isclose(float(original_selected["lambda"]), float(selected["lambda"]))
    v10_same_side = (float(original_v10["hp_corr_input"]) < 0.959773) and (
        float(v10_row["hp_corr_input"]) < float(tgv_ref["hp_corr_input"])
    )
    lines = [
        "# Fusion Window 2 Summary",
        "",
        f"- CPU-only entry: `CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', '')}`",
        f"- Original window: `{window_name(ORIGINAL_WINDOW)}`",
        f"- Window 2 rule: same shape, shifted down by one original-window height plus 24 px; `{window_name(WINDOW2)}`.",
        f"- CSV: `{metrics_csv}`",
        "",
        "## Re-selected TGV x V9A-60K Lambda",
        "",
        dataframe_to_markdown(
            selected_table,
            ["name", "window", "lambda", "hp_corr_input", "hp_corr_tgv", "sharp_p95", "lattice_score", "frontier_vs_tgv"],
        ),
        "",
        (
            f"Selected lambda is {'stable' if stable_lambda else 'not stable'} across the two windows "
            f"({float(original_selected['lambda']):.1f} -> {float(selected['lambda']):.1f})."
        ),
        "",
        "## V10 Comparison Point",
        "",
        dataframe_to_markdown(
            v10_compare,
            ["name", "window", "hp_corr_input", "hp_corr_tgv", "sharp_p95", "lattice_score"],
        ),
        "",
        (
            "V10 stays on the same side of the TGV fidelity reference in both windows "
            if v10_same_side
            else "V10 changes side relative to the TGV fidelity reference across windows "
        )
        + "(lower `hp_corr_input` than TGV); this is a proxy-stability readout, not a fidelity proof.",
        "",
        "## Window 2 Reference Rows",
        "",
        dataframe_to_markdown(
            refs,
            ["name", "hp_corr_input", "hp_corr_tgv", "sharp_p95", "lattice_score"],
        ),
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_output_dirs(args.output_dir, args.cache_dir)
    lambdas = parse_lambdas(args.lambdas)

    _, _, refs = load_real_inputs(
        frame_limit=args.frame_limit,
        alignment_method=args.alignment_method,
        tgv_path=args.tgv_path,
    )
    refs_dict = {
        "drizzle": refs.drizzle_mean,
        "tgv": refs.tgv,
    }
    predictions = {
        "v9a20": load_required_npy(args.v9a20_path),
        "v9a60": load_required_npy(args.v9a60_path),
    }
    v10 = load_required_npy(args.v10_path)

    df = compute_window2_rows(
        refs=refs_dict,
        predictions=predictions,
        v10=v10,
        lambdas=lambdas,
        window=WINDOW2,
    )
    df = add_tgv_relative_columns(df)
    selected, tgv_v9a60 = select_tgv_v9a60_lambda(df)
    original_selected = original_selected_lambda(args.original_fusion_csv)
    original_v10 = original_v10_row(args.original_v10_csv)

    metrics_csv = args.output_dir / "fusion_window2_metrics.csv"
    summary_md = args.output_dir / "fusion_window2_summary.md"
    df.to_csv(metrics_csv, index=False)
    write_summary(
        summary_md,
        metrics_csv=metrics_csv,
        df=df,
        selected=selected,
        original_selected=original_selected,
        original_v10=original_v10,
    )

    print(f"window2={window_name(WINDOW2)}")
    print("TGV x V9A-60K candidates:")
    print(
        tgv_v9a60[
            [
                "lambda",
                "hp_corr_input",
                "hp_corr_tgv",
                "sharp_p95",
                "lattice_score",
                "frontier_vs_tgv",
            ]
        ].to_string(index=False, float_format=lambda v: f"{v:.4f}")
    )
    print()
    print(
        "selected lambda="
        f"{float(selected['lambda']):.1f} "
        f"hp_corr_input={fmt(selected['hp_corr_input'])} "
        f"hp_corr_tgv={fmt(selected['hp_corr_tgv'])} "
        f"sharp_p95={fmt(selected['sharp_p95'])} "
        f"lattice={fmt(selected['lattice_score'])} "
        f"frontier_vs_tgv={bool(selected['frontier_vs_tgv'])}"
    )
    v10_row = df[df["name"] == V10_NAME].iloc[0]
    print(
        f"v10 {V10_NAME}: "
        f"hp_corr_input={fmt(v10_row['hp_corr_input'])} "
        f"hp_corr_tgv={fmt(v10_row['hp_corr_tgv'])} "
        f"sharp_p95={fmt(v10_row['sharp_p95'])} "
        f"lattice={fmt(v10_row['lattice_score'])}"
    )
    print(f"saved -> {metrics_csv}")
    print(f"saved -> {summary_md}")


if __name__ == "__main__":
    main()
