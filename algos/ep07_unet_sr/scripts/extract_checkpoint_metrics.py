#!/usr/bin/env python3
"""Extract real-eval checkpoint metrics from EP07 TensorBoard logs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EP07_OUTPUTS = PROJECT_ROOT / "algos" / "ep07_unet_sr" / "outputs"
DEFAULT_OUTPUT_CSV = (
    PROJECT_ROOT
    / "output"
    / "ep11_dl_benchmark"
    / "checkpoint_selection"
    / "checkpoint_metrics.csv"
)
DEFAULT_FORWARD_OUTPUT_CSV = (
    PROJECT_ROOT
    / "output"
    / "ep11_dl_benchmark"
    / "checkpoint_selection"
    / "forward_loss_curves.csv"
)

DEFAULT_ARMS: dict[str, str] = {
    "v6": "ep07_v6_physics",
    "v8.1a": "ep07_v8_1a_loss_cooldown",
    "v8.1b": "ep07_v8_1b_pixelshuffle",
    "v9b": "ep07_v9b_fwd_consistency",
}
EXTRA_ARMS: dict[str, str] = {
    "v9a": "ep07_v9a_hybrid_drizzle",
    "v9d": "ep07_v9d_fwd_fullband",
}
ARMS: dict[str, str] = {**DEFAULT_ARMS, **EXTRA_ARMS}
DEFAULT_FORWARD_ARMS = ("v6", "v9b", "v9d")

SCALAR_TAGS = {
    "artifact_score": "eval_real/artifact_score",
    "raw_control_corr": "eval_real/raw_control_corr",
}
FORWARD_LOSS_TAG = "loss/forward_model"


def _dedup_latest(events: list) -> dict[int, float]:
    """Return step -> latest scalar value, keeping the final duplicate step."""

    values: dict[int, float] = {}
    for event in events:
        values[int(event.step)] = float(event.value)
    return values


def _normalise_arm_list(raw_arms: list[str] | tuple[str, ...] | None) -> list[str]:
    if raw_arms is None:
        return list(DEFAULT_ARMS)

    arms: list[str] = []
    for item in raw_arms:
        for arm in str(item).split(","):
            arm = arm.strip()
            if not arm:
                continue
            if arm == "all":
                arms.extend(ARMS)
            else:
                arms.append(arm)

    unknown = sorted(set(arms) - set(ARMS))
    if unknown:
        raise KeyError(f"Unknown arm(s): {unknown}. Known arms: {sorted(ARMS)}")

    deduped: list[str] = []
    seen: set[str] = set()
    for arm in arms:
        if arm not in seen:
            deduped.append(arm)
            seen.add(arm)
    return deduped


def _load_accumulator(arm: str, run_dir: Path) -> EventAccumulator:
    tb_dir = run_dir / "tb_logs"
    if not tb_dir.exists():
        raise FileNotFoundError(f"TensorBoard log dir not found for {arm}: {tb_dir}")

    accumulator = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
    return accumulator


def extract_arm_metrics(arm: str, run_dir: Path) -> list[dict[str, float | int | str]]:
    accumulator = _load_accumulator(arm, run_dir)
    available_tags = set(accumulator.Tags().get("scalars", []))
    missing = [tag for tag in SCALAR_TAGS.values() if tag not in available_tags]
    if missing:
        raise KeyError(f"{arm} missing scalar tag(s): {missing}")

    by_metric = {
        metric: _dedup_latest(accumulator.Scalars(tag))
        for metric, tag in SCALAR_TAGS.items()
    }
    common_steps = sorted(set.intersection(*(set(values) for values in by_metric.values())))
    rows: list[dict[str, float | int | str]] = []
    for step in common_steps:
        rows.append(
            {
                "arm": arm,
                "step": int(step),
                "artifact_score": by_metric["artifact_score"][step],
                "raw_control_corr": by_metric["raw_control_corr"][step],
            }
        )
    return rows


def extract_all(
    output_csv: Path | None = DEFAULT_OUTPUT_CSV,
    *,
    arms: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    selected_arms = _normalise_arm_list(arms)
    rows: list[dict[str, float | int | str]] = []
    for arm in selected_arms:
        dirname = ARMS[arm]
        rows.extend(extract_arm_metrics(arm, EP07_OUTPUTS / dirname))
    df = pd.DataFrame(rows, columns=["arm", "step", "artifact_score", "raw_control_corr"])
    df["arm"] = pd.Categorical(df["arm"], categories=selected_arms, ordered=True)
    df = df.sort_values(["arm", "step"]).reset_index(drop=True)
    df["arm"] = df["arm"].astype(str)

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df


def extract_arm_forward_loss(
    arm: str,
    run_dir: Path,
    *,
    strict: bool = False,
) -> list[dict[str, float | int | str]]:
    accumulator = _load_accumulator(arm, run_dir)
    available_tags = set(accumulator.Tags().get("scalars", []))
    if FORWARD_LOSS_TAG not in available_tags:
        message = f"{arm} missing scalar tag: {FORWARD_LOSS_TAG}"
        if strict:
            raise KeyError(message)
        print(f"WARNING: {message}")
        return []

    values = _dedup_latest(accumulator.Scalars(FORWARD_LOSS_TAG))
    return [
        {"arm": arm, "step": int(step), "value": float(value)}
        for step, value in sorted(values.items())
    ]


def extract_forward_loss(
    output_csv: Path | None = DEFAULT_FORWARD_OUTPUT_CSV,
    *,
    arms: list[str] | tuple[str, ...] | None = DEFAULT_FORWARD_ARMS,
    strict: bool = False,
) -> pd.DataFrame:
    selected_arms = _normalise_arm_list(arms)
    rows: list[dict[str, float | int | str]] = []
    for arm in selected_arms:
        dirname = ARMS[arm]
        rows.extend(extract_arm_forward_loss(arm, EP07_OUTPUTS / dirname, strict=strict))

    df = pd.DataFrame(rows, columns=["arm", "step", "value"])
    if not df.empty:
        df["arm"] = pd.Categorical(df["arm"], categories=selected_arms, ordered=True)
        df = df.sort_values(["arm", "step"]).reset_index(drop=True)
        df["arm"] = df["arm"].astype(str)

    if output_csv is not None:
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    parser.add_argument(
        "--arms",
        nargs="+",
        default=None,
        help=(
            "Arms to extract. Defaults to the original EP11 set "
            "(v6 v8.1a v8.1b v9b). Use 'all' or add v9a/v9d explicitly."
        ),
    )
    parser.add_argument(
        "--extract-forward-loss",
        action="store_true",
        help="Also extract TensorBoard loss/forward_model curves.",
    )
    parser.add_argument(
        "--forward-output-csv",
        type=Path,
        default=DEFAULT_FORWARD_OUTPUT_CSV,
    )
    parser.add_argument(
        "--forward-arms",
        nargs="+",
        default=None,
        help="Forward-loss arms. Defaults to v6 v9b v9d when --extract-forward-loss is set.",
    )
    parser.add_argument(
        "--strict-forward-loss",
        action="store_true",
        help="Fail if a requested forward-loss arm is missing loss/forward_model.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = extract_all(args.output_csv, arms=args.arms)
    counts = df.groupby("arm", sort=False)["step"].agg(["min", "max", "count"])
    print(f"Wrote {args.output_csv}")
    print(counts.to_string())
    if args.extract_forward_loss:
        forward_arms = args.forward_arms if args.forward_arms is not None else DEFAULT_FORWARD_ARMS
        fwd = extract_forward_loss(
            args.forward_output_csv,
            arms=forward_arms,
            strict=args.strict_forward_loss,
        )
        print(f"Wrote {args.forward_output_csv}")
        if fwd.empty:
            print("No forward-loss rows extracted.")
        else:
            fwd_counts = fwd.groupby("arm", sort=False)["step"].agg(["min", "max", "count"])
            print(fwd_counts.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
