#!/usr/bin/env python3
"""Extract real-eval checkpoint metrics from EP07 TensorBoard logs."""

from __future__ import annotations

import argparse
from pathlib import Path

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

ARMS: dict[str, str] = {
    "v6": "ep07_v6_physics",
    "v8.1a": "ep07_v8_1a_loss_cooldown",
    "v8.1b": "ep07_v8_1b_pixelshuffle",
    "v9b": "ep07_v9b_fwd_consistency",
}

SCALAR_TAGS = {
    "artifact_score": "eval_real/artifact_score",
    "raw_control_corr": "eval_real/raw_control_corr",
}


def _dedup_latest(events: list) -> dict[int, float]:
    """Return step -> latest scalar value, keeping the final duplicate step."""

    values: dict[int, float] = {}
    for event in events:
        values[int(event.step)] = float(event.value)
    return values


def extract_arm_metrics(arm: str, run_dir: Path) -> list[dict[str, float | int | str]]:
    tb_dir = run_dir / "tb_logs"
    if not tb_dir.exists():
        raise FileNotFoundError(f"TensorBoard log dir not found for {arm}: {tb_dir}")

    accumulator = EventAccumulator(str(tb_dir), size_guidance={"scalars": 0})
    accumulator.Reload()
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


def extract_all(output_csv: Path) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    for arm, dirname in ARMS.items():
        rows.extend(extract_arm_metrics(arm, EP07_OUTPUTS / dirname))
    df = pd.DataFrame(rows)
    df["arm"] = pd.Categorical(df["arm"], categories=list(ARMS), ordered=True)
    df = df.sort_values(["arm", "step"]).reset_index(drop=True)
    df["arm"] = df["arm"].astype(str)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-csv", type=Path, default=DEFAULT_OUTPUT_CSV)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    df = extract_all(args.output_csv)
    counts = df.groupby("arm", sort=False)["step"].agg(["min", "max", "count"])
    print(f"Wrote {args.output_csv}")
    print(counts.to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
