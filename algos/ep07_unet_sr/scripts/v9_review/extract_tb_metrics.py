"""Extract EP07 V9 TensorBoard eval scalars into a reproducible CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from common import DEFAULT_EVAL_RUNS, DEFAULT_OUTPUT_DIR, DEFAULT_RUN_ROOT, ensure_output_dirs, project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=project_path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-root", type=project_path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--runs", nargs="+", default=DEFAULT_EVAL_RUNS, help="Run directory names under --run-root.")
    parser.add_argument(
        "--include-forward-loss",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Also include loss/forward_model when present.",
    )
    return parser.parse_args()


def should_keep_tag(tag: str, *, include_forward_loss: bool) -> bool:
    return tag.startswith("eval_real") or (include_forward_loss and tag == "loss/forward_model")


def extract_metrics(run_root: Path, runs: list[str], *, include_forward_loss: bool) -> pd.DataFrame:
    rows: list[dict[str, int | float | str]] = []
    for run in runs:
        logdir = run_root / run / "tb_logs"
        if not logdir.exists():
            print(f"skip missing tb_logs: {logdir}")
            continue
        for ev_file in sorted(logdir.glob("events.*")):
            acc = EventAccumulator(str(ev_file), size_guidance={"scalars": 0})
            acc.Reload()
            for tag in acc.Tags().get("scalars", []):
                if not should_keep_tag(tag, include_forward_loss=include_forward_loss):
                    continue
                for event in acc.Scalars(tag):
                    rows.append({"run": run, "tag": tag, "step": event.step, "value": event.value})
    if not rows:
        return pd.DataFrame(columns=["run", "tag", "step", "value"])
    df = pd.DataFrame(rows).drop_duplicates(subset=["run", "tag", "step"], keep="last")
    return df.sort_values(["run", "tag", "step"]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    ensure_output_dirs(args.output_dir)
    df = extract_metrics(args.run_root, args.runs, include_forward_loss=args.include_forward_loss)
    out_csv = args.output_dir / "ep07_eval_real_metrics.csv"
    df.to_csv(out_csv, index=False)

    if not df.empty:
        pivot = df[df.tag.str.startswith("eval_real")].pivot_table(
            index=["run", "step"], columns="tag", values="value"
        )
        pd.set_option("display.width", 200)
        print(pivot.to_string(float_format=lambda v: f"{v:.4f}"))
    print(f"\nsaved -> {out_csv}")


if __name__ == "__main__":
    main()
