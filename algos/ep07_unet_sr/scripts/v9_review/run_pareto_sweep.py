"""Rebuild fine-window V9 Pareto metrics and checkpoint strip figures."""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common import (
    DEFAULT_CACHE_DIR,
    DEFAULT_OUTPUT_DIR,
    DEFAULT_RUN_ROOT,
    DEFAULT_TGV_PATH,
    CheckpointSpec,
    default_v9a_specs,
    ensure_output_dirs,
    highpass_fine,
    infer_checkpoint_cached,
    load_real_inputs,
    metric_row,
    parse_checkpoint_spec,
    project_path,
    save_fine_panel,
    save_pareto_scatter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=project_path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--cache-dir", type=project_path, default=DEFAULT_CACHE_DIR)
    parser.add_argument("--run-root", type=project_path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--tgv-path", type=project_path, default=DEFAULT_TGV_PATH)
    parser.add_argument("--frame-limit", type=int, default=248)
    parser.add_argument("--alignment-method", default="contour_refined")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda", "auto"], help="Default is CPU.")
    parser.add_argument("--force", action="store_true", help="Recompute checkpoint npy caches.")
    parser.add_argument("--patch-size-hr", type=int, default=256)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="LABEL=RUN_DIR:STEP:INPUT_MODE",
        help="Checkpoint spec. RUN_DIR may be absolute or a directory under --run-root.",
    )
    return parser.parse_args()


def specs_from_args(args: argparse.Namespace) -> list[CheckpointSpec]:
    if args.checkpoint:
        return [parse_checkpoint_spec(text, run_root=args.run_root) for text in args.checkpoint]
    return default_v9a_specs(args.run_root)


def main() -> None:
    args = parse_args()
    ensure_output_dirs(args.output_dir, args.cache_dir)
    specs = specs_from_args(args)

    raw_frames, shifts, refs = load_real_inputs(
        frame_limit=args.frame_limit,
        alignment_method=args.alignment_method,
        tgv_path=args.tgv_path,
    )
    drizzle_hp = highpass_fine(refs.drizzle_mean)
    tgv_hp = highpass_fine(refs.tgv)

    rows = [
        metric_row("input_drizzle", refs.drizzle_mean, drizzle_hp_fine=drizzle_hp, tgv_hp_fine=tgv_hp),
        metric_row("tgv", refs.tgv, drizzle_hp_fine=drizzle_hp, tgv_hp_fine=tgv_hp),
    ]
    predictions: dict[str, np.ndarray] = {}
    for spec in specs:
        pred = infer_checkpoint_cached(
            spec,
            cache_dir=args.cache_dir,
            raw_frames=raw_frames,
            shifts=shifts,
            device=args.device,
            force=args.force,
            patch_size_hr=args.patch_size_hr,
            overlap=args.overlap,
        )
        predictions[spec.label] = pred
        rows.append(
            metric_row(
                spec.label,
                pred,
                drizzle_hp_fine=drizzle_hp,
                tgv_hp_fine=tgv_hp,
                step=spec.step,
            )
        )
        print(f"done {spec.label} step={spec.step}")

    df = pd.DataFrame(rows)
    metrics_path = args.output_dir / "v9a_pareto_metrics.csv"
    df.to_csv(metrics_path, index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    save_pareto_scatter(args.output_dir / "v9a_pareto_scatter.png", df)

    strip_specs = specs[:6] + specs[-1:] if len(specs) > 7 else specs
    strip_panels = [("drizzle input", refs.drizzle_mean)] + [
        (spec.label, predictions[spec.label]) for spec in strip_specs
    ]
    save_fine_panel(
        args.output_dir / "v9a_checkpoint_strip.png",
        strip_panels,
        title="V9 checkpoint fine zigzag window (raw 2x grid, no zoom)",
        ncols=4,
    )
    print(f"saved -> {metrics_path}")
    print(f"saved -> {args.output_dir / 'v9a_pareto_scatter.png'}")
    print(f"saved -> {args.output_dir / 'v9a_checkpoint_strip.png'}")


if __name__ == "__main__":
    main()
