"""Render tracked V9 review comparison panels from inputs and checkpoints."""

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
    default_panel_specs,
    ensure_output_dirs,
    highpass_fine,
    infer_checkpoint_cached,
    load_real_inputs,
    metric_row,
    parse_checkpoint_spec,
    project_path,
    save_fine_panel,
    save_temperature_panel,
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
        help="Checkpoint spec. Defaults to v8.1a@60K, V9A@10K, V9A@60K.",
    )
    return parser.parse_args()


def specs_from_args(args: argparse.Namespace) -> list[CheckpointSpec]:
    if args.checkpoint:
        return [parse_checkpoint_spec(text, run_root=args.run_root) for text in args.checkpoint]
    return default_panel_specs(args.run_root)


def main() -> None:
    args = parse_args()
    ensure_output_dirs(args.output_dir, args.cache_dir)
    specs = specs_from_args(args)

    raw_frames, shifts, refs = load_real_inputs(
        frame_limit=args.frame_limit,
        alignment_method=args.alignment_method,
        tgv_path=args.tgv_path,
    )

    save_temperature_panel(args.output_dir / "raw_bicubic2x_center_zoom3x.png", refs.raw_control, title="raw_bicubic2x")
    save_temperature_panel(
        args.output_dir / "input_drizzle2x_mean_center_zoom3x.png",
        refs.drizzle_mean,
        title="input_drizzle2x_mean",
    )
    save_temperature_panel(args.output_dir / "ep10_tgv_best_center_zoom3x.png", refs.tgv, title="ep10_tgv_best")

    predictions: dict[str, np.ndarray] = {}
    for spec in specs:
        predictions[spec.label] = infer_checkpoint_cached(
            spec,
            cache_dir=args.cache_dir,
            raw_frames=raw_frames,
            shifts=shifts,
            device=args.device,
            force=args.force,
            patch_size_hr=args.patch_size_hr,
            overlap=args.overlap,
        )
        print(f"loaded {spec.label} step={spec.step}")

    save_fine_panel(
        args.output_dir / "tight_center_inputs.png",
        [
            ("raw bicubic 2x", refs.raw_control),
            ("network INPUT: drizzle2x mean", refs.drizzle_mean),
            ("EP10 TGV best", refs.tgv),
        ],
        title="Fine zigzag inputs and classical references",
        ncols=3,
    )

    comparison_panels = [
        ("raw bicubic 2x (control)", refs.raw_control),
        ("network INPUT: drizzle2x mean", refs.drizzle_mean),
        ("EP10 TGV best (classical)", refs.tgv),
    ] + [(spec.label, predictions[spec.label]) for spec in specs if spec.label.startswith("v9a")]
    save_fine_panel(
        args.output_dir / "fine_zigzag_comparison.png",
        comparison_panels,
        title="Fine zigzag fused area: what the network sees vs what it outputs",
        ncols=3,
    )

    final_panels = [
        ("network INPUT: drizzle2x mean", refs.drizzle_mean),
        ("EP10 TGV best (classical)", refs.tgv),
    ] + [(spec.label, predictions[spec.label]) for spec in specs]
    save_fine_panel(
        args.output_dir / "fine_zigzag_final_panel.png",
        final_panels,
        title="Fine zigzag area: old input vs hybrid input vs classical",
        ncols=3,
    )

    drizzle_hp = highpass_fine(refs.drizzle_mean)
    tgv_hp = highpass_fine(refs.tgv)
    rows = []
    for spec in specs:
        rows.append(
            metric_row(
                spec.label,
                predictions[spec.label],
                drizzle_hp_fine=drizzle_hp,
                tgv_hp_fine=tgv_hp,
                step=spec.step,
            )
        )
    df = pd.DataFrame(rows)
    scores_path = args.output_dir / "fine_zigzag_panel_metrics.csv"
    df.to_csv(scores_path, index=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"saved -> {scores_path}")


if __name__ == "__main__":
    main()
