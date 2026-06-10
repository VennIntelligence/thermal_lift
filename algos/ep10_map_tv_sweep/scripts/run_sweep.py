#!/usr/bin/env python3
"""Run EP10 MAP-TV lambda/PSF sigma joint sweep."""

from __future__ import annotations

from thermal_core.alignment_paths import default_contour_alignment_csv
import argparse
import hashlib
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
ALGO_ROOT = SCRIPT_DIR.parents[0]
PROJECT_ROOT = SCRIPT_DIR.parents[2]
for path in (
    ALGO_ROOT / "src",
    PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src",
    PROJECT_ROOT / "algos" / "ep06_sr_poc" / "scripts",
    PROJECT_ROOT / "core" / "src",
):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from common.alignment import load_alignment_shifts  # noqa: E402
from common.data_loader import (  # noqa: E402
    bicubic_upsample,
    highpass_preprocess,
    load_main_session_frames,
    offset_correction,
)
from common.forward_model import forward  # noqa: E402
from ep10_map_tv_sweep.map_tv_sweep import (  # noqa: E402
    ParamSpec,
    combine_detail_tables,
    evaluate_param,
    existing_completed,
    init_worker,
    psnr,
    reconstruct_for_spec,
    save_best_params,
    save_heatmap,
    save_results_table,
    write_json,
)


LAMBDA_GRID = (0.0001, 0.0003, 0.0005, 0.001, 0.002, 0.005, 0.01)
PSF_GRID = (0.10, 0.18, 0.30, 0.50)


def parse_float_grid(text: str) -> list[float]:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError("grid must contain at least one value")
    return values


def synthetic_fallback(
    *,
    n_frames: int,
    scale: int,
    psf_sigma: float,
    noise_sigma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:128, 0:160]
    truth = 0.3 * np.sin(x / 8.0) + 0.2 * np.cos(y / 11.0)
    truth += ((x > 30) & (x < 120) & (y > 24) & (y < 96)).astype(float)
    truth += 0.8 * ((x - 92) ** 2 + (y - 64) ** 2 < 18**2)
    truth -= 0.5 * ((x > 54) & (x < 72) & (y > 42) & (y < 106)).astype(float)
    truth = truth.astype(np.float32)
    shifts = rng.uniform(-0.48, 0.48, size=(n_frames, 2)).astype(np.float32)
    frames = np.stack([forward(truth, shift, psf_sigma=psf_sigma, scale=scale) for shift in shifts])
    frames += rng.normal(0.0, noise_sigma, size=frames.shape).astype(np.float32)
    return truth, frames.astype(np.float32), shifts


def run_synthetic_validation(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    try:
        from run_saa import make_synthetic_observations as make_synthetic  # type: ignore
        source = "algos/ep06_sr_poc/scripts/run_saa.make_synthetic_observations"
    except Exception:
        make_synthetic = synthetic_fallback
        source = "algos/ep10_map_tv_sweep/scripts/run_sweep.synthetic_fallback"

    truth, frames, shifts = make_synthetic(
        n_frames=int(args.synthetic_frames),
        scale=2,
        psf_sigma=float(args.synthetic_psf_sigma),
        noise_sigma=float(args.synthetic_noise),
        seed=int(args.random_state),
    )
    spec = ParamSpec(lambda_tv=float(args.synthetic_lambda_tv), psf_sigma=float(args.synthetic_psf_sigma))
    config = {
        "max_iter": min(int(args.max_iter), int(args.synthetic_max_iter)),
        "step_size": float(args.step_size),
        "use_fista": bool(args.use_fista),
        "map_tv_workers": int(args.map_tv_workers),
    }
    hr = reconstruct_for_spec(frames, shifts, spec, config)
    baseline = bicubic_upsample(np.mean(frames, axis=0), scale=2)
    payload = {
        "source": source,
        "truth_shape_hr": list(np.asarray(truth).shape),
        "frames_shape_lr": list(np.asarray(frames).shape),
        "lambda_tv": spec.lambda_tv,
        "psf_sigma": spec.psf_sigma,
        "iterations": int(config["max_iter"]),
        "map_tv_finite": bool(np.isfinite(hr).all()),
        "bicubic_mean_psnr_db": psnr(truth, baseline),
        "map_tv_psnr_db": psnr(truth, hr),
    }
    write_json(output_dir / "synthetic_validation.json", payload)
    if not payload["map_tv_finite"]:
        raise RuntimeError("Synthetic MAP-TV validation produced non-finite values")
    return payload


def _path_signature(path: Path) -> dict[str, Any]:
    resolved = path.resolve()
    payload: dict[str, Any] = {"path": str(resolved), "exists": resolved.exists()}
    if resolved.exists() and resolved.is_file():
        stat = resolved.stat()
        payload.update({"size": int(stat.st_size), "mtime_ns": int(stat.st_mtime_ns)})
    return payload


def _input_signature(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "data_dir": _path_signature(args.data_dir),
        "frame_audit_csv": _path_signature(args.frame_audit_csv),
        "alignment_csv": _path_signature(args.alignment_csv),
        "alignment_method": str(args.alignment_method),
        "sigma_bg": float(args.sigma_bg),
        "ref_hp_sigma_bg_hr_px": float(args.sigma_bg) * 2.0,
    }


def _signature_digest(signature: dict[str, Any]) -> str:
    text = json.dumps(signature, sort_keys=True).encode("utf-8")
    return hashlib.sha256(text).hexdigest()


def prepare_real_inputs(args: argparse.Namespace, output_dir: Path) -> dict[str, str]:
    cache_dir = output_dir / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    hp_path = cache_dir / "hp_frames.npy"
    shifts_path = cache_dir / "shifts.npy"
    ref_hp_path = cache_dir / "ref_hp_hr.npy"
    meta_path = cache_dir / "metadata.csv"
    summary_path = cache_dir / "input_summary.json"
    signature = _input_signature(args)
    signature_digest = _signature_digest(signature)

    if args.reuse_cached_inputs and hp_path.exists() and shifts_path.exists() and ref_hp_path.exists():
        if summary_path.exists():
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if summary.get("input_signature_digest") == signature_digest:
                return {"hp_frames": str(hp_path), "shifts": str(shifts_path), "ref_hp_hr": str(ref_hp_path)}
        print("Cached inputs exist but input signature changed; rebuilding cache.")

    frames, metadata = load_main_session_frames(
        data_dir=args.data_dir,
        frame_audit_path=args.frame_audit_csv,
        workers=int(args.io_workers),
        dtype=np.float32,
    )
    shifts = load_alignment_shifts(
        method=args.alignment_method,
        metadata=metadata,
        alignment_csv=args.alignment_csv,
        strict=True,
    )
    hp_frames = highpass_preprocess(frames, sigma_bg=float(args.sigma_bg), workers=int(args.io_workers))
    raw_corrected = offset_correction(frames, method="median")
    ref_idx = len(frames) // 2
    ref_raw_hr = bicubic_upsample(raw_corrected[ref_idx], scale=2)
    ref_hp_hr = highpass_preprocess(ref_raw_hr, sigma_bg=float(args.sigma_bg) * 2.0)

    np.save(hp_path, hp_frames.astype(np.float32, copy=False))
    np.save(shifts_path, shifts.astype(np.float32, copy=False))
    np.save(ref_hp_path, ref_hp_hr.astype(np.float32, copy=False))
    metadata.to_csv(meta_path, index=False)
    write_json(
        summary_path,
        {
            "frames_shape": list(frames.shape),
            "hp_frames": str(hp_path.relative_to(PROJECT_ROOT)),
            "shifts_shape": list(shifts.shape),
            "ref_idx": int(ref_idx),
            "alignment_method": args.alignment_method,
            "sigma_bg": float(args.sigma_bg),
            "ref_hp_sigma_bg_hr_px": float(args.sigma_bg) * 2.0,
            "input_signature": signature,
            "input_signature_digest": signature_digest,
        },
    )
    return {"hp_frames": str(hp_path), "shifts": str(shifts_path), "ref_hp_hr": str(ref_hp_path)}


def _sweep_config(args: argparse.Namespace, output_dir: Path) -> dict[str, Any]:
    return {
        "max_iter": int(args.max_iter),
        "step_size": float(args.step_size),
        "use_fista": bool(args.use_fista),
        "map_tv_workers": int(args.map_tv_workers),
        "split_half_splits": int(args.split_half_splits),
        "random_state": int(args.random_state),
        "holdout_mod": int(args.holdout_mod),
        "cache_dir": str(output_dir / "cache"),
        "detail_dir": str(output_dir / "details"),
    }


def run_real_sweep(args: argparse.Namespace, output_dir: Path) -> pd.DataFrame:
    paths = prepare_real_inputs(args, output_dir)
    config = _sweep_config(args, output_dir)

    specs = [ParamSpec(lam, sigma) for sigma in args.psf_grid for lam in args.lambda_grid]
    if args.max_params is not None:
        specs = specs[: int(args.max_params)]

    results_path = output_dir / "sweep_results.csv"
    rows: list[dict[str, Any]] = []
    if args.resume and results_path.exists():
        rows = pd.read_csv(results_path).to_dict("records")
        done = existing_completed(results_path)
        specs = [spec for spec in specs if (float(spec.lambda_tv), float(spec.psf_sigma)) not in done]
    elif results_path.exists() and not args.force:
        raise FileExistsError(f"{results_path} exists; pass --force to overwrite or keep --resume")

    if args.force and results_path.exists() and not args.resume:
        results_path.unlink()

    if not specs:
        table = save_results_table(results_path, rows)
        save_best_params(table, paths, config, output_dir)
        save_heatmap(table, output_dir)
        combine_detail_tables(table, output_dir)
        return table

    if int(args.workers) == 1:
        init_worker(paths, config)
        iterator = tqdm(specs, desc="MAP-TV sweep", unit="param")
        for spec in iterator:
            row = evaluate_param(spec)
            rows.append(row)
            save_results_table(results_path, rows)
    else:
        with ProcessPoolExecutor(
            max_workers=int(args.workers),
            initializer=init_worker,
            initargs=(paths, config),
        ) as executor:
            futures = {executor.submit(evaluate_param, spec): spec for spec in specs}
            for future in tqdm(as_completed(futures), total=len(futures), desc="MAP-TV sweep", unit="param"):
                row = future.result()
                rows.append(row)
                save_results_table(results_path, rows)

    table = save_results_table(results_path, rows)
    save_best_params(table, paths, config, output_dir)
    save_heatmap(table, output_dir)
    combine_detail_tables(table, output_dir)
    return table


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "data_raw" / "infrared_avi")
    parser.add_argument("--frame-audit-csv", type=Path, default=PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--alignment-csv", type=Path, default=default_contour_alignment_csv(project_root_path=PROJECT_ROOT))
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "ep10_map_tv_sweep")
    parser.add_argument("--alignment-method", default="contour_refined")
    parser.add_argument("--lambda-grid", type=parse_float_grid, default=list(LAMBDA_GRID))
    parser.add_argument("--psf-grid", type=parse_float_grid, default=list(PSF_GRID))
    parser.add_argument("--max-iter", type=int, default=100)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--split-half-splits", type=int, default=5)
    parser.add_argument("--holdout-mod", type=int, default=5)
    parser.add_argument("--sigma-bg", type=float, default=5.0)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--workers", type=int, default=None, help="outer parameter-grid worker processes (default: auto, <=4)")
    parser.add_argument("--map-tv-workers", type=int, default=None, help="inner MAP-TV thread workers per reconstruction (default: auto)")
    parser.add_argument("--io-workers", type=int, default=None, help="IO workers for data loading (default: auto)")
    parser.add_argument("--no-fista", dest="use_fista", action="store_false")
    parser.set_defaults(use_fista=True)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--reuse-cached-inputs", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-params", type=int, default=None, help="debug limit on number of grid points")
    parser.add_argument("--skip-synthetic-validation", action="store_true")
    parser.add_argument("--synthetic-only", action="store_true")
    parser.add_argument("--synthetic-frames", type=int, default=20)
    parser.add_argument("--synthetic-noise", type=float, default=0.002)
    parser.add_argument("--synthetic-lambda-tv", type=float, default=0.001)
    parser.add_argument("--synthetic-psf-sigma", type=float, default=0.5)
    parser.add_argument("--synthetic-max-iter", type=int, default=6)
    return parser


def _resolve_parallelism(args: argparse.Namespace) -> None:
    total = max(1, (os.cpu_count() or 2) // 2)
    if args.workers is None and args.map_tv_workers is None:
        args.workers = min(4, max(1, total // 4))
        args.map_tv_workers = max(1, total // args.workers)
    elif args.workers is None:
        args.workers = max(1, total // max(1, args.map_tv_workers))
        args.workers = min(4, args.workers)
    elif args.map_tv_workers is None:
        args.map_tv_workers = max(1, total // max(1, args.workers))
    if args.io_workers is None:
        args.io_workers = max(1, min(8, total))
    print(
        f"Parallelism: workers={args.workers} x map_tv_workers={args.map_tv_workers}"
        f" = {args.workers * args.map_tv_workers} cores"
        f" (budget={total}, io_workers={args.io_workers})"
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    _resolve_parallelism(args)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_synthetic_validation:
        synthetic = run_synthetic_validation(args, output_dir)
        print(f"synthetic validation: MAP-TV PSNR={synthetic['map_tv_psnr_db']:.3f} dB")
    if args.synthetic_only:
        return 0

    table = run_real_sweep(args, output_dir)
    print(f"wrote {output_dir / 'sweep_results.csv'} ({len(table)} rows)")
    print(f"wrote {output_dir / 'best_params.json'}")
    print(f"wrote {output_dir / 'sweep_heatmap.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
