#!/usr/bin/env python3
"""Self-supervised PSF sigma calibration CLI (ACL-056).

Distribution-agnostic: works on any burst (frames + per-frame shifts + scale).
Two modes:

  1. Generic burst mode (prereg Step 2 — real data):
       --burst-npy burst.npy --shifts-csv shifts.csv --scale 2
     burst.npy is (N, H, W); shifts are per-frame [dx_px, dy_px] in LR pixels
     (a .npy of shape (N, 2), or a CSV containing dx_px/dy_px columns in frame
     order — the same columns load_alignment_shifts consumes).

  2. Bench validation mode (prereg Step 1 — MUST pass before Step 2 is allowed):
       --bench-pool-dir data/synthetic/pool_2x_v6_bench48 --workers 48
     Runs the estimator per scene, compares against the true metadata sigma and
     emits the preregistered verdict (median |rel err| <= tol, no systematic
     noise-tertile bias).

WARNING (ACL-056): the default e1e2 kernel was found near-degenerate in sigma
at realistic image sizes (see module docstring). Expect its Step 1 to FAIL the
prereg acceptance; run it only as a cheap falsification check (--scene-limit 3).

RECOMMENDED (ACL-057): --kernel esf — multi-frame projected ESF. Identifiable
via a parametric straight-edge scene prior; models the render/detector aperture
explicitly (--aperture, default auto). Refuses to output sigma when no usable
straight edge exists (exit 4 in generic mode).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def bootstrap() -> Path:
    path = Path(__file__).resolve()
    root = path
    while not (root / "AGENTS.md").exists() and root != root.parent:
        root = root.parent
    for add in [
        path.parents[1] / "src",
        root / "algos" / "ep06_sr_poc" / "src",
        root / "core" / "src",
        root / "tcforge" / "src",
    ]:
        text = str(add)
        if text not in sys.path:
            sys.path.insert(0, text)
    return root


PROJECT_ROOT = bootstrap()

from psf_calibration.esf_selfcal import (  # noqa: E402
    APERTURES,
    EsfSelfCalConfig,
    resolve_aperture,
    run_esf_bench_validation,
    run_esf_selfcal,
)
from psf_calibration.sigma_selfcal import (  # noqa: E402
    DEFAULT_SIGMA_GRID,
    PREREG_MEDIAN_REL_ERR_TOL,
    SelfCalConfig,
    run_bench_validation,
    run_selfcal,
)


def _csv_floats(text: str) -> tuple[float, ...]:
    try:
        values = tuple(float(tok) for tok in text.replace(";", ",").split(",") if tok.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from exc
    if len(values) < 4:
        raise argparse.ArgumentTypeError("sigma grid needs at least 4 points")
    return values


def _load_shifts(path: Path):
    import numpy as np  # noqa: PLC0415

    if path.suffix == ".npy":
        shifts = np.load(path)
    else:
        import pandas as pd  # noqa: PLC0415

        table = pd.read_csv(path)
        missing = {"dx_px", "dy_px"} - set(table.columns)
        if missing:
            raise SystemExit(f"shifts CSV lacks columns {sorted(missing)}")
        shifts = table[["dx_px", "dy_px"]].to_numpy()
    if shifts.ndim != 2 or shifts.shape[1] != 2:
        raise SystemExit("shifts must have shape (N, 2)")
    return shifts.astype(float)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bench-pool-dir", type=Path, default=None, help="prereg Step 1: synthetic pool with known true sigma")
    parser.add_argument("--burst-npy", type=Path, default=None, help="generic mode: (N,H,W) burst")
    parser.add_argument("--shifts-csv", type=Path, default=None, help="generic mode: (N,2) .npy or CSV with dx_px/dy_px")
    parser.add_argument("--label", default="burst", help="generic mode: output file prefix")
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "sigma_selfcal")
    parser.add_argument("--sigma-grid", type=_csv_floats, default=DEFAULT_SIGMA_GRID)
    parser.add_argument("--rounds", type=int, default=8)
    parser.add_argument("--holdout-frac", type=float, default=0.25)
    parser.add_argument("--cg-iters", type=int, default=40)
    parser.add_argument("--lam", type=float, default=1e-3, help="Tikhonov stabilizer, relative to the empirical mean-normal diagonal scale; constant across the sigma grid (numerical only, cannot bias the argmin)")
    parser.add_argument("--max-frames", type=int, default=48, help="cap frames per burst for tractability (-1 = use all)")
    parser.add_argument("--crop-lr", type=int, default=None, help="centered LR crop side (px) — compute knob; same crop for every sigma so it cannot bias the argmin. Recommended 160 for full 480x640 bursts")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale", type=int, default=2)
    parser.add_argument("--workers", type=int, default=1, help="bench mode: scene-level process pool (BLAS pinned to 1 per worker)")
    parser.add_argument("--scene-limit", type=int, default=None, help="bench mode: only first K scenes (smoke)")
    parser.add_argument("--no-scene-plots", action="store_true")
    parser.add_argument("--median-tol", type=float, default=PREREG_MEDIAN_REL_ERR_TOL)
    parser.add_argument(
        "--kernel",
        choices=["e1e2", "esf"],
        default="e1e2",
        help="estimator kernel. e1e2 = ACL-056 self-supervised pair (KNOWN near-degenerate in sigma; "
        "falsification instrument only). esf = ACL-057 multi-frame projected ESF (identifiable via the "
        "straight-edge scene prior; RECOMMENDED)",
    )
    parser.add_argument(
        "--aperture",
        choices=["auto", *APERTURES],
        default="auto",
        help="esf kernel: aperture model. auto = pool scenes use their recorded forward_mode preset; "
        "generic bursts use detector_box (real detector). pool_block_average adds the known render-chain "
        "extra blur (raster box + bilinear tent, sigma_extra=0.25 LR px at scale=2)",
    )
    parser.add_argument("--esf-half-width", type=float, default=5.0, help="esf: profile half-width around the edge (LR px)")
    parser.add_argument("--esf-max-edges", type=int, default=8)
    parser.add_argument("--esf-min-r2", type=float, default=0.90)
    parser.add_argument("--esf-grad-quantile", type=float, default=0.985)
    parser.add_argument("--esf-bootstrap", type=int, default=500)
    return parser.parse_args()


def _main_esf(args: argparse.Namespace) -> int:
    cfg = EsfSelfCalConfig(
        scale=args.scale,
        aperture="auto",  # resolved per scene (bench) or below (generic)
        half_width_lr=args.esf_half_width,
        max_edges=args.esf_max_edges,
        min_r2=args.esf_min_r2,
        grad_quantile=args.esf_grad_quantile,
        bootstrap_rounds=args.esf_bootstrap,
        seed=args.seed,
    )
    if args.bench_pool_dir is not None:
        result = run_esf_bench_validation(
            args.bench_pool_dir,
            cfg,
            args.output_dir,
            aperture=args.aperture,
            workers=args.workers,
            scene_limit=args.scene_limit,
            scene_plots=not args.no_scene_plots,
            median_tol=args.median_tol,
        )
        verdict = result["verdict"]
        print(json.dumps(verdict, indent=2))
        print(
            f"[sigma_selfcal/esf] bench verdict: {'PASS' if verdict['prereg_pass'] else 'FAIL'} "
            f"(median |rel err| = {verdict['median_abs_rel_err']:.4f}, tol {verdict['median_tol']}, "
            f"no-edge scenes {verdict['n_no_edge_scenes']}/{verdict['n_scenes_total']})"
        )
        return 0 if verdict["prereg_pass"] else 3
    if args.burst_npy is None or args.shifts_csv is None:
        raise SystemExit("either --bench-pool-dir or both --burst-npy and --shifts-csv are required")
    import numpy as np  # noqa: PLC0415

    cfg.aperture = resolve_aperture(args.aperture, None)  # generic burst: auto -> detector_box
    burst = np.load(args.burst_npy)
    shifts = _load_shifts(args.shifts_csv)
    summary = run_esf_selfcal(burst, shifts, cfg, out_dir=args.output_dir, label=args.label)
    printable = {
        k: summary[k]
        for k in ("label", "status", "sigma_hat", "ci_lo", "ci_hi", "n_edges_valid", "rel_spread", "warnings", "aperture")
    }
    print(json.dumps(printable, indent=2))
    if summary["status"] == "no_usable_edges":
        print("[sigma_selfcal/esf] REFUSED: no usable straight edges found — sigma_hat withheld (no silent fallback)")
        return 4
    return 0


def main() -> int:
    args = parse_args()
    if args.kernel == "esf":
        return _main_esf(args)
    cfg = SelfCalConfig(
        sigma_grid=tuple(args.sigma_grid),
        rounds=args.rounds,
        holdout_frac=args.holdout_frac,
        cg_iters=args.cg_iters,
        lam=args.lam,
        max_frames=None if args.max_frames is not None and args.max_frames < 0 else args.max_frames,
        crop_lr=args.crop_lr,
        seed=args.seed,
        scale=args.scale,
    )
    if args.bench_pool_dir is not None:
        result = run_bench_validation(
            args.bench_pool_dir,
            cfg,
            args.output_dir,
            workers=args.workers,
            scene_limit=args.scene_limit,
            scene_plots=not args.no_scene_plots,
            median_tol=args.median_tol,
        )
        verdict = result["verdict"]
        print(json.dumps(verdict, indent=2))
        print(f"[sigma_selfcal] bench verdict: {'PASS' if verdict['prereg_pass'] else 'FAIL'} "
              f"(median |rel err| = {verdict['median_abs_rel_err']:.4f}, tol {verdict['median_tol']})")
        return 0 if verdict["prereg_pass"] else 3
    if args.burst_npy is None or args.shifts_csv is None:
        raise SystemExit("either --bench-pool-dir or both --burst-npy and --shifts-csv are required")
    import numpy as np  # noqa: PLC0415

    burst = np.load(args.burst_npy)
    shifts = _load_shifts(args.shifts_csv)
    summary = run_selfcal(burst, shifts, cfg, out_dir=args.output_dir, label=args.label)
    printable = {k: summary[k] for k in ("label", "sigma_hat_e1", "ci_lo", "ci_hi", "sigma_hat_e2_flatness", "sigma_hat_e2_decay", "agreement")}
    print(json.dumps(printable, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
