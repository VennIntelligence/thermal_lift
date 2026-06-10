#!/usr/bin/env python3
"""Quick single-param TGV run with anisotropic + coverage-weighted fixes.

Uses the previously-identified best parameters (lambda=0.003, sigma=0.5)
and applies the ACL-007 fixes: aniso_ratio_y=1.5, coverage_weighted=True.
Skips the full sweep/split-half/holdout evaluation — just does one full
reconstruction + comparison figure.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ALGO_ROOT = SCRIPT_DIR.parents[0]
PROJECT_ROOT = SCRIPT_DIR.parents[2]
for path in (
    ALGO_ROOT / "src",
    PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src",
    PROJECT_ROOT / "core" / "src",
):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from thermal_core.alignment_paths import default_contour_alignment_csv  # noqa: E402

from common.alignment import load_alignment_shifts
from common.data_loader import (
    bicubic_upsample,
    highpass_preprocess,
    load_main_session_frames,
    offset_correction,
)
from common.metrics import artifact_score
from ep10_tgv_sr import get_tgv_backend_provenance, reconstruct_map_tgv
from thermal_core.plotting import COLORMAPS, FIGURE_SIZES, savefig_academic, setup_academic_style


OUTPUT_DIR = PROJECT_ROOT / "output" / "ep10_tgv_sr"
DATA_DIR = PROJECT_ROOT / "data" / "data_raw" / "infrared_avi"
FRAME_AUDIT_CSV = PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv"
ALIGNMENT_CSV = default_contour_alignment_csv(project_root_path=PROJECT_ROOT)

# Best params from previous sweep
LAMBDA_TV = 0.003
PSF_SIGMA = 0.50
ALPHA_RATIO = 2.0
MAX_ITER = 100
STEP_SIZE = 1.0
TGV_INNER_ITER = 80
HIGHPASS_SIGMA = 5.0
ANISO_RATIO_Y = 1.5
COVERAGE_WEIGHTED = True


def center_crop(image: np.ndarray, fraction: float = 1.0 / 3.0) -> np.ndarray:
    h, w = image.shape
    ch = max(1, int(round(h * fraction)))
    cw = max(1, int(round(w * fraction)))
    y0 = (h - ch) // 2
    x0 = (w - cw) // 2
    return image[y0: y0 + ch, x0: x0 + cw]


def robust_limits(images: list[np.ndarray], *, symmetric: bool = True) -> tuple[float, float]:
    values = np.concatenate([np.asarray(img, dtype=np.float32).ravel() for img in images])
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    if symmetric:
        limit = float(np.nanpercentile(np.abs(values), 99.0))
        return -limit, limit
    return float(np.nanpercentile(values, 1.0)), float(np.nanpercentile(values, 99.0))


def pearson_corr(a: np.ndarray, b: np.ndarray) -> float:
    aa = np.asarray(a, dtype=np.float64).ravel()
    bb = np.asarray(b, dtype=np.float64).ravel()
    valid = np.isfinite(aa) & np.isfinite(bb)
    if int(valid.sum()) < 2:
        return float("nan")
    return float(np.corrcoef(aa[valid], bb[valid])[0, 1])


def main() -> None:
    start = time.perf_counter()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load data (reuse cache if available)
    cache_dir = OUTPUT_DIR / "cache"
    hp_path = cache_dir / "hp_frames.npy"
    shifts_path = cache_dir / "shifts.npy"
    ref_hp_path = cache_dir / "ref_hp_hr.npy"

    if hp_path.exists() and shifts_path.exists() and ref_hp_path.exists():
        print("Reusing cached inputs...")
        hp_frames = np.load(hp_path, mmap_mode="r")
        shifts = np.load(shifts_path, mmap_mode="r")
        ref_hp_hr = np.load(ref_hp_path, mmap_mode="r")
    else:
        print("Loading and preprocessing frames...")
        raw_frames, metadata = load_main_session_frames(
            DATA_DIR, FRAME_AUDIT_CSV, workers=8, dtype=np.float32,
        )
        shifts = load_alignment_shifts(
            method="contour_refined", metadata=metadata, alignment_csv=ALIGNMENT_CSV,
        )
        hp_frames = highpass_preprocess(raw_frames, sigma_bg=HIGHPASS_SIGMA, workers=8)
        raw_offset = offset_correction(raw_frames)
        ref_idx = len(raw_offset) // 2
        ref_raw_hr = bicubic_upsample(raw_offset[ref_idx], scale=2)
        ref_hp_hr = highpass_preprocess(ref_raw_hr, sigma_bg=HIGHPASS_SIGMA * 2.0)

        cache_dir.mkdir(parents=True, exist_ok=True)
        np.save(hp_path, hp_frames.astype(np.float32, copy=False))
        np.save(shifts_path, shifts.astype(np.float32, copy=False))
        np.save(ref_hp_path, ref_hp_hr.astype(np.float32, copy=False))

    print(f"Frames: {hp_frames.shape}, Shifts: {shifts.shape}")

    # Run single reconstruction with aniso + coverage
    print(f"Running MAP-TGV: lambda={LAMBDA_TV}, sigma={PSF_SIGMA}, "
          f"aniso_ratio_y={ANISO_RATIO_Y}, coverage_weighted={COVERAGE_WEIGHTED}")
    hr_new, records = reconstruct_map_tgv(
        hp_frames,
        shifts,
        lambda_tv=LAMBDA_TV,
        alpha_ratio=ALPHA_RATIO,
        psf_sigma=PSF_SIGMA,
        max_iter=MAX_ITER,
        step_size=STEP_SIZE,
        use_fista=True,
        workers=4,
        tgv_inner_iter=TGV_INNER_ITER,
        tgv_device="cpu",
        aniso_ratio_y=ANISO_RATIO_Y,
        coverage_weighted=COVERAGE_WEIGHTED,
    )
    hr_new = np.asarray(hr_new, dtype=np.float32)
    backend = get_tgv_backend_provenance()
    elapsed = time.perf_counter() - start
    print(f"Reconstruction done in {elapsed:.1f}s, backend: {backend['backend']}/{backend['status']}")

    # Save
    np.save(OUTPUT_DIR / "best_hr_highpass.npy", hr_new)

    # Load old isotropic result for comparison (if exists)
    old_hr_path = OUTPUT_DIR / "cache" / "full_hr_lambda0p003_sigma0p5.npy"
    has_old = old_hr_path.exists()
    if has_old:
        hr_old = np.load(old_hr_path).astype(np.float32, copy=False)
        print("Loaded previous isotropic TGV result for comparison")
    else:
        print("No previous isotropic TGV result found; skipping before/after comparison")

    # Metrics
    art_new = artifact_score(hr_new)
    corr_new = pearson_corr(hr_new, ref_hp_hr)
    print(f"NEW (aniso+cov): artifact_score={art_new:.6f}, raw_control_corr={corr_new:.6f}")
    if has_old:
        art_old = artifact_score(hr_old)
        corr_old = pearson_corr(hr_old, ref_hp_hr)
        print(f"OLD (isotropic): artifact_score={art_old:.6f}, raw_control_corr={corr_old:.6f}")

    # Generate comparison figure
    setup_academic_style()
    if has_old:
        crops = [center_crop(img) for img in (ref_hp_hr, hr_old, hr_new)]
        vmin, vmax = robust_limits(crops, symmetric=True)
        fig, axes = plt.subplots(1, 3, figsize=FIGURE_SIZES["double_col"], constrained_layout=True)
        panels = [
            (crops[0], "Raw-control highpass"),
            (crops[1], f"TGV isotropic\nartifact={art_old:.4f}"),
            (crops[2], f"TGV aniso+cov\nartifact={art_new:.4f}"),
        ]
    else:
        crops = [center_crop(img) for img in (ref_hp_hr, hr_new)]
        vmin, vmax = robust_limits(crops, symmetric=True)
        fig, axes = plt.subplots(1, 2, figsize=(6.4, 3.6), constrained_layout=True)
        panels = [
            (crops[0], "Raw-control highpass"),
            (crops[1], f"TGV aniso+cov\nartifact={art_new:.4f}"),
        ]

    for ax, (img, label) in zip(axes, panels, strict=True):
        ax.imshow(img, cmap=COLORMAPS["residual_diff"], vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(label, fontsize=8.5, pad=2)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        f"ACL-007: aniso_ratio_y={ANISO_RATIO_Y}, coverage_weighted={COVERAGE_WEIGHTED}",
        fontsize=10, fontweight="bold",
    )
    savefig_academic(fig, OUTPUT_DIR / "tgv_vs_tv_comparison.png")
    print(f"Saved comparison figure to {OUTPUT_DIR / 'tgv_vs_tv_comparison.png'}")

    # Save minimal run_summary.json
    summary = {
        "best_label": f"tgv_lam{LAMBDA_TV}_sigma{PSF_SIGMA}",
        "best_label_scope": "map_tgv_only",
        "best_tgv_label": f"tgv_lam{LAMBDA_TV}_sigma{PSF_SIGMA}",
        "elapsed_sec": elapsed,
        "n_frames": int(hp_frames.shape[0]),
        "alignment_method": "contour_refined",
        "highpass_sigma": HIGHPASS_SIGMA,
        "lambda_grid": [LAMBDA_TV],
        "psf_grid": [PSF_SIGMA],
        "alpha_ratio": ALPHA_RATIO,
        "aniso_ratio_y": ANISO_RATIO_Y,
        "coverage_weighted": COVERAGE_WEIGHTED,
        "max_iter": MAX_ITER,
        "step_size": STEP_SIZE,
        "tgv_inner_iter": TGV_INNER_ITER,
        "gradient_workers": 4,
        "outer_workers": 1,
        "tgv_devices": ["cpu"],
        "full_session_metrics": True,
        "tgv_backend": str(backend["backend"]),
        "tgv_backend_status": str(backend["status"]),
    }
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8",
    )

    # Save minimal sweep_results.csv for EP10 cache compatibility
    import pandas as pd
    row = {
        "method": "map_tgv",
        "label": f"tgv_lam{LAMBDA_TV}_sigma{PSF_SIGMA}",
        "n_input_frames": int(hp_frames.shape[0]),
        "input_frame_count": int(hp_frames.shape[0]),
        "lambda_tv": LAMBDA_TV,
        "alpha_ratio": ALPHA_RATIO,
        "aniso_ratio_y": ANISO_RATIO_Y,
        "coverage_weighted": COVERAGE_WEIGHTED,
        "psf_sigma": PSF_SIGMA,
        "max_iter": MAX_ITER,
        "step_size": STEP_SIZE,
        "n_splits": 0,
        "tgv_inner_iter": TGV_INNER_ITER,
        "tv_inner_iter": np.nan,
        "split_half_nrmse_median": np.nan,
        "split_half_corr_median": np.nan,
        "holdout_mse": np.nan,
        "artifact_score": float(art_new),
        "raw_control_corr": float(corr_new),
        "tgv_device": "cpu",
        "tgv_backend": str(backend["backend"]),
        "tgv_backend_status": str(backend["status"]),
        "tgv_backend_device": str(backend.get("selected_device", "")),
        "tgv_backend_error": "",
        "full_session_metrics": True,
        "hr_cache_file": str(OUTPUT_DIR / "best_hr_highpass.npy"),
        "convergence_file": "",
    }
    pd.DataFrame([row]).to_csv(OUTPUT_DIR / "sweep_results.csv", index=False)
    print(f"Done. Total elapsed: {elapsed:.1f}s")


if __name__ == "__main__":
    main()
