"""Stage 2b synthetic benchmark harness (H1 domain-gap vs H2 architecture verdict).

Runs classical (TGV / MAP-TV) and neural (unrolled-solver checkpoints) reconstructions on a
held-out synthetic bench pool (configs/synthetic/pool_2x_v6_bench48.json) over a paired
N-frame prefix ladder, then scores band-limited FRC-vs-GT (25-40 um), band RMSE and PSNR.
Design doc: research_log/stage2b_synth_benchmark_design.md.  Phases:

  recon-classical  TGV/MAP-TV x {portable, oracle} psf conditions (CPU)
  recon-neural     solver checkpoints via infer_solver_from_burst_full_halo (true scene PSF)
  gate             per-arm constant grid-offset measurement vs GT (ACL-049 hazard class);
                   writes gate.json; aborts (exit 2) if any arm cannot be brought <0.15 HR px
  metrics          applies gate corrections, scores all recons, writes long + summary CSVs

Run from algos/ep07_unet_sr (uv env has torch/tcforge/pandas/scipy).  Classical and ep15 FRC
machinery are imported cross-episode by path (same pattern as real_eval._import_ep06_common).
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Cross-episode imports (lazy heavy ones; light path setup here)
# ---------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _ep15_scripts() -> tuple[Any, Any, Any]:
    """Import ep15 FRC machinery (returns (v2, m2, probe)).

    v2 = run_real_split_frc_v2 (frc_curve_v2 / interpolate_frc / crop_for_frc — the SAME conventions
    as every leaderboard number since Stage 0f), m2 = run_m2_frc (find_cutoff), probe =
    probe_pair_offset (estimate_global_offset / fourier_shift, synthetic-verified +-0.02 px).
    """
    scripts_dir = str(_repo_root() / "algos" / "ep15_info_limit" / "scripts")
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    v2 = importlib.import_module("run_real_split_frc_v2")
    m2 = importlib.import_module("run_m2_frc")
    probe = importlib.import_module("probe_pair_offset")
    return v2, m2, probe


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------

BAND_UM = (25.0, 40.0)
REPORT_PERIODS_UM = (24.0, 30.0)
GATE_APPLY_THRESHOLD_HR_PX = 0.02
GATE_PASS_RESIDUAL_HR_PX = 0.15
GATE_ABORT_OFFSET_HR_PX = 1.5

# Real-data classical parameters, verbatim from stage0h renders (ACL-049 baseline lineage).
TGV_KWARGS = dict(
    lambda_tv=0.003,
    alpha_ratio=2.0,
    max_iter=100,
    step_size=1.0,
    use_fista=True,
    tgv_inner_iter=80,
    tgv_device="cpu",
    aniso_ratio_y=1.5,
    coverage_weighted=True,
)
MAPTV_KWARGS = dict(lambda_tv=0.001, max_iter=150, step_size=0.1, use_fista=True)
PORTABLE_PSF = {"tgv": 0.5, "maptv": 0.2}


def select_prefix_frames(burst: np.ndarray, shifts: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """First-n prefix of a scene burst (paired N-ladder). Raises if the scene is too short."""
    if burst.ndim != 3 or shifts.shape != (burst.shape[0], 2):
        raise ValueError(f"burst/shifts shapes inconsistent: {burst.shape} vs {shifts.shape}")
    if not 0 < n <= burst.shape[0]:
        raise ValueError(f"prefix n={n} out of range for burst with {burst.shape[0]} frames")
    return (
        np.asarray(burst[:n], dtype=np.float32),
        np.asarray(shifts[:n], dtype=np.float32),
    )


def classify_gate(offset_norm_hr_px: float, corrected_residual_hr_px: float | None) -> str:
    """Gate verdict for one arm: 'ok' (no correction), 'corrected', or 'abort'."""
    if offset_norm_hr_px > GATE_ABORT_OFFSET_HR_PX:
        return "abort"
    if offset_norm_hr_px <= GATE_APPLY_THRESHOLD_HR_PX:
        return "ok"
    if corrected_residual_hr_px is None or corrected_residual_hr_px > GATE_PASS_RESIDUAL_HR_PX:
        return "abort"
    return "corrected"


def measure_offset_iterative(
    reference: np.ndarray, image: np.ndarray, probe: Any, *, iters: int = 4
) -> tuple[float, float, float]:
    """Iterative phase-correlation offset of ``image`` vs ``reference`` -> (dx, dy, residual_norm).

    Single-shot parabolic peak refinement systematically under-estimates sub-pixel fractions on
    band-limited content; the estimate is a contraction, so estimate->correct->re-estimate
    converges (verified: residual < 0.025 HR px for true offsets up to 0.7 px).
    """
    dx = dy = 0.0
    current = image
    for _ in range(int(iters)):
        off = probe.estimate_global_offset(reference, current)
        dx += float(off["dx_px"])
        dy += float(off["dy_px"])
        current = probe.fourier_shift(image, dx_px=-dx, dy_px=-dy)
    res = probe.estimate_global_offset(reference, current)
    return dx, dy, float(np.hypot(res["dx_px"], res["dy_px"]))


def tertile_bins(values: list[float]) -> list[str]:
    """Assign lo/mid/hi tertile labels by quantile over the given values."""
    arr = np.asarray(values, dtype=float)
    q1, q2 = np.nanquantile(arr, [1.0 / 3.0, 2.0 / 3.0])
    out = []
    for v in arr:
        if v <= q1:
            out.append("lo")
        elif v <= q2:
            out.append("mid")
        else:
            out.append("hi")
    return out


def band_rmse(
    image: np.ndarray,
    reference: np.ndarray,
    *,
    scale: int,
    crop_lr_px: int,
    pixel_size_um: float = 20.0,
    band_um: tuple[float, float] = BAND_UM,
    tukey_alpha: float = 0.25,
) -> float:
    """RMS of the (image - reference) difference restricted to the band_um period band.

    Windowed FFT bandpass on the difference; a relative measure (window attenuates borders),
    consistent across arms because every arm is scored with identical windowing.
    """
    from scipy.signal.windows import tukey  # noqa: PLC0415

    v2, _, _ = _ep15_scripts()
    a = v2.crop_for_frc(np.asarray(image, dtype=np.float32), scale=scale, crop_lr_px=crop_lr_px)
    b = v2.crop_for_frc(np.asarray(reference, dtype=np.float32), scale=scale, crop_lr_px=crop_lr_px)
    if a.shape != b.shape:
        raise ValueError(f"band_rmse shapes differ: {a.shape} vs {b.shape}")
    diff = a - b
    win = tukey(diff.shape[0], alpha=tukey_alpha)[:, None] * tukey(diff.shape[1], alpha=tukey_alpha)[None, :]
    hr_pitch_um = pixel_size_um / scale
    fy = np.fft.fftfreq(diff.shape[0], d=hr_pitch_um)
    fx = np.fft.fftfreq(diff.shape[1], d=hr_pitch_um)
    radial = np.hypot(fy[:, None], fx[None, :])
    with np.errstate(divide="ignore"):
        period = np.where(radial > 0, 1.0 / radial, np.inf)
    mask = (period >= band_um[0]) & (period <= band_um[1])
    spectrum = np.fft.fft2(diff * win)
    band = np.real(np.fft.ifft2(np.where(mask, spectrum, 0.0)))
    return float(np.sqrt(np.mean(band**2)))


def psnr(image: np.ndarray, reference: np.ndarray, *, scale: int, crop_lr_px: int) -> float:
    v2, _, _ = _ep15_scripts()
    a = v2.crop_for_frc(np.asarray(image, dtype=np.float32), scale=scale, crop_lr_px=crop_lr_px)
    b = v2.crop_for_frc(np.asarray(reference, dtype=np.float32), scale=scale, crop_lr_px=crop_lr_px)
    peak = float(b.max() - b.min())
    mse = float(np.mean((a - b) ** 2))
    if mse <= 0 or peak <= 0:
        return float("inf")
    return float(20.0 * np.log10(peak) - 10.0 * np.log10(mse))


def lowfreq_stability(image: np.ndarray, reference: np.ndarray, *, scale: int, crop_lr_px: int) -> dict[str, float]:
    """Low-frequency amplitude-drift columns (ACL-054 verification B: band metrics are blind to it).

    All three are relative to the scene's own GT — distribution-agnostic by construction:
      fullband_rmse   RMS(recon - GT), no bandpass (catches DC/low-freq drift band_rmse ignores).
      mean_offset     mean(recon - GT) — signed large-scale bias (v14 showed +1.1 C).
      range_excursion (recon range)/(GT range) — v14 showed 7-10x blowup; healthy arms ~1.
    """
    v2, _, _ = _ep15_scripts()
    a = v2.crop_for_frc(np.asarray(image, dtype=np.float32), scale=scale, crop_lr_px=crop_lr_px)
    b = v2.crop_for_frc(np.asarray(reference, dtype=np.float32), scale=scale, crop_lr_px=crop_lr_px)
    gt_range = float(b.max() - b.min())
    return {
        "fullband_rmse": float(np.sqrt(np.mean((a - b) ** 2))),
        "mean_offset": float(np.mean(a - b)),
        "range_excursion": float((a.max() - a.min()) / gt_range) if gt_range > 0 else float("inf"),
    }


# ---------------------------------------------------------------------------
# Scene access
# ---------------------------------------------------------------------------


def list_scene_dirs(pool_dir: Path) -> list[Path]:
    dirs = sorted(p for p in pool_dir.iterdir() if p.is_dir() and (p / "metadata.json").exists())
    if not dirs:
        raise FileNotFoundError(f"no scene directories under {pool_dir}")
    return dirs


def load_bench_scene(scene_dir: Path) -> dict[str, Any]:
    """Load one bench scene: burst (float32), shifts, GT temperature, metadata."""
    from tcforge.storage import load_scene_compact  # noqa: PLC0415

    scene = load_scene_compact(scene_dir)
    burst = scene.get("lr_burst")
    if burst is None:
        raise ValueError(f"{scene_dir} has no lr_burst (bench pool must be generated with save_lr_burst)")
    if "hr_temperature" in scene:
        gt = np.asarray(scene["hr_temperature"], dtype=np.float32)
    else:
        from unet_sr.dataset import _reconstruct_target  # noqa: PLC0415

        gt = np.asarray(_reconstruct_target(scene), dtype=np.float32)
    return {
        "scene_id": scene_dir.name,
        "burst": np.asarray(burst, dtype=np.float32),
        "shifts": np.asarray(scene["shifts"], dtype=np.float32),
        "gt": gt,
        "metadata": scene["metadata"],
    }


def recon_path(output_dir: Path, arm: str, scene_id: str, n: int) -> Path:
    return output_dir / "recons" / arm / f"{scene_id}__N{n:03d}.npy"


def _metadata_row(md: dict[str, Any]) -> dict[str, Any]:
    geo = md.get("geometry_metadata") or {}
    motif = geo.get("motif") or geo.get("motif_type") or geo.get("scene_motif") or "unknown"
    return {
        "noise_sigma_c": float(md.get("noise_sigma_c", float("nan"))),
        "delta_T_c": float(md.get("delta_T_c", float("nan"))),
        "psf_sigma_lr_px": float(md.get("psf_sigma_lr_px", float("nan"))),
        "psf_shape": str(md.get("psf_shape", "gaussian")),
        "rotation_deg": float(md.get("rotation_deg", float("nan"))),
        "difficulty": str(md.get("difficulty", "")),
        "motif": str(motif),
    }


# ---------------------------------------------------------------------------
# Phases
# ---------------------------------------------------------------------------


SLOW_TASK_SEC = 300.0  # owner review threshold: single classical recon above this gets flagged
_BLAS_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def _classical_task(task: dict[str, Any]) -> dict[str, Any]:
    """One classical reconstruction (runs in a spawned worker; BLAS env inherited from parent)."""
    import os  # noqa: PLC0415

    for key in _BLAS_ENV:  # belt-and-braces; authoritative set happens in the parent pre-spawn
        os.environ.setdefault(key, str(task["blas_threads"]))
    root = Path(task["repo_root"])
    for rel in ("algos/ep10_tgv_sr/src", "algos/ep06_sr_poc/src"):
        p = str(root / rel)
        if p not in sys.path:
            sys.path.insert(0, p)
    scene = load_bench_scene(Path(task["scene_dir"]))
    frames, shifts = select_prefix_frames(scene["burst"], scene["shifts"], task["n"])
    t0 = time.time()
    if task["method"] == "tgv":
        from ep10_tgv_sr.tgv import reconstruct_map_tgv  # noqa: PLC0415

        image, _ = reconstruct_map_tgv(
            frames, shifts, psf_sigma=task["sigma"], scale=2, workers=task["inner_workers"], **TGV_KWARGS
        )
    else:
        from map_tv.map_tv import reconstruct_map_tv  # noqa: PLC0415

        image, _ = reconstruct_map_tv(
            frames, shifts, psf_sigma=task["sigma"], scale=2, workers=task["inner_workers"], **MAPTV_KWARGS
        )
    out = Path(task["out_path"])
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, np.asarray(image, dtype=np.float32))
    elapsed = round(time.time() - t0, 2)
    return {
        "scene_id": scene["scene_id"],
        "arm": task["arm"],
        "n": task["n"],
        "psf_sigma_used": task["sigma"],
        "elapsed_sec": elapsed,
        "slow": elapsed > SLOW_TASK_SEC,
    }


def phase_recon_classical(args: argparse.Namespace) -> None:
    """Scene-level process parallelism (owner: 60-core cap, target full sweep <=1h).

    Uses a *spawn* pool with BLAS thread caps exported in the parent BEFORE spawning, so each
    fresh child interpreter imports numpy under the cap (fork would inherit an already-initialized
    BLAS pool and ignore the env).  Keep workers x inner_workers <= ~55.
    """
    import os  # noqa: PLC0415
    from concurrent.futures import ProcessPoolExecutor, as_completed  # noqa: PLC0415
    from multiprocessing import get_context  # noqa: PLC0415

    scene_dirs = _selected_scenes(args)
    ladder = _ladder(args)
    jobs = [(m, c, f"{m}__{c}") for m in args.methods.split(",") for c in args.conditions.split(",")]
    for method, cond, _arm in jobs:
        if method not in ("tgv", "maptv") or cond not in ("portable", "oracle"):
            raise ValueError(f"unknown method/condition: {method}/{cond}")
    if args.workers > 1 and args.inner_workers != 1:
        print(
            f"[stage2b] WARNING: workers={args.workers} with inner_workers={args.inner_workers} "
            "oversubscribes CPU; forcing inner_workers=1 (set --workers 1 for internal parallelism)",
            flush=True,
        )
        args.inner_workers = 1
    tasks: list[dict[str, Any]] = []
    for scene_dir in scene_dirs:
        metadata = json.loads((scene_dir / "metadata.json").read_text())
        for n in ladder:
            for method, cond, arm in jobs:
                out = recon_path(args.output_dir, arm, scene_dir.name, n)
                if args.skip_existing and out.exists():
                    continue
                sigma = PORTABLE_PSF[method] if cond == "portable" else float(metadata["psf_sigma_lr_px"])
                tasks.append(
                    {
                        "scene_dir": str(scene_dir),
                        "repo_root": str(_repo_root()),
                        "method": method,
                        "arm": arm,
                        "n": n,
                        "sigma": sigma,
                        "inner_workers": args.inner_workers,
                        "blas_threads": args.blas_threads,
                        "out_path": str(out),
                    }
                )
    if not tasks:
        print("[stage2b] nothing to do (all outputs exist)", flush=True)
        return
    print(f"[stage2b] {len(tasks)} classical tasks, workers={args.workers}, blas_threads={args.blas_threads}", flush=True)
    manifest: list[dict[str, Any]] = []
    slow: list[dict[str, Any]] = []
    if args.workers <= 1:
        for task in tasks:
            row = _classical_task(task)
            manifest.append(row)
            print(f"[stage2b] {row}", flush=True)
    else:
        for key in _BLAS_ENV:  # authoritative: children spawn fresh and import numpy under this cap
            os.environ[key] = str(args.blas_threads)
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=get_context("spawn")) as pool:
            futures = {pool.submit(_classical_task, t): t for t in tasks}
            for i, fut in enumerate(as_completed(futures), 1):
                row = fut.result()
                manifest.append(row)
                if row["slow"]:
                    slow.append(row)
                print(f"[stage2b] ({i}/{len(tasks)}) {row}", flush=True)
    if slow:
        print(
            f"[stage2b] SLOW_TASKS: {len(slow)} recon(s) exceeded {SLOW_TASK_SEC:.0f}s — owner review "
            "(candidate knobs: tgv_inner_iter, max_iter, scene count)",
            flush=True,
        )
    _append_manifest(args.output_dir, "recon_classical_manifest.jsonl", manifest)


def phase_recon_neural(args: argparse.Namespace) -> None:
    import torch  # noqa: PLC0415

    from unet_sr.config import TrainingConfig  # noqa: PLC0415
    from unet_sr.forward_torch import ScenePSF  # noqa: PLC0415
    from unet_sr.real_eval import infer_solver_from_burst_full_halo  # noqa: PLC0415
    from unet_sr.solver_train import build_solver  # noqa: PLC0415

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    scene_dirs = _selected_scenes(args)
    ladder = _ladder(args)
    manifest: list[dict[str, Any]] = []
    for spec in args.arm:
        name, _, ckpt_path = spec.partition("=")
        if not ckpt_path:
            raise ValueError(f"--arm expects name=/path/to/solver_final.pt, got: {spec}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cfg_dict = dict(ckpt["config"])
        field_names = {f.name for f in dataclasses.fields(TrainingConfig)}
        dropped = sorted(set(cfg_dict) - field_names)
        config = TrainingConfig(**{k: v for k, v in cfg_dict.items() if k in field_names})
        cond_channels = 5 if config.solver_no_drizzle else config.in_channels
        solver = build_solver(config, device, cond_channels)
        solver.load_state_dict(ckpt["model_state_dict"])
        solver.eval()
        print(f"[stage2b] arm={name} ckpt_step={ckpt.get('step')} dropped_cfg_keys={dropped}", flush=True)
        for scene_dir in scene_dirs:
            scene = load_bench_scene(scene_dir)
            md = scene["metadata"]
            psf = ScenePSF(
                sigma_lr_px=torch.tensor([float(md["psf_sigma_lr_px"])], dtype=torch.float32, device=device),
                shape=[str(md.get("psf_shape", "gaussian"))],
                sigma_y_lr_px=[
                    None if md.get("psf_sigma_y_lr_px") is None else float(md["psf_sigma_y_lr_px"])
                ],
                angle_deg=torch.tensor([float(md.get("psf_angle_deg", 0.0))], dtype=torch.float32, device=device),
            )
            for n in ladder:
                out = recon_path(args.output_dir, name, scene["scene_id"], n)
                if args.skip_existing and out.exists():
                    continue
                frames, shifts = select_prefix_frames(scene["burst"], scene["shifts"], n)
                t0 = time.time()
                pred = infer_solver_from_burst_full_halo(
                    solver,
                    frames,
                    shifts,
                    training_config=config,
                    halo_hr=args.halo_hr,
                    device=device,
                    output_grid="native",
                    psf_override=psf,
                )
                out.parent.mkdir(parents=True, exist_ok=True)
                np.save(out, np.asarray(pred, dtype=np.float32))
                manifest.append(
                    {
                        "scene_id": scene["scene_id"],
                        "arm": name,
                        "n": n,
                        "elapsed_sec": round(time.time() - t0, 2),
                    }
                )
        print(f"[stage2b] arm={name} done", flush=True)
    _append_manifest(args.output_dir, "recon_neural_manifest.jsonl", manifest)


def phase_gate(args: argparse.Namespace) -> None:
    """Measure per-arm constant offset vs GT on the first gate scenes at max N; write gate.json."""
    _, _, probe = _ep15_scripts()
    scene_dirs = _selected_scenes(args)[: args.gate_scenes]
    ladder = _ladder(args)
    n_ref = max(ladder)
    arms = sorted(p.name for p in (args.output_dir / "recons").iterdir() if p.is_dir())
    if not arms:
        raise FileNotFoundError("no recon arms found; run recon phases first")
    gate: dict[str, Any] = {"n_ref": n_ref, "scenes": [d.name for d in scene_dirs], "arms": {}}
    failures = []
    for arm in arms:
        offsets = []
        for scene_dir in scene_dirs:
            scene = load_bench_scene(scene_dir)
            rp = recon_path(args.output_dir, arm, scene["scene_id"], n_ref)
            if not rp.exists():
                raise FileNotFoundError(f"gate needs {rp}; run recon for gate scenes first")
            recon = np.load(rp)
            odx, ody, _ = measure_offset_iterative(scene["gt"], recon, probe)
            offsets.append((odx, ody))
        dx = float(np.mean([o[0] for o in offsets]))
        dy = float(np.mean([o[1] for o in offsets]))
        norm = float(np.hypot(dx, dy))
        residual = None
        if GATE_APPLY_THRESHOLD_HR_PX < norm <= GATE_ABORT_OFFSET_HR_PX:
            res = []
            for scene_dir in scene_dirs:
                scene = load_bench_scene(scene_dir)
                recon = np.load(recon_path(args.output_dir, arm, scene["scene_id"], n_ref))
                corrected = probe.fourier_shift(recon, dx_px=-dx, dy_px=-dy)
                _, _, rnorm = measure_offset_iterative(scene["gt"], corrected, probe)
                res.append(rnorm)
            residual = float(np.median(res))
        verdict = classify_gate(norm, residual)
        gate["arms"][arm] = {
            "dx_px": dx,
            "dy_px": dy,
            "norm_px": norm,
            "post_correction_residual_px": residual,
            "verdict": verdict,
            "per_scene": offsets,
        }
        print(f"[stage2b:gate] {arm}: offset=({dx:+.3f},{dy:+.3f}) |{norm:.3f}| residual={residual} -> {verdict}", flush=True)
        if verdict == "abort":
            failures.append(arm)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gate.json").write_text(json.dumps(gate, indent=1))
    if failures:
        print(f"[stage2b:gate] ABORT — arms failed grid gate: {failures} (ACL-049 hazard class; fix before metrics)", flush=True)
        raise SystemExit(2)


def phase_metrics(args: argparse.Namespace) -> None:
    import pandas as pd  # noqa: PLC0415

    v2, m2, probe = _ep15_scripts()
    gate_path = args.output_dir / "gate.json"
    if not gate_path.exists():
        raise FileNotFoundError("gate.json missing — run the gate phase first (grid sanity is mandatory)")
    gate = json.loads(gate_path.read_text())
    scene_dirs = _selected_scenes(args)
    ladder = _ladder(args)
    rows: list[dict[str, Any]] = []
    for scene_dir in scene_dirs:
        scene = load_bench_scene(scene_dir)
        md_row = _metadata_row(scene["metadata"])
        gt = scene["gt"]
        for arm, g in gate["arms"].items():
            apply_corr = g["verdict"] == "corrected"
            for n in ladder:
                rp = recon_path(args.output_dir, arm, scene["scene_id"], n)
                if not rp.exists():
                    print(f"[stage2b:metrics] MISSING (logged, not silently skipped): {rp}", flush=True)
                    continue
                recon = np.load(rp)
                if apply_corr:
                    recon = probe.fourier_shift(recon, dx_px=-g["dx_px"], dy_px=-g["dy_px"])
                curve = v2.frc_curve_v2(
                    recon, gt, scale=2, pixel_size_um=20.0, crop_lr_px=args.crop_lr_px, tukey_alpha=0.25
                )
                band = curve[(curve["period_um"] >= BAND_UM[0]) & (curve["period_um"] <= BAND_UM[1])]
                cut17 = m2.find_cutoff(curve, "threshold_1_7")
                cuthb = m2.find_cutoff(curve, "threshold_half_bit")
                row = {
                    "scene_id": scene["scene_id"],
                    "arm": arm,
                    "n_frames_used": n,
                    "frc_band_mean_25_40": float(np.nanmean(band["frc"].to_numpy(dtype=float))),
                    "cutoff_1_7_um": cut17.period_um,
                    "cutoff_1_7_crossed": bool(cut17.crossed),
                    "cutoff_half_bit_um": cuthb.period_um,
                    "cutoff_half_bit_crossed": bool(cuthb.crossed),
                    "band_rmse": band_rmse(recon, gt, scale=2, crop_lr_px=args.crop_lr_px),
                    "psnr": psnr(recon, gt, scale=2, crop_lr_px=args.crop_lr_px),
                    **lowfreq_stability(recon, gt, scale=2, crop_lr_px=args.crop_lr_px),
                    "gate_verdict": g["verdict"],
                    "gate_dx_px": g["dx_px"],
                    "gate_dy_px": g["dy_px"],
                    **{k: v2.interpolate_frc(curve, p) for k, p in (("frc_at_24um", 24.0), ("frc_at_30um", 30.0))},
                    **md_row,
                }
                rows.append(row)
    if not rows:
        raise RuntimeError("metrics produced no rows — nothing scored")
    long_df = pd.DataFrame(rows)
    long_df["noise_tertile"] = _tertiles_by_scene(long_df, "noise_sigma_c")
    long_df["delta_T_tertile"] = _tertiles_by_scene(long_df, "delta_T_c")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    long_path = args.output_dir / "stage2b_long.csv"
    long_df.to_csv(long_path, index=False)
    metric_cols = [
        "frc_at_30um", "frc_at_24um", "frc_band_mean_25_40", "band_rmse", "psnr",
        "fullband_rmse", "mean_offset", "range_excursion", "cutoff_half_bit_um",
    ]
    summary = (
        long_df.groupby(["arm", "n_frames_used"])[metric_cols].agg(["mean", "std", "count"]).round(5)
    )
    summary.to_csv(args.output_dir / "stage2b_summary.csv")
    strat = (
        long_df.groupby(["arm", "n_frames_used", "noise_tertile"])[metric_cols[:3]].agg(["mean", "std"]).round(5)
    )
    strat.to_csv(args.output_dir / "stage2b_stratified_noise.csv")
    strat2 = (
        long_df.groupby(["arm", "n_frames_used", "delta_T_tertile"])[metric_cols[:3]].agg(["mean", "std"]).round(5)
    )
    strat2.to_csv(args.output_dir / "stage2b_stratified_deltaT.csv")
    print(f"[stage2b:metrics] wrote {long_path} ({len(long_df)} rows) + 3 summary CSVs", flush=True)


def _tertiles_by_scene(df: Any, column: str) -> list[str]:
    """Tertile labels computed over unique scenes (not rows) then broadcast to rows."""
    per_scene = df.drop_duplicates("scene_id")[["scene_id", column]]
    labels = dict(zip(per_scene["scene_id"], tertile_bins(list(per_scene[column])), strict=True))
    return [labels[s] for s in df["scene_id"]]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _selected_scenes(args: argparse.Namespace) -> list[Path]:
    dirs = list_scene_dirs(args.pool_dir)
    end = None if args.scene_count < 0 else args.scene_start + args.scene_count
    sel = dirs[args.scene_start : end]
    if not sel:
        raise ValueError(f"scene selection empty (start={args.scene_start}, count={args.scene_count})")
    return sel


def _ladder(args: argparse.Namespace) -> list[int]:
    ladder = sorted({int(x) for x in args.n_ladder.split(",")})
    if not ladder:
        raise ValueError("empty --n-ladder")
    return ladder


def _append_manifest(output_dir: Path, name: str, rows: list[dict[str, Any]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / name).open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["recon-classical", "recon-neural", "gate", "metrics"])
    parser.add_argument("--pool-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-ladder", default="24,48,96")
    parser.add_argument("--scene-start", type=int, default=0)
    parser.add_argument("--scene-count", type=int, default=-1, help="-1 = all scenes from --scene-start")
    parser.add_argument("--methods", default="tgv,maptv", help="recon-classical: comma list of tgv,maptv")
    parser.add_argument("--conditions", default="portable,oracle")
    parser.add_argument(
        "--workers",
        type=int,
        default=55,
        help="recon-classical scene-level process pool size (keep workers*inner_workers <= ~55; WSL cap 60)",
    )
    parser.add_argument(
        "--inner-workers",
        type=int,
        default=1,
        help="threads INSIDE one TGV/MAP-TV solve; forced to 1 when --workers > 1",
    )
    parser.add_argument("--blas-threads", type=int, default=1, help="per-worker BLAS/OMP thread cap")
    parser.add_argument("--arm", action="append", default=[], help="recon-neural: name=/path/solver_final.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--halo-hr", type=int, default=96)
    parser.add_argument("--gate-scenes", type=int, default=3)
    parser.add_argument("--crop-lr-px", type=int, default=16)
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    phase = {
        "recon-classical": phase_recon_classical,
        "recon-neural": phase_recon_neural,
        "gate": phase_gate,
        "metrics": phase_metrics,
    }[args.phase]
    phase(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
