#!/usr/bin/env python
"""Batch real-domain dot-retention probe for freshly trained micro-ablation checkpoints.

Implements the procedure in `dot_probe_runbook.md` (2026-07-09 reconstruction of the
uncommitted `render_v23_depb9v7_real_halves.py` / v24 render script + the committed
`probe_dot_retention.py --extra-arms` path) so that new checkpoints can be scored on the
*same* frozen ACL-063 3562-point / isolation-labelled real-domain detection set used by
ACL-063/064/066/067, with the same preprocessing conventions, producing directly
comparable "ALL retention" / "isolated erased%" numbers.

Two-stage architecture (this mirrors how the pipeline has always actually been run):

  --stage render   GPU checkpoint -> real-domain reconstruction -> drizzle-referenced
                   offset correction.  Needs the 248-frame real burst (5090 only) and a
                   CUDA device.  Writes an inbox-layout directory: {arm}_a_corrected.npy,
                   {arm}_b_corrected.npy, drizzle_a.npy, drizzle_b.npy, raw/{arm}_{a,b}.npy,
                   render_manifest.json.

  --stage probe    CPU dot-retention measurement via probe_dot_retention.append_extra_arms,
                   reusing the frozen canonical output/dot_probe/per_dot.csv (repo root) and
                   requiring tgv_a.npy/tgv_b.npy (copied from the canonical inbox, NEVER
                   regenerated).  Runs on the Mac once the render stage's --outdir has been
                   synced over.  Writes probe_out/{arm}/summary_v22_arms.csv per arm,
                   probe_out/summary_v22_arms_combined.csv, and summary_micro.json.

  --stage all      Runs both stages back-to-back in one process (only meaningful on a
                   single machine that has both the GPU/real burst AND per_dot.csv -- the
                   overnight run should use --stage render on the 5090 and --stage probe on
                   the Mac separately, per the runbook's actual historical split).

Each checkpoint is rebuilt with its OWN saved `ckpt["config"]` via
`unet_sr.solver_train.build_solver` -- never `render_checkpoint_evolution_drop._build_solver`
(that helper predates solver_dc_normalize/solver_fusion and silently mis-builds meanDC /
eta-variant / fusion arms) and never depb9-family default assumptions.

Example (render, on the 5090):
    uv run python scripts/eval_arms_dot_probe.py --stage render \\
        --arms micro_a=outputs/micro_v7end_8k/solver_final.pt,micro_b=outputs/micro_v7end2_8k/solver_final.pt \\
        --sanity-arm depb9v6=outputs/solver_v21_depb9v6_9bin/solver_final.pt \\
        --outdir ../../output/dot_probe_micro --device cuda

Example (probe, on the Mac, after rsyncing --outdir over):
    uv run python scripts/eval_arms_dot_probe.py --stage probe \\
        --arms micro_a,micro_b --sanity-arm depb9v6 \\
        --outdir ../../output/dot_probe_micro

Dry run (either stage, either machine -- validates imports/config/solver construction only):
    uv run python scripts/eval_arms_dot_probe.py --stage render --dry-run \\
        --arms micro_a=outputs/micro_v7end_8k/solver_final.pt --outdir /tmp/dp_dry
    uv run python scripts/eval_arms_dot_probe.py --stage probe --dry-run \\
        --arms micro_a --outdir ../../output/dot_probe_micro

See `dot_probe_runbook.md` section 6 for what could not be verified from the committed
repo alone and should be double-checked on the remote host before trusting this blindly
(a stray uncommitted `render_v2*_real_halves.py` on remote scratch, if it still exists, is
strictly safer to reuse -- diff it against this script's assumptions first).
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any

os.environ.setdefault("MPLBACKEND", "Agg")
import matplotlib  # noqa: E402

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

# --------------------------------------------------------------------------
# sys.path bootstrap -- mirrors the pattern every script in this repo uses at
# its own top (see run_real_split_frc_v2.py / probe_pair_offset.py).  `unet_sr`
# itself resolves via the ep07_unet_sr uv project's own editable install when
# run with `uv run` from this directory; EP07_SRC is inserted too as a
# defensive fallback for plain `python` invocation.
# --------------------------------------------------------------------------
SCRIPT_PATH = Path(__file__).resolve()
EP07_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[3]
EP07_SRC = EP07_ROOT / "src"
EP15_SCRIPTS = PROJECT_ROOT / "algos" / "ep15_info_limit" / "scripts"
EP06_SRC = PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src"
CORE_SRC = PROJECT_ROOT / "core" / "src"

for _p in (SCRIPT_PATH.parent, EP15_SCRIPTS, EP06_SRC, CORE_SRC, EP07_SRC):
    _t = str(_p)
    if _t not in sys.path:
        sys.path.insert(0, _t)

import probe_dot_retention as pdr  # noqa: E402  (algos/ep07_unet_sr/scripts, same dir)
import run_m2_frc as m2frc  # noqa: E402  (algos/ep15_info_limit/scripts)
import run_real_split_frc_v2 as rsfv2  # noqa: E402  (algos/ep15_info_limit/scripts)
import probe_pair_offset as ppo  # noqa: E402  (algos/ep15_info_limit/scripts)
from common.alignment import load_alignment_shifts  # noqa: E402  (ep06_sr_poc/src)
from common.data_loader import load_main_session_frames, offset_correction  # noqa: E402
from unet_sr.real_eval import infer_solver_from_burst_full_halo  # noqa: E402
from unet_sr.solver_train import build_solver  # noqa: E402

# --------------------------------------------------------------------------
SPLIT_SCALE = 2  # phase-stratified A/B split grid (independent of a checkpoint's own
                 # phase_bin_channels conditioning bin count, e.g. 9-bin vs 4-bin)
DEFAULT_STAGE_CONFIG = PROJECT_ROOT / "configs" / "stage_calibration.json"
DEFAULT_PER_DOT_CSV = PROJECT_ROOT / "output" / "dot_probe" / "per_dot.csv"
DEFAULT_CANONICAL_INBOX = PROJECT_ROOT / "remote_inbox" / "20260713_dotprobe"

# ACL-067 control-experiment table (dot_probe_runbook.md section 4) -- built-in
# instrument-drift / pipeline-regression check.  Tolerances per the runbook: retention
# +/- 0.02, erased% +/- 1 percentage point ("v6 历史臂重测漂移 <0.003" is the observed
# historical drift; the tolerance here is deliberately looser to avoid false alarms).
SANITY_REFERENCE = {
    "depb9v6": {"retention": 0.598, "erased_pct": 4.66, "tol_retention": 0.02, "tol_erased_pct": 1.0},
    "depb9v7": {"retention": 0.331, "erased_pct": 43.48, "tol_retention": 0.02, "tol_erased_pct": 1.0},
    "depb9v7_ctrl": {"retention": 0.337, "erased_pct": 39.75, "tol_retention": 0.02, "tol_erased_pct": 1.0},
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


# --------------------------------------------------------------------------
# --arms / --sanity-arm parsing: "name=path" (render) or bare "name" (probe, path
# ignored/optional so the same --arms value can be reused across both stages).
# --------------------------------------------------------------------------
def parse_arm_spec(token: str) -> tuple[str, str | None]:
    token = token.strip()
    if "=" in token:
        name, path = token.split("=", 1)
        return name.strip(), path.strip()
    return token, None


def parse_arms_arg(value: str | None) -> list[tuple[str, str | None]]:
    if not value:
        return []
    return [parse_arm_spec(tok) for tok in value.split(",") if tok.strip()]


# --------------------------------------------------------------------------
# Checkpoint -> solver (section 2.1 of the runbook).  Deliberately reuses
# unet_sr.solver_train.build_solver verbatim -- do NOT hand-roll UnrolledSolver(...)
# kwargs and do NOT reuse render_checkpoint_evolution_drop.py's stale _build_solver
# (missing solver_dc_normalize/solver_fusion, per runbook pitfall 3).
# --------------------------------------------------------------------------
def load_checkpoint_solver(ckpt_path: Path, device: str) -> tuple[torch.nn.Module, SimpleNamespace]:
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "config" not in ckpt or "model_state_dict" not in ckpt:
        raise ValueError(f"{ckpt_path}: not a recognized solver checkpoint (missing 'config'/'model_state_dict')")
    cfg = SimpleNamespace(**ckpt["config"])
    cond_channels = 5 if bool(getattr(cfg, "solver_no_drizzle", False)) else int(cfg.in_channels)
    solver = build_solver(cfg, device, cond_channels)
    solver.load_state_dict(ckpt["model_state_dict"])
    solver.eval()
    return solver, cfg


# --------------------------------------------------------------------------
# Real burst + phase-stratified split (section 2.2).  _load_real_eval_cache does NOT
# expose metadata (needed for command_phase_bins), so this mirrors
# run_real_split_frc_v2.prepare_inputs's loader calls directly instead (runbook
# section 6, item 3) and feeds RAW (untouched) frames to the solver -- matching
# unet_sr.real_eval.maybe_log_solver_real_eval, which also passes raw_frames straight
# through with no offset/highpass domain conversion.
# --------------------------------------------------------------------------
def load_real_inputs(frame_limit: int, alignment_method: str, seed: int):
    log(f"loading real burst (frame_limit={frame_limit}, alignment_method={alignment_method}) ...")
    t0 = time.time()
    raw_frames, metadata = load_main_session_frames(
        workers=2, dtype=np.float32, limit=frame_limit if frame_limit > 0 else None
    )
    full_metadata = load_main_session_frames(workers=2, dtype=np.float32, limit=None)[1]
    full_shifts = load_alignment_shifts(alignment_method, metadata=full_metadata).astype(np.float32, copy=False)
    shifts = full_shifts[: len(metadata)]
    log(f"  loaded {raw_frames.shape[0]} frames, shape={raw_frames.shape[1:]}, in {time.time() - t0:.1f}s")

    stage = rsfv2.load_stage_config(DEFAULT_STAGE_CONFIG)
    bin_ids = m2frc.command_phase_bins(
        metadata, scale=SPLIT_SCALE, theta_deg=float(stage["theta_deg"]), pixel_size_um=float(stage["pixel_size_um"])
    )
    a_idx, b_idx, balance = m2frc.stratified_split(np.asarray(bin_ids, dtype=int), scale=SPLIT_SCALE, seed=seed)
    log(
        f"  phase_stratified split seed={seed}: n_a={len(a_idx)} n_b={len(b_idx)} "
        f"max_bin_abs_diff={int(balance['abs_diff'].max())}"
    )
    return raw_frames, shifts, a_idx, b_idx


# --------------------------------------------------------------------------
# Drizzle reference halves for render-time offset correction (section 2.4).  Three
# sources, in priority order: explicit --drizzle-a/--drizzle-b > already-cached in
# --outdir (resume) > canonical-inbox fallback (remote_inbox/20260713_dotprobe's
# drizzle_phase_stratified_seed42_a/b.npy is the identical split) > generate fresh
# from the raw burst (offset-domain: per-frame median removed, then bilinear_drizzle,
# matching run_real_split_frc_v2's frame_domain="offset" + method="drizzle" path).
# --------------------------------------------------------------------------
def get_or_build_drizzle_halves(
    outdir: Path,
    canonical_inbox: Path,
    raw_frames: np.ndarray,
    shifts: np.ndarray,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    drizzle_a_arg: str | None,
    drizzle_b_arg: str | None,
) -> tuple[np.ndarray, np.ndarray]:
    out_a, out_b = outdir / "drizzle_a.npy", outdir / "drizzle_b.npy"

    source = None
    if drizzle_a_arg and drizzle_b_arg:
        source = (Path(drizzle_a_arg), Path(drizzle_b_arg), "explicit --drizzle-a/--drizzle-b")
    elif out_a.exists() and out_b.exists():
        log(f"reusing drizzle halves already cached in --outdir ({out_a}, {out_b})")
        return np.load(out_a).astype(np.float64), np.load(out_b).astype(np.float64)
    else:
        cand_a, cand_b = canonical_inbox / "drizzle_a.npy", canonical_inbox / "drizzle_b.npy"
        if cand_a.exists() and cand_b.exists():
            source = (cand_a, cand_b, "canonical-inbox fallback")

    if source is not None:
        src_a, src_b, why = source
        log(f"using existing drizzle halves ({why}): {src_a}, {src_b}")
        shutil.copy(src_a, out_a)
        shutil.copy(src_b, out_b)
        return np.load(out_a).astype(np.float64), np.load(out_b).astype(np.float64)

    log("no existing drizzle halves found -- generating from the raw burst "
        "(offset-domain: per-frame median removed, then bilinear_drizzle, scale=2) ...")
    t0 = time.time()
    offset_frames = offset_correction(raw_frames, method="median").astype(np.float32, copy=False)
    rec_a = m2frc.bilinear_drizzle(
        np.asarray(offset_frames[a_idx], dtype=np.float32), np.asarray(shifts[a_idx], dtype=np.float32),
        scale=SPLIT_SCALE, desc="drizzle_a",
    )
    rec_b = m2frc.bilinear_drizzle(
        np.asarray(offset_frames[b_idx], dtype=np.float32), np.asarray(shifts[b_idx], dtype=np.float32),
        scale=SPLIT_SCALE, desc="drizzle_b",
    )
    np.save(out_a, rec_a.image.astype(np.float32))
    np.save(out_b, rec_b.image.astype(np.float32))
    log(
        f"  drizzle halves generated in {time.time() - t0:.1f}s "
        f"(zero_coverage a={rec_a.zero_coverage_pct:.2f}% b={rec_b.zero_coverage_pct:.2f}%)"
    )
    return rec_a.image.astype(np.float64), rec_b.image.astype(np.float64)


# --------------------------------------------------------------------------
# Offset-correct a raw render half against the matching drizzle half (section 2.4 /
# pitfall 10: pair is drizzle=reference, arm=corrected).  Reimplements probe_pair()'s
# two-pass estimate/correct loop (probe_pair_offset.py lines ~161-175) by importing
# estimate_global_offset/fourier_shift directly instead of shelling out --pair specs
# per half (runbook section 6, item 6) -- skips the FRC-curve bookkeeping probe_pair()
# also does, since the dot probe itself doesn't need it.
# --------------------------------------------------------------------------
def offset_correct_to_reference(ref: np.ndarray, arr: np.ndarray) -> tuple[np.ndarray, dict]:
    total_dx, total_dy = 0.0, 0.0
    corrected = arr
    peak_corr = float("nan")
    for _ in range(2):
        offset = ppo.estimate_global_offset(ref, corrected)
        total_dx += offset["dx_px"]
        total_dy += offset["dy_px"]
        peak_corr = offset["peak_corr"]
        corrected = ppo.fourier_shift(arr, dx_px=-total_dx, dy_px=-total_dy)
    return corrected, {"dx_px": total_dx, "dy_px": total_dy, "peak_corr": peak_corr}


# --------------------------------------------------------------------------
# Stage: render
# --------------------------------------------------------------------------
def run_render_stage(args: argparse.Namespace) -> dict:
    arms = parse_arms_arg(args.arms)
    if args.sanity_arm:
        s_name, s_path = parse_arm_spec(args.sanity_arm)
        if s_name not in [n for n, _ in arms]:
            arms.append((s_name, s_path))
    missing_paths = [n for n, p in arms if p is None]
    if missing_paths:
        raise SystemExit(
            f"--stage render requires name=checkpoint_path for every arm; missing a path for: {missing_paths}"
        )

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    raw_dir = outdir / "raw"
    raw_dir.mkdir(exist_ok=True)

    manifest: dict[str, Any] = {
        "stage": "render", "halo_hr": args.halo_hr, "frame_limit": args.frame_limit,
        "alignment_method": args.alignment_method, "split_seed": args.seed, "device": args.device,
        "arms": {},
    }

    if args.dry_run:
        log(f"[dry-run] validating {len(arms)} checkpoint(s) on CPU only (no burst load, no GPU render) ...")
        for name, path in arms:
            try:
                solver, cfg = load_checkpoint_solver(Path(path), "cpu")
                n_params = sum(p.numel() for p in solver.parameters())
                log(
                    f"  [ok] {name}: {path} -> solver built ({n_params:,} params, scale={cfg.scale}, "
                    f"unroll_steps={cfg.unroll_steps}, in_channels={cfg.in_channels}, "
                    f"solver_no_drizzle={getattr(cfg, 'solver_no_drizzle', False)}, "
                    f"solver_m_frames={getattr(cfg, 'solver_m_frames', 'n/a')})"
                )
                manifest["arms"][name] = {"status": "dry_run_ok", "checkpoint": str(path), "n_params": int(n_params)}
            except Exception as exc:
                log(f"  [FAIL] {name}: {path} -> {exc!r}")
                traceback.print_exc()
                manifest["arms"][name] = {"status": "dry_run_failed", "checkpoint": str(path), "error": repr(exc)}
        write_json(outdir / "render_manifest.json", manifest)
        n_fail = sum(1 for v in manifest["arms"].values() if v["status"] != "dry_run_ok")
        log(f"[dry-run] done: {len(arms) - n_fail}/{len(arms)} checkpoints OK.")
        return manifest

    raw_frames, shifts, a_idx, b_idx = load_real_inputs(args.frame_limit, args.alignment_method, args.seed)
    drizzle_a, drizzle_b = get_or_build_drizzle_halves(
        outdir, Path(args.canonical_inbox), raw_frames, shifts, a_idx, b_idx, args.drizzle_a, args.drizzle_b
    )

    for name, path in arms:
        t_arm0 = time.time()
        log(f"=== arm {name}: {path} ===")
        try:
            solver, cfg = load_checkpoint_solver(Path(path), args.device)
            arm_info: dict[str, Any] = {
                "checkpoint": str(path),
                "unroll_steps": int(cfg.unroll_steps),
                "solver_m_frames": int(getattr(cfg, "solver_m_frames", -1)),
                "solver_no_drizzle": bool(getattr(cfg, "solver_no_drizzle", False)),
            }
            halves: dict[str, np.ndarray] = {}
            for half, idx in (("a", a_idx), ("b", b_idx)):
                if args.device.startswith("cuda"):
                    torch.cuda.empty_cache()
                t0 = time.time()
                pred = infer_solver_from_burst_full_halo(
                    solver, raw_frames[idx], shifts[idx], training_config=cfg,
                    halo_hr=args.halo_hr, device=args.device, output_grid="centered",
                )
                dt = time.time() - t0
                log(f"  half {half}: rendered {pred.shape} in {dt:.1f}s")
                np.save(raw_dir / f"{name}_{half}.npy", pred.astype(np.float32))
                halves[half] = pred.astype(np.float64)
                arm_info[f"render_sec_{half}"] = dt

            del solver
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()

            for half, drz_ref in (("a", drizzle_a), ("b", drizzle_b)):
                corrected, off = offset_correct_to_reference(drz_ref, halves[half])
                np.save(outdir / f"{name}_{half}_corrected.npy", corrected.astype(np.float32))
                arm_info[f"offset_{half}"] = off
                log(
                    f"  half {half}: offset-corrected vs drizzle -> "
                    f"({off['dx_px']:+.4f},{off['dy_px']:+.4f}) px, peak_corr={off['peak_corr']:.4f}"
                )

            arm_info["status"] = "ok"
            arm_info["elapsed_sec"] = time.time() - t_arm0
            manifest["arms"][name] = arm_info
            log(f"=== arm {name}: done in {arm_info['elapsed_sec']:.1f}s ===")
        except Exception as exc:
            log(f"  [FAIL] arm {name}: {exc!r}")
            traceback.print_exc()
            manifest["arms"][name] = {
                "checkpoint": str(path), "status": "failed", "error": repr(exc),
                "elapsed_sec": time.time() - t_arm0,
            }
            if args.device.startswith("cuda"):
                torch.cuda.empty_cache()
            continue

    write_json(outdir / "render_manifest.json", manifest)
    n_ok = sum(1 for v in manifest["arms"].values() if v.get("status") == "ok")
    log(f"render stage done: {n_ok}/{len(arms)} arms OK. outputs in {outdir}")
    return manifest


# --------------------------------------------------------------------------
# Stage: probe
# --------------------------------------------------------------------------
def check_sanity(name: str, result: dict | None) -> dict:
    ref = SANITY_REFERENCE.get(name)
    if ref is None:
        log(f"[sanity] no reference numbers hardcoded for arm '{name}' -- skipping "
            f"(known: {list(SANITY_REFERENCE)})")
        return {"checked": False, "reason": "no reference for this arm name"}
    if not result or result.get("status") != "ok":
        print("\n" + "!" * 78)
        print(f"! SANITY ARM '{name}' PRODUCED NO RESULT ({result}) -- cannot verify pipeline health.")
        print("!" * 78 + "\n")
        return {"checked": True, "passed": False, "reason": "no result"}

    ret_ok = abs(result["all_retention"] - ref["retention"]) <= ref["tol_retention"]
    erased_ok = abs(result["isolated_erased_pct"] - ref["erased_pct"]) <= ref["tol_erased_pct"]
    passed = ret_ok and erased_ok
    report = {
        "checked": True, "passed": passed,
        "observed_retention": result["all_retention"], "expected_retention": ref["retention"],
        "observed_erased_pct": result["isolated_erased_pct"], "expected_erased_pct": ref["erased_pct"],
    }
    if passed:
        log(
            f"[sanity] OK: {name} reproduces the historical numbers "
            f"(retention {result['all_retention']:.4f} vs {ref['retention']}, "
            f"erased% {result['isolated_erased_pct']:.2f} vs {ref['erased_pct']})"
        )
    else:
        print("\n" + "#" * 78)
        print(f"# SANITY CHECK FAILED for arm '{name}' -- PROBE/PIPELINE REGRESSION SUSPECTED")
        print(f"# observed ALL retention    = {result['all_retention']:.4f} "
              f"(expected {ref['retention']} +/- {ref['tol_retention']})")
        print(f"# observed isolated erased% = {result['isolated_erased_pct']:.2f} "
              f"(expected {ref['erased_pct']} +/- {ref['tol_erased_pct']})")
        print("# DO NOT TRUST tonight's new-arm numbers until this is resolved.")
        print("#" * 78 + "\n")
    return report


def run_probe_stage(args: argparse.Namespace) -> dict:
    arms = parse_arms_arg(args.arms)
    if args.sanity_arm:
        s_name, _ = parse_arm_spec(args.sanity_arm)
        if s_name not in [n for n, _ in arms]:
            arms.append((s_name, None))
    arm_names = [n for n, _ in arms]
    if not arm_names:
        raise SystemExit("--stage probe requires --arms name1,name2,... (checkpoint paths optional/ignored)")

    inbox = Path(args.inbox) if args.inbox else Path(args.outdir)
    per_dot_csv = Path(args.per_dot_csv) if args.per_dot_csv else DEFAULT_PER_DOT_CSV
    canonical_inbox = Path(args.canonical_inbox)
    tgv_a = Path(args.tgv_a) if args.tgv_a else canonical_inbox / "tgv_a.npy"
    tgv_b = Path(args.tgv_b) if args.tgv_b else canonical_inbox / "tgv_b.npy"

    log(f"inbox={inbox}")
    log(f"per_dot_csv={per_dot_csv}")
    log(f"tgv_a={tgv_a}")
    log(f"tgv_b={tgv_b}")

    problems: list[str] = []
    if not per_dot_csv.exists():
        problems.append(f"per_dot_csv not found: {per_dot_csv}")
    if not tgv_a.exists():
        problems.append(f"tgv_a not found: {tgv_a}")
    if not tgv_b.exists():
        problems.append(f"tgv_b not found: {tgv_b}")
    arm_files_expected: dict[str, tuple[Path, Path]] = {}
    for name in arm_names:
        fa, fb = inbox / f"{name}_a_corrected.npy", inbox / f"{name}_b_corrected.npy"
        arm_files_expected[name] = (fa, fb)
        if not fa.exists():
            problems.append(f"{name}: missing {fa}")
        if not fb.exists():
            problems.append(f"{name}: missing {fb}")

    if args.dry_run:
        log("[dry-run] probe stage: import + file-existence check only (no measurement run)")
        log(f"  probe_dot_retention module: {pdr.__file__}")
        log(f"  append_extra_arms callable: {callable(pdr.append_extra_arms)}")
        if problems:
            for p in problems:
                log(f"  [MISSING] {p}")
        else:
            log("  all expected input files are present.")
        return {"problems": problems}

    if not tgv_a.exists() or not tgv_b.exists():
        raise SystemExit(
            "tgv_a.npy/tgv_b.npy are required and are NEVER regenerated by this script -- "
            f"copy them from the canonical inbox ({canonical_inbox}) first."
        )
    if not per_dot_csv.exists():
        raise SystemExit(f"per_dot_csv not found: {per_dot_csv}")

    inbox.mkdir(parents=True, exist_ok=True)
    for src, dst_name in ((tgv_a, "tgv_a.npy"), (tgv_b, "tgv_b.npy")):
        dst = inbox / dst_name
        if src.resolve() != dst.resolve():
            shutil.copy(src, dst)

    probe_out_root = Path(args.outdir) / "probe_out"
    probe_out_root.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    combined_frames: list[pd.DataFrame] = []
    for name in arm_names:
        fa, fb = arm_files_expected[name]
        if not fa.exists() or not fb.exists():
            log(f"  [SKIP] {name}: missing corrected npy in inbox ({fa.name}/{fb.name})")
            results[name] = {"status": "missing_input"}
            continue
        t0 = time.time()
        try:
            pdr.ARM_FILES[name] = (fa.name, fb.name)
            arm_outdir = probe_out_root / name
            log(f"  probing arm {name} ...")
            pdr.append_extra_arms(str(inbox), str(per_dot_csv), str(arm_outdir), arm_list=[name])
            summary_csv = arm_outdir / "summary_v22_arms.csv"
            df = pd.read_csv(summary_csv)
            combined_frames.append(df)
            # "ALL retention" (the ACL-063..067 headline number) is the WHOLE-population
            # median retention -- table="by_size", size_bin="ALL" (no isolation filter).
            # Empirically verified against the historical files in
            # remote_inbox/20260713_dotprobe/ + the canonical per_dot.csv: depb9v6's
            # by_isolation/isolated/ALL row's median_retention is 0.6207 (isolated dots
            # only), NOT the ACL-067 reference 0.598 -- but its by_size/ALL row is
            # 0.598408, an exact match. The runbook's section 1.4 claim that both numbers
            # come from the same by_isolation/isolated/ALL row is INCORRECT for retention;
            # "isolated erased%" (erased_pct), by contrast, genuinely does come from that
            # by_isolation/isolated/ALL row (verified: 4.658 vs reference 4.66).
            size_row = df[(df["table"] == "by_size") & (df["size_bin"] == "ALL")]
            iso_row = df[(df["table"] == "by_isolation") & (df["isolation"] == "isolated") & (df["size_bin"] == "ALL")]
            if len(size_row) != 1:
                raise RuntimeError(f"expected exactly one by_size/ALL row, got {len(size_row)}")
            if len(iso_row) != 1:
                raise RuntimeError(f"expected exactly one by_isolation/isolated/ALL row, got {len(iso_row)}")
            retention = float(size_row["median_retention"].iloc[0])
            erased_pct = float(iso_row["erased_pct"].iloc[0])
            isolated_retention = float(iso_row["median_retention"].iloc[0])
            results[name] = {
                "status": "ok", "all_retention": retention, "isolated_erased_pct": erased_pct,
                "isolated_retention": isolated_retention, "elapsed_sec": time.time() - t0,
            }
            log(f"  {name}: ALL retention={retention:.4f}  isolated erased%={erased_pct:.2f}  "
                f"(isolated-only retention={isolated_retention:.4f})  ({time.time() - t0:.1f}s)")
        except Exception as exc:
            log(f"  [FAIL] {name}: {exc!r}")
            traceback.print_exc()
            results[name] = {"status": "failed", "error": repr(exc), "elapsed_sec": time.time() - t0}

    if combined_frames:
        pd.concat(combined_frames, ignore_index=True).to_csv(
            probe_out_root / "summary_v22_arms_combined.csv", index=False
        )

    print("\n=== dot-retention summary "
          "(ALL retention = whole-population by_size/ALL median_retention; "
          "isolated erased% = by_isolation/isolated/ALL erased_pct) ===")
    print(f"{'arm':<24}{'ALL retention':>15}{'isolated erased%':>20}{'status':>12}")
    for name in arm_names:
        r = results[name]
        if r.get("status") == "ok":
            print(f"{name:<24}{r['all_retention']:>15.4f}{r['isolated_erased_pct']:>20.2f}{'ok':>12}")
        else:
            print(f"{name:<24}{'--':>15}{'--':>20}{r.get('status', '?'):>12}")

    sanity_report = None
    if args.sanity_arm:
        s_name, _ = parse_arm_spec(args.sanity_arm)
        sanity_report = check_sanity(s_name, results.get(s_name))

    summary_payload = {
        "inbox": str(inbox), "per_dot_csv": str(per_dot_csv), "results": results, "sanity": sanity_report,
    }
    write_json(Path(args.outdir) / "summary_micro.json", summary_payload)
    log(f"wrote {Path(args.outdir) / 'summary_micro.json'}")
    return summary_payload


# --------------------------------------------------------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--stage", choices=["render", "probe", "all"], default="all",
        help="render: GPU checkpoint->corrected-halves (run on the 5090). probe: CPU dot-retention "
             "measurement (run on the Mac, needs output/dot_probe/per_dot.csv). all: both in one "
             "process (only viable on a single machine with both the GPU/real burst AND per_dot.csv).",
    )
    ap.add_argument(
        "--arms", required=True,
        help="comma-separated name=checkpoint_path pairs. In --stage probe the =checkpoint_path "
             "suffix is optional/ignored -- only the arm name is used to look up "
             "{name}_a_corrected.npy/{name}_b_corrected.npy in --inbox.",
    )
    ap.add_argument(
        "--sanity-arm", default=None,
        help=f"name=checkpoint_path (render) or bare name (probe) of a historical arm "
             f"(known: {list(SANITY_REFERENCE)}) to reproduce as a built-in instrument-drift check.",
    )
    ap.add_argument(
        "--outdir", required=True, type=str,
        help="render: inbox-layout output dir ({arm}_a_corrected.npy, drizzle_a/b.npy, raw/, "
             "render_manifest.json). probe: working dir for probe_out/ + summary_micro.json "
             "(also the default --inbox if --inbox is omitted).",
    )
    ap.add_argument(
        "--inbox", default=None,
        help="probe stage: dir with {arm}_a_corrected.npy/{arm}_b_corrected.npy + tgv_a.npy/tgv_b.npy. "
             "Defaults to --outdir (reuse the render stage's output dir once rsynced over).",
    )
    ap.add_argument(
        "--canonical-inbox", default=str(DEFAULT_CANONICAL_INBOX),
        help="source of tgv_a.npy/tgv_b.npy (required, never regenerated) and, as a fallback, "
             "drizzle_a.npy/drizzle_b.npy for --stage render if --drizzle-a/-b are not given. "
             f"Default: {DEFAULT_CANONICAL_INBOX}",
    )
    ap.add_argument(
        "--per-dot-csv", default=None,
        help=f"probe stage. Default: {DEFAULT_PER_DOT_CSV} (the ACL-063 canonical 3562-point "
             "detection set -- never regenerate this; always reuse it via --extra-arms).",
    )
    ap.add_argument("--tgv-a", default=None, help="probe stage override; default <canonical-inbox>/tgv_a.npy")
    ap.add_argument("--tgv-b", default=None, help="probe stage override; default <canonical-inbox>/tgv_b.npy")
    ap.add_argument(
        "--drizzle-a", default=None,
        help="render stage: existing drizzle half A .npy to use as the offset-correction reference "
             "(same phase_stratified/seed=42 split). Generated on the fly if omitted and not found "
             "in --outdir or --canonical-inbox.",
    )
    ap.add_argument("--drizzle-b", default=None)
    ap.add_argument("--frame-limit", type=int, default=248)
    ap.add_argument(
        "--seed", type=int, default=42,
        help="phase_stratified split seed. MUST be 42 to match the historical drizzle/TGV halves "
             "and per_dot.csv (ACL-063..067) -- only change this for a deliberate audit.",
    )
    ap.add_argument("--alignment-method", default="contour_refined")
    ap.add_argument("--halo-hr", type=int, default=96)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dry-run", action="store_true",
                     help="Validate imports/config/solver-construction (render) or file presence "
                          "(probe) without loading real data / running GPU inference / measuring.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.seed != 42:
        log(f"WARNING: --seed={args.seed} != 42 -- this will NOT match the historical drizzle/TGV/"
            f"per_dot.csv split; only use this for a deliberate audit, never for an arm you intend "
            f"to compare against ACL-063..067 numbers.")

    if args.stage in ("render", "all"):
        run_render_stage(args)
    if args.stage in ("probe", "all"):
        run_probe_stage(args)


if __name__ == "__main__":
    main()
