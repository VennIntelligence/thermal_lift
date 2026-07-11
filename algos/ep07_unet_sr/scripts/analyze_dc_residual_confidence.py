#!/usr/bin/env python
"""DC-residual confidence analysis: can |y - A(x_hat)| expose dot-erasure sites?

Task #6 (owner-approved 2026-07-11, ACL-074 follow-up). Zero-training, local-only.
For each rendered arm half (native solver grid recovered from the centered-grid
render via the inverse of real_eval.to_center_grid), pushes the reconstruction
back through the SAME forward operator the solver's DC used at inference
(forward_torch.forward_burst, gaussian sigma = forward_model_psf_sigma placeholder
0.5), on frames HELD OUT from the DC subset (mirrors real_eval's honest-residual
frame selection), then splat-aggregates per-frame |residual| onto the drizzle HR
grid with run_m2_frc.bilinear_drizzle so the map is registered to per_dot.csv
coordinates (render manifest shows the centered renders sit within 0.09 HR px of
the drizzle grid - negligible vs the >=3 px stat windows).

Verdict statistic: per-dot local residual stats (window max / mean, and the same
after subtracting a sigma_bg gaussian background so the sigma=0.5 model-error
structure doesn't drown local contrast), AUC (Mann-Whitney) for erased vs
non-erased dots and erased vs background null.

Honesty notes baked in: the real PSF is the misspecified sigma=0.5 placeholder
(ACL-032/059) - treat AUC as an instrument test of the DEPLOYABLE self-check
(the operator the solver itself carries), not as physics truth. Alignment is
verified by measuring preserved-dot depth in the drizzle half at per_dot (y,x).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
EP07_ROOT = SCRIPT_PATH.parent.parent
PROJECT_ROOT = EP07_ROOT.parent.parent
EP15_SCRIPTS = PROJECT_ROOT / "algos" / "ep15_info_limit" / "scripts"
EP06_SRC = PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src"
CORE_SRC = PROJECT_ROOT / "core" / "src"
EP07_SRC = EP07_ROOT / "src"
for _p in (SCRIPT_PATH.parent, EP15_SCRIPTS, EP06_SRC, CORE_SRC, EP07_SRC):
    _t = str(_p)
    if _t not in sys.path:
        sys.path.insert(0, _t)

import pandas as pd  # noqa: E402
import torch  # noqa: E402

import run_m2_frc as m2frc  # noqa: E402
import run_real_split_frc_v2 as rsfv2  # noqa: E402
from common.alignment import load_alignment_shifts  # noqa: E402
from common.data_loader import load_main_session_frames  # noqa: E402
from unet_sr.forward_torch import ScenePSF, forward_burst  # noqa: E402

SPLIT_SCALE = 2
DEFAULT_STAGE_CONFIG = PROJECT_ROOT / "configs" / "stage_calibration.json"


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def from_center_grid(image: np.ndarray, *, scale: int = 2) -> np.ndarray:
    """Inverse of real_eval.to_center_grid (content moves by +offset instead of -offset)."""
    offset = (int(scale) - 1) / 2.0
    arr = np.asarray(image, dtype=np.float64)
    if offset == 0.0:
        return arr.astype(np.float32, copy=False)
    fy = np.fft.fftfreq(arr.shape[0])[:, None]
    fx = np.fft.fftfreq(arr.shape[1])[None, :]
    ramp = np.exp(-2j * np.pi * (fy + fx) * (+offset))
    return np.real(np.fft.ifft2(np.fft.fft2(arr) * ramp)).astype(np.float32)


def load_real_inputs(frame_limit: int, alignment_method: str, seed: int):
    """Mirror of eval_arms_dot_probe.load_real_inputs (same loader calls, same split)."""
    log(f"loading real burst (frame_limit={frame_limit}, alignment={alignment_method}) ...")
    raw_frames, metadata = load_main_session_frames(
        workers=2, dtype=np.float32, limit=frame_limit if frame_limit > 0 else None
    )
    full_metadata = load_main_session_frames(workers=2, dtype=np.float32, limit=None)[1]
    shifts = load_alignment_shifts(alignment_method, metadata=full_metadata).astype(
        np.float32, copy=False
    )[: len(metadata)]
    stage = rsfv2.load_stage_config(DEFAULT_STAGE_CONFIG)
    bin_ids = m2frc.command_phase_bins(
        metadata, scale=SPLIT_SCALE, theta_deg=float(stage["theta_deg"]),
        pixel_size_um=float(stage["pixel_size_um"]),
    )
    a_idx, b_idx, _bal = m2frc.stratified_split(
        np.asarray(bin_ids, dtype=int), scale=SPLIT_SCALE, seed=seed
    )
    log(f"  {raw_frames.shape[0]} frames; split n_a={len(a_idx)} n_b={len(b_idx)}")
    return raw_frames, shifts, np.asarray(a_idx), np.asarray(b_idx)


def dc_heldout_indices(n_half: int, m_dc: int) -> np.ndarray:
    """Frames of the half NOT consumed by the solver's DC subset (mirrors
    real_eval._select_solver_eval_frames / _select_holdout_eval_frames)."""
    dc_idx = set(np.unique(np.linspace(0, n_half - 1, min(m_dc, n_half), dtype=np.int64)).tolist())
    held = np.array([i for i in range(n_half) if i not in dc_idx], dtype=np.int64)
    return held if held.size else np.arange(n_half, dtype=np.int64)


def residual_map_for_half(
    x_centered: np.ndarray,
    y_frames: np.ndarray,
    sh_frames: np.ndarray,
    m_dc: int,
    psf_sigma: float,
    chunk: int = 16,
) -> tuple[np.ndarray, dict]:
    """|y - A(x_native)| on held-out frames, splatted onto the drizzle HR grid."""
    x_native = from_center_grid(x_centered, scale=SPLIT_SCALE)
    held = dc_heldout_indices(y_frames.shape[0], m_dc)
    y_h, sh_h = y_frames[held], sh_frames[held]
    x_t = torch.from_numpy(np.ascontiguousarray(x_native[None, None])).float()
    resid_frames = np.empty_like(y_h)
    for i0 in range(0, y_h.shape[0], chunk):
        i1 = min(i0 + chunk, y_h.shape[0])
        sh_t = torch.from_numpy(np.ascontiguousarray(sh_h[i0:i1][None])).float()
        psf = ScenePSF(
            sigma_lr_px=torch.full((1,), float(psf_sigma), dtype=torch.float32),
            shape=["gaussian"], sigma_y_lr_px=[float(psf_sigma)],
            angle_deg=torch.zeros(1, dtype=torch.float32),
        )
        with torch.no_grad():
            pred = forward_burst(x_t, sh_t, psf, SPLIT_SCALE)  # (1,n,h,w)
        resid_frames[i0:i1] = pred.numpy()[0] - y_h[i0:i1]
    rec = m2frc.bilinear_drizzle(
        np.abs(resid_frames).astype(np.float32), sh_h.astype(np.float32),
        scale=SPLIT_SCALE, desc="resid_drizzle",
    )
    stats = {
        "n_heldout": int(y_h.shape[0]),
        "resid_rms_lr": float(np.sqrt(np.mean(resid_frames**2))),
        "zero_coverage_pct": float(rec.zero_coverage_pct),
    }
    return rec.image.astype(np.float32), stats


def subtract_background(m: np.ndarray, sigma_bg: float) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return m - gaussian_filter(m, sigma_bg)


def window_stats(m: np.ndarray, y: int, x: int, half: int) -> tuple[float, float]:
    sl = (slice(max(0, y - half), y + half + 1), slice(max(0, x - half), x + half + 1))
    w = m[sl]
    return float(w.max()), float(w.mean())


def auc_mannwhitney(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUC = P(pos > neg), rank-based, no sklearn."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    pos, neg = pos[np.isfinite(pos)], neg[np.isfinite(neg)]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty_like(allv)
    ranks[order] = np.arange(1, len(allv) + 1)
    # midranks for ties
    sv = allv[order]
    i = 0
    while i < len(sv):
        j = i
        while j + 1 < len(sv) and sv[j + 1] == sv[i]:
            j += 1
        if j > i:
            ranks[order[i : j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    r_pos = ranks[: len(pos)].sum()
    u = r_pos - len(pos) * (len(pos) + 1) / 2.0
    return float(u / (len(pos) * len(neg)))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inbox", default=str(PROJECT_ROOT / "remote_inbox" / "20260710_expab"))
    ap.add_argument("--arms", default="depb9v6,depb9v9s2,depb9v9_3k")
    ap.add_argument("--per-dot-csv", default=str(PROJECT_ROOT / "output" / "dot_probe" / "per_dot.csv"))
    ap.add_argument("--outdir", default=str(PROJECT_ROOT / "output" / "dc_residual_confidence"))
    ap.add_argument("--alignment-method", default="contour_refined")
    ap.add_argument("--frame-limit", type=int, default=248)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--psf-sigma", type=float, default=0.5,
                    help="forward_model_psf_sigma placeholder the checkpoints trained/infer with")
    ap.add_argument("--sigma-bg", type=float, default=8.0,
                    help="HR-px gaussian for background subtraction of the residual map")
    ap.add_argument("--n-null", type=int, default=3000)
    args = ap.parse_args()

    inbox = Path(args.inbox)
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]

    manifest = json.loads((inbox / "render_manifest.json").read_text())
    raw_frames, shifts, a_idx, b_idx = load_real_inputs(
        args.frame_limit, args.alignment_method, args.seed
    )

    dots = pd.read_csv(args.per_dot_csv)
    log(f"per_dot: {len(dots)} dots")

    # --- residual maps per arm (mean over the two halves) --------------------
    maps: dict[str, np.ndarray] = {}
    run_meta: dict[str, dict] = {}
    for arm in arms:
        halves = []
        meta = {}
        m_dc = int(manifest["arms"][arm].get("solver_m_frames", 12))
        for half, idx in (("a", a_idx), ("b", b_idx)):
            x_c = np.load(inbox / "raw" / f"{arm}_{half}.npy")
            t0 = time.time()
            rmap, st = residual_map_for_half(
                x_c, raw_frames[idx], shifts[idx], m_dc, args.psf_sigma
            )
            log(f"{arm} half {half}: resid map in {time.time()-t0:.1f}s  "
                f"rms_lr={st['resid_rms_lr']:.4f}  n_heldout={st['n_heldout']}")
            np.save(outdir / f"{arm}_residmap_{half}.npy", rmap)
            halves.append(rmap)
            meta[half] = st
        maps[arm] = 0.5 * (halves[0] + halves[1])
        run_meta[arm] = meta

    # --- alignment sanity: preserved big dots must be dark in drizzle at (y,x)
    drz = 0.5 * (np.load(inbox / "drizzle_a.npy") + np.load(inbox / "drizzle_b.npy"))
    anchor_col = f"class_{arms[0]}"
    per_arm_dots: dict[str, pd.DataFrame] = {}
    for arm in arms:
        f = inbox / "probe_out" / arm / "intermediate" / "per_dot_v22_arms.csv"
        d = pd.read_csv(f) if f.exists() else dots.copy()
        per_arm_dots[arm] = d
    big = dots[(dots["size_bin"].isin([">8px", "5-8px"]))].nlargest(12, "sigma")
    depth_hits = 0
    for _, r in big.iterrows():
        y, x = int(r["y"]), int(r["x"])
        half = 6
        sl = (slice(y - 12, y + 13), slice(x - 12, x + 13))
        ann = drz[sl]
        wmax, wmean = window_stats(drz, y, x, half)
        if wmean < np.median(ann):  # dot darker than local surroundings
            depth_hits += 1
    align_ok = depth_hits >= 8
    log(f"alignment sanity: {depth_hits}/12 large dots darker than local median -> "
        f"{'OK' if align_ok else 'SUSPECT'}")

    # --- per-dot stats + AUC --------------------------------------------------
    rng = np.random.default_rng(0)
    h_img, w_img = next(iter(maps.values())).shape
    dot_mask = np.zeros((h_img, w_img), bool)
    for _, r in dots.iterrows():
        y, x = int(r["y"]), int(r["x"])
        dot_mask[max(0, y - 10):y + 11, max(0, x - 10):x + 11] = True
    null_pts = []
    while len(null_pts) < args.n_null:
        y = int(rng.integers(24, h_img - 24)); x = int(rng.integers(24, w_img - 24))
        if not dot_mask[y, x]:
            null_pts.append((y, x))

    rows, auc_rows = [], []
    for arm in arms:
        rmap = maps[arm]
        rmap_bs = subtract_background(rmap, args.sigma_bg)
        d = per_arm_dots[arm]
        cls_col = f"class_{arm}"
        if cls_col not in d.columns:
            log(f"{arm}: no {cls_col} in per-dot csv -> background control only")
            cls = pd.Series(["unknown"] * len(d))
        else:
            cls = d[cls_col]
        stats = {k: [] for k in ("win_max", "win_mean", "bs_max", "bs_mean")}
        for _, r in d.iterrows():
            y, x = int(r["y"]), int(r["x"])
            half = int(np.clip(np.ceil(2.0 * float(r["sigma"])) + 2, 3, 8))
            mx, mn = window_stats(rmap, y, x, half)
            bmx, bmn = window_stats(rmap_bs, y, x, half)
            stats["win_max"].append(mx); stats["win_mean"].append(mn)
            stats["bs_max"].append(bmx); stats["bs_mean"].append(bmn)
        for k, v in stats.items():
            d[f"resid_{k}"] = v
        d["dot_class"] = cls.values
        d["arm"] = arm
        rows.append(d[["dot_id", "y", "x", "sigma", "size_bin", "isolation", "arm",
                       "dot_class"] + [f"resid_{k}" for k in stats]])

        null_stats = {k: [] for k in ("win_max", "win_mean", "bs_max", "bs_mean")}
        for (y, x) in null_pts:
            mx, mn = window_stats(rmap, y, x, 5)
            bmx, bmn = window_stats(rmap_bs, y, x, 5)
            null_stats["win_max"].append(mx); null_stats["win_mean"].append(mn)
            null_stats["bs_max"].append(bmx); null_stats["bs_mean"].append(bmn)

        is_erased = (cls == "erased").values
        is_kept = cls.isin(["preserved", "blurred"]).values
        for k in stats:
            v = np.asarray(stats[k], float)
            auc_rows.append({
                "arm": arm, "stat": k,
                "n_erased": int(is_erased.sum()), "n_kept": int(is_kept.sum()),
                "auc_erased_vs_kept": auc_mannwhitney(v[is_erased], v[is_kept]),
                "auc_erased_vs_null": auc_mannwhitney(v[is_erased], np.asarray(null_stats[k])),
                "auc_alldots_vs_null": auc_mannwhitney(v, np.asarray(null_stats[k])),
                "median_erased": float(np.nanmedian(v[is_erased])) if is_erased.any() else float("nan"),
                "median_kept": float(np.nanmedian(v[is_kept])) if is_kept.any() else float("nan"),
                "median_null": float(np.median(null_stats[k])),
            })

    per_dot_out = pd.concat(rows, ignore_index=True)
    per_dot_out.to_csv(outdir / "per_dot_residual_stats.csv", index=False)
    auc_df = pd.DataFrame(auc_rows)
    auc_df.to_csv(outdir / "auc_table.csv", index=False)
    print(auc_df.to_string(index=False))

    # --- example crops PNG ----------------------------------------------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        arm = arms[1] if len(arms) > 1 else arms[0]
        d = per_arm_dots[arm]
        cls_col = f"class_{arm}"
        ex_er = d[d[cls_col] == "erased"].nlargest(3, "sigma")
        ex_pr = d[d[cls_col] == "preserved"].nlargest(3, "sigma")
        fig, axes = plt.subplots(2, 6, figsize=(18, 6))
        for j, (_, r) in enumerate(pd.concat([ex_er, ex_pr]).iterrows()):
            y, x = int(r["y"]), int(r["x"])
            sl = (slice(y - 16, y + 17), slice(x - 16, x + 17))
            axes[0, j].imshow(drz[sl], cmap="gray")
            axes[0, j].set_title(f"drz {r[cls_col]} d{int(r['dot_id'])}", fontsize=8)
            axes[1, j].imshow(maps[arm][sl], cmap="magma")
            axes[1, j].set_title(f"|resid| {arm}", fontsize=8)
            for ax in (axes[0, j], axes[1, j]):
                ax.plot(16, 16, "c+", ms=8); ax.axis("off")
        fig.tight_layout()
        fig.savefig(outdir / f"example_crops_{arm}.png", dpi=130)
        log(f"wrote example crops for {arm}")
    except Exception as exc:  # matplotlib genuinely optional
        log(f"crops skipped: {exc!r}")

    meta = {
        "args": vars(args), "alignment_sanity_hits": depth_hits,
        "alignment_ok": bool(align_ok), "run_meta": run_meta,
        "note": "PSF is the sigma=0.5 real-eval placeholder (misspecified, ACL-032/059): "
                "this tests the DEPLOYABLE self-check the solver itself carries.",
    }
    (outdir / "run_meta.json").write_text(json.dumps(meta, indent=2))
    log(f"done -> {outdir}")


if __name__ == "__main__":
    main()
