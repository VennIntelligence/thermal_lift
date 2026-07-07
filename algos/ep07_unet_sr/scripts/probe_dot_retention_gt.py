#!/usr/bin/env python
"""GT-based small-dot retention probe for the A1-dots pilot pool (ACL-06x).

Unlike probe_dot_retention.py (which DETECTS dots on the TGV work image, real
domain, no ground truth), this script exploits the A1-dots pilot pool's known
ground truth: every hole's exact (y, x, radius_px, depth_fraction) is recorded
in metadata.json's geometry_metadata.defects (hole_centers_yx / hole_radii /
hole_depths -- realism.py ACL extension, ACL-063+A1dots-pilot). It measures
depth directly at the KNOWN hole location on GT and on each frozen arm's
stage2b reconstruction -- no detection step, no funnel, no isolation
classifier. retention = depth_arm / depth_GT. This is a HARDER test than a
real-domain probe because GT is exactly known (no reference-arm proxy error).

Pipeline:
  0. load stage2b gate.json (per-arm constant-offset correction, same
     convention as run_stage2b_synth_benchmark.py's metrics phase) and apply
     it to each arm's recon before measurement.
  1. for every scene, every recorded hole: measure_dot() on GT and on each
     arm's (gate-corrected) recon at the SAME (y, x), window/annulus sized
     from the hole's OWN radius (no shared bandwidth across holes).
  2. retention = depth_arm / depth_GT; erased := retention < 0.30 (same
     threshold as probe_dot_retention.py's CONFIG["class_erased"]).
  3. stratify by radius bin (round to nearest int, clipped to [1,4]) and by
     depth tertile (lo/mid/hi over the realized hole_depths, pooled).
  4. crop board: a small stratified sample of holes (per radius bin) rendered
     as GT vs each arm crops, labelled with retention.

Usage (from algos/ep07_unet_sr; uv env has torch/tcforge/pandas/scipy):
  uv run python scripts/probe_dot_retention_gt.py \\
      --pool-dir ../../data/synthetic/pool_2x_dots_pilot \\
      --stage2b-dir ../../output/dot_probe_pilot/stage2b \\
      --outdir ../../output/dot_probe_pilot \\
      --n 96 \\
      --arms tgv__oracle,tgv__portable,v14,v19_etaB,depb9v6,depb9v6_bin4
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import run_stage2b_synth_benchmark as s2b  # noqa: E402

CLASS_ERASED = 0.30  # consistent with probe_dot_retention.py's CONFIG["class_erased"]
WINDOW_HALFWIDTH_SIGMA = 2.0
ANNULUS_R_SIGMA = (3.0, 5.0)
N_LOWEST = 3


# --------------------------------------------------------------------------
# Measurement (ported from probe_dot_retention.measure_dot; sigma here is the
# hole's OWN known radius_px, not a detected LoG scale)
# --------------------------------------------------------------------------
def window_slices(y: int, x: int, half: int, shape: tuple[int, int]) -> tuple[slice, slice]:
    return (
        slice(max(0, y - half), min(shape[0], y + half + 1)),
        slice(max(0, x - half), min(shape[1], x + half + 1)),
    )


def measure_dot(img: np.ndarray, y: int, x: int, sigma: float) -> dict[str, Any]:
    """Depth / background at a KNOWN (y, x); window/annulus scaled from sigma (= hole radius_px)."""
    half = int(np.ceil(WINDOW_HALFWIDTH_SIGMA * sigma))
    r_in = int(np.ceil(ANNULUS_R_SIGMA[0] * sigma))
    r_out = int(np.ceil(ANNULUS_R_SIGMA[1] * sigma))
    sl = window_slices(y, x, r_out, img.shape)
    patch = img[sl]
    yy, xx = np.mgrid[sl[0], sl[1]]
    rr = np.hypot(yy - y, xx - x)
    ann = patch[(rr >= r_in) & (rr <= r_out)]
    edge_clipped = ann.size == 0 or (sl[0].stop - sl[0].start) < (2 * r_out + 1) or (
        sl[1].stop - sl[1].start
    ) < (2 * r_out + 1)
    bg = float(np.median(ann)) if ann.size else float("nan")
    wsl = window_slices(y, x, half, img.shape)
    wpatch = img[wsl].ravel()
    n_take = min(N_LOWEST, wpatch.size)
    low = np.sort(wpatch)[:n_take]
    depth = bg - float(low.mean()) if n_take else float("nan")
    return {"depth": depth, "bg": bg, "edge_clipped": bool(edge_clipped)}


def tertile_bins(values: np.ndarray) -> list[str]:
    arr = np.asarray(values, dtype=float)
    q1, q2 = np.nanquantile(arr, [1.0 / 3.0, 2.0 / 3.0])
    out = []
    for v in arr:
        if not np.isfinite(v):
            out.append("nan")
        elif v <= q1:
            out.append("lo")
        elif v <= q2:
            out.append("mid")
        else:
            out.append("hi")
    return out


# --------------------------------------------------------------------------
# Main measurement loop
# --------------------------------------------------------------------------
def collect_per_dot(
    pool_dir: Path, stage2b_dir: Path, arms: list[str], n: int
) -> pd.DataFrame:
    gate_path = stage2b_dir / "gate.json"
    gate = json.loads(gate_path.read_text()) if gate_path.exists() else {"arms": {}}
    _, _, probe = s2b._ep15_scripts()
    scene_dirs = s2b.list_scene_dirs(pool_dir)

    rows: list[dict[str, Any]] = []
    for scene_dir in scene_dirs:
        scene = s2b.load_bench_scene(scene_dir)
        gt = scene["gt"]
        defects = (scene["metadata"].get("geometry_metadata") or {}).get("defects") or {}
        centers = defects.get("hole_centers_yx")
        radii = defects.get("hole_radii")
        depths_meta = defects.get("hole_depths")
        if not centers or not radii or not depths_meta:
            print(f"[dotgt] {scene_dir.name}: missing hole_centers_yx/hole_radii/hole_depths "
                  f"in metadata (old-schema scene?) -- skipped", flush=True)
            continue
        if not (len(centers) == len(radii) == len(depths_meta)):
            raise ValueError(f"{scene_dir.name}: hole_centers_yx/hole_radii/hole_depths length mismatch")

        # cache recon arrays per arm for this scene (gate-corrected once)
        arm_imgs: dict[str, np.ndarray | None] = {}
        for arm in arms:
            g = gate["arms"].get(arm)
            rp = s2b.recon_path(stage2b_dir, arm, scene_dir.name, n)
            if g is None or not rp.exists():
                arm_imgs[arm] = None
                continue
            recon = np.load(rp).astype(np.float64)
            if g["verdict"] == "corrected":
                recon = probe.fourier_shift(recon, dx_px=-g["dx_px"], dy_px=-g["dy_px"])
            arm_imgs[arm] = recon

        for hole_id, ((y, x), r, d_meta) in enumerate(zip(centers, radii, depths_meta, strict=True)):
            y, x = int(y), int(x)
            m_gt = measure_dot(gt, y, x, r)
            depth_gt = m_gt["depth"]
            base = {
                "scene_id": scene_dir.name,
                "hole_id": hole_id,
                "y": y,
                "x": x,
                "radius_px": float(r),
                "depth_meta": float(d_meta),
                "depth_gt": depth_gt,
                "bg_gt": m_gt["bg"],
                "edge_clipped": m_gt["edge_clipped"],
            }
            for arm in arms:
                img = arm_imgs.get(arm)
                if img is None:
                    rows.append({**base, "arm": arm, "depth_arm": np.nan,
                                 "retention": np.nan, "erased": None, "missing": True})
                    continue
                m_arm = measure_dot(img, y, x, r)
                if np.isfinite(depth_gt) and depth_gt > 1e-6:
                    retention = m_arm["depth"] / depth_gt
                else:
                    retention = float("nan")
                rows.append({
                    **base,
                    "arm": arm,
                    "depth_arm": m_arm["depth"],
                    "retention": retention,
                    "erased": bool(np.isfinite(retention) and retention < CLASS_ERASED),
                    "missing": False,
                })
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    valid = df[~df["missing"] & df["retention"].notna()]
    out = (
        valid.groupby(["arm", group_col])
        .agg(
            n=("retention", "size"),
            median_retention=("retention", "median"),
            mean_retention=("retention", "mean"),
            erased_pct=("erased", lambda s: 100.0 * s.mean()),
        )
        .reset_index()
    )
    return out


# --------------------------------------------------------------------------
# Crop board
# --------------------------------------------------------------------------
def pick_board_dots(df: pd.DataFrame, rng: np.random.Generator, per_bin: int = 2) -> pd.DataFrame:
    one_per_hole = df.drop_duplicates(["scene_id", "hole_id"]).copy()
    chosen = []
    for rb in sorted(one_per_hole["radius_bin_px"].unique()):
        cell = one_per_hole[one_per_hole["radius_bin_px"] == rb]
        take = min(per_bin, len(cell))
        if take == 0:
            continue
        idx = rng.choice(cell.index.to_numpy(), take, replace=False)
        chosen.extend(cell.loc[i] for i in idx)
    return pd.DataFrame(chosen)


def render_board(board: pd.DataFrame, df: pd.DataFrame, pool_dir: Path, stage2b_dir: Path,
                  arms: list[str], n: int, path: Path) -> None:
    if board.empty:
        print("[dotgt] board empty -- skipping crop board render", flush=True)
        return
    _, _, probe = s2b._ep15_scripts()
    gate = json.loads((stage2b_dir / "gate.json").read_text())
    cols = ["GT", *arms]
    fig, axes = plt.subplots(len(board), len(cols), figsize=(1.9 * len(cols), 1.9 * len(board)))
    axes = np.atleast_2d(axes)
    scene_cache: dict[str, dict[str, Any]] = {}
    for i, r in enumerate(board.itertuples()):
        if r.scene_id not in scene_cache:
            scene = s2b.load_bench_scene(pool_dir / r.scene_id)
            imgs = {"GT": scene["gt"].astype(np.float64)}
            for arm in arms:
                g = gate["arms"].get(arm)
                rp = s2b.recon_path(stage2b_dir, arm, r.scene_id, n)
                if g is None or not rp.exists():
                    imgs[arm] = None
                    continue
                recon = np.load(rp).astype(np.float64)
                if g["verdict"] == "corrected":
                    recon = probe.fourier_shift(recon, dx_px=-g["dx_px"], dy_px=-g["dy_px"])
                imgs[arm] = recon
            scene_cache[r.scene_id] = imgs
        imgs = scene_cache[r.scene_id]
        hw = max(10, int(np.ceil(6 * r.radius_px)))
        sl = window_slices(r.y, r.x, hw, imgs["GT"].shape)
        vmin, vmax = -1.5 * r.depth_gt, 0.75 * r.depth_gt
        row_match = df[(df["scene_id"] == r.scene_id) & (df["hole_id"] == r.hole_id)].set_index("arm")
        for j, name in enumerate(cols):
            ax = axes[i, j]
            img = imgs.get(name)
            if img is None:
                ax.axis("off")
                continue
            mm = measure_dot(img, r.y, r.x, r.radius_px)
            crop = img[sl] - mm["bg"]
            ax.imshow(crop, cmap="gray", vmin=vmin, vmax=vmax, interpolation="nearest")
            cy, cx = r.y - sl[0].start, r.x - sl[1].start
            ax.add_patch(plt.Circle((cx, cy), r.radius_px, fill=False, color="tab:red", lw=0.6, alpha=0.8))
            if name == "GT":
                lab = "ref"
            elif name in row_match.index:
                lab = f"{row_match.loc[name, 'retention']:.2f}"
            else:
                lab = "n/a"
            ax.text(0.03, 0.03, lab, transform=ax.transAxes, fontsize=6, color="yellow", va="bottom")
            if i == 0:
                ax.set_title(name, fontsize=8)
            if j == 0:
                ax.set_ylabel(
                    f"{r.scene_id}#{r.hole_id}\nr={r.radius_px:.1f}px d={r.depth_meta:.2f}",
                    fontsize=5,
                )
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle("GT-known dot crops, local bg removed, window=[-1.5,+0.75]xdepth_GT; label=retention",
                 fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.995))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pool-dir", type=Path, required=True)
    ap.add_argument("--stage2b-dir", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--arms", required=True, help="comma list of arm names (as they appear under recons/)")
    ap.add_argument("--board-per-bin", type=int, default=2)
    ap.add_argument("--seed", type=int, default=20260706)
    args = ap.parse_args()

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    df = collect_per_dot(args.pool_dir, args.stage2b_dir, arms, args.n)
    if df.empty:
        raise SystemExit("no holes collected -- check pool metadata schema / --arms / gate.json")

    df["radius_bin_px"] = df["radius_px"].round().clip(1, 4).astype(int)
    per_hole_depth = df.drop_duplicates(["scene_id", "hole_id"])[["scene_id", "hole_id", "depth_meta"]]
    depth_labels = dict(
        zip(
            zip(per_hole_depth["scene_id"], per_hole_depth["hole_id"]),
            tertile_bins(per_hole_depth["depth_meta"].to_numpy()),
            strict=True,
        )
    )
    df["depth_tertile"] = [depth_labels[(s, h)] for s, h in zip(df["scene_id"], df["hole_id"], strict=True)]

    args.outdir.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.outdir / "per_dot_gt.csv", index=False)

    summary_size = summarize(df, "radius_bin_px").sort_values(["arm", "radius_bin_px"])
    summary_size.to_csv(args.outdir / "summary_by_size.csv", index=False)

    summary_depth = summarize(df, "depth_tertile").sort_values(["arm", "depth_tertile"])
    summary_depth.to_csv(args.outdir / "summary_by_depth.csv", index=False)

    n_holes = df.drop_duplicates(["scene_id", "hole_id"]).shape[0]
    n_scenes = df["scene_id"].nunique()
    print(f"[dotgt] {n_scenes} scenes, {n_holes} holes total, arms={arms}", flush=True)
    print(summary_size.to_string(index=False), flush=True)
    print(summary_depth.to_string(index=False), flush=True)

    rng = np.random.default_rng(args.seed)
    board = pick_board_dots(df, rng, per_bin=args.board_per_bin)
    render_board(board, df, args.pool_dir, args.stage2b_dir, arms, args.n,
                 args.outdir / "crop_board_by_size.png")


if __name__ == "__main__":
    main()
