#!/usr/bin/env python3
"""Mechanical mini-pool generator using the r4 v7 composer demo (owner-approved
form). Does NOT modify scripts/v7_composer_demo.py; only imports and calls its
public `_make_scene` / `interior_crop` / `_pct` helpers to:

  1. Generate 50 scenes (22 mid, 24 high, 4 high+force_xl), seeds 3030001..3030050.
  2. Save each scene to outputs/v7_demo_minipool/scene_{i:03d}.npz
     (cov/T/lr as float16; occ/dT/sigma/angle scalars; tier string).
  3. Write an index outputs/v7_demo_minipool/index.json.
  4. Render 5 eyeball sheets (10 scenes each, 5x4 grid of GT|LR pairs) to
     research_log/assets/v7_planning/composer_demo_r4/minipool_sheets/.
  5. Print occ distribution per tier, total dot count, and zero-dot scene count.

Usage: uv run python scripts/generate_v7_demo_minipool.py
"""
from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, "scripts")
import v7_composer_demo as D  # noqa: E402

SEED0 = 3030001
N_MID = 22
N_HIGH = 24
N_XL = 4
N_TOTAL = N_MID + N_HIGH + N_XL  # 50

NPZ_OUT = Path("outputs/v7_demo_minipool")
SHEET_OUT = Path(
    "research_log/assets/v7_planning/composer_demo_r4/minipool_sheets"
)


def _plan() -> list[tuple[str, bool]]:
    plan: list[tuple[str, bool]] = []
    plan += [("mid", False)] * N_MID
    plan += [("high", False)] * N_HIGH
    plan += [("high", True)] * N_XL
    assert len(plan) == N_TOTAL
    return plan


def generate_pool() -> list[dict]:
    NPZ_OUT.mkdir(parents=True, exist_ok=True)
    plan = _plan()
    records: list[dict] = []
    errors: list[tuple[int, int, str]] = []
    for i, (tier, force_xl) in enumerate(plan):
        seed = SEED0 + i
        try:
            s = D._make_scene(seed, tier, force_xl=force_xl)
        except Exception as exc:  # noqa: BLE001 - keep going, record & continue
            msg = f"{type(exc).__name__}: {exc}"
            print(f"[scene {i:03d} seed={seed}] FAILED: {msg}")
            traceback.print_exc()
            errors.append((i, seed, msg))
            continue

        m, dfx = s["meta"], s["dfx"]
        n_dots, n_hots, n_darks = len(dfx.dots), len(dfx.hots), len(dfx.darks)
        n_notch, n_break = len(m.notch_pts), len(m.break_pts)
        n_panels = len(m.blocks)

        npz_path = NPZ_OUT / f"scene_{i:03d}.npz"
        np.savez_compressed(
            npz_path,
            cov=s["cov"].astype(np.float16),
            T=s["T"].astype(np.float16),
            lr=s["lr"].astype(np.float16),
            occ=np.float32(s["occ"]),
            dT=np.float32(s["dT"]),
            sigma=np.float32(s["sigma"]),
            angle=np.float32(s["angle"]),
            tier=str(tier),
        )

        rec = dict(
            i=i, seed=seed, tier=tier, force_xl=force_xl,
            occ=float(s["occ"]), dT=float(s["dT"]), sigma=float(s["sigma"]),
            angle=float(s["angle"]),
            n_dots=n_dots, n_hots=n_hots, n_darks=n_darks,
            n_notch=n_notch, n_break=n_break, n_panels=n_panels,
            secondary=bool(m.secondary),
        )
        records.append(rec)
        print(
            f"[scene {i:03d} seed={seed}] tier={tier}{'+xl' if force_xl else '':<3} "
            f"occ={s['occ']:.3f} dots={n_dots} panels={n_panels}"
        )

    index_records = [
        {k: rec[k] for k in (
            "i", "seed", "tier", "occ", "dT", "sigma",
            "n_dots", "n_hots", "n_darks", "n_notch", "n_break",
            "n_panels", "secondary",
        )}
        for rec in records
    ]
    with open(NPZ_OUT / "index.json", "w") as f:
        json.dump(index_records, f, indent=2)

    if errors:
        print(f"\n{len(errors)} scene(s) FAILED (see log above):")
        for i, seed, msg in errors:
            print(f"  scene {i:03d} seed={seed}: {msg}")
    print(f"\n{len(records)}/{N_TOTAL} scenes generated OK.")
    return records


def render_sheets(records: list[dict]) -> list[Path]:
    """5 sheets x 10 scenes, 5 rows x 4 cols (each scene = GT|LR pair).
    Re-runs _make_scene per recorded seed/tier (cheap vs. re-reading npz,
    and gives us full-precision arrays for the percentile scaling)."""
    SHEET_OUT.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    chunks = [records[k:k + 10] for k in range(0, len(records), 10)]
    for sheet_idx, chunk in enumerate(chunks, start=1):
        fig, axes = plt.subplots(5, 4, figsize=(4.4 * 4, 3.6 * 5))
        fig.suptitle(
            f"v7 demo r4 minipool sheet {sheet_idx}/{len(chunks)} — "
            "GT (interior crop) | LR (full frame), nearest, per-panel 1-99pct scale",
            fontsize=12.5, y=0.998,
        )
        for k, rec in enumerate(chunk):
            s = D._make_scene(rec["seed"], rec["tier"], force_xl=rec["force_xl"])
            gt = D.interior_crop(s["T"])
            lr = s["lr"]
            r, c = divmod(k, 2)
            ax_gt, ax_lr = axes[r][2 * c], axes[r][2 * c + 1]
            lo, hi = D._pct(gt)
            ax_gt.imshow(gt, cmap="inferno", vmin=lo, vmax=hi,
                         interpolation="nearest")
            tier_lbl = rec["tier"] + (" XL" if rec["force_xl"] else "")
            ax_gt.set_title(
                f"GT {tier_lbl}  occ={rec['occ']:.2f}  seed={rec['seed']}  "
                f"dots={rec['n_dots']}",
                fontsize=8,
            )
            llo, lhi = D._pct(lr)
            ax_lr.imshow(lr, cmap="inferno", vmin=llo, vmax=lhi,
                         interpolation="nearest")
            ax_lr.set_title(
                f"LR 480x640  sigma={rec['sigma']:.2f}px  i={rec['i']:03d}",
                fontsize=8,
            )
            for ax in (ax_gt, ax_lr):
                ax.set_xticks([]); ax.set_yticks([])
        # blank out any unused cells (last sheet may have < 10 scenes)
        for k in range(len(chunk), 10):
            r, c = divmod(k, 2)
            for ax in (axes[r][2 * c], axes[r][2 * c + 1]):
                ax.axis("off")
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        out_path = SHEET_OUT / f"sheet_minipool_{sheet_idx}.png"
        fig.savefig(out_path, dpi=115, bbox_inches="tight")
        plt.close(fig)
        paths.append(out_path.resolve())
        print("wrote", out_path.resolve())
    return paths


def summarize(records: list[dict]) -> None:
    print("\n=== summary ===")
    for tier_lbl in ("mid", "high", "high_xl"):
        if tier_lbl == "high_xl":
            occs = [r["occ"] for r in records if r["tier"] == "high" and r["force_xl"]]
        else:
            occs = [r["occ"] for r in records
                    if r["tier"] == tier_lbl and not r["force_xl"]]
        if occs:
            print(
                f"  tier={tier_lbl:8s} n={len(occs):2d}  "
                f"occ min={min(occs):.3f} median={float(np.median(occs)):.3f} "
                f"max={max(occs):.3f}"
            )
        else:
            print(f"  tier={tier_lbl:8s} n=0")
    total_dots = sum(r["n_dots"] for r in records)
    zero_dot_scenes = sum(1 for r in records if r["n_dots"] < 20)
    print(f"  total dots across pool: {total_dots}")
    print(f"  zero-dot scenes (n_dots<20): {zero_dot_scenes}")


def main() -> None:
    records = generate_pool()
    sheet_paths = render_sheets(records)
    summarize(records)
    print("\nsheets:")
    for p in sheet_paths:
        print(" ", p)


if __name__ == "__main__":
    main()
