#!/usr/bin/env python3
"""Quantitative gates for the v7 composer demo r4 (owner: 不能光看好看).

Pre-registered gates (thresholds fixed BEFORE the run; sources in brackets):

  G1 band floor, bright  [ACL-023 red line; composition-interaction check]
     Sub-28um bright slivers from composed geometry (window-edge strips, notch
     corners, crossing-trace pinches). Metric: area fraction of structure
     removed by a 28um-diameter disc opening, on the UNROTATED draw-res mask.
     Gate: median <= 0.5%, p95 <= 1.5% of structure area.
  G2 band floor, gaps  [ACL-023]
     Sub-28um dark gaps inside the cluster bbox (pinched streets, carve
     interactions). Same opening on background within the structure bbox.
     Gate: median <= 1.0%, p95 <= 3.0% of in-bbox background.
  G3 occupancy tiers  [density-audit verdict + owner "no low tier"]
     Full-frame audit metric (HR coverage > 0.5, rotated+disc, same as
     audit_v6_density.py). Gate: mid median in [0.05, 0.16]; high median
     >= 0.16; pooled max >= 0.40 (capability v6 lacked); no scene < 0.02.
  G4 fragmentation  [density-audit: v6 dust median comp area 13px]
     Gate: >= 90% of structure mass in components >= 100 HR px (median over
     scenes >= 0.90); p95 component count <= 1500.
  G5 in-scene sparse/dense contrast  [owner 疏密 verdict; the r4 fix target]
     Interior-crop 8x8 tile occupancy std, compared with the SAME statistic on
     v6-current geometry (tcforge, pool motif weights). Gate: v7 median tile
     std >= 1.5x v6 median.
  G6 dot placement  [dots pilot silent-floor accident: 11/24 zero-dot scenes]
     Gate: 100% of scenes place >= 20 dots; pool-wide both strata present:
     isolated-on-uniform >= 15%, structure-embedded >= 15% (ACL-063 two-class
     requirement).
  G7 bright/dark diversity  [owner level_min 0.6 verdict]
     Per-scene level spread across components >= 50 HR px. Gate: >= 80% of
     scenes have spread >= 0.15.
  G8 mask-defect coverage
     Gate: >= 95% scenes have >= 1 edge notch; >= 40% have >= 1 broken trace.

Runs the r4 prototype (scripts/v7_composer_demo.py) as a library. The real
pool audit (verify_pool_sharpness.py band energies, defect schema checks)
happens after tcforge integration — this file gates the COMPOSER GEOMETRY.

Usage:
  uv run python scripts/audit_v7_demo_gates.py \
      [--n-mid 40 --n-high 40 --n-xl 8 --n-v6 24] \
      [--out research_log/assets/v7_planning/composer_demo_r4]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
import v7_composer_demo as D  # noqa: E402

TCFORGE_SRC = SCRIPTS.parent / "tcforge" / "src"
if TCFORGE_SRC.exists() and str(TCFORGE_SRC) not in sys.path:
    sys.path.insert(0, str(TCFORGE_SRC))
from tcforge.geometry import build_scene_mask_with_metadata  # noqa: E402

V6_WEIGHTS = {  # pool_2x_v6_cpu.json verbatim
    "pga_grid": 0.28, "die_bga": 0.24, "multi_die": 0.18,
    "trace_bus": 0.14, "heat_spreader": 0.10, "generic": 0.06,
}


def _disc_struct(radius_px: float) -> np.ndarray:
    r = int(np.ceil(radius_px))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return (np.hypot(yy, xx) <= radius_px)


# 28um diameter opening at draw resolution (UPD um/px)
OPEN_STRUCT = _disc_struct(14.0 / D.UPD)


# Metric v2: disc opening counts the sharp corners of every LEGIT >=28um
# axis-aligned rectangle/pad as "sliver" (~(4-pi)r^2 ≈ 15 draw px per corner;
# a 1000-pad scene accumulates this into the p95 driver — measured round 2).
# Real sub-floor strips are elongated (>=28um x >=200um ≈ 250+ px). Filtering
# sliver components below MIN_SLIVER_PX cleanly separates the two.
MIN_SLIVER_PX = 60


def _sliver_frac(region: np.ndarray, denom: float) -> float:
    resid = region & ~ndimage.binary_opening(region, structure=OPEN_STRUCT)
    if not resid.any():
        return 0.0
    lab, n = ndimage.label(resid)
    areas = np.bincount(lab.ravel())[1:]
    return float(areas[areas >= MIN_SLIVER_PX].sum()) / denom


def sliver_fractions(mask_draw: np.ndarray) -> tuple[float, float]:
    """(bright sliver frac of structure, gap sliver frac of in-bbox background)
    via 28um-disc opening on the unrotated draw-res binary mask; only sliver
    components >= MIN_SLIVER_PX count (metric v2, corner-artifact filtered)."""
    m = mask_draw.astype(bool)
    if not m.any():
        return 0.0, 0.0
    bright_sliver = _sliver_frac(m, float(m.sum()))
    ys, xs = np.where(m)
    pad = int(100.0 / D.UPD)                       # 100um margin inside bbox
    y0, y1 = ys.min() + pad, ys.max() - pad
    x0, x1 = xs.min() + pad, xs.max() - pad
    if y1 - y0 < 10 or x1 - x0 < 10:
        return bright_sliver, 0.0
    bg = ~m[y0:y1, x0:x1]
    if not bg.any():
        return bright_sliver, 0.0
    gap_sliver = _sliver_frac(bg, float(bg.sum()))
    return bright_sliver, gap_sliver


def tile_occ_std(cov_hr: np.ndarray, grid: int = 8) -> float:
    """Std of tile occupancy over a grid x grid partition, restricted to tiles
    that intersect the structure bounding box (metric v2). v1 included pure
    background tiles, so the structure-vs-empty split dominated the statistic
    for BOTH pools and masked the in-structure sparse/dense contrast the gate
    is meant to measure. Applied identically to v7 and the v6 baseline."""
    crop = D.interior_crop(cov_hr)
    binm = crop > 0.5
    if not binm.any():
        return 0.0
    ys, xs = np.where(binm)
    h, w = crop.shape
    th, tw = h // grid, w // grid
    occ = []
    for i in range(grid):
        for j in range(grid):
            y0, y1 = i * th, (i + 1) * th
            x0, x1 = j * tw, (j + 1) * tw
            if y1 < ys.min() or y0 > ys.max() or x1 < xs.min() or x0 > xs.max():
                continue                            # tile outside structure bbox
            occ.append(float(binm[y0:y1, x0:x1].mean()))
    return float(np.std(occ)) if len(occ) >= 4 else 0.0


def component_stats(cov_hr: np.ndarray) -> tuple[int, float]:
    """(n components, structure-mass fraction in components >= 100 HR px)."""
    binm = cov_hr > 0.5
    lab, n = ndimage.label(binm)
    if n == 0:
        return 0, 1.0
    areas = np.bincount(lab.ravel())[1:]
    total = float(areas.sum())
    return int(n), float(areas[areas >= 100].sum()) / max(total, 1.0)


def level_spread(cov_hr: np.ndarray, lvl_img: np.ndarray) -> float:
    """Spread (max-min) of per-component mean levels, comps >= 50 HR px."""
    binm = cov_hr > 0.5
    lab, n = ndimage.label(binm)
    if n == 0:
        return 0.0
    areas = np.bincount(lab.ravel())[1:]
    idx = np.where(areas >= 50)[0] + 1
    if len(idx) < 2:
        return 0.0
    means = ndimage.mean(np.divide(lvl_img, np.maximum(cov_hr, 1e-6)),
                         lab, index=idx)
    return float(np.max(means) - np.min(means))


def dot_strata(cov_hr: np.ndarray, dots: list, win: int = 12) -> tuple[int, int]:
    """(isolated-on-uniform, structure-embedded) dot counts. A dot is isolated
    when its local window is ~pure structure (background fraction < 5%)."""
    binm = cov_hr > 0.5
    iso = emb = 0
    for (y, x, _r, _d) in dots:
        y0, y1 = max(y - win, 0), min(y + win + 1, binm.shape[0])
        x0, x1 = max(x - win, 0), min(x + win + 1, binm.shape[1])
        if (~binm[y0:y1, x0:x1]).mean() < 0.05:
            iso += 1
        else:
            emb += 1
    return iso, emb


def audit_scene(seed: int, tier: str, force_xl: bool) -> dict:
    rng = np.random.default_rng(seed)
    cv, meta = D.compose_scene(rng, tier, force_xl=force_xl)
    bright_sl, gap_sl = sliver_fractions(cv.final())
    angle = D._u(rng, 0.0, 360.0)
    cov = D.render_coverage(cv, angle)
    lvl = D.level_render(cov, rng, zones=meta.zones, angle=angle)
    _T, _bg, _dT, dfx = D.temp_render(rng, cov, lvl)
    n_comp, mass_big = component_stats(cov)
    iso, emb = dot_strata(cov, dfx.dots)
    return {
        "tier": tier + ("_xl" if force_xl else ""),
        "seed": seed,
        "occ": float((cov > 0.5).mean()),
        "bright_sliver": bright_sl,
        "gap_sliver": gap_sl,
        "tile_std": tile_occ_std(cov),
        "n_comp": n_comp,
        "mass_big": mass_big,
        "lvl_spread": level_spread(cov, lvl),
        "n_dots": len(dfx.dots),
        "dots_iso": iso,
        "dots_emb": emb,
        "n_notch": len(meta.notch_pts),
        "n_break": len(meta.break_pts),
    }


def audit_v6_scene(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    mask, _meta = build_scene_mask_with_metadata(
        "medium", seed,
        rotation_deg_center=float(rng.uniform(0.0, 360.0)),
        rotation_jitter_deg=0.0,
        canvas_shape=(960, 1280), pixel_size_um=20.0, scale=2,
        antialias=True, ssaa_factor=4, inscribe_disc=True,
        motif_weights=V6_WEIGHTS,
    )
    return {"tile_std": tile_occ_std(np.asarray(mask, dtype=np.float32))}


def q(vals, p):
    return float(np.percentile(np.asarray(vals, dtype=np.float64), p))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mid", type=int, default=40)
    ap.add_argument("--n-high", type=int, default=40)
    ap.add_argument("--n-xl", type=int, default=8)
    ap.add_argument("--n-v6", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=2020001)
    ap.add_argument("--out", type=Path,
                    default=Path("research_log/assets/v7_planning/composer_demo_r4"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    plan = ([("mid", False)] * args.n_mid + [("high", False)] * args.n_high
            + [("high", True)] * args.n_xl)
    for k, (tier, xl) in enumerate(plan):
        rows.append(audit_scene(args.seed0 + k, tier, xl))
        if (k + 1) % 20 == 0:
            print(f"  ... {k + 1}/{len(plan)} v7 scenes")
    v6_stds = [audit_v6_scene(args.seed0 + 900000 + k)["tile_std"]
               for k in range(args.n_v6)]
    print(f"  ... {args.n_v6} v6 baseline scenes")

    def col(name, tiers=None):
        return [r[name] for r in rows
                if tiers is None or r["tier"] in tiers]

    occ_mid = col("occ", {"mid"})
    occ_high = col("occ", {"high", "high_xl"})
    iso_total = sum(col("dots_iso"))
    emb_total = sum(col("dots_emb"))
    dots_total = max(iso_total + emb_total, 1)
    v7_tile_med = q(col("tile_std"), 50)
    v6_tile_med = q(v6_stds, 50)

    gates = {
        "G1_bright_floor": {
            "median": q(col("bright_sliver"), 50), "p95": q(col("bright_sliver"), 95),
            "pass": q(col("bright_sliver"), 50) <= 0.005
                    and q(col("bright_sliver"), 95) <= 0.015,
            "rule": "median<=0.5%, p95<=1.5% of structure area",
        },
        "G2_gap_floor": {
            "median": q(col("gap_sliver"), 50), "p95": q(col("gap_sliver"), 95),
            "pass": q(col("gap_sliver"), 50) <= 0.010
                    and q(col("gap_sliver"), 95) <= 0.030,
            "rule": "median<=1.0%, p95<=3.0% of in-bbox background",
        },
        "G3_occupancy": {
            "mid_median": q(occ_mid, 50), "high_median": q(occ_high, 50),
            "pool_max": max(col("occ")), "pool_min": min(col("occ")),
            "pass": (0.05 <= q(occ_mid, 50) <= 0.16)
                    and q(occ_high, 50) >= 0.16
                    and max(col("occ")) >= 0.40 and min(col("occ")) >= 0.02,
            "rule": "mid med in [.05,.16]; high med>=.16; max>=.40; min>=.02",
        },
        "G4_fragmentation": {
            "mass_big_median": q(col("mass_big"), 50),
            "n_comp_p95": q(col("n_comp"), 95),
            "pass": q(col("mass_big"), 50) >= 0.90
                    and q(col("n_comp"), 95) <= 1500,
            "rule": "mass-in->=100px comps median>=0.90; n_comp p95<=1500",
        },
        "G5_inscene_contrast": {
            "v7_tile_std_median": v7_tile_med, "v6_tile_std_median": v6_tile_med,
            "ratio": v7_tile_med / max(v6_tile_med, 1e-9),
            "pass": v7_tile_med >= 1.5 * v6_tile_med,
            "rule": "v7 median tile-occ std >= 1.5x v6 median",
        },
        "G6_dots": {
            "min_dots": min(col("n_dots")),
            "iso_share": iso_total / dots_total,
            "emb_share": emb_total / dots_total,
            "pass": min(col("n_dots")) >= 20
                    and iso_total / dots_total >= 0.15
                    and emb_total / dots_total >= 0.15,
            "rule": "all scenes >=20 dots; iso>=15%; embedded>=15%",
        },
        "G7_levels": {
            "spread_ge_015_share":
                float(np.mean([s >= 0.15 for s in col("lvl_spread")])),
            "pass": float(np.mean([s >= 0.15 for s in col("lvl_spread")])) >= 0.80,
            "rule": ">=80% scenes with component-level spread >=0.15",
        },
        "G8_mask_defects": {
            "notch_share": float(np.mean([n >= 1 for n in col("n_notch")])),
            "break_share": float(np.mean([n >= 1 for n in col("n_break")])),
            "pass": float(np.mean([n >= 1 for n in col("n_notch")])) >= 0.95
                    and float(np.mean([n >= 1 for n in col("n_break")])) >= 0.40,
            "rule": "notch>=1 in >=95%; break>=1 in >=40%",
        },
    }

    print("\n== v7 composer demo r4 — gate results ==")
    all_pass = True
    for name, g in gates.items():
        status = "PASS" if g["pass"] else "FAIL"
        all_pass &= bool(g["pass"])
        detail = {k: (round(v, 4) if isinstance(v, float) else v)
                  for k, v in g.items() if k not in ("pass", "rule")}
        print(f"  [{status}] {name}: {detail}   rule: {g['rule']}")
    print(f"\n  OVERALL: {'ALL PASS' if all_pass else 'FAILURES PRESENT'}")

    out_json = args.out / "gate_audit.json"
    out_json.write_text(json.dumps(
        {"gates": gates, "n_scenes": len(rows), "rows": rows,
         "v6_tile_stds": v6_stds}, indent=2, default=float))
    print("wrote", out_json.resolve())

    fig, axes = plt.subplots(2, 3, figsize=(16, 8.5))
    fig.suptitle("v7 demo r4 gate audit — distributions "
                 f"(n={len(rows)} v7 scenes, {args.n_v6} v6 baseline)",
                 fontsize=13)
    panels = [
        ("occupancy (full-frame)", col("occ"), None),
        ("bright sliver frac", col("bright_sliver"), 0.015),
        ("gap sliver frac", col("gap_sliver"), 0.030),
        ("tile-occ std (v7 vs v6)", col("tile_std"), None),
        ("mass in >=100px comps", col("mass_big"), 0.90),
        ("dots per scene", col("n_dots"), 20),
    ]
    for ax, (title, vals, line) in zip(axes.ravel(), panels):
        ax.hist(vals, bins=24, color="#e08214", alpha=0.85)
        if title.startswith("tile-occ"):
            ax.hist(v6_stds, bins=24, color="#5e3c99", alpha=0.6, label="v6")
            ax.legend(fontsize=8)
        if line is not None:
            ax.axvline(line, color="crimson", ls="--", lw=1)
        ax.set_title(title, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig_path = args.out / "gate_audit_hist.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fig_path.resolve())


if __name__ == "__main__":
    main()
