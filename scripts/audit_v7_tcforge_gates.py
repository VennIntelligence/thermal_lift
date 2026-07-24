#!/usr/bin/env python3
"""Re-run the v7 pre-registered gates G1-G8 against the REAL tcforge pipeline (§6 test 14).

scripts/audit_v7_demo_gates.py is a read-only reference that gates the PROTOTYPE composer;
it cannot be modified to add an --engine flag. This sibling reuses its (UPD-independent) gate
metrics verbatim and drives the tcforge production path instead:

  geometry.build_scene_mask_with_metadata(scene_composer="panel_cluster_v7") for the composer,
  realism.carve_trace_breaks / apply_defects (dots config) / render_isothermal_field for the
  realism-stage defects and per-component levels.

Thresholds are IDENTICAL to the prototype gates. G5 (sparse/dense contrast) is marginal by
design in the prototype (1.446x vs 1.5x) and is NOT the integration's responsibility — the
overall bar is 7/8.

Usage:
  uv run python scripts/audit_v7_tcforge_gates.py [--n-mid 40 --n-high 40 --n-xl 8 --n-v6 24]
  （另有 --seed0 起始种子，默认 2020001）

输入: tcforge/src（panel_cluster_v7 composer + realism）与 scripts/audit_v7_demo_gates.py 的门控函数
输出: 仅终端 PASS/FAIL 表 + JSON 行，不写文件；exit 0 = 通过（>=7/8 门控）
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

SCRIPTS = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPTS))
TCFORGE_SRC = SCRIPTS.parent / "tcforge" / "src"
if str(TCFORGE_SRC) not in sys.path:
    sys.path.insert(0, str(TCFORGE_SRC))

import audit_v7_demo_gates as G  # noqa: E402  (metric functions + v6 baseline, read-only import)
from tcforge.geometry import build_scene_mask_with_metadata  # noqa: E402
from tcforge import realism  # noqa: E402

HR_SHAPE = (960, 1280)
CANVAS_W_UM = HR_SHAPE[1] * 10.0
CANVAS_H_UM = HR_SHAPE[0] * 10.0
SSAA = 4
UPD_TC = CANVAS_W_UM / (HR_SHAPE[1] * SSAA)   # 2.5 um/draw px at ssaa=4
DRAW_SHAPE = (HR_SHAPE[0] * SSAA, HR_SHAPE[1] * SSAA)
OPEN_STRUCT_TC = G._disc_struct(14.0 / UPD_TC)   # 28um-diameter opening at tc draw res

# v7 dots-pilot defect config (§3.5) + broken-trace / margin knobs (§3.4).
DEFECT_KW = dict(
    severity_range=(1.0, 1.0), hole_radius_px=(1, 4), notch_radius_px=(3, 10),
    crack_len_px=(60, 260), crack_width_px=(2, 4), max_holes=50, max_notches=6,
    max_cracks=4, min_holes=20, min_notches=2, hole_depth_range=(0.3, 1.0),
    hole_edge_softness_px=1.0, hole_margin_px=8, hole_margin_adaptive=True,
    hole_edge_fraction=0.12,
)


def _sliver_frac_tc(region: np.ndarray, denom: float) -> float:
    resid = region & ~ndimage.binary_opening(region, structure=OPEN_STRUCT_TC)
    if not resid.any():
        return 0.0
    lab, _ = ndimage.label(resid)
    areas = np.bincount(lab.ravel())[1:]
    return float(areas[areas >= G.MIN_SLIVER_PX].sum()) / denom


def sliver_fractions_tc(mask_draw: np.ndarray) -> tuple[float, float]:
    m = mask_draw.astype(bool)
    if not m.any():
        return 0.0, 0.0
    bright = _sliver_frac_tc(m, float(m.sum()))
    ys, xs = np.where(m)
    pad = int(100.0 / UPD_TC)
    y0, y1 = ys.min() + pad, ys.max() - pad
    x0, x1 = xs.min() + pad, xs.max() - pad
    if y1 - y0 < 10 or x1 - x0 < 10:
        return bright, 0.0
    bg = ~m[y0:y1, x0:x1]
    return (bright, _sliver_frac_tc(bg, float(bg.sum())) if bg.any() else 0.0)


def _finalize_hr(mask_draw: np.ndarray, angle: float) -> np.ndarray:
    """Disc-inscribe + order-1 rotate + block-average downsample (mirrors _finalize_scene_mask)."""
    dr, dc = mask_draw.shape
    yy, xx = np.mgrid[:dr, :dc]
    disc = ((yy - dr / 2.0) ** 2 + (xx - dc / 2.0) ** 2) <= (min(dr, dc) / 2.0) ** 2
    m = (mask_draw.astype(bool) & disc).astype(np.float32)
    if abs(angle) >= 0.01:
        m = ndimage.rotate(m, angle, reshape=False, order=1, mode="constant")
        m = np.clip(m, 0.0, 1.0)
    return np.clip(m.reshape(HR_SHAPE[0], SSAA, HR_SHAPE[1], SSAA).mean(axis=(1, 3)), 0.0, 1.0)


def audit_scene_tc(seed: int, tier: str, force_xl: bool) -> dict:
    from tcforge.composer_v7 import compose_panel_cluster_scene
    rng = np.random.default_rng(seed)
    ft = "xl" if force_xl else tier
    common = dict(canvas_shape=DRAW_SHAPE, pixel_size_um=20.0, scale=2 * SSAA)
    mask_draw, _prims, extra = compose_panel_cluster_scene(
        rng, params={"force_tier": ft}, common=common, draw_shape=DRAW_SHAPE,
        canvas_w_um=CANVAS_W_UM, canvas_h_um=CANVAS_H_UM, detector_pitch_um=20.0)
    bright_sl, gap_sl = sliver_fractions_tc(mask_draw)
    angle = float(rng.uniform(0.0, 360.0))
    cov = _finalize_hr(mask_draw, angle)

    # realism-stage defects on the post-rotation HR coverage
    center = ((HR_SHAPE[0] - 1) / 2.0, (HR_SHAPE[1] - 1) / 2.0)
    cov_b, trace_inst, nid = realism.carve_trace_breaks(
        cov, extra["traces"], rng, scene_rotation_deg=angle, canvas_center_yx=center,
        hr_pitch_um=10.0, break_p=0.7, count_range=(1, 2), gap_px=(6.0, 20.0),
        record_instances=True, label_map=None, next_id=1)
    defected, dmeta = realism.apply_defects(
        cov_b, rng, record_instances=True, label_map=None, next_id=nid, **DEFECT_KW)
    inst = dmeta.get("instances", [])
    holes = [i for i in inst if i["type"] == "hole"]
    dots = [(i["center_yx_hr"][0], i["center_yx_hr"][1], i["radius_px"], i["depth_or_amplitude"])
            for i in holes]
    n_notch = sum(1 for i in inst if i["type"] == "notch")
    n_break = len(trace_inst)

    # per-component level image (level_min 0.60, anchored) for G7
    lvl = realism.render_isothermal_field(
        defected, rng, t_bg_c=0.0, delta_t_c=1.0, level_min=0.60, edge_sigma=0.6,
        zones=(extra["zones"] or None), zone_rotation_deg=angle, zone_level_jitter=0.03,
        hr_pitch_um=10.0, stratified_anchor=True)

    n_comp, mass_big = G.component_stats(cov)
    iso, emb = G.dot_strata(defected, dots)
    return {
        "tier": tier + ("_xl" if force_xl else ""), "seed": seed,
        "occ": float((cov > 0.5).mean()), "bright_sliver": bright_sl, "gap_sliver": gap_sl,
        "tile_std": G.tile_occ_std(cov), "n_comp": n_comp, "mass_big": mass_big,
        "lvl_spread": G.level_spread(defected, lvl), "n_dots": len(dots),
        "dots_iso": iso, "dots_emb": emb, "n_notch": n_notch, "n_break": n_break,
    }


def q(vals, p):
    return float(np.percentile(np.asarray(vals, dtype=np.float64), p))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-mid", type=int, default=40)
    ap.add_argument("--n-high", type=int, default=40)
    ap.add_argument("--n-xl", type=int, default=8)
    ap.add_argument("--n-v6", type=int, default=24)
    ap.add_argument("--seed0", type=int, default=2020001)
    args = ap.parse_args()

    plan = ([("mid", False)] * args.n_mid + [("high", False)] * args.n_high
            + [("high", True)] * args.n_xl)
    rows = [audit_scene_tc(args.seed0 + k, t, xl) for k, (t, xl) in enumerate(plan)]
    v6_stds = [G.audit_v6_scene(args.seed0 + 900000 + k)["tile_std"] for k in range(args.n_v6)]

    def col(name, tiers=None):
        return [r[name] for r in rows if tiers is None or r["tier"] in tiers]

    occ_mid = col("occ", {"mid"})
    occ_high = col("occ", {"high", "high_xl"})
    iso_t, emb_t = sum(col("dots_iso")), sum(col("dots_emb"))
    dots_total = max(iso_t + emb_t, 1)
    v7_med, v6_med = q(col("tile_std"), 50), q(v6_stds, 50)

    gates = {
        "G1_bright_floor": (q(col("bright_sliver"), 50) <= 0.005 and q(col("bright_sliver"), 95) <= 0.015,
                            {"median": q(col("bright_sliver"), 50), "p95": q(col("bright_sliver"), 95)}),
        "G2_gap_floor": (q(col("gap_sliver"), 50) <= 0.010 and q(col("gap_sliver"), 95) <= 0.030,
                         {"median": q(col("gap_sliver"), 50), "p95": q(col("gap_sliver"), 95)}),
        "G3_occupancy": ((0.05 <= q(occ_mid, 50) <= 0.16) and q(occ_high, 50) >= 0.16
                         and max(col("occ")) >= 0.40 and min(col("occ")) >= 0.02,
                         {"mid_med": q(occ_mid, 50), "high_med": q(occ_high, 50),
                          "max": max(col("occ")), "min": min(col("occ"))}),
        "G4_fragmentation": (q(col("mass_big"), 50) >= 0.90 and q(col("n_comp"), 95) <= 1500,
                             {"mass_big_med": q(col("mass_big"), 50), "n_comp_p95": q(col("n_comp"), 95)}),
        "G5_inscene_contrast": (v7_med >= 1.5 * v6_med,
                                {"v7": v7_med, "v6": v6_med, "ratio": v7_med / max(v6_med, 1e-9)}),
        "G6_dots": (min(col("n_dots")) >= 20 and iso_t / dots_total >= 0.15 and emb_t / dots_total >= 0.15,
                    {"min_dots": min(col("n_dots")), "iso": iso_t / dots_total, "emb": emb_t / dots_total}),
        "G7_levels": (float(np.mean([s >= 0.15 for s in col("lvl_spread")])) >= 0.80,
                      {"spread>=0.15_share": float(np.mean([s >= 0.15 for s in col("lvl_spread")]))}),
        "G8_mask_defects": (float(np.mean([n >= 1 for n in col("n_notch")])) >= 0.95
                            and float(np.mean([n >= 1 for n in col("n_break")])) >= 0.40,
                            {"notch_share": float(np.mean([n >= 1 for n in col("n_notch")])),
                             "break_share": float(np.mean([n >= 1 for n in col("n_break")]))}),
    }
    print(f"\n== v7 tcforge gates (n={len(rows)} scenes, {args.n_v6} v6 baseline) ==")
    n_pass = 0
    for name, (ok, detail) in gates.items():
        n_pass += bool(ok)
        d = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in detail.items()}
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {d}")
    non_g5 = sum(1 for k, (ok, _) in gates.items() if ok and k != "G5_inscene_contrast")
    print(f"\n  {n_pass}/8 pass ({non_g5}/7 excluding G5). Bar: >=7/8 (G5 not in scope).")
    print(json.dumps({k: bool(v[0]) for k, v in gates.items()}))
    return 0 if n_pass >= 7 or non_g5 >= 7 else 1


if __name__ == "__main__":
    raise SystemExit(main())
