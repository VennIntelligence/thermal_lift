#!/usr/bin/env python
"""Verify a generated pool's GT sharpness (ACL-030 / v5 check) + basic integrity.

The v5 requirement: the SAVED ground truth (`hr_temperature_2x.npy`) must carry
recoverable super-resolution-band detail again. We measure `out_of_band_ratio`
of the GT directly:

    edge_sigma=1.4 (v4, blurry):  out_of_band(GT) ~ 0.00008   <- the target had no detail
    edge_sigma=0.8 (v5, sharp):   out_of_band(GT) ~ 0.002-0.005

So a v5 pool PASSES if the median GT out_of_band is well above the v4 floor
(>= 0.0015, ~20x v4) and not absurdly high (<= 0.02; above that the GT is sub-pitch
and would demand hallucination -- the FM-1 beading risk).

numpy-only (no torch), so it runs in the data-generation env.

    python scripts/verify_pool_sharpness.py --pool data/synthetic/pool_2x_v5_5k --k 96
    python scripts/verify_pool_sharpness.py --pool ... --out output/v5_gt_crops    # also dump PNGs
    python scripts/verify_pool_sharpness.py --pool ... --ref-pool data/synthetic/pool_2x_v4_5k  # A/B vs v4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

CRITICAL = "hr_temperature_2x.npy"
EXPECT_FILES = [CRITICAL, "phase_bin_drizzle_2x.npy", "obs_features_1x.npz",
                "lr_burst.npy", "shifts.npy", "hr_mask_4x.png", "metadata.json"]
V4_REF = 0.00008          # edge_sigma=1.4 reference (the blurry target)
# Calibrated through the REAL AA+defects pipeline (ACL-030): the SSAA coverage mask softens edges
# ~0.7 HR px on its own, so edge_sigma 0.8 only reaches ~0.0013 (still soft); 0.6 -> ~0.0039 (edges
# at ~1 pitch = target); <=0.4 -> >=0.013 (approaches the AA floor = sub-pitch). Target edge_sigma=0.6.
PASS_MIN = 0.0025         # below -> too soft (e.g. edge_sigma>=0.8, lost to AA) -> sharpen the GT
PASS_MAX = 0.010          # above -> sub-pitch (edge_sigma<=0.4) -> hallucination / FM-1 beading risk
SCALE = 2


def out_of_band_ratio(img: np.ndarray, scale: int = SCALE) -> float:
    """Fraction of spectral energy above the LR-Nyquist (recoverable) cutoff."""
    arr = np.nan_to_num(np.asarray(img, np.float64))
    arr = arr - arr.mean()
    h, w = arr.shape
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    power = np.abs(np.fft.fftshift(np.fft.fft2(arr * win))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    radial = np.sqrt(fy ** 2 + fx ** 2)
    total = float(power.sum())
    return float(power[radial > 0.5 / scale].sum() / total) if total > 0 else 0.0


def scan_pool(pool: Path, k: int, seed: int) -> dict:
    scenes = sorted(pool.glob("scene_*"))
    if not scenes:
        raise SystemExit(f"FAIL: no scene_* dirs under {pool}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(scenes), size=min(k, len(scenes)), replace=False)
    pick = [scenes[int(i)] for i in idx]

    oob, p99, missing, shapes, dtypes, has_defects, crops = [], [], [], set(), set(), 0, []
    for sd in pick:
        for f in EXPECT_FILES:
            if not (sd / f).exists():
                missing.append(f"{sd.name}/{f}")
        gtp = sd / CRITICAL
        if not gtp.exists():
            continue
        raw = np.load(gtp)
        gt = raw.astype(np.float64)
        shapes.add(tuple(gt.shape))
        dtypes.add(str(raw.dtype))
        oob.append(out_of_band_ratio(gt))
        gy, gx = np.gradient(gt)
        p99.append(float(np.percentile(np.sqrt(gx * gx + gy * gy), 99)))
        if len(crops) < 4:
            crops.append((sd.name, gt))
        mp = sd / "metadata.json"
        if mp.exists():
            md = json.loads(mp.read_text())
            geo = md.get("geometry_metadata") or md.get("geo_meta") or md
            d = geo.get("defects") if isinstance(geo, dict) else None
            # defect_meta is a counts dict: {holes, notches, cracks, severity}
            if isinstance(d, dict) and any(int(d.get(k2, 0) or 0) > 0 for k2 in ("holes", "notches", "cracks")):
                has_defects += 1
    return {
        "n_total": len(scenes), "n_checked": len(oob), "oob": np.array(oob),
        "p99": np.array(p99), "missing": missing, "shapes": shapes,
        "dtypes": dtypes, "has_defects": has_defects, "crops": crops,
    }


def _summ(a: np.ndarray) -> str:
    return (f"median={np.median(a):.5f}  mean={a.mean():.5f}  "
            f"min={a.min():.5f}  max={a.max():.5f}")


def dump_crops(crops, out: Path) -> None:
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    for name, gt in crops:
        h, w = gt.shape
        cy, cx, rh, rw = h // 2, w // 2, h // 6, w // 6
        crop = gt[cy - rh:cy + rh, cx - rw:cx + rw]
        lo, hi = np.percentile(crop, [1, 99])
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(crop, cmap="inferno", vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(f"GT center crop — {name}")
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046).set_label("deg C")
        fig.savefig(out / f"gt_{name}.png", dpi=150, bbox_inches="tight")
        plt.close(fig)
    print(f"  wrote {len(crops)} GT center-crop PNGs -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--k", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="optional dir to dump GT center-crop PNGs")
    ap.add_argument("--ref-pool", default="", help="optional pool to A/B against (e.g. v4)")
    args = ap.parse_args()

    r = scan_pool(Path(args.pool), args.k, args.seed)
    print(f"\n=== {args.pool}  ({r['n_checked']}/{r['n_total']} scenes sampled) ===")
    print(f"  GT shape(s): {r['shapes']}   dtype(s): {r['dtypes']}")
    print(f"  out_of_band(GT):  {_summ(r['oob'])}")
    print(f"  p99 |grad| (C/px): {_summ(r['p99'])}")
    print(f"  scenes with defects in metadata: {r['has_defects']}/{r['n_checked']}")
    print(f"  reference: v4 edge_sigma=1.4 ~ {V4_REF}  |  v5 edge_sigma=0.6 expect ~0.003-0.005 (real AA pipeline)")

    if args.ref_pool:
        rr = scan_pool(Path(args.ref_pool), args.k, args.seed)
        ratio = np.median(r["oob"]) / max(np.median(rr["oob"]), 1e-12)
        print(f"\n  A/B vs {args.ref_pool}: median out_of_band {np.median(rr['oob']):.5f} "
              f"-> {np.median(r['oob']):.5f}  ({ratio:.1f}x)")

    if args.out and r["crops"]:
        dump_crops(r["crops"], Path(args.out))

    med = float(np.median(r["oob"]))
    ok = True
    if r["missing"]:
        ok = False
        print(f"\n  ! MISSING FILES ({len(r['missing'])}): {r['missing'][:6]}{' ...' if len(r['missing']) > 6 else ''}")
    if med < PASS_MIN:
        ok = False
        print(f"\n  ! GT out_of_band median {med:.5f} < {PASS_MIN} — sharpening did NOT land "
              f"(still ~v4-blurry). Check edge_sigma in the config / that the right pool was generated.")
    elif med > PASS_MAX:
        ok = False
        print(f"\n  ! GT out_of_band median {med:.5f} > {PASS_MAX} — GT is sub-pitch sharp; "
              f"risks hallucination/beading. Consider raising edge_sigma toward 0.8-1.0.")

    print(f"\n{'PASS' if ok else 'FAIL'}: GT sharpness "
          f"({med:.5f}, ~{med / V4_REF:.0f}x v4) + integrity"
          f"{'' if ok else ' — see warnings above'}\n")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
