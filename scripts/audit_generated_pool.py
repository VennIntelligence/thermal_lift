#!/usr/bin/env python3
"""Audit a generated v3 pool from the unrolled-solver's standpoint (numpy only).

For K sampled scenes it verifies the data is SELF-CONSISTENT and HONEST:
  * GT<->burst consistency: recompute clean A(reconstruct(hr_mask, metadata)) with the SAVED
    shifts + per-scene PSF, compare to the SAVED burst after highpass (drift removed).  High
    correlation => the save/reconstruct/metadata/shift/PSF roundtrip is sound; low => a bug.
  * residual level vs the recorded noise floor (sanity on noise/drift/defect magnitude).
  * band honesty (ACL-023): GT energy above LR-Nyquist, and above the per-scene PSF MTF<0.1.
  * shapes/dtypes/ranges sanity (obs 5ch, burst (N,h,w), shifts (N,2), psf/scale/pitch).

Run on the remote box that holds the pool:
    python scripts/audit_generated_pool.py --pool data/synthetic/pool_2x_v3_5k --k 32
Exit 0 = pool looks good; nonzero = investigate before committing to the long run.

本地运行: uv run python scripts/audit_generated_pool.py --pool data/synthetic/pool_2x_v3_5k --k 32
输入: --pool 指定的合成场景池目录（scene_*/，由 generate_training_pool.py 生成）
输出: 终端逐场景表格 + PASS/INVESTIGATE 结论（exit 0/1/2）；--out 指定路径时另写 JSON 汇总
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "tcforge" / "src"))
from tcforge import generate_lr_burst, reconstruct_hr_temperature  # noqa: E402
from tcforge.storage import load_scene_compact  # noqa: E402


def gaussian_blur(x, sigma):
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(x, sigma=sigma, mode="reflect")


def highpass(x, sigma=5.0):
    return x - gaussian_blur(x, sigma)


def radial_power_fraction(field, f_lo, f_hi=0.5):
    f = field - field.mean()
    F = np.abs(np.fft.fftshift(np.fft.fft2(f))) ** 2
    H, W = f.shape
    fy = np.fft.fftshift(np.fft.fftfreq(H))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(W))[None, :]
    rr = np.sqrt(fy ** 2 + fx ** 2)
    return float(F[(rr > f_lo) & (rr <= f_hi)].sum() / max(F.sum(), 1e-30))


def mtf_cutoff(sigma_hr, thresh=0.1):
    fg = np.linspace(0, 0.5, 512)
    m = np.exp(-2.0 * (np.pi * sigma_hr * fg) ** 2)
    return float(fg[np.argmax(m < thresh)]) if (m < thresh).any() else 0.5


def check_defect_schema(scene: dict) -> dict:
    """v7 defect-annotation spot check (§6 test 15): schema_version 2, dense 1-based ids,
    label-map ids subset of the instance ids, dot (hole) count. Absent annotations => inert
    ({} => scene ignored by the pool-level zero-dot / schema aggregation)."""
    m = scene["metadata"]
    da = m.get("defect_annotations")
    if not da:
        return {}
    inst = da.get("instances", [])
    ids = [i.get("id") for i in inst]
    problems: list[str] = []
    if da.get("schema_version") != 2:
        problems.append(f"schema_version={da.get('schema_version')}")
    if ids != list(range(1, len(inst) + 1)):
        problems.append("ids not dense/1-based")
    di = scene.get("defect_instances")
    if di is not None:
        import numpy as _np
        lm_ids = set(int(v) for v in _np.unique(di)) - {0}
        if not lm_ids.issubset(set(ids)):
            problems.append("label-map ids not subset of instances")
    counts = da.get("counts_by_type", {})
    n_dots = int(counts.get("hole", sum(1 for i in inst if i.get("type") == "hole")))
    return {"has_annotations": True, "n_dots": n_dots, "defect_problems": problems}


def audit_scene(scene_dir: Path) -> dict:
    s = load_scene_compact(scene_dir)
    m = s["metadata"]
    scale = int(m["scale"])
    shifts = np.asarray(s["shifts"], dtype=np.float32)
    burst = np.asarray(s["lr_burst"], dtype=np.float32)  # (N,h,w)
    obs = np.asarray(s["obs_features"])
    target = reconstruct_hr_temperature(
        s["hr_mask"],
        T_bg_c=float(m["T_bg_c"]), delta_T_c=float(m["delta_T_c"]),
        low_freq_amplitude_c=float(m["low_freq_amplitude_c"]),
        low_freq_sigma_px=float(m["low_freq_sigma_px"]), seed=int(m["low_freq_seed"]),
    )
    clean = generate_lr_burst(
        target.astype(np.float32), shifts, forward_mode=m["forward_mode"],
        psf_sigma_lr_px=float(m["psf_sigma_lr_px"]), psf_shape=m.get("psf_shape", "gaussian"),
        psf_sigma_y_lr_px=m.get("psf_sigma_y_lr_px"), psf_angle_deg=float(m.get("psf_angle_deg", 0.0)),
        scale=scale,
    )
    n = min(len(burst), len(clean))
    corrs, rms = [], []
    for i in range(n):
        a, b = highpass(burst[i]), highpass(clean[i])
        a, b = a - a.mean(), b - b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        corrs.append(float((a * b).sum() / denom) if denom > 0 else np.nan)
        rms.append(float(np.sqrt(np.mean((burst[i] - clean[i]) ** 2))))
    f_lrnyq = 1.0 / (2 * scale)
    sig_hr = float(m["psf_sigma_lr_px"]) * scale
    e_above_lr = radial_power_fraction(target, f_lrnyq)
    e_unrec = radial_power_fraction(target, mtf_cutoff(sig_hr))
    # sanity
    problems = []
    if obs.shape != (5, *burst.shape[1:]):
        problems.append(f"obs_features shape {obs.shape}")
    if shifts.shape != (len(burst), 2):
        problems.append(f"shifts {shifts.shape} vs burst {burst.shape}")
    if not (0.10 <= float(m["psf_sigma_lr_px"]) <= 0.60):
        problems.append(f"psf_sigma {m['psf_sigma_lr_px']}")
    if not np.isfinite(burst).all():
        problems.append("burst has non-finite")
    defect_info = check_defect_schema(s)
    problems.extend(defect_info.get("defect_problems", []))
    return {
        "scene": scene_dir.name, "n_frames": int(len(burst)), "difficulty": m.get("difficulty"),
        "psf_shape": m.get("psf_shape"), "psf_sigma": round(float(m["psf_sigma_lr_px"]), 3),
        "hp_corr_med": round(float(np.nanmedian(corrs)), 4),
        "hp_corr_min": round(float(np.nanmin(corrs)), 4),
        "resid_rms": round(float(np.median(rms)), 4),
        "noise_sigma_c": round(float(m.get("noise_sigma_c", np.nan)), 4),
        "E_above_LRnyq": round(e_above_lr, 4), "E_unrecoverable": round(e_unrec, 4),
        "has_annotations": defect_info.get("has_annotations", False),
        "n_dots": defect_info.get("n_dots"),
        "problems": problems,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--out", default="")
    args = ap.parse_args()
    pool = Path(args.pool)
    scenes = sorted(d for d in pool.glob("scene_*") if d.is_dir()) or sorted(
        d.parent for d in pool.glob("*/metadata.json"))
    if not scenes:
        print(f"no scenes under {pool}"); return 2
    rng = np.random.default_rng(0)
    pick = [scenes[i] for i in rng.choice(len(scenes), size=min(args.k, len(scenes)), replace=False)]
    print(f"auditing {len(pick)} / {len(scenes)} scenes in {pool}\n")
    rows = []
    hdr = f"{'scene':>16}{'diff':>8}{'psf':>10}{'σ':>6}{'hpCorrMed':>10}{'hpCorrMin':>10}{'residRMS':>9}{'E>LRnq':>8}{'Eunrec':>8}"
    print(hdr)
    for d in pick:
        try:
            r = audit_scene(d)
        except Exception as e:  # noqa: BLE001
            print(f"{d.name:>16}  ERROR: {e}"); rows.append({"scene": d.name, "error": str(e)}); continue
        rows.append(r)
        flag = "  !!" + ",".join(r["problems"]) if r["problems"] else ""
        print(f"{r['scene']:>16}{str(r['difficulty']):>8}{str(r['psf_shape'])[:9]:>10}{r['psf_sigma']:>6}"
              f"{r['hp_corr_med']:>10}{r['hp_corr_min']:>10}{r['resid_rms']:>9}"
              f"{r['E_above_LRnyq']:>8}{r['E_unrecoverable']:>8}{flag}")
    ok_rows = [r for r in rows if "hp_corr_med" in r]
    corr_med = float(np.median([r["hp_corr_med"] for r in ok_rows])) if ok_rows else 0.0
    n_problem = sum(1 for r in rows if r.get("problems") or r.get("error"))
    e_unrec_med = float(np.median([r["E_unrecoverable"] for r in ok_rows])) if ok_rows else 0.0
    # v7 defect-schema aggregate: zero-dot-scene ratio MUST be 0 (the G6 silent-floor fix).
    annotated = [r for r in ok_rows if r.get("has_annotations")]
    zero_dot = [r for r in annotated if (r.get("n_dots") or 0) == 0]
    zero_dot_ratio = (len(zero_dot) / len(annotated)) if annotated else 0.0
    print(f"\nmedian hp-corr(clean A(target) vs saved burst) = {corr_med:.4f}   (want > 0.90: roundtrip sound)")
    print(f"median GT unrecoverable-band energy            = {e_unrec_med:.4f}   (want < ~0.05: data honest)")
    print(f"scenes with structural problems/errors         = {n_problem} / {len(rows)}")
    if annotated:
        print(f"v7 defect annotations                          = {len(annotated)} / {len(ok_rows)} scenes")
        print(f"zero-dot-scene ratio (want == 0)               = {zero_dot_ratio:.4f}  "
              f"({len(zero_dot)} zero-dot scenes)")
    verdict = corr_med > 0.90 and n_problem == 0 and zero_dot_ratio == 0.0
    print("\nPOOL AUDIT:", "PASS" if verdict else "INVESTIGATE")
    if args.out:
        Path(args.out).write_text(json.dumps({"median_hp_corr": corr_med, "rows": rows}, indent=2))
        print(f"wrote {args.out}")
    return 0 if verdict else 1


if __name__ == "__main__":
    raise SystemExit(main())
