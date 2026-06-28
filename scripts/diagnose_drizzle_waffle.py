#!/usr/bin/env python
"""Diagnose the phase-bin drizzle 'coverage waffle' (ACL-032, the background checkerboard).

The hybrid solver warm-starts x0 from the FIRST phase-bin drizzle channel (ch5). Phase-bin drizzle
routes each frame to one of n sub-pixel-phase bins and drizzles each bin separately onto the 2x HR
grid; on flat background with uneven phase coverage, the 2x2 HR sub-positions in each block are
filled from different frame subsets -> a 2-HR-px-period (= 1 detector pitch = 20um) intensity
modulation. That waffle is the faint background grid in the real-data solver output.

This script proves it WITHOUT any training or GPU: for sampled pool scenes it FFT-compares the
two candidate warm-starts that live in the saved obs --
    * ch0  = fused aligned_mean (smooth)          -> the de-waffled candidate (--solver-warmstart aligned_mean)
    * ch5  = phase_bin_drizzle[0] (current x0)     -> carries the waffle
and the GT, reporting out_of_band energy plus a 'grid' score (power piled at the HR-Nyquist
corners/axes where a 2px checkerboard concentrates). Expectation:
    grid score / out_of_band:   ch5 (phasebin)  >>  ch0 (aligned_mean)  ~  GT
which means seeding x0 from ch0 removes the waffle at the source while keeping all 9 cond channels.

numpy/scipy only (runs in the data-gen env):

    uv run python scripts/diagnose_drizzle_waffle.py --pool data/synthetic/pool_2x_v5_5k --k 48
    uv run python scripts/diagnose_drizzle_waffle.py --pool ... --out output/waffle_diag   # + PNGs
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.ndimage import zoom as ndi_zoom

# tcforge is importable in the data-gen env (same env that generated the pool).
from tcforge.storage import load_scene_compact

SCALE = 2
ALIGNED_MEAN_CH = 0   # obs_features ch0 = fused aligned_mean (see tcforge.fusion.fuse_burst_to_features)


def _window(h: int, w: int) -> np.ndarray:
    return np.hanning(h)[:, None] * np.hanning(w)[None, :]


def _power(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hann-windowed 2D power spectrum (fftshifted) and the matching radial-frequency grid."""
    arr = np.nan_to_num(np.asarray(img, np.float64))
    arr = arr - arr.mean()
    h, w = arr.shape
    power = np.abs(np.fft.fftshift(np.fft.fft2(arr * _window(h, w)))) ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    radial = np.sqrt(fy ** 2 + fx ** 2)
    return power, radial


def out_of_band_ratio(img: np.ndarray, scale: int = SCALE) -> float:
    power, radial = _power(img)
    total = float(power.sum())
    return float(power[radial > 0.5 / scale].sum() / total) if total > 0 else 0.0


def grid_score(img: np.ndarray) -> float:
    """Fraction of spectral power concentrated at the HR-Nyquist shell (|f|>=0.45 on an axis).

    A 2-HR-px grid/checkerboard piles energy at the corners (~0.5,~0.5) and axes (~0.5,0)/(0,~0.5)
    of the spectrum; a smooth field has ~none there. This isolates the waffle better than the broad
    out_of_band ratio."""
    power, _ = _power(img)
    h, w = power.shape
    fy = np.abs(np.fft.fftshift(np.fft.fftfreq(h)))[:, None]
    fx = np.abs(np.fft.fftshift(np.fft.fftfreq(w)))[None, :]
    near_nyq = (np.maximum(fy, fx) >= 0.45)   # within ~10% of Nyquist on either axis
    total = float(power.sum())
    return float(power[near_nyq].sum() / total) if total > 0 else 0.0


def _upsample(img2d: np.ndarray, scale: int) -> np.ndarray:
    return ndi_zoom(np.asarray(img2d, np.float32), (scale, scale), order=1).astype(np.float32)


def _flattest_window(gt: np.ndarray, size: int = 96) -> tuple[int, int]:
    """Top-left corner of the lowest-gradient size x size window in the GT (a flat background ROI)."""
    gy, gx = np.gradient(np.asarray(gt, np.float64))
    mag = np.sqrt(gx * gx + gy * gy)
    h, w = mag.shape
    size = min(size, h, w)
    # coarse search on a stride so it stays cheap
    best, best_v = (0, 0), np.inf
    stride = max(1, size // 3)
    for y in range(0, h - size + 1, stride):
        for x in range(0, w - size + 1, stride):
            v = float(mag[y:y + size, x:x + size].mean())
            if v < best_v:
                best_v, best = v, (y, x)
    return best


def scan(pool: Path, k: int, seed: int) -> dict:
    scenes = sorted(pool.glob("scene_*"))
    if not scenes:
        raise SystemExit(f"FAIL: no scene_* dirs under {pool}")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(scenes), size=min(k, len(scenes)), replace=False)
    rows = {"aligned_mean(ch0)": ([], []), "phasebin0(ch5=x0)": ([], []),
            "phasebin_mean": ([], []), "GT": ([], [])}
    crops = []
    for i in idx:
        sd = scenes[int(i)]
        scene = load_scene_compact(sd)
        obs = np.asarray(scene["obs_features"], np.float32)          # (5, H1, W1)
        pbd = scene.get("phase_bin_drizzle")
        if pbd is None:
            continue
        pbd = np.asarray(pbd, np.float32)                            # (n_bins, H2, W2)
        gt = np.asarray(scene["hr_temperature"], np.float32)         # (H2, W2)

        aligned2x = _upsample(obs[ALIGNED_MEAN_CH], SCALE)
        fields = {
            "aligned_mean(ch0)": aligned2x,
            "phasebin0(ch5=x0)": pbd[0],
            "phasebin_mean": pbd.mean(axis=0),
            "GT": gt,
        }
        for name, f in fields.items():
            rows[name][0].append(out_of_band_ratio(f))
            rows[name][1].append(grid_score(f))
        if len(crops) < 4:
            y, x = _flattest_window(gt)
            crops.append((sd.name, y, x, {n: f for n, f in fields.items()}))
    return {"n": len(rows["GT"][0]), "rows": rows, "crops": crops, "n_total": len(scenes)}


def dump_crops(crops, out: Path) -> None:
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    order = ["aligned_mean(ch0)", "phasebin0(ch5=x0)", "GT"]
    sz = 96
    for name, y, x, fields in crops:
        fig, axes = plt.subplots(1, len(order), figsize=(4 * len(order), 4))
        for ax, key in zip(axes, order):
            crop = np.asarray(fields[key])[y:y + sz, x:x + sz]
            lo, hi = np.percentile(crop, [2, 98])
            im = ax.imshow(crop, cmap="inferno", vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_title(f"{key}\noob={out_of_band_ratio(fields[key]):.4f} grid={grid_score(fields[key]):.4f}",
                         fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
            fig.colorbar(im, ax=ax, fraction=0.046)
        fig.suptitle(f"{name} — flat-background ROI ({sz}x{sz})  [waffle should appear only in ch5]")
        fig.tight_layout()
        fig.savefig(out / f"waffle_{name}.png", dpi=140, bbox_inches="tight")
        plt.close(fig)
    print(f"  wrote {len(crops)} flat-ROI montages -> {out}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True)
    ap.add_argument("--k", type=int, default=48)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="", help="optional dir for flat-ROI montage PNGs")
    args = ap.parse_args()

    r = scan(Path(args.pool), args.k, args.seed)
    print(f"\n=== drizzle waffle diagnostic: {args.pool} ({r['n']}/{r['n_total']} scenes) ===")
    print(f"  {'field':22s} {'out_of_band (median)':>22s} {'grid_score (median)':>22s}")
    med = {}
    for name, (oob, grid) in r["rows"].items():
        oob_m, grid_m = float(np.median(oob)), float(np.median(grid))
        med[name] = (oob_m, grid_m)
        print(f"  {name:22s} {oob_m:>22.5f} {grid_m:>22.5f}")

    ch5 = med["phasebin0(ch5=x0)"]
    ch0 = med["aligned_mean(ch0)"]
    gt = med["GT"]
    grid_ratio = ch5[1] / max(ch0[1], 1e-12)
    print(f"\n  phasebin/aligned grid_score ratio: {grid_ratio:.1f}x"
          f"   (ch5 x0 carries {grid_ratio:.1f}x the Nyquist-grid energy of the smooth ch0)")
    verdict = ("CONFIRMED: the waffle lives in the ch5 phase-bin warm-start and is ~absent from ch0 "
               "-> --solver-warmstart aligned_mean removes it at the source."
               if grid_ratio >= 3.0 and ch0[1] <= max(gt[1] * 2.5, 0.01)
               else "INCONCLUSIVE: ch0 not clearly cleaner than ch5 — inspect the montages (--out) "
                    "and consider --solver-no-drizzle as the cleaner control.")
    print(f"  {verdict}")

    if args.out and r["crops"]:
        dump_crops(r["crops"], Path(args.out))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
