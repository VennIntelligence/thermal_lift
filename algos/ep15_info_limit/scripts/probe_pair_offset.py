#!/usr/bin/env python3
"""Probe a reconstruction pair for a global sub-pixel grid offset.

Stage 0g found same-half cross-method FRC curves oscillating for V11 vs
classical reconstructions (ACL-047/048) — the signature of a rigid grid offset
modulating ring correlations, as opposed to the monotone decay of genuine
content divergence.  This tool settles it per pair:

1. estimate the global (dx, dy) offset of image_b relative to image_a via
   windowed phase correlation with parabolic sub-pixel refinement;
2. apply the inverse shift to image_b in the Fourier domain;
3. recompute the FRC curve before/after and report cutoffs and band samples.

If the post-correction FRC recovers to classical anchor levels and the
oscillation disappears, the pair difference was a registration artifact; if it
stays low and monotone, the content genuinely differs (prior replacement).
The estimated offset is in HR grid pixels (multiply by hr pitch for um).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.signal.windows import tukey

SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[3]
for _path in (SCRIPT_PATH.parent, PROJECT_ROOT / "core" / "src", PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src"):
    _text = str(_path)
    if _text not in sys.path:
        sys.path.insert(0, _text)

from run_real_split_frc_v2 import (  # noqa: E402
    DEFAULT_SCALE,
    fill_nan,
    frc_curve_v2,
    interpolate_frc,
    rel,
    write_json,
)
from run_m2_frc import find_cutoff  # noqa: E402


def _windowed(image: np.ndarray, tukey_alpha: float) -> np.ndarray:
    arr = fill_nan(np.asarray(image, dtype=np.float64))
    arr = arr - float(arr.mean())
    win_y = tukey(arr.shape[0], alpha=float(tukey_alpha))
    win_x = tukey(arr.shape[1], alpha=float(tukey_alpha))
    return arr * win_y[:, None] * win_x[None, :]


def estimate_global_offset(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    tukey_alpha: float = 0.25,
    max_offset_px: float = 4.0,
    eps: float = 1e-12,
) -> dict[str, float]:
    """Estimate the rigid (dx, dy) of ``image_b`` relative to ``image_a``.

    Windowed phase correlation (normalized cross-power spectrum) with a
    3-point parabolic sub-pixel peak refinement per axis.  Positive dx means
    image_b's content sits at larger x than image_a's.  ``max_offset_px``
    guards against locking onto a far spurious peak: the integer peak is
    searched only inside that radius around zero lag.
    """

    a = _windowed(image_a, tukey_alpha)
    b = _windowed(image_b, tukey_alpha)
    if a.shape != b.shape:
        raise ValueError(f"pair shapes differ: {a.shape} vs {b.shape}")
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    cross = fa * np.conj(fb)
    cross /= np.abs(cross) + eps
    corr = np.real(np.fft.ifft2(cross))

    h, w = corr.shape
    rad = int(np.ceil(float(max_offset_px)))
    yy = np.fft.fftfreq(h, d=1.0 / h)  # signed lag values per row index
    xx = np.fft.fftfreq(w, d=1.0 / w)
    mask = (np.abs(yy)[:, None] <= rad) & (np.abs(xx)[None, :] <= rad)
    masked = np.where(mask, corr, -np.inf)
    peak_flat = int(np.argmax(masked))
    py, px = np.unravel_index(peak_flat, corr.shape)

    def _parabolic(center: int, axis_len: int, values: np.ndarray) -> float:
        prev = values[(center - 1) % axis_len]
        cur = values[center]
        nxt = values[(center + 1) % axis_len]
        denom = prev - 2.0 * cur + nxt
        if abs(denom) < 1e-15:
            return 0.0
        frac = 0.5 * (prev - nxt) / denom
        return float(np.clip(frac, -0.5, 0.5))

    frac_y = _parabolic(py, h, corr[:, px])
    frac_x = _parabolic(px, w, corr[py, :])
    # If b(x) = a(x - d) (b displaced by +d), FA*conj(FB) has phase +2*pi*f*d,
    # whose inverse FFT peaks at lag -d — so negate the peak lag to report the
    # displacement of b relative to a.
    dy = -(float(yy[py]) + frac_y)
    dx = -(float(xx[px]) + frac_x)
    peak_value = float(corr[py, px])
    return {"dx_px": dx, "dy_px": dy, "peak_corr": peak_value}


def fourier_shift(image: np.ndarray, *, dx_px: float, dy_px: float) -> np.ndarray:
    """Translate an image by (dx, dy) pixels with Fourier phase ramps."""

    arr = fill_nan(np.asarray(image, dtype=np.float64))
    fy = np.fft.fftfreq(arr.shape[0])[:, None]
    fx = np.fft.fftfreq(arr.shape[1])[None, :]
    ramp = np.exp(-2j * np.pi * (fy * float(dy_px) + fx * float(dx_px)))
    shifted = np.fft.ifft2(np.fft.fft2(arr) * ramp)
    return np.real(shifted).astype(np.float32, copy=False)


def _band_samples(curve: pd.DataFrame, periods: tuple[float, ...]) -> dict[str, float]:
    return {f"frc_at_{p:g}um": interpolate_frc(curve, float(p)) for p in periods}


def _sign_changes(curve: pd.DataFrame, *, period_lo_um: float, period_hi_um: float) -> int:
    sel = curve[(curve["period_um"] >= period_lo_um) & (curve["period_um"] <= period_hi_um)]
    values = sel["frc"].to_numpy(dtype=float)
    values = values[np.isfinite(values)]
    if values.size < 2:
        return 0
    return int(np.sum(np.diff(np.sign(values)) != 0))


def parse_pair(spec: str) -> tuple[str, Path, Path, int]:
    parts = str(spec).split(":")
    if len(parts) not in (3, 4):
        raise ValueError(f"--pair must be name:path_a.npy:path_b.npy[:scale]; got {spec!r}")
    scale = int(parts[3]) if len(parts) == 4 and parts[3] else DEFAULT_SCALE
    return parts[0], Path(parts[1]), Path(parts[2]), scale


def probe_pair(
    name: str,
    path_a: Path,
    path_b: Path,
    scale: int,
    *,
    args: argparse.Namespace,
    pixel_size_um: float,
) -> tuple[dict[str, Any], list[pd.DataFrame]]:
    image_a = np.load(path_a).astype(np.float32, copy=False)
    image_b = np.load(path_b).astype(np.float32, copy=False)
    # Two estimate->correct passes: the parabolic peak refinement carries a
    # small (~0.03 px) bias at large offsets, so re-estimating on the corrected
    # image and accumulating removes most of it.
    total_dx, total_dy = 0.0, 0.0
    corrected_b = image_b
    peak_corr = float("nan")
    for _ in range(2):
        offset = estimate_global_offset(
            image_a,
            corrected_b,
            tukey_alpha=float(args.tukey_alpha),
            max_offset_px=float(args.max_offset_px),
        )
        total_dx += offset["dx_px"]
        total_dy += offset["dy_px"]
        peak_corr = offset["peak_corr"]
        corrected_b = fourier_shift(image_b, dx_px=-total_dx, dy_px=-total_dy)
    offset = {"dx_px": total_dx, "dy_px": total_dy, "peak_corr": peak_corr}

    def _curve(b_img: np.ndarray) -> pd.DataFrame:
        return frc_curve_v2(
            image_a,
            b_img,
            scale=scale,
            pixel_size_um=pixel_size_um,
            crop_lr_px=int(args.crop_lr_px),
            tukey_alpha=float(args.tukey_alpha),
        )

    before = _curve(image_b)
    after = _curve(corrected_b)
    curves: list[pd.DataFrame] = []
    curve_dir = args.output_dir / "frc_curves"
    curve_dir.mkdir(parents=True, exist_ok=True)
    for tag, curve in (("before", before), ("after", after)):
        labeled = curve.copy()
        labeled.insert(0, "stage", tag)
        labeled.insert(0, "method", name)
        labeled.to_csv(curve_dir / f"{name}_offset_{tag}_frc_curve.csv", index=False)
        curves.append(labeled)

    hr_pitch_um = pixel_size_um / float(scale)
    periods = tuple(float(p) for p in args.periods_um)
    row: dict[str, Any] = {
        "pair": name,
        "status": "success",
        "path_a": rel(path_a),
        "path_b": rel(path_b),
        "scale": int(scale),
        "hr_pitch_um": hr_pitch_um,
        "offset_dx_hr_px": offset["dx_px"],
        "offset_dy_hr_px": offset["dy_px"],
        "offset_norm_hr_px": float(np.hypot(offset["dx_px"], offset["dy_px"])),
        "offset_dx_um": offset["dx_px"] * hr_pitch_um,
        "offset_dy_um": offset["dy_px"] * hr_pitch_um,
        "phase_corr_peak": offset["peak_corr"],
        "cutoff_1_7_before_um": float(find_cutoff(before, "threshold_1_7").period_um),
        "cutoff_1_7_after_um": float(find_cutoff(after, "threshold_1_7").period_um),
        "sign_changes_45_21um_before": _sign_changes(before, period_lo_um=21.0, period_hi_um=45.0),
        "sign_changes_45_21um_after": _sign_changes(after, period_lo_um=21.0, period_hi_um=45.0),
    }
    for key, value in _band_samples(before, periods).items():
        row[f"{key}_before"] = value
    for key, value in _band_samples(after, periods).items():
        row[f"{key}_after"] = value
    return row, curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        required=True,
        help="name:path_a.npy:path_b.npy[:scale] (repeatable). b is probed/shifted relative to a.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--pixel-size-um", type=float, default=20.0)
    parser.add_argument("--crop-lr-px", type=int, default=16)
    parser.add_argument("--tukey-alpha", type=float, default=0.25)
    parser.add_argument("--max-offset-px", type=float, default=4.0)
    parser.add_argument(
        "--periods-um",
        type=lambda text: tuple(float(v) for v in text.split(",") if v.strip()),
        default=(40.0, 33.0, 30.0, 27.0, 24.0),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    rows: list[dict[str, Any]] = []
    curves: list[pd.DataFrame] = []
    for spec in args.pair:
        name, raw_a, raw_b, scale = parse_pair(spec)
        path_a = raw_a if raw_a.is_absolute() else PROJECT_ROOT / raw_a
        path_b = raw_b if raw_b.is_absolute() else PROJECT_ROOT / raw_b
        try:
            row, pair_curves = probe_pair(name, path_a, path_b, scale, args=args, pixel_size_um=float(args.pixel_size_um))
            rows.append(row)
            curves.extend(pair_curves)
            print(
                f"[offset-probe] {name}: offset=({row['offset_dx_hr_px']:+.3f}, {row['offset_dy_hr_px']:+.3f}) HR px "
                f"(|{row['offset_norm_hr_px']:.3f}|), cutoff {row['cutoff_1_7_before_um']:.2f} -> "
                f"{row['cutoff_1_7_after_um']:.2f} um, sign changes {row['sign_changes_45_21um_before']} -> "
                f"{row['sign_changes_45_21um_after']}"
            )
        except Exception as exc:  # noqa: BLE001 - keep the audit row.
            rows.append({"pair": name, "status": "failed", "error": repr(exc)})
            print(f"[offset-probe] {name}: FAILED {exc!r}", file=sys.stderr)

    summary_path = args.output_dir / "offset_probe_summary.csv"
    pd.DataFrame(rows).to_csv(summary_path, index=False)
    if curves:
        pd.concat(curves, ignore_index=True).to_csv(args.output_dir / "offset_probe_curves_long.csv", index=False)
    write_json(
        args.output_dir / "offset_probe_manifest.json",
        {
            "task": "stage0h pair grid-offset probe",
            "pairs": list(args.pair),
            "pixel_size_um": float(args.pixel_size_um),
            "interpretation": (
                "offset is b relative to a in HR grid px; 'after' rows are FRC with b shifted back by the "
                "estimated offset. Recovery + oscillation collapse => registration artifact; low monotone "
                "after-curve => genuine content divergence."
            ),
            "elapsed_sec": float(time.perf_counter() - started),
            "failed": [r for r in rows if r.get("status") != "success"],
        },
    )
    print(f"[offset-probe] wrote {rel(summary_path)}")
    return 0 if all(r.get("status") == "success" for r in rows) else 2


if __name__ == "__main__":
    raise SystemExit(main())
