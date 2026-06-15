#!/usr/bin/env python3
"""Generate F5b: held-out upper-right structure ROI for the F5 visual gate.

ROI2 is predeclared by fixed fractional bounds on the 2x HR grid:
rows [0.270, 0.415), cols [0.530, 0.685).  This differs from the F5 center
comb ROI (rows [384/960, 518/960), cols [478/1280, 674/1280)) and covers the
upper-right peripheral plate-edge / zigzag-branch structure.  The window is a
fixed geometry check, not selected by proxy values or tuned after viewing
method outputs.

Inputs:
    output/ep11_unified_harness/hr/{drizzle,tgv,v9a_late_60k,
    v10_lam120_15k}_{temperature,highpass}.npy

Outputs:
    output/paper_figures/fig05b_main_visual_roi2.{png,pdf}
    output/ep11_unified_harness/roi2_structure_proxies.csv
    output/ep11_unified_harness/roi2_manifest.json

The figure and metrics are visual/proxy evidence only: they do not establish
resolution, temperature metrology, or GT fidelity.

Run from the repository root:
    uv run python scripts/paper_figures/fig05b_roi2_holdout.py
"""

from __future__ import annotations

import json
import math
import os
import time
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLBACKEND", "Agg")
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage
from scipy.signal import find_peaks

from thermal_core.plotting import COLORMAPS, savefig_academic, setup_academic_style

PROJECT_ROOT = Path(__file__).resolve().parents[2]
HARNESS_DIR = PROJECT_ROOT / "output" / "ep11_unified_harness"
HR_DIR = HARNESS_DIR / "hr"
PAPER_FIGURE_DIR = PROJECT_ROOT / "output" / "paper_figures"

PIXEL_SIZE_UM = 10.0
SCALE_2X = 2

CENTER_ROI_FRAC = {
    "row0": 384.0 / 960.0,
    "row1": 518.0 / 960.0,
    "col0": 478.0 / 1280.0,
    "col1": 674.0 / 1280.0,
}
ROI2_FRAC = {"row0": 0.270, "row1": 0.415, "col0": 0.530, "col1": 0.685}
ROI2_DESCRIPTION = "fixed upper-right peripheral plate-edge / zigzag-branch structure ROI"
CLAIM_BOUNDARY = "visual/proxy evidence only; not resolution, metrology, or GT fidelity evidence"

ARMS = [
    ("drizzle", "Drizzle"),
    ("tgv", "TGV"),
    ("v9a_late_60k", "Hybrid 60K"),
    ("v10_lam120_15k", "Hybrid+ResObs\nlambda=1.2 15K"),
]

RANK_SPECS = {
    "lattice_score": True,
    "sharp_p95": False,
    "zigzag_fwhm_median_um": True,
    "zigzag_dip_depth_median": False,
    "zigzag_profiles_separated": False,
}


def rel(path: Path | str) -> str:
    p = Path(path)
    try:
        return str(p.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(p)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return rel(value)
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def fill_nan(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    finite = np.isfinite(arr)
    if finite.all():
        return arr
    fill = float(np.nanmedian(arr[finite])) if bool(finite.any()) else 0.0
    return np.where(finite, arr, fill).astype(np.float32, copy=False)


def bounds_from_frac(shape: tuple[int, int], frac: dict[str, float]) -> tuple[int, int, int, int]:
    rows, cols = shape
    y0 = int(round(rows * frac["row0"]))
    y1 = int(round(rows * frac["row1"]))
    x0 = int(round(cols * frac["col0"]))
    x1 = int(round(cols * frac["col1"]))
    if not (0 <= y0 < y1 <= rows and 0 <= x0 < x1 <= cols):
        raise ValueError(f"Invalid ROI bounds {(y0, y1, x0, x1)} for shape {shape}")
    return y0, y1, x0, x1


def crop_roi(image: np.ndarray, frac: dict[str, float]) -> np.ndarray:
    y0, y1, x0, x1 = bounds_from_frac(np.asarray(image).shape, frac)
    return np.asarray(image)[y0:y1, x0:x1]


def load_arrays() -> dict[str, dict[str, np.ndarray | str]]:
    out: dict[str, dict[str, np.ndarray | str]] = {}
    for arm_id, label in ARMS:
        temp_path = HR_DIR / f"{arm_id}_temperature.npy"
        hp_path = HR_DIR / f"{arm_id}_highpass.npy"
        if not temp_path.exists() or not hp_path.exists():
            raise FileNotFoundError(f"Missing cached HR pair for {arm_id}: {temp_path}, {hp_path}")
        temp = np.load(temp_path).astype(np.float32, copy=False)
        hp = np.load(hp_path).astype(np.float32, copy=False)
        if temp.shape != hp.shape:
            raise ValueError(f"Shape mismatch for {arm_id}: temperature {temp.shape}, highpass {hp.shape}")
        out[arm_id] = {
            "label": label,
            "temperature": temp,
            "highpass": hp,
            "temperature_path": temp_path,
            "highpass_path": hp_path,
        }
    return out


def lattice_score(crop_hp: np.ndarray) -> float:
    x = np.asarray(crop_hp, dtype=np.float64)
    if x.size == 0 or not np.isfinite(x).any():
        return float("nan")
    x = fill_nan(x).astype(np.float64, copy=False)
    x -= float(np.nanmean(x))
    power = np.abs(np.fft.fftshift(np.fft.fft2(x))) ** 2
    rows, cols = x.shape
    fy = np.fft.fftshift(np.fft.fftfreq(rows))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(cols))[None, :]
    band = (np.abs(fy) > 0.35) | (np.abs(fx) > 0.35)
    return float(power[band].sum() / max(float(power.sum()), 1e-12))


def sharp_p95(crop_temp: np.ndarray) -> float:
    crop = np.asarray(crop_temp, dtype=np.float64)
    if crop.size == 0 or not np.isfinite(crop).any():
        return float("nan")
    gy, gx = np.gradient(fill_nan(crop).astype(np.float64, copy=False))
    return float(np.nanpercentile(np.hypot(gy, gx), 95.0))


def sample_line(image: np.ndarray, y0: float, x0: float, y1: float, x1: float) -> np.ndarray:
    length = float(np.hypot(y1 - y0, x1 - x0))
    n = int(max(16, round(length) + 1))
    ys = np.linspace(y0, y1, n)
    xs = np.linspace(x0, x1, n)
    return ndimage.map_coordinates(fill_nan(image), [ys, xs], order=1, mode="nearest")


def line_signal_for_dark_trace(profile: np.ndarray) -> np.ndarray:
    baseline = float(np.percentile(profile, 90.0))
    return (baseline - np.asarray(profile, dtype=np.float32)).astype(np.float32, copy=False)


def profile_metrics(signal: np.ndarray, *, pitch_um: float, min_spacing_um: float = 8.0) -> dict[str, float | bool | int]:
    sig = ndimage.gaussian_filter1d(np.asarray(signal, dtype=np.float32), sigma=1.0, mode="nearest")
    base = float(np.percentile(sig, 10.0))
    height = float(np.max(sig) - base)
    if height <= 1e-8:
        return {"fwhm_um": float("nan"), "dip_depth": float("nan"), "lines_separated": False, "n_peaks": 0}
    peaks, _props = find_peaks(sig, prominence=0.10 * height, distance=max(2, int(round(min_spacing_um / pitch_um))))
    if peaks.size == 0:
        peaks = np.asarray([int(np.argmax(sig))], dtype=int)
    primary = int(peaks[np.argmax(sig[peaks])])
    half = base + 0.5 * (float(sig[primary]) - base)
    left = primary
    while left > 0 and sig[left] > half:
        left -= 1
    right = primary
    while right < sig.size - 1 and sig[right] > half:
        right += 1
    fwhm_um = float(max(0, right - left) * pitch_um)
    if peaks.size >= 2:
        ranked = peaks[np.argsort(sig[peaks])[-2:]]
        p0, p1 = sorted(int(p) for p in ranked)
        valley = float(np.min(sig[p0 : p1 + 1]))
        peak_ref = min(float(sig[p0]), float(sig[p1]))
        dip_ratio = (valley - base) / max(peak_ref - base, 1e-8)
        dip_depth = float(1.0 - np.clip(dip_ratio, 0.0, 1.0))
        lines_separated = bool(dip_depth >= 0.25 and (p1 - p0) * pitch_um >= min_spacing_um)
    else:
        dip_depth = float("nan")
        lines_separated = False
    return {"fwhm_um": fwhm_um, "dip_depth": dip_depth, "lines_separated": lines_separated, "n_peaks": int(peaks.size)}


def roi_profile_specs(image_shape: tuple[int, int]) -> list[dict[str, float | str]]:
    rows, cols = image_shape
    return [
        {"profile_id": "roi_upper_left", "y0": 0.24 * rows, "x0": 0.02 * cols, "y1": 0.24 * rows, "x1": 0.44 * cols},
        {"profile_id": "roi_mid_left", "y0": 0.36 * rows, "x0": 0.03 * cols, "y1": 0.36 * rows, "x1": 0.46 * cols},
        {"profile_id": "roi_lower_left", "y0": 0.49 * rows, "x0": 0.04 * cols, "y1": 0.49 * rows, "x1": 0.48 * cols},
    ]


def summarize_zigzag(crop_hp: np.ndarray, *, output_grid_scale: int = SCALE_2X) -> dict[str, Any]:
    pitch_um = PIXEL_SIZE_UM / float(output_grid_scale)
    rows: list[dict[str, Any]] = []
    for spec in roi_profile_specs(np.asarray(crop_hp).shape):
        profile = sample_line(
            crop_hp,
            float(spec["y0"]),
            float(spec["x0"]),
            float(spec["y1"]),
            float(spec["x1"]),
        )
        signal = line_signal_for_dark_trace(profile)
        rows.append({"profile_id": spec["profile_id"], **profile_metrics(signal, pitch_um=pitch_um)})
    table = pd.DataFrame(rows)
    return {
        "zigzag_fwhm_median_um": float(np.nanmedian(table["fwhm_um"].to_numpy(dtype=float))),
        "zigzag_dip_depth_median": float(np.nanmedian(table["dip_depth"].to_numpy(dtype=float))),
        "zigzag_profiles_separated": int(table["lines_separated"].astype(bool).sum()),
    }


def compute_roi2_rows(arrays: dict[str, dict[str, np.ndarray | str]]) -> pd.DataFrame:
    center_metrics = pd.read_csv(HARNESS_DIR / "all_arm_metrics.csv").set_index("arm_id")
    records: list[dict[str, Any]] = []
    for arm_id, label in ARMS:
        item = arrays[arm_id]
        temp = item["temperature"]
        hp = item["highpass"]
        assert isinstance(temp, np.ndarray)
        assert isinstance(hp, np.ndarray)
        y0, y1, x0, x1 = bounds_from_frac(temp.shape, ROI2_FRAC)
        temp_crop = temp[y0:y1, x0:x1]
        hp_crop = hp[y0:y1, x0:x1]
        row: dict[str, Any] = {
            "arm_id": arm_id,
            "display_name": label,
            "roi2_row0": y0,
            "roi2_row1": y1,
            "roi2_col0": x0,
            "roi2_col1": x1,
            "roi2_lattice_score": lattice_score(hp_crop),
            "roi2_sharp_p95": sharp_p95(temp_crop),
            "source_temperature_npy": rel(Path(str(item["temperature_path"]))),
            "source_highpass_npy": rel(Path(str(item["highpass_path"]))),
        }
        row.update({f"roi2_{key}": value for key, value in summarize_zigzag(hp_crop).items()})
        if arm_id in center_metrics.index:
            center = center_metrics.loc[arm_id]
            for metric in RANK_SPECS:
                row[f"center_{metric}"] = float(center.get(metric, np.nan))
        records.append(row)

    df = pd.DataFrame(records)
    for metric, ascending in RANK_SPECS.items():
        roi_col = f"roi2_{metric}"
        center_col = f"center_{metric}"
        df[f"roi2_rank_{metric}"] = df[roi_col].rank(ascending=ascending, method="min")
        if center_col in df.columns:
            df[f"center_rank_{metric}"] = df[center_col].rank(ascending=ascending, method="min")
            df[f"rank_delta_{metric}"] = df[f"roi2_rank_{metric}"] - df[f"center_rank_{metric}"]
    return df


def ranking_summary(df: pd.DataFrame) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for metric in RANK_SPECS:
        roi_rank_col = f"roi2_rank_{metric}"
        center_rank_col = f"center_rank_{metric}"
        roi_order = df.sort_values([roi_rank_col, "arm_id"])["arm_id"].tolist()
        center_order = df.sort_values([center_rank_col, "arm_id"])["arm_id"].tolist()
        out[metric] = {
            "roi2_order": roi_order,
            "center_order": center_order,
            "same_order": roi_order == center_order,
        }
    return out


def draw_f5b(arrays: dict[str, dict[str, np.ndarray | str]], output_path: Path) -> Path:
    labels = [label for _arm_id, label in ARMS]
    temp_crops = []
    hp_crops = []
    for arm_id, _label in ARMS:
        temp = arrays[arm_id]["temperature"]
        hp = arrays[arm_id]["highpass"]
        assert isinstance(temp, np.ndarray)
        assert isinstance(hp, np.ndarray)
        temp_crops.append(crop_roi(temp, ROI2_FRAC))
        hp_crops.append(crop_roi(hp, ROI2_FRAC))

    temp_values = np.concatenate([crop[np.isfinite(crop)].ravel() for crop in temp_crops if np.isfinite(crop).any()])
    hp_values = np.concatenate([crop[np.isfinite(crop)].ravel() for crop in hp_crops if np.isfinite(crop).any()])
    temp_vmin, temp_vmax = float(np.percentile(temp_values, 1.0)), float(np.percentile(temp_values, 99.0))
    hp_vmax = max(float(np.percentile(np.abs(hp_values), 99.0)), 1e-6)

    fig, axes = plt.subplots(2, len(labels), figsize=(7.2, 3.05), constrained_layout=True)
    im0 = im1 = None
    for col, label in enumerate(labels):
        im0 = axes[0, col].imshow(
            fill_nan(temp_crops[col]),
            cmap=COLORMAPS["temperature"],
            vmin=temp_vmin,
            vmax=temp_vmax,
            interpolation="nearest",
        )
        im1 = axes[1, col].imshow(
            fill_nan(hp_crops[col]),
            cmap=COLORMAPS["residual_diff"],
            vmin=-hp_vmax,
            vmax=hp_vmax,
            interpolation="nearest",
        )
        axes[0, col].set_title(label)
        for row in (0, 1):
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    axes[0, 0].set_ylabel("Temp.")
    axes[1, 0].set_ylabel("Highpass")
    assert im0 is not None and im1 is not None
    fig.colorbar(im0, ax=axes[0, :].tolist(), fraction=0.025, pad=0.01).set_label("deg C")
    fig.colorbar(im1, ax=axes[1, :].tolist(), fraction=0.025, pad=0.01).set_label("deg C")
    return savefig_academic(fig, output_path)


def main() -> int:
    setup_academic_style()
    started = time.perf_counter()
    arrays = load_arrays()
    first = arrays[ARMS[0][0]]["temperature"]
    assert isinstance(first, np.ndarray)
    roi2_bounds = bounds_from_frac(first.shape, ROI2_FRAC)
    center_bounds = bounds_from_frac(first.shape, CENTER_ROI_FRAC)

    png_path = PAPER_FIGURE_DIR / "fig05b_main_visual_roi2.png"
    pdf_path = PAPER_FIGURE_DIR / "fig05b_main_visual_roi2.pdf"
    draw_f5b(arrays, png_path)
    draw_f5b(arrays, pdf_path)

    proxy_df = compute_roi2_rows(arrays)
    proxy_path = HARNESS_DIR / "roi2_structure_proxies.csv"
    proxy_path.parent.mkdir(parents=True, exist_ok=True)
    proxy_df.to_csv(proxy_path, index=False)

    summary = ranking_summary(proxy_df)
    manifest_path = HARNESS_DIR / "roi2_manifest.json"
    write_json(
        manifest_path,
        {
            "task": "Task E2 F5b second held-out ROI",
            "created_or_updated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "elapsed_sec": float(time.perf_counter() - started),
            "cpu_only": True,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            "roi2_description": ROI2_DESCRIPTION,
            "roi2_fraction_bounds": ROI2_FRAC,
            "roi2_pixel_bounds_y0_y1_x0_x1": roi2_bounds,
            "center_f5_fraction_bounds": CENTER_ROI_FRAC,
            "center_f5_pixel_bounds_y0_y1_x0_x1": center_bounds,
            "selection_rule": "Fixed fractional geometry; no proxy search and no method-output tuning.",
            "claim_boundary": CLAIM_BOUNDARY,
            "ranking_summary": summary,
            "outputs": {
                "fig05b_png": png_path,
                "fig05b_pdf": pdf_path,
                "roi2_structure_proxies": proxy_path,
            },
            "sources": {
                arm_id: {
                    "temperature_npy": rel(Path(str(arrays[arm_id]["temperature_path"]))),
                    "highpass_npy": rel(Path(str(arrays[arm_id]["highpass_path"]))),
                }
                for arm_id, _label in ARMS
            },
        },
    )

    print(f"[fig05b] wrote {rel(png_path)}")
    print(f"[fig05b] wrote {rel(pdf_path)}")
    print(f"[fig05b] wrote {rel(proxy_path)}")
    print(f"[fig05b] wrote {rel(manifest_path)}")
    for metric, payload in summary.items():
        print(
            f"[fig05b] {metric}: roi2={payload['roi2_order']} "
            f"center={payload['center_order']} same={payload['same_order']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
