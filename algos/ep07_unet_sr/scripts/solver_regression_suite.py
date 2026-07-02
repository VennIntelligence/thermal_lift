"""Stage 0d solver regression suite for known EP07 failure cases.

The suite is intentionally usable without a checkpoint: it can score saved
render arrays, the local ignored V8/K4 diagnostic arrays when present, or a
single aligned-mean real-data baseline.  When a checkpoint is supplied it
renders both tiled and full-halo outputs through the existing real_eval path
and then applies the same array-level probes.

All physical audit constants below follow AGENTS.md as of 2026-07-02:
TXT detector pitch = 20 um/pixel and current spatial resolution = 20 um.
No threshold in this file is inherited from historical 10 um assumptions.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, laplace


DETECTOR_PITCH_UM = 20.0
SPATIAL_RESOLUTION_UM = 20.0
NOISE_FLOOR_C = 0.0724
SCHEMA_VERSION = 1


def _project_root() -> Path:
    p = Path(__file__).resolve()
    for q in [p, *p.parents]:
        if (q / "AGENTS.md").exists():
            return q
    return p.parents[3]


PROJECT_ROOT = _project_root()
ALGO_ROOT = PROJECT_ROOT / "algos" / "ep07_unet_sr"
SRC_ROOT = ALGO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


@dataclass
class CaseResult:
    name: str
    status: str
    metrics: dict[str, Any] = field(default_factory=dict)
    thresholds: dict[str, Any] = field(default_factory=dict)
    skip_reason: str = ""
    notes: list[str] = field(default_factory=list)


@dataclass
class SuiteThresholds:
    flat_lowpass_p95_delta_c: float = 0.08
    flat_lowpass_std_delta_c: float = 0.04
    extent_nrmse: float = 0.15
    extent_p95_abs_diff_c: float = 0.15
    seam_autocorr: float = 0.35
    bead_edge_ratio: float = 4.0
    bead_excess_p95_c: float = 0.04


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return "unknown"


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _finite_2d(name: str, image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
    if not np.isfinite(arr).any():
        raise ValueError(f"{name} has no finite pixels")
    return arr


def highpass(image: np.ndarray, sigma: float) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float64)
    return arr - gaussian_filter(arr, float(sigma), mode="nearest")


def _robust_std(arr: np.ndarray) -> float:
    vals = np.asarray(arr, dtype=np.float64)
    vals = vals[np.isfinite(vals)]
    if vals.size < 2:
        return 0.0
    return float(np.std(vals))


def locate_flattest_roi(base: np.ndarray, *, size: int = 96) -> tuple[int, int, int]:
    """Find a low-variance square away from the border."""

    arr = _finite_2d("base", base)
    h, w = arr.shape
    size = int(max(8, min(size, h, w)))
    sigma = max(2.0, size / 6.0)
    mean = gaussian_filter(arr, sigma, mode="nearest")
    mean2 = gaussian_filter(arr * arr, sigma, mode="nearest")
    var = np.clip(mean2 - mean * mean, 0.0, None)
    margin = max(1, size // 2)
    masked = var.copy()
    masked[:margin, :] = np.inf
    masked[-margin:, :] = np.inf
    masked[:, :margin] = np.inf
    masked[:, -margin:] = np.inf
    if not np.isfinite(masked).any():
        iy, ix = h // 2, w // 2
    else:
        iy, ix = np.unravel_index(int(np.nanargmin(masked)), masked.shape)
    y0 = int(np.clip(iy - size // 2, 0, h - size))
    x0 = int(np.clip(ix - size // 2, 0, w - size))
    return y0, x0, size


def _find_label(labels: list[str], exact: str | None = None, tokens: tuple[str, ...] = ()) -> str | None:
    if exact and exact in labels:
        return exact
    for label in labels:
        low = label.lower()
        if all(tok in low for tok in tokens):
            return label
    return None


def choose_reference_label(arrays: dict[str, np.ndarray]) -> str | None:
    labels = list(arrays)
    for candidate in ("aligned_mean", "baseline", "input_drizzle", "drizzle_mean", "raw_control"):
        found = _find_label(labels, exact=candidate, tokens=(candidate,))
        if found is not None:
            return found
    return labels[0] if labels else None


def flat_roi_probe(
    arrays: dict[str, np.ndarray],
    thresholds: SuiteThresholds,
    *,
    roi_size: int = 96,
    lowpass_sigma: float = 16.0,
) -> CaseResult:
    ref_label = choose_reference_label(arrays)
    if ref_label is None:
        return CaseResult("flat_roi_artifact", "skip", skip_reason="no arrays available")
    try:
        ref = _finite_2d(ref_label, arrays[ref_label])
    except ValueError as exc:
        return CaseResult("flat_roi_artifact", "skip", skip_reason=str(exc))

    y0, x0, size = locate_flattest_roi(ref, size=roi_size)
    labels = [k for k in arrays if k != ref_label] or [ref_label]
    metrics: dict[str, Any] = {
        "reference": ref_label,
        "roi": {"y": y0, "x": x0, "size_hr_px": size},
        "lowpass_sigma_hr_px": float(lowpass_sigma),
        "candidates": {},
    }
    failed: list[str] = []
    for label in labels:
        try:
            img = _finite_2d(label, arrays[label])
        except ValueError as exc:
            metrics["candidates"][label] = {"skip_reason": str(exc)}
            continue
        if img.shape != ref.shape:
            metrics["candidates"][label] = {
                "skip_reason": f"shape mismatch vs reference {ref.shape}: {img.shape}",
            }
            continue
        delta_lp = gaussian_filter(img - ref, lowpass_sigma, mode="nearest")
        roi = delta_lp[y0 : y0 + size, x0 : x0 + size]
        p95 = float(np.nanpercentile(np.abs(roi), 95.0))
        std = _robust_std(roi)
        rng = float(np.nanmax(roi) - np.nanmin(roi))
        ok = (
            p95 <= thresholds.flat_lowpass_p95_delta_c
            and std <= thresholds.flat_lowpass_std_delta_c
        )
        metrics["candidates"][label] = {
            "lowpass_p95_abs_delta_c": p95,
            "lowpass_std_delta_c": std,
            "lowpass_range_c": rng,
            "status": "pass" if ok else "fail",
        }
        if not ok:
            failed.append(label)

    evaluated = [
        v for v in metrics["candidates"].values()
        if isinstance(v, dict) and "status" in v
    ]
    if not evaluated:
        return CaseResult(
            "flat_roi_artifact",
            "skip",
            metrics=metrics,
            thresholds={
                "flat_lowpass_p95_delta_c": thresholds.flat_lowpass_p95_delta_c,
                "flat_lowpass_std_delta_c": thresholds.flat_lowpass_std_delta_c,
            },
            skip_reason="no shape-compatible candidate arrays",
        )
    notes = []
    if labels == [ref_label]:
        notes.append("only the reference baseline was available; artifact delta is self-vs-self")
    return CaseResult(
        "flat_roi_artifact",
        "fail" if failed else "pass",
        metrics=metrics,
        thresholds={
            "flat_lowpass_p95_delta_c": thresholds.flat_lowpass_p95_delta_c,
            "flat_lowpass_std_delta_c": thresholds.flat_lowpass_std_delta_c,
        },
        notes=notes,
    )


def extent_consistency_probe(
    arrays: dict[str, np.ndarray],
    thresholds: SuiteThresholds,
    *,
    crop_margin_hr: int = 0,
) -> CaseResult:
    labels = list(arrays)
    tiled_label = _find_label(labels, exact="tiled_p192_o128", tokens=("tiled",))
    full_label = _find_label(labels, exact="full_halo96", tokens=("full", "halo"))
    if tiled_label is None or full_label is None:
        return CaseResult(
            "tiled_full_halo_extent_consistency",
            "skip",
            skip_reason=(
                "requires both a tiled output and a full_halo output; "
                f"available labels={labels}"
            ),
        )
    try:
        tiled = _finite_2d(tiled_label, arrays[tiled_label])
        full = _finite_2d(full_label, arrays[full_label])
    except ValueError as exc:
        return CaseResult("tiled_full_halo_extent_consistency", "skip", skip_reason=str(exc))
    if tiled.shape != full.shape:
        return CaseResult(
            "tiled_full_halo_extent_consistency",
            "skip",
            skip_reason=f"shape mismatch: {tiled_label} {tiled.shape} vs {full_label} {full.shape}",
        )
    m = int(max(0, crop_margin_hr))
    if m > 0 and 2 * m < min(tiled.shape):
        tiled_eval = tiled[m:-m, m:-m]
        full_eval = full[m:-m, m:-m]
    else:
        tiled_eval = tiled
        full_eval = full
    diff = full_eval - tiled_eval
    rms = float(np.sqrt(np.nanmean(diff * diff)))
    denom = max(_robust_std(tiled_eval), 1e-9)
    nrmse = rms / denom
    p95 = float(np.nanpercentile(np.abs(diff), 95.0))
    mean = float(np.nanmean(diff))
    ok = nrmse <= thresholds.extent_nrmse and p95 <= thresholds.extent_p95_abs_diff_c
    return CaseResult(
        "tiled_full_halo_extent_consistency",
        "pass" if ok else "fail",
        metrics={
            "tiled_label": tiled_label,
            "full_halo_label": full_label,
            "crop_margin_hr_px": int(m),
            "rms_diff_c": rms,
            "nrmse_vs_tiled_std": nrmse,
            "p95_abs_diff_c": p95,
            "mean_diff_c": mean,
        },
        thresholds={
            "extent_nrmse": thresholds.extent_nrmse,
            "extent_p95_abs_diff_c": thresholds.extent_p95_abs_diff_c,
        },
    )


def _autocorr_at_lags(profile: np.ndarray, lags: list[int]) -> dict[int, float]:
    p = np.asarray(profile, dtype=np.float64)
    p = p[np.isfinite(p)]
    if p.size < 3:
        return {}
    p = p - float(np.mean(p))
    denom = float(np.dot(p, p))
    if denom <= 1e-18:
        return {lag: 0.0 for lag in lags if 0 < lag < p.size}
    out: dict[int, float] = {}
    for lag in lags:
        if 0 < int(lag) < p.size:
            out[int(lag)] = float(np.dot(p[:-lag], p[lag:]) / denom)
    return out


def seam_metrics(
    image: np.ndarray,
    reference: np.ndarray,
    *,
    tile_step_hr: int,
    highpass_sigma: float = 12.0,
) -> dict[str, Any]:
    img = _finite_2d("image", image)
    ref = _finite_2d("reference", reference)
    if img.shape != ref.shape:
        raise ValueError(f"shape mismatch vs reference {ref.shape}: {img.shape}")
    if tile_step_hr <= 0:
        raise ValueError("tile_step_hr must be positive")
    diff_hp = highpass(img - ref, highpass_sigma)
    profile_x = np.nanmean(np.abs(diff_hp), axis=0)
    profile_y = np.nanmean(np.abs(diff_hp), axis=1)
    lags = sorted({max(1, tile_step_hr // 2), int(tile_step_hr), int(tile_step_hr * 3 // 2), int(tile_step_hr * 2)})
    ac_x = _autocorr_at_lags(profile_x, lags)
    ac_y = _autocorr_at_lags(profile_y, lags)
    values = [abs(v) for v in [*ac_x.values(), *ac_y.values()]]
    max_ac = max(values) if values else float("nan")
    return {
        "tile_step_hr_px": int(tile_step_hr),
        "checked_lags_hr_px": lags,
        "autocorr_x": {str(k): v for k, v in ac_x.items()},
        "autocorr_y": {str(k): v for k, v in ac_y.items()},
        "max_abs_autocorr": float(max_ac),
        "diff_highpass_std_c": _robust_std(diff_hp),
    }


def seam_spectrum_probe(
    arrays: dict[str, np.ndarray],
    thresholds: SuiteThresholds,
    *,
    tile_step_hr: int,
    highpass_sigma: float = 12.0,
) -> CaseResult:
    ref_label = choose_reference_label(arrays)
    if ref_label is None:
        return CaseResult("seam_spectrum", "skip", skip_reason="no arrays available")
    labels = [k for k in arrays if k != ref_label]
    if not labels:
        return CaseResult(
            "seam_spectrum",
            "skip",
            metrics={"reference": ref_label},
            skip_reason="no candidate reconstruction beyond the reference baseline",
        )
    try:
        ref = _finite_2d(ref_label, arrays[ref_label])
    except ValueError as exc:
        return CaseResult("seam_spectrum", "skip", skip_reason=str(exc))

    metrics: dict[str, Any] = {
        "reference": ref_label,
        "highpass_sigma_hr_px": float(highpass_sigma),
        "candidates": {},
    }
    failed: list[str] = []
    for label in labels:
        try:
            m = seam_metrics(
                arrays[label],
                ref,
                tile_step_hr=tile_step_hr,
                highpass_sigma=highpass_sigma,
            )
            ok = m["max_abs_autocorr"] <= thresholds.seam_autocorr
            m["status"] = "pass" if ok else "fail"
            metrics["candidates"][label] = m
            if not ok:
                failed.append(label)
        except ValueError as exc:
            metrics["candidates"][label] = {"skip_reason": str(exc)}
    evaluated = [v for v in metrics["candidates"].values() if "status" in v]
    if not evaluated:
        return CaseResult(
            "seam_spectrum",
            "skip",
            metrics=metrics,
            thresholds={"seam_autocorr": thresholds.seam_autocorr},
            skip_reason="no shape-compatible candidate arrays",
        )
    return CaseResult(
        "seam_spectrum",
        "fail" if failed else "pass",
        metrics=metrics,
        thresholds={"seam_autocorr": thresholds.seam_autocorr},
    )


def _edge_mask(reference: np.ndarray, *, percentile: float = 90.0, dilation: int = 2) -> np.ndarray:
    ref = _finite_2d("reference", reference)
    smooth = gaussian_filter(ref, 1.5, mode="nearest")
    gy, gx = np.gradient(smooth)
    grad = np.hypot(gx, gy)
    finite = grad[np.isfinite(grad)]
    if finite.size < 16 or float(np.nanmax(finite)) <= 1e-12:
        return np.zeros_like(ref, dtype=bool)
    threshold = float(np.nanpercentile(finite, percentile))
    mask = grad >= threshold
    for _ in range(max(0, int(dilation))):
        mask = binary_dilation(mask)
    return mask.astype(bool, copy=False)


def bead_probe_metrics(image: np.ndarray, reference: np.ndarray, edge: np.ndarray) -> dict[str, float]:
    img = _finite_2d("image", image)
    ref = _finite_2d("reference", reference)
    if img.shape != ref.shape:
        raise ValueError(f"shape mismatch vs reference {ref.shape}: {img.shape}")
    if edge.shape != img.shape:
        raise ValueError("edge mask shape mismatch")
    if int(edge.sum()) < 32:
        raise ValueError("not enough edge pixels for beading probe")
    img_sig = np.abs(laplace(highpass(img, 1.2), mode="nearest"))
    ref_sig = np.abs(laplace(highpass(ref, 1.2), mode="nearest"))
    edge_img = img_sig[edge]
    edge_ref = ref_sig[edge]
    edge_delta = np.abs(img_sig - ref_sig)[edge]
    ref_p95 = float(np.nanpercentile(edge_ref, 95.0))
    img_p95 = float(np.nanpercentile(edge_img, 95.0))
    excess_p95 = float(np.nanpercentile(edge_delta, 95.0))
    flat = ~binary_dilation(edge, iterations=3)
    flat_ratio = float("nan")
    if int(flat.sum()) >= 32:
        flat_p95 = float(np.nanpercentile(img_sig[flat], 95.0))
        flat_ratio = img_p95 / max(flat_p95, 1e-9)
    return {
        "edge_signal_p95_c": img_p95,
        "reference_edge_signal_p95_c": ref_p95,
        "edge_signal_ratio_vs_reference": img_p95 / max(ref_p95, 1e-9),
        "edge_excess_p95_c": excess_p95,
        "edge_to_flat_signal_ratio": flat_ratio,
    }


def beading_probe(arrays: dict[str, np.ndarray], thresholds: SuiteThresholds) -> CaseResult:
    ref_label = choose_reference_label(arrays)
    if ref_label is None:
        return CaseResult("beading_probe", "skip", skip_reason="no arrays available")
    labels = [k for k in arrays if k != ref_label]
    if not labels:
        return CaseResult(
            "beading_probe",
            "skip",
            metrics={"reference": ref_label},
            skip_reason="no candidate reconstruction beyond the reference baseline",
        )
    try:
        ref = _finite_2d(ref_label, arrays[ref_label])
    except ValueError as exc:
        return CaseResult("beading_probe", "skip", skip_reason=str(exc))
    edge = _edge_mask(ref)
    if int(edge.sum()) < 32:
        return CaseResult(
            "beading_probe",
            "skip",
            metrics={"reference": ref_label, "edge_pixels": int(edge.sum())},
            skip_reason="reference has too few edge pixels for a beading probe",
        )

    metrics: dict[str, Any] = {
        "reference": ref_label,
        "edge_pixels": int(edge.sum()),
        "candidates": {},
    }
    failed: list[str] = []
    for label in labels:
        try:
            m = bead_probe_metrics(arrays[label], ref, edge)
            ok = (
                m["edge_signal_ratio_vs_reference"] <= thresholds.bead_edge_ratio
                and m["edge_excess_p95_c"] <= thresholds.bead_excess_p95_c
            )
            m["status"] = "pass" if ok else "fail"
            metrics["candidates"][label] = m
            if not ok:
                failed.append(label)
        except ValueError as exc:
            metrics["candidates"][label] = {"skip_reason": str(exc)}
    evaluated = [v for v in metrics["candidates"].values() if "status" in v]
    if not evaluated:
        return CaseResult(
            "beading_probe",
            "skip",
            metrics=metrics,
            thresholds={
                "bead_edge_ratio": thresholds.bead_edge_ratio,
                "bead_excess_p95_c": thresholds.bead_excess_p95_c,
            },
            skip_reason="no shape-compatible candidate arrays",
        )
    return CaseResult(
        "beading_probe",
        "fail" if failed else "pass",
        metrics=metrics,
        thresholds={
            "bead_edge_ratio": thresholds.bead_edge_ratio,
            "bead_excess_p95_c": thresholds.bead_excess_p95_c,
        },
    )


def load_npz_arrays(path: Path) -> dict[str, np.ndarray]:
    z = np.load(path)
    arrays: dict[str, np.ndarray] = {}
    keys = list(z.keys())
    if keys == ["temperature_c"]:
        arrays[path.stem] = np.asarray(z["temperature_c"], dtype=np.float32)
    else:
        for key in keys:
            arr = np.asarray(z[key])
            if arr.ndim == 2:
                arrays[str(key)] = arr.astype(np.float32, copy=False)
    return arrays


def parse_array_npy_spec(text: str) -> tuple[str, Path]:
    if "=" not in text:
        raise argparse.ArgumentTypeError("--array-npy expects NAME=PATH")
    name, raw_path = text.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("array name must be non-empty")
    return name, project_path(raw_path)


def project_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def load_explicit_arrays(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], list[str]]:
    arrays: dict[str, np.ndarray] = {}
    notes: list[str] = []
    for raw in args.arrays_npz or []:
        path = project_path(raw)
        if not path.exists():
            notes.append(f"arrays npz missing: {_rel(path)}")
            continue
        loaded = load_npz_arrays(path)
        arrays.update(loaded)
        notes.append(f"loaded {len(loaded)} arrays from {_rel(path)}")
    for spec in args.array_npy or []:
        name, path = parse_array_npy_spec(spec)
        if not path.exists():
            notes.append(f"array npy missing: {name}={_rel(path)}")
            continue
        arr = np.load(path)
        if arr.ndim != 2:
            notes.append(f"array npy skipped (not 2D): {name}={_rel(path)} shape={arr.shape}")
            continue
        arrays[name] = arr.astype(np.float32, copy=False)
        notes.append(f"loaded {name} from {_rel(path)}")
    return arrays, notes


def find_default_diagnostic_arrays() -> tuple[dict[str, np.ndarray], list[str]]:
    """Load ignored V8/K4 diagnostic arrays if this checkout happens to have them."""

    notes: list[str] = []
    arrays: dict[str, np.ndarray] = {}
    npz_candidates = [
        *PROJECT_ROOT.glob("remote_inbox/**/v8k4_step10000_render_arrays.npz"),
        *PROJECT_ROOT.glob("outputs/ep07_solver_diag/**/v8k4_step10000_render_arrays.npz"),
        *PROJECT_ROOT.glob("output/ep07_solver_diag/**/v8k4_step10000_render_arrays.npz"),
    ]
    for path in sorted(npz_candidates):
        loaded = load_npz_arrays(path)
        if loaded:
            arrays.update(loaded)
            notes.append(f"loaded default diagnostic npz: {_rel(path)}")
            return arrays, notes

    base = (
        ALGO_ROOT
        / "outputs"
        / "solver_v8_k4_nodrizzle_fullhalo_eval"
        / "eval_compare_v8k4_step10000"
    )
    tilehalo = (
        ALGO_ROOT
        / "outputs"
        / "solver_v8_k4_nodrizzle_fullhalo_eval"
        / "eval_compare_v8k4_step10000_tilehalo"
    )
    mapping = {
        "aligned_mean": base / "aligned_mean.npy",
        "tiled_p192_o128": base / "tiled_p192_o128.npy",
        "dense_p192_o160": base / "dense_p192_o160.npy",
        "full_halo96": base / "full_halo96.npy",
        "tile_halo32": tilehalo / "tile_halo32.npy",
        "tile_halo64": tilehalo / "tile_halo64.npy",
        "tile_halo96": tilehalo / "tile_halo96.npy",
    }
    for name, path in mapping.items():
        if path.exists():
            arr = np.load(path)
            if arr.ndim == 2:
                arrays[name] = arr.astype(np.float32, copy=False)
    if arrays:
        notes.append(f"loaded {len(arrays)} local ignored V8/K4 diagnostic npy arrays")
    else:
        notes.append("no default diagnostic arrays found")
    return arrays, notes


def _default_training_config() -> dict[str, Any]:
    return {
        "scale": 2,
        "in_channels": 9,
        "base_channels": 64,
        "patch_size_hr": 384,
        "highpass_sigma": 5.0,
        "forward_model_psf_sigma": 0.5,
        "unroll_steps": 2,
        "solver_m_frames": 16,
        "solver_band_sigma": 5.0,
        "solver_huber_delta": 0.0,
        "solver_share_weights": True,
        "solver_eta_init": 0.5,
        "solver_learn_eta": False,
        "solver_dc_rim_lr_px": 8,
        "solver_no_drizzle": True,
        "solver_warmstart": "aligned_mean",
        "solver_prox_use_se": False,
        "solver_prox_norm": "none",
        "solver_prox_highpass_residual": False,
        "solver_prox_highpass_sigma_hr": 5.0,
        "phase_bin_channels": 4,
    }


def _load_solver_checkpoint(path: Path, device: str):
    import torch
    from unet_sr.unroll import UnrolledSolver

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"unsupported solver checkpoint format: {path}")
    cfg = _default_training_config()
    cfg.update(dict(ckpt.get("config") or {}))
    cond_channels = 5 if bool(cfg.get("solver_no_drizzle", False)) else int(cfg.get("in_channels", 9))
    solver = UnrolledSolver(
        n_steps=int(cfg.get("unroll_steps", 2)),
        cond_channels=cond_channels,
        base_channels=int(cfg.get("base_channels", 64)),
        scale=int(cfg.get("scale", 2)),
        share_weights=bool(cfg.get("solver_share_weights", True)),
        band_highpass_sigma_lr_px=float(cfg.get("solver_band_sigma", 5.0)),
        huber_delta=float(cfg.get("solver_huber_delta", 0.0)),
        eta_init=float(cfg.get("solver_eta_init", 0.5)),
        learn_eta=bool(cfg.get("solver_learn_eta", False)),
        prox_use_se=bool(cfg.get("solver_prox_use_se", False)),
        prox_norm=str(cfg.get("solver_prox_norm", "none")),
        prox_highpass_residual=bool(cfg.get("solver_prox_highpass_residual", False)),
        prox_highpass_sigma_hr=float(cfg.get("solver_prox_highpass_sigma_hr", 5.0)),
    )
    solver.load_state_dict(ckpt["model_state_dict"])
    solver.eval()
    requested = torch.device(device)
    if requested.type == "cuda" and not torch.cuda.is_available():
        requested = torch.device("cpu")
    solver.to(requested)
    return solver, cfg, int(ckpt.get("step", 0)), requested


def render_checkpoint_arrays(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], list[str]]:
    if not args.checkpoint:
        return {}, ["checkpoint render not requested"]
    checkpoint = project_path(args.checkpoint)
    if not checkpoint.exists():
        return {}, [f"checkpoint missing: {_rel(checkpoint)}"]

    notes: list[str] = []
    try:
        from unet_sr.real_eval import (
            _load_real_eval_cache,
            _solver_conditioning_from_burst,
            infer_solver_from_burst,
            infer_solver_from_burst_full_halo,
        )

        solver, cfg, step, device = _load_solver_checkpoint(checkpoint, args.device)
        raw_frames, shifts = _load_real_eval_cache(args.frame_limit, args.alignment_method)
        eval_cfg = dict(cfg)
        patch_size_hr = int(args.patch_size_hr or eval_cfg.get("patch_size_hr", 384))
        eval_cfg["patch_size_hr"] = patch_size_hr
        train_ns = SimpleNamespace(**eval_cfg)
        cond, _ = _solver_conditioning_from_burst(
            raw_frames,
            shifts,
            training_config=train_ns,
            scale=int(eval_cfg.get("scale", 2)),
        )
        arrays = {"aligned_mean": cond[0].astype(np.float32, copy=False)}
        t0 = time.monotonic()
        arrays["tiled"] = infer_solver_from_burst(
            solver,
            raw_frames,
            shifts,
            training_config=train_ns,
            patch_size_hr=patch_size_hr,
            overlap=int(args.overlap),
            device=device,
            tile_batch_size=int(args.tile_batch),
        ).astype(np.float32, copy=False)
        arrays[f"full_halo{int(args.full_halo_hr)}"] = infer_solver_from_burst_full_halo(
            solver,
            raw_frames,
            shifts,
            training_config=train_ns,
            halo_hr=int(args.full_halo_hr),
            device=device,
        ).astype(np.float32, copy=False)
        notes.append(
            "rendered checkpoint "
            f"{_rel(checkpoint)} step={step} patch={patch_size_hr} overlap={args.overlap} "
            f"full_halo={args.full_halo_hr} device={device} seconds={time.monotonic() - t0:.1f}"
        )
        if args.rendered_arrays_npz:
            out = project_path(args.rendered_arrays_npz)
            out.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(out, **arrays)
            notes.append(f"saved rendered arrays: {_rel(out)}")
        return arrays, notes
    except Exception as exc:
        return {}, [f"checkpoint render skipped ({type(exc).__name__}: {exc})"]


def load_real_baseline_arrays(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], list[str]]:
    if args.no_real_baseline:
        return {}, ["real aligned baseline disabled"]
    try:
        from unet_sr.real_eval import _load_real_eval_cache, _solver_conditioning_from_burst

        raw_frames, shifts = _load_real_eval_cache(args.frame_limit, args.alignment_method)
        cfg = SimpleNamespace(
            scale=2,
            highpass_sigma=5.0,
            solver_no_drizzle=True,
            solver_m_frames=16,
            solver_dc_rim_lr_px=8,
            forward_model_psf_sigma=0.5,
            phase_bin_channels=4,
            solver_warmstart="aligned_mean",
        )
        cond, _ = _solver_conditioning_from_burst(raw_frames, shifts, training_config=cfg, scale=2)
        return {"aligned_mean": cond[0].astype(np.float32, copy=False)}, [
            f"loaded real aligned_mean baseline frame_limit={args.frame_limit}"
        ]
    except Exception as exc:
        return {}, [f"real aligned baseline skipped ({type(exc).__name__}: {exc})"]


def build_arrays(args: argparse.Namespace) -> tuple[dict[str, np.ndarray], list[str]]:
    arrays, notes = load_explicit_arrays(args)
    explicit_arrays_requested = bool(args.arrays_npz or args.array_npy)
    if explicit_arrays_requested:
        return arrays, notes

    rendered, render_notes = render_checkpoint_arrays(args)
    notes.extend(render_notes)
    if args.checkpoint:
        return rendered, notes

    default_arrays, default_notes = find_default_diagnostic_arrays()
    notes.extend(default_notes)
    if default_arrays:
        return default_arrays, notes

    baseline, baseline_notes = load_real_baseline_arrays(args)
    notes.extend(baseline_notes)
    return baseline, notes


def infer_tile_step_hr(args: argparse.Namespace) -> int:
    if args.tile_step_hr:
        return int(args.tile_step_hr)
    patch = int(args.patch_size_hr or 0)
    if args.checkpoint and patch > int(args.overlap):
        return patch - int(args.overlap)
    return 64


def run_suite(args: argparse.Namespace) -> dict[str, Any]:
    arrays, input_notes = build_arrays(args)
    thresholds = SuiteThresholds(
        flat_lowpass_p95_delta_c=float(args.flat_p95_threshold_c),
        flat_lowpass_std_delta_c=float(args.flat_std_threshold_c),
        extent_nrmse=float(args.extent_nrmse_threshold),
        extent_p95_abs_diff_c=float(args.extent_p95_threshold_c),
        seam_autocorr=float(args.seam_autocorr_threshold),
        bead_edge_ratio=float(args.bead_ratio_threshold),
        bead_excess_p95_c=float(args.bead_excess_threshold_c),
    )
    tile_step = infer_tile_step_hr(args)
    cases = [
        flat_roi_probe(
            arrays,
            thresholds,
            roi_size=int(args.flat_roi_size),
            lowpass_sigma=float(args.flat_lowpass_sigma),
        ),
        extent_consistency_probe(
            arrays,
            thresholds,
            crop_margin_hr=int(args.extent_crop_margin_hr),
        ),
        seam_spectrum_probe(
            arrays,
            thresholds,
            tile_step_hr=tile_step,
            highpass_sigma=float(args.seam_highpass_sigma),
        ),
        beading_probe(arrays, thresholds),
    ]
    counts = {status: sum(1 for c in cases if c.status == status) for status in ("pass", "fail", "skip")}
    if counts["fail"]:
        suite_status = "fail"
    elif counts["pass"] and counts["skip"]:
        suite_status = "pass_with_skips"
    elif counts["pass"]:
        suite_status = "pass"
    else:
        suite_status = "skipped"

    report = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _now_utc(),
        "suite_status": suite_status,
        "summary": counts,
        "audit": {
            "detector_pitch_um_per_pixel": DETECTOR_PITCH_UM,
            "spatial_resolution_um": SPATIAL_RESOLUTION_UM,
            "noise_floor_c": NOISE_FLOOR_C,
            "historical_10um_policy": (
                "Historical 10um-derived physical thresholds are not used. "
                "The probes use degC, unitless ratios, or HR-pixel periodicity; "
                "the physical truth audited for this run is 20um/pixel and 20um spatial resolution."
            ),
        },
        "run": {
            "cwd": str(Path.cwd()),
            "project_root": str(PROJECT_ROOT),
            "git_sha": _git_sha(),
            "command": " ".join(sys.argv),
            "device": args.device,
            "checkpoint": str(args.checkpoint or ""),
            "tile_step_hr_px": tile_step,
        },
        "inputs": {
            "labels": list(arrays.keys()),
            "shapes": {k: list(np.asarray(v).shape) for k, v in arrays.items()},
            "notes": input_notes,
        },
        "cases": [asdict(c) for c in cases],
    }
    return _json_ready(report)


def write_report(report: dict[str, Any], output_json: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", default="", help="Optional solver checkpoint (.pt) to render and score.")
    parser.add_argument("--arrays-npz", action="append", default=[], help="Saved render arrays npz.")
    parser.add_argument("--array-npy", action="append", default=[], help="Named 2D array, NAME=PATH.")
    parser.add_argument(
        "--output-json",
        default=str(PROJECT_ROOT / "output" / "ep07_solver_regression_suite" / "report.json"),
    )
    parser.add_argument(
        "--rendered-arrays-npz",
        default="",
        help="Optional path to save checkpoint-rendered aligned/tiled/full_halo arrays.",
    )
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or cuda:N for checkpoint rendering.")
    parser.add_argument("--frame-limit", type=int, default=248)
    parser.add_argument("--alignment-method", default="contour_refined")
    parser.add_argument("--patch-size-hr", type=int, default=0, help="0 = checkpoint/default patch size.")
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--tile-batch", type=int, default=16)
    parser.add_argument("--full-halo-hr", type=int, default=96)
    parser.add_argument("--tile-step-hr", type=int, default=0, help="0 = infer; arrays default to 64 HR px.")
    parser.add_argument("--no-real-baseline", action="store_true")
    parser.add_argument("--fail-on-skip", action="store_true")

    parser.add_argument("--flat-roi-size", type=int, default=96)
    parser.add_argument("--flat-lowpass-sigma", type=float, default=16.0)
    parser.add_argument("--flat-p95-threshold-c", type=float, default=SuiteThresholds.flat_lowpass_p95_delta_c)
    parser.add_argument("--flat-std-threshold-c", type=float, default=SuiteThresholds.flat_lowpass_std_delta_c)
    parser.add_argument("--extent-crop-margin-hr", type=int, default=0)
    parser.add_argument("--extent-nrmse-threshold", type=float, default=SuiteThresholds.extent_nrmse)
    parser.add_argument("--extent-p95-threshold-c", type=float, default=SuiteThresholds.extent_p95_abs_diff_c)
    parser.add_argument("--seam-highpass-sigma", type=float, default=12.0)
    parser.add_argument("--seam-autocorr-threshold", type=float, default=SuiteThresholds.seam_autocorr)
    parser.add_argument("--bead-ratio-threshold", type=float, default=SuiteThresholds.bead_edge_ratio)
    parser.add_argument("--bead-excess-threshold-c", type=float, default=SuiteThresholds.bead_excess_p95_c)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = run_suite(args)
    out = project_path(args.output_json)
    write_report(report, out)
    print(json.dumps({
        "suite_status": report["suite_status"],
        "summary": report["summary"],
        "output_json": _rel(out),
        "labels": report["inputs"]["labels"],
    }, indent=2, sort_keys=True))
    if report["summary"]["fail"] > 0:
        return 1
    if args.fail_on_skip and report["summary"]["skip"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
