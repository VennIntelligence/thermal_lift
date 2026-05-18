"""Metrics for EP06 contour-level SR without scikit-image dependencies."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt, laplace


def gradient_magnitude(img: np.ndarray) -> np.ndarray:
    """Compute gradient magnitude over the last two axes."""

    arr = np.asarray(img, dtype=np.float32)
    if arr.ndim < 2:
        raise ValueError("img must have at least two dimensions")
    gy, gx = np.gradient(arr, axis=(-2, -1))
    return np.hypot(gx, gy).astype(np.float32, copy=False)


def psnr(reference: np.ndarray, estimate: np.ndarray, *, data_range: float | None = None) -> float:
    """Peak signal-to-noise ratio in dB."""

    ref = np.asarray(reference, dtype=np.float64)
    est = np.asarray(estimate, dtype=np.float64)
    if ref.shape != est.shape:
        raise ValueError("reference and estimate shapes differ")
    valid = np.isfinite(ref) & np.isfinite(est)
    if not np.any(valid):
        return float("nan")
    mse = float(np.mean((ref[valid] - est[valid]) ** 2))
    if mse <= 0:
        return float("inf")
    if data_range is None:
        data_range = float(np.nanmax(ref[valid]) - np.nanmin(ref[valid]))
        if data_range <= 0 or not np.isfinite(data_range):
            data_range = 1.0
    return float(20.0 * np.log10(data_range / np.sqrt(mse)))


def _mask_from_edges(
    edges_or_points: np.ndarray,
    shape: tuple[int, int],
    *,
    threshold: float | None = None,
    percentile: float | None = None,
    point_order: str = "yx",
) -> np.ndarray:
    arr = np.asarray(edges_or_points)
    if arr.ndim == 2 and arr.shape == shape:
        if arr.dtype == bool:
            return arr.copy()
        finite = np.isfinite(arr)
        if threshold is None:
            threshold = float(np.nanpercentile(arr[finite], percentile or 95.0)) if finite.any() else 0.0
        return finite & (arr >= threshold)

    pts = np.asarray(edges_or_points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError("edge input must be a mask/image or an (N, 2) point array")
    rr = pts[:, 0] if point_order == "yx" else pts[:, 1]
    cc = pts[:, 1] if point_order == "yx" else pts[:, 0]
    out = np.zeros(shape, dtype=bool)
    r = np.rint(rr).astype(int)
    c = np.rint(cc).astype(int)
    ok = (r >= 0) & (r < shape[0]) & (c >= 0) & (c < shape[1])
    out[r[ok], c[ok]] = True
    return out


def contour_chamfer_from_edges(
    sr_edges: np.ndarray,
    reference_edges: np.ndarray,
    *,
    image_shape: tuple[int, int] | None = None,
    threshold: float | None = None,
    percentile: float | None = None,
    symmetric: bool = True,
    point_order: str = "yx",
) -> float:
    """Chamfer distance between two edge masks or contour point sets."""

    shape = image_shape
    if shape is None:
        for item in (sr_edges, reference_edges):
            arr = np.asarray(item)
            if arr.ndim == 2 and arr.shape[1] != 2:
                shape = arr.shape
                break
    if shape is None:
        raise ValueError("image_shape is required for point-array inputs")

    sr_mask = _mask_from_edges(sr_edges, shape, threshold=threshold, percentile=percentile, point_order=point_order)
    ref_mask = _mask_from_edges(reference_edges, shape, threshold=threshold, percentile=percentile, point_order=point_order)
    if not sr_mask.any() or not ref_mask.any():
        return float("inf")
    dist_to_sr = distance_transform_edt(~sr_mask)
    ref_to_sr = float(np.mean(dist_to_sr[ref_mask]))
    if not symmetric:
        return ref_to_sr
    dist_to_ref = distance_transform_edt(~ref_mask)
    sr_to_ref = float(np.mean(dist_to_ref[sr_mask]))
    return 0.5 * (ref_to_sr + sr_to_ref)


contour_chamfer = contour_chamfer_from_edges


def _accepts(fn: Callable[..., Any], arg: str) -> bool:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return False
    return arg in sig.parameters or any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())


def _call_sr(
    sr_method: Callable[..., Any],
    frames: np.ndarray,
    shifts: np.ndarray,
    *,
    weights: np.ndarray | None,
    kwargs: dict[str, Any],
) -> np.ndarray:
    call_kwargs = dict(kwargs)
    if weights is not None and _accepts(sr_method, "weights"):
        call_kwargs["weights"] = weights
    result = sr_method(frames, shifts, **call_kwargs)
    if isinstance(result, tuple):
        result = result[0]
    return np.asarray(result, dtype=np.float32)


def split_half_consistency(
    frames: np.ndarray,
    shifts: np.ndarray,
    sr_method: Callable[..., Any],
    *,
    n_splits: int = 10,
    random_state: int = 0,
    weights: np.ndarray | None = None,
    **sr_kwargs: Any,
) -> pd.DataFrame:
    """Random split-half reconstruction consistency table."""

    frame_arr = np.asarray(frames)
    shift_arr = np.asarray(shifts, dtype=float)
    if frame_arr.ndim != 3 or shift_arr.shape != (len(frame_arr), 2):
        raise ValueError("frames must be (N,H,W) and shifts must be (N,2)")
    rng = np.random.default_rng(random_state)
    rows: list[dict[str, float | int]] = []
    for split_idx in range(int(n_splits)):
        perm = rng.permutation(len(frame_arr))
        mid = len(perm) // 2
        idx_a = np.sort(perm[:mid])
        idx_b = np.sort(perm[mid:])
        w_a = None if weights is None else np.asarray(weights)[idx_a]
        w_b = None if weights is None else np.asarray(weights)[idx_b]
        sr_a = _call_sr(sr_method, frame_arr[idx_a], shift_arr[idx_a], weights=w_a, kwargs=sr_kwargs)
        sr_b = _call_sr(sr_method, frame_arr[idx_b], shift_arr[idx_b], weights=w_b, kwargs=sr_kwargs)
        valid = np.isfinite(sr_a) & np.isfinite(sr_b)
        diff = sr_a[valid] - sr_b[valid]
        rmse = float(np.sqrt(np.mean(diff * diff)))
        denom = float(np.std(sr_a[valid]) + np.std(sr_b[valid]))
        corr = float(np.corrcoef(sr_a[valid].ravel(), sr_b[valid].ravel())[0, 1]) if valid.sum() > 1 else float("nan")
        rows.append(
            {
                "split": split_idx,
                "n_a": int(len(idx_a)),
                "n_b": int(len(idx_b)),
                "rmse": rmse,
                "nrmse": rmse / max(denom, 1e-12),
                "corr": corr,
                "psnr_db": psnr(sr_a, sr_b),
            }
        )
    return pd.DataFrame(rows)


def artifact_score(
    sr_img: np.ndarray,
    lr_img: np.ndarray | None = None,
    *,
    scale: int = 2,
    return_components: bool = False,
) -> float | dict[str, float]:
    """Heuristic ringing/blockiness/overshoot score."""

    arr = np.asarray(sr_img, dtype=np.float32)
    if arr.ndim != 2:
        raise ValueError("sr_img must be 2D")
    finite = np.isfinite(arr)
    if not finite.any():
        comps = {"ringing": float("inf"), "blockiness": float("inf"), "overshoot": float("inf"), "score": float("inf")}
        return comps if return_components else comps["score"]

    fill = float(np.nanmedian(arr[finite]))
    clean = np.where(finite, arr, fill)
    grad = gradient_magnitude(clean)
    ringing = float(np.nanmedian(np.abs(laplace(clean, mode="nearest"))) / (np.nanmedian(grad) + 1e-12))

    row_diff = np.abs(np.diff(clean, axis=0))
    col_diff = np.abs(np.diff(clean, axis=1))
    rb = ((np.arange(row_diff.shape[0]) + 1) % max(1, scale)) == 0
    cb = ((np.arange(col_diff.shape[1]) + 1) % max(1, scale)) == 0
    boundary = []
    interior = []
    if row_diff.size:
        boundary.append(row_diff[rb, :].ravel())
        interior.append(row_diff[~rb, :].ravel())
    if col_diff.size:
        boundary.append(col_diff[:, cb].ravel())
        interior.append(col_diff[:, ~cb].ravel())
    bvals = np.concatenate([v for v in boundary if v.size]) if boundary else np.array([0.0])
    ivals = np.concatenate([v for v in interior if v.size]) if interior else np.array([1.0])
    blockiness = float(np.nanmedian(bvals) / (np.nanmedian(ivals) + 1e-12))

    overshoot = 0.0
    if lr_img is not None:
        lr = np.asarray(lr_img, dtype=np.float32)
        lo, hi = float(np.nanpercentile(lr, 0.5)), float(np.nanpercentile(lr, 99.5))
        overshoot = float(np.mean(np.maximum(clean - hi, 0.0) + np.maximum(lo - clean, 0.0)) / (hi - lo + 1e-12))

    score = ringing + 0.25 * blockiness + 5.0 * overshoot
    comps = {"ringing": ringing, "blockiness": blockiness, "overshoot": overshoot, "score": float(score)}
    return comps if return_components else comps["score"]
