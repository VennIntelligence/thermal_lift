"""Evaluation metrics for thermal SR (numpy/scipy only — no torch, CPU-friendly).

Shared by ``real_eval`` (GT-free, on real frames) and ``synth_eval`` (vs the
synthetic GT).  These replace the old ``artifact_score`` / ``raw_control_corr``
*objectives*, which compared the clean SR output against the **degraded** input
(bicubic of the raw mean) and so rewarded staying close to the blur and
penalised genuine restoration — "any cleaner image scored worse".

The principled split:

- ``out_of_band_ratio`` — GT-free artifact / hallucination monitor.  Spectral
  energy above the pitch-set recoverable cutoff; needs no reference and no PSF.
  The FM-1 beading/checkerboard cliff spikes it, so a jump across checkpoints
  flags the cliff — without punishing legitimate in-band sharpening.
- ``psnr`` / ``region_rmse`` / ``boundary_f1`` — rigorous fidelity *vs the GT*,
  for the held-out synthetic set (where we actually have ground truth).  These
  measure what we care about on v4 data: recoverable-band accuracy, the
  isothermal temperature level *inside* a body, and defect/edge preservation.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation


def _finite_mask(*arrays: np.ndarray) -> np.ndarray:
    base = np.isfinite(np.asarray(arrays[0], dtype=np.float64))
    for arr in arrays[1:]:
        base = base & np.isfinite(np.asarray(arr, dtype=np.float64))
    return base


def out_of_band_ratio(img: np.ndarray, *, scale: int = 2) -> float:
    """Fraction of spectral energy above the recoverable (pitch-set) cutoff.

    On an ``scale``x SR grid the LR sampling constrains content only up to the LR
    Nyquist, at radial frequency ``1 / (2 * scale)`` cycles per HR pixel.  Energy
    above that is either legitimate edge sharpening or — when it spikes, as in the
    FM-1 beading / pearl-necklace / checkerboard cliff — hallucination.  A Hann
    window suppresses FFT border leakage so the ratio is stable across runs.

    GT-free and PSF-free: usable on real frames where there is no ground truth.
    """

    arr = np.asarray(img, dtype=np.float64)
    if arr.ndim != 2:
        raise ValueError(f"out_of_band_ratio expects a 2D image, got shape {arr.shape}")
    arr = np.nan_to_num(arr, copy=True)
    arr = arr - float(arr.mean())
    h, w = arr.shape
    win = np.hanning(h)[:, None] * np.hanning(w)[None, :]
    spec = np.fft.fftshift(np.fft.fft2(arr * win))
    power = spec.real ** 2 + spec.imag ** 2
    fy = np.fft.fftshift(np.fft.fftfreq(h))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(w))[None, :]
    radial = np.sqrt(fy ** 2 + fx ** 2)
    cutoff = 0.5 / float(scale)
    total = float(power.sum())
    if total <= 0.0:
        return 0.0
    return float(power[radial > cutoff].sum() / total)


def psnr(pred: np.ndarray, target: np.ndarray, *, data_range: float | None = None) -> float:
    """Peak SNR (dB) of ``pred`` vs the GT ``target``.  Higher is better.

    ``data_range`` defaults to the GT's own min–max span, so the score is
    invariant to the scene's absolute temperature offset.
    """

    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    mask = _finite_mask(p, t)
    if not mask.any():
        return float("nan")
    mse = float(np.mean((p[mask] - t[mask]) ** 2))
    if mse <= 0.0:
        return float("inf")
    dr = float(data_range) if data_range is not None else float(t[mask].max() - t[mask].min())
    if dr <= 0.0:
        return float("inf")
    return float(10.0 * np.log10(dr * dr / mse))


def region_rmse(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, *, threshold: float = 0.5) -> float:
    """RMSE between ``pred`` and ``target`` *inside* the structure mask.

    On near-isothermal v4 data this is the temperature-level fidelity inside the
    chip body — the quantity the old edge-only metrics never measured.  Returns
    NaN when the patch has no support.
    """

    p = np.asarray(pred, dtype=np.float64)
    t = np.asarray(target, dtype=np.float64)
    region = (np.asarray(mask) > float(threshold)) & _finite_mask(p, t)
    if not region.any():
        return float("nan")
    return float(np.sqrt(np.mean((p[region] - t[region]) ** 2)))


def _mask_boundary(binary: np.ndarray) -> np.ndarray:
    edge = np.zeros_like(binary, dtype=bool)
    diff_v = binary[:-1, :] != binary[1:, :]
    diff_h = binary[:, :-1] != binary[:, 1:]
    edge[:-1, :] |= diff_v
    edge[1:, :] |= diff_v
    edge[:, :-1] |= diff_h
    edge[:, 1:] |= diff_h
    return edge


def boundary_f1(
    pred: np.ndarray,
    target_mask: np.ndarray,
    *,
    tol_px: int = 2,
    grad_percentile: float = 92.0,
    threshold: float = 0.5,
) -> dict[str, float]:
    """F1 between predicted edges and the true structure boundary, within ``tol_px``.

    True edges = morphological boundary of ``target_mask`` (chip outline + every
    defect rim: holes, cracks, notches).  Predicted edges = strong-gradient
    pixels of ``pred`` above a relative percentile threshold.  Matching dilates
    by ``tol_px`` so sub-pixel edge placement is not punished.

    Captures defect preservation directly: a filled-in hole drops **recall**, a
    hallucinated edge / beading drops **precision**.  Returns precision, recall, f1.
    """

    pred = np.asarray(pred, dtype=np.float64)
    gt_edge = _mask_boundary(np.asarray(target_mask) > float(threshold))

    gx = np.zeros_like(pred)
    gy = np.zeros_like(pred)
    gx[:, 1:-1] = pred[:, 2:] - pred[:, :-2]
    gy[1:-1, :] = pred[2:, :] - pred[:-2, :]
    gmag = np.sqrt(gx * gx + gy * gy)
    finite = np.isfinite(gmag)
    if not finite.any() or not gt_edge.any():
        return {"f1": float("nan"), "precision": float("nan"), "recall": float("nan")}

    thr = float(np.percentile(gmag[finite], grad_percentile))
    pred_edge = gmag >= max(thr, 1e-12)

    struct = np.ones((2 * int(tol_px) + 1, 2 * int(tol_px) + 1), dtype=bool)
    gt_dilated = binary_dilation(gt_edge, structure=struct)
    pred_dilated = binary_dilation(pred_edge, structure=struct)

    n_pred = int(pred_edge.sum())
    n_gt = int(gt_edge.sum())
    precision = int((pred_edge & gt_dilated).sum()) / n_pred if n_pred else 0.0
    recall = int((gt_edge & pred_dilated).sum()) / n_gt if n_gt else 0.0
    f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"f1": float(f1), "precision": float(precision), "recall": float(recall)}
