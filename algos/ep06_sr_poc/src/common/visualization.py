"""Visualization helpers for EP06 direct SR comparisons."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter, zoom

from thermal_core.plotting import COLORMAPS, METHOD_COLOR_LIST, savefig_academic, setup_academic_style


METHOD_LABELS = ("LR", "Bicubic", "SAA-u", "SAA-w", "IBP", "MAP-TV")
ROI = tuple[int, int, int, int]


def _limits(images: Sequence[np.ndarray], *, symmetric: bool = True) -> tuple[float, float]:
    values = []
    for image in images:
        arr = np.asarray(image, dtype=np.float32)
        sy = max(1, arr.shape[0] // 512)
        sx = max(1, arr.shape[1] // 512)
        values.append(arr[::sy, ::sx].ravel())
    vals = np.concatenate(values)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return -1.0, 1.0
    if symmetric:
        lim = float(np.nanpercentile(np.abs(vals), 99.0))
        return -lim, lim
    return float(np.nanpercentile(vals, 1.0)), float(np.nanpercentile(vals, 99.0))


def _up_lr(lr: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return zoom(lr, (shape[0] / lr.shape[0], shape[1] / lr.shape[1]), order=0).astype(np.float32)


def plot_sr_comparison(
    lr: np.ndarray,
    bicubic: np.ndarray,
    saa_uniform: np.ndarray,
    saa_weighted: np.ndarray,
    ibp: np.ndarray,
    map_tv: np.ndarray,
    *,
    roi: ROI | None = None,
    title: str | None = None,
    save_path: str | Path | None = None,
    close: bool = True,
) -> plt.Figure:
    """Plot LR/bicubic/SAA/IBP/MAP-TV side by side."""

    setup_academic_style()
    target_shape = np.asarray(bicubic).shape
    images = [_up_lr(np.asarray(lr), target_shape), bicubic, saa_uniform, saa_weighted, ibp, map_tv]
    if roi is not None:
        r0, r1, c0, c1 = roi
        images = [np.asarray(img)[r0:r1, c0:c1] for img in images]
    vmin, vmax = _limits(images, symmetric=True)
    fig, axes = plt.subplots(1, 6, figsize=(11.0, 2.35), constrained_layout=True)
    for ax, label, image in zip(axes, METHOD_LABELS, images, strict=True):
        ax.imshow(image, cmap=COLORMAPS["residual_diff"], vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(label)
        ax.set_xticks([])
        ax.set_yticks([])
    if title:
        fig.suptitle(title, fontsize=11, fontweight="bold")
    if save_path is not None:
        savefig_academic(fig, save_path, close=close)
    return fig


def plot_control_track_comparison(
    highpass_images: Mapping[str, np.ndarray],
    raw_images: Mapping[str, np.ndarray],
    *,
    sigma: float = 10.0,
    save_path: str | Path | None = None,
    close: bool = True,
) -> plt.Figure:
    """Plot highpass main-track outputs against raw-control highpass views."""

    setup_academic_style()
    keys = list(highpass_images)
    raw_struct = {key: np.asarray(raw_images[key]) - gaussian_filter(raw_images[key], sigma=sigma, mode="nearest") for key in keys}
    values = [np.asarray(highpass_images[key]) for key in keys] + [raw_struct[key] for key in keys]
    vmin, vmax = _limits(values, symmetric=True)
    fig, axes = plt.subplots(2, len(keys), figsize=(3.0 * len(keys), 4.2), constrained_layout=True)
    for col, key in enumerate(keys):
        for row, image, prefix in ((0, highpass_images[key], "highpass"), (1, raw_struct[key], "raw control")):
            axes[row, col].imshow(image, cmap=COLORMAPS["residual_diff"], vmin=vmin, vmax=vmax, interpolation="nearest")
            axes[row, col].set_title(f"{key}\n{prefix}")
            axes[row, col].set_xticks([])
            axes[row, col].set_yticks([])
    if save_path is not None:
        savefig_academic(fig, save_path, close=close)
    return fig


def plot_metric_bars(
    values: Mapping[str, float],
    *,
    ylabel: str,
    title: str,
    save_path: str | Path | None = None,
    close: bool = True,
) -> plt.Figure:
    """Save a compact method metric bar chart."""

    setup_academic_style()
    labels = list(values)
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(5.5, 3.0), constrained_layout=True)
    ax.bar(x, [values[k] for k in labels], color=METHOD_COLOR_LIST[: len(labels)])
    ax.set_xticks(x, labels, rotation=30, ha="right")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if save_path is not None:
        savefig_academic(fig, save_path, close=close)
    return fig
