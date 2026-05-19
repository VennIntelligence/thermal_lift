"""Small plotting helpers with lazy matplotlib imports."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _plt():
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise ImportError("matplotlib is required for tcforge.visualization functions") from exc
    return plt


def plot_image(
    image: np.ndarray,
    *,
    title: str | None = None,
    cmap: str = "inferno",
    colorbar: bool = True,
    ax=None,
):
    """Plot one 2D image and return ``(fig, ax)``."""

    arr = np.asarray(image)
    if arr.ndim != 2:
        raise ValueError("image must be 2D")
    plt = _plt()
    if ax is None:
        fig, ax = plt.subplots(figsize=(6.4, 4.8), dpi=300)
    else:
        fig = ax.figure
    im = ax.imshow(arr, cmap=cmap)
    ax.set_axis_off()
    if title:
        ax.set_title(title)
    if colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return fig, ax


def plot_burst_preview(
    burst: np.ndarray,
    *,
    indices: tuple[int, ...] = (0, 1, 2, 3),
    cmap: str = "inferno",
):
    """Plot a compact preview of selected frames from an LR burst."""

    arr = np.asarray(burst)
    if arr.ndim != 3:
        raise ValueError("burst must have shape (N, H, W)")
    chosen = [idx for idx in indices if 0 <= int(idx) < arr.shape[0]]
    if not chosen:
        raise ValueError("indices select no frames from burst")
    plt = _plt()
    fig, axes = plt.subplots(1, len(chosen), figsize=(3.0 * len(chosen), 3.0), dpi=300, squeeze=False)
    for ax, idx in zip(axes.ravel(), chosen, strict=True):
        ax.imshow(arr[idx], cmap=cmap)
        ax.set_title(f"frame {idx}")
        ax.set_axis_off()
    fig.tight_layout()
    return fig, axes.ravel().tolist()


def save_figure(fig, path: str | Path, *, dpi: int = 300, close: bool = True) -> Path:
    """Save a matplotlib figure, optionally closing it afterward."""

    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=int(dpi), bbox_inches="tight", facecolor="white")
    if close:
        _plt().close(fig)
    return out
