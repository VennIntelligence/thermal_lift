"""CVPR-standard academic plotting utilities for the Thermal Lift project.

Full specification: docs/plotting_standards.md

Usage
-----
    from thermal_core.plotting import setup_academic_style, FIGURE_SIZES, METHOD_COLORS

    setup_academic_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["single_col"])
    ...
    savefig_academic(fig, "output/ep01/my_figure.pdf")
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# ── Colour palette (≤ 6 methods) ──────────────────────────────────────────
METHOD_COLORS = {
    "primary":   "#4C72B0",  # steel blue
    "secondary": "#55A868",  # muted green
    "accent_1":  "#C44E52",  # soft red
    "accent_2":  "#8172B2",  # lavender purple
    "accent_3":  "#DD8452",  # warm orange
    "neutral":   "#937860",  # taupe
}

# Ordered list for quick indexing: colors[0], colors[1], ...
METHOD_COLOR_LIST: list[str] = list(METHOD_COLORS.values())

# ── Recommended colormaps ─────────────────────────────────────────────────
COLORMAPS = {
    "temperature":   "inferno",   # thermal field / HR image
    "residual_pos":  "YlOrRd",    # single-sided residual (≥ 0)
    "residual_diff": "RdBu_r",    # diverging residual (±)
    "coverage":      "viridis",   # count / coverage
}

# ── Figure sizes (width, height) in inches ────────────────────────────────
FIGURE_SIZES = {
    "single_col":   (3.5, 2.6),
    "one_half_col": (5.5, 3.5),
    "double_col":   (7.2, 4.0),
    "notebook":     (8.0, 5.0),
}

# ── Marker cycle for colour-blind safety ──────────────────────────────────
MARKER_CYCLE = ["o", "s", "^", "D", "v", "P", "X", "*"]

# ── Line-style cycle ──────────────────────────────────────────────────────
LINESTYLE_CYCLE = ["-", "--", "-.", ":", (0, (3, 1, 1, 1))]

# ---------------------------------------------------------------------------
# rcParams dictionary (the single source of truth)
# ---------------------------------------------------------------------------

ACADEMIC_RCPARAMS: dict = {
    # ── Font ──
    "font.family":       "serif",
    "font.serif":        ["Times New Roman", "Times", "DejaVu Serif", "serif"],
    "font.size":         9,
    "mathtext.fontset":  "stix",

    # ── Axes ──
    "axes.titlesize":    10,
    "axes.titleweight":  "bold",
    "axes.labelsize":    9,
    "axes.labelweight":  "normal",
    "axes.linewidth":    0.8,
    "axes.spines.top":   False,
    "axes.spines.right": False,

    # ── Ticks ──
    "xtick.labelsize":   8,
    "ytick.labelsize":   8,
    "xtick.direction":   "out",
    "ytick.direction":   "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size":  3.5,
    "ytick.major.size":  3.5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,

    # ── Legend ──
    "legend.fontsize":      8,
    "legend.frameon":       False,
    "legend.handlelength":  1.5,
    "legend.handletextpad": 0.5,
    "legend.borderpad":     0.4,

    # ── Lines ──
    "lines.linewidth":  1.4,
    "lines.markersize": 5,

    # ── Grid ──
    "axes.grid":     False,
    "grid.alpha":    0.3,
    "grid.linewidth": 0.5,
    "grid.color":    "#cccccc",

    # ── Figure ──
    "figure.dpi":       150,
    "figure.facecolor": "white",
    "figure.constrained_layout.use": True,

    # ── Saving ──
    "savefig.dpi":       300,
    "savefig.bbox":      "tight",
    "savefig.pad_inches": 0.03,
    "savefig.facecolor": "white",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def setup_academic_style() -> None:
    """Apply the project-wide CVPR-standard academic figure style.

    Must be called at the beginning of every figure_* function or notebook cell
    that produces figures.  See ``docs/plotting_standards.md`` for the full
    specification.
    """
    plt.rcParams.update(ACADEMIC_RCPARAMS)


def savefig_academic(
    fig: plt.Figure,
    path: str | Path,
    *,
    dpi: int = 300,
    close: bool = True,
) -> Path:
    """Save a figure following the project saving conventions.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to save.
    path : str or Path
        Destination file path.  Parent directories are created automatically.
        Use ``.pdf`` for paper submission, ``.png`` for notebook display.
    dpi : int, optional
        Resolution in dots per inch.  Must be ≥ 300 for paper figures.
    close : bool, optional
        If *True* (default), close the figure after saving to free memory.

    Returns
    -------
    Path
        The resolved path to the saved file.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    if close:
        plt.close(fig)
    return path


def get_method_style(
    index: int,
) -> dict:
    """Return a consistent (color, marker, linestyle) dict for method *index*.

    Use this to cycle through visual styles when plotting multiple methods,
    ensuring colour-blind safety (distinct markers + linestyles alongside
    colours).

    Parameters
    ----------
    index : int
        Zero-based method index.

    Returns
    -------
    dict
        Keys: ``color``, ``marker``, ``linestyle``.
    """
    return {
        "color":     METHOD_COLOR_LIST[index % len(METHOD_COLOR_LIST)],
        "marker":    MARKER_CYCLE[index % len(MARKER_CYCLE)],
        "linestyle": LINESTYLE_CYCLE[index % len(LINESTYLE_CYCLE)],
    }


def add_reference_line(
    ax: plt.Axes,
    value: float,
    label: str,
    *,
    axis: Literal["x", "y"] = "y",
    color: str = "#666666",
    linewidth: float = 0.9,
) -> None:
    """Add a dashed reference / threshold line to *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    value : float
        Position of the reference line.
    label : str
        Legend label describing the line's meaning.
    axis : {"x", "y"}, optional
        Which axis the line is parallel to.
    color : str, optional
        Line colour.
    linewidth : float, optional
        Line width.
    """
    fn = ax.axhline if axis == "y" else ax.axvline
    fn(value, ls="--", color=color, linewidth=linewidth, label=label, zorder=1)


def format_colorbar(
    cbar,
    label: str,
    *,
    fontsize: int = 8,
) -> None:
    """Apply standard formatting to a colorbar.

    Parameters
    ----------
    cbar : matplotlib.colorbar.Colorbar
        The colorbar to format.
    label : str
        Descriptive label including units, e.g. ``"Temperature [°C]"``.
    fontsize : int, optional
        Label font size.
    """
    cbar.set_label(label, fontsize=fontsize)
    cbar.ax.tick_params(labelsize=fontsize)


def make_figure(
    layout: Literal["single_col", "one_half_col", "double_col", "notebook"] = "single_col",
    *,
    nrows: int = 1,
    ncols: int = 1,
    height: float | None = None,
    **subplot_kw,
) -> tuple[plt.Figure, np.ndarray | plt.Axes]:
    """Create a figure with standardised sizing.

    Parameters
    ----------
    layout : str
        One of the predefined layout keys in :data:`FIGURE_SIZES`.
    nrows, ncols : int
        Subplot grid dimensions.
    height : float or None
        Override the default height (inches).  Width is always from
        the layout preset.
    **subplot_kw
        Extra keyword arguments forwarded to ``plt.subplots``.

    Returns
    -------
    fig, axes
    """
    setup_academic_style()
    w, h = FIGURE_SIZES[layout]
    if height is not None:
        h = height
    # Scale height with number of rows
    if nrows > 1:
        h = h * nrows * 0.7  # slight compression per row
    fig, axes = plt.subplots(nrows, ncols, figsize=(w, h), **subplot_kw)
    return fig, axes
