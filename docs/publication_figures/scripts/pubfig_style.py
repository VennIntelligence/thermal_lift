"""Shared publication-figure style for thermal_lift.

Implements docs/plotting_standards.md (CVPR, serif/Times, 300 dpi).
Every figure script in this directory imports from here; do NOT hardcode
font sizes inside individual figure functions.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# ── Canonical directories ────────────────────────────────────────────
PKG_DIR = Path(__file__).resolve().parent.parent          # docs/publication_figures
FIG_DIR = PKG_DIR / "figures"
DATA_DIR = PKG_DIR / "data"
REPO_ROOT = PKG_DIR.parent.parent                          # thermal_lift repo root

# ── CVPR column widths [inch] ────────────────────────────────────────
W_SINGLE = 3.5     # single column
W_1P5 = 5.5        # 1.5 column
W_DOUBLE = 7.2     # double column (never exceed 7.1 in final paper)

ACADEMIC_RCPARAMS = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif", "serif"],
    "font.size": 9,
    "mathtext.fontset": "stix",
    "axes.titlesize": 10,
    "axes.titleweight": "bold",
    "axes.labelsize": 9,
    "axes.labelweight": "normal",
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 3.5,
    "ytick.major.size": 3.5,
    "xtick.minor.visible": False,
    "ytick.minor.visible": False,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.5,
    "legend.borderpad": 0.4,
    "lines.linewidth": 1.4,
    "lines.markersize": 5,
    "axes.grid": False,
    "grid.alpha": 0.3,
    "grid.linewidth": 0.5,
    "grid.color": "#cccccc",
    "figure.dpi": 300,
    "figure.facecolor": "white",
    "figure.constrained_layout.use": True,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
    "savefig.facecolor": "white",
}

METHOD_PALETTE = {
    "primary": "#4C72B0",    # steel blue
    "secondary": "#55A868",  # muted green
    "accent_1": "#C44E52",   # soft red
    "accent_2": "#8172B2",   # lavender purple
    "accent_3": "#DD8452",   # warm orange
    "neutral": "#937860",    # taupe
}

# Fixed method → (color, marker, linestyle) mapping used across ALL figures
# so the same arm always looks the same. Extend here, never ad hoc.
METHOD_STYLE = {
    "tgv":        dict(color="#333333", marker="s", ls="-",  label="TGV $\\times$ drizzle"),
    "tgv_oracle": dict(color="#333333", marker="s", ls="-",  label="TGV (oracle)"),
    "tgv_portable": dict(color="#888888", marker="s", ls="--", label="TGV (portable)"),
    "maptv":      dict(color="#937860", marker="D", ls="-",  label="MAP-TV $\\times$ drizzle"),
    "drizzle":    dict(color="#DD8452", marker="^", ls="-",  label="Drizzle"),
    "depb9v6":    dict(color="#4C72B0", marker="o", ls="-",  label="Ours (v6 pool)"),
    "depb9v9_9bin": dict(color="#C44E52", marker="v", ls="-", label="Ours (v9 pool, 9-bin)"),
    "depb9v9_3k": dict(color="#55A868", marker="P", ls="-",  label="Ours (v9 pool, 3k)"),
    "depb9v8":    dict(color="#8172B2", marker="X", ls="-",  label="Ours (v8 pool)"),
}

# Colormaps per plotting_standards.md
CMAP_TEMPERATURE = "inferno"   # temperature / HR image
CMAP_RESID_POS = "YlOrRd"      # one-sided residual >= 0
CMAP_RESID_DIV = "RdBu_r"      # two-sided diff
CMAP_COVERAGE = "viridis"      # coverage / counts

REF_LINE = dict(color="#222222", ls="--", lw=0.9)
REF_LINE_GRAY = dict(color="#666666", ls="--", lw=0.9)


def setup_academic_style() -> None:
    """Apply the repo-wide academic rcParams. Call once per script."""
    mpl.use("Agg")
    mpl.rcParams.update(ACADEMIC_RCPARAMS)


def save_fig(fig: plt.Figure, name: str, formats: tuple[str, ...] = ("png", "pdf")) -> list[Path]:
    """Save into figures/ as png (preview) + pdf (submission vector)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in formats:
        path = FIG_DIR / f"{name}.{ext}"
        fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
        out.append(path)
    plt.close(fig)
    return out


def ylabel_with_unit(desc: str, unit: str) -> str:
    return f"{desc} [{unit}]"


# ── Checkpoint-strip montage template (shared by fig12 / fig28) ──────────────
def strip_montage(
    panels,
    *,
    col_titles=None,
    row_labels=None,
    cmap=None,
    vmin=None,
    vmax=None,
    cbar_label=None,
    fig_width: float = W_DOUBLE,
    panel_aspect: float = 1.0,
    wspace: float = 0.02,
    hspace: float = 0.02,
    title_fontsize: float = 9.0,
    label_fontsize: float = 9.0,
    note: str | None = None,
    note_fontsize: float = 6.5,
):
    """Dense image-strip montage: a grid of image panels + row/column labels.

    Shared template for checkpoint-evolution strips (fig12, fig28 and any
    future ones) so all strips keep identical row-label / step-title /
    colorbar styling and tight panel spacing.

    Parameters
    ----------
    panels : nested list, ``panels[row][col] -> ndarray``. 2-D arrays are
        drawn with ``cmap``/``vmin``/``vmax`` (scalar data, e.g. absolute
        temperature); 3-D arrays are shown as-is (pre-rendered RGB exports).
    col_titles : per-row list of per-panel titles (rows may use different
        checkpoint steps, so titles are per panel, not per grid column).
    row_labels : one label per row, placed as the leftmost ylabel.
    cbar_label : if given (scalar panels only), one shared colorbar is
        appended on the right, driven by the last scalar mappable.
    panel_aspect : width/height of one image panel; used to size the figure
        so panels sit flush (minimal letterboxing) at the given spacing.
    wspace, hspace : constrained-layout panel gaps (fraction of axes size).
    note : optional small display-convention footnote under the strip (e.g.
        when panels are pre-rendered exports without a shared scale).

    Returns ``(fig, axes)`` (axes is a 2-D array) so callers can add scale
    bars or other overlays before ``save_fig``.
    """
    nrows = len(panels)
    ncols = max(len(row) for row in panels)

    # Estimate the figure height that makes image panels fill their slots at
    # the requested width: colorbar and row labels eat horizontal space, while
    # per-panel titles and the optional footnote eat vertical space.
    cbar_w_frac = 0.055 if cbar_label else 0.0
    label_w_in = 0.14 if row_labels else 0.0
    panel_w = (fig_width * (1.0 - cbar_w_frac) - label_w_in) / ncols
    panel_h = panel_w / panel_aspect
    title_h = 0.24 if col_titles else 0.0
    note_h = 0.18 if note else 0.0
    fig_h = nrows * (panel_h + title_h) + note_h

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_h), squeeze=False)
    fig.get_layout_engine().set(wspace=wspace, hspace=hspace, w_pad=0.01, h_pad=0.01)

    mappable = None
    for r, row in enumerate(panels):
        for c in range(ncols):
            ax = axes[r][c]
            if c >= len(row):
                ax.set_axis_off()
                continue
            img = row[c]
            if img.ndim == 2:
                mappable = ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax,
                                     interpolation="nearest")
            else:
                ax.imshow(img)
            ax.set_xticks([])
            ax.set_yticks([])
            for sp in ax.spines.values():
                sp.set_visible(False)
            if col_titles is not None and c < len(col_titles[r]):
                ax.set_title(col_titles[r][c], fontsize=title_fontsize, pad=2.5)
        if row_labels is not None:
            axes[r][0].set_ylabel(row_labels[r], fontsize=label_fontsize)

    if cbar_label is not None and mappable is not None:
        cbar = fig.colorbar(
            mappable, ax=[ax for row_axes in axes for ax in row_axes],
            fraction=0.032, pad=0.015,
        )
        cbar.set_label(cbar_label, fontsize=label_fontsize)

    if note is not None:
        fig.supxlabel(note, fontsize=note_fontsize, color="#555555",
                      x=0.01, ha="left")

    return fig, axes
