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
