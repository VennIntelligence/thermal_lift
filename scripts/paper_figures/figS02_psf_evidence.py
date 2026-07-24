"""Supplementary figure S-F2: the three-route PSF evidence chain + M3 arbitration.

Panels
------
(a) Residual-sweep routes: EP09 Route A (forward residual, 248 frames) and
    Route C (joint hold-out) as relative residual curves over sigma.
(b) Route B apparent ESF width distribution (EP09 outer-border segments)
    with EP15 M3 multi-edge family medians -- apparent width = PSF (x) edge.
(c) M3 arbitration: FRC-shape fit score vs sigma with the adopted credible
    range sigma in [0.2, 0.5] LR px and all three route estimates marked.

Data sources:
    output/ep09_psf_calibration/{forward_residual_sweep,forward_residual_fine_sweep,
    joint_sigma_sweep,esf_sigma_distribution,route_sigma_summary}.csv
    output/ep15_info_limit/m3_sigma/{edge_summary,frc_shape_fit_scores}.csv
    output/ep15_info_limit/m3_sigma/sigma_summary.json

Run from the repository root:
    uv run python scripts/paper_figures/figS02_psf_evidence.py

输出: output/paper_figures/figS02_psf_evidence.{png,pdf}

历史定位: scripts/paper_figures/ 是 2026-06 时代的旧论文图脚本；现行权威图集见
docs/publication_figures/（每图一个脚本、自带规范）。

关联: EP09 / EP15
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from thermal_core.plotting import (
    METHOD_COLORS,
    savefig_academic,
    setup_academic_style,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
EP09_DIR = PROJECT_ROOT / "output" / "ep09_psf_calibration"
M3_DIR = PROJECT_ROOT / "output" / "ep15_info_limit" / "m3_sigma"
OUT_DIR = PROJECT_ROOT / "output" / "paper_figures"

ADOPTED_RANGE = (0.2, 0.5)  # LR px, M3 arbitration result

EDGE_FAMILY_LABELS = {
    "die_outer_border": "Outer border",
    "internal_metal_strong": "Strong interior",
    "steepest_temperature_edge": "Steepest edge",
}
EDGE_FAMILY_COLORS = {
    "die_outer_border": METHOD_COLORS["accent_1"],
    "internal_metal_strong": METHOD_COLORS["secondary"],
    "steepest_temperature_edge": METHOD_COLORS["accent_3"],
}


def _shade_adopted(ax: plt.Axes, label: bool = False) -> None:
    ax.axvspan(
        *ADOPTED_RANGE, color=METHOD_COLORS["primary"], alpha=0.12, lw=0,
        label=r"Adopted $\sigma \in [0.2, 0.5]$ LR px" if label else None,
    )


def _panel_residual_routes(ax: plt.Axes, routes: pd.DataFrame) -> None:
    coarse = pd.read_csv(EP09_DIR / "forward_residual_sweep.csv")
    fine = pd.read_csv(EP09_DIR / "forward_residual_fine_sweep.csv")
    joint = pd.read_csv(EP09_DIR / "joint_sigma_sweep.csv")

    fwd = pd.concat([coarse, fine], ignore_index=True)
    fwd_val = (
        fwd[fwd["split"] == "val"].groupby("sigma_lr_px", as_index=False)["median_mse"].mean()
        .sort_values("sigma_lr_px")
    )
    rel_a = fwd_val["median_mse"] / fwd_val["median_mse"].min()
    ax.plot(
        fwd_val["sigma_lr_px"], rel_a, color=METHOD_COLORS["primary"],
        marker="o", markersize=2.5, label="Route A: forward residual (val)",
    )

    joint = joint.sort_values("sigma_lr_px")
    rel_c = joint["holdout_mse"] / joint["holdout_mse"].min()
    ax.plot(
        joint["sigma_lr_px"], rel_c, color=METHOD_COLORS["accent_2"],
        marker="s", markersize=2.5, ls="--", label="Route C: joint hold-out",
    )

    row_a = routes[routes["route"] == "A_forward"].iloc[0]
    ax.axvline(row_a["sigma_lr_px"], color=METHOD_COLORS["primary"], ls=":", lw=0.9)
    ax.axvspan(
        row_a["ci95_low_lr_px"], row_a["ci95_high_lr_px"],
        color=METHOD_COLORS["primary"], alpha=0.28, lw=0,
        label=f"Route A min {row_a['sigma_lr_px']:.3f} (95% CI)",
    )

    x_max = max(fwd_val["sigma_lr_px"].max(), joint["sigma_lr_px"].max())
    ax.set_xlim(0.05, x_max + 0.05)
    ax.set_xlabel(r"PSF $\sigma$ [LR px]")
    ax.set_ylabel("Relative residual MSE (per route)")
    ax.legend(fontsize=6.8, loc="upper left")


def _panel_esf(ax: plt.Axes, routes: pd.DataFrame) -> None:
    esf = pd.read_csv(EP09_DIR / "esf_sigma_distribution.csv")
    edge = pd.read_csv(M3_DIR / "edge_summary.csv")

    ax.hist(
        esf["sigma_lr_px"], bins=np.arange(0.4, 2.2, 0.1),
        color=METHOD_COLORS["primary"], alpha=0.55, edgecolor="white",
        label=f"EP09 outer-border fits (n={len(esf)})",
    )
    for _, row in edge.iterrows():
        key = row["edge_type"]
        ax.axvline(
            row["sigma_total_median_lr_px"], color=EDGE_FAMILY_COLORS[key],
            ls="--", lw=1.1,
            label=f"{EDGE_FAMILY_LABELS[key]} median {row['sigma_total_median_lr_px']:.2f}",
        )
    sharpest = edge["sigma_total_min_lr_px"].min()
    ax.axvline(
        sharpest, color="#222222", ls=":", lw=1.1,
        label=f"Sharpest single edge {sharpest:.2f}",
    )
    _shade_adopted(ax)
    ax.text(
        float(np.mean(ADOPTED_RANGE)), 0.97, "adopted\n$\\sigma$ range",
        transform=ax.get_xaxis_transform(), ha="center", va="top",
        fontsize=6.8, color=METHOD_COLORS["primary"],
    )

    ax.set_xlim(0.0, 2.2)
    ax.set_xlabel(r"Apparent ESF $\sigma_{\mathrm{total}}$ [LR px]")
    ax.set_ylabel("Edge-segment count")
    ax.legend(fontsize=6.4, loc="upper right")
    ax.annotate(
        r"$\sigma_{\mathrm{total}}^2 = \sigma_{\mathrm{PSF}}^2 + w_{\mathrm{edge}}^2$",
        xy=(0.97, 0.48), xycoords="axes fraction", ha="right", fontsize=8,
    )


def _panel_arbitration(ax: plt.Axes, routes: pd.DataFrame) -> None:
    scores = pd.read_csv(M3_DIR / "frc_shape_fit_scores.csv").sort_values("sigma_lr_px")
    summary = json.loads((M3_DIR / "sigma_summary.json").read_text())

    ax.plot(
        scores["sigma_lr_px"], scores["shape_mse"], color=METHOD_COLORS["primary"],
        marker="o", label="M3 FRC-shape fit MSE",
    )
    best = summary["best_frc_fit_sigma"]
    best_mse = scores.loc[scores["sigma_lr_px"] == best, "shape_mse"].iloc[0]
    ax.annotate(
        f"best fit $\\sigma$={best:.1f}", xy=(best, best_mse),
        xytext=(best + 0.08, best_mse + 0.05), fontsize=7.5,
        arrowprops=dict(arrowstyle="-", color="#666666", lw=0.6),
    )
    _shade_adopted(ax, label=True)

    markers = {"A_forward": ("^", "Route A"), "B_esf": ("X", "Route B (apparent)"), "C_joint": ("D", "Route C")}
    for _, row in routes.iterrows():
        m, lbl = markers[row["route"]]
        ax.plot(
            row["sigma_lr_px"], 0.012, marker=m, ls="none", markersize=6,
            color=METHOD_COLORS["accent_1"], clip_on=False,
            label=f"{lbl} {row['sigma_lr_px']:.2f}",
        )

    ax.set_xlim(0.05, 1.25)
    ax.set_ylim(0, 0.62)
    ax.set_xlabel(r"PSF $\sigma$ [LR px]")
    ax.set_ylabel("FRC-shape fit MSE")
    ax.legend(fontsize=6.4, loc="upper left")


def main() -> None:
    setup_academic_style()
    routes = pd.read_csv(EP09_DIR / "route_sigma_summary.csv")

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7))
    _panel_residual_routes(axes[0], routes)
    axes[0].set_title("(a) Residual sweeps (Routes A, C)", fontsize=9)
    _panel_esf(axes[1], routes)
    axes[1].set_title("(b) Apparent ESF widths (Route B + M3)", fontsize=9)
    _panel_arbitration(axes[2], routes)
    axes[2].set_title("(c) M3 arbitration", fontsize=9)

    savefig_academic(fig, OUT_DIR / "figS02_psf_evidence.png", close=False)
    savefig_academic(fig, OUT_DIR / "figS02_psf_evidence.pdf")
    print(f"saved {OUT_DIR / 'figS02_psf_evidence.png'} (+.pdf)")


if __name__ == "__main__":
    main()
