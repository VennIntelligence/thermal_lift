"""Paper figure F2 + supplementary figure S-F1: phase-stratified split-half FRC.

F2  (single column): main mean FRC curve with 1/7 and half-bit criteria plus
    the four-control panel; the 10-12 um rebound band is flagged as risk.
S-F1 (double column): full archive -- per-seed cutoffs, control curves with
    their own cutoffs, the period band table, and zero-coverage statistics.

Data sources (regenerate with algos/ep15_info_limit/scripts/run_m2_frc.py):
    output/ep15_info_limit/m2_frc/{frc_curve,frc_controls,frc_band_table,
    frc_repeats}.csv + frc_summary.json

Run from the repository root:
    uv run python scripts/paper_figures/fig02_frc.py
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

M2_DIR = PROJECT_ROOT / "output" / "ep15_info_limit" / "m2_frc"
OUT_DIR = PROJECT_ROOT / "output" / "paper_figures"

FREQ_MAX = 0.130  # cyc/um, i.e. periods >= ~7.7 um
REBOUND_BAND_UM = (10.0, 12.0)

CURVE_STYLE = {
    "main": dict(color=METHOD_COLORS["primary"], ls="-", label="Main split-half"),
    "positive_bicubic": dict(color=METHOD_COLORS["secondary"], ls="--", label="Positive ctrl (bicubic+noise)"),
    "negative_shift_shuffle": dict(color=METHOD_COLORS["accent_1"], ls="-.", label="Negative ctrl (shift shuffle)"),
    "drift_acquisition_half": dict(color=METHOD_COLORS["accent_3"], ls=":", label="Drift ctrl (acq. halves)"),
}

PERIOD_TICKS_UM = [20, 16, 14, 12, 10, 8]


def _load() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    curve = pd.read_csv(M2_DIR / "frc_curve.csv")
    controls = pd.read_csv(M2_DIR / "frc_controls.csv")
    band = pd.read_csv(M2_DIR / "frc_band_table.csv")
    repeats = pd.read_csv(M2_DIR / "frc_repeats.csv")
    summary = json.loads((M2_DIR / "frc_summary.json").read_text())
    return curve, controls, band, repeats, summary


def _clip(df: pd.DataFrame) -> pd.DataFrame:
    return df[(df["frequency_um_inv"] > 0) & (df["frequency_um_inv"] <= FREQ_MAX)]


def _period_axis(ax: plt.Axes) -> None:
    sec = ax.secondary_xaxis(
        "top", functions=(lambda f: f, lambda f: f)
    )
    sec.set_xticks([1.0 / p for p in PERIOD_TICKS_UM])
    sec.set_xticklabels([str(p) for p in PERIOD_TICKS_UM])
    sec.set_xlabel(r"Period [$\mu$m]")
    sec.tick_params(labelsize=8)


def _shade_rebound(ax: plt.Axes, label: bool = False) -> None:
    f_lo, f_hi = 1.0 / REBOUND_BAND_UM[1], 1.0 / REBOUND_BAND_UM[0]
    ax.axvspan(
        f_lo, f_hi, color="#bbbbbb", alpha=0.25, lw=0,
        label="10-12 $\\mu$m rebound (coverage/drift risk)" if label else None,
    )


def _panel_main(ax: plt.Axes, curve: pd.DataFrame, repeats: pd.DataFrame, summary: dict) -> None:
    c = _clip(curve)
    ax.plot(c["frequency_um_inv"], c["frc"], **CURVE_STYLE["main"])
    ax.plot(
        c["frequency_um_inv"], c["threshold_half_bit"],
        color="#444444", ls=":", lw=0.9, label="Half-bit criterion",
    )
    ax.axhline(1.0 / 7.0, color="#222222", ls="--", lw=0.9, label="1/7 criterion")

    f_c = summary["f_c_frequency_um_inv"]
    ax.axvline(f_c, color=METHOD_COLORS["primary"], ls="--", lw=0.9)
    ax.annotate(
        f"cutoff {summary['f_c_period_um']:.1f} $\\mu$m\n(std {summary['f_c_std_um']:.2f} $\\mu$m, 3 seeds)",
        xy=(f_c, 0.42), xytext=(0.0615, 0.52), fontsize=7.5,
        arrowprops=dict(arrowstyle="-", color="#666666", lw=0.6),
    )
    seed_fc = repeats["f_c_frequency_um_inv_1_7"].to_numpy()
    ax.plot(
        seed_fc, np.full_like(seed_fc, 1.0 / 7.0), marker="|", ls="none",
        color=METHOD_COLORS["primary"], markersize=7, label="Per-seed 1/7 cutoffs",
    )
    _shade_rebound(ax, label=True)

    ax.set_xlim(0, FREQ_MAX)
    ax.set_ylim(-0.05, 1.02)
    ax.set_ylabel("FRC")
    ax.legend(loc="lower left", fontsize=6.8)


def _panel_controls(ax: plt.Axes, controls: pd.DataFrame, mark_cutoffs: bool = False, summary: dict | None = None) -> None:
    for name, style in CURVE_STYLE.items():
        cc = _clip(controls[controls["curve"] == name])
        ax.plot(cc["frequency_um_inv"], cc["frc"], **style)
    ax.axhline(1.0 / 7.0, color="#222222", ls="--", lw=0.9, label="1/7 criterion")
    _shade_rebound(ax)
    if mark_cutoffs and summary is not None:
        ctl = summary["controls"]
        marks = [
            (1.0 / summary["f_c_period_um"], CURVE_STYLE["main"]["color"]),
            (1.0 / ctl["positive_control_fc_period_um"], CURVE_STYLE["positive_bicubic"]["color"]),
            (1.0 / ctl["negative_control_fc_period_um"], CURVE_STYLE["negative_shift_shuffle"]["color"]),
            (1.0 / ctl["drift_control_fc_period_um"], CURVE_STYLE["drift_acquisition_half"]["color"]),
        ]
        for f_c, color in marks:
            ax.axvline(f_c, color=color, ls="--", lw=0.8, alpha=0.85)
    ax.set_xlim(0, FREQ_MAX)
    ax.set_ylim(-0.45, 1.02)
    ax.set_xlabel(r"Spatial frequency [$\mu$m$^{-1}$]")
    ax.set_ylabel("FRC")
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.30),
        ncol=2, fontsize=6.8, columnspacing=1.0,
    )


def make_fig02(curve, controls, repeats, summary) -> plt.Figure:
    fig, axes = plt.subplots(2, 1, figsize=(3.5, 4.6), sharex=True)
    _panel_main(axes[0], curve, repeats, summary)
    _period_axis(axes[0])
    axes[0].set_title("(a) Split-half FRC, 248-frame clean session", fontsize=9)
    _panel_controls(axes[1], controls)
    axes[1].set_title("(b) Controls", fontsize=9)
    return fig


def _panel_band_table(ax: plt.Axes, band: pd.DataFrame) -> None:
    periods = band["period_um"].to_numpy()
    x = np.arange(len(periods))
    width = 0.2
    cols = [
        ("main_frc", "main"),
        ("positive_bicubic_frc", "positive_bicubic"),
        ("negative_shift_shuffle_frc", "negative_shift_shuffle"),
        ("drift_acquisition_half_frc", "drift_acquisition_half"),
    ]
    for i, (col, key) in enumerate(cols):
        style = CURVE_STYLE[key]
        ax.bar(
            x + (i - 1.5) * width, band[col], width,
            color=style["color"], label=style["label"].split(" (")[0],
        )
    ax.axhline(1.0 / 7.0, color="#222222", ls="--", lw=0.9)
    ax.axhline(0.0, color="#888888", lw=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{p:.0f}" for p in periods])
    ax.set_xlabel(r"Period [$\mu$m]")
    ax.set_ylabel("FRC")
    # Curve names already given in panel (b)'s legend; colors are shared.
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)


def _panel_zero_coverage(ax: plt.Axes, repeats: pd.DataFrame) -> None:
    x = np.arange(len(repeats))
    width = 0.38
    ax.bar(x - width / 2, repeats["zero_coverage_pct_a"], width,
           color=METHOD_COLORS["primary"], label="Half A")
    ax.bar(x + width / 2, repeats["zero_coverage_pct_b"], width,
           color=METHOD_COLORS["accent_2"], label="Half B")
    ax.set_xticks(x)
    ax.set_xticklabels([f"seed {s}" for s in repeats["seed"]])
    ax.set_ylabel("Zero-coverage fraction [%]")
    mean_zc = float(np.mean(repeats[["zero_coverage_pct_a", "zero_coverage_pct_b"]].to_numpy()))
    ax.axhline(mean_zc, color="#666666", ls="--", lw=0.9, label=f"Mean {mean_zc:.1f}%")
    ax.set_ylim(0, 40)
    ax.legend(fontsize=6.8)
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)


def make_figS01(curve, controls, band, repeats, summary) -> plt.Figure:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.4))
    _panel_main(axes[0, 0], curve, repeats, summary)
    _period_axis(axes[0, 0])
    axes[0, 0].set_xlabel(r"Spatial frequency [$\mu$m$^{-1}$]")
    axes[0, 0].set_title("(a) Main curve, criteria, per-seed cutoffs", fontsize=9)

    _panel_controls(axes[0, 1], controls, mark_cutoffs=True, summary=summary)
    _period_axis(axes[0, 1])
    axes[0, 1].set_title("(b) Control curves with 1/7 cutoffs", fontsize=9)

    _panel_band_table(axes[1, 0], band)
    axes[1, 0].set_title("(c) Band table (period-interpolated FRC)", fontsize=9)

    _panel_zero_coverage(axes[1, 1], repeats)
    axes[1, 1].set_title("(d) Fine-grid zero coverage per split half", fontsize=9)
    return fig


def main() -> None:
    setup_academic_style()
    curve, controls, band, repeats, summary = _load()

    fig = make_fig02(curve, controls, repeats, summary)
    savefig_academic(fig, OUT_DIR / "fig02_frc.png", close=False)
    savefig_academic(fig, OUT_DIR / "fig02_frc.pdf")
    print(f"saved {OUT_DIR / 'fig02_frc.png'} (+.pdf)")

    fig_s = make_figS01(curve, controls, band, repeats, summary)
    savefig_academic(fig_s, OUT_DIR / "figS01_frc_archive.png", close=False)
    savefig_academic(fig_s, OUT_DIR / "figS01_frc_archive.pdf")
    print(f"saved {OUT_DIR / 'figS01_frc_archive.png'} (+.pdf)")


if __name__ == "__main__":
    main()
