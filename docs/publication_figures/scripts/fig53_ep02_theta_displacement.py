"""fig53 — EP02 displacement calibration: stage-rotation theta forest + visible-vs-commanded displacement.

The thermal detector is mounted rotated by theta relative to the XY translation
stage (configured theta = 47.6 deg, configs/stage_calibration.json; detector
pitch 20 um/px). EP02 validates the stage prior with time-adjacent frame pairs
and coordinate-pair NCC registration under raw / high-pass / gradient
preprocessing, on the 248 clean main-session frames (263 raw TXT).

(a) theta FOREST: per-pair theta estimates under the repo rotation model
    (coordinate_to_shift: image = R(-theta) . stage / pitch, so
    theta_hat = atan2(dY,dX) - angle_deg_y_up), one row per evidence tier x
    method. Row transitions (40 um ~ 2 px, the largest commanded moves) land
    at 40.5-41.5 deg; sub-pixel X-step regressions (0.1-0.2 px) at 30-34 deg;
    drift-contaminated Y-coordinate pairs at 35.6-37.3 deg. Per EP02's
    documented verdict, TXT NCC directions are smoke tests that do NOT
    independently pin theta; the configured 47.6 deg is covered only by the
    recorded AVI continuous-scan estimate (47.14 deg, 95% CI [46.36, 47.92],
    EP02 README 2026-05-17 "AVI theta independent verification"; those AVI
    artifacts are not part of this rebuild and the value is quoted, not
    recomputed).
(b) DISPLACEMENT evidence: per-pair visible projection along the commanded
    direction vs commanded prior magnitude (log-log). On the corrected
    20 um/px basis every time-adjacent move class sits on y = x
    (projection ratio 0.89-1.09) from 0.1 px (2 um) up to 2 px (40 um);
    Y-coordinate pairs (acquisition gap ~ 16 frames) blow up 3.2x at 2 um —
    thermal drift, which is why time adjacency is required.
(c) Projection ratio (visible / commanded) per method and move class with a
    ratio = 1 reference.

Data:  remote_inbox/20260715_ep02_recal/{time_adjacent_method_measurements,
       y_coordinate_method_measurements, time_adjacent_x_step_fit}.csv
       + cache_manifest.json (theta_deg, pixel_size_um cross-check).
       Cache rebuilt on the 5090 on 2026-07-15 (uv run python
       scripts/build_ep02_cache.py) after fixing an ep02.py query bug.
Ref:   research_log/episodes/ep02_displacement_calibration/README.md
Run:   uv run python docs/publication_figures/scripts/fig53_ep02_theta_displacement.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pubfig_style import (  # noqa: E402
    METHOD_PALETTE,
    REF_LINE,
    REF_LINE_GRAY,
    REPO_ROOT,
    W_DOUBLE,
    save_fig,
    setup_academic_style,
)

DATA_DIR = REPO_ROOT / "remote_inbox" / "20260715_ep02_recal"

# Fixed method -> (color, marker) mapping, consistent across all three panels.
METHODS = ["raw_ncc", "highpass_ncc", "gradient_ncc"]
METHOD_FMT = {
    "raw_ncc": dict(color=METHOD_PALETTE["primary"], marker="o", label="raw NCC"),
    "highpass_ncc": dict(color=METHOD_PALETTE["accent_3"], marker="s", label="high-pass NCC"),
    "gradient_ncc": dict(color=METHOD_PALETTE["secondary"], marker="^", label="gradient NCC"),
}

# Recorded EP02 AVI continuous-scan theta verification (gradient NCC, combined
# X+Y scans, 16 AVIs). Source: research_log/episodes/ep02_displacement_
# calibration/README.md, 2026-05-17 "AVI theta 独立验证" (avi_theta_result.json).
# QUOTED, not recomputed: the AVI artifacts are not part of the 2026-07-15
# rebuild (raw-data server offline; AVI is 8-bit rendered video, aux evidence).
AVI_THETA_MEAN = 47.14
AVI_THETA_CI = (46.36, 47.92)

# Small-annotation font size derived from the house style, never hardcoded.
ANNOT_FS = mpl.rcParams["font.size"] - 2


def wrap_deg(a: np.ndarray) -> np.ndarray:
    return (np.asarray(a, float) + 180.0) % 360.0 - 180.0


def theta_hat_deg(df: pd.DataFrame) -> pd.Series:
    """Per-pair theta under the repo model image = R(-theta) . stage.

    coordinate_to_shift (thermal_core.displacement) maps a stage command at
    angle phi_stage to an image (y-up) direction phi_stage - theta, so
    theta_hat = phi_stage - angle_deg_y_up. Matches fit_rotation_angle:
    the x-step medians reproduce theta_from_x_steps_deg to <0.3 deg.
    """
    phi_stage = np.degrees(np.arctan2(df["delta_Y_um"], df["delta_X_um"]))
    return pd.Series(wrap_deg(phi_stage - df["angle_deg_y_up"]), index=df.index)


def iqr(s: pd.Series) -> tuple[float, float]:
    return float(s.quantile(0.25)), float(s.quantile(0.75))


def main() -> None:
    setup_academic_style()

    manifest = json.loads((DATA_DIR / "cache_manifest.json").read_text())
    theta_ref = float(manifest["theta_deg"])
    pitch_um = float(manifest["pixel_size_um"])
    assert abs(theta_ref - 47.6) < 1e-9 and abs(pitch_um - 20.0) < 1e-9

    ta = pd.read_csv(DATA_DIR / "time_adjacent_method_measurements.csv")
    yc = pd.read_csv(DATA_DIR / "y_coordinate_method_measurements.csv")
    xfit = pd.read_csv(DATA_DIR / "time_adjacent_x_step_fit.csv").set_index("method_label")
    ta = ta[ta["method_label"].isin(METHODS)].copy()  # phase_corr degenerates on 0.1 px
    ta["theta_hat"] = theta_hat_deg(ta)
    yc["theta_hat"] = theta_hat_deg(yc)

    fig, (ax_a, ax_b, ax_c) = plt.subplots(
        1, 3, figsize=(W_DOUBLE, 3.15), gridspec_kw=dict(width_ratios=[1.30, 1.02, 0.95]))

    # ================= (a) theta forest =================
    # rows: (group ytick label, per-method [marker_x, (whisker_lo, whisker_hi)])
    groups: list[tuple[str, list[tuple[str, float, tuple[float, float], bool]]]] = []

    rt = ta[ta["move_type"] == "row_transition"]
    groups.append((
        "row transitions\n40 $\\mu$m $\\approx$ 2 px\n($n{=}15$/method)",
        [(m, float(g["theta_hat"].median()), iqr(g["theta_hat"]), True)
         for m in METHODS for g in [rt[rt["method_label"] == m]]],
    ))
    xs = ta[ta["move_type"] == "x_step"]
    groups.append((
        "X-step regression\n2–4 $\\mu$m $\\approx$ 0.1–0.2 px\n($n{=}231$/method)",
        [(m, float(xfit.loc[m, "theta_from_x_steps_deg"]),
          iqr(xs[xs["method_label"] == m]["theta_hat"]), True)
         for m in METHODS],
    ))
    groups.append((
        "Y-coordinate pairs\ngap $\\approx$ 16 frames\n($n{=}232$/method)",
        [(m, float(g["theta_hat"].median()), iqr(g["theta_hat"]), False)
         for m in METHODS for g in [yc[yc["method_label"] == m]]],
    ))

    # Build row positions: AVI row on top, then groups separated by gaps.
    y = 0.0
    rows = []  # (y, method|None, x, lo, hi, filled)
    y_avi = y
    rows.append((y_avi, None, AVI_THETA_MEAN, *AVI_THETA_CI, True))
    y -= 1.6
    ylab_pos, ylab_txt = [y_avi], ["AVI continuous scan\n(recorded, 16 AVIs)"]
    for label, items in groups:
        ys = []
        for m, x, (lo, hi), filled in items:
            rows.append((y, m, x, lo, hi, filled))
            ys.append(y)
            y -= 1.0
        ylab_pos.append(float(np.mean(ys)))
        ylab_txt.append(label)
        y -= 0.6

    ax_a.axvline(theta_ref, **REF_LINE, zorder=1)
    for yy, m, x, lo, hi, filled in rows:
        if m is None:  # AVI (recorded, quoted)
            ax_a.plot([lo, hi], [yy, yy], color="#888888", lw=1.1, zorder=3)
            ax_a.scatter([x], [yy], s=34, marker="D", facecolor="white",
                         edgecolor="#555555", lw=1.0, zorder=4)
            txt_c = "#555555"
        else:
            fmt = METHOD_FMT[m]
            ax_a.plot([lo, hi], [yy, yy], color=fmt["color"], lw=1.4,
                      alpha=0.9 if filled else 0.55,
                      ls="-" if filled else (0, (2.2, 1.4)), zorder=3)
            ax_a.scatter([x], [yy], s=26, marker=fmt["marker"],
                         facecolor=fmt["color"] if filled else "white",
                         edgecolor="white" if filled else fmt["color"],
                         lw=0.5 if filled else 0.9, zorder=4)
            txt_c = "#666666"
        ax_a.text(58.5, yy, f"{x:.1f}$^\\circ$", va="center", ha="right",
                  fontsize=ANNOT_FS, color=txt_c)

    ax_a.text(theta_ref + 0.6, y_avi + 1.05,
              f"configured $\\theta$ = {theta_ref:.1f}$^\\circ$",
              fontsize=ANNOT_FS, color=REF_LINE["color"], ha="left", va="bottom")
    ax_a.set_yticks(ylab_pos)
    ax_a.set_yticklabels(ylab_txt, fontsize=mpl.rcParams["ytick.labelsize"] - 1)
    ax_a.set_ylim(y + 0.4, y_avi + 1.35)
    ax_a.set_xlim(19, 59)
    ax_a.set_xlabel("Stage-rotation $\\theta$ estimate [deg]")
    ax_a.set_title("(a) $\\theta$ by estimation method", loc="left")

    # ================= (b) visible vs commanded =================
    rng = np.random.default_rng(53)
    ax_b.axline((0.08, 0.08), (3, 3), **REF_LINE, zorder=1)
    for df, filled in [(ta, True), (yc, False)]:
        for m in METHODS:
            g = df[df["method_label"] == m]
            fmt = METHOD_FMT[m]
            xj = g["ref_mag_px"] * 10 ** rng.uniform(-0.055, 0.055, len(g))
            ax_b.scatter(
                xj, g["parallel_px"], s=7, marker=fmt["marker"],
                facecolor=fmt["color"] if filled else "none",
                edgecolor="none" if filled else fmt["color"],
                lw=0.0 if filled else 0.5,
                alpha=0.30 if filled else 0.35, zorder=3)
    ax_b.set_xscale("log")
    ax_b.set_yscale("log")
    ax_b.set_xlim(0.062, 3.4)
    ax_b.set_ylim(0.062, 3.4)
    ticks = [0.1, 0.2, 0.5, 1, 2]
    for axis in (ax_b.xaxis, ax_b.yaxis):
        axis.set_major_locator(mpl.ticker.FixedLocator(ticks))
        axis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_locator(mpl.ticker.NullLocator())
    ax_b.set_xlabel(f"Commanded magnitude [px] ({pitch_um:.0f} $\\mu$m/px)")
    ax_b.set_ylabel("Visible projection [px]")
    ax_b.set_title("(b) Visible vs. commanded", loc="left")
    ax_b.annotate("2 $\\mu$m\nX-steps", (0.128, 0.091), fontsize=ANNOT_FS,
                  color="#444444", ha="left", va="center")
    ax_b.annotate("4 $\\mu$m", (0.2, 0.158), fontsize=ANNOT_FS,
                  color="#444444", ha="center", va="top")
    ax_b.annotate(
        "row transitions\n(40 $\\mu$m)", xy=(1.72, 2.12), xytext=(0.52, 2.05),
        fontsize=ANNOT_FS, color="#444444", ha="center", va="center",
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.7))
    ax_b.annotate(
        "Y-coordinate pairs:\ngap $\\approx$ 16 frames,\ndrift-dominated",
        xy=(0.105, 0.34), xytext=(0.24, 0.80), fontsize=ANNOT_FS, color="#444444",
        ha="left", va="center",
        arrowprops=dict(arrowstyle="-", color="#888888", lw=0.7))

    # ================= (c) projection ratio =================
    grp_defs = [
        ("2 $\\mu$m\nX", ta, (ta["move_type"] == "x_step") & (ta["delta_um"] == 2), True),
        ("4 $\\mu$m\nX", ta, (ta["move_type"] == "x_step") & (ta["delta_um"] == 4), True),
        ("40 $\\mu$m\nrow", ta, ta["move_type"] == "row_transition", True),
        ("4 $\\mu$m\nY-pair", yc, yc["delta_um"] == 4, False),
        ("2 $\\mu$m\nY-pair", yc, yc["delta_um"] == 2, False),
    ]
    ax_c.axhline(1.0, **REF_LINE, zorder=1)
    ax_c.axvspan(2.55, 4.55, color="#888888", alpha=0.10, zorder=0)
    for i, (glabel, df, sel, filled) in enumerate(grp_defs):
        for j, m in enumerate(METHODS):
            g = df[sel & (df["method_label"] == m)]["projection_ratio"]
            lo, hi = iqr(g)
            fmt = METHOD_FMT[m]
            xx = i + (j - 1) * 0.24
            ax_c.plot([xx, xx], [lo, hi], color=fmt["color"], lw=1.2, zorder=3)
            ax_c.scatter([xx], [g.median()], s=22, marker=fmt["marker"],
                         facecolor=fmt["color"] if filled else "white",
                         edgecolor="white" if filled else fmt["color"],
                         lw=0.5 if filled else 0.9, zorder=4)
    ax_c.set_yscale("log")
    ax_c.set_ylim(0.72, 4.6)
    yticks = [0.8, 1.0, 1.5, 2.0, 3.0, 4.0]
    ax_c.yaxis.set_major_locator(mpl.ticker.FixedLocator(yticks))
    ax_c.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v:g}"))
    ax_c.yaxis.set_minor_locator(mpl.ticker.NullLocator())
    ax_c.set_xticks(range(len(grp_defs)))
    ax_c.set_xticklabels([g[0] for g in grp_defs],
                         fontsize=mpl.rcParams["xtick.labelsize"] - 1)
    ax_c.set_xlim(-0.55, 4.55)
    ax_c.set_ylabel("Visible / commanded ratio")
    ax_c.set_title("(c) Projection ratio", loc="left")
    ax_c.text(1.0, 1.32, "time-adjacent:\n0.89–1.09", fontsize=ANNOT_FS,
              color="#444444", ha="center", va="bottom")
    ax_c.annotate("drift\n$\\times$3.2–3.4", (3.15, 3.3), fontsize=ANNOT_FS,
                  color=METHOD_PALETTE["accent_1"], ha="right", va="center")

    # ================= shared legend =================
    hollow = dict(markerfacecolor="white", markeredgewidth=0.9)
    handles = [
        plt.Line2D([], [], ls="none", marker=f["marker"], color=f["color"],
                   markeredgecolor="white", markeredgewidth=0.4, label=f["label"])
        for f in (METHOD_FMT[m] for m in METHODS)
    ]
    handles += [
        plt.Line2D([], [], ls="none", marker="o", color="#555555",
                   label="open marker = coordinate pairs (drift-flagged)", **hollow),
        plt.Line2D([], [], ls="none", marker="D", color="#555555",
                   label="AVI $\\theta$ (recorded 2026-05, 95% CI)", **hollow),
        plt.Line2D([], [], color=REF_LINE["color"], ls=REF_LINE["ls"],
                   lw=REF_LINE["lw"], label="reference ($\\theta{=}47.6^\\circ$; $y{=}x$; ratio 1)"),
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3,
               columnspacing=1.4, handletextpad=0.5)

    paths = save_fig(fig, "fig53_ep02_theta_displacement")
    print("\n".join(str(p) for p in paths))

    # Console cross-check summary (numbers quoted in README/GALLERY entries).
    print(f"\nconfigured theta = {theta_ref} deg, pitch = {pitch_um} um/px")
    for label, items in groups:
        for m, x, (lo, hi), _ in items:
            print(f"{label.splitlines()[0]:24s} {m:13s} {x:6.2f} deg  IQR [{lo:.2f}, {hi:.2f}]")
    print(f"AVI (recorded)           gradient      {AVI_THETA_MEAN:6.2f} deg  CI [{AVI_THETA_CI[0]:.2f}, {AVI_THETA_CI[1]:.2f}]")


if __name__ == "__main__":
    main()
