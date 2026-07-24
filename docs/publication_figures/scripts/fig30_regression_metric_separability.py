"""fig30: Which 0d regression-suite metrics separate good from bad reconstructions?

Finding (ACL-046 Task B / ACL-047 Task 3, Stage 0f): of the seven scalar
metrics in the 0d solver regression suite, ONLY the tiled/full-halo
extent-consistency probe separates known-good from known-bad arms with a
usable margin (bad/good-max ratio 2.4-2.9x). The flat-ROI and beading
metrics technically place the bad arm above the good ones, but the margin
(1.06-1.15x) is comparable to the spread WITHIN the good arms and rests on
a single bad case, so they carry no reliable discriminative power. The
seam-spectrum probe is inverted: the bad arm scores slightly LOWER than
both good arms. Consequence (changelog quick-truth #6): extent is the only
candidate hard gate; flat/beading are trend-tracking at best; seam should
be retired or redesigned.

Probe/metric definitions (inferred from suite source + REPORT.md):
- flat_roi_artifact: low-pass residual structure injected into a flat
  (featureless) ROI; p95 |delta| and std of delta, in deg C.
- tiled_full_halo_extent_consistency ("extent probe"): agreement between
  tiled-with-halo and full-frame inference of the same scene; NRMSE vs the
  tiled std, and p95 |difference| in deg C. Sensitive to context-dependent
  (non-local) solver behaviour.
- seam_spectrum: max |autocorrelation| of the tile-seam signal
  (periodic tiling artifact detector).
- beading_probe: edge over-sharpening ("beading"): edge signal ratio vs a
  reference reconstruction, and p95 edge excess in deg C.

Arms: good = v11_40k, promptA_5k (n=2); bad = v8k4 (n=1; checkpoint
load_state_dict failed on the remote runner, scores computed from the
pre-rendered diagnostic arrays stored alongside it - provenance degraded,
recorded in ACL-047).

Data (verbatim, no recomputation):
- remote_inbox/20260704_stage0f/t3_regression_metric_values.csv
  (per-case metric values)
- remote_inbox/20260704_stage0f/t3_regression_metric_separability.csv
  (good/bad ranges, separation ratio = bad_min / good_max)
Context: remote_inbox/20260704_stage0f/REPORT.md Task 3 (note: its claim
that all suggested thresholds are drop-in replacements was corrected in
ACL-047 - only extent holds up).

Layout: one row per metric, grouped by probe; x = metric value normalized
by the good-arm maximum of that metric (so the dashed line at 1.0 is the
good-arm envelope and the bad marker's abscissa IS the separation ratio).
Good arms: blue circles; bad arm: red square. Extent-probe rows are
shaded. Right margin annotates the ratio.

Run from repo root:
    uv run python docs/publication_figures/scripts/fig30_regression_metric_separability.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pubfig_style import (  # noqa: E402
    METHOD_PALETTE,
    REF_LINE,
    REPO_ROOT,
    W_1P5,
    save_fig,
    setup_academic_style,
)

DATA_DIR = REPO_ROOT / "remote_inbox" / "20260704_stage0f"
VALUES_CSV = DATA_DIR / "t3_regression_metric_values.csv"
SEP_CSV = DATA_DIR / "t3_regression_metric_separability.csv"

GOOD_C = METHOD_PALETTE["primary"]    # #4C72B0
BAD_C = METHOD_PALETTE["accent_1"]    # #C44E52

# Display order (top row first) and pretty names.
PROBE_LABEL = {
    "flat_roi_artifact": "Flat-ROI\nartifact",
    "tiled_full_halo_extent_consistency": "Extent\nprobe",
    "seam_spectrum": "Seam\nspectrum",
    "beading_probe": "Beading\nprobe",
}
METRIC_LABEL = {
    "lowpass_p95_abs_delta_c": "low-pass p95 $|\\Delta|$",
    "lowpass_std_delta_c": "low-pass std $\\Delta$",
    "nrmse_vs_tiled_std": "NRMSE vs tiled std",
    "p95_abs_diff_c": "p95 $|$diff$|$",
    "max_abs_autocorr": "max $|$autocorr$|$",
    "edge_signal_ratio_vs_reference": "edge signal ratio",
    "edge_excess_p95_c": "edge excess p95",
}
ROW_ORDER = [
    ("flat_roi_artifact", "lowpass_p95_abs_delta_c"),
    ("flat_roi_artifact", "lowpass_std_delta_c"),
    ("tiled_full_halo_extent_consistency", "nrmse_vs_tiled_std"),
    ("tiled_full_halo_extent_consistency", "p95_abs_diff_c"),
    ("seam_spectrum", "max_abs_autocorr"),
    ("beading_probe", "edge_signal_ratio_vs_reference"),
    ("beading_probe", "edge_excess_p95_c"),
]
EXTENT_PROBE = "tiled_full_halo_extent_consistency"


def main() -> None:
    setup_academic_style()

    vals = pd.read_csv(VALUES_CSV)
    sep = pd.read_csv(SEP_CSV).set_index(["probe", "metric"])

    fig, ax = plt.subplots(figsize=(W_1P5, 3.1))

    n = len(ROW_ORDER)
    yticks, ylabels = [], []
    for i, (probe, metric) in enumerate(ROW_ORDER):
        y = n - 1 - i  # top row first
        yticks.append(y)
        ylabels.append(METRIC_LABEL[metric])

        srow = sep.loc[(probe, metric)]
        gmax = float(srow["good_max"])
        sub = vals[(vals["probe"] == probe) & (vals["metric"] == metric)]
        good = sub[sub["class"] == "good"]["value"].to_numpy() / gmax
        bad = sub[sub["class"] == "bad"]["value"].to_numpy() / gmax
        ratio = float(srow["separation_ratio_badmin_over_goodmax"])

        # good-arm envelope (min..max) as a thin blue bar behind the dots
        ax.plot([good.min(), good.max()], [y, y], color=GOOD_C, lw=2.2,
                alpha=0.30, solid_capstyle="butt", zorder=2)
        ax.scatter(good, [y] * len(good), s=26, marker="o",
                   facecolor=GOOD_C, edgecolor="white", lw=0.4, zorder=4)
        # nudge the bad marker down when it would sit on top of a good dot
        bad_dy = -0.22 if min(abs(bad.min() - g) for g in good) < 0.04 else 0.0
        ax.scatter(bad, [y + bad_dy] * len(bad), s=30, marker="s",
                   facecolor=BAD_C, edgecolor="white", lw=0.4, zorder=4)

        # separation ratio annotation at right margin
        is_extent = probe == EXTENT_PROBE
        if ratio >= 1.0:
            txt = f"$\\times${ratio:.2f}"
        else:
            txt = f"$\\times${ratio:.2f} (inverted)"
        ax.text(3.30, y, txt, va="center", ha="left", fontsize=8,
                color="#222222" if is_extent else "#888888",
                fontweight="bold" if is_extent else "normal")

    # shade the extent-probe rows
    extent_rows = [n - 1 - i for i, (p, _) in enumerate(ROW_ORDER)
                   if p == EXTENT_PROBE]
    ax.axhspan(min(extent_rows) - 0.42, max(extent_rows) + 0.42,
               color=GOOD_C, alpha=0.07, zorder=0)
    ax.text(3.30, n - 0.40, "bad / good-max",
            ha="left", va="bottom", fontsize=7, color="#666666")

    # good-arm envelope reference line
    ax.axvline(1.0, **REF_LINE, zorder=1)

    ax.set_yticks(yticks)
    ax.set_yticklabels(ylabels)
    ax.set_ylim(-0.55, n - 0.45)
    ax.set_xlim(0.60, 3.25)
    ax.set_xlabel("Metric value / good-arm maximum (per metric)")
    ax.set_title("Only the extent probe separates good from bad arms")

    # probe group labels on far left
    for probe in dict.fromkeys(p for p, _ in ROW_ORDER):
        rows = [n - 1 - i for i, (p, _) in enumerate(ROW_ORDER) if p == probe]
        ax.text(-0.31, (min(rows) + max(rows)) / 2.0, PROBE_LABEL[probe],
                transform=ax.get_yaxis_transform(), ha="center", va="center",
                fontsize=8, color="#444444", fontstyle="italic")

    # legend below the axes so it never occludes data
    handles = [
        plt.Line2D([], [], ls="none", marker="o", color=GOOD_C,
                   label="good arms (v11_40k, promptA_5k; $n{=}2$)"),
        plt.Line2D([], [], ls="none", marker="s", color=BAD_C,
                   label="bad arm (v8k4; $n{=}1$)"),
        plt.Line2D([], [], color=REF_LINE["color"], ls=REF_LINE["ls"],
                   lw=REF_LINE["lw"], label="good-arm max"),
    ]
    ax.legend(handles=handles, loc="upper center",
              bbox_to_anchor=(0.5, -0.22), ncol=3, columnspacing=1.0)

    save_fig(fig, "fig30_regression_metric_separability")


if __name__ == "__main__":
    main()
