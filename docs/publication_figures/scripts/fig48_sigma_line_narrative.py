"""fig48 — The sigma (PSF width) calibration line: a three-act narrative.

(a) EP09 three-route point calibration diverges wildly (spread ~1.01 LR px vs
    the +/-0.05 px agreement gate) -> FAIL; later explained by ACL-056: the
    self-supervised estimator family is nearly degenerate in sigma (each route
    breaks the degeneracy with a different implicit prior).
(b) E3 multi-frame projected-ESF kernel (parametric straight-edge scene prior,
    ACL-057) breaks the degeneracy; after the bench ground-truth fix (ACL-058:
    scene edge_sigma folded in quadrature), the 48-scene synthetic bench
    re-verdict PASSES: median |rel err| 4.1% (27 evaluable scenes; 3.1%
    gaussian-only) vs the pre-registered 15% tolerance, no systematic bias.
(c) Real 248-frame data (ACL-059): the frozen estimator detects 8 candidate
    straight edges but 0/8 pass the pre-registered quality gate
    (r^2 >= 0.90 AND amp_SNR >= 5.0) -> legitimate rejection: sigma is not
    self-calibratable on this target -> strategy pivots from point calibration
    to a robust band sigma in [~0.1, ~0.4] LR px.

Sources (numbers transcribed, none recomputed):
- research_log/episodes/ep09_psf_calibration/README.md
    Route A forward-residual sigma = 0.2257 LR px
    Route B ESF fitting        sigma = 1.1286 LR px
    Route C joint MAP-TV       sigma = 0.1190 LR px
    spread ~1.010 px vs +/-0.05 px agreement gate -> FAIL
- research_log/algorithm_changelog.md ACL-056/057/058/059
- output/sigma_esf_bench_reverdict/bench_verdict.json
    median_abs_rel_err = 0.04090 (n=27), gaussian-only 0.03053,
    median_tol = 0.15, prereg_pass = true
- remote_inbox/20260712_sigma/sigma_esf_real/real248_esf_summary.json
    8 edges detected, 0 valid; per-edge (r2, amp_snr) plotted in (c)

Outputs:
- docs/publication_figures/figures/fig48_sigma_line_narrative.{png,pdf}
- docs/publication_figures/data/sigma_line.csv

Run from repo root:
    uv run python docs/publication_figures/scripts/fig48_sigma_line_narrative.py
"""

from __future__ import annotations

import csv

import matplotlib.pyplot as plt
import numpy as np

from pubfig_style import (
    DATA_DIR,
    METHOD_PALETTE,
    REF_LINE,
    W_DOUBLE,
    save_fig,
    setup_academic_style,
)

# ── Transcribed data ─────────────────────────────────────────────────

# (a) EP09 three-route point estimates [LR px]
ROUTES = [
    ("A", "Forward residual", 0.2257),
    ("B", "ESF fitting", 1.1286),
    ("C", "Joint MAP-TV", 0.1190),
]
GATE_HALF_WIDTH = 0.05          # +/- px agreement gate
ROUTE_SPREAD = 1.010            # max-min spread [px]

# (b) E3 synthetic bench re-verdict (ACL-058)
BENCH_MEDIAN_ALL = 4.09         # median |rel err| [%], 27 evaluable scenes
BENCH_MEDIAN_GAUSS = 3.052      # gaussian-only subset [%]
BENCH_TOL = 15.0                # pre-registered tolerance [%]

# (c) Real 248-frame Step 2 per-edge quality (ACL-059)
REAL_EDGES = [  # (edge_id, r2, amp_snr)
    (0, 0.7491, 10.65),
    (1, 0.6858, 8.78),
    (2, 0.7564, 11.68),
    (3, 0.6945, 9.37),
    (4, 0.7657, 13.74),
    (5, 0.1731, 2.37),
    (6, 0.1492, 2.60),
    (7, 0.6702, 11.25),
]
GATE_R2 = 0.90
GATE_SNR = 5.0

BLUE = METHOD_PALETTE["primary"]
GREEN = METHOD_PALETTE["secondary"]
RED = METHOD_PALETTE["accent_1"]


def write_csv() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rows = [
        ("ep09_route_A_forward_residual_sigma_lr_px", 0.2257, "EP09 README"),
        ("ep09_route_B_esf_sigma_lr_px", 1.1286, "EP09 README"),
        ("ep09_route_C_joint_maptv_sigma_lr_px", 0.1190, "EP09 README"),
        ("ep09_agreement_gate_half_width_px", GATE_HALF_WIDTH, "EP09 README"),
        ("ep09_route_spread_px", ROUTE_SPREAD, "EP09 README (FAIL)"),
        ("e1e2_degeneracy_verdict", "sigma nearly unidentifiable", "ACL-056"),
        ("e3_bench_median_abs_rel_err_pct", BENCH_MEDIAN_ALL,
         "ACL-058 / output/sigma_esf_bench_reverdict/bench_verdict.json (n=27, PASS)"),
        ("e3_bench_median_abs_rel_err_gaussian_only_pct", BENCH_MEDIAN_GAUSS,
         "ACL-058 / bench_verdict.json"),
        ("e3_bench_tolerance_pct", BENCH_TOL, "ACL-057 prereg"),
        ("real248_edges_detected", 8, "ACL-059 / real248_esf_summary.json"),
        ("real248_edges_pass_quality_gate", 0, "ACL-059 (legitimate rejection)"),
        ("quality_gate_r2_min", GATE_R2, "ACL-059 prereg"),
        ("quality_gate_amp_snr_min", GATE_SNR, "ACL-059 prereg"),
        ("robust_band_sigma_lo_lr_px", 0.1, "ACL-059 strategy pivot"),
        ("robust_band_sigma_hi_lr_px", 0.4, "ACL-059 strategy pivot"),
    ]
    for eid, r2, snr in REAL_EDGES:
        rows.append((f"real248_edge{eid}_r2", r2,
                     "remote_inbox/20260712_sigma/sigma_esf_real/real248_esf_summary.json"))
        rows.append((f"real248_edge{eid}_amp_snr", snr, "same"))
    with open(DATA_DIR / "sigma_line.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["quantity", "value", "source"])
        w.writerows(rows)


def panel_a(ax: plt.Axes) -> None:
    """EP09 three routes vs the +/-0.05 px agreement gate."""
    xs = np.arange(len(ROUTES))
    sigmas = [r[2] for r in ROUTES]
    med = float(np.median(sigmas))
    ax.axhspan(med - GATE_HALF_WIDTH, med + GATE_HALF_WIDTH,
               color="#bbbbbb", alpha=0.45, zorder=0,
               label="$\\pm$0.05 px agreement gate")
    markers = ["o", "s", "D"]
    for x, (tag, _name, s), m in zip(xs, ROUTES, markers):
        ax.plot(x, s, marker=m, ms=6, color=BLUE, ls="none", zorder=3)
        ax.annotate(f"{s:.3f}", (x, s), xytext=(0, 6),
                    textcoords="offset points", ha="center", fontsize=7)
    # spread bracket
    ax.annotate("", xy=(2.42, max(sigmas)), xytext=(2.42, min(sigmas)),
                arrowprops=dict(arrowstyle="<->", color=RED, lw=0.9))
    ax.text(2.30, 0.90, "spread $\\approx$ 1.01 px\n$\\rightarrow$ FAIL",
            color=RED, fontsize=7, ha="right", va="center")
    ax.set_xticks(xs)
    # break route names over two lines so neighbouring tick labels stay apart
    ax.set_xticklabels([f"{t}\n{n.replace(' ', chr(10))}" for t, n, _ in ROUTES],
                       fontsize=7)
    ax.set_xlim(-0.5, 2.55)
    ax.set_ylim(0, 1.38)
    ax.set_ylabel(r"$\hat{\sigma}$ [LR px]")
    ax.set_title("(a) EP09: three routes diverge")
    ax.legend(loc="upper left", fontsize=7)
    ax.text(0.02, 0.60, "root cause: self-supervised\nestimators degenerate in $\\sigma$\n(ACL-056)",
            transform=ax.transAxes, fontsize=7, color="#555555", va="top")


def panel_b(ax: plt.Axes) -> None:
    """E3 synthetic bench verdict (compact, no scene scatter)."""
    ax.axvspan(0, BENCH_TOL, color=GREEN, alpha=0.15, zorder=0,
               label="prereg tolerance ($\\leq$15%)")
    ax.axvline(BENCH_TOL, **REF_LINE)
    ys = [1, 0]
    vals = [BENCH_MEDIAN_ALL, BENCH_MEDIAN_GAUSS]
    labels = ["all evaluable\n(n = 27)", "gaussian-only\nsubset"]
    for y, v, m in zip(ys, vals, ["o", "^"]):
        ax.plot([0, v], [y, y], color=GREEN, lw=2.2, solid_capstyle="round")
        ax.plot(v, y, marker=m, ms=6, color=GREEN, ls="none")
        ax.annotate(f"{v:.1f}%", (v, y), xytext=(6, 5),
                    textcoords="offset points", fontsize=7)
    ax.text(0.71, 0.06, "PASS\n(ACL-058)", transform=ax.transAxes,
            fontsize=8, color=GREEN, ha="right", va="bottom", fontweight="bold")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_ylim(-0.7, 1.8)
    ax.set_xlim(0, 20)
    ax.set_xlabel(r"median $|\hat{\sigma}/\sigma_{\rm true}-1|$ [%]")
    ax.set_title("(b) E3 kernel: synthetic bench")
    ax.legend(loc="upper left", fontsize=7)


def panel_c(ax: plt.Axes) -> None:
    """Real 248-frame Step 2: 0/8 edges pass the quality gate."""
    ax.axvline(GATE_SNR, **REF_LINE)
    ax.axhline(GATE_R2, **REF_LINE)
    # pass region (upper-right quadrant)
    ax.axhspan(GATE_R2, 1.02, xmin=(GATE_SNR - 0) / 16.0,
               color=GREEN, alpha=0.12, zorder=0)
    ax.text(15.6, 0.985, "pass region (0/8)", fontsize=7, ha="right",
            va="top", color=GREEN)
    r2s = [e[1] for e in REAL_EDGES]
    snrs = [e[2] for e in REAL_EDGES]
    ax.plot(snrs, r2s, marker="x", ms=6, mew=1.4, color=RED, ls="none",
            label="detected edge (fail)")
    ax.text(0.34, 0.875, "gate: $r^2 \\geq 0.90$ & amp SNR $\\geq 5.0$",
            transform=ax.transAxes, fontsize=6.5, va="top", ha="left",
            color="#222222")
    ax.annotate("$\\sigma$ point-calibration rejected\n$\\rightarrow$ robust band\n"
                "$\\sigma \\in [0.1, 0.4]$ px",
                xy=(0.985, 0.32), xycoords="axes fraction", ha="right", va="center",
                fontsize=7, color="black", fontweight="bold")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("edge amplitude SNR")
    ax.set_ylabel(r"ESF fit $r^2$")
    ax.set_title("(c) Real 248-frame: 0/8 edges")
    ax.legend(loc="lower right", fontsize=7)


def main() -> None:
    setup_academic_style()
    write_csv()
    # 1x3 row of equal square panels; width_ratios only absorb the differing
    # y-decoration widths so the visible boxes stay equal and evenly spaced.
    fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 2.6),
                             gridspec_kw=dict(width_ratios=[1.0, 1.06, 1.08]))
    for ax in axes:
        ax.set_box_aspect(1.0)
    panel_a(axes[0])
    panel_b(axes[1])
    panel_c(axes[2])
    fig.suptitle("The $\\sigma$ (PSF width) calibration line: divergence "
                 "$\\rightarrow$ degeneracy broken $\\rightarrow$ legitimate rejection",
                 fontsize=11, fontweight="bold")
    paths = save_fig(fig, "fig48_sigma_line_narrative")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
