"""Fig 13 — Self-supervised sigma estimator (E3) bench re-verdict (ACL-056/057/058).

Context: self-supervised PSF-sigma estimators E1/E2 were found to be nearly
degenerate in sigma (ACL-056/057, negative finding) -- they cannot distinguish
scenes with different true blur width. E3, a multi-frame projected-ESF
(edge-spread-function) kernel estimator, breaks this degeneracy. After fixing
the benchmark ground truth to the quadrature edge-sigma (combining PSF-only
sigma and the scene's intrinsic edge sigma in quadrature), the ACL-058
re-verdict PASSes: median |sigma_hat - sigma_true| / sigma_true ~= 4.1% over
27 scenes with usable edges (21/48 scenes had no usable edges and are
excluded, shown separately as a bar). (a) Estimated vs. true sigma scatter,
colored by PSF shape, with the y=x reference line and the prereg pass/fail
gate shaded. (b) Relative-error distribution per scene, sorted, against the
15% prereg tolerance line, plus the excluded/no-usable-edge scene count.

Data: output/sigma_esf_bench_reverdict/{bench_rows.csv,bench_verdict.json}
Run:  cd /Users/ujs/mycode/thermal_lift && uv run python docs/publication_figures/scripts/fig13_sigma_selfcal_bench.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import METHOD_PALETTE, REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

SRC = REPO_ROOT / "output/sigma_esf_bench_reverdict"
rows = pd.read_csv(SRC / "bench_rows.csv")
verdict = json.loads((SRC / "bench_verdict.json").read_text())

MEDIAN_TOL = verdict["median_tol"]  # 0.15
MEDIAN_ERR = verdict["median_abs_rel_err"]  # ~0.041
N_NO_EDGE = verdict["n_no_edge_scenes"]
N_TOTAL = verdict["n_scenes_total"]

ok = rows[rows["status"] == "ok"].copy()
ok = ok.sort_values("abs_rel_err")

SHAPE_STYLE = {
    "gaussian": dict(color=METHOD_PALETTE["primary"], marker="o", label="Gaussian"),
    "elliptical_gaussian": dict(color=METHOD_PALETTE["accent_1"], marker="^", label="Elliptical Gaussian"),
    "airy_disk": dict(color=METHOD_PALETTE["accent_3"], marker="s", label="Airy disk"),
}

fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(W_DOUBLE, 3.0), gridspec_kw=dict(width_ratios=[1.05, 1.0]))

# ── (a) sigma_hat vs sigma_true scatter ────────────────────────────────
lo = min(ok["sigma_true"].min(), ok["sigma_hat_esf"].min()) * 0.9
hi = max(ok["sigma_true"].max(), ok["sigma_hat_esf"].max()) * 1.08

# +/-15% prereg tolerance band around y = x
xs = np.linspace(lo, hi, 200)
ax_a.fill_between(xs, xs * (1 - MEDIAN_TOL), xs * (1 + MEDIAN_TOL),
                   color="#4C9A2A", alpha=0.10, zorder=1,
                   label=f"$\\pm${MEDIAN_TOL:.0%} tol.")
ax_a.plot(xs, xs, **{"color": "#222222", "ls": "--", "lw": 0.9}, zorder=2, label="$y=x$")

for shape, sty in SHAPE_STYLE.items():
    sub = ok[ok["psf_shape"] == shape]
    if len(sub) == 0:
        continue
    ax_a.errorbar(
        sub["sigma_true"], sub["sigma_hat_esf"],
        yerr=[sub["sigma_hat_esf"] - sub["ci_lo"], sub["ci_hi"] - sub["sigma_hat_esf"]],
        fmt=sty["marker"], color=sty["color"], mec="#333333", mew=0.4,
        ecolor=sty["color"], elinewidth=0.6, capsize=1.5, alpha=0.85,
        ms=5, zorder=3, label=sty["label"])

ax_a.set_xlim(lo, hi)
ax_a.set_ylim(lo, hi)
ax_a.set_aspect("equal", adjustable="box")
ax_a.set_xlabel(r"True $\sigma$ (quadrature edge-sigma) [px]")
ax_a.set_ylabel(r"Estimated $\hat\sigma$ (E3 ESF kernel) [px]")
ax_a.set_title("(a) Estimated vs. true $\\sigma$", loc="left")
ax_a.legend(loc="upper left", fontsize=6.6, ncol=1, handlelength=1.2)

# ── (b) sorted relative error per scene ─────────────────────────────────
n = len(ok)
xr = np.arange(1, n + 1)
colors_b = [SHAPE_STYLE[s]["color"] for s in ok["psf_shape"]]
ax_b.bar(xr, ok["abs_rel_err"] * 100, color=colors_b, width=0.75,
         edgecolor="#333333", linewidth=0.3, zorder=3)
ax_b.axhline(MEDIAN_ERR * 100, color=METHOD_PALETTE["primary"], ls="-", lw=1.1, zorder=4,
             label=f"median = {MEDIAN_ERR * 100:.1f}%")
ax_b.axhline(MEDIAN_TOL * 100, color="#666666", ls="--", lw=0.9, zorder=4,
             label=f"prereg tol. = {MEDIAN_TOL * 100:.0f}%")
ax_b.set_xlabel("Scene rank (sorted by relative error)")
ax_b.set_ylabel(r"$|\hat\sigma-\sigma_{\rm true}|/\sigma_{\rm true}$ [%]")
title_status = "PASS" if verdict["prereg_pass"] else "FAIL"
ax_b.set_title(f"(b) Relative error, prereg {title_status}", loc="left")
ax_b.legend(loc="upper left", fontsize=7)
ax_b.annotate(
    f"{N_NO_EDGE}/{N_TOTAL} scenes\nexcluded\n(no usable edges)",
    xy=(0.98, 0.95), xycoords="axes fraction", ha="right", va="top",
    fontsize=6.8, color="#666666",
    bbox=dict(fc="white", ec="#cccccc", lw=0.5, pad=1.6))

paths = save_fig(fig, "fig13_sigma_selfcal_bench")
print("\n".join(str(p) for p in paths))
