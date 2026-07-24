"""Fig 37 -- Continuous retention vs. dot size + optical ground-truth subset (ACL-063 detail).

Companion to fig09_dot_probe_stratified.py: fig09 shows median retention in four
coarse size *strata*. This figure re-derives the same ACL-063 story directly from
per-dot data at full resolution -- a continuous running-median retention curve
against dot diameter for all six arms, with a light per-dot scatter underlay for
the two most informative arms (drizzle, which is flat, and depb9v6, the current
"ours" champion pool, which degrades) so the reader can see the spread the
strata average away. The 42 dots independently matched against the optical
micrograph (in_optical == True in per_dot.csv) are overlaid as black-edged
markers to show they sit on the same trend as the bulk population, i.e. the
probe's small-dot attenuation finding is not an artifact of the synthetic
depth-fit pipeline.

Data provenance:
  output/dot_probe/per_dot.csv         -- per-dot fwhm_diam_tgv_um (diameter,
                                           reference/TGV FWHM-equivalent) and
                                           retention_<arm> = depth_arm / depth_tgv
                                           for arm in {drizzle, v14, v19, de_pb9,
                                           depb9v6, meanDC}; in_optical flags the
                                           42-dot optical ground-truth subset.
  output/dot_probe/optical_subset.csv  -- same 42 rows, kept as a named export;
                                           read here only to cross-check the row
                                           count against per_dot.csv's in_optical
                                           flag (n=42 in both).
  output/dot_probe/retention_vs_size.png -- prior quick-look (all 6 arms, 1-px
                                           integer-bin medians over full scatter);
                                           used only to confirm arm ordering and
                                           x-range intent, not re-plotted here.

Method:
  Running median computed on a physical-diameter grid (window = +/-6 um,
  center-aligned), masked to NaN wherever the window holds fewer than 15 dots
  so the "continuous" curve never over-claims precision in the sparse tails.
  A bottom density strip reports the per-10-um-bin dot count (all dots, and the
  optical subset) so the reader can see exactly how much support each part of
  the curve has -- the ACL-063 finding softens toward the smallest diameters,
  where the strip shows support drops fastest.

Caveats:
  - "diameter" here is the TGV-fit FWHM-equivalent diameter (fwhm_diam_tgv_um),
    the same reference used by fig09's px bins (1 px = 10 um there); it is a
    fitted quantity, not the true injected dot diameter, so very small dots
    carry more relative sizing noise than large ones -- another reason the
    running median is suppressed below n=15 support.
  - Per-dot retention has heavy-tailed noise (drizzle's 99th pct is ~3.3);
    the y-axis is clipped to keep the trend readable, and the count of
    off-scale points is reported in the caption text, not hidden silently.
  - Optical-subset markers are drawn only against the drizzle/depb9v6 scatter
    (the two arms with a visible point cloud) to avoid duplicating the same
    42 markers six times; their diameters/retention are arm-specific columns
    even though the marker style is shared.

Run:
  cd /Users/ujs/mycode/thermal_lift && uv run python docs/publication_figures/scripts/fig37_retention_vs_size.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

DATA_DIR = REPO_ROOT / "output" / "dot_probe"

df = pd.read_csv(DATA_DIR / "per_dot.csv")
opt = pd.read_csv(DATA_DIR / "optical_subset.csv")
assert opt["dot_id"].nunique() == int(df["in_optical"].sum()) == 42, (
    "optical_subset.csv row count must match per_dot.csv in_optical flag"
)

DIAM_COL = "fwhm_diam_tgv_um"

# Fixed arm -> style mapping, identical to fig09_dot_probe_stratified.py.
ARMS = {
    "drizzle": dict(color="#DD8452", marker="^", ls="-", label="Drizzle"),
    "de_pb9":  dict(color="#55A868", marker="D", ls="-", label="Ours (de–pb9)"),
    "depb9v6": dict(color="#4C72B0", marker="o", ls="-", label="Ours (v6)"),
    "v14":     dict(color="#8172B2", marker="X", ls="-", label="Ours (v14)"),
    "v19":     dict(color="#937860", marker="P", ls="-", label="Ours (v19)"),
    "meanDC":  dict(color="#C44E52", marker="v", ls="-", label="Mean-DC"),
}
# Arms shown with a light per-dot scatter underlay (kept to 2 to avoid overplot).
SCATTER_ARMS = ["drizzle", "depb9v6"]

Y_LO, Y_HI = -0.15, 1.65

# ── Running median on a physical-diameter grid ─────────────────────────
WINDOW_HALF_UM = 6.0
MIN_SUPPORT = 15
d_min, d_max = np.nanpercentile(df[DIAM_COL], [0.5, 99.5])
x_grid = np.linspace(d_min, d_max, 240)
diam = df[DIAM_COL].to_numpy()

running = {}
support = np.zeros_like(x_grid)
for arm in ARMS:
    y = df[f"retention_{arm}"].to_numpy()
    med = np.full_like(x_grid, np.nan)
    for i, xc in enumerate(x_grid):
        mask = np.abs(diam - xc) <= WINDOW_HALF_UM
        n = mask.sum()
        if arm == "drizzle":
            support[i] = n
        if n >= MIN_SUPPORT:
            med[i] = np.nanmedian(y[mask])
    running[arm] = med

# ── Figure: main panel + bottom sample-density strip (shared x) ───────
# Wider than W_1P5 to reserve a clear right-hand margin for the legend, so it
# never sits on top of data (the top-left corner is the busiest region).
fig, (ax, ax_n) = plt.subplots(
    2, 1, figsize=(W_DOUBLE * 0.82, 3.9), sharex=True,
    gridspec_kw=dict(height_ratios=[3.1, 1.0], hspace=0.08),
)

# Light scatter underlay for the two key arms (rasterized: thousands of pts).
for arm in SCATTER_ARMS:
    st = ARMS[arm]
    y = df[f"retention_{arm}"].clip(Y_LO, Y_HI)
    ax.scatter(
        df[DIAM_COL], y, s=3.5, color=st["color"], alpha=0.09,
        linewidths=0, rasterized=True, zorder=1,
    )

# Running-median curves, all six arms.
for arm, st in ARMS.items():
    ax.plot(
        x_grid, running[arm], color=st["color"], ls=st["ls"], lw=1.6,
        label=st["label"], zorder=3, clip_on=True,
    )

# Optical-verified subset: overlay on the two scattered arms only.
opt_mask = df["in_optical"]
for arm in SCATTER_ARMS:
    st = ARMS[arm]
    ax.scatter(
        df.loc[opt_mask, DIAM_COL],
        df.loc[opt_mask, f"retention_{arm}"].clip(Y_LO, Y_HI),
        s=20, facecolors="none", edgecolors="black", linewidths=0.7,
        zorder=4,
    )

ax.axhline(1.0, color="#888888", lw=0.7, ls=":", zorder=0)
ax.set_ylim(Y_LO, Y_HI)
ax.set_ylabel("Retention (depth$_{\\mathrm{arm}}$ / depth$_{\\mathrm{TGV}}$)")
ax.set_title("Retention vs. dot diameter, with optical-verified subset", loc="left")

# Legend: arms + one shared marker entry for the optical subset.
handles, labels = ax.get_legend_handles_labels()
opt_handle = plt.Line2D(
    [], [], marker="o", ls="", markerfacecolor="none", markeredgecolor="black",
    markeredgewidth=0.7, markersize=4.5,
)
handles.append(opt_handle)
labels.append("Optical-verified subset (n=42)")
# Legend placed outside the axes (right margin) so it never occludes data --
# the top-left of the plot is the busiest region (small-dot cluster).
ax.legend(
    handles, labels, loc="center left", bbox_to_anchor=(1.01, 0.42),
    ncol=1, fontsize=7.0, borderpad=0.4,
)

n_off_scale = int(
    sum((df[f"retention_{a}"] < Y_LO) | (df[f"retention_{a}"] > Y_HI) for a in ARMS).sum()
)
ax.text(
    0.99, 0.02,
    f"{n_off_scale} arm-dot points off-scale (clipped, not dropped)",
    transform=ax.transAxes, ha="right", va="bottom", fontsize=6.3, color="#555555",
)

# ── Bottom strip: honest n-per-bin support ─────────────────────────────
bin_edges = np.arange(10, 125, 10)
all_counts, _ = np.histogram(df[DIAM_COL], bins=bin_edges)
opt_counts, _ = np.histogram(df.loc[opt_mask, DIAM_COL], bins=bin_edges)
centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
width = np.diff(bin_edges) * 0.9
ax_n.bar(centers, all_counts, width=width, color="#bbbbbb", label="All dots", zorder=1)
ax_n.bar(centers, opt_counts, width=width * 0.45, color="black", label="Optical subset", zorder=2)
ax_n.set_ylabel("n / 10 $\\mu$m", fontsize=7.5)
ax_n.set_xlabel("Dot diameter [$\\mu$m] (TGV FWHM-equivalent)")
ax_n.set_yscale("log")
ax_n.set_ylim(0.7, 1200)
ax_n.legend(loc="upper right", fontsize=6.3, ncol=1, handlelength=1.0, borderpad=0.3)
ax_n.set_xlim(d_min - 2, d_max + 2)

paths = save_fig(fig, "fig37_retention_vs_size")
print("\n".join(str(p) for p in paths))
