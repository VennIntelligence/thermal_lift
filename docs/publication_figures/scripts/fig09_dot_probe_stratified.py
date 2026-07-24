"""Fig 09 -- Dot-probe retention stratified by size, depth, and isolation (ACL-063).

ACL-063 established that all neural arms attenuate small, dark defects more
than drizzle/classical baselines. This figure shows WHERE the neural arms
fail: it stratifies median dot retention (recovered depth / injected depth)
by dot size bin, depth (contrast) bin, and structural isolation, so the
small x shallow x isolated-corner failure mode is visible directly rather
than averaged away in a single headline number.

Data provenance:
  output/dot_probe/summary_by_arm_size.csv       (panel a: size_bin x arm)
  output/dot_probe/summary_by_arm_depth.csv      (panel b: depth_bin x arm)
  output/dot_probe/summary_by_arm_isolation.csv  (panel c: isolation x arm,
                                                    size_bin == "ALL" rows)
  output/dot_probe/per_dot.csv                   (inspected only, not
                                                    plotted directly here)

Caveats:
  - The sibling directories output/dot_probe_v7/ and output/dot_probe_v24ctrl/
    only contain summary_v22_arms.csv, a preprocessing/calibration table
    (gain, offset, by_size, by_isolation for depb9v7 / depb9v7_ctrl variants).
    They do not carry median_retention broken out by depth_bin, and their
    arms (depb9v7, depb9v7_bin4, depb9v7_ctrl) are outside the fixed
    arm-color mapping specified for this figure. They add no new arms to
    the primary retention-stratification story and are not plotted.
  - "tgv" appears only as a per-dot depth-fit reference column in per_dot.csv
    (retention_tgv_self == 1.0 identically); it is not one of the six arms
    carrying a median_retention column in the summary CSVs, so it is
    omitted from the legend (only arms present in the data are shown).
  - Panel c (isolation) uses the size_bin == "ALL" aggregate rows in
    summary_by_arm_isolation.csv (isolated vs. structured), since isolation
    is not itself binned further within this figure.

Run:
  cd /Users/ujs/mycode/thermal_lift && uv run python docs/publication_figures/scripts/fig09_dot_probe_stratified.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

DATA_DIR = REPO_ROOT / "output" / "dot_probe"

size_df = pd.read_csv(DATA_DIR / "summary_by_arm_size.csv")
depth_df = pd.read_csv(DATA_DIR / "summary_by_arm_depth.csv")
iso_df = pd.read_csv(DATA_DIR / "summary_by_arm_isolation.csv")

# Fixed arm -> style mapping (colors as specified; only arms present in data).
ARMS = {
    "drizzle": dict(color="#DD8452", marker="^", ls="-", label="Drizzle"),
    "de_pb9":  dict(color="#55A868", marker="D", ls="-", label="Ours (de–pb9)"),
    "depb9v6": dict(color="#4C72B0", marker="o", ls="-", label="Ours (v6)"),
    "v14":     dict(color="#8172B2", marker="X", ls="-", label="Ours (v14)"),
    "v19":     dict(color="#937860", marker="P", ls="-", label="Ours (v19)"),
    "meanDC":  dict(color="#C44E52", marker="v", ls="-", label="Mean-DC"),
}

SIZE_ORDER = ["<=3px", "3-5px", "5-8px", ">8px"]
SIZE_LABELS = ["$\\leq$3", "3–5", "5–8", "$>$8"]
DEPTH_ORDER = ["shallow", "mid", "deep"]
DEPTH_LABELS = ["Shallow", "Mid", "Deep"]
ISO_ORDER = ["isolated", "structured"]
ISO_LABELS = ["Isolated", "Structured"]

fig, axes = plt.subplots(1, 3, figsize=(W_DOUBLE, 2.5), sharey=True)
ax_a, ax_b, ax_c = axes

# ── Panel (a): retention vs. size bin ──────────────────────────────
for arm, st in ARMS.items():
    sub = size_df[size_df["arm"] == arm].set_index("size_bin")
    y = [sub.loc[s, "median_retention"] for s in SIZE_ORDER]
    ax_a.plot(range(len(SIZE_ORDER)), y, color=st["color"], marker=st["marker"],
              ls=st["ls"], label=st["label"], markersize=4.5, clip_on=False)
ax_a.set_xticks(range(len(SIZE_ORDER)))
ax_a.set_xticklabels(SIZE_LABELS)
ax_a.set_xlabel("Dot diameter bin [px]")
ax_a.set_title("(a) By size", loc="left")

# ── Panel (b): retention vs. depth (contrast) bin ──────────────────
for arm, st in ARMS.items():
    sub = depth_df[depth_df["arm"] == arm].set_index("depth_bin")
    y = [sub.loc[d, "median_retention"] for d in DEPTH_ORDER]
    ax_b.plot(range(len(DEPTH_ORDER)), y, color=st["color"], marker=st["marker"],
              ls=st["ls"], label=st["label"], markersize=4.5, clip_on=False)
ax_b.set_xticks(range(len(DEPTH_ORDER)))
ax_b.set_xticklabels(DEPTH_LABELS)
ax_b.set_xlabel("Injected-depth bin")
ax_b.set_title("(b) By depth", loc="left")

# ── Panel (c): retention vs. isolation (size_bin == ALL) ───────────
for arm, st in ARMS.items():
    sub = iso_df[(iso_df["arm"] == arm) & (iso_df["size_bin"] == "ALL")].set_index("isolation")
    y = [sub.loc[i, "median_retention"] for i in ISO_ORDER]
    ax_c.plot(range(len(ISO_ORDER)), y, color=st["color"], marker=st["marker"],
              ls=st["ls"], label=st["label"], markersize=4.5, clip_on=False)
ax_c.set_xticks(range(len(ISO_ORDER)))
ax_c.set_xticklabels(ISO_LABELS)
ax_c.set_xlim(-0.35, len(ISO_ORDER) - 1 + 0.35)
ax_c.set_xlabel("Structural context")
ax_c.set_title("(c) By isolation", loc="left")

ax_a.set_ylabel("Median retention")
ax_a.set_ylim(0.0, 1.02)
for ax in axes:
    ax.margins(x=0.08)

# Mark the worst corner (small x shallow) for context: point at the
# v19/meanDC low cluster at the 3-5px bin, text placed clear of all lines.
ax_a.annotate(
    "small dots hit\nhardest (ACL-063)",
    xy=(1, 0.225), xytext=(1.65, 0.08),
    textcoords="data", fontsize=6.6, color="#333333", ha="left", va="center",
    arrowprops=dict(arrowstyle="-", color="#666666", lw=0.6, shrinkA=2, shrinkB=2),
)

handles, labels = ax_a.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.06),
           fontsize=7.5, columnspacing=1.1, handletextpad=0.5, frameon=False)

fig.suptitle("")
paths = save_fig(fig, "fig09_dot_probe_stratified")
print("\n".join(str(p) for p in paths))
