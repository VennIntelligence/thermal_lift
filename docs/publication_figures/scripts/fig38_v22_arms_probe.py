"""Fig 38 -- v22-era arm calibration on the real-domain dot probe.

The "v22" dot-probe preprocessing script (summary_v22_arms*.csv) was run as
a side-channel verification/calibration pass across several checkpoint
audits that are otherwise scattered across research-log entries: the
v24-ctrl batch/patch-alignment check (ACL-067), the 300-scene micro-calib
arms (ACL-069), and the depb9v8 pool verdict (ACL-070). Unlike the primary
dot-probe summaries used by fig09 (output/dot_probe/summary_by_arm_*.csv),
these tables also carry a per-arm *registration/gain calibration* block
(photometric gain slope+intercept, sub-pixel xy offset) alongside the usual
by-size / by-isolation retention breakdown -- so this figure asks a
question fig09 cannot: is the retention spread across arms explained by
drifting registration/gain calibration, or is it a real model-quality
effect with calibration held essentially constant?

Data provenance (comment='#' not used; plain CSV):
  output/dot_probe_v24ctrl/summary_v22_arms.csv
      -> depb9v7_ctrl, depb9v7, depb9v7_bin4, depb9v6, depb9v6_bin4
  remote_inbox/20260716_micro_calib/probe_out/summary_v22_arms_combined.csv
      -> micro_v7end, micro_v7end_4k, micro_v6end, micro_v6end_4k
         (+ a repeated depb9v6 block, identical to the v24ctrl file, used
         only for dedup / cross-file consistency, not re-plotted twice)
  remote_inbox/20260716_v8_verdict/probe_out/summary_v22_arms_combined.csv
      -> depb9v8 (+ a repeated depb9v6 block, same as above)

Each file has three stacked tables (a `table` column selects rows):
  preprocessing -- one row per arm: gain_intercept, gain_slope,
                   offset_dx/dy/norm_px (registration + photometric
                   calibration of the preprocessing step itself).
  by_size       -- median_retention per arm x size_bin (<=3/3-5/5-8/>8/ALL).
  by_isolation  -- median_retention per arm x isolation x size_bin, used
                   here only at size_bin == "ALL".

Arm key (kept short on-figure; spelled out here):
  *_ctrl      = ACL-067 batch/patch-aligned control (v7 pool, no shift/gain
                applied deliberately -- the "is our harness itself honest"
                check).
  *_bin4      = same checkpoint re-probed after 4x4 detector binning.
  micro_*     = ACL-069 300-scene micro-calib arms (small held-out set;
                *_4k suffix = 4k-iteration checkpoint vs. *end = final).
  depb9v8     = ACL-070 pool-v8 verification arm.
  depb9v6     = anchor/reference only -- its by_size / by_isolation
                retention numbers are already the primary fig09 content
                (output/dot_probe/summary_by_arm_*.csv); values here are
                numerically consistent (agree to ~1e-3) and are shown as
                thin dashed reference lines in panel (b), not re-plotted
                as bars, to avoid duplicating fig09.

Design:
  Panel (a): registration/gain calibration scatter, one point per arm
             (offset_norm_px vs. gain_slope). All 10 arms cluster tightly
             (slope 1.13-1.21, offset <0.06 px) regardless of family.
  Panel (b): median dot retention (ALL sizes, size_bin=="ALL"), isolated
             vs. structured, grouped by arm family. Retention spans
             0.33-1.13 across arms with near-identical calibration in (a)
             -- i.e. the panel (a) tightness rules out registration/gain
             drift as the explanation for the panel (b) spread.

Caveat: "v22-era" here names the preprocessing-script vintage, not a model
pool generation; do not confuse depb9v7/v8 (pool checkpoints) with "v22"
itself. retention values >1.0 (micro_v6end/_4k, micro_v7end_4k) indicate
recovered depth exceeding injected depth (over-restoration at these
checkpoints), not a data error -- flagged with a light gray >1.0 band.

Run:
  cd /Users/ujs/mycode/thermal_lift && uv run python docs/publication_figures/scripts/fig38_v22_arms_probe.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import (
    METHOD_PALETTE,
    METHOD_STYLE,
    REF_LINE_GRAY,
    REPO_ROOT,
    W_DOUBLE,
    save_fig,
    setup_academic_style,
)

setup_academic_style()

V24CTRL = REPO_ROOT / "output" / "dot_probe_v24ctrl" / "summary_v22_arms.csv"
MICRO = (
    REPO_ROOT
    / "remote_inbox"
    / "20260716_micro_calib"
    / "probe_out"
    / "summary_v22_arms_combined.csv"
)
V8VERDICT = (
    REPO_ROOT
    / "remote_inbox"
    / "20260716_v8_verdict"
    / "probe_out"
    / "summary_v22_arms_combined.csv"
)

raw = pd.concat(
    [pd.read_csv(V24CTRL), pd.read_csv(MICRO), pd.read_csv(V8VERDICT)],
    ignore_index=True,
)
# Dedup exact repeated (table, arm, isolation, size_bin) blocks across files
# (depb9v6 appears verbatim in all three; keep first occurrence).
raw = raw.drop_duplicates(subset=["table", "arm", "isolation", "size_bin"], keep="first")

# ── Panel (a): calibration ──────────────────────────────────────────
calib = raw[raw["table"] == "preprocessing"].set_index("arm")

ARM_STYLE = {
    "depb9v7_ctrl":  dict(color=METHOD_PALETTE["accent_1"], marker="*", ms=11, label="v7 ctrl"),
    "depb9v7":       dict(color=METHOD_PALETTE["accent_1"], marker="o", ms=6,  label="v7"),
    "depb9v7_bin4":  dict(color=METHOD_PALETTE["accent_1"], marker="^", ms=6,  label="v7 bin4"),
    "depb9v6":       dict(color=METHOD_STYLE["depb9v6"]["color"], marker="o", ms=6, label="v6 (anchor)"),
    "depb9v6_bin4":  dict(color=METHOD_STYLE["depb9v6"]["color"], marker="^", ms=6, label="v6 bin4"),
    "depb9v8":       dict(color=METHOD_STYLE["depb9v8"]["color"], marker="X", ms=7, label="v8 (anchor)"),
    "micro_v6end":   dict(color=METHOD_PALETTE["secondary"], marker="o", ms=6, label="micro v6"),
    "micro_v6end_4k": dict(color=METHOD_PALETTE["secondary"], marker="^", ms=6, label="micro v6 4k"),
    "micro_v7end":   dict(color=METHOD_PALETTE["accent_3"], marker="o", ms=6, label="micro v7"),
    "micro_v7end_4k": dict(color=METHOD_PALETTE["accent_3"], marker="^", ms=6, label="micro v7 4k"),
}

# Manual label offsets (data units) to avoid overlap in a tight cluster.
LABEL_OFFSET = {
    "depb9v7_ctrl": (0.001, 0.006),
    "depb9v7": (0.001, -0.010),
    "depb9v7_bin4": (0.003, -0.006),
    "depb9v6": (0.003, -0.009),
    "depb9v6_bin4": (0.002, 0.007),
    "depb9v8": (0.002, -0.010),
    "micro_v6end": (-0.014, 0.006),
    "micro_v6end_4k": (0.002, 0.007),
    "micro_v7end": (0.002, 0.006),
    "micro_v7end_4k": (-0.001, -0.011),
}

fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(W_DOUBLE, 3.3), width_ratios=[0.95, 1.55])

for arm, sty in ARM_STYLE.items():
    if arm not in calib.index:
        continue
    row = calib.loc[arm]
    ax_a.scatter(
        row["offset_norm_px"], row["gain_slope"],
        color=sty["color"], marker=sty["marker"], s=sty["ms"] ** 2 * 2.2,
        edgecolor="black", linewidth=0.5, zorder=3,
    )
    dx, dy = LABEL_OFFSET[arm]
    ax_a.annotate(
        sty["label"], (row["offset_norm_px"], row["gain_slope"]),
        xytext=(row["offset_norm_px"] + dx, row["gain_slope"] + dy),
        fontsize=6.5, ha="left" if dx >= 0 else "right",
    )

ax_a.text(
    0.98, 0.03, "gain = 1 (no drift) is off-scale below this panel",
    transform=ax_a.transAxes, fontsize=6.5, color="#666666", ha="right", va="bottom",
)
ax_a.set_xlabel("Registration offset, $\\|(dx,dy)\\|$ [px]")
ax_a.set_ylabel("Photometric gain slope")
ax_a.set_title("(a) Preprocessing calibration, all arms")
ax_a.set_xlim(-0.004, 0.062)
ax_a.set_ylim(1.10, 1.225)
ax_a.grid(axis="both", alpha=0.25, linewidth=0.5)

# ── Panel (b): retention by isolation, ALL size bin ─────────────────
iso = raw[(raw["table"] == "by_isolation") & (raw["size_bin"] == "ALL")]
iso = iso.set_index(["arm", "isolation"])["median_retention"]

FAMILY_ORDER = [
    "depb9v7_ctrl", "depb9v7", "depb9v7_bin4",
    "depb9v6", "depb9v6_bin4", "depb9v8",
    "micro_v7end", "micro_v7end_4k", "micro_v6end", "micro_v6end_4k",
]
ANCHORS = {"depb9v6", "depb9v8"}
LABELS_B = {
    "depb9v7_ctrl": "v7\nctrl", "depb9v7": "v7", "depb9v7_bin4": "v7\nbin4",
    "depb9v6": "v6\n(anchor)", "depb9v6_bin4": "v6\nbin4", "depb9v8": "v8\n(anchor)",
    "micro_v7end": "micro\nv7", "micro_v7end_4k": "micro\nv7 4k",
    "micro_v6end": "micro\nv6", "micro_v6end_4k": "micro\nv6 4k",
}

x = np.arange(len(FAMILY_ORDER))
width = 0.36
iso_vals = [iso.loc[(a, "isolated")] for a in FAMILY_ORDER]
struct_vals = [iso.loc[(a, "structured")] for a in FAMILY_ORDER]
edge_colors = ["#222222" if a in ANCHORS else "none" for a in FAMILY_ORDER]
hatches = ["///" if a in ANCHORS else None for a in FAMILY_ORDER]

ax_b.axhspan(1.0, 1.25, color="#eeeeee", zorder=0)
ax_b.text(
    0.985, 0.965, "over-restoration\n(retention $>$ 1)",
    transform=ax_b.transAxes, fontsize=6.5, color="#888888", ha="right", va="top",
)

bars1 = ax_b.bar(x - width / 2, iso_vals, width, color=METHOD_PALETTE["primary"],
                  edgecolor=edge_colors, hatch=None, label="Isolated", zorder=3)
bars2 = ax_b.bar(x + width / 2, struct_vals, width, color=METHOD_PALETTE["accent_3"],
                  edgecolor=edge_colors, hatch=None, label="Structured", zorder=3)
for bars, is_anchor_hatch in ((bars1, hatches), (bars2, hatches)):
    for rect, h, a in zip(bars, is_anchor_hatch, FAMILY_ORDER):
        if a in ANCHORS:
            rect.set_hatch("///")
            rect.set_linewidth(1.0)

ax_b.axhline(1.0, **REF_LINE_GRAY)
ax_b.set_xticks(x)
ax_b.set_xticklabels([LABELS_B[a] for a in FAMILY_ORDER], fontsize=7)
ax_b.set_ylabel("Median dot retention (ALL sizes)")
ax_b.set_title("(b) Retention by isolation, hatched = fig09 anchor arm")
ax_b.set_ylim(0, 1.28)
ax_b.grid(axis="y", alpha=0.3, linewidth=0.5)
ax_b.legend(loc="upper left", ncol=1, fontsize=7.5)

fig.suptitle(
    "v22-era dot-probe: registration/gain calibration is stable across arms;"
    " retention is not",
    fontsize=9.5,
)

save_fig(fig, "fig38_v22_arms_probe")
print("Saved fig38_v22_arms_probe.{png,pdf}")
