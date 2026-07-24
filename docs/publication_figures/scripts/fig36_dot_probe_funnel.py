"""Fig 36 -- Real-domain dot probe: detection funnel + example board crops.

(a) Horizontal waterfall of the dot-detection pipeline (ACL-063 P0 dot
retention probe), TGV work image only. Stages, in order, are the keys of
output/dot_probe/detection_funnel.json (also reproduced in section "1.
Detection funnel" of output/dot_probe/summary.md):
  raw_3d_local_maxima  -> candidate maxima in the 3-D (x, y, depth) TGV
                          reconstruction stack.
  after_dedup           -> after collapsing duplicate maxima from adjacent
                            depth slices onto one detection.
  after_edge             -> after dropping detections within 16 px of the
                            work-image border (removed_edge_16px).
  after_depth_snr        -> after requiring depth SNR >= 4 at the maximum
                            (removed_depth_snr_lt4).
  after_half_consistency -> after requiring the detection to reproduce in
                            both independent frame-halves (removed_half_
                            consistency; 0 removed for this run -- every
                            depth/edge/SNR survivor was half-consistent).
  after_size == final_dots -> after excluding detections with fitted TGV
                            spot radius > 6 px (removed_size_r_gt6px); this
                            is the final probe set used for the retention
                            analysis (N = 3562, matches summary.md sections
                            2b/3/3b).
The "removed_half_consistency: 0" step is annotated as a text note rather
than drawn as a zero-width bar segment.

(b) One example dot (#2243, TGV FWHM d=4.9 px, depth=0.067, size bin
3-5px, depth-mid tertile), cropped tightly (no header/title band) from a
clean interior row of output/dot_probe/board_crops.png, spanning all 7
processing arms. Crop window and yellow retention-score insets are exactly
as pre-rendered by the upstream board-crop script (window = [-1.5, +0.75]
x depth_tgv around the dot, background-subtracted (a+b)/2 average of the
two frame-halves); this script only slices the row out of the existing
PNG and adds its own column headers -- no recolormap, no re-rendering of
the crops themselves. "ref" (tgv column) marks the reference arm the
retention score for other arms is computed against.

Data: output/dot_probe/detection_funnel.json, output/dot_probe/summary.md,
      output/dot_probe/board_crops.png (row 4, y in [880, 1146) px of the
      full 6300-px-tall montage).
Run:  uv run python docs/publication_figures/scripts/fig36_dot_probe_funnel.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from pubfig_style import METHOD_PALETTE, REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

DOT_PROBE_DIR = REPO_ROOT / "output" / "dot_probe"

with open(DOT_PROBE_DIR / "detection_funnel.json") as f:
    funnel = json.load(f)

# ── Panel (a): waterfall stages ─────────────────────────────────────────
# (label, count, removed_this_step, removed_reason)
STAGES = [
    ("Raw 3-D local maxima", funnel["raw_3d_local_maxima"], None, None),
    ("After dedup", funnel["after_dedup"],
     funnel["raw_3d_local_maxima"] - funnel["after_dedup"], "duplicate maxima"),
    ("After edge exclusion", funnel["after_edge"],
     funnel["removed_edge_16px"], "within 16 px of border"),
    ("After depth-SNR filter", funnel["after_depth_snr"],
     funnel["removed_depth_snr_lt4"], "depth SNR $<$ 4"),
    ("After size filter\n(final probe set)", funnel["after_size"],
     funnel["removed_size_r_gt6px"], "spot radius $>$ 6 px"),
]
assert funnel["after_size"] == funnel["final_dots"]
assert funnel["after_half_consistency"] == funnel["after_depth_snr"]

labels = [s[0] for s in STAGES]
counts = [s[1] for s in STAGES]
n_stages = len(STAGES)
y_pos = np.arange(n_stages)[::-1]  # top = raw, bottom = final

fig = plt.figure(figsize=(W_DOUBLE, 4.15))
gs = fig.add_gridspec(2, 1, height_ratios=[2.0, 1.0], hspace=0.05)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])

bar_color = METHOD_PALETTE["primary"]
final_color = METHOD_PALETTE["secondary"]
bar_colors = [bar_color] * (n_stages - 1) + [final_color]

ax_a.barh(y_pos, counts, height=0.6, color=bar_colors, edgecolor="#222222",
          linewidth=0.6, zorder=3)

for yi, (label, count, removed, reason) in zip(y_pos, STAGES):
    ax_a.text(count + 60, yi, f"{count:,}", va="center", ha="left",
               fontsize=8, color="#222222")

# removal annotations between consecutive bars
for i in range(1, n_stages):
    y_top = y_pos[i - 1]
    y_bot = y_pos[i]
    removed = STAGES[i][2]
    reason = STAGES[i][3]
    y_mid = (y_top + y_bot) / 2.0
    ax_a.annotate(
        "", xy=(counts[i - 1] * 0.985, y_bot + 0.32),
        xytext=(counts[i - 1] * 0.985, y_top - 0.32),
        arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7, ls="--"),
    )
    ax_a.text(
        counts[i - 1] * 1.03, y_mid,
        f"$-${removed:,} ({reason})",
        va="center", ha="left", fontsize=7, color=METHOD_PALETTE["accent_1"],
        style="italic",
    )

# note about the zero-removal half-consistency check, placed near the
# final bar since it sits between after_depth_snr and after_size
ax_a.text(
    counts[3] * 1.03, y_pos[3] - 0.85,
    "(half-consistency check: 0 removed)",
    va="center", ha="left", fontsize=6.5, color="#666666",
)

ax_a.set_yticks(y_pos)
ax_a.set_yticklabels(labels, fontsize=8)
ax_a.set_xlabel("Dot count")
ax_a.set_xlim(0, max(counts) * 1.55)
ax_a.set_title("(a) Detection funnel: candidate maxima $\\to$ final probe set")
ax_a.grid(axis="x", alpha=0.3, linewidth=0.5)
ax_a.set_axisbelow(True)

# ── Panel (b): example dot crops across arms ────────────────────────────
board = plt.imread(DOT_PROBE_DIR / "board_crops.png")
h, w = board.shape[0], board.shape[1]
row_top, row_bot = 880, 1146  # clean interior row, no title/header band
crop = board[row_top:row_bot, :, :]

ax_b.imshow(crop, interpolation="nearest", aspect="auto")
ax_b.set_xticks([])
ax_b.set_yticks([])
for sp in ax_b.spines.values():
    sp.set_visible(False)

arm_labels = ["drizzle", "tgv (ref)", "v14", "v19", "de_pb9", "depb9v6", "meanDC"]
n_arms = len(arm_labels)
col_w = w / n_arms
for j, name in enumerate(arm_labels):
    xc = col_w * (j + 0.5)
    ax_b.text(xc, -14, name, ha="center", va="bottom", fontsize=7)

ax_b.set_title(
    "(b) Example dot #2243 (TGV $d$=4.9 px) across processing arms "
    "-- inset = retention score",
    pad=14,
)

paths = save_fig(fig, "fig36_dot_probe_funnel")
print("\n".join(str(p) for p in paths))
