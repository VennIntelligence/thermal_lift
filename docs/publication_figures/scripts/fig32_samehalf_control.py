"""Fig 32 -- Same-half control: shared-prior hallucination inflates self-consistency.

Companion to fig06 (which reports the honest, symmetrized cross-method FRC
that fig06 relies on as the resolution instrument). This figure justifies
*why* that convention is necessary: it contrasts each method's own
self split-half FRC (X_a vs X_b, same method, both halves reconstructed by
that method alone) against its cross-method FRC vs an independent reference
(drizzle; classical methods also get an independent tgv-vs-maptv check).
Neural methods (V11, C_nodr, D_dr01) show self-FRC almost as high as the
classical arms (0.92-0.96 @ 30um) -- but their cross-method FRC against a
truly independent reconstruction collapses to ~0.10-0.12, an order of
magnitude below the classical arms' cross-method FRC (0.37-0.56 @ 30um).
Self split-half does not decorrelate a deterministic network's learned
prior -- both halves reproduce the same hallucinated high-frequency detail,
so self-FRC rewards reproducible hallucination rather than real resolved
signal (ACL-046 Task E, ACL-047 Task 2).

Data provenance (inspected before drawing; the task-supplied
remote_inbox/20260705_stage0g/task2_samehalf_method_summary.csv turned out,
on inspection, to hold a *different* comparison -- cross-method agreement
restricted to a shared input half, i.e. X_a vs Y_a for two different
methods, not the self-vs-cross split-half comparison ACL-047 is about; its
own conclusion ("degree of resolution deferred, not hallucination") does
not support this figure's claim). The correct paired data for the
self-vs-cross point already exists in two earlier same-session archives:
  - self split-half FRC (external_artifact_pair, i.e. same method's own
    two-half reconstructions):
    remote_inbox/20260703_stage1a/artifacts/task_e/method_summary.csv
  - cross-method FRC vs an independent reference, symmetrized
    (cross_method_pair):
    remote_inbox/20260704_stage0f/t2_frc_method_summary.csv
Both are read-only archives of already-computed CSV outputs; no
recomputation performed here. frc_at_30um / frc_at_24um columns used
(credible band per ACL-047; below 20um is aperture-zero, audit-only).

IMPORTANT calibration split (ACL-049): the stage0f cross-FRC values above
predate the +0.5 px grid-convention correction, so for the neural arms they
conflate two effects -- the registration artifact (~0.4 FRC points,
retroactively acquitted) and the genuine self-split inflation. The figure
therefore shows a THIRD marker per row: cross-FRC after the measured-offset
correction (remote_inbox/20260713_dotprobe/offset_probe_summary_stage0h.csv,
frc_at_30um_after). The honest decomposition per neural arm reads:
self-split inflation ~0.4 (self vs corrected-cross) + registration artifact
~0.4 (corrected vs uncorrected cross). Classical arms are nearly unmoved by
the correction.

Run: uv run python docs/publication_figures/scripts/fig32_samehalf_control.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import REPO_ROOT, W_1P5, save_fig, setup_academic_style

setup_academic_style()

SELF_CSV = REPO_ROOT / "remote_inbox/20260703_stage1a/artifacts/task_e/method_summary.csv"
CROSS_CSV = REPO_ROOT / "remote_inbox/20260704_stage0f/t2_frc_method_summary.csv"

self_df = pd.read_csv(SELF_CSV).set_index("method")
cross_df = pd.read_csv(CROSS_CSV).set_index("method")
OFFSET_CSV = REPO_ROOT / "remote_inbox/20260713_dotprobe/offset_probe_summary_stage0h.csv"
corr_df = pd.read_csv(OFFSET_CSV).set_index("pair")
CORR_KEY = {"v11": "v11_vs_drz", "c_nodr": "c_nodr_vs_drz",
            "d_dr01": "d_dr01_vs_drz", "tgv": "tgv_vs_drz",
            "maptv": "maptv_vs_drz"}

# method -> (self-split-half row key, cross-vs-independent-ref row key, display label, is_neural)
PAIRS = [
    ("tgv",    "tgv_x_drz",    "TGV",           False),
    ("maptv",  "maptv_x_drz",  "MAP-TV",        False),
    ("v11",    "v11_x_drz",    "Ours V11",      True),
    ("c_nodr", "c_nodr_x_drz", "Ours C (no DR)", True),
    ("d_dr01", "d_dr01_x_drz", "Ours D (DR 0.1px)", True),
]

BAND = "frc_at_30um"
BAND_LABEL = "FRC @ 30 $\\mu$m"

CLASSICAL_COLOR = "#333333"
NEURAL_COLOR = "#C44E52"
SELF_COLOR = "#B0B0B0"

fig, ax = plt.subplots(figsize=(W_1P5 + 1.5, 3.0))

y_positions = np.arange(len(PAIRS))[::-1]
for y, (self_key, cross_key, label, is_neural) in zip(y_positions, PAIRS):
    v_self = self_df.loc[self_key, BAND]
    v_cross_raw = cross_df.loc[cross_key, BAND]
    v_cross_corr = corr_df.loc[CORR_KEY[self_key], "frc_at_30um_after"]
    color = NEURAL_COLOR if is_neural else CLASSICAL_COLOR
    # uncorrected -> corrected: the +0.5 px registration artifact (acquitted)
    ax.plot([v_cross_raw, v_cross_corr], [y, y], color=color, lw=1.1,
            zorder=1, alpha=0.35, ls=":")
    # corrected cross -> self: the genuine self-split inflation
    ax.plot([v_cross_corr, v_self], [y, y], color=color, lw=1.1, zorder=1,
            alpha=0.7)
    ax.scatter([v_self], [y], marker="o", s=34, facecolor=SELF_COLOR,
               edgecolor=color, linewidth=1.0, zorder=3)
    ax.scatter([v_cross_raw], [y], marker="o", s=34, facecolor="white",
               edgecolor=color, linewidth=1.0, zorder=3)
    ax.scatter([v_cross_corr], [y], marker="o", s=34, facecolor=color,
               edgecolor=color, linewidth=1.0, zorder=3)
    gap = v_self - v_cross_corr
    ax.annotate(f"$-${gap:.2f}", xy=((v_self + v_cross_corr) / 2, y),
                xytext=(0, 6.5), textcoords="offset points", ha="center",
                fontsize=6.5, color=color)

ax.set_yticks(y_positions)
ax.set_yticklabels([lbl for _, _, lbl, _ in PAIRS])
ax.axvline(0, color="#cccccc", lw=0.6, zorder=0)
ax.set_xlim(-0.05, 1.05)
ax.set_xlabel(BAND_LABEL)
ax.set_title("Self split-half vs. honest cross-method FRC", loc="left")

# de-occluded legend via proxy artists, placed in empty upper-right space
legend_handles = [
    plt.Line2D([0], [0], marker="o", color="none", markerfacecolor=SELF_COLOR,
               markeredgecolor="#333333", markersize=6, label="self split-half\n(X$_a$ vs X$_b$, invalid)"),
    plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#666666",
               markeredgecolor="#666666", markersize=6,
               label="cross vs. indep. ref.\n(offset-corrected)"),
    plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="white",
               markeredgecolor="#666666", markersize=6,
               label="cross, uncorrected\n(+0.5 px artifact,\nACL-049 acquitted)"),
]
ax.legend(handles=legend_handles, loc="center left", fontsize=6.3, labelspacing=1.4,
          handletextpad=0.6, bbox_to_anchor=(1.03, 0.5))

ax.annotate("shared prior survives self-split\n$\\rightarrow$ inflated agreement ($\\sim$0.4)",
            xy=(0.75, y_positions[2] + 0.32), ha="center", va="bottom",
            fontsize=7, color=NEURAL_COLOR, style="italic")

paths = save_fig(fig, "fig32_samehalf_control")
print("\n".join(str(p) for p in paths))
