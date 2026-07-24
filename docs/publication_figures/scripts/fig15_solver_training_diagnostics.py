"""Fig 15 -- V8/K4 full-halo unrolled-solver training diagnostics.

Six-panel dashboard of TensorBoard scalar exports from the archived V8/K4
full-halo solver run (unroll_steps=4, no drizzle, real_eval_solver_mode=
full_halo, real_eval_solver_halo_hr=96; see the archive README.md and
diagnosis_20260630/DIAGNOSIS.md for the full grid-artifact / GroupNorm+SE
extent-shift investigation this checkpoint was produced under).

Panels:
  (a) Loss components (log y): total, DC, struct, highpass.
      NOTE (caveat): loss/total and loss/struct are numerically IDENTICAL at
      every logged step in this export -- consistent with struct loss
      dominating the total in this run's weighting, not a plotting error.
      Kept both curves to make that identity visible rather than hiding it.
  (b) Real-domain physics consistency: DC-residual band vs full, same units.
      Only steps 5000/10000 were logged (real eval hook is infrequent) --
      shown as markers, not a dense curve.
  (c) Real artifact score / out-of-band ratio: scales differ by ~100x, so
      split into two stacked mini-panels sharing the x-axis (no dual axes,
      per plotting_standards.md).
  (d) Synthetic PSNR vs step (4 eval points at 2500-step cadence).
  (e) Synthetic boundary F1 / region RMSE: split into two stacked
      mini-panels (again, no dual axes).
  (f) train/eta and train/anneal: BOTH ARE FLAT LINES throughout this run
      (eta = 0.5, anneal = 1 at every logged step). This is not a plotting
      artifact -- it is the documented frozen-eta / frozen-prior-annealing
      training design for this checkpoint (ACL-025 / ACL-026): eta and the
      prior-annealing weight are fixed, not scheduled, during V8/K4 training.

Data: research_log/episodes/ep07_solver_v8_k4_fullhalo_eval_archive/scalars/
      *.csv -- TensorBoard scalar exports, columns (wall_time, step, value).
Run:  cd /Users/ujs/mycode/thermal_lift && \
      uv run python docs/publication_figures/scripts/fig15_solver_training_diagnostics.py
"""

import matplotlib.pyplot as plt
import pandas as pd

from pubfig_style import METHOD_PALETTE, REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

SRC = (
    REPO_ROOT
    / "research_log/episodes/ep07_solver_v8_k4_fullhalo_eval_archive/scalars"
)


def load(tag: str) -> pd.DataFrame:
    df = pd.read_csv(SRC / f"{tag}.csv")
    return df.sort_values("step")


def style_ax(ax, title, ylabel, xlabel=True):
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel("Training step [$\\times 10^3$]")


def line(ax, df, color, marker, label, lw=1.2):
    dense = len(df) > 20
    ax.plot(
        df["step"] / 1000, df["value"],
        color=color, lw=lw,
        marker=None if dense else marker, markersize=4.5,
        label=label,
    )


# No in-figure suptitle (title belongs in the caption); spacing kept tight
# so panels, not gaps, dominate the canvas.
fig = plt.figure(figsize=(W_DOUBLE, 5.4), constrained_layout=False)
gs = fig.add_gridspec(
    2, 3, hspace=0.45, wspace=0.42,
    left=0.075, right=0.99, top=0.955, bottom=0.085,
)

# ── (a) Loss components, log y ──────────────────────────────────────
ax_a = fig.add_subplot(gs[0, 0])
loss_specs = [
    ("loss_total", "Total", METHOD_PALETTE["primary"], "o"),
    ("loss_dc", "DC", METHOD_PALETTE["accent_1"], "s"),
    ("loss_struct", "Struct", METHOD_PALETTE["secondary"], "^"),
    ("loss_highpass", "Highpass", METHOD_PALETTE["accent_2"], "D"),
]
for tag, label, color, marker in loss_specs:
    line(ax_a, load(tag), color, marker, label)
ax_a.set_yscale("log")
ax_a.set_ylim(top=3.0)
style_ax(ax_a, "(a) Loss components", "Loss (log scale)")
ax_a.legend(loc="upper right", fontsize=6.5, ncol=2)

# ── (b) Real DC-residual band vs full ───────────────────────────────
ax_b = fig.add_subplot(gs[0, 1])
line(ax_b, load("eval_real_dc_resid_band"), METHOD_PALETTE["primary"], "o", "DC resid., band")
line(ax_b, load("eval_real_dc_resid_full"), METHOD_PALETTE["accent_1"], "s", "DC resid., full")
style_ax(ax_b, "(b) Real: DC-residual consistency", "DC residual [$^\\circ$C]")
ax_b.set_xlim(4, 11)
ax_b.legend(loc="center right", fontsize=7)

# ── (c) Real artifact score / out-of-band ratio: stacked mini-panels ─
gs_c = gs[0, 2].subgridspec(2, 1, hspace=0.15)
ax_c1 = fig.add_subplot(gs_c[0])
ax_c2 = fig.add_subplot(gs_c[1], sharex=ax_c1)
line(ax_c1, load("eval_real_artifact_score"), METHOD_PALETTE["primary"], "o", None)
line(ax_c2, load("eval_real_out_of_band_ratio"), METHOD_PALETTE["accent_1"], "s", None)
ax_c1.set_title("(c) Real: artifact score / OOB ratio", loc="left")
ax_c1.set_ylabel("Artifact score", fontsize=8)
ax_c2.set_ylabel("Out-of-band ratio", fontsize=8)
ax_c2.set_xlabel("Training step [$\\times 10^3$]")
plt.setp(ax_c1.get_xticklabels(), visible=False)
ax_c1.set_xlim(4, 11)
ax_c2.set_xlim(4, 11)

# ── (d) Synthetic PSNR ───────────────────────────────────────────────
ax_d = fig.add_subplot(gs[1, 0])
line(ax_d, load("eval_synth_psnr"), METHOD_PALETTE["primary"], "o", None)
style_ax(ax_d, "(d) Synthetic: PSNR", "PSNR [dB]")

# ── (e) Synthetic boundary F1 / region RMSE: stacked mini-panels ────
gs_e = gs[1, 1].subgridspec(2, 1, hspace=0.15)
ax_e1 = fig.add_subplot(gs_e[0])
ax_e2 = fig.add_subplot(gs_e[1], sharex=ax_e1)
line(ax_e1, load("eval_synth_boundary_f1"), METHOD_PALETTE["primary"], "o", None)
line(ax_e2, load("eval_synth_region_rmse"), METHOD_PALETTE["accent_1"], "s", None)
ax_e1.set_title("(e) Synthetic: boundary F1 / RMSE", loc="left")
ax_e1.set_ylabel("Boundary F1", fontsize=8)
ax_e2.set_ylabel("Region RMSE [$^\\circ$C]", fontsize=8)
ax_e2.set_xlabel("Training step [$\\times 10^3$]")
plt.setp(ax_e1.get_xticklabels(), visible=False)

# ── (f) eta / anneal schedule (frozen by design) ─────────────────────
ax_f = fig.add_subplot(gs[1, 2])
line(ax_f, load("train_eta"), METHOD_PALETTE["primary"], "o", "$\\eta$ (solver step size)")
line(ax_f, load("train_anneal"), METHOD_PALETTE["accent_1"], "s", "Prior anneal weight")
style_ax(ax_f, "(f) $\\eta$ / prior-anneal schedule", "Value (dimensionless)")
ax_f.set_ylim(-0.05, 1.35)
ax_f.legend(loc="upper center", fontsize=6.5, ncol=1)
ax_f.annotate(
    "both frozen\n(ACL-025/026)", xy=(0.5, 0.22), xycoords="axes fraction",
    ha="center", va="center", fontsize=7, color="#666666", style="italic",
)

paths = save_fig(fig, "fig15_solver_training_diagnostics")
print("\n".join(str(p) for p in paths))
