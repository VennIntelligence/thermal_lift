"""Fig 52 -- INR/decoder priors vs classical MAP-TV for 2x contour SR (EP08).

(a) Stage 2 TCForge synthetic benchmark (lr_shape=(256,256), n_frames=32,
    scale=2, HR highpass GT domain): highpass PSNR and global SSIM proxy for
    the four learned-prior methods. Confirms all four close the forward/
    highpass loop correctly (PSNR > 18 dB) before trusting real-data numbers.
(b) Real 32-frame patch five-method comparison (seed=42 phase-stratified
    split, 256x256 LR patch, all rows gate=complete): the three
    decision-relevant stability/fidelity metrics the EP08 README uses to
    reach its verdict -- split-half NRMSE, artifact score, and raw-control
    agreement. SIREN (highlighted, #4C72B0) is the Stage 3 recommendation:
    best split-half stability and near-best raw-control agreement among the
    learned priors, while WIRE and DeepInverse-DIP show materially higher
    artifact risk despite lower hold-out residual.

Data: docs/publication_figures/data/inr_methods.csv (transcribed from
research_log/episodes/ep08_inr_sr/README.md Stage 2 tables).
Run:  uv run python docs/publication_figures/scripts/fig52_inr_methods.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import DATA_DIR, METHOD_PALETTE, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

df = pd.read_csv(DATA_DIR / "inr_methods.csv", comment="#")

SYN_METHODS = ["SIREN", "WIRE", "Deep Decoder", "DeepInverse-DIP"]
REAL_METHODS = ["EP06 MAP-TV", "SIREN", "WIRE", "Deep Decoder", "DeepInverse-DIP"]

# Fixed colors: SIREN highlighted per instructions; MAP-TV as classic-baseline
# neutral gray; remaining learned priors from the secondary palette slots.
COLOR = {
    "SIREN": METHOD_PALETTE["primary"],      # #4C72B0
    "WIRE": METHOD_PALETTE["accent_3"],      # #DD8452
    "Deep Decoder": METHOD_PALETTE["secondary"],  # #55A868
    "DeepInverse-DIP": METHOD_PALETTE["accent_1"],  # #C44E52
    "EP06 MAP-TV": "#666666",
}

fig = plt.figure(figsize=(W_DOUBLE, 4.6), layout=None)
fig.set_constrained_layout(False)
gs = fig.add_gridspec(3, 2, width_ratios=[1.0, 1.35], height_ratios=[1, 1, 1],
                       hspace=0.65, wspace=0.35, top=0.86, bottom=0.14,
                       left=0.09, right=0.98)

# ── Panel (a): synthetic TCForge benchmark, stacked PSNR / SSIM bars ──────
syn = df[df["panel"] == "synthetic"].set_index("method").loc[SYN_METHODS]
x = np.arange(len(SYN_METHODS))
colors_a = [COLOR[m] for m in SYN_METHODS]

SYN_METRICS = [
    ("highpass_psnr_db", "Highpass PSNR [dB]", 28),
    ("global_ssim_proxy", "Global SSIM proxy", 1.0),
]
for row, (col, label, ymax) in enumerate(SYN_METRICS):
    ax_a = fig.add_subplot(gs[row, 0])
    vals = syn[col].values
    ax_a.bar(x, vals, width=0.6, color=colors_a, edgecolor="white", linewidth=0.6)
    ax_a.set_ylabel(label, fontsize=7.5)
    ax_a.set_ylim(0, ymax)
    ax_a.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax_a.set_axisbelow(True)
    fmt = "{:.1f}" if col == "highpass_psnr_db" else "{:.2f}"
    for xi, v in zip(x, vals):
        ax_a.annotate(fmt.format(v), (xi, v), ha="center", va="bottom", fontsize=6.5)
    if row == 0:
        ax_a.set_xticks(x)
        ax_a.set_xticklabels([])
        ax_a.set_title("(a) Synthetic TCForge benchmark\n(HR highpass GT domain, $n{=}32$ frames)",
                        loc="left", fontsize=9)
    else:
        ax_a.set_xticks(x)
        ax_a.set_xticklabels(SYN_METHODS, rotation=20, ha="right", fontsize=7.5)

# Bottom-left cell: brief takeaway note instead of a third synthetic bar.
ax_note = fig.add_subplot(gs[2, 0])
ax_note.axis("off")
ax_note.text(
    0.0, 1.0,
    "All four methods exceed 18 dB highpass PSNR on the\n"
    "TCForge GT benchmark, confirming the forward/highpass\n"
    "wrapper is geometry-correct before trusting real-data\n"
    "numbers in (b).",
    ha="left", va="top", fontsize=7.5, color="#333333", transform=ax_note.transAxes,
    wrap=True,
)

# ── Panel (b): real-patch decision metrics, 3 stacked small multiples ─────
REAL_METRICS = [
    ("split_half_nrmse", "Split-half NRMSE"),
    ("artifact_score", "Artifact score"),
    ("raw_control_agreement", "Raw-control agreement"),
]
real = df[df["panel"] == "real"].set_index("method").loc[REAL_METHODS]

xr = np.arange(len(REAL_METHODS))
colors_b = [COLOR[m] for m in REAL_METHODS]
edge_b = ["#333333" if m == "SIREN" else "none" for m in REAL_METHODS]
lw_b = [1.3 if m == "SIREN" else 0.0 for m in REAL_METHODS]

for row, (col, label) in enumerate(REAL_METRICS):
    ax_b = fig.add_subplot(gs[row, 1])
    vals = real[col].values
    ax_b.bar(xr, vals, width=0.6, color=colors_b, edgecolor=edge_b, linewidth=lw_b)
    ax_b.set_ylabel(label, fontsize=7.5)
    ax_b.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax_b.set_axisbelow(True)
    ymax = vals.max()
    ax_b.set_ylim(0, ymax * 1.28)
    for xi, v in zip(xr, vals):
        ax_b.annotate(f"{v:.2f}", (xi, v), ha="center", va="bottom", fontsize=6.2)
    if row < len(REAL_METRICS) - 1:
        ax_b.set_xticks(xr)
        ax_b.set_xticklabels([])
    else:
        ax_b.set_xticks(xr)
        ax_b.set_xticklabels(REAL_METHODS, rotation=20, ha="right", fontsize=7.5)
    if row == 0:
        ax_b.set_title("(b) Real 32-frame patch comparison\n(seed=42 split, all rows gate=complete)",
                        loc="left", fontsize=9)

fig.suptitle("EP08: learned-prior 2$\\times$ contour SR vs classical MAP-TV",
             fontsize=11, fontweight="bold", x=0.02, y=0.995, ha="left")

paths = save_fig(fig, "fig52_inr_methods")
print("\n".join(str(p) for p in paths))
