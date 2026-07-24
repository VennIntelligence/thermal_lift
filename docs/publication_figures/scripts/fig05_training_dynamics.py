"""Fig 05 — Training dynamics: unrolled physics solver vs plain UNet baseline.

Six panels of TensorBoard scalars over training steps, comparing the
physics-constrained unrolled solver (solver_v5_sharp_hybrid, 20k steps) with
the V10 residual UNet baseline (v10_v5_sharp, 50k steps) on identical eval
hooks: real-domain artifact score / out-of-band ratio, synthetic PSNR /
boundary F1 / region RMSE, and total training loss.

Data: remote_inbox/20260627_checkpoint_evolution/20260628_hybrid_solver/
      {solver_v5_sharp_hybrid,v10_v5_sharp}_scalars.csv (TB exports).
Run:  uv run python docs/publication_figures/scripts/fig05_training_dynamics.py
"""

import matplotlib.pyplot as plt
import pandas as pd

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

SRC = REPO_ROOT / "remote_inbox/20260627_checkpoint_evolution/20260628_hybrid_solver"
solver = pd.read_csv(SRC / "solver_v5_sharp_hybrid_scalars.csv")
unet = pd.read_csv(SRC / "v10_v5_sharp_scalars.csv")

RUNS = [
    ("Unrolled solver (ours)", solver, "#4C72B0", "o"),
    ("Residual UNet (V10)", unet, "#C44E52", "s"),
]

PANELS = [
    ("eval_real/artifact_score", "(a) Real: artifact score", "Artifact score (lower better)", False),
    ("eval_real/out_of_band_ratio", "(b) Real: out-of-band ratio", "Out-of-band ratio (lower better)", False),
    ("eval_synth/psnr", "(c) Synthetic: PSNR", "PSNR [dB] (higher better)", False),
    ("eval_synth/boundary_f1", "(d) Synthetic: boundary F1", "Boundary F1 (higher better)", False),
    ("eval_synth/region_rmse", "(e) Synthetic: region RMSE", "Region RMSE [$^\\circ$C] (lower better)", False),
    ("loss/total", "(f) Training loss", "Total loss", True),
]

fig, axes = plt.subplots(2, 3, figsize=(W_DOUBLE, 4.6))

for ax, (tag, title, ylab, logy) in zip(axes.flat, PANELS):
    for label, df, color, marker in RUNS:
        d = df[df["tag"] == tag].sort_values("step")
        if d.empty:
            continue
        dense = len(d) > 30
        ax.plot(d["step"] / 1000, d["value"], color=color,
                marker=None if dense else marker, markersize=3.5,
                lw=1.2, label=label)
    if logy:
        ax.set_yscale("log")
    ax.set_title(title, loc="left")
    ax.set_ylabel(ylab)
    ax.set_xlabel("Training step [$\\times 10^3$]")

axes[0, 0].legend(loc="best", fontsize=7)

paths = save_fig(fig, "fig05_training_dynamics")
print("\n".join(str(p) for p in paths))
