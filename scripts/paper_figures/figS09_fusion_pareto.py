"""Supplementary figure S-F9: zero-training fusion baseline on the fine window.

Fidelity (highpass corr vs the drizzle 2x input channel) against the sharpness
proxy (P95 gradient) on the center fine-line window:
  * V9A training-time trajectory (5K-60K),
  * post-hoc linear fusion curves fused(lambda) = (1-lambda)*anchor + lambda*UNet,
  * TGV / drizzle-input reference points and the dominance quadrant of the
    TGV work point.

Data sources (regenerate with algos/ep07_unet_sr/scripts/v9_review/):
    output/ep07_v9_review/fusion_baseline_metrics.csv
    output/ep07_v9_review/v9a_pareto_metrics.csv

Run from the repository root:
    uv run python scripts/paper_figures/figS09_fusion_pareto.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from thermal_core.plotting import (
    METHOD_COLORS,
    savefig_academic,
    setup_academic_style,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REVIEW_DIR = PROJECT_ROOT / "output" / "ep07_v9_review"
OUT_DIR = PROJECT_ROOT / "output" / "paper_figures"

FUSION_STYLE = {
    ("tgv", "v9a60"): dict(color=METHOD_COLORS["accent_1"], ls="-", marker="o",
                           label=r"TGV $+\lambda\cdot$V9A-60K"),
    ("tgv", "v9a20"): dict(color=METHOD_COLORS["accent_1"], ls="--", marker="s",
                           label=r"TGV $+\lambda\cdot$V9A-20K"),
    ("drizzle", "v9a60"): dict(color=METHOD_COLORS["secondary"], ls="-", marker="o",
                               label=r"Drizzle $+\lambda\cdot$V9A-60K"),
    ("drizzle", "v9a20"): dict(color=METHOD_COLORS["secondary"], ls="--", marker="s",
                               label=r"Drizzle $+\lambda\cdot$V9A-20K"),
}
ANNOTATED_STEPS = {"v9a_5k": "5K", "v9a_20k": "20K", "v9a_30k": "30K", "v9a_60k": "60K"}


def main() -> None:
    setup_academic_style()
    fusion = pd.read_csv(REVIEW_DIR / "fusion_baseline_metrics.csv")
    traj = pd.read_csv(REVIEW_DIR / "v9a_pareto_metrics.csv")

    refs = fusion[fusion["kind"] == "reference"].set_index("name")
    tgv_x = refs.loc["tgv", "hp_corr_input"]
    tgv_y = refs.loc["tgv", "sharp_p95"]
    drz_x = refs.loc["input_drizzle", "hp_corr_input"]
    drz_y = refs.loc["input_drizzle", "sharp_p95"]

    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    # Dominance quadrant of the TGV work point.
    ax.axvspan(tgv_x, 1.005, ymin=0, ymax=1, color="#dddddd", alpha=0.0)  # keep xlim space
    ax.fill_betweenx([tgv_y, 1.35], tgv_x, 1.005, color=METHOD_COLORS["accent_1"], alpha=0.08, lw=0)
    ax.annotate(
        "dominates TGV\nwork point", xy=(0.9985, 1.05), ha="right", fontsize=7.5,
        color=METHOD_COLORS["accent_1"],
    )

    # V9A training trajectory.
    steps = traj[traj["step"].notna() & (traj["name"].str.startswith("v9a"))]
    steps = steps.sort_values("step")
    ax.plot(
        steps["hp_corr_input"], steps["sharp_p95"], color="#999999", lw=1.0,
        marker="o", markersize=3.5, label="V9A checkpoints (5K-60K)", zorder=2,
    )
    for _, row in steps.iterrows():
        if row["name"] in ANNOTATED_STEPS:
            ax.annotate(
                ANNOTATED_STEPS[row["name"]],
                xy=(row["hp_corr_input"], row["sharp_p95"]),
                xytext=(3, 4), textcoords="offset points", fontsize=7, color="#555555",
            )

    # Fusion curves.
    fus = fusion[fusion["kind"] != "reference"]
    for (anchor, pred), style in FUSION_STYLE.items():
        sel = fus[(fus["anchor"] == anchor) & (fus["unet_pred"] == pred)].sort_values("lambda")
        if sel.empty:
            continue
        ax.plot(
            sel["hp_corr_input"], sel["sharp_p95"], markersize=3.0, lw=1.2,
            zorder=3, **style,
        )

    # Best dominating candidate.
    best = fus[(fus["anchor"] == "tgv") & (fus["unet_pred"] == "v9a60") & (fus["lambda"] == 0.2)]
    if not best.empty:
        bx, by = best.iloc[0][["hp_corr_input", "sharp_p95"]]
        ax.annotate(
            r"TGV $+\,0.2\cdot$V9A-60K",
            xy=(bx, by), xytext=(bx - 0.0205, by + 0.115), fontsize=7.5,
            arrowprops=dict(arrowstyle="-", color="#666666", lw=0.6),
        )

    # Reference points.
    ax.plot(tgv_x, tgv_y, marker="*", markersize=13, color=METHOD_COLORS["accent_3"],
            ls="none", zorder=4, label="TGV work point")
    ax.plot(drz_x, drz_y, marker="s", markersize=7, color="#222222",
            ls="none", zorder=4, label="Drizzle input channel")

    ax.set_xlim(0.898, 1.005)
    ax.set_ylim(0.42, 1.32)
    ax.set_xlabel("Fidelity: highpass corr vs drizzle input channel")
    ax.set_ylabel("Sharpness proxy: P95 gradient (fine window)")
    ax.legend(fontsize=6.8, loc="lower left", ncol=1)

    savefig_academic(fig, OUT_DIR / "figS09_fusion_pareto.png", close=False)
    savefig_academic(fig, OUT_DIR / "figS09_fusion_pareto.pdf")
    print(f"saved {OUT_DIR / 'figS09_fusion_pareto.png'} (+.pdf)")


if __name__ == "__main__":
    main()
