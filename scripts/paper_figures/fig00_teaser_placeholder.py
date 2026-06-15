"""Generate a BLANK placeholder for the §1 teaser (Figure 1 / F0).

This is intentionally empty — it only sketches the intended panel layout so the
draft has a real image link to render. Replace with the final crop from
`output/paper_figures/fig05_main_visual.png` once the teaser is composed.
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parents[2] / "output" / "paper_figures"
OUT.mkdir(parents=True, exist_ok=True)


def main() -> None:
    fig = plt.figure(figsize=(7.2, 3.0))  # double-column-ish teaser
    fig.patch.set_facecolor("white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(
        0.5, 0.93,
        "[ PLACEHOLDER ]  Figure 1 (F0) — Teaser",
        ha="center", va="center", fontsize=12, fontweight="bold", color="#444",
    )
    ax.text(
        0.5, 0.85,
        "Detailed but Drifting — to be cropped from fig05_main_visual.png",
        ha="center", va="center", fontsize=8, color="#888", style="italic",
    )

    # four empty panels: LR  ->  TGV (faithful)  ->  Learned (sharpest)  ->  drift catch
    labels = [
        "LR / bicubic\n(input)",
        "TGV\nfaithful, mild staircase",
        "Learned\nsharpest by eye",
        "drift catch:\nforward-loss flat,\nartifact proxy rises",
    ]
    n = len(labels)
    pad, top, bot = 0.02, 0.74, 0.10
    w = (1.0 - pad * (n + 1)) / n
    for i, lab in enumerate(labels):
        x0 = pad + i * (w + pad)
        box = FancyBboxPatch(
            (x0, bot), w, top - bot,
            boxstyle="round,pad=0.005",
            linewidth=1.0, edgecolor="#bbb",
            facecolor="#f6f6f6", linestyle="--",
        )
        ax.add_patch(box)
        ax.text(
            x0 + w / 2, (top + bot) / 2, lab,
            ha="center", va="center", fontsize=7.5, color="#999",
        )

    ax.text(
        0.5, 0.04,
        "blank placeholder — replace before submission",
        ha="center", va="center", fontsize=6.5, color="#bbb",
    )

    out_png = OUT / "fig00_teaser_placeholder.png"
    fig.savefig(out_png, dpi=300, facecolor="white")
    plt.close(fig)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
