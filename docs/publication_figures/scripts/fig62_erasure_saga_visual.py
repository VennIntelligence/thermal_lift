"""Fig 62 — The dot-erasure saga, seen directly in the reconstructions.

Six isolated board defects ("dots", clearly visible in the drizzle
reference) followed across pool generations. The v7-pool arm — the
generation with shallow-depth training scenes — wipes them from the image
(43% of isolated dots erased); the deep-scene fixes bring them back
(v8 4.3%, v9 1.6%, v9-3k 0.0% erased; ACL-066/071/074). Columns are
individual dots selected as: isolated, erased by depb9v7_bin4, visible in
drizzle (top depth_drizzle; ids 750/673/753/1381/925/793 spanning size
bins). Each panel is a 25x25 HR-px crop, per-panel 2-98% gray stretch,
crosshair at the catalog dot position.

Alignment caveat: arms come from three inbox drops (20260710_expab,
20260713_dotprobe, 20260716_v8_verdict), all registration-corrected onto
the same drizzle grid; cross-inbox alignment was verified visually on
shared structure before selection.

Data: remote_inbox reconstructions + output/dot_probe_v7/intermediate/
per_dot_v22_arms.csv (dot catalog & classes).
Run:  uv run python docs/publication_figures/scripts/fig62_erasure_saga_visual.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

cat = pd.read_csv(REPO_ROOT / "output/dot_probe_v7/intermediate/"
                              "per_dot_v22_arms.csv").set_index("dot_id")

ARMS = [
    ("Drizzle", "remote_inbox/20260710_expab/drizzle_a.npy"),
    ("TGV", "remote_inbox/20260710_expab/tgv_a.npy"),
    ("v6 pool", "remote_inbox/20260710_expab/depb9v6_a_corrected.npy"),
    ("v7 pool", "remote_inbox/20260713_dotprobe/depb9v7_bin4_a_corrected.npy"),
    ("v8 pool", "remote_inbox/20260716_v8_verdict/depb9v8_a_corrected.npy"),
    ("v9 pool", "remote_inbox/20260710_expab/depb9v9s2_a_corrected.npy"),
    ("v9 3k", "remote_inbox/20260710_expab/depb9v9_3k_a_corrected.npy"),
]
DOTS = [750, 673, 753, 1381, 925, 793]
W = 12  # half-window

fig, axes = plt.subplots(len(ARMS), len(DOTS),
                         figsize=(W_DOUBLE * 0.62, 5.6))
fig.set_layout_engine("none")

for i, (name, path) in enumerate(ARMS):
    img = np.load(REPO_ROOT / path)
    for j, dot in enumerate(DOTS):
        y, x = int(cat.loc[dot, "y"]), int(cat.loc[dot, "x"])
        c = img[y - W:y + W + 1, x - W:x + W + 1]
        ax = axes[i, j]
        ax.imshow(c, cmap="gray", vmin=np.percentile(c, 2),
                  vmax=np.percentile(c, 98), interpolation="nearest")
        ax.plot(W, W, "+", color="#C44E52", ms=5, mew=0.9, alpha=0.85)
        ax.set_xticks([]), ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color("#aaaaaa"), s.set_linewidth(0.4)
    color = "#C44E52" if name == "v7 pool" else "#222222"
    axes[i, 0].set_ylabel(name, fontsize=7.5, color=color)

for j, dot in enumerate(DOTS):
    axes[0, j].set_title(f"dot {dot}\n{cat.loc[dot, 'size_bin']}",
                         fontsize=7)

fig.text(0.5, 0.015,
         "Isolated-dot erasure rates: v7 43% $\\rightarrow$ v8 4.3% "
         "$\\rightarrow$ v9 1.6% $\\rightarrow$ v9-3k 0.0% "
         "(ACL-066/071/074)", ha="center", va="bottom", fontsize=7,
         style="italic", color="#444444")
fig.suptitle("Shallow-scene training erases real defects; deep-scene pools "
             "restore them", x=0.03, y=0.985, ha="left", fontsize=9)
fig.subplots_adjust(left=0.065, right=0.985, top=0.905, bottom=0.05,
                    wspace=0.06, hspace=0.10)

paths = save_fig(fig, "fig62_erasure_saga_visual")
print("\n".join(str(p) for p in paths))
