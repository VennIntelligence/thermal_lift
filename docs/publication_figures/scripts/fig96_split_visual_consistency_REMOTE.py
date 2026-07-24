"""Remote-rendered on the 5090 (run from ~/thermal_lift with `uv run python`).

Fig 96 -- Reconstructions are visually invariant to the split seed
(ACL-077 companion). Rows = arms, columns = phase-stratified split seeds
42/123/456. TEMPERATURE-MAP variant (owner request 2026-07-13, replaces
the earlier high-pass version).

DOMAIN NOTE (why this figure is built the way it is): the neural arms
(depb9v6, depb9v9_3k) reconstruct in ABSOLUTE temperature (~22-26 C,
strictly positive), but the classical arms (TGV, Drizzle) reconstruct in
the background-subtracted high-pass domain (zero-mean, no absolute level).
To show all four rows as temperature maps on one common Celsius scale, we
compose every panel as  [shared scene background]  +  [that panel's own
recovered mid-band structure]:
  * shared background per seed  = gaussian_filter(sigma=10) of the mean of
    the two neural absolute-temperature a-halves for that seed (the real
    low-frequency thermal field of the chip).
  * per-panel structure        = c - gaussian_filter(c, sigma=10)  for
    every arm (removes each arm's own low-freq so only the recovered
    structure rides on the shared background).
So the ABSOLUTE LEVEL is a common scene background shared by all panels;
panels differ only in the recovered mid-band structure. For the neural
arms this equals their true reconstruction; for TGV/Drizzle the absolute
level is the shared background added back for visualization (they do not
reconstruct absolute temperature). Footnote states this.

The a-half FRAME SETS differ between seeds, so panel-to-panel sameness
within a row shows the reconstruction is driven by the data, not the
split. Outputs /tmp/fig96_split_visual_consistency.{png,pdf}. ACL-077.
Font is DejaVu serif (the 5090 has no Times).
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.titlesize": 8,
    "figure.dpi": 110, "savefig.dpi": 300,
})

MS = "output/stage2p5_multisplit_v2"
PATHS = {  # (arm, seed) -> a-half path
    ("Drizzle", 42): "output/stage0g_frc_refined/recons/drizzle_phase_stratified_seed42_a.npy",
    ("Drizzle", 123): f"{MS}/refs/recons/drizzle_phase_stratified_seed123_a.npy",
    ("Drizzle", 456): f"{MS}/refs/recons/drizzle_phase_stratified_seed456_a.npy",
    ("TGV", 42): "output/stage0h_frc_recons/tgv_a.npy",
    ("TGV", 123): f"{MS}/tgv_seed123/tgv_a.npy",
    ("TGV", 456): f"{MS}/tgv_seed456/tgv_a.npy",
    ("Ours, v6 pool", 42): "output/dot_probe_expab/depb9v6_a_corrected.npy",
    ("Ours, v6 pool", 123): f"{MS}/inbox_seed123/depb9v6_a_corrected.npy",
    ("Ours, v6 pool", 456): f"{MS}/inbox_seed456/depb9v6_a_corrected.npy",
    ("Ours, v9 3k", 42): "output/dot_probe_expab/depb9v9_3k_a_corrected.npy",
    ("Ours, v9 3k", 123): f"{MS}/inbox_seed123/depb9v9_3k_a_corrected.npy",
    ("Ours, v9 3k", 456): f"{MS}/inbox_seed456/depb9v9_3k_a_corrected.npy",
}
ARMS = ["Drizzle", "TGV", "Ours, v6 pool", "Ours, v9 3k"]
NEURAL = ["Ours, v6 pool", "Ours, v9 3k"]  # arms that carry absolute temperature
SEEDS = [42, 123, 456]
Y, X, S = 361, 522, 128  # fig10 center-detail ROI
BG_SIGMA = 10.0          # HR px; separates smooth scene background from recovered structure

# --- load raw ROI crops ---
raw = {}
for (arm, seed), p in PATHS.items():
    a = np.load(p).astype(np.float64)
    raw[arm, seed] = a[Y:Y + S, X:X + S]

# --- shared absolute-temperature scene background per seed (from neural arms) ---
bg = {}
for seed in SEEDS:
    neural_mean = np.mean([raw[arm, seed] for arm in NEURAL], axis=0)
    bg[seed] = gaussian_filter(neural_mean, BG_SIGMA)

# --- compose every panel: shared background + that panel's own recovered structure ---
panel = {}
for (arm, seed) in raw:
    c = raw[arm, seed]
    structure = c - gaussian_filter(c, BG_SIGMA)   # strip each arm's own low-freq
    panel[arm, seed] = bg[seed] + structure        # ride structure on the shared bg

# --- common Celsius scale across the whole figure ---
allv = np.concatenate([panel[k].ravel() for k in panel])
vmin, vmax = np.percentile(allv, [1.0, 99.0])

fig, axes = plt.subplots(len(ARMS), len(SEEDS), figsize=(4.6, 5.9))
im = None
for i, arm in enumerate(ARMS):
    for j, seed in enumerate(SEEDS):
        ax = axes[i, j]
        im = ax.imshow(panel[arm, seed], cmap="inferno", vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        ax.set_xticks([]), ax.set_yticks([])
        for sp in ax.spines.values():
            sp.set_color("#888888"), sp.set_linewidth(0.4)
    axes[i, 0].set_ylabel(arm, fontsize=7.5)
for j, seed in enumerate(SEEDS):
    axes[0, j].set_title(f"split seed {seed}", fontsize=8)

# scale bar: 30 px = 300 um (white on the thermal map)
axes[-1, 0].plot([4, 34], [122, 122], color="white", lw=1.8,
                 solid_capstyle="butt")
axes[-1, 0].annotate("300 $\\mu$m", (19, 116), ha="center", va="bottom",
                     fontsize=6, color="white")

fig.suptitle("Same data, three different splits: reconstructions do not "
             "move\n(temperature, a-halves; frame sets differ per seed)",
             x=0.04, y=0.995, ha="left", fontsize=8.5)

fig.subplots_adjust(left=0.085, right=0.86, top=0.9, bottom=0.055,
                    wspace=0.05, hspace=0.07)

# shared Celsius colorbar
cax = fig.add_axes([0.88, 0.055, 0.025, 0.845])
cb = fig.colorbar(im, cax=cax)
cb.set_label("temperature (°C)", fontsize=7)
cb.ax.tick_params(labelsize=6)

# honesty footnote about the shared background for classical arms
fig.text(0.04, 0.012,
         "Absolute level is a shared scene background (low-pass of the "
         "aligned reconstruction); panels differ only in recovered mid-band "
         "structure.\nTGV/Drizzle reconstruct in the background-subtracted "
         "domain — the common background is added back for display.",
         fontsize=5.2, color="#444444", va="bottom")

for ext in ("png", "pdf"):
    fig.savefig(f"/tmp/fig96_split_visual_consistency.{ext}",
                bbox_inches="tight", facecolor="white")
print("done vmin=%.2f vmax=%.2f" % (vmin, vmax))
