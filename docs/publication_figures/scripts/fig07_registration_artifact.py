"""Fig 07 — The +0.5 px grid-convention artifact and its correction (ACL-049).

The solver's output grid carries a designed +0.5 HR px corner convention
(forward_torch.py). Every neural×classical FRC comparison silently paid for
it: (a) the probe measures a ~0.6–0.8 HR px inter-method grid offset for
neural pairs but ~0.03 px for classical pairs; (b) after subtracting the
measured offset, cross-FRC @30µm recovers dramatically (v11×TGV 0.04→0.83)
and FRC sign thrashing in the 45–21µm band disappears (14→0 crossings for
v11×drizzle). This retroactively acquitted the neural arms of "in-band
destruction" claimed in ACL-046/047.

Data: remote_inbox/20260713_dotprobe/offset_probe_summary_stage0h.csv.
Run:  uv run python docs/publication_figures/scripts/fig07_registration_artifact.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

setup_academic_style()

SRC = REPO_ROOT / "remote_inbox/20260713_dotprobe/offset_probe_summary_stage0h.csv"
df = pd.read_csv(SRC).set_index("pair")

PAIRS = ["v11_vs_drz", "c_nodr_vs_drz", "d_dr01_vs_drz", "v11_vs_tgv",
         "tgv_vs_drz", "maptv_vs_drz"]
LABEL = {
    "v11_vs_drz": "V11 $\\times$ drizzle",
    "c_nodr_vs_drz": "C-noDR $\\times$ drizzle",
    "d_dr01_vs_drz": "D-DR0.1 $\\times$ drizzle",
    "v11_vs_tgv": "V11 $\\times$ TGV",
    "tgv_vs_drz": "TGV $\\times$ drizzle",
    "maptv_vs_drz": "MAP-TV $\\times$ drizzle",
}
NEURAL = {"v11_vs_drz", "c_nodr_vs_drz", "d_dr01_vs_drz", "v11_vs_tgv"}
d = df.loc[PAIRS]
y = np.arange(len(PAIRS))[::-1]  # top-to-bottom in listed order

fig, (ax_a, ax_b) = plt.subplots(
    1, 2, figsize=(W_DOUBLE, 2.7), gridspec_kw=dict(width_ratios=[1, 1.25]))

# ── (a) measured inter-method grid offset ────────────────────────────
colors = ["#4C72B0" if p in NEURAL else "#888888" for p in PAIRS]
ax_a.barh(y, d["offset_norm_hr_px"], height=0.6, color=colors, zorder=3)
for yi, v in zip(y, d["offset_norm_hr_px"]):
    ax_a.annotate(f"{v:.2f}", (v, yi), xytext=(3, 0), textcoords="offset points",
                  va="center", fontsize=7,
                  bbox=dict(fc="white", ec="none", pad=0.2), zorder=5)
ax_a.axvline(np.hypot(0.5, 0.5), color="#222222", ls="--", lw=0.9, zorder=2)
ax_a.annotate("designed corner convention\n$\\|(+0.5,+0.5)\\|=0.71$ HR px",
              (np.hypot(0.5, 0.5), 0.02), xycoords=("data", "axes fraction"),
              xytext=(4, 0), textcoords="offset points",
              fontsize=6.5, color="#222222", ha="left", va="bottom")
ax_a.set_yticks(y)
ax_a.set_yticklabels([LABEL[p] for p in PAIRS], fontsize=7.5)
ax_a.set_xlabel("Measured grid offset $\\|\\Delta\\|$ [HR px]")
ax_a.set_xlim(0, 1.02)
ax_a.set_title("(a) Probe: neural grids sit off by half a pixel", loc="left")

# ── (b) FRC@30µm before → after offset correction ────────────────────
before = d["frc_at_30um_before"].to_numpy()
after = d["frc_at_30um_after"].to_numpy()
ax_b.hlines(y, before, after, color="#bbbbbb", lw=1.2, zorder=2)
ax_b.scatter(before, y, s=32, color="#C44E52", zorder=3, label="before correction")
ax_b.scatter(after, y, s=32, color="#4C72B0", zorder=3, label="after correction")
for yi, b, a in zip(y, before, after):
    if abs(a - b) > 0.05:
        ax_b.annotate(f"+{a - b:.2f}", ((a + b) / 2, yi), xytext=(0, 5),
                      textcoords="offset points", ha="center", fontsize=6.5,
                      color="#555555")
ax_b.set_yticks(y)
ax_b.set_yticklabels([])
ax_b.set_xlabel("Cross-FRC @ 30 $\\mu$m period")
ax_b.set_xlim(0, 0.9)
ax_b.set_title("(b) FRC recovers once the offset is removed", loc="left")
ax_b.legend(loc="lower right", fontsize=7)

paths = save_fig(fig, "fig07_registration_artifact")
print("\n".join(str(p) for p in paths))
