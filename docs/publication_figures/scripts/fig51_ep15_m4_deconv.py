"""Fig 51 — EP15 M4 deconvolution anchor: four-arm comparison + zigzag profiles.

M4 is the classical/reference "baseline-to-beat" for 2.5x SR: a GPU MAP-TV
deconvolution (PSF-sigma scan 0.2-0.5 LR px x lambda_TV {3e-4,1e-3,3e-3},
detector-box forward model, split-half selection proxy) on the 248-frame clean
real thermal session. Selected parameters: sigma_PSF = 0.2 LR px,
lambda_TV = 0.001 (parameter_selection.csv, selected_global row).
Four reconstruction arms on the same real data:
  bare_drizzle | bicubic | map_tv (the M4 classical anchor) | ep07_v6 (neural).

Panels: (a) four-arm temperature ROI montage (center 1/3 of the 2400x3200 HR
field, shared 1-99 pct limits); (b) matching high-pass montage (shared
symmetric limits) revealing resolved fine structure; (c) three zigzag
line-profile overlays (inverted line contrast vs distance, upper/mid/lower)
— MAP-TV vs bare drizzle: 3/3 line pairs separated, median FWHM 220->204 um,
median dip depth 0.915->0.950 (zigzag_profile_metrics.csv / m4_summary.json);
(d) M4 self split-half FRC verification — bare drizzle half-bit cutoff at
22.8 um vs MAP-TV above the half-bit criterion down to the 8 um measurement
floor (frc_verification.csv; self-FRC shown as M4's internal verification
only, not as a cross-method metric — see fig31/32/63 caveats).

EP07 ARM PROVENANCE (honest labeling, do not simplify): the ep07_v6 arm is
NOT the original v6 model. The original ep07_v6_physics checkpoint was
deleted in a pool migration, so it was retrained 2026-07-16 with the
v6-physics loss recipe (grad_vector 0.3 / laplacian 0.1 / forward_model 0.1,
scale 2) on the CURRENT pool `pool_2x_v9_5k` (the original v6 training pool
no longer exists). It is therefore labeled "EP07 (v6 recipe, v9 pool)" both
here and on the figure.

Data: remote_inbox/20260716_ep15_m4/{fig51_data.npz, fig51_meta.json,
      zigzag_profile_metrics.csv, parameter_selection.csv,
      frc_verification.csv, m4_summary.json}  (EP15 M4 run, md5-verified).
Ref:  EP15 M4 "MAP-TV deconvolution anchor" (m4_summary.json task field).
Run:  uv run python docs/publication_figures/scripts/fig51_ep15_m4_deconv.py
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import (
    CMAP_RESID_DIV,
    CMAP_TEMPERATURE,
    METHOD_STYLE,  # noqa: F401  (repo-wide arm styles; see ARM_COLOR note)
    REPO_ROOT,
    W_DOUBLE,
    save_fig,
    setup_academic_style,
)

setup_academic_style()

SRC = REPO_ROOT / "remote_inbox/20260716_ep15_m4"

# This figure is a self-contained four-arm comparison internal to the M4 run;
# it deliberately follows M4's own arm palette (bare_drizzle blue, map_tv red)
# rather than METHOD_STYLE's cross-figure drizzle-orange/maptv-brown, because
# two of the four arms (bicubic, this retrained ep07 arm) have no METHOD_STYLE
# entry and the M4 source plots already fixed blue/red semantics.
ARMS = ["bare_drizzle", "bicubic", "map_tv", "ep07_v6"]
ARM_COLOR = {
    "bare_drizzle": "#4C72B0",
    "bicubic": "#888888",
    "map_tv": "#C44E52",
    "ep07_v6": "#55A868",
}
ARM_TITLE = {
    "bare_drizzle": "Bare drizzle",
    "bicubic": "Bicubic",
    "map_tv": "MAP-TV (M4)",
    "ep07_v6": "EP07 (v6 recipe,\nv9 pool)",
}
ARM_LABEL = {  # legend (single line)
    "bare_drizzle": "Bare drizzle",
    "bicubic": "Bicubic",
    "map_tv": "MAP-TV (M4)",
    "ep07_v6": "EP07 (v6 recipe, v9 pool)",
}

d = np.load(SRC / "fig51_data.npz")
meta = json.loads((SRC / "fig51_meta.json").read_text())
summary = json.loads((SRC / "m4_summary.json").read_text())
frc = pd.read_csv(SRC / "frc_verification.csv")

tvmin, tvmax = (float(v) for v in d["temp_vminmax"])
hvmin, hvmax = (float(v) for v in d["hp_vminmax"])

# ROI geometry: crops are the center 1/3 of the HR field, downsampled to
# 330x440 -> um per display px from meta (pitch 10 um / HR px).
pitch_um = float(meta["pitch_um"])
n_disp_rows = d["temp__bare_drizzle"].shape[0]
um_per_disp_px = (meta["shape"][0] / 3) * pitch_um / n_disp_rows  # ~24.2

fig = plt.figure(figsize=(W_DOUBLE, 6.4))
gs = fig.add_gridspec(3, 1, height_ratios=[1.16, 1.10, 2.45])
gs_t = gs[0].subgridspec(1, 4)
gs_h = gs[1].subgridspec(1, 4)
gs_b = gs[2].subgridspec(3, 2, width_ratios=[2.05, 1.0])
fig.get_layout_engine().set(w_pad=0.02, h_pad=0.02, wspace=0.02, hspace=0.03)

# ── (a) temperature montage ──────────────────────────────────────────
axes_t = [fig.add_subplot(gs_t[0, j]) for j in range(4)]
for ax, arm in zip(axes_t, ARMS):
    im_t = ax.imshow(
        d[f"temp__{arm}"].astype(np.float32),
        cmap=CMAP_TEMPERATURE, vmin=tvmin, vmax=tvmax, interpolation="nearest",
    )
    ax.set_title(ARM_TITLE[arm], fontsize=8, pad=3)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
axes_t[0].set_ylabel("(a) Temperature", fontsize=8)

# 2 mm scale bar (pitch 10 um / HR px -> ~24.2 um per display px)
bar_px = 2000.0 / um_per_disp_px
axes_t[0].plot([14, 14 + bar_px], [306, 306], color="white", lw=2.2,
               solid_capstyle="butt")
axes_t[0].annotate("2 mm", (14 + bar_px / 2, 298), color="white",
                   ha="center", va="bottom", fontsize=7)

cbar_t = fig.colorbar(im_t, ax=axes_t, fraction=0.045, pad=0.015)
cbar_t.set_label("Temperature [$^\\circ$C]", fontsize=8)
cbar_t.ax.tick_params(labelsize=7)

# ── (b) high-pass montage ────────────────────────────────────────────
axes_h = [fig.add_subplot(gs_h[0, j]) for j in range(4)]
for ax, arm in zip(axes_h, ARMS):
    im_h = ax.imshow(
        d[f"hp__{arm}"].astype(np.float32),
        cmap=CMAP_RESID_DIV, vmin=hvmin, vmax=hvmax, interpolation="nearest",
    )
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():  # arm-colored frame, matches zigzag curves
        sp.set_visible(True)
        sp.set_linewidth(1.4)
        sp.set_color(ARM_COLOR[arm])
axes_h[0].set_ylabel("(b) High-pass", fontsize=8)

cbar_h = fig.colorbar(im_h, ax=axes_h, fraction=0.045, pad=0.015)
cbar_h.set_label("High-pass [$^\\circ$C]", fontsize=8)
cbar_h.ax.tick_params(labelsize=7)

# ── (c) zigzag line profiles ─────────────────────────────────────────
PROFILE_TAG = {0: "upper", 1: "mid", 2: "lower"}
axes_z = [fig.add_subplot(gs_b[i, 0]) for i in range(3)]
for i, ax in enumerate(axes_z):
    dist = d[f"zig{i}__dist"]
    for arm in ARMS:  # bicubic/bare first, deconv/neural arms drawn on top
        ax.plot(dist, d[f"zig{i}__{arm}"], color=ARM_COLOR[arm], lw=1.0,
                label=ARM_LABEL[arm] if i == 0 else None,
                zorder=3 if arm in ("map_tv", "ep07_v6") else 2)
    ax.axhline(0, color="#cccccc", lw=0.6, zorder=1)
    ax.set_xlim(0, float(dist[-1]))
    # top headroom (data max 2.10) reserves a data-free legend band
    ax.set_ylim(-0.55, 2.75)
    ax.set_yticks([0, 1, 2])
    ax.tick_params(labelsize=7)
    ax.annotate(PROFILE_TAG[i], (0.012, 0.86), xycoords="axes fraction",
                fontsize=7.5, fontstyle="italic", color="#444444")
    if i < 2:
        ax.set_xticklabels([])
axes_z[0].set_title("(c) Zigzag line profiles (dark traces $\\to$ positive peaks)",
                    loc="left", fontsize=9)
axes_z[0].legend(loc="upper right", fontsize=6.5, ncol=4,
                 borderaxespad=0.2, handlelength=1.1, columnspacing=0.8,
                 handletextpad=0.4)
axes_z[1].set_ylabel("Inverted line contrast [$^\\circ$C]")
axes_z[2].set_xlabel("Distance along profile [$\\mu$m]")

# Verdict numbers from zigzag_profile_metrics.csv medians (= m4_summary.json
# zigzag block): profiles_map_tv_separated 3/3, fwhm 220->204 um,
# dip depth 0.915->0.950 (MAP-TV vs bare drizzle).
axes_z[2].annotate(
    "MAP-TV vs bare drizzle: 3/3 line pairs separated;\n"
    "median FWHM 220$\\to$204 $\\mu$m; median dip depth 0.915$\\to$0.950",
    (0.985, 0.94), xycoords="axes fraction", ha="right", va="top",
    fontsize=6.5, color="#333333",
    bbox=dict(fc="white", ec="#bbbbbb", lw=0.5, boxstyle="round,pad=0.3"),
)

# ── (d) M4 split-half FRC self-verification ──────────────────────────
ax_f = fig.add_subplot(gs_b[:, 1])
PMIN, PMAX = 8.0, 60.0
bare_cut = summary["frc"]["bare_cutoff_period_um"]  # 22.80 um
FRC_LABEL = {  # cutoff facts from m4_summary.json "frc" block
    "bare_drizzle": f"Bare drizzle (cutoff {bare_cut:.1f} $\\mu$m)",
    "map_tv": "MAP-TV ($\\geq$ half-bit to 8 $\\mu$m floor)",
}
for arm in ["bare_drizzle", "map_tv"]:
    g = frc[(frc["method"] == arm)
            & (frc["period_um"] >= PMIN) & (frc["period_um"] <= PMAX)]
    g = g.sort_values("period_um", ascending=False)
    ax_f.plot(g["period_um"], g["frc"], color=ARM_COLOR[arm], lw=1.2,
              label=FRC_LABEL[arm])
g0 = frc[(frc["method"] == "bare_drizzle")
         & (frc["period_um"] >= PMIN) & (frc["period_um"] <= PMAX)]
g0 = g0.sort_values("period_um", ascending=False)
ax_f.plot(g0["period_um"], g0["threshold_half_bit"], color="#999999",
          ls=":", lw=1.0, label="Half-bit criterion")
ax_f.axvline(bare_cut, color=ARM_COLOR["bare_drizzle"], ls=":", lw=0.9)
ax_f.annotate("MAP-TV: $\\sigma_{\\mathrm{PSF}}$ = 0.2 LR px,"
              " $\\lambda_{\\mathrm{TV}}$ = 0.001",
              (0.98, 0.24), xycoords="axes fraction", fontsize=6.5,
              ha="right", va="top", color="#333333")
ax_f.set_xscale("log")
ax_f.set_xlim(PMAX, PMIN)
ticks = [60, 40, 30, 20, 15, 12, 10, 8]
ax_f.set_xticks(ticks)
ax_f.set_xticklabels([str(t) for t in ticks], fontsize=7)
ax_f.minorticks_off()
# bottom shelf below the data range (FRC min -0.26) keeps the legend and the
# sigma/lambda annotation off the curves
ax_f.set_ylim(-0.72, 1.05)
ax_f.set_yticks([-0.2, 0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
ax_f.axhline(0, color="#cccccc", lw=0.6, zorder=0)
ax_f.set_xlabel("Period [$\\mu$m]")
ax_f.set_ylabel("Split-half FRC")
ax_f.set_title("(d) M4 self split-half FRC", loc="left", fontsize=9)
ax_f.legend(loc="lower left", bbox_to_anchor=(0.0, 0.0), fontsize=6.5,
            borderaxespad=0.2, handlelength=1.2)

# Honest EP07 provenance note (must stay on-figure; see docstring).
fig.text(
    0.5, -0.012,
    "EP07 arm: original ep07_v6_physics checkpoint lost in a pool migration;"
    " retrained 2026-07-16 with the v6-physics loss recipe (grad_vector 0.3 /"
    " laplacian 0.1 / forward_model 0.1, scale 2)\non the current pool"
    " pool_2x_v9_5k — not the original v6 model."
    " FRC panel is M4's internal split-half verification (self-FRC), not a"
    " cross-method metric.",
    ha="center", va="top", fontsize=6.5, color="#555555",
)

paths = save_fig(fig, "fig51_ep15_m4_deconv")
print("\n".join(str(p) for p in paths))
