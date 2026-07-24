"""Fig 41 - TCForge composer quality-gate audit (G1-G8).

Per-gate pass/fail summary for the v7 composer-demo dry run (88 scenes,
mid+high occupancy tiers). Each of the 8 gates checks a different pair of
scalar statistics against its own rule (see JSON "rule" strings), so the
metrics are heterogeneous in unit: fractions of image area (G1, G2),
occupancy fractions (G3), a mass fraction + a raw component count (G4),
a tile-std ratio (G5), a raw dot count + two share fractions (G6), a share
fraction (G7), and two defect-share fractions (G8). A single shared x-axis
would be dishonest (units/scales collide, esp. G4's count vs. fraction and
G1/G2's log-spanning floors), so this is drawn as small multiples: one row
per gate, each with axis limits/scale chosen for its own metrics. Gates
with two incompatible units within themselves (G4, G6) get a split
left/right sub-panel instead of forcing both onto one x-axis.

Each measured statistic is a dot; dashed vertical lines mark the gate's
pass/fail threshold(s) (transcribed from the "rule" field). Marker/line
color encodes the gate's overall pass (#55A868) / fail (#C44E52) verdict.

Data: research_log/assets/v7_planning/composer_demo_r4/gate_audit.json
      (top-level "gates" dict; produced by the composer_demo_r4 dry run,
      n_scenes=88; see research_log/algorithm_changelog.md ACL entries for
      the v7 composer gate definitions).
Run:  uv run python docs/publication_figures/scripts/fig41_gate_audit.py
"""

import json

import matplotlib.pyplot as plt
import numpy as np

from pubfig_style import (
    METHOD_PALETTE,
    REPO_ROOT,
    W_DOUBLE,
    save_fig,
    setup_academic_style,
)

setup_academic_style()

DATA_PATH = (
    REPO_ROOT
    / "research_log"
    / "assets"
    / "v7_planning"
    / "composer_demo_r4"
    / "gate_audit.json"
)
gates = json.loads(DATA_PATH.read_text())["gates"]

PASS_C = METHOD_PALETTE["secondary"]  # #55A868
FAIL_C = METHOD_PALETTE["accent_1"]   # #C44E52
THRESH_KW = dict(color="#222222", ls="--", lw=0.9, zorder=1)

MARK = dict(median="o", p95="s", other1="o", other2="s", other3="D", other4="^")


def gate_color(name: str) -> str:
    return PASS_C if gates[name]["pass"] else FAIL_C


def dot_row(ax, points, color, y=0.0, markers=("o", "s", "D", "^")):
    """Draw dot(+range line) for a list of (x, marker_key) on one row."""
    xs = [p[0] for p in points]
    if len(xs) > 1:
        ax.plot([min(xs), max(xs)], [y, y], color=color, lw=1.4, zorder=2)
    for (x, mk), marker in zip(points, markers):
        ax.plot(x, y, marker=marker, color=color, ms=5.5, mec="white",
                 mew=0.5, zorder=3)


fig = plt.figure(figsize=(W_DOUBLE, 8.4))
fig.set_constrained_layout(False)  # manual gridspec margins below instead
gs = fig.add_gridspec(8, 2, hspace=0.85, wspace=0.35,
                       left=0.20, right=0.98, top=0.965, bottom=0.045)

label_kw = dict(fontsize=8, va="center")


def style_row(ax, title, xlabel):
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.set_title(title, loc="left", fontsize=8, fontweight="bold", pad=3)
    ax.set_xlabel(xlabel, fontsize=7.5, labelpad=2)
    ax.tick_params(axis="x", labelsize=7)
    ax.spines["left"].set_visible(False)


# ── G1 bright floor (log, area fraction) ──────────────────────────────
ax = fig.add_subplot(gs[0, :])
g = gates["G1_bright_floor"]
c = gate_color("G1_bright_floor")
dot_row(ax, [(g["median"], "o"), (g["p95"], "s")], c)
for thr, lbl in [(0.005, "median thr"), (0.015, "p95 thr")]:
    ax.axvline(thr, **THRESH_KW)
ax.set_xscale("log")
ax.set_xlim(1e-5, 3e-2)
style_row(ax, "G1 bright floor  (PASS)", "fraction of structure area (log)")
ax.legend(handles=[
    plt.Line2D([], [], marker="o", color=c, mec="white", ls="", label="median"),
    plt.Line2D([], [], marker="s", color=c, mec="white", ls="", label="p95"),
    plt.Line2D([], [], color="#222222", ls="--", lw=0.9, label="threshold"),
], loc="upper left", bbox_to_anchor=(1.01, 1.15), fontsize=6.5, handlelength=1.3,
    borderaxespad=0.0)

# ── G2 gap floor (log, area fraction) ──────────────────────────────────
ax = fig.add_subplot(gs[1, :])
g = gates["G2_gap_floor"]
c = gate_color("G2_gap_floor")
dot_row(ax, [(g["median"], "o"), (g["p95"], "s")], c)
for thr in (0.01, 0.03):
    ax.axvline(thr, **THRESH_KW)
ax.set_xscale("log")
ax.set_xlim(1e-5, 3e-2)
style_row(ax, "G2 gap floor  (PASS)", "fraction of in-bbox background (log)")

# ── G3 occupancy (linear, 4 heterogeneous-threshold points) ────────────
ax = fig.add_subplot(gs[2, :])
g = gates["G3_occupancy"]
c = gate_color("G3_occupancy")
pts = [(g["mid_median"], "o"), (g["high_median"], "s"),
       (g["pool_max"], "D"), (g["pool_min"], "^")]
for x, mk in pts:
    ax.plot(x, 0, marker=mk, color=c, ms=5.5, mec="white", mew=0.5, zorder=3)
ax.axvspan(0.05, 0.16, color="#222222", alpha=0.08, zorder=0)
for thr in (0.16, 0.40, 0.02):
    ax.axvline(thr, **THRESH_KW)
ax.set_xlim(0.0, 0.45)
style_row(ax, "G3 occupancy  (PASS)", "tile occupancy fraction")
ax.legend(handles=[
    plt.Line2D([], [], marker="o", color=c, mec="white", ls="", label="mid median"),
    plt.Line2D([], [], marker="s", color=c, mec="white", ls="", label="high median"),
    plt.Line2D([], [], marker="D", color=c, mec="white", ls="", label="pool max"),
    plt.Line2D([], [], marker="^", color=c, mec="white", ls="", label="pool min"),
], loc="upper left", bbox_to_anchor=(1.01, 1.3), fontsize=6.5, handlelength=1.3,
    borderaxespad=0.0, ncol=1)

# ── G4 fragmentation: split (mass fraction | component count) ──────────
g = gates["G4_fragmentation"]
c = gate_color("G4_fragmentation")
ax_l = fig.add_subplot(gs[3, 0])
ax_l.plot(g["mass_big_median"], 0, marker="o", color=c, ms=5.5, mec="white",
          mew=0.5, zorder=3)
ax_l.axvline(0.90, **THRESH_KW)
ax_l.set_xlim(0.80, 1.02)
style_row(ax_l, "G4 fragmentation  (PASS)", "mass in $\\geq$100px comps [frac]")

ax_r = fig.add_subplot(gs[3, 1])
ax_r.plot(g["n_comp_p95"], 0, marker="s", color=c, ms=5.5, mec="white",
          mew=0.5, zorder=3)
ax_r.axvline(1500, **THRESH_KW)
ax_r.set_xlim(0, 1700)
style_row(ax_r, "", "n components, p95 [count]")

# ── G5 in-scene contrast (linear, tile-std) ─────────────────────────────
ax = fig.add_subplot(gs[4, :])
g = gates["G5_inscene_contrast"]
c = gate_color("G5_inscene_contrast")
dot_row(ax, [(g["v6_tile_std_median"], "o"), (g["v7_tile_std_median"], "s")], c)
thr_x = 1.5 * g["v6_tile_std_median"]
ax.axvline(thr_x, **THRESH_KW)
ax.set_xlim(0.0, 0.40)
style_row(ax, f"G5 in-scene contrast  (FAIL, ratio={g['ratio']:.2f}$\\times$ < 1.5$\\times$)",
          "tile-occupancy std [a.u.]")
ax.legend(handles=[
    plt.Line2D([], [], marker="o", color=c, mec="white", ls="", label="v6 median"),
    plt.Line2D([], [], marker="s", color=c, mec="white", ls="", label="v7 median"),
    plt.Line2D([], [], color="#222222", ls="--", lw=0.9, label="1.5$\\times$ v6 thr"),
], loc="upper left", bbox_to_anchor=(1.01, 1.15), fontsize=6.5, handlelength=1.3,
    borderaxespad=0.0)

# ── G6 dots: split (min dot count | iso/embedded share) ────────────────
g = gates["G6_dots"]
c = gate_color("G6_dots")
ax_l = fig.add_subplot(gs[5, 0])
ax_l.plot(g["min_dots"], 0, marker="o", color=c, ms=5.5, mec="white",
          mew=0.5, zorder=3)
ax_l.axvline(20, **THRESH_KW)
ax_l.set_xlim(10, 30)
style_row(ax_l, "G6 dots  (PASS)", "min dots / scene [count]")

ax_r = fig.add_subplot(gs[5, 1])
dot_row(ax_r, [(g["iso_share"], "s"), (g["emb_share"], "D")], c)
ax_r.axvline(0.15, **THRESH_KW)
ax_r.set_xlim(0.0, 0.85)
style_row(ax_r, "", "iso / embedded share [frac]")
ax_r.legend(handles=[
    plt.Line2D([], [], marker="s", color=c, mec="white", ls="", label="isolated"),
    plt.Line2D([], [], marker="D", color=c, mec="white", ls="", label="embedded"),
], loc="upper left", bbox_to_anchor=(1.02, 1.35), fontsize=6.5, handlelength=1.2,
    borderaxespad=0.0)

# ── G7 levels (linear, share) ───────────────────────────────────────────
ax = fig.add_subplot(gs[6, :])
g = gates["G7_levels"]
c = gate_color("G7_levels")
ax.plot(g["spread_ge_015_share"], 0, marker="o", color=c, ms=5.5, mec="white",
        mew=0.5, zorder=3)
ax.axvline(0.80, **THRESH_KW)
ax.set_xlim(0.0, 1.0)
style_row(ax, "G7 levels  (PASS)", "scenes with level spread $\\geq$0.15 [frac]")

# ── G8 mask defects (linear, share) ─────────────────────────────────────
ax = fig.add_subplot(gs[7, :])
g = gates["G8_mask_defects"]
c = gate_color("G8_mask_defects")
dot_row(ax, [(g["notch_share"], "o"), (g["break_share"], "s")], c)
ax.axvline(0.95, **THRESH_KW)
ax.axvline(0.40, **THRESH_KW)
ax.set_xlim(0.0, 1.05)
style_row(ax, "G8 mask defects  (PASS)", "scene share with $\\geq$1 defect [frac]")
ax.legend(handles=[
    plt.Line2D([], [], marker="o", color=c, mec="white", ls="", label="notch"),
    plt.Line2D([], [], marker="s", color=c, mec="white", ls="", label="break"),
], loc="upper left", bbox_to_anchor=(1.01, 1.15), fontsize=6.5, handlelength=1.3,
    borderaxespad=0.0)

fig.suptitle(
    "TCForge composer quality-gate audit "
    f"(n={json.loads(DATA_PATH.read_text())['n_scenes']} scenes, v7 composer_demo_r4)",
    fontsize=10, fontweight="bold", y=0.995,
)

save_fig(fig, "fig41_gate_audit")
print("saved fig41_gate_audit.png/.pdf")
