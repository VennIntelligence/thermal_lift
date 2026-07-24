"""fig45 -- Stage 2b synthetic-benchmark panorama: all arms on both held-out benches.

Scatter of band-FRC (25-40 um, mean over 48 held-out scenes, n_frames=96) vs
range excursion (log x). One marker per (arm x bench); bench encoded by marker
shape (v6 bench = circle, v8 bench = square), arm family by color (classical
TGV/MAP-TV black/gray, neural v6-recipe pools blue #4C72B0, v8 pool purple
#8172B2, other/corrupted red #C44E52). Ideal corner = top-left (high FRC, low
range excursion). tgv__oracle is ringed as the cross-bench reference
instrument. v20_champion is included but flagged: it is the corrupted
checkpoint documented as an ACL-054-era case study (range_excursion ~1e5).

Provenance
----------
Data (means at n_frames_used = 96, count = 48 scenes each):
  * remote_inbox/20260711_stage2b/stage2b_summary.csv
      first Stage 2b round on the v6 bench: tgv oracle/portable,
      maptv oracle/portable, neural v14 / v17 eta-sweep / v19_etaB /
      v20_champion (corrupted ckpt).
  * remote_inbox/20260717_v8_champion/stage2b_bench_v8/stage2b_summary.csv
      v8-era arms (depb9v8_bin4, depb9v8_9bin) + tgv__oracle on the v8 bench.
  * remote_inbox/20260717_v8_champion/stage2b_v8arms_on_v6bench/stage2b_summary.csv
      same v8-era arms evaluated on the v6 bench (tgv__oracle row there is
      byte-identical to the 20260711 v6-bench row; plotted once).
Context: Stage 2b separates domain-gap (H1) vs architecture (H2) explanations
(ACL-054); v8-era arms show range-excursion inflation (ACL-071).

Run from repo root:
  uv run python docs/publication_figures/scripts/fig45_stage2b_panorama.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pubfig_style import REPO_ROOT, W_DOUBLE, save_fig, setup_academic_style

# ── Data sources ─────────────────────────────────────────────────────
CSV_V6_ROUND1 = (
    REPO_ROOT / "remote_inbox/20260711_stage2b/stage2b_summary.csv"
)
CSV_V8_BENCH = (
    REPO_ROOT
    / "remote_inbox/20260717_v8_champion/stage2b_bench_v8/stage2b_summary.csv"
)
CSV_V8ARMS_V6BENCH = (
    REPO_ROOT
    / "remote_inbox/20260717_v8_champion/stage2b_v8arms_on_v6bench/stage2b_summary.csv"
)

N_FRAMES = 96  # plot the 96-frame operating point of each arm

# Family colors (plotting_standards.md palette; classical in black/gray)
C_TGV = "#111111"
C_MAPTV = "#7a7a7a"
C_V6POOL = "#4C72B0"   # neural, v6-recipe pools (v14 / v17 / v19)
C_V8POOL = "#8172B2"   # neural, v8 pool (depb9v8_*)
C_OTHER = "#C44E52"    # corrupted / other (v20_champion)

FAMILY_COLOR = {
    "tgv__oracle": C_TGV,
    "tgv__portable": C_TGV,
    "maptv__oracle": C_MAPTV,
    "maptv__portable": C_MAPTV,
    "v14": C_V6POOL,
    "v17_eta0p0625": C_V6POOL,
    "v17_eta0p09": C_V6POOL,
    "v17_eta0p125": C_V6POOL,
    "v17_eta0p1875": C_V6POOL,
    "v17_eta0p25": C_V6POOL,
    "v17_eta1p0": C_V6POOL,
    "v17_eta2p0": C_V6POOL,
    "v19_etaB": C_V6POOL,
    "v20_champion": C_OTHER,
    "depb9v8_bin4": C_V8POOL,
    "depb9v8_9bin": C_V8POOL,
}

# Portable classical variants drawn as open markers (same hue as oracle)
OPEN_FACE = {"tgv__portable", "maptv__portable"}

SHORT = {
    "tgv__oracle": "TGV oracle",
    "tgv__portable": "TGV portable",
    "maptv__oracle": "MAP-TV oracle",
    "maptv__portable": "MAP-TV portable",
    "v14": "v14",
    "v17_eta0p0625": r"$\eta$=.0625",
    "v17_eta0p09": r"$\eta$=.09",
    "v17_eta0p125": r"$\eta$=.125",
    "v17_eta0p1875": r"$\eta$=.1875",
    "v17_eta0p25": r"$\eta$=.25",
    "v17_eta1p0": r"$\eta$=1.0",
    "v17_eta2p0": r"$\eta$=2.0",
    "v19_etaB": r"v19 $\eta$B",
    "v20_champion": "v20 champion",
    "depb9v8_bin4": "v8 bin4",
    "depb9v8_9bin": "v8 9-bin",
}

BENCH_MARKER = {"v6": "o", "v8": "s"}


def load_means(path: Path) -> pd.DataFrame:
    """Return DataFrame indexed by arm with metric means at N_FRAMES."""
    df = pd.read_csv(path, header=[0, 1], index_col=[0, 1])
    df = df.xs("mean", axis=1, level=1)
    df = df.xs(N_FRAMES, level=1)
    return df[["frc_band_mean_25_40", "range_excursion"]]


def collect_points() -> pd.DataFrame:
    rows = []
    for path, bench in (
        (CSV_V6_ROUND1, "v6"),
        (CSV_V8ARMS_V6BENCH, "v6"),
        (CSV_V8_BENCH, "v8"),
    ):
        means = load_means(path)
        for arm, r in means.iterrows():
            rows.append(
                dict(
                    arm=arm,
                    bench=bench,
                    frc=r["frc_band_mean_25_40"],
                    rexc=r["range_excursion"],
                )
            )
    pts = pd.DataFrame(rows)
    # tgv__oracle on the v6 bench appears identically in two files; keep one.
    pts = pts.drop_duplicates(subset=["arm", "bench"]).reset_index(drop=True)
    return pts


# Manual label offsets (points, relative to marker) tuned against the render.
# key = (arm, bench): (dx, dy, ha, va)
LEADER = {("tgv__portable", "v6")}  # labels far enough to need a leader line
LABEL_POS = {
    ("tgv__oracle", "v6"): (6, -12, "left", "top"),
    ("tgv__portable", "v6"): (10, 16, "left", "bottom"),
    ("maptv__oracle", "v6"): (2, -9, "center", "top"),
    ("maptv__portable", "v6"): (0, -9, "center", "top"),
    ("v14", "v6"): (-6, 0, "right", "center"),
    ("v17_eta0p0625", "v6"): (-6, 4, "right", "center"),
    ("v17_eta0p09", "v6"): (5, 3, "left", "center"),
    ("v17_eta0p125", "v6"): (6, -8, "left", "top"),
    ("v17_eta0p1875", "v6"): (6, 2, "left", "center"),
    ("v17_eta0p25", "v6"): (0, 7, "center", "bottom"),
    ("v17_eta1p0", "v6"): (0, 7, "center", "bottom"),
    ("v17_eta2p0", "v6"): (0, -9, "center", "top"),
    ("v19_etaB", "v6"): (-6, -5, "right", "center"),
    ("v20_champion", "v6"): (0, 8, "center", "bottom"),
    ("depb9v8_bin4", "v6"): (6, -1, "left", "center"),
    ("depb9v8_9bin", "v6"): (0, 7, "center", "bottom"),
    ("tgv__oracle", "v8"): (8, 0, "left", "center"),
    ("depb9v8_bin4", "v8"): (6, 0, "left", "center"),
    ("depb9v8_9bin", "v8"): (6, 0, "left", "center"),
}


def main() -> None:
    setup_academic_style()
    pts = collect_points()

    fig, ax = plt.subplots(figsize=(W_DOUBLE, 4.0))
    ax.set_xscale("log")

    for _, p in pts.iterrows():
        color = FAMILY_COLOR[p.arm]
        marker = BENCH_MARKER[p.bench]
        open_face = p.arm in OPEN_FACE
        # White edge on filled markers keeps overlapping points (e.g. the
        # near-coincident v17 eta=.0625 / v19_etaB pair) separable.
        ax.scatter(
            p.rexc,
            p.frc,
            s=30,
            marker=marker,
            facecolor="white" if open_face else color,
            edgecolor=color if open_face else "white",
            linewidths=0.9 if open_face else 0.6,
            zorder=3,
        )
        dx, dy, ha, va = LABEL_POS[(p.arm, p.bench)]
        ax.annotate(
            SHORT[p.arm],
            xy=(p.rexc, p.frc),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=ha,
            va=va,
            fontsize=7,
            color="#222222",
            zorder=4,
            arrowprops=(
                dict(arrowstyle="-", lw=0.5, color="#999999",
                     shrinkA=1, shrinkB=3)
                if (p.arm, p.bench) in LEADER
                else None
            ),
        )

    # Reference instrument: ring tgv__oracle on both benches
    ref = pts[(pts.arm == "tgv__oracle")]
    for _, p in ref.iterrows():
        ax.scatter(
            p.rexc, p.frc, s=110, marker=BENCH_MARKER[p.bench],
            facecolor="none", edgecolor="#111111", linewidths=0.7, zorder=2,
        )
    ax.annotate(
        "rings: reference instrument\n(TGV oracle, both benches)",
        xy=(5.2, 0.60),
        ha="left", va="top", fontsize=7, color="#444444", style="italic",
    )

    # v17 eta-sweep cue (short eta labels belong to the v17 recipe sweep)
    ax.annotate(
        r"$\eta$ points: v17 recipe sweep",
        xy=(6.5, 0.935),
        ha="left", va="center", fontsize=7, color=C_V6POOL, style="italic",
    )

    # Corrupted-checkpoint flag on v20_champion
    v20 = pts[(pts.arm == "v20_champion")].iloc[0]
    ax.annotate(
        "corrupted checkpoint\n(ACL-054 case study)",
        xy=(v20.rexc, v20.frc),
        xytext=(-8, -14),
        textcoords="offset points",
        ha="right", va="top", fontsize=7, color=C_OTHER, style="italic",
    )

    # Ideal corner cue (top-left)
    ax.annotate(
        "ideal",
        xy=(0.70, 0.952),
        xytext=(1.7, 0.905),
        ha="left", va="center", fontsize=8, color="#555555",
        arrowprops=dict(arrowstyle="->", lw=0.8, color="#555555"),
    )

    ax.set_xlabel("Range excursion (log scale)")
    ax.set_ylabel(r"Band FRC, 25$-$40 $\mu$m")
    ax.set_title("Stage 2b panorama: all arms on both held-out benches")
    ax.set_ylim(0.30, 0.97)
    ax.set_xlim(0.6, 4e5)

    # Legend: color = arm family, shape = bench
    def mk(marker, color, label, face=None):
        return Line2D(
            [], [], marker=marker, linestyle="none", markersize=5,
            markerfacecolor=color if face is None else face,
            markeredgecolor=color, markeredgewidth=0.9, label=label,
        )

    handles = [
        mk("o", C_TGV, "TGV (classical)"),
        mk("o", C_MAPTV, "MAP-TV (classical)"),
        mk("o", C_TGV, "portable variant", face="white"),
        mk("o", C_V6POOL, "neural, v6-recipe pools"),
        mk("o", C_V8POOL, "neural, v8 pool"),
        mk("o", C_OTHER, "v20 (corrupted)"),
        mk("o", "#555555", "v6 bench"),
        mk("s", "#555555", "v8 bench"),
    ]
    ax.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        ncol=4,
        columnspacing=1.2,
        handletextpad=0.4,
    )

    paths = save_fig(fig, "fig45_stage2b_panorama")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
