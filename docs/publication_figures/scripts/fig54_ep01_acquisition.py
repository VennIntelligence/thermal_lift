"""fig54 -- EP01 acquisition-session detection & step-stop raster trajectory.

The raw dataset is 263 static thermal frames named X_Y_R.txt, captured by
step-stop rastering the stage over a 16x16 sub-pixel offset grid
(X, Y in um; 2 um steps over 0-20 um, then 4 um steps to 40 um; detector
pitch 20 um, so the full 40 um throw spans 2 detector pixels). mtime-based
session detection (EP01) splits the 263 frames into three sessions:

  session 0 --   1 frame   prewarm shot (0_0_1)
  session 1 --   7 frames  aborted cold first pass over row Y=0 (~19.9 degC)
  session 2 -- 255 frames  main acquisition (~23.3-23.8 degC), of which
               248 are SR-usable (frame_role=sr_default) and 7 are
               repeat/diagnostic frames (1 main-start repeat 8_0_1 +
               6 post-main row-0 re-checks R=2).

Grid positions (0..8, 0) are covered only by the cold session-1 pass /
repeats (no usable main-session frame), and (14,6), (16,6), (16,16) were
never acquired at all -- hence 248 = 256 - 8 SR inputs.

Panels
------
(a) Raster scan trajectory over the X-Y stage grid: path through the 248
    SR-usable frames colored by acquisition order (row-major raster with
    flyback, starting mid-row at X=10 um because row 0's left edge was
    consumed by the aborted session-1 pass). Excluded frames (all on row
    Y=0) are dodged below the axis by role; never-acquired positions are
    open gray circles.
(b) Acquisition timeline: per-frame mean temperature vs elapsed time.
    The cold aborted pass, the ~3.6 degC jump into the settled main
    session, its slow ~0.6 degC drift, and the post-main re-checks are
    all visible; excluded frames use the role palette.
(c) Frame budget: count breakdown by role (doubles as the role legend),
    with the session split annotated.

Data: output/ep01_data_processing/frame_audit.csv (263 rows, one per raw
      frame; EP01 data-processing audit).
Ref:  EP01 (research_log/episodes/ep01_*).
Run:  uv run python docs/publication_figures/scripts/fig54_ep01_acquisition.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection

from pubfig_style import (
    CMAP_COVERAGE,
    REPO_ROOT,
    W_DOUBLE,
    save_fig,
    setup_academic_style,
)

AUDIT_CSV = REPO_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv"

REQUIRED_COLS = [
    "file", "X", "Y", "R", "mtime", "acquisition_order", "session",
    "is_sr_usable", "frame_role", "T_mean", "T_std",
    "repeat_exclusion_reason",
]

# Qualitative role palette (frame roles, NOT solver arms -> not METHOD_STYLE)
ROLE_STYLE = {
    "sr_default":        dict(color="#4C72B0", marker="o", label="SR-usable (main session)"),
    "repeat_diagnostic": dict(color="#DD8452", marker="^", label="Repeat / diagnostic"),
    "other_session":     dict(color="#C44E52", marker="s", label="Other session (cold pass)"),
}
MISSING_STYLE = dict(color="#888888", label="Never acquired")


def load_audit() -> pd.DataFrame:
    df = pd.read_csv(AUDIT_CSV)
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise KeyError(f"frame_audit.csv missing expected columns: {missing}")
    unknown_roles = set(df["frame_role"].unique()) - set(ROLE_STYLE)
    if unknown_roles:
        raise ValueError(f"Unmapped frame_role values: {unknown_roles}")
    df = df.sort_values("acquisition_order").reset_index(drop=True)
    df["t_min"] = (df["mtime"] - df["mtime"].min()) / 60.0
    return df


def panel_trajectory(ax, fig, df: pd.DataFrame) -> None:
    u = df[df["is_sr_usable"]]
    order = u["acquisition_order"].to_numpy(float)
    pts = u[["X", "Y"]].to_numpy(float)

    # scan path, colored by acquisition order (reveals row-major raster + flyback)
    segs = np.stack([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=CMAP_COVERAGE, lw=0.7, alpha=0.75, zorder=2)
    lc.set_array(0.5 * (order[:-1] + order[1:]))
    ax.add_collection(lc)
    sc = ax.scatter(pts[:, 0], pts[:, 1], c=order, cmap=CMAP_COVERAGE,
                    s=11, lw=0, zorder=3)

    # never-acquired grid positions
    xs, ys = sorted(u["X"].unique()), sorted(u["Y"].unique())
    have = set(zip(df["X"], df["Y"]))
    miss = np.array([(x, y) for x in xs for y in ys if (x, y) not in have], float)
    ax.scatter(miss[:, 0], miss[:, 1], facecolor="none",
               edgecolor=MISSING_STYLE["color"], marker="o", s=26, lw=0.8,
               zorder=3, label=MISSING_STYLE["label"])

    # excluded frames -- all on row Y=0; dodge below the axis, stacked per position
    exc = df[~df["is_sr_usable"]]
    if not (exc["Y"] == 0).all():
        raise ValueError("Layout assumption broken: excluded frames not all on row Y=0")
    for role in ["other_session", "repeat_diagnostic"]:
        st = ROLE_STYLE[role]
        g = exc[exc["frame_role"] == role]
        yoff = np.zeros(len(g))
        for i, (_, r) in enumerate(g.iterrows()):
            k = ((exc["X"] == r["X"]) & (exc.index < r.name)).sum()
            yoff[i] = -2.2 * (k + 1)
        ax.scatter(g["X"], yoff, facecolor="none", edgecolor=st["color"],
                   marker=st["marker"], s=22, lw=0.9, zorder=3, label=st["label"])
    ax.axhline(-1.1, color="#bbbbbb", lw=0.5, ls=":")
    ax.text(-2.5, -9.2, "excluded frames\n(row 0, dodged)", fontsize=6,
            color="#666666", ha="left", va="top")

    # 1 detector pixel bracket (pitch 20 um -> whole throw = 2 px)
    ax.plot([0, 20], [42.8, 42.8], color="#333333", lw=0.9)
    for x in (0, 20):
        ax.plot([x, x], [42.0, 43.6], color="#333333", lw=0.9)
    ax.text(10, 44.2, "1 detector px (20 µm)", ha="center", va="bottom",
            fontsize=6.5, color="#333333")

    ax.set_xlim(-3.5, 43.5)
    ax.set_ylim(-14, 47.5)
    ax.set_aspect("equal")
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([0, 10, 20, 30, 40])
    ax.set_xlabel("Stage offset X [µm]")
    ax.set_ylabel("Stage offset Y [µm]")
    ax.set_title("(a) Step-stop raster trajectory")

    cb = fig.colorbar(sc, ax=ax, location="bottom", fraction=0.05, pad=0.02,
                      aspect=28, shrink=0.92)
    cb.set_label("Acquisition order (of 263 frames)", fontsize=7)
    cb.ax.tick_params(labelsize=7)
    ax.legend(loc="lower right", bbox_to_anchor=(1.03, -0.015), fontsize=5.5,
              handletextpad=0.25, borderpad=0.15, labelspacing=0.25,
              handlelength=1.0)


def panel_timeline(ax, df: pd.DataFrame) -> None:
    u = df[df["is_sr_usable"]]
    st = ROLE_STYLE["sr_default"]
    ax.fill_between(u["t_min"], u["T_mean"] - u["T_std"], u["T_mean"] + u["T_std"],
                    color=st["color"], alpha=0.22, lw=0,
                    label="±1 s.d. (spatial)")
    ax.plot(u["t_min"], u["T_mean"], color=st["color"], lw=1.0,
            label=st["label"])
    for role in ["repeat_diagnostic", "other_session"]:
        rs = ROLE_STYLE[role]
        g = df[df["frame_role"] == role]
        ax.scatter(g["t_min"], g["T_mean"], facecolor="none",
                   edgecolor=rs["color"], marker=rs["marker"], s=16, lw=0.9,
                   zorder=4, label=rs["label"])

    # session extents from the audit itself
    s1 = df[df["session"] == 1]
    s2 = df[df["session"] == 2]
    ax.axvspan(s1["t_min"].min() - 0.3, s1["t_min"].max() + 0.3,
               color="#C44E52", alpha=0.07, lw=0)
    ax.text(-1.2, 21.35, "prewarm\n(session 0)", fontsize=6.5,
            color="#666666", ha="left", va="top")
    ax.annotate("aborted cold pass\n(session 1, 7 fr.)",
                xy=(s1["t_min"].max() + 0.4, s1["T_mean"].iloc[-1]),
                xytext=(14.5, 19.9), fontsize=6.5, color="#C44E52",
                ha="left", va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#C44E52"))
    ax.text(0.5 * (s2["t_min"].min() + s2["t_min"].max()), 24.45,
            "main acquisition (session 2, 255 fr.)", fontsize=6.5,
            color="#333333", ha="center", va="top")
    ax.annotate("row-0 re-checks\n(post-main, R=2)",
                xy=(df["t_min"].iloc[-3], df["T_mean"].iloc[-3] - 0.10),
                xytext=(78, 21.6), fontsize=6.5, color="#DD8452", ha="right",
                va="center",
                arrowprops=dict(arrowstyle="-", lw=0.6, color="#DD8452"))
    # settling jump between sessions
    ax.annotate("", xy=(9.9, 23.6), xytext=(9.9, 20.3),
                arrowprops=dict(arrowstyle="->", lw=0.8, color="#666666"))
    ax.text(11.0, 22.7, "+3.6 °C", fontsize=6.5, color="#666666", ha="left")

    ax.set_xlim(-2, 86)
    ax.set_ylim(19.0, 24.6)
    ax.set_xlabel("Elapsed time [min]")
    ax.set_ylabel("Frame mean T [°C]")
    ax.set_title("(b) Acquisition timeline & thermal state")


def panel_budget(ax, df: pd.DataFrame) -> None:
    roles = ["sr_default", "repeat_diagnostic", "other_session"]
    counts = [int((df["frame_role"] == r).sum()) for r in roles]
    n_post = int((df["repeat_exclusion_reason"] == "post_main_repeat").sum())
    n_start = counts[1] - n_post
    labels = ["SR-usable", "Repeat /\ndiagnostic", "Other\nsession"]
    ypos = np.arange(len(roles))[::-1]
    ax.barh(ypos, counts, height=0.62,
            color=[ROLE_STYLE[r]["color"] for r in roles])
    notes = [
        f"{counts[0]}  (94.3% of {len(df)})",
        f"{counts[1]}  ({n_start} start + {n_post} post-main)",
        f"{counts[2]}  (cold pass)",
    ]
    for y, c, note in zip(ypos, counts, notes):
        ax.text(c + 5, y, note, va="center", ha="left", fontsize=7)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlim(0, 263)
    ax.set_xticks([0, 100, 200, 263])
    ax.set_xlabel("Frames")
    ax.set_title("(c) Frame budget (n = 263, 3 sessions: 1 / 7 / 255)")


def main() -> None:
    setup_academic_style()
    import matplotlib.pyplot as plt

    df = load_audit()
    n_usable = int(df["is_sr_usable"].sum())
    print(f"{len(df)} frames, {df['session'].nunique()} sessions, "
          f"{n_usable} SR-usable")

    fig = plt.figure(figsize=(W_DOUBLE, 3.5))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.32],
                          height_ratios=[1.75, 1.0])
    panel_trajectory(fig.add_subplot(gs[:, 0]), fig, df)
    panel_timeline(fig.add_subplot(gs[0, 1]), df)
    panel_budget(fig.add_subplot(gs[1, 1]), df)

    for p in save_fig(fig, "fig54_ep01_acquisition"):
        print(p)


if __name__ == "__main__":
    main()
