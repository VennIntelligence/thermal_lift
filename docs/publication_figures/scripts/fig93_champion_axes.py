"""Fig 93 — Champion candidates on all four acceptance axes (parallel coordinates).

The owner-decision material of ACL-076 drawn as a parallel-coordinates plot
(reads better in print than a radar). Four candidates x four acceptance axes:
cross-FRC @30um (ACL-071/072), isolated-defect erasure (ACL-074), OOD wins /9
(ACL-076), and range_exc (ACL-071/072, log-scaled before normalization).
Each axis is min-max normalized so UP = BETTER everywhere (erased% and
range_exc inverted); raw values are labeled at each vertex.

Missing values are shown as gaps with dotted bridge connectors, never
fabricated: depb9v9_3k has no OOD measurement (open blind spot), and the
classical TGV reference has no dot probe and is itself the OOD reference.

Data: data/champion_axes.csv. Run:
uv run python docs/publication_figures/scripts/fig93_champion_axes.py
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pubfig_style import DATA_DIR, METHOD_STYLE, W_1P5, save_fig, setup_academic_style

setup_academic_style()

df = pd.read_csv(DATA_DIR / "champion_axes.csv", comment="#").set_index("arm")

# ── Axes: (column, header, sub-note, better-direction transform, raw formatter)
# transform maps raw -> "goodness" (monotone increasing = better);
# per-axis min-max normalization is applied afterwards.
AXES = [
    ("frc30", "Cross-FRC\n@ 30 $\\mu$m", "higher better",
     lambda v: v, lambda v: f"{v:.4f}"),
    ("erased_pct", "Isolated defects\nerased [%]", "lower better",
     lambda v: -v, lambda v: f"{v:.2f}"),
    ("ood_wins", "OOD scene\nwins / 9", "higher better",
     lambda v: v, lambda v: f"{v:.0f}/9"),
    ("range_exc_pct", "Range excursion\n(log)", "lower better",
     lambda v: -math.log(v), lambda v: f"{v:.2f}"),
]

ARMS = ["depb9v6", "depb9v9_9bin", "depb9v9_3k", "tgv"]
# Per-figure overrides: task spec draws the classical reference dashed.
LINESTYLE = {"tgv": "--"}

PAD_LO, PAD_HI = 0.06, 0.94  # keep vertices off the axis ends


def normalize() -> dict[str, list[float | None]]:
    """Per-axis min-max on transformed values, mapped into [PAD_LO, PAD_HI]."""
    pos: dict[str, list[float | None]] = {a: [] for a in ARMS}
    for col, _, _, tf, _ in AXES:
        good = {a: tf(df.loc[a, col]) for a in ARMS if pd.notna(df.loc[a, col])}
        lo, hi = min(good.values()), max(good.values())
        for a in ARMS:
            if a in good:
                g = (good[a] - lo) / (hi - lo)
                pos[a].append(PAD_LO + (PAD_HI - PAD_LO) * g)
            else:
                pos[a].append(None)
    return pos


POS = normalize()

fig, ax = plt.subplots(figsize=(W_1P5, 3.7))
ax.set_xlim(-0.42, 3.42)
ax.set_ylim(-0.10, 1.24)
ax.axis("off")

# ── Vertical axis lines + headers ────────────────────────────────────
for i, (_, header, note, _, _) in enumerate(AXES):
    ax.plot([i, i], [0, 1], color="#bbbbbb", lw=0.8, zorder=1)
    ax.annotate(header, (i, 1.115), ha="center", va="bottom", fontsize=8,
                color="#222222")
    ax.annotate(f"({note})", (i, 1.035), ha="center", va="bottom", fontsize=7,
                color="#777777", style="italic")

# "up = better" cue on the far left
ax.annotate("", xy=(-0.36, 0.72), xytext=(-0.36, 0.32),
            arrowprops=dict(arrowstyle="->", color="#666666", lw=0.8))
ax.annotate("better", (-0.36, 0.76), ha="center", va="bottom", fontsize=7,
            color="#666666", rotation=90)

# ── Polylines: solid where measured, dotted bridges across gaps ──────
for arm in ARMS:
    st = METHOD_STYLE[arm]
    ls = LINESTYLE.get(arm, st["ls"])
    ys = POS[arm]
    idx = [i for i, y in enumerate(ys) if y is not None]
    for a, b in zip(idx[:-1], idx[1:]):
        bridging = b - a > 1  # spans one or more missing axes
        ax.plot([a, b], [ys[a], ys[b]],
                color=st["color"], ls=":" if bridging else ls,
                lw=1.1 if bridging else 1.4,
                alpha=0.55 if bridging else 1.0, zorder=2,
                solid_capstyle="round")
    ax.plot(idx, [ys[i] for i in idx], ls="none", marker=st["marker"],
            color=st["color"], ms=5, mec="white", mew=0.5, zorder=3,
            label=st["label"])


def dodge(items: list[tuple[float, str, str]], min_gap: float = 0.062):
    """items = (y, text, color); return dodged label y's, order-preserving."""
    order = sorted(range(len(items)), key=lambda k: items[k][0])
    ys = [items[k][0] for k in order]
    for j in range(1, len(ys)):          # push up
        ys[j] = max(ys[j], ys[j - 1] + min_gap)
    over = ys[-1] - 1.0
    if over > 0:                          # push back down if we ran off the top
        ys = [y - over for y in ys]
        for j in range(len(ys) - 2, -1, -1):
            ys[j] = min(ys[j], ys[j + 1] - min_gap)
    out = [0.0] * len(items)
    for j, k in enumerate(order):
        out[k] = ys[j]
    return out


# ── Raw-value labels at each vertex (dodged per axis side) ───────────
# Outermost axes label on the empty outward side; interior labels get a
# small white bbox so crossing polylines never strike through the text.
for i, (col, _, _, _, fmt) in enumerate(AXES):
    side = -1 if i == 0 else 1                 # axis 0 -> left, others -> right
    items = []
    for arm in ARMS:
        y = POS[arm][i]
        if y is None:
            continue
        items.append((y, fmt(df.loc[arm, col]), METHOD_STYLE[arm]["color"]))
    label_ys = dodge(items)
    for (y, text, color), ly in zip(items, label_ys):
        ax.annotate(text, (i, ly), xytext=(6 * side, 0),
                    textcoords="offset points", fontsize=7,
                    ha="left" if side > 0 else "right", va="center",
                    color=color, zorder=5,
                    bbox=dict(fc="white", ec="none", pad=0.15, alpha=0.85))

# ── Missing-value annotations (gaps are deliberate, not zeros) ───────
GAP_NOTES = [  # (axis idx, arm, text)
    (2, "depb9v9_3k", "untested"),
    (1, "tgv", "not probed"),
    (2, "tgv", "reference"),
]
for i, arm, text in GAP_NOTES:
    ys = POS[arm]
    lo = max(j for j in range(i) if ys[j] is not None)
    hi = min(j for j in range(i + 1, len(ys)) if ys[j] is not None)
    y_cross = ys[lo] + (ys[hi] - ys[lo]) * (i - lo) / (hi - lo)
    ax.annotate(text, (i, y_cross), xytext=(0, -8), textcoords="offset points",
                ha="center", va="top", fontsize=7, style="italic",
                color=METHOD_STYLE[arm]["color"], zorder=4)

# ── Legend + decision-tension note ───────────────────────────────────
fig.legend(loc="lower center", bbox_to_anchor=(0.5, -0.035), ncol=4,
           columnspacing=1.0, handletextpad=0.3)
fig.text(0.5, -0.085,
         "Decision tension: FRC-bound v6 vs dot-fidelity-bound v9-3k; "
         "v9-3k's OOD is the open blind spot.\n"
         "OOD wins are vs the oracle anchor; vs the stronger portable "
         "baseline v6 wins 8/9 in-grammar but only ties out-of-grammar "
         "(ACL-077/078, fig94/95).",
         ha="center", va="top", fontsize=8, style="italic", color="#444444")

ax.set_title("Champion candidates on the four acceptance axes (ACL-076/078)",
             loc="left", y=1.13)

paths = save_fig(fig, "fig93_champion_axes")
print("\n".join(str(p) for p in paths))
