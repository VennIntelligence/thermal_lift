"""Fig 90 — Project chronicle: all ACL entries on a progress axis, six phases.

Redesign 2026-07-24 (owner feedback on the date-axis swimlane version):
the old layout left large horizontal whitespace (entries cluster on a few
dates and stack into tall columns) and spent color on lane titles. New
layout: vertical ACL *progress* axis (uniform density, no whitespace);
six phase bands with BLACK titles + plain-text summaries of what was
actually done in each phase; color appears only on the per-entry dots
(5 theme colors). Because the axis is entry order rather than calendar
time, the Jun-2025-dated prehistory ACL-001..003 can be drawn too.
No footnote (owner: nobody reads it) — the milestone-star semantics live
in the legend, ACL-073's absence is stated in the GALLERY caption.

Data: parsed live from research_log/algorithm_changelog.md headings
      (regex '^### [ACL-'). ACL-073 has no heading in the log (exists only
      as a commit / body reference) and is therefore absent. Phase texts
      are editorial transcriptions of the same changelog (entry numbers
      cited inline for audit).
Run:  uv run python docs/publication_figures/scripts/fig90_project_timeline.py
"""

import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from pubfig_style import (
    METHOD_PALETTE,
    REPO_ROOT,
    W_DOUBLE,
    save_fig,
    setup_academic_style,
)

setup_academic_style()

CHANGELOG = REPO_ROOT / "research_log" / "algorithm_changelog.md"
HEADING_RE = re.compile(r"^### \[ACL-(\d{3})\] (\d{4}-\d{2}-[0-9x]{2})(.*)$")

# ── Themes (dot colors only; all figure text stays black) ────────────
THEMES = [
    "UNet era",
    "Solver architecture",
    "Metrology & calibration",
    "Pools & dot fidelity",
    "Benchmarks & OOD",
]
THEME_COLOR = {
    "UNet era": METHOD_PALETTE["neutral"],
    "Solver architecture": METHOD_PALETTE["primary"],
    "Metrology & calibration": METHOD_PALETTE["accent_3"],
    "Pools & dot fidelity": METHOD_PALETTE["accent_1"],
    "Benchmarks & OOD": METHOD_PALETTE["secondary"],
}
THEME_X = {t: 1.0 + 1.4 * i for i, t in enumerate(THEMES)}

# Keyword rules, checked IN ORDER against the (mostly Chinese) heading
# title; latin keywords matched case-insensitively. First hit wins.
# (sigma-calibration entries are folded into Metrology & calibration.)
LANE_RULES = [
    ("Benchmarks & OOD", ["ood", "oodchain", "2b", "track c"]),
    ("Metrology & calibration", ["自校准", "esf", "σ 线"]),
    ("Pools & dot fidelity",
     ["池", "v7", "v8", "v9", "depb9", "de_pb9", "点保真", "黑点",
      "抹除", "缺陷", "dot", "composer", "motif", "erased"]),
    ("Metrology & calibration",
     ["stage 0", "frc", "shift", "对比", "配准", "偏移", "对齐",
      "仪表", "重标定", "网格", "pitch"]),
    ("Solver architecture",
     ["solver", "prox", "dc", "eta", "η", "unroll", "halo",
      "drizzle", "phase-bin"]),
]
UNET_ERA_MAX = 22          # ACL-001..022 = pre-solver era regardless
OVERRIDES = {
    30: "Pools & dot fidelity",       # v5 GT edge_sigma change = pool change
    52: "Solver architecture",        # eta sweep = DC-weight calibration
    55: "Solver architecture",        # DC sum->mean normalisation
    58: "Metrology & calibration",    # sigma-campaign root cause
    77: "Benchmarks & OOD",           # multi-split champion-ranking verdict
    80: "Metrology & calibration",    # "sharpest != best" re-eval verdict
}
FALLBACK = "Solver architecture"

MILESTONES = {23, 24, 37, 48, 49, 53, 63, 66, 70, 74, 76, 79, 80}

# ── Phases: (roman, lo, hi, dates, title, description lines) ─────────
PHASES = [
    ("I", 1, 22, "Jun 05–14",
     "UNet-era baselines & paper harness",
     ["Residual-UNet SR with loss engineering: V4/V5 anti-adhesion, Laplacian + PSF forward loss (001–005)",
      "TGV baseline repaired: anisotropic regularization + coverage-weighted data term (007)",
      "TCForge anti-aliased training pools; EP12 4x pipeline fixes (008/013/014)",
      "V9A/B/C hybrid drizzle inputs + forward-consistency losses (016–019)",
      "V10 residual-over-observation head (020); paper T1/T2/F5 real-data harness (021/022)"]),
    ("II", 23, 43, "Jun 25–Jul 01",
     "Physics reset & the unrolled solver",
     ["Detector pitch recalibrated to 20 µm + forward operator certified (023)",
      "Commitment: physics-constrained unrolled prox–DC solver, no diffusion (024)",
      "solver v1→v5: hard DC, prior annealing, loss/metric redesign, real-eval hooks (025–031)",
      "de-waffle warm start kills the 2 px checkerboard (032)",
      "K4 glow-box root cause: GroupNorm+SE break extent-invariance (037/040)",
      "→ mainline noSE+noGN + halo-96 inference (038/041); D–E prox arms negative (042/043)"]),
    ("III", 44, 50, "Jul 02–06",
     "Instrument repair: the stage-0 saga",
     ["The ruler was broken (046): per-frame alignment refined (~0.3 px error)",
      "→ authoritative recoverable band 34.07 → 25.45 µm (047/048)",
      "self split-half FRC invalidated: it rewards reproducible hallucination (047)",
      "+0.5 px grid-convention exoneration: cross-FRC 0.04 → 0.83 (049); pipeline locked (050)"]),
    ("IV", 51, 62, "Jul 07–08",
     "Calibration & held-out benches",
     ["η (DC weight) recalibrated 0.5 → 0.09: closes ~35% of the TGV gap (051–053)",
      "Stage-2b held-out synthetic bench: classical arms evaluated on it for the first time (054/055)",
      "σ line: E1/E2 degenerate → E3 ESF kernel PASS (median 4.1%) → robust band [0.1, 0.4] px (056–059)",
      "de_pb9 revival = most balanced arm (061); v6 advantage attributed to recipe, not pool (062)"]),
    ("V", 63, 74, "Jul 08–10",
     "Dot fidelity & pool evolution",
     ["3562-dot real probe: all neural arms attenuate small dark defects; cross-FRC nearly blind (063)",
      "v7 pool pathology: isolated-dot erasure 43%; controls blame pool design, not training (065–067)",
      "L1 audit + 300-scene probes: shallow depth is the culprit; erasure prior needs pool scale (068/069)",
      "repair by attribution: erased 43% → 4.35% (v8) → 1.55% (v9) → 0.00% (v9-3k) (070/072/074)"]),
    ("VI", 75, 80, "Jul 11–13",
     "Final verdicts & the champion",
     ["DC-residual self-audit: erased dots stay detectable in held-out residuals, AUC 0.68–0.84 (075)",
      "13-pool OOD: v6 sweeps oracle 13/13; fidelity–robustness trade-off, 3k worst (076/078/079)",
      "split-robust ranking (077); \u201csharpest \u2260 best\u201d v5 re-eval (080) → champion: depb9v6"]),
]

X_TEXT = 7.8        # left edge of phase text blocks
X_MAX = 21.0        # keep ~3.33 units/inch with the 6.3 in figure width


def classify(num: int, title: str) -> str:
    if num in OVERRIDES:
        return OVERRIDES[num]
    if num <= UNET_ERA_MAX:
        return "UNet era"
    low = title.lower()
    for lane, kws in LANE_RULES:
        if any(kw in low for kw in kws):
            return lane
    return FALLBACK


def parse_entries():
    entries = []
    for line in CHANGELOG.read_text(encoding="utf-8").splitlines():
        m = HEADING_RE.match(line)
        if m:
            num, title = int(m.group(1)), m.group(3)
            entries.append((num, classify(num, title)))
    return entries


def main() -> None:
    entries = parse_entries()
    counts = {t: sum(1 for _, th in entries if th == t) for t in THEMES}
    print(f"{len(entries)} entries; theme counts:", counts)
    for num, theme in sorted(entries):
        print(f"  ACL-{num:03d} -> {theme}")

    fig, ax = plt.subplots(figsize=(W_DOUBLE * X_MAX / 24.0, 8.0))
    ax.set_xlim(0, X_MAX)
    ax.set_ylim(81.2, -1.2)            # inverted: ACL 001 at the top

    # Phase bands, separators, black titles + description text
    for i, (roman, lo, hi, dates, title, lines) in enumerate(PHASES):
        top, bot = lo - 0.5, hi + 0.5
        if i % 2 == 1:
            ax.axhspan(top, bot, color="#000000", alpha=0.035, lw=0)
        if i > 0:
            ax.hlines(top, 0.15, X_MAX - 0.15, color="#aaaaaa", lw=0.6)
        ax.text(X_TEXT, top + 0.95,
                f"Phase {roman} — {title} · {dates}",
                fontsize=7.8, fontweight="bold", color="black",
                ha="left", va="center")
        ax.text(X_MAX - 0.15, top + 0.95, f"ACL {lo:03d}–{hi:03d}",
                fontsize=6.2, color="#444444", ha="right", va="center")
        ax.text(X_TEXT, top + 1.8, "\n".join(lines),
                fontsize=6.2, color="#222222", ha="left", va="top",
                linespacing=1.28)

    # One dot per ACL entry (color = theme; star = milestone)
    for num, theme in entries:
        x, c = THEME_X[theme], THEME_COLOR[theme]
        if num in MILESTONES:
            ax.plot(x, num, marker="*", ms=8.5, color=c,
                    mec="#222222", mew=0.5, ls="none", zorder=4)
            ax.annotate(f"{num:03d}", (x, num), xytext=(4.6, 0),
                        textcoords="offset points", fontsize=5.5,
                        fontweight="bold", color="black",
                        va="center", ha="left", zorder=5)
        else:
            ax.plot(x, num, marker="o", ms=3.4, color=c,
                    mec="white", mew=0.35, ls="none", zorder=3)

    ax.set_yticks(range(10, 81, 10))
    ax.tick_params(axis="y", labelsize=6.5)
    ax.set_ylabel("Algorithm-changelog entry (ACL #, project progress \u2192)")
    ax.set_xticks([])
    ax.spines["bottom"].set_visible(False)
    ax.set_title("Project chronicle: 80 algorithm-changelog entries in six phases")

    handles = [
        Line2D([], [], marker="o", ls="none", ms=4.2, color=THEME_COLOR[t],
               mec="white", mew=0.35, label=t)
        for t in THEMES
    ]
    handles.append(Line2D([], [], marker="*", ls="none", ms=7.5,
                          color="#888888", mec="#222222", mew=0.5,
                          label="milestone"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.006),
              ncol=6, fontsize=6.0, columnspacing=0.7, handletextpad=0.25,
              frameon=False)

    paths = save_fig(fig, "fig90_project_timeline")
    print("saved:", *paths)


if __name__ == "__main__":
    main()
