#!/usr/bin/env python3
"""fig35 -- L1 zero-training detectability audit: the pathological radius x depth corner.

Story (ACL-068 / ACL-070): the layer-1 input-side audit
(scripts/audit_defect_detectability.py) falsified the hypothesis that the v7
pool's small dark dots are *physically invisible* in the noisy LR burst --
overall they are detectable -- but it located a genuine pathology in the
smallest-radius x shallowest-depth corner: at r~1 HR px and the low depth
tertile, 55% of v7 holes fall below empirical CNR = 3. The v8 pilot pool
(hole count 20-50 -> 2-8 per scene, depth range [0.3,1.0] -> [0.55,1.0])
was gated on this same audit and reduced the corner to ~22%.

Each panel: fraction of hole instances whose EMPIRICAL multi-frame CNR
(aligned mean crop, local-annulus robust noise -- i.e. structured noise that
survives frame averaging is counted) is below 3, binned by rounded HR hole
radius (rows) x within-pool depth_frac tertile (columns). Depth tertiles are
computed within each pool, so the absolute depth ranges (printed in the
column tick labels) differ between panels -- v8's "lo" is deeper than v7's
by design; that recipe change IS the intervention being shown.

Data provenance:
  Windows-5090 box (Administrator@100.98.99.29, WSL user ujs)
    /home/ujs/thermal_lift/output/defect_detectability/v7_5k/     (2026-07-08,
        pool_2x_v7_5k, 200 scenes, 7074 recorded hole instances)
    /home/ujs/thermal_lift/output/defect_detectability/v8_pilot/  (2026-07-09,
        pool_2x_v8_pilot, 24 scenes, 120 recorded hole instances)
  produced by scripts/audit_defect_detectability.py (recorded-instances mode,
  seed 20260709, thresholds 1,2,3,5). per_hole.csv + summary.json copied
  2026-07-13 via base64-over-ssh (md5-verified) into
  remote_inbox/20260713_l1audit/{v7_5k,v8_pilot}/.

Run from repo root:
  uv run python docs/publication_figures/scripts/fig35_l1_detectability_corner.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
from pubfig_style import REPO_ROOT, W_1P5, save_fig, setup_academic_style  # noqa: E402

INBOX = REPO_ROOT / "remote_inbox" / "20260713_l1audit"
CNR_THRESHOLD = 3.0
RADIUS_BINS = [1, 2, 3, 4]
TERTILES = ["lo", "mid", "hi"]
POOLS = [
    ("v7_5k", "v7 pool (5k, pre-fix)"),
    ("v8_pilot", "v8 pilot (post-fix)"),
]


def load_pool(name: str) -> pd.DataFrame:
    df = pd.read_csv(INBOX / name / "per_hole.csv")
    need = {"radius_bin_px", "depth_tertile", "depth_frac", "cnr_empirical"}
    missing = need - set(df.columns)
    if missing:
        raise SystemExit(f"{name}/per_hole.csv missing columns: {missing}")
    return df


def corner_stats(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """(frac below CNR threshold, n) per radius bin x depth tertile."""
    frac = np.full((len(RADIUS_BINS), len(TERTILES)), np.nan)
    n = np.zeros_like(frac)
    for i, rb in enumerate(RADIUS_BINS):
        for j, dt in enumerate(TERTILES):
            v = df.loc[
                (df["radius_bin_px"] == rb) & (df["depth_tertile"] == dt),
                "cnr_empirical",
            ].dropna()
            if len(v):
                frac[i, j] = float((v < CNR_THRESHOLD).mean())
                n[i, j] = len(v)
    return frac, n


def tertile_ranges(df: pd.DataFrame) -> dict[str, tuple[float, float]]:
    g = df.groupby("depth_tertile")["depth_frac"].agg(["min", "max"])
    return {t: (float(g.loc[t, "min"]), float(g.loc[t, "max"])) for t in TERTILES}


def main() -> None:
    setup_academic_style()

    fig, axes = plt.subplots(1, 2, figsize=(W_1P5, 2.7), sharey=True)
    vmax = 0.6
    im = None
    for ax, (name, title) in zip(axes, POOLS):
        df = load_pool(name)
        frac, n = corner_stats(df)
        rng = tertile_ranges(df)
        im = ax.imshow(frac, cmap="YlOrRd", vmin=0.0, vmax=vmax, aspect="auto")
        for i in range(frac.shape[0]):
            for j in range(frac.shape[1]):
                if np.isfinite(frac[i, j]):
                    dark = frac[i, j] > 0.62 * vmax
                    ax.text(j, i, f"{frac[i, j] * 100:.0f}%\nn={int(n[i, j])}",
                            ha="center", va="center", fontsize=7,
                            color="white" if dark else "#222222")
        ax.set_xticks(range(len(TERTILES)),
                      [f"{t}\n{rng[t][0]:.2f}–{rng[t][1]:.2f}" for t in TERTILES])
        ax.set_xlabel("Hole depth tertile (depth fraction)")
        ax.set_title(title)
        # pathological corner = smallest radius x shallowest depth (top-left cell)
        ax.add_patch(Rectangle((-0.5, -0.5), 1.0, 1.0, fill=False,
                               edgecolor="#222222", lw=1.4))
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].set_yticks(range(len(RADIUS_BINS)),
                       [f"{r}" for r in RADIUS_BINS])
    axes[0].set_ylabel("Hole radius bin [HR px]")
    fig.text(0.02, -0.045,
             "Black box: pathological corner (smallest radius $\\times$ shallowest "
             "depth), 55% $\\rightarrow$ 22% after the v8 recipe fix.",
             fontsize=8, ha="left", va="top", color="#222222")
    cbar = fig.colorbar(im, ax=axes, shrink=0.9, pad=0.02)
    cbar.set_label(f"Fraction of holes below empirical CNR = {CNR_THRESHOLD:g} [–]",
                   fontsize=8)
    cbar.ax.tick_params(labelsize=8)

    paths = save_fig(fig, "fig35_l1_detectability_corner")
    for p in paths:
        print(p)


if __name__ == "__main__":
    main()
