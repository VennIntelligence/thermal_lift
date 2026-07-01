#!/usr/bin/env python3
"""Side-by-side preview figures for the v6 CPU/part synthetic pool.

Renders (all with interpolation='nearest', small ΔT, matched inferno scale):
  fig1_motif_grid.png    — HR ground-truth variety across CPU/part motifs
  fig2_recoverability.png — HR GT | one LR frame | drizzle-fused reconstruction
  fig3_vs_real_anchor.png — a real IR frame vs a v6 LR frame vs v6 HR GT
  fig4_zoom_grid.png     — zoomed centre crops of fine array motifs (GT vs LR)

Usage:
  uv run python scripts/preview_v6_cpu.py \
    --pool outputs/data_gen_v6_cpu_preview/pool_sample \
    --figs outputs/data_gen_v6_cpu_preview/figs \
    --anchor data/data_raw/infrared_avi/0_0_0.txt
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

CMAP = "inferno"


def _dlabel(s: dict) -> str:
    """Continuous-density label (the difficulty tier is vestigial in the v6 CPU pool)."""
    d = s.get("density")
    return f"d={d:.2f}" if d is not None else "d=?"


def _pct(img: np.ndarray, lo: float = 1.0, hi: float = 99.0) -> tuple[float, float]:
    a = np.asarray(img, dtype=np.float64)
    return float(np.percentile(a, lo)), float(np.percentile(a, hi))


def _load_scene(scene_dir: Path) -> dict:
    meta = json.loads((scene_dir / "metadata.json").read_text())
    gt = np.load(scene_dir / "hr_temperature_2x.npy").astype(np.float32)
    lr = np.load(scene_dir / "lr_burst.npy", mmap_mode="r")
    pbd = np.load(scene_dir / "phase_bin_drizzle_2x.npy", mmap_mode="r").astype(np.float32)
    geo = meta.get("geometry_metadata", {})
    fam = geo.get("scene_family", "?")
    dens = geo.get("density")
    return {
        "id": meta["scene_id"],
        "difficulty": meta["difficulty"],
        "density": (float(dens) if dens is not None else None),
        "family": fam,
        "delta_T": float(meta["delta_T_c"]),
        "gt": gt,
        "lr_mid": np.asarray(lr[lr.shape[0] // 2], dtype=np.float32),
        "drizzle": np.asarray(pbd.mean(axis=0), dtype=np.float32),
    }


def _load_anchor(path: Path) -> np.ndarray:
    # 480x640 tab-separated °C matrix with a trailing tab → read first 640 cols.
    return np.loadtxt(path, delimiter="\t", usecols=range(640)).astype(np.float32)


def fig_motif_grid(scenes: list[dict], out: Path) -> None:
    # Prefer coverage of distinct families, then fill to 16.
    ordered: list[dict] = []
    seen: set[str] = set()
    for s in scenes:
        if s["family"] not in seen:
            ordered.append(s)
            seen.add(s["family"])
    for s in scenes:
        if s not in ordered:
            ordered.append(s)
    picks = ordered[:16]
    n = len(picks)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.1 * rows))
    fig.suptitle(
        "v6 CPU/part pool — HR GT variety (continuous density, constant-large chips, "
        "28µm feature floor; inferno, nearest)",
        fontsize=13, y=0.995,
    )
    axes = np.atleast_2d(axes)
    for i, ax in enumerate(axes.ravel()):
        if i >= n:
            ax.axis("off")
            continue
        s = picks[i]
        lo, hi = _pct(s["gt"])
        ax.imshow(s["gt"], cmap=CMAP, vmin=lo, vmax=hi, interpolation="nearest")
        ax.set_title(f"{s['family']} [{_dlabel(s)}]  ΔT={s['delta_T']:.1f}°C\n{s['id']}",
                     fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_recoverability(scenes: list[dict], out: Path) -> None:
    # Pick fine-array-heavy families where recoverability is visible.
    prefer = ("pga_grid", "die_bga", "heat_spreader", "multi_die", "trace_bus")
    picks: list[dict] = []
    for fam in prefer:
        for s in scenes:
            if s["family"] == fam:
                picks.append(s)
                break
        if len(picks) == 3:
            break
    while len(picks) < 3 and len(picks) < len(scenes):
        for s in scenes:
            if s not in picks:
                picks.append(s)
                break
    fig, axes = plt.subplots(len(picks), 3, figsize=(13.5, 4.3 * len(picks)))
    fig.suptitle(
        "v6 recoverability — HR GT | one LR frame (2× coarser + noise) | 4-bin drizzle fused",
        fontsize=13, y=0.997,
    )
    axes = np.atleast_2d(axes)
    for r, s in enumerate(picks):
        lo, hi = _pct(s["gt"])
        panels = [
            (s["gt"], f"HR GT 960×1280\n{s['family']} [{_dlabel(s)}]"),
            (s["lr_mid"], "LR frame 480×640 (aliased + FPN/grain)"),
            (s["drizzle"], "drizzle fused 960×1280"),
        ]
        for c, (img, title) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap=CMAP, vmin=lo, vmax=hi, interpolation="nearest")
            ax.set_title(title, fontsize=9)
            ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_vs_real(scenes: list[dict], anchor: np.ndarray, out: Path) -> None:
    # Prefer a filled die_bga (round-2 fix) so the improved solid-hot-die-with-routing
    # match to the real anchor is directly visible; fall back to pga_grid, then any scene.
    s = (next((x for x in scenes if x["family"] == "die_bga"), None)
         or next((x for x in scenes if x["family"] == "pga_grid"), None)
         or scenes[0])
    alo, ahi = _pct(anchor)
    llo, lhi = _pct(s["lr_mid"])
    glo, ghi = _pct(s["gt"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "Realism check — real IR anchor vs v6 synthetic (inferno, nearest, per-panel 1–99% scale)",
        fontsize=13,
    )
    axes[0].imshow(anchor, cmap=CMAP, vmin=alo, vmax=ahi, interpolation="nearest")
    axes[0].set_title(f"REAL anchor 480×640\n{alo:.2f}–{ahi:.2f}°C", fontsize=10)
    axes[1].imshow(s["lr_mid"], cmap=CMAP, vmin=llo, vmax=lhi, interpolation="nearest")
    axes[1].set_title(f"v6 LR frame 480×640 [{s['family']}/{_dlabel(s)}]\n{llo:.2f}–{lhi:.2f}°C",
                      fontsize=10)
    axes[2].imshow(s["gt"], cmap=CMAP, vmin=glo, vmax=ghi, interpolation="nearest")
    axes[2].set_title(f"v6 HR GT 960×1280\nΔT={s['delta_T']:.1f}°C", fontsize=10)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def fig_zoom(scenes: list[dict], out: Path) -> None:
    # Centre crops of the finest array motifs: GT (2×) vs matching LR (1×).
    prefer = ("pga_grid", "die_bga", "heat_spreader")
    picks: list[dict] = []
    for fam in prefer:
        for s in scenes:
            if s["family"] == fam and s not in picks:
                picks.append(s)
                break
    picks = picks[:3] or scenes[:3]
    fig, axes = plt.subplots(len(picks), 2, figsize=(9, 4.4 * len(picks)))
    fig.suptitle("Fine-array zoom — GT centre crop (2×) vs LR centre crop (1×), nearest",
                 fontsize=13, y=0.997)
    axes = np.atleast_2d(axes)
    for r, s in enumerate(picks):
        h, w = s["gt"].shape
        cy, cx, rh, rw = h // 2, w // 2, h // 8, w // 8
        gt_c = s["gt"][cy - rh:cy + rh, cx - rw:cx + rw]
        lh, lw = s["lr_mid"].shape
        lr_c = s["lr_mid"][lh // 2 - rh // 2:lh // 2 + rh // 2, lw // 2 - rw // 2:lw // 2 + rw // 2]
        lo, hi = _pct(gt_c)
        axes[r, 0].imshow(gt_c, cmap=CMAP, vmin=lo, vmax=hi, interpolation="nearest")
        axes[r, 0].set_title(f"GT crop — {s['family']} [{_dlabel(s)}]", fontsize=9)
        axes[r, 1].imshow(lr_c, cmap=CMAP, vmin=lo, vmax=hi, interpolation="nearest")
        axes[r, 1].set_title("LR crop (aliased)", fontsize=9)
        for c in (0, 1):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, type=Path)
    ap.add_argument("--figs", required=True, type=Path)
    ap.add_argument("--anchor", required=True, type=Path)
    args = ap.parse_args()

    args.figs.mkdir(parents=True, exist_ok=True)
    scene_dirs = sorted(args.pool.glob("scene_*"))
    if not scene_dirs:
        raise SystemExit(f"no scene_* under {args.pool}")
    scenes = [_load_scene(d) for d in scene_dirs]
    anchor = _load_anchor(args.anchor)

    fig_motif_grid(scenes, args.figs / "fig1_motif_grid.png")
    fig_recoverability(scenes, args.figs / "fig2_recoverability.png")
    fig_vs_real(scenes, anchor, args.figs / "fig3_vs_real_anchor.png")
    fig_zoom(scenes, args.figs / "fig4_zoom_grid.png")
    print("wrote:")
    for p in sorted(args.figs.glob("*.png")):
        print("  ", p.resolve())


if __name__ == "__main__":
    main()
