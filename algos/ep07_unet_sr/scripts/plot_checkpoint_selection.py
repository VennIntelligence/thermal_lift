#!/usr/bin/env python3
"""Plot EP07 checkpoint-selection proxy trajectories and visual panels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from thermal_core.plotting import get_method_style, savefig_academic, setup_academic_style


PROJECT_ROOT = Path(__file__).resolve().parents[3]
EP07_OUTPUTS = PROJECT_ROOT / "algos" / "ep07_unet_sr" / "outputs"
DEFAULT_INPUT_CSV = (
    PROJECT_ROOT
    / "output"
    / "ep11_dl_benchmark"
    / "checkpoint_selection"
    / "checkpoint_metrics.csv"
)
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "ep11_dl_benchmark" / "checkpoint_selection"
TGV_REFERENCE = (0.695, 0.916)
CANONICAL_LABEL_OFFSETS = {
    "v6": (10, 8),
    "v8.1a": (10, -12),
    "v8.1b": (-34, -18),
    "v9b": (8, 16),
}

ARMS: dict[str, str] = {
    "v6": "ep07_v6_physics",
    "v8.1a": "ep07_v8_1a_loss_cooldown",
    "v8.1b": "ep07_v8_1b_pixelshuffle",
    "v9b": "ep07_v9b_fwd_consistency",
}


def _abs(path: Path) -> str:
    return str(path.resolve())


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path.resolve())


def _checkpoint_path(arm: str, step: int) -> Path:
    return EP07_OUTPUTS / ARMS[arm] / f"checkpoint_step_{int(step):06d}.pt"


def _png_path(arm: str, step: int) -> Path:
    return (
        EP07_OUTPUTS
        / ARMS[arm]
        / "eval_real"
        / f"unet_step{int(step)}_center_zoom3x_temperature.png"
    )


def _normalise_distance(group: pd.DataFrame) -> pd.DataFrame:
    out = group.copy()
    art = out["artifact_score"].astype(float)
    corr = out["raw_control_corr"].astype(float)
    art_span = float(art.max() - art.min())
    corr_span = float(corr.max() - corr.min())
    out["artifact_norm"] = 0.0 if art_span == 0.0 else (art - art.min()) / art_span
    out["corr_norm"] = 0.0 if corr_span == 0.0 else (corr.max() - corr) / corr_span
    out["ideal_distance"] = np.sqrt(out["artifact_norm"] ** 2 + out["corr_norm"] ** 2)
    return out


def select_candidates(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        group = metrics[metrics["arm"] == arm].sort_values("step").reset_index(drop=True)
        if group.empty:
            continue
        scored = _normalise_distance(group)
        selected_steps: list[int] = []
        selected_windows: set[int] = set()
        for record in scored.sort_values(["ideal_distance", "step"]).to_dict("records"):
            step = int(record["step"])
            window = step // 5000
            if window in selected_windows:
                continue
            selected_steps.append(step)
            selected_windows.add(window)
            if len(selected_steps) == 3:
                break

        if int(scored["step"].max()) == 60000 and 60000 not in selected_steps:
            selected_steps.append(60000)

        selected = scored[scored["step"].isin(selected_steps)].copy()
        selected = selected.sort_values(["ideal_distance", "step"]).reset_index(drop=True)
        canonical_step = int(selected.iloc[0]["step"])
        for rank, record in enumerate(selected.to_dict("records"), start=1):
            step = int(record["step"])
            checkpoint = _checkpoint_path(arm, step)
            png = _png_path(arm, step)
            record.update(
                {
                    "candidate_rank": rank,
                    "is_terminal_60k": step == 60000,
                    "is_canonical": step == canonical_step,
                    "checkpoint_path": _abs(checkpoint),
                    "png_path": _abs(png),
                    "checkpoint_exists": checkpoint.exists(),
                    "png_exists": png.exists(),
                }
            )
            rows.append(record)

    columns = [
        "arm",
        "step",
        "artifact_score",
        "raw_control_corr",
        "artifact_norm",
        "corr_norm",
        "ideal_distance",
        "candidate_rank",
        "is_terminal_60k",
        "is_canonical",
        "checkpoint_path",
        "png_path",
        "checkpoint_exists",
        "png_exists",
    ]
    return pd.DataFrame(rows, columns=columns)


def plot_trajectories(metrics: pd.DataFrame, output_dir: Path) -> Path:
    setup_academic_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharex=True)
    for idx, arm in enumerate(ARMS):
        group = metrics[metrics["arm"] == arm].sort_values("step")
        style = get_method_style(idx)
        label = arm
        axes[0].plot(
            group["step"],
            group["artifact_score"],
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markevery=max(1, len(group) // 8),
        )
        axes[1].plot(
            group["step"],
            group["raw_control_corr"],
            label=label,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            markevery=max(1, len(group) // 8),
        )

    axes[0].set_title("Artifact proxy")
    axes[0].set_ylabel("artifact_score (lower is better)")
    axes[1].set_title("Raw-control agreement")
    axes[1].set_ylabel("raw_control_corr (higher is better)")
    for ax in axes:
        ax.set_xlabel("checkpoint step")
        ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    axes[1].legend(loc="best")

    path = output_dir / "fig_trajectories.png"
    savefig_academic(fig, path)
    return path


def plot_pareto(metrics: pd.DataFrame, candidates: pd.DataFrame, output_dir: Path) -> Path:
    setup_academic_style()
    fig, ax = plt.subplots(1, 1, figsize=(5.5, 4.0))
    for idx, arm in enumerate(ARMS):
        group = metrics[metrics["arm"] == arm].sort_values("step")
        style = get_method_style(idx)
        ax.plot(
            group["artifact_score"],
            group["raw_control_corr"],
            label=arm,
            color=style["color"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            alpha=0.8,
        )
        cand = candidates[candidates["arm"] == arm]
        ax.scatter(
            cand["artifact_score"],
            cand["raw_control_corr"],
            s=72,
            facecolors="none",
            edgecolors="black",
            linewidths=1.0,
            zorder=5,
        )
        canonical = cand[cand["is_canonical"]]
        if not canonical.empty:
            row = canonical.iloc[0]
            offset = CANONICAL_LABEL_OFFSETS.get(arm, (4, 4))
            ax.annotate(
                f"{arm} {int(row['step']) // 1000}K",
                (float(row["artifact_score"]), float(row["raw_control_corr"])),
                xytext=offset,
                textcoords="offset points",
                fontsize=7,
            )

    ax.scatter(
        [TGV_REFERENCE[0]],
        [TGV_REFERENCE[1]],
        marker="*",
        s=110,
        color="black",
        label="EP10 TGV reference",
        zorder=6,
    )
    ax.annotate(
        "EP10 TGV (0.695, 0.916)",
        TGV_REFERENCE,
        xytext=(-88, -16),
        textcoords="offset points",
        fontsize=8,
        arrowprops={"arrowstyle": "->", "linewidth": 0.6},
    )
    ax.set_xlabel("artifact_score (lower is better)")
    ax.set_ylabel("raw_control_corr (higher is better)")
    ax.set_title("Real-eval proxy Pareto view")
    ax.grid(axis="both", alpha=0.25, linewidth=0.5)
    ax.legend(loc="lower left")

    finite_x = metrics["artifact_score"].astype(float)
    finite_y = metrics["raw_control_corr"].astype(float)
    x_pad = max(0.02, 0.04 * float(finite_x.max() - finite_x.min()))
    y_pad = max(0.01, 0.04 * float(finite_y.max() - finite_y.min()))
    ax.set_xlim(min(float(finite_x.min()), TGV_REFERENCE[0]) - x_pad, max(float(finite_x.max()), TGV_REFERENCE[0]) + x_pad)
    ax.set_ylim(min(float(finite_y.min()), TGV_REFERENCE[1]) - y_pad, max(float(finite_y.max()), TGV_REFERENCE[1]) + y_pad)

    path = output_dir / "fig_pareto.png"
    savefig_academic(fig, path)
    return path


def _placeholder_image(label: str, size: tuple[int, int] = (900, 650)) -> Image.Image:
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, size[0] - 1, size[1] - 1), outline=(180, 180, 180), width=3)
    draw.text((24, 24), label, fill=(20, 20, 20), font=_panel_font(24))
    return image


def _panel_font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSerif.ttf", "Times New Roman.ttf", "Times.ttf"):
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _footer(width: int, text: str, height: int = 110) -> Image.Image:
    footer = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(footer)
    draw.line((0, 0, width, 0), fill=(210, 210, 210), width=1)
    draw.text((18, 36), text, fill=(20, 20, 20), font=_panel_font(32))
    return footer


def build_panel(arm: str, candidates: pd.DataFrame, output_dir: Path) -> Path:
    tiles: list[Image.Image] = []
    for row in candidates[candidates["arm"] == arm].sort_values(["candidate_rank", "step"]).to_dict("records"):
        step = int(row["step"])
        png = Path(str(row["png_path"]))
        if png.exists():
            image = Image.open(png).convert("RGB")
        else:
            image = _placeholder_image(f"Missing PNG: {png.name}")
        label = (
            f"step {step} | artifact={float(row['artifact_score']):.3f} | "
            f"corr={float(row['raw_control_corr']):.3f}"
        )
        footer = _footer(image.width, label)
        tile = Image.new("RGB", (image.width, image.height + footer.height), "white")
        tile.paste(image, (0, 0))
        tile.paste(footer, (0, image.height))
        tiles.append(tile)

    if not tiles:
        panel = _placeholder_image(f"No candidates for {arm}")
    else:
        max_height = max(tile.height for tile in tiles)
        total_width = sum(tile.width for tile in tiles)
        panel = Image.new("RGB", (total_width, max_height), "white")
        x0 = 0
        for tile in tiles:
            panel.paste(tile, (x0, 0))
            x0 += tile.width

    path = output_dir / f"panel_{arm}.png"
    panel.save(path, dpi=(300, 300))
    return path


def run(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(args.input_csv)
    required = {"arm", "step", "artifact_score", "raw_control_corr"}
    missing = required - set(metrics.columns)
    if missing:
        raise KeyError(f"Missing input column(s): {sorted(missing)}")
    metrics = metrics[metrics["arm"].isin(ARMS)].copy()
    metrics["step"] = metrics["step"].astype(int)

    candidates = select_candidates(metrics)
    candidates_csv = output_dir / "checkpoint_candidates.csv"
    candidates.to_csv(candidates_csv, index=False)

    trajectories = plot_trajectories(metrics, output_dir)
    pareto = plot_pareto(metrics, candidates, output_dir)
    panels = {arm: build_panel(arm, candidates, output_dir) for arm in ARMS}

    manifest = {
        "input_csv": _rel(args.input_csv),
        "output_dir": _rel(output_dir),
        "candidate_rule": (
            "Per arm, min-max normalise artifact_score and inverted "
            "raw_control_corr, choose three nearest distinct 5K windows, "
            "then append step 60000 as terminal control when present."
        ),
        "tgv_reference": {
            "artifact_score": TGV_REFERENCE[0],
            "raw_control_corr": TGV_REFERENCE[1],
        },
        "artifacts": {
            "candidates_csv": _rel(candidates_csv),
            "trajectories": _rel(trajectories),
            "pareto": _rel(pareto),
            "panels": {arm: _rel(path) for arm, path in panels.items()},
        },
        "missing_png": candidates.loc[~candidates["png_exists"], ["arm", "step", "png_path"]].to_dict("records"),
        "missing_checkpoint": candidates.loc[
            ~candidates["checkpoint_exists"], ["arm", "step", "checkpoint_path"]
        ].to_dict("records"),
    }
    manifest_path = output_dir / "checkpoint_selection_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(f"Wrote {_rel(candidates_csv)}")
    print(f"Wrote {_rel(trajectories)}")
    print(f"Wrote {_rel(pareto)}")
    for arm, path in panels.items():
        print(f"Wrote {arm}: {_rel(path)}")
    print(f"Wrote {_rel(manifest_path)}")
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input_csv}")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
