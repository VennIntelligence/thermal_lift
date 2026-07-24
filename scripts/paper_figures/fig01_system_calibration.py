#!/usr/bin/env python3
"""Generate Figure F1: system geometry, calibration chain, and grid scales.

用法（项目根目录，无 CLI 参数）::

    uv run python scripts/paper_figures/fig01_system_calibration.py

输入依赖: configs/stage_calibration.json、configs/noise_floor.json，
    可选 output/ep15_info_limit/m3_sigma/sigma_summary.json（缺失时用默认 σ 区间）
输出: output/paper_figures/fig01_system_calibration.{png,pdf,json}

历史定位: scripts/paper_figures/ 是 2026-06 时代的旧论文图脚本；现行权威图集见
docs/publication_figures/（每图一个脚本、自带规范）。
"""

from __future__ import annotations

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("MPLBACKEND", "Agg")

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, FancyArrowPatch, Rectangle

from thermal_core.displacement import coordinate_to_shift
from thermal_core.ep03 import build_sampling_resolution_table
from thermal_core.plotting import METHOD_COLOR_LIST, METHOD_COLORS, savefig_academic, setup_academic_style


ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = ROOT / "output" / "paper_figures"
PNG_PATH = OUTPUT_DIR / "fig01_system_calibration.png"
PDF_PATH = OUTPUT_DIR / "fig01_system_calibration.pdf"
JSON_PATH = OUTPUT_DIR / "fig01_system_calibration.json"

STAGE_CONFIG_PATH = ROOT / "configs" / "stage_calibration.json"
NOISE_CONFIG_PATH = ROOT / "configs" / "noise_floor.json"
SIGMA_SUMMARY_PATH = ROOT / "output" / "ep15_info_limit" / "m3_sigma" / "sigma_summary.json"

COORDINATES_UM = np.array(
    [0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32, 36, 40],
    dtype=float,
)
COMMAND_UM = 40.0


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _load_constants() -> dict[str, Any]:
    stage = _read_json(STAGE_CONFIG_PATH)
    noise = _read_json(NOISE_CONFIG_PATH)
    sigma_summary: dict[str, Any] = {}
    if SIGMA_SUMMARY_PATH.exists():
        sigma_summary = _read_json(SIGMA_SUMMARY_PATH)

    detector_pitch_um = float(stage["pixel_size_um"])
    spatial_resolution_um = float(stage["current_spatial_resolution_um"])
    theta_deg = float(stage["theta_deg"])
    noise_floor_c = float(noise["noise_floor_celsius"])
    sigma_range = sigma_summary.get("sigma_credible_range_lr_px", [0.2, 0.5])
    sigma_min, sigma_max = [float(v) for v in sigma_range]
    target_grid_um = detector_pitch_um / 2.0

    dx_40, dy_40 = coordinate_to_shift(
        COMMAND_UM,
        0.0,
        theta_deg=theta_deg,
        pixel_size_um=detector_pitch_um,
    )
    command_shift_px = float(np.hypot(dx_40, dy_40))

    build_sampling_resolution_table(
        detector_pitch_um=detector_pitch_um,
        spatial_resolution_um=spatial_resolution_um,
        target_grid_um=target_grid_um,
    )

    return {
        "theta_deg": theta_deg,
        "detector_pitch_um": detector_pitch_um,
        "spatial_resolution_um": spatial_resolution_um,
        "target_grid_um": target_grid_um,
        "detector_rows": int(stage["detector_rows"]),
        "detector_cols": int(stage["detector_cols"]),
        "wavelength_band": str(stage["wavelength_band"]),
        "noise_floor_c": noise_floor_c,
        "sigma_min_px": sigma_min,
        "sigma_max_px": sigma_max,
        "command_um": COMMAND_UM,
        "command_dx_px": float(dx_40),
        "command_dy_px": float(dy_40),
        "command_shift_px": command_shift_px,
    }


def _panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.015,
        0.985,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9.2,
        fontweight="bold",
        bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.4, "alpha": 0.85},
    )


def _draw_panel_a(ax: plt.Axes, constants: dict[str, Any]) -> None:
    theta_deg = constants["theta_deg"]
    detector_pitch_um = constants["detector_pitch_um"]
    x_grid, y_grid = np.meshgrid(COORDINATES_UM, COORDINATES_UM)

    ax.scatter(
        x_grid.ravel(),
        y_grid.ravel(),
        s=7,
        color=METHOD_COLORS["primary"],
        alpha=0.85,
        linewidths=0.0,
        zorder=3,
    )

    for y in COORDINATES_UM:
        ax.plot(
            COORDINATES_UM,
            np.full_like(COORDINATES_UM, y),
            color=METHOD_COLORS["primary"],
            alpha=0.16,
            linewidth=0.75,
            zorder=1,
        )
        ax.add_patch(
            FancyArrowPatch(
                (6.0, y),
                (38.5, y),
                arrowstyle="-|>",
                mutation_scale=5,
                linewidth=0.55,
                color=METHOD_COLORS["primary"],
                alpha=0.35,
                zorder=2,
            )
        )

    for y0, y1 in zip(COORDINATES_UM[:-1], COORDINATES_UM[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (42.5, y0),
                (42.5, y1),
                arrowstyle="-|>",
                mutation_scale=4.5,
                linewidth=0.55,
                color="#666666",
                alpha=0.35,
                zorder=2,
            )
        )

    # Pixel basis shown in the stage-coordinate plane: inverse of the
    # coordinate_to_shift rotation, with one drawn detector pixel enlarged.
    origin = np.array([9.0, 31.0])
    basis_len_um = detector_pitch_um * 0.9
    theta = np.deg2rad(theta_deg)
    e_pix_x = basis_len_um * np.array([np.cos(theta), np.sin(theta)])
    e_pix_y = basis_len_um * np.array([-np.sin(theta), np.cos(theta)])
    ax.add_patch(
        FancyArrowPatch(
            origin,
            origin + e_pix_x,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.0,
            color=METHOD_COLORS["accent_1"],
            zorder=4,
        )
    )
    ax.add_patch(
        FancyArrowPatch(
            origin,
            origin + e_pix_y,
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.0,
            color=METHOD_COLORS["accent_2"],
            zorder=4,
        )
    )
    ax.text(*(origin + e_pix_x + np.array([0.6, 0.3])), r"$e_x^{pix}$", fontsize=7, color=METHOD_COLORS["accent_1"])
    ax.text(*(origin + e_pix_y + np.array([0.5, -0.9])), r"$e_y^{pix}$", fontsize=7, color=METHOD_COLORS["accent_2"])
    ax.add_patch(
        Arc(
            origin,
            9.0,
            9.0,
            angle=0.0,
            theta1=0.0,
            theta2=theta_deg,
            color="#444444",
            linewidth=0.7,
        )
    )
    ax.text(origin[0] + 5.0, origin[1] + 1.8, rf"$\theta={theta_deg:.1f}^\circ$", fontsize=7, color="#222222")

    ax.add_patch(
        FancyArrowPatch(
            (0.0, -2.0),
            (COMMAND_UM, -2.0),
            arrowstyle="-|>",
            mutation_scale=8,
            linewidth=1.2,
            color=METHOD_COLORS["accent_3"],
            zorder=4,
        )
    )
    ax.text(
        20.0,
        -3.6,
        rf"40 um command $\rightarrow$ {constants['command_shift_px']:.1f} px",
        ha="center",
        va="top",
        fontsize=7,
        color=METHOD_COLORS["accent_3"],
    )
    ax.text(
        26.5,
        31.5,
        "Stage commands\nare priors,\nnot truth",
        ha="left",
        va="top",
        fontsize=7,
        color="#222222",
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "pad": 1.8, "alpha": 0.92},
    )
    ax.text(
        43.8,
        20.0,
        "row jump",
        ha="left",
        va="center",
        fontsize=6.6,
        color="#555555",
        rotation=90,
    )

    ax.set_title("Micro-scan geometry", fontsize=8.6, pad=3)
    ax.set_xlabel("Stage X [um]")
    ax.set_ylabel("Stage Y [um]")
    ax.set_xlim(-3.0, 47.5)
    ax.set_ylim(-5.0, 43.0)
    ax.set_xticks([0, 10, 20, 40])
    ax.set_yticks([0, 10, 20, 40])
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, color="#dddddd", linewidth=0.45, alpha=0.5)
    _panel_label(ax, "(a)")


def _draw_box(
    ax: plt.Axes,
    center_x: float,
    label: str,
    source: str,
    *,
    width: float,
    facecolor: str,
) -> tuple[float, float]:
    y0 = 0.50
    height = 0.24
    x0 = center_x - width / 2.0
    rect = Rectangle(
        (x0, y0),
        width,
        height,
        transform=ax.transAxes,
        facecolor=facecolor,
        alpha=0.16,
        edgecolor=facecolor,
        linewidth=1.0,
    )
    ax.add_patch(rect)
    ax.text(
        center_x,
        y0 + height / 2.0,
        label,
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.3,
        color="#111111",
        linespacing=0.96,
    )
    ax.text(
        center_x,
        y0 - 0.10,
        source,
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=5.7,
        color="#555555",
        linespacing=0.96,
    )
    return x0, x0 + width


def _draw_panel_b(ax: plt.Axes, constants: dict[str, Any]) -> None:
    ax.set_axis_off()
    ax.set_title("Calibration chain and observation model", fontsize=8.6, pad=3)

    centers = [0.055, 0.220, 0.410, 0.585, 0.735, 0.905]
    widths = [0.095, 0.120, 0.165, 0.115, 0.105, 0.140]
    labels = [
        "$x$\nlatent field",
        "$S_k$\nshift",
        "$H$\nPSF\n" + rf"$\sigma={constants['sigma_min_px']:.1f}$--{constants['sigma_max_px']:.1f} px",
        "$B$\n10 um\nbox",
        "$D$\n$\\downarrow 2$",
        "$y_k+n$\n" + rf"{constants['noise_floor_c']:.4f}$^\circ$C",
    ]
    sources = [
        "target",
        "EP02\ntheta",
        "EP09+M3\nsigma",
        "EP03\npitch",
        "EP03\ngrid",
        "EP01\nnoise",
    ]
    colors = [
        "#888888",
        METHOD_COLORS["primary"],
        METHOD_COLORS["accent_1"],
        METHOD_COLORS["secondary"],
        METHOD_COLORS["accent_2"],
        METHOD_COLORS["neutral"],
    ]

    edges = [
        _draw_box(ax, c, label, source, width=w, facecolor=color)
        for c, w, label, source, color in zip(centers, widths, labels, sources, colors)
    ]
    for (_, right), (left, _) in zip(edges[:-1], edges[1:]):
        ax.add_patch(
            FancyArrowPatch(
                (right + 0.010, 0.625),
                (left - 0.010, 0.625),
                transform=ax.transAxes,
                arrowstyle="-|>",
                mutation_scale=8,
                linewidth=0.8,
                color="#333333",
            )
        )

    ax.text(
        0.5,
        0.91,
        "$y_k = D B H S_k x + n_k$",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=8.3,
        color="#111111",
    )
    ax.text(
        0.5,
        0.18,
        "Measured operators constrain reconstruction; command shifts initialize alignment.",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=6.2,
        color="#333333",
    )
    _panel_label(ax, "(b)")


def _draw_panel_c(ax: plt.Axes, constants: dict[str, Any]) -> None:
    detector_pitch_um = constants["detector_pitch_um"]
    spatial_resolution_um = constants["spatial_resolution_um"]
    target_grid_um = constants["target_grid_um"]
    four_x_um = detector_pitch_um / 4.0

    ax.set_xlim(-1.0, 41.0)
    ax.set_ylim(-0.22, 4.0)
    ax.set_xlabel("Physical distance [um]")
    ax.set_yticks([3.2, 2.2, 1.2, 0.2])
    ax.set_yticklabels(["Detector\nsamples", "Spatial\nresolution", "2x SR\ngrid", "4x grid"])
    ax.tick_params(axis="y", length=0, pad=2)

    for x in np.arange(0.0, 40.1, detector_pitch_um):
        ax.plot([x, x], [2.92, 3.48], color=METHOD_COLOR_LIST[0], linewidth=1.1)
    ax.plot([0.0, 40.0], [3.2, 3.2], color=METHOD_COLOR_LIST[0], linewidth=0.75)
    ax.text(
        19.8,
        3.56,
        "10 um / pixel",
        ha="center",
        va="bottom",
        fontsize=9.0,
        color=METHOD_COLOR_LIST[0],
    )

    rect = Rectangle(
        (0.0, 1.95),
        spatial_resolution_um,
        0.50,
        facecolor=METHOD_COLOR_LIST[2],
        alpha=0.25,
        edgecolor=METHOD_COLOR_LIST[2],
        linewidth=0.9,
    )
    ax.add_patch(rect)
    ax.text(
        spatial_resolution_um / 2.0,
        2.52,
        "20 um resolution",
        ha="center",
        va="bottom",
        fontsize=9.0,
        color=METHOD_COLOR_LIST[2],
    )

    for x in np.arange(0.0, 40.1, target_grid_um):
        ax.plot([x, x], [0.95, 1.45], color=METHOD_COLOR_LIST[1], linewidth=0.85)
    ax.text(
        24.0,
        1.55,
        "5 um 2x grid",
        ha="center",
        va="bottom",
        fontsize=9.0,
        color=METHOD_COLOR_LIST[1],
    )

    for x in np.arange(0.0, 40.1, four_x_um):
        ax.plot([x, x], [-0.05, 0.45], color=METHOD_COLOR_LIST[3], linewidth=0.45, alpha=0.78)
    ax.text(
        24.0,
        0.55,
        "2.5 um 4x grid",
        ha="center",
        va="bottom",
        fontsize=9.0,
        color=METHOD_COLOR_LIST[3],
    )

    ax.set_title("Pitch is not resolution", fontsize=8.6, pad=3)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.tick_params(axis="y", labelsize=7.0)
    ax.grid(axis="x", alpha=0.15, linewidth=0.5)
    ax.spines["left"].set_visible(False)
    _panel_label(ax, "(c)")


def _build_source_mapping(constants: dict[str, Any]) -> dict[str, Any]:
    def rel(path: Path | str) -> str:
        path_obj = Path(path)
        try:
            return str(path_obj.relative_to(ROOT))
        except ValueError:
            return str(path_obj)

    return {
        "figure": "F1 system + calibration chain + sampling/resolution distinction",
        "script": rel(Path(__file__).resolve()),
        "outputs": {
            "png": rel(PNG_PATH),
            "pdf": rel(PDF_PATH),
            "json": rel(JSON_PATH),
        },
        "source_mapping": {
            "stage_to_pixel_rotation_theta_deg": {
                "value": constants["theta_deg"],
                "unit": "degree",
                "source_files": [rel(STAGE_CONFIG_PATH), "paper/reports/ep02_displacement_calibration/calibration_report.md"],
            },
            "detector_pitch_um": {
                "value": constants["detector_pitch_um"],
                "unit": "um/pixel",
                "source_files": [rel(STAGE_CONFIG_PATH), "core/src/thermal_core/ep03.py"],
            },
            "spatial_resolution_um": {
                "value": constants["spatial_resolution_um"],
                "unit": "um",
                "source_files": [rel(STAGE_CONFIG_PATH), "core/src/thermal_core/ep03.py"],
            },
            "two_x_sr_grid_um": {
                "value": constants["target_grid_um"],
                "unit": "um/sample",
                "source_files": ["core/src/thermal_core/ep03.py"],
            },
            "four_x_reference_grid_um": {
                "value": constants["detector_pitch_um"] / 4.0,
                "unit": "um/sample",
                "source_files": ["core/src/thermal_core/ep03.py"],
            },
            "coordinate_set_um": {
                "value": COORDINATES_UM.astype(int).tolist(),
                "unit": "um",
                "source_files": ["AGENTS.md", "todos/paper_prompts.md"],
            },
            "command_40um_shift_prior": {
                "stage_command_um": constants["command_um"],
                "dx_px": constants["command_dx_px"],
                "dy_px": constants["command_dy_px"],
                "magnitude_px": constants["command_shift_px"],
                "source_files": [rel(STAGE_CONFIG_PATH), "core/src/thermal_core/displacement.py"],
            },
            "psf_sigma_range_lr_px": {
                "value": [constants["sigma_min_px"], constants["sigma_max_px"]],
                "unit": "LR pixel sigma",
                "source_files": [rel(SIGMA_SUMMARY_PATH), "docs/paper/04_problem_forward_model.md"],
            },
            "noise_floor_c": {
                "value": constants["noise_floor_c"],
                "unit": "degree C",
                "source_files": [rel(NOISE_CONFIG_PATH), "core/src/thermal_core/ep01.py", "core/src/thermal_core/ep03.py"],
            },
            "detector_shape": {
                "rows": constants["detector_rows"],
                "cols": constants["detector_cols"],
                "source_files": [rel(STAGE_CONFIG_PATH)],
            },
            "wavelength_band": {
                "value": constants["wavelength_band"],
                "source_files": [rel(STAGE_CONFIG_PATH)],
            },
        },
    }


def build_figure() -> plt.Figure:
    setup_academic_style()
    constants = _load_constants()

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(7.2, 2.60),
        gridspec_kw={"width_ratios": [1.02, 1.48, 1.02]},
        constrained_layout=True,
    )
    fig.patch.set_facecolor("white")

    _draw_panel_a(axes[0], constants)
    _draw_panel_b(axes[1], constants)
    _draw_panel_c(axes[2], constants)
    return fig


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    constants = _load_constants()
    fig = build_figure()
    savefig_academic(fig, PNG_PATH, dpi=300, close=False)
    savefig_academic(fig, PDF_PATH, dpi=300, close=True)

    mapping = _build_source_mapping(constants)
    with JSON_PATH.open("w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)
        f.write("\n")

    print(f"Wrote {PNG_PATH.relative_to(ROOT)}")
    print(f"Wrote {PDF_PATH.relative_to(ROOT)}")
    print(f"Wrote {JSON_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
