"""EP07 demo dataset cache builder and loader for notebook fragments."""

from __future__ import annotations

import json
import sys
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.ndimage import gaussian_filter, shift as ndi_shift

from thermal_core.notebook_cache import (
    cache_is_complete,
    load_manifest,
    missing_artifacts,
    project_root,
    require_artifacts,
    write_manifest,
)
from thermal_core.ep03 import gaussian_mtf
from thermal_core.plotting import (
    COLORMAPS,
    METHOD_COLORS,
    format_colorbar,
    savefig_academic,
    setup_academic_style,
)

EP07_CACHE_VERSION = 3

EP07_FIGURE_ARTIFACTS = (
    "demo_hr_scene.png",
    "demo_thermal_field_decomposition.png",
    "demo_psf_blur_check.png",
    "demo_noise_check.png",
    "demo_noise_real_vs_synthetic.png",
    "demo_snr_budget.png",
    "demo_forward_highpass.png",
    "demo_dataset_overview.png",
    "demo_profiles_generation_vs_observation.png",
)

EP07_TABLE_ARTIFACTS = (
    "scene_stats.csv",
    "forward_stats.csv",
    "physics_checks.csv",
    "snr_budget.csv",
    "noise_model_checks.csv",
    "demo_metrics.csv",
    "demo_metadata.json",
)

EP07_CACHE_ARTIFACTS = (*EP07_FIGURE_ARTIFACTS, *EP07_TABLE_ARTIFACTS, "cache_manifest.json")
REBUILD_COMMAND = "uv run python scripts/build_ep07_cache.py"


@dataclass(frozen=True)
class DemoConfig:
    scene_id: str = "ep07_demo_easy_000"
    seed: int = 7
    lr_shape: tuple[int, int] = (64, 96)
    scale: int = 2
    n_frames: int = 16
    pixel_size_um: float = 20.0
    base_temp_c: float = 21.0
    delta_temp_c: float = 2.5
    low_freq_amplitude_c: float = 0.2
    noise_sigma_c: float = 0.0724
    noise_model: str = "mixed"
    fpn_sigma_px: float = 5.0
    stripe_sigma_c: float = 0.015
    psf_profile: str = "ep09_provisional"
    psf_profile_source: str = "configs/psf_calibration.json"
    psf_calibration_status: str = "provisional_needs_review"
    psf_four_x_verdict: str = "not_cleared_inconsistent_routes"
    psf_sigma_hr_px: float = 0.45144300017693384
    psf_sigma_lr_px: float = 0.22572150008846692
    highpass_sigma_lr_px: float = 5.0


@dataclass(frozen=True)
class Ep07Cache:
    output_dir: Path
    demo_dir: Path
    manifest: dict
    demo_config: DemoConfig
    scene_generation_mode: str
    forward_mode: str
    shift_source: str
    highpass_source: str
    block_forward_mode: str
    edge_vis_margin_lr_px: int
    eval_source: str
    scene_stats: pd.DataFrame
    forward_stats: pd.DataFrame
    physics_checks: pd.DataFrame
    snr_budget: pd.DataFrame
    noise_model_checks: pd.DataFrame
    demo_metrics: pd.DataFrame
    metadata: dict
    demo_skipped: bool
    tcforge_available: bool

    def figure_path(self, name: str) -> Path:
        return self.output_dir / name


def _tcforge_src(root: Path) -> Path:
    return root / "tcforge" / "src"


def _ensure_tcforge_path(root: Path) -> bool:
    src = _tcforge_src(root)
    if src.exists() and str(src) not in sys.path:
        sys.path.insert(0, str(src))
    try:
        import tcforge  # noqa: F401

        return True
    except Exception:
        return False


def _load_psf_calibration(root: Path) -> dict[str, object]:
    path = root / "configs" / "psf_calibration.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _resolve_demo_config(config: DemoConfig, root: Path) -> DemoConfig:
    if config.psf_profile != "ep09_provisional":
        return config
    calibration = _load_psf_calibration(root)
    if not calibration:
        return config
    sigma_lr = float(calibration["psf_sigma_lr_px"])
    return replace(
        config,
        psf_sigma_lr_px=sigma_lr,
        psf_sigma_hr_px=float(calibration.get("psf_sigma_hr_px_at_2x", sigma_lr * config.scale)),
        psf_profile_source="configs/psf_calibration.json",
        psf_calibration_status=str(calibration.get("status", "")),
        psf_four_x_verdict=str(calibration.get("four_x_verdict", "")),
    )


def _boolish(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"1", "true", "yes", "y", "t"})


def ep06_like_highpass(frames: np.ndarray, sigma_bg: float = 5.0, mode: str = "nearest") -> np.ndarray:
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 2:
        return arr - gaussian_filter(arr, sigma=sigma_bg, mode=mode)
    if arr.ndim != 3:
        raise ValueError("frames must be 2D or 3D")
    bg = gaussian_filter(arr, sigma=(0.0, sigma_bg, sigma_bg), mode=mode)
    return (arr - bg).astype(np.float32, copy=False)


def array_contract_row(name: str, arr: np.ndarray, *, role: str) -> dict[str, object]:
    values = np.asarray(arr)
    finite = np.isfinite(values).all() if np.issubdtype(values.dtype, np.number) else True
    return {
        "array": name,
        "role": role,
        "shape": "x".join(map(str, values.shape)),
        "dtype": str(values.dtype),
        "min": float(np.min(values)) if values.size and np.issubdtype(values.dtype, np.number) else np.nan,
        "max": float(np.max(values)) if values.size and np.issubdtype(values.dtype, np.number) else np.nan,
        "finite": bool(finite),
    }


def _image_limits(image: np.ndarray, *, symmetric: bool = False, q: float = 99.0) -> tuple[float, float]:
    values = np.asarray(image, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0.0, 1.0)
    if symmetric:
        vmax = float(np.percentile(np.abs(values), q))
        return (-max(vmax, 1e-8), max(vmax, 1e-8))
    return (float(np.percentile(values, 100 - q)), float(np.percentile(values, q)))


def _add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        fontweight="bold",
        color="black",
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.78, "pad": 1.5},
    )


def _compact_cbar_title(label: str | None) -> str:
    if not label:
        return ""
    if "$^\\circ$C" in label or "°C" in label or "[C]" in label:
        return "C"
    if "Weight" in label:
        return "w"
    if "Mask" in label:
        return "mask"
    if "Edge" in label:
        return "edge"
    return ""


def _show_image(
    ax: plt.Axes,
    image: np.ndarray,
    *,
    title: str,
    cmap: str,
    colorbar_label: str | None = None,
    vmin: float | None = None,
    vmax: float | None = None,
    robust: bool = False,
    symmetric: bool = False,
    show_colorbar: bool = True,
):
    if robust:
        vmin, vmax = _image_limits(image, symmetric=symmetric)
    arr = np.asarray(image)
    if arr.ndim >= 2 and arr.shape[1] > 0:
        ax.set_box_aspect(float(arr.shape[0]) / float(arr.shape[1]))
    im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if not show_colorbar:
        return im
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.6%", pad=0.03)
    cbar = ax.figure.colorbar(im, cax=cax)
    cbar.ax.tick_params(length=2.0, width=0.5, pad=1.3, labelsize=7)
    cbar_title = _compact_cbar_title(colorbar_label)
    if cbar_title:
        cbar.ax.set_title(cbar_title, fontsize=6.5, pad=2.0)
    return im


def _fallback_make_scene(config: DemoConfig) -> tuple[dict[str, np.ndarray], str]:
    rng = np.random.default_rng(config.seed)
    hr_shape = (config.lr_shape[0] * config.scale, config.lr_shape[1] * config.scale)
    yy, xx = np.mgrid[: hr_shape[0], : hr_shape[1]]

    mask = np.zeros(hr_shape, dtype=np.uint8)
    mask[24:104, 32:160] = 1
    mask[48:80, 58:138] = 0
    for col in range(36, 156, 16):
        mask[18:28, col : col + 6] = 1
        mask[100:112, col : col + 6] = 1
    for row in range(34, 98, 14):
        mask[row : row + 5, 24:36] = 1
        mask[row : row + 5, 156:168] = 1

    channel = np.exp(-((yy - 64) ** 2) / (2 * 5.0**2)) * ((xx > 50) & (xx < 144))
    hotspot = np.exp(-(((yy - 44) ** 2) + ((xx - 124) ** 2)) / (2 * 10.0**2))
    lowfreq = 0.08 * np.sin(xx / 45.0) + 0.05 * np.cos(yy / 38.0)
    hr_temperature = (
        config.base_temp_c
        + config.delta_temp_c * mask.astype(np.float32)
        + 0.35 * channel
        + 0.25 * hotspot
        + lowfreq
    ).astype(np.float32)
    edge_y, edge_x = np.gradient(mask.astype(np.float32))
    edge_map = np.hypot(edge_y, edge_x).astype(np.float32)

    hr_shape_4x = (config.lr_shape[0] * 4, config.lr_shape[1] * 4)
    yy_4x, xx_4x = np.mgrid[: hr_shape_4x[0], : hr_shape_4x[1]]
    mask_4x = np.zeros(hr_shape_4x, dtype=np.uint8)
    mask_4x[48:208, 64:320] = 1
    mask_4x[96:160, 116:276] = 0
    channel_4x = np.exp(-((yy_4x - 128) ** 2) / (2 * 10.0**2)) * ((xx_4x > 100) & (xx_4x < 288))
    hotspot_4x = np.exp(-(((yy_4x - 88) ** 2) + ((xx_4x - 248) ** 2)) / (2 * 20.0**2))
    lowfreq_4x = 0.08 * np.sin(xx_4x / 90.0) + 0.05 * np.cos(yy_4x / 76.0)
    hr_temperature_4x = (
        config.base_temp_c
        + config.delta_temp_c * mask_4x.astype(np.float32)
        + 0.35 * channel_4x
        + 0.25 * hotspot_4x
        + lowfreq_4x
    ).astype(np.float32)
    return {
        "hr_mask_2x": mask,
        "hr_temperature_2x": hr_temperature.astype(np.float32),
        "hr_edge_map_2x": edge_map,
        "hr_mask_4x": mask_4x,
        "hr_temperature_4x": hr_temperature_4x,
    }, "notebook_fallback_geometry_physics"


def _make_scene(config: DemoConfig, *, use_tcforge: bool) -> tuple[dict[str, np.ndarray], str]:
    if not use_tcforge:
        return _fallback_make_scene(config)
    try:
        from tcforge.geometry import build_scene_mask
        from tcforge.physics import edge_map, render_temperature_field

        hr_shape = (config.lr_shape[0] * config.scale, config.lr_shape[1] * config.scale)
        mask = build_scene_mask(
            "easy",
            config.seed,
            canvas_shape=hr_shape,
            pixel_size_um=config.pixel_size_um,
            scale=config.scale,
            rotation_jitter_deg=0.0,
        )
        low_freq_sigma_px = max(8.0, min(hr_shape) / 12.0)
        hr_temperature = render_temperature_field(
            mask,
            t_bg_c=config.base_temp_c,
            delta_t_c=config.delta_temp_c,
            low_freq_amplitude_c=config.low_freq_amplitude_c,
            low_freq_sigma_px=low_freq_sigma_px,
            seed=config.seed + 17,
        )

        hr_shape_4x = (config.lr_shape[0] * 4, config.lr_shape[1] * 4)
        mask_4x = build_scene_mask(
            "easy",
            config.seed,
            canvas_shape=hr_shape_4x,
            pixel_size_um=config.pixel_size_um,
            scale=4,
            rotation_jitter_deg=0.0,
        )
        hr_temperature_4x = render_temperature_field(
            mask_4x,
            t_bg_c=config.base_temp_c,
            delta_t_c=config.delta_temp_c,
            low_freq_amplitude_c=config.low_freq_amplitude_c,
            low_freq_sigma_px=max(8.0, min(hr_shape_4x) / 12.0),
            seed=config.seed + 17,
        )

        return {
            "hr_mask_2x": np.asarray(mask, dtype=np.uint8),
            "hr_temperature_2x": np.asarray(hr_temperature, dtype=np.float32),
            "hr_edge_map_2x": np.asarray(edge_map(mask), dtype=np.float32),
            "hr_mask_4x": np.asarray(mask_4x, dtype=np.uint8),
            "hr_temperature_4x": np.asarray(hr_temperature_4x, dtype=np.float32),
        }, "tcforge.geometry+tcforge.physics"
    except Exception:
        return _fallback_make_scene(config)


def _phase_grid_shifts(n_frames: int) -> np.ndarray:
    phases = np.array(
        [(dx, dy) for dy in (0.0, 0.25, 0.5, 0.75) for dx in (0.0, 0.25, 0.5, 0.75)],
        dtype=np.float32,
    )
    if n_frames <= len(phases):
        return phases[:n_frames]
    reps = int(np.ceil(n_frames / len(phases)))
    return np.tile(phases, (reps, 1))[:n_frames]


def _fallback_forward_block_average(
    hr_temperature: np.ndarray,
    shifts_lr_dxdy: np.ndarray,
    config: DemoConfig,
) -> np.ndarray:
    frames = []
    for shift_lr_x, shift_lr_y in shifts_lr_dxdy:
        shifted = ndi_shift(
            hr_temperature,
            shift=(-shift_lr_y * config.scale, -shift_lr_x * config.scale),
            order=1,
            mode="nearest",
            prefilter=False,
        )
        lr = shifted.reshape(config.lr_shape[0], config.scale, config.lr_shape[1], config.scale).mean(axis=(1, 3))
        frames.append(lr.astype(np.float32))
    return np.stack(frames, axis=0).astype(np.float32)


def _make_shifts(config: DemoConfig, *, use_tcforge: bool) -> tuple[np.ndarray, str]:
    if use_tcforge:
        try:
            from tcforge.shifts import load_shift_profile

            shifts, info = load_shift_profile(
                "ideal_phase_grid",
                n_frames=config.n_frames,
                scale=config.scale,
                phase_steps=4,
                seed=config.seed,
            )
            return np.asarray(shifts, dtype=np.float32), f"tcforge.shifts.load_shift_profile({info['profile']})"
        except Exception:
            pass
    return _phase_grid_shifts(config.n_frames), "notebook_4x4_phase_grid"


def _make_lr_burst(
    hr_temperature: np.ndarray,
    shifts_lr_dxdy: np.ndarray,
    config: DemoConfig,
    *,
    forward_mode: str,
    use_tcforge: bool,
) -> tuple[np.ndarray, str]:
    if use_tcforge:
        try:
            from tcforge.forward import generate_lr_burst

            raw = generate_lr_burst(
                hr_temperature,
                shifts_lr_dxdy,
                forward_mode=forward_mode,
                psf_sigma_lr_px=config.psf_sigma_lr_px,
                scale=config.scale,
            )
            return raw.astype(np.float32, copy=False), f"tcforge.forward.generate_lr_burst({forward_mode})"
        except Exception:
            pass
    return (
        _fallback_forward_block_average(hr_temperature, shifts_lr_dxdy, config),
        "notebook_fallback_physical_block_average",
    )


def _make_highpass(frames: np.ndarray, config: DemoConfig, *, use_tcforge: bool) -> tuple[np.ndarray, str]:
    if use_tcforge:
        try:
            from tcforge.highpass import highpass_preprocess

            return (
                highpass_preprocess(frames, sigma_bg=config.highpass_sigma_lr_px, mode="nearest"),
                "tcforge.highpass.highpass_preprocess",
            )
        except Exception:
            pass
    return (
        ep06_like_highpass(frames, sigma_bg=config.highpass_sigma_lr_px),
        "notebook_ep06_like_highpass",
    )


def _crop_lr_image(image: np.ndarray, margin: int) -> np.ndarray:
    margin = int(margin)
    if margin <= 0:
        return image
    if min(image.shape) <= 2 * margin:
        return image
    return image[margin:-margin, margin:-margin]


def _crop_lr_stack(frames: np.ndarray, margin: int) -> np.ndarray:
    margin = int(margin)
    if margin <= 0:
        return frames
    if min(frames.shape[-2:]) <= 2 * margin:
        return frames
    return frames[:, margin:-margin, margin:-margin]


def _build_metadata(
    config: DemoConfig,
    *,
    scene_generation_mode: str,
    forward_mode: str,
    shift_source: str,
    highpass_source: str,
    tcforge_version: str,
) -> dict:
    return {
        "schema_version": "0.1-demo",
        "dataset": "ThermalChipPhantom",
        "engine": "TCForge",
        "scene_id": config.scene_id,
        "generator": "build_ep07_cache",
        "tcforge_import_status": tcforge_version,
        "scene_generation_mode": scene_generation_mode,
        "seed": config.seed,
        "scale": config.scale,
        "lr_shape": list(config.lr_shape),
        "hr_shape": [config.lr_shape[0] * config.scale, config.lr_shape[1] * config.scale],
        "pixel_size_um": config.pixel_size_um,
        "spatial_resolution_um": 20.0,
        "geometry": {
            "difficulty": "easy",
            "synthetic_truth_files": ["hr_mask_2x.npy", "hr_temperature_2x.npy", "hr_edge_map_2x.npy"],
        },
        "physics": {
            "T_bg_c": config.base_temp_c,
            "delta_T_c": config.delta_temp_c,
            "low_freq_amplitude_c": config.low_freq_amplitude_c,
            "noise_sigma_c": config.noise_sigma_c,
            "noise_model": config.noise_model,
            "fpn_sigma_px": config.fpn_sigma_px,
            "stripe_sigma_c": config.stripe_sigma_c,
            "noise_sigma_source": "configs/noise_floor.json",
            "noise_sigma_note": "0.0724 C is a smooth adjacent-coordinate MAE anchor, not proof of iid pixel noise",
            "psf_profile": config.psf_profile,
            "psf_profile_source": config.psf_profile_source,
            "psf_calibration_status": config.psf_calibration_status,
            "psf_four_x_verdict": config.psf_four_x_verdict,
            "psf_sigma_lr_px": config.psf_sigma_lr_px,
            "psf_sigma_hr_px": config.psf_sigma_hr_px,
            "forward_mode": forward_mode,
            "highpass_sigma_lr_px": config.highpass_sigma_lr_px,
            "highpass_mode": "nearest",
            "drift_model": "none",
            "noise_injection": "LR burst after forward (matches CLI generator)",
        },
        "shifts": {
            "source": shift_source,
            "convention": "LR-to-reference alignment shift",
            "columns": ["dx_px", "dy_px"],
        },
    }


def _plot_hr_scene(output_dir: Path, scene: dict[str, np.ndarray], config: DemoConfig) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.2))
    axes = axes.ravel()
    _show_image(axes[0], scene["hr_mask_2x"], title="HR mask 2x", cmap="gray", colorbar_label="Mask", vmin=0, vmax=1)
    _show_image(
        axes[1],
        scene["hr_temperature_2x"],
        title="HR temperature 2x",
        cmap=COLORMAPS["temperature"],
        colorbar_label="Temperature [$^\\circ$C]",
        robust=True,
        show_colorbar=False,
    )
    _show_image(
        axes[2],
        scene["hr_edge_map_2x"],
        title="HR edge proxy 2x",
        cmap=COLORMAPS["coverage"],
        colorbar_label="Edge",
        vmin=0,
        vmax=max(1.0, float(np.max(scene["hr_edge_map_2x"]))),
    )
    _show_image(axes[3], scene["hr_mask_4x"], title="HR mask 4x sanity", cmap="gray", colorbar_label="Mask", vmin=0, vmax=1)
    _show_image(
        axes[4],
        scene["hr_temperature_4x"],
        title="HR temperature 4x sanity",
        cmap=COLORMAPS["temperature"],
        colorbar_label="Temperature [$^\\circ$C]",
        robust=True,
        show_colorbar=False,
    )
    axes[5].axis("off")
    axes[5].text(0.0, 0.86, "Generation contract", fontsize=10, fontweight="bold", transform=axes[5].transAxes)
    axes[5].text(
        0.0,
        0.68,
        f"scene_id: {config.scene_id}\n"
        f"seed: {config.seed}\n"
        f"LR shape: {config.lr_shape[0]} x {config.lr_shape[1]}\n"
        f"2x HR shape: {scene['hr_temperature_2x'].shape[0]} x {scene['hr_temperature_2x'].shape[1]}\n"
        f"pixel pitch: {config.pixel_size_um:.1f} um/LR px",
        fontsize=8,
        va="top",
        transform=axes[5].transAxes,
    )
    for label, ax in zip(["a", "b", "c", "d", "e"], axes[:5], strict=True):
        _add_panel_label(ax, label)
    savefig_academic(fig, output_dir / "demo_hr_scene.png")


def _plot_forward_highpass(
    output_dir: Path,
    *,
    scene: dict[str, np.ndarray],
    shifts: np.ndarray,
    lr_burst_raw: np.ndarray,
    lr_burst_highpass: np.ndarray,
    block_preview: np.ndarray,
    config: DemoConfig,
    edge_margin: int,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.4))
    axes = axes.ravel()
    frame_idx = 5
    raw_frame_vis = _crop_lr_image(lr_burst_raw[frame_idx], edge_margin)
    highpass_frame_vis = _crop_lr_image(lr_burst_highpass[frame_idx], edge_margin)
    highpass_abs_vis = np.mean(np.abs(_crop_lr_stack(lr_burst_highpass, edge_margin)), axis=0)
    point_preview = _crop_lr_image(lr_burst_raw[0], edge_margin)
    block_preview_vis = _crop_lr_image(block_preview[0], edge_margin)
    forward_diff = point_preview - block_preview_vis

    _show_image(
        axes[0],
        raw_frame_vis,
        title="LR raw frame (interior)",
        cmap=COLORMAPS["temperature"],
        colorbar_label="Temperature [$^\\circ$C]",
        robust=True,
        show_colorbar=False,
    )
    _show_image(
        axes[1],
        highpass_frame_vis,
        title="LR highpass frame (interior)",
        cmap=COLORMAPS["residual_diff"],
        colorbar_label="Highpass [$^\\circ$C]",
        robust=True,
        symmetric=True,
    )
    _show_image(
        axes[2],
        highpass_abs_vis,
        title="Mean abs highpass (interior)",
        cmap="magma",
        colorbar_label="Abs response [$^\\circ$C]",
        robust=True,
    )
    axes[3].scatter(
        shifts[:, 0],
        shifts[:, 1],
        c=np.arange(len(shifts)),
        cmap=COLORMAPS["coverage"],
        s=36,
        edgecolors="black",
        linewidths=0.25,
    )
    axes[3].set_title("Sub-pixel phase coverage")
    axes[3].set_xlabel("dx [LR px]")
    axes[3].set_ylabel("dy [LR px]")
    axes[3].invert_yaxis()
    axes[3].set_aspect("equal", adjustable="box")
    axes[3].grid(True, alpha=0.25, linewidth=0.5)
    _show_image(
        axes[4],
        forward_diff,
        title="Point minus block preview",
        cmap=COLORMAPS["residual_diff"],
        colorbar_label="Difference [$^\\circ$C]",
        robust=True,
        symmetric=True,
    )
    axes[5].axis("off")
    axes[5].text(0.0, 0.9, "Forward/highpass settings", fontsize=10, fontweight="bold", transform=axes[5].transAxes)
    axes[5].text(
        0.0,
        0.72,
        f"forward: exact_ep06_point\n"
        f"PSF: {config.psf_profile}\n"
        f"sigma: {config.psf_sigma_lr_px:.3f} LR px\n"
        f"noise: {config.noise_model}, rms={config.noise_sigma_c:.4g} C\n"
        f"highpass sigma: {config.highpass_sigma_lr_px:.1f} LR px\n"
        f"frames: {config.n_frames}\n"
        f"shift max norm: {np.linalg.norm(shifts, axis=1).max():.3f} LR px",
        fontsize=8,
        va="top",
        transform=axes[5].transAxes,
    )
    for label, ax in zip(["a", "b", "c", "d", "e"], axes[:5], strict=True):
        _add_panel_label(ax, label)
    savefig_academic(fig, output_dir / "demo_forward_highpass.png")


def _plot_dataset_overview(
    output_dir: Path,
    demo_dir: Path,
    edge_margin: int,
) -> None:
    hr_temperature_disk = np.load(demo_dir / "hr_temperature_2x.npy")
    hr_mask_disk = np.load(demo_dir / "hr_mask_2x.npy")
    hr_edge_disk = np.load(demo_dir / "hr_edge_map_2x.npy")
    lr_raw_disk = np.load(demo_dir / "lr_burst_raw.npy")
    lr_hp_disk = np.load(demo_dir / "lr_burst_highpass.npy")
    shifts_disk = np.load(demo_dir / "shifts.npy")
    lr_raw_disk_vis = _crop_lr_stack(lr_raw_disk, edge_margin)
    lr_hp_disk_vis = _crop_lr_stack(lr_hp_disk, edge_margin)

    fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.4))
    axes = axes.ravel()
    _show_image(
        axes[0],
        hr_temperature_disk,
        title="Synthetic HR temperature truth",
        cmap=COLORMAPS["temperature"],
        colorbar_label="Temperature [$^\\circ$C]",
        robust=True,
        show_colorbar=False,
    )
    _show_image(axes[1], hr_mask_disk, title="HR structure mask", cmap="gray", colorbar_label="Mask", vmin=0, vmax=1)
    _show_image(
        axes[2],
        hr_edge_disk,
        title="HR contour proxy",
        cmap=COLORMAPS["coverage"],
        colorbar_label="Edge",
        vmin=0,
        vmax=max(1.0, float(hr_edge_disk.max())),
    )
    _show_image(
        axes[3],
        lr_raw_disk_vis.mean(axis=0),
        title="LR raw mean (interior)",
        cmap=COLORMAPS["temperature"],
        colorbar_label="Temperature [$^\\circ$C]",
        robust=True,
    )
    _show_image(
        axes[4],
        lr_hp_disk_vis.std(axis=0),
        title="LR highpass std (interior)",
        cmap="magma",
        colorbar_label="Std response [$^\\circ$C]",
        robust=True,
    )
    sc = axes[5].scatter(
        shifts_disk[:, 0],
        shifts_disk[:, 1],
        c=np.arange(len(shifts_disk)),
        cmap=COLORMAPS["coverage"],
        s=36,
        edgecolors="black",
        linewidths=0.25,
    )
    axes[5].set_title("Phase coverage")
    axes[5].set_xlabel("dx [LR px]")
    axes[5].set_ylabel("dy [LR px]")
    axes[5].invert_yaxis()
    axes[5].set_aspect("equal", adjustable="box")
    axes[5].grid(True, alpha=0.25, linewidth=0.5)
    format_colorbar(fig.colorbar(sc, ax=axes[5], fraction=0.046, pad=0.03), "Frame index")
    for label, ax in zip(["a", "b", "c", "d", "e", "f"], axes, strict=True):
        _add_panel_label(ax, label)
    savefig_academic(fig, output_dir / "demo_dataset_overview.png")


def _plot_profiles(
    output_dir: Path,
    demo_dir: Path,
    edge_margin: int,
    *,
    frame_idx: int = 5,
) -> None:
    hr_temperature_disk = np.load(demo_dir / "hr_temperature_2x.npy")
    hr_mask_disk = np.load(demo_dir / "hr_mask_2x.npy")
    lr_raw_disk = np.load(demo_dir / "lr_burst_raw.npy")
    lr_hp_disk = np.load(demo_dir / "lr_burst_highpass.npy")
    lr_raw_disk_vis = _crop_lr_stack(lr_raw_disk, edge_margin)
    lr_hp_disk_vis = _crop_lr_stack(lr_hp_disk, edge_margin)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    row_idx = int(hr_temperature_disk.shape[0] * 0.5)
    profile_slice = slice(int(hr_temperature_disk.shape[1] * 0.20), int(hr_temperature_disk.shape[1] * 0.82))
    profile_x_hr = np.arange(profile_slice.start, profile_slice.stop)
    mask_profile = hr_mask_disk[row_idx, profile_slice].astype(float)
    temp_profile = hr_temperature_disk[row_idx, profile_slice]
    temp_norm = (temp_profile - temp_profile.min()) / max(float(temp_profile.max() - temp_profile.min()), 1e-8)

    axes[0].plot(profile_x_hr, mask_profile, color="#222222", linestyle="--", linewidth=1.4, label="HR mask")
    axes[0].plot(profile_x_hr, temp_norm, color=METHOD_COLORS["accent_1"], linewidth=1.4, label="HR temperature (normalized)")
    axes[0].set_title("HR profile across chip structure")
    axes[0].set_xlabel("HR column [px]")
    axes[0].set_ylabel("Normalized value")
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2)

    lr_row_idx = int(lr_raw_disk_vis.shape[1] * 0.5)
    lr_profile_x = np.arange(lr_raw_disk_vis.shape[2])
    raw_profile = lr_raw_disk_vis[frame_idx, lr_row_idx]
    hp_profile = lr_hp_disk_vis[frame_idx, lr_row_idx]
    raw_norm = (raw_profile - raw_profile.min()) / max(float(raw_profile.max() - raw_profile.min()), 1e-8)
    hp_norm = hp_profile / max(float(np.max(np.abs(hp_profile))), 1e-8)
    axes[1].plot(lr_profile_x, raw_norm, color=METHOD_COLORS["primary"], linewidth=1.4, label="LR raw (normalized)")
    axes[1].plot(lr_profile_x, hp_norm, color=METHOD_COLORS["secondary"], linewidth=1.4, label="LR highpass (signed)")
    axes[1].axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
    axes[1].set_title("LR observation profile")
    axes[1].set_xlabel("Interior LR column [px]")
    axes[1].set_ylabel("Normalized / signed value")
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2)
    for label, ax in zip(["a", "b"], axes, strict=True):
        _add_panel_label(ax, label)
    savefig_academic(fig, output_dir / "demo_profiles_generation_vs_observation.png")


def _decompose_thermal_field(
    mask: np.ndarray,
    hr_temperature: np.ndarray,
    config: DemoConfig,
    *,
    use_tcforge: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return structure-only, low-frequency-only, and full clean HR temperature."""
    if use_tcforge:
        try:
            from tcforge.physics import render_temperature_field

            hr_shape = hr_temperature.shape
            structure_only = render_temperature_field(
                mask,
                t_bg_c=config.base_temp_c,
                delta_t_c=config.delta_temp_c,
                low_freq_amplitude_c=0.0,
                low_freq_sigma_px=max(8.0, min(hr_shape) / 12.0),
                seed=config.seed + 17,
            )
            lowfreq_only = (hr_temperature - structure_only).astype(np.float32, copy=False)
            return (
                np.asarray(structure_only, dtype=np.float32),
                lowfreq_only,
                np.asarray(hr_temperature, dtype=np.float32),
            )
        except Exception:
            pass
    structure_only = (
        config.base_temp_c + config.delta_temp_c * mask.astype(np.float32)
    ).astype(np.float32)
    lowfreq_only = (hr_temperature - structure_only).astype(np.float32, copy=False)
    return structure_only, lowfreq_only, np.asarray(hr_temperature, dtype=np.float32)


def _psf_kernel_2d(sigma_lr_px: float, scale: int, *, size: int = 31) -> np.ndarray:
    sigma_hr = max(0.0, float(sigma_lr_px) * int(scale))
    half = int(size) // 2
    yy, xx = np.mgrid[-half : half + 1, -half : half + 1]
    if sigma_hr <= 0:
        kernel = np.zeros((size, size), dtype=np.float32)
        kernel[half, half] = 1.0
        return kernel
    kernel = np.exp(-(xx**2 + yy**2) / (2.0 * sigma_hr**2)).astype(np.float32)
    kernel /= float(kernel.sum()) or 1.0
    return kernel


def _build_physics_checks(
    config: DemoConfig,
    *,
    structure_only: np.ndarray,
    lowfreq_only: np.ndarray,
    hr_temperature: np.ndarray,
    lr_burst_clean: np.ndarray,
    lr_burst_raw: np.ndarray,
    mask: np.ndarray,
) -> pd.DataFrame:
    frame_idx = min(5, lr_burst_clean.shape[0] - 1)
    noise_residual = (lr_burst_raw[frame_idx] - lr_burst_clean[frame_idx]).astype(np.float32)
    residual_std = float(np.std(noise_residual))
    residual_mean = float(np.mean(noise_residual))
    sigma_hr = max(0.0, config.psf_sigma_lr_px * config.scale)
    blurred_hr = gaussian_filter(hr_temperature.astype(np.float64), sigma=sigma_hr, mode="constant", cval=0.0)
    pre_grad_y, pre_grad_x = np.gradient(hr_temperature.astype(np.float64))
    post_grad_y, post_grad_x = np.gradient(blurred_hr.astype(np.float64))
    pre_grad = np.hypot(pre_grad_y, pre_grad_x)
    post_grad = np.hypot(post_grad_y, post_grad_x)
    crop = (slice(4, -4), slice(4, -4)) if min(pre_grad.shape) > 12 else (slice(None), slice(None))
    pre_edge_grad = float(np.percentile(pre_grad[crop], 95))
    post_edge_grad = float(np.percentile(post_grad[crop], 95))
    blur_ratio = post_edge_grad / max(pre_edge_grad, 1e-8)
    demo_snr = float(config.delta_temp_c / config.noise_sigma_c)
    mtf_2x = float(gaussian_mtf(0.5, config.psf_sigma_lr_px))
    effective_snr = demo_snr * mtf_2x

    rows = [
        {
            "component": "thermal_field",
            "check": "structure_delta_t_c",
            "value": config.delta_temp_c,
            "unit": "C",
            "expected": "phantom_smoke easy=2.5",
            "pass": abs(config.delta_temp_c - 2.5) < 1e-6,
        },
        {
            "component": "thermal_field",
            "check": "low_freq_peak_c",
            "value": float(np.max(np.abs(lowfreq_only))),
            "unit": "C",
            "expected": f"<= {config.low_freq_amplitude_c:.3g}",
            "pass": float(np.max(np.abs(lowfreq_only))) <= config.low_freq_amplitude_c + 1e-5,
        },
        {
            "component": "thermal_field",
            "check": "hr_temp_range_c",
            "value": float(hr_temperature.max() - hr_temperature.min()),
            "unit": "C",
            "expected": "finite, > delta_T",
            "pass": bool(np.isfinite(hr_temperature).all() and (hr_temperature.max() - hr_temperature.min()) > 0.5),
        },
        {
            "component": "psf",
            "check": "psf_sigma_lr_px",
            "value": config.psf_sigma_lr_px,
            "unit": "LR px",
            "expected": "EP09 Route A provisional ~=0.226; legacy_upper=0.5 only for stress",
            "pass": (
                config.psf_profile == "legacy_upper" and abs(config.psf_sigma_lr_px - 0.5) < 1e-6
            )
            or (
                config.psf_profile == "ep09_provisional"
                and 0.20 <= config.psf_sigma_lr_px <= 0.25
                and config.psf_calibration_status == "provisional_needs_review"
            ),
        },
        {
            "component": "psf",
            "check": "edge_gradient_blur_ratio",
            "value": blur_ratio,
            "unit": "ratio",
            "expected": "< 1.0 (PSF softens edges)",
            "pass": blur_ratio < 0.98,
        },
        {
            "component": "noise",
            "check": "configured_sigma_c",
            "value": config.noise_sigma_c,
            "unit": "C",
            "expected": "0.0724 (noise floor)",
            "pass": abs(config.noise_sigma_c - 0.0724) < 1e-6,
        },
        {
            "component": "noise",
            "check": "noise_model",
            "value": config.noise_model,
            "unit": "-",
            "expected": "mixed default; iid_gaussian retained as baseline",
            "pass": config.noise_model in {"iid_gaussian", "fpn_lowfreq", "column_stripe", "mixed"},
        },
        {
            "component": "noise",
            "check": "measured_residual_std_c",
            "value": residual_std,
            "unit": "C",
            "expected": f"~{config.noise_sigma_c:.4g}",
            "pass": abs(residual_std - config.noise_sigma_c) < 0.03,
        },
        {
            "component": "noise",
            "check": "residual_mean_c",
            "value": residual_mean,
            "unit": "C",
            "expected": "~0",
            "pass": abs(residual_mean) < 0.02,
        },
        {
            "component": "snr",
            "check": "demo_input_snr",
            "value": demo_snr,
            "unit": "ratio",
            "expected": ">= 3 (borderline recoverability)",
            "pass": demo_snr >= 3.0,
        },
        {
            "component": "snr",
            "check": "demo_effective_snr_2x",
            "value": effective_snr,
            "unit": "ratio",
            "expected": "delta_T * MTF(0.5) / noise",
            "pass": effective_snr >= 1.0,
        },
        {
            "component": "snr",
            "check": "mask_coverage",
            "value": float(np.mean(mask > 0)),
            "unit": "fraction",
            "expected": "0 < coverage < 1",
            "pass": 0.0 < float(np.mean(mask > 0)) < 1.0,
        },
    ]
    return pd.DataFrame(rows)


def _build_snr_budget(config: DemoConfig) -> pd.DataFrame:
    difficulty_delta_t = {
        "easy": 2.5,
        "medium": 1.5,
        "hard": 1.0,
        "stress": 0.7,
    }
    rows = []
    for difficulty, delta_t in difficulty_delta_t.items():
        input_snr = float(delta_t / config.noise_sigma_c)
        mtf = float(gaussian_mtf(0.5, config.psf_sigma_lr_px))
        effective_snr = input_snr * mtf
        if effective_snr >= 5.0:
            risk_band = "observable"
        elif effective_snr >= 3.0:
            risk_band = "borderline"
        elif effective_snr >= 1.0:
            risk_band = "weak"
        else:
            risk_band = "noise-dominated"
        rows.append(
            {
                "difficulty": difficulty,
                "delta_t_c": delta_t,
                "noise_sigma_c": config.noise_sigma_c,
                "input_snr": input_snr,
                "psf_sigma_lr_px": config.psf_sigma_lr_px,
                "mtf_at_2x_nyquist": mtf,
                "effective_snr_2x": effective_snr,
                "passes_3x_noise": bool(effective_snr >= 3.0),
                "passes_5x_noise": bool(effective_snr >= 5.0),
                "risk_band": risk_band,
            }
        )
    return pd.DataFrame(rows)


def _plot_thermal_field_decomposition(
    output_dir: Path,
    *,
    mask: np.ndarray,
    structure_only: np.ndarray,
    lowfreq_only: np.ndarray,
    hr_temperature: np.ndarray,
    config: DemoConfig,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    axes = axes.ravel()
    _show_image(axes[0], mask.astype(float), title="Structure mask", cmap="gray", colorbar_label="Mask", vmin=0, vmax=1)
    _show_image(
        axes[1],
        structure_only,
        title="Structure temperature only",
        cmap=COLORMAPS["temperature"],
        colorbar_label="Temperature [$^\\circ$C]",
        robust=True,
    )
    _show_image(
        axes[2],
        lowfreq_only,
        title="Low-frequency background only",
        cmap=COLORMAPS["residual_diff"],
        colorbar_label="Background [$^\\circ$C]",
        robust=True,
        symmetric=True,
    )
    _show_image(
        axes[3],
        hr_temperature,
        title="Full HR temperature (clean)",
        cmap=COLORMAPS["temperature"],
        colorbar_label="Temperature [$^\\circ$C]",
        robust=True,
    )
    row_idx = int(hr_temperature.shape[0] * 0.5)
    profile_slice = slice(int(hr_temperature.shape[1] * 0.18), int(hr_temperature.shape[1] * 0.82))
    profile_x = np.arange(profile_slice.start, profile_slice.stop)
    inset = axes[3].inset_axes([0.52, 0.08, 0.44, 0.34])
    inset.plot(profile_x, structure_only[row_idx, profile_slice], color=METHOD_COLORS["primary"], linewidth=1.0, label="structure")
    inset.plot(profile_x, lowfreq_only[row_idx, profile_slice], color=METHOD_COLORS["secondary"], linewidth=1.0, label="low-freq")
    inset.plot(profile_x, hr_temperature[row_idx, profile_slice], color="#222222", linewidth=1.2, label="sum")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.legend(fontsize=6, loc="upper right")
    for label, ax in zip(["a", "b", "c", "d"], axes, strict=True):
        _add_panel_label(ax, label)
    fig.suptitle(
        f"Thermal field decomposition | delta_T={config.delta_temp_c:.2f} C, low-freq amp={config.low_freq_amplitude_c:.2f} C",
        fontsize=10,
        y=1.02,
    )
    savefig_academic(fig, output_dir / "demo_thermal_field_decomposition.png")


def _plot_psf_blur_check(
    output_dir: Path,
    *,
    hr_temperature: np.ndarray,
    lr_burst_clean: np.ndarray,
    config: DemoConfig,
    edge_margin: int,
) -> None:
    sigma_hr = max(0.0, config.psf_sigma_lr_px * config.scale)
    blurred_hr = gaussian_filter(hr_temperature.astype(np.float64), sigma=sigma_hr, mode="constant", cval=0.0)
    kernel = _psf_kernel_2d(config.psf_sigma_lr_px, config.scale)
    lr_clean = _crop_lr_image(lr_burst_clean[0], edge_margin)
    hr_vis = hr_temperature
    blurred_vis = blurred_hr.astype(np.float32)

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    axes = axes.ravel()
    _show_image(axes[0], kernel, title="PSF kernel (HR grid)", cmap="magma", colorbar_label="Weight", vmin=0, vmax=float(kernel.max()))
    _show_image(
        axes[1],
        hr_vis,
        title="HR temperature before PSF",
        cmap=COLORMAPS["temperature"],
        colorbar_label="Temperature [$^\\circ$C]",
        robust=True,
        show_colorbar=False,
    )
    _show_image(
        axes[2],
        blurred_vis,
        title=f"HR after Gaussian blur (sigma={sigma_hr:.2f} HR px)",
        cmap=COLORMAPS["temperature"],
        colorbar_label="Temperature [$^\\circ$C]",
        robust=True,
        show_colorbar=False,
    )
    _show_image(
        axes[3],
        lr_clean,
        title="LR frame 0 after forward (no noise)",
        cmap=COLORMAPS["temperature"],
        colorbar_label="Temperature [$^\\circ$C]",
        robust=True,
    )

    row_idx = int(hr_vis.shape[0] * 0.5)
    profile_slice = slice(int(hr_vis.shape[1] * 0.40), int(hr_vis.shape[1] * 0.58))
    profile_x = np.arange(profile_slice.start, profile_slice.stop)
    inset = axes[3].inset_axes([0.52, 0.08, 0.44, 0.34])
    inset.plot(profile_x, hr_vis[row_idx, profile_slice], color=METHOD_COLORS["primary"], linewidth=1.0, label="pre-PSF")
    inset.plot(profile_x, blurred_vis[row_idx, profile_slice], color=METHOD_COLORS["secondary"], linewidth=1.0, label="post-PSF")
    inset.set_xticks([])
    inset.set_yticks([])
    inset.legend(fontsize=6, loc="upper right")
    for label, ax in zip(["a", "b", "c", "d"], axes, strict=True):
        _add_panel_label(ax, label)
    fig.suptitle(
        f"PSF blur check | sigma_LR={config.psf_sigma_lr_px:.2f} px, sigma_HR={sigma_hr:.2f} px",
        fontsize=10,
        y=1.02,
    )
    savefig_academic(fig, output_dir / "demo_psf_blur_check.png")


def _plot_noise_check(
    output_dir: Path,
    *,
    lr_burst_clean: np.ndarray,
    lr_burst_raw: np.ndarray,
    config: DemoConfig,
    edge_margin: int,
) -> None:
    frame_idx = min(5, lr_burst_clean.shape[0] - 1)
    clean = _crop_lr_image(lr_burst_clean[frame_idx], edge_margin)
    noisy = _crop_lr_image(lr_burst_raw[frame_idx], edge_margin)
    residual = (noisy - clean).astype(np.float32)
    residual_flat = residual.ravel()

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 5.6))
    axes = axes.ravel()
    _show_image(
        axes[0],
        clean,
        title=f"LR frame {frame_idx} (clean forward)",
        cmap=COLORMAPS["temperature"],
        colorbar_label="Temperature [$^\\circ$C]",
        robust=True,
    )
    _show_image(
        axes[1],
        noisy,
        title=f"LR frame {frame_idx} (+ {config.noise_model} noise)",
        cmap=COLORMAPS["temperature"],
        colorbar_label="Temperature [$^\\circ$C]",
        robust=True,
    )
    _show_image(
        axes[2],
        residual,
        title="Noise residual (noisy - clean)",
        cmap=COLORMAPS["residual_diff"],
        colorbar_label="Residual [$^\\circ$C]",
        robust=True,
        symmetric=True,
    )
    axes[3].hist(residual_flat, bins=40, color=METHOD_COLORS["primary"], alpha=0.85, edgecolor="white", linewidth=0.4)
    measured_std = float(np.std(residual_flat))
    x = np.linspace(residual_flat.min(), residual_flat.max(), 200)
    if measured_std > 0:
        pdf = np.exp(-0.5 * ((x - float(np.mean(residual_flat))) / measured_std) ** 2) / (measured_std * np.sqrt(2 * np.pi))
        pdf *= residual_flat.size * (x[1] - x[0])
        axes[3].plot(x, pdf, color=METHOD_COLORS["secondary"], linewidth=1.4, label="Gaussian with measured std")
    axes[3].axvline(0.0, color="#666666", linestyle="--", linewidth=0.8)
    axes[3].set_title("Residual histogram")
    axes[3].set_xlabel("Residual [$^\\circ$C]")
    axes[3].set_ylabel("Pixel count")
    axes[3].grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[3].text(
        0.98,
        0.95,
        f"model = {config.noise_model}\nconfigured rms = {config.noise_sigma_c:.4g} C\nmeasured std = {measured_std:.4g} C",
        transform=axes[3].transAxes,
        ha="right",
        va="top",
        fontsize=8,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85, "pad": 2.0},
    )
    for label, ax in zip(["a", "b", "c", "d"], axes, strict=True):
        _add_panel_label(ax, label)
    fig.suptitle("Detector noise injection check (LR burst, post-forward; RMS anchored)", fontsize=10, y=1.02)
    savefig_academic(fig, output_dir / "demo_noise_check.png")


def _lag1_corr(image: np.ndarray) -> float:
    arr = np.asarray(image, dtype=np.float64)
    arr = arr - float(np.mean(arr))
    if arr.shape[1] < 2:
        return float("nan")
    a = arr[:, :-1].ravel()
    b = arr[:, 1:].ravel()
    denom = float(np.std(a) * np.std(b))
    if denom <= 0:
        return float("nan")
    return float(np.mean((a - a.mean()) * (b - b.mean())) / denom)


def _build_noise_model_checks(config: DemoConfig, *, use_tcforge: bool) -> pd.DataFrame:
    rows = []
    models = ("iid_gaussian", "fpn_lowfreq", "column_stripe", "mixed")
    for idx, model in enumerate(models):
        if use_tcforge:
            try:
                from tcforge.physics import make_noise

                residual = make_noise(
                    config.lr_shape,
                    noise_sigma_c=config.noise_sigma_c,
                    seed=config.seed + 500 + idx,
                    noise_model=model,
                    fpn_sigma_px=config.fpn_sigma_px,
                    stripe_sigma_c=config.stripe_sigma_c,
                )
            except Exception:
                rng = np.random.default_rng(config.seed + 500 + idx)
                residual = rng.normal(0.0, config.noise_sigma_c, size=config.lr_shape).astype(np.float32)
        else:
            rng = np.random.default_rng(config.seed + 500 + idx)
            residual = rng.normal(0.0, config.noise_sigma_c, size=config.lr_shape).astype(np.float32)
        col_bias = np.mean(residual, axis=0)
        row_bias = np.mean(residual, axis=1)
        rows.append(
            {
                "noise_model": model,
                "mean_c": float(np.mean(residual)),
                "std_c": float(np.std(residual)),
                "target_sigma_c": config.noise_sigma_c,
                "lag1_column_corr": _lag1_corr(residual),
                "column_bias_std_c": float(np.std(col_bias)),
                "row_bias_std_c": float(np.std(row_bias)),
                "fpn_sigma_px": config.fpn_sigma_px,
                "stripe_sigma_c": config.stripe_sigma_c,
            }
        )
    return pd.DataFrame(rows)


def _center_crop(image: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    rows, cols = map(int, shape)
    arr = np.asarray(image, dtype=np.float32)
    if arr.shape[0] < rows or arr.shape[1] < cols:
        return arr
    r0 = (arr.shape[0] - rows) // 2
    c0 = (arr.shape[1] - cols) // 2
    return arr[r0 : r0 + rows, c0 : c0 + cols].astype(np.float32, copy=False)


def _load_real_residual_crop(root: Path, config: DemoConfig) -> tuple[np.ndarray | None, str]:
    audit_path = root / "output" / "ep01_data_processing" / "frame_audit.csv"
    data_dir = root / "data" / "data_raw" / "infrared_avi"
    if not audit_path.exists() or not data_dir.exists():
        return None, "real frame unavailable: missing EP01 audit or raw data"
    try:
        from thermal_core.io import load_frame

        audit = pd.read_csv(audit_path)
        if "is_sr_usable" in audit.columns:
            subset = audit[_boolish(audit["is_sr_usable"])].copy()
        elif "is_main_session" in audit.columns:
            subset = audit[_boolish(audit["is_main_session"])].copy()
        else:
            subset = audit.copy()
        subset["acquisition_order"] = pd.to_numeric(subset["acquisition_order"], errors="coerce")
        subset = subset.sort_values(["acquisition_order", "file"]).reset_index(drop=True)
        frame = np.asarray(load_frame(data_dir / str(subset.iloc[0]["file"])), dtype=np.float32)
        residual = ep06_like_highpass(frame, sigma_bg=config.highpass_sigma_lr_px)
        return _center_crop(residual, config.lr_shape), f"real clean main-session frame: {subset.iloc[0]['file']}"
    except Exception as exc:
        return None, f"real frame unavailable: {exc.__class__.__name__}"


def _radial_power_spectrum(image: np.ndarray, *, n_bins: int = 28) -> tuple[np.ndarray, np.ndarray]:
    arr = np.asarray(image, dtype=np.float64)
    arr = arr - float(np.mean(arr))
    window_y = np.hanning(arr.shape[0])[:, None]
    window_x = np.hanning(arr.shape[1])[None, :]
    spec = np.abs(np.fft.fftshift(np.fft.fft2(arr * window_y * window_x))) ** 2
    yy, xx = np.mgrid[: arr.shape[0], : arr.shape[1]]
    cy = (arr.shape[0] - 1) / 2.0
    cx = (arr.shape[1] - 1) / 2.0
    radius = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)
    rmax = float(radius.max())
    bins = np.linspace(0.0, rmax, n_bins + 1)
    centers = 0.5 * (bins[:-1] + bins[1:]) / max(rmax, 1e-8)
    power = np.zeros(n_bins, dtype=np.float64)
    for idx in range(n_bins):
        mask = (radius >= bins[idx]) & (radius < bins[idx + 1])
        power[idx] = float(np.mean(spec[mask])) if np.any(mask) else np.nan
    finite = np.isfinite(power) & (power > 0)
    if np.any(finite):
        power = power / float(np.nanmax(power[finite]))
    return centers.astype(np.float32), power.astype(np.float32)


def _plot_noise_real_vs_synthetic(
    output_dir: Path,
    *,
    root: Path,
    lr_burst_clean: np.ndarray,
    lr_burst_raw: np.ndarray,
    config: DemoConfig,
) -> None:
    real_residual, real_source = _load_real_residual_crop(root, config)
    synthetic_residual = (lr_burst_raw[0] - lr_burst_clean[0]).astype(np.float32)
    synthetic_residual = _center_crop(synthetic_residual, config.lr_shape)
    synth_label = f"synthetic residual: {config.noise_model}, sigma={config.noise_sigma_c:.4g} C"

    fig, axes = plt.subplots(2, 2, figsize=(7.4, 5.6))
    axes = axes.ravel()
    if real_residual is None:
        axes[0].axis("off")
        axes[0].text(0.02, 0.8, real_source, transform=axes[0].transAxes, fontsize=8, va="top")
        real_for_plot = np.zeros_like(synthetic_residual)
    else:
        real_for_plot = real_residual
        _show_image(
            axes[0],
            real_residual,
            title="Real residual crop",
            cmap=COLORMAPS["residual_diff"],
            colorbar_label="Residual [$^\\circ$C]",
            robust=True,
            symmetric=True,
        )
    _show_image(
        axes[1],
        synthetic_residual,
        title="Synthetic noise residual",
        cmap=COLORMAPS["residual_diff"],
        colorbar_label="Residual [$^\\circ$C]",
        robust=True,
        symmetric=True,
    )
    x_real = np.arange(real_for_plot.shape[1])
    x_synth = np.arange(synthetic_residual.shape[1])
    axes[2].plot(x_real, np.mean(real_for_plot, axis=0), color=METHOD_COLORS["primary"], linewidth=1.2, label="real column mean")
    axes[2].plot(x_synth, np.mean(synthetic_residual, axis=0), color=METHOD_COLORS["secondary"], linewidth=1.2, label="synthetic column mean")
    axes[2].axhline(0.0, color="#777777", linestyle="--", linewidth=0.8)
    axes[2].set_title("Column mean residual")
    axes[2].set_xlabel("Column [px]")
    axes[2].set_ylabel("Mean residual [$^\\circ$C]")
    axes[2].grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[2].legend(fontsize=7)

    freq_s, power_s = _radial_power_spectrum(synthetic_residual)
    axes[3].plot(freq_s, power_s, color=METHOD_COLORS["secondary"], linewidth=1.4, label="synthetic")
    if real_residual is not None:
        freq_r, power_r = _radial_power_spectrum(real_residual)
        axes[3].plot(freq_r, power_r, color=METHOD_COLORS["primary"], linewidth=1.4, label="real")
    axes[3].set_yscale("log")
    axes[3].set_title("Radial power spectrum (normalized)")
    axes[3].set_xlabel("Normalized spatial frequency")
    axes[3].set_ylabel("Relative power")
    axes[3].grid(True, alpha=0.25, linewidth=0.5)
    axes[3].legend(fontsize=7)
    axes[3].text(
        0.02,
        0.04,
        f"{real_source}\n{synth_label}",
        transform=axes[3].transAxes,
        ha="left",
        va="bottom",
        fontsize=7,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.85, "pad": 2.0},
    )
    for label, ax in zip(["a", "b", "c", "d"], axes, strict=True):
        _add_panel_label(ax, label)
    fig.suptitle("Real vs synthetic residual texture check (single-frame, lightweight)", fontsize=10, y=1.02)
    savefig_academic(fig, output_dir / "demo_noise_real_vs_synthetic.png")


def _plot_snr_budget(output_dir: Path, snr_budget: pd.DataFrame, config: DemoConfig) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.2))
    difficulties = snr_budget["difficulty"].tolist()
    x = np.arange(len(difficulties))
    width = 0.34
    axes[0].bar(x - width / 2, snr_budget["input_snr"], width=width, color=METHOD_COLORS["primary"], label="input SNR")
    axes[0].bar(x + width / 2, snr_budget["effective_snr_2x"], width=width, color=METHOD_COLORS["secondary"], label="effective SNR (2x)")
    axes[0].axhline(3.0, color="#666666", linestyle="--", linewidth=0.9, label="3x noise gate")
    axes[0].axhline(5.0, color="#999999", linestyle=":", linewidth=0.9, label="5x noise gate")
    axes[0].set_xticks(x, difficulties)
    axes[0].set_ylabel("SNR = contrast / noise")
    axes[0].set_title("Difficulty SNR budget")
    axes[0].grid(axis="y", alpha=0.25, linewidth=0.5)
    axes[0].legend(loc="upper right", fontsize=7)

    risk_colors = {
        "observable": METHOD_COLORS["accent_1"],
        "borderline": METHOD_COLORS["secondary"],
        "weak": METHOD_COLORS["primary"],
        "noise-dominated": "#888888",
    }
    colors = [risk_colors.get(str(row), "#888888") for row in snr_budget["risk_band"]]
    axes[1].bar(x, snr_budget["effective_snr_2x"], color=colors, edgecolor="black", linewidth=0.25)
    axes[1].set_xticks(x, difficulties)
    axes[1].set_ylabel("Effective SNR (2x)")
    axes[1].set_title("Risk band by difficulty")
    axes[1].grid(axis="y", alpha=0.25, linewidth=0.5)
    for idx, row in snr_budget.iterrows():
        axes[1].text(
            int(idx),
            float(row["effective_snr_2x"]) + 0.15,
            str(row["risk_band"]),
            ha="center",
            va="bottom",
            fontsize=7,
            rotation=25,
        )
    for label, ax in zip(["a", "b"], axes, strict=True):
        _add_panel_label(ax, label)
    fig.suptitle(
        f"SNR budget | noise floor={config.noise_sigma_c:.4g} C, PSF sigma={config.psf_sigma_lr_px:.2f} LR px",
        fontsize=10,
        y=1.04,
    )
    savefig_academic(fig, output_dir / "demo_snr_budget.png")


def _summarize_demo_metrics(
    demo_dir: Path,
    config: DemoConfig,
    *,
    use_tcforge: bool,
) -> tuple[pd.DataFrame, str]:
    hr_mask = np.load(demo_dir / "hr_mask_2x.npy")
    hr_edge = np.load(demo_dir / "hr_edge_map_2x.npy")
    lr_raw = np.load(demo_dir / "lr_burst_raw.npy")
    lr_hp = np.load(demo_dir / "lr_burst_highpass.npy")
    shifts = np.load(demo_dir / "shifts.npy")

    if use_tcforge:
        try:
            from tcforge import evaluate as tc_eval

            summary = tc_eval.summarize_scene(demo_dir)
            eval_source = "tcforge.evaluate.summarize_scene"
        except Exception as exc:
            summary = None
            eval_source = f"notebook fallback summary ({exc.__class__.__name__})"
    else:
        summary = None
        eval_source = "notebook fallback summary (tcforge unavailable)"

    if summary is None:
        shift_norms = np.linalg.norm(shifts, axis=1)
        summary = {
            "scene_id": config.scene_id,
            "n_frames": int(lr_raw.shape[0]),
            "lr_rows": int(lr_raw.shape[1]),
            "lr_cols": int(lr_raw.shape[2]),
            "hr_rows": int(np.load(demo_dir / "hr_temperature_2x.npy").shape[0]),
            "hr_cols": int(np.load(demo_dir / "hr_temperature_2x.npy").shape[1]),
            "mask_coverage": float(np.mean(hr_mask > 0)),
            "edge_density": float(np.mean(hr_edge > 0)),
            "lr_raw_mean_c": float(np.mean(lr_raw)),
            "lr_raw_std_c": float(np.std(lr_raw)),
            "lr_highpass_abs_p95_c": float(np.percentile(np.abs(lr_hp), 95)),
            "shift_norm_mean_px": float(np.mean(shift_norms)),
            "shift_norm_max_px": float(np.max(shift_norms)),
        }

    metric_explain = {
        "n_frames": "synthetic LR burst length",
        "mask_coverage": "fraction of HR pixels inside the synthetic chip structure",
        "edge_density": "fraction of HR pixels marked by the contour proxy",
        "lr_raw_mean_c": "ordinary LR temperature mean",
        "lr_raw_std_c": "ordinary LR temperature standard deviation",
        "lr_highpass_abs_p95_c": "95th percentile of absolute highpass response",
        "shift_norm_mean_px": "mean shift magnitude in LR pixels",
        "shift_norm_max_px": "max shift magnitude in LR pixels",
    }
    rows = []
    for key, explanation in metric_explain.items():
        rows.append({"metric": key, "value": summary.get(key, np.nan), "interpretation": explanation})
    return pd.DataFrame(rows), eval_source


def build_ep07_cache(
    *,
    project_root_path: Path | None = None,
    output_dir: Path | None = None,
    demo_config: DemoConfig | None = None,
    force: bool = False,
) -> Ep07Cache:
    """Generate EP07 demo dataset, figures, tables, and cache manifest."""
    root = project_root(project_root_path)
    output_dir = (output_dir or root / "output" / "ep07_thermal_chip_phantom").resolve()
    demo_dir = output_dir / "demo_dataset"
    config = _resolve_demo_config(demo_config or DemoConfig(), root)
    output_dir.mkdir(parents=True, exist_ok=True)
    demo_dir.mkdir(parents=True, exist_ok=True)

    tcforge_available = _ensure_tcforge_path(root)
    tcforge_version = "unknown"
    if tcforge_available:
        import tcforge

        tcforge_version = getattr(tcforge, "__version__", "unknown")

    if not force and cache_is_complete(output_dir, EP07_CACHE_ARTIFACTS):
        manifest = load_manifest(output_dir)
        if manifest.get("demo_skipped") is False or (
            manifest.get("demo_skipped") is True and not tcforge_available
        ):
            return load_ep07_cache(output_dir=output_dir, project_root_path=root)

    if not tcforge_available:
        manifest = write_manifest(
            output_dir,
            version=EP07_CACHE_VERSION,
            artifacts=["cache_manifest.json"],
            rebuild_command=REBUILD_COMMAND,
            extra={
                "demo_skipped": True,
                "tcforge_available": False,
                "tcforge_src": str(_tcforge_src(root)),
            },
        )
        return Ep07Cache(
            output_dir=output_dir,
            demo_dir=demo_dir,
            manifest=manifest,
            demo_config=config,
            scene_generation_mode="skipped",
            forward_mode="skipped",
            shift_source="skipped",
            highpass_source="skipped",
            block_forward_mode="skipped",
            edge_vis_margin_lr_px=0,
            eval_source="skipped",
            scene_stats=pd.DataFrame(),
            forward_stats=pd.DataFrame(),
            physics_checks=pd.DataFrame(),
            snr_budget=pd.DataFrame(),
            noise_model_checks=pd.DataFrame(),
            demo_metrics=pd.DataFrame(),
            metadata={},
            demo_skipped=True,
            tcforge_available=False,
        )

    setup_academic_style()
    scene, scene_generation_mode = _make_scene(config, use_tcforge=True)
    for name, arr in scene.items():
        np.save(demo_dir / f"{name}.npy", arr)

    shifts, shift_source = _make_shifts(config, use_tcforge=True)
    lr_burst_clean, forward_mode = _make_lr_burst(
        scene["hr_temperature_2x"],
        shifts,
        config,
        forward_mode="exact_ep06_point",
        use_tcforge=True,
    )
    if config.noise_sigma_c > 0 and tcforge_available:
        try:
            from tcforge.physics import add_noise

            lr_burst_raw = add_noise(
                lr_burst_clean,
                noise_sigma_c=config.noise_sigma_c,
                seed=config.seed + 23,
                noise_model=config.noise_model,
                fpn_sigma_px=config.fpn_sigma_px,
                stripe_sigma_c=config.stripe_sigma_c,
            )
            noise_source = f"tcforge.physics.add_noise({config.noise_model}; LR burst, post-forward)"
        except Exception:
            rng = np.random.default_rng(config.seed + 23)
            lr_burst_raw = lr_burst_clean + rng.normal(0.0, config.noise_sigma_c, size=lr_burst_clean.shape).astype(np.float32)
            noise_source = "notebook_gaussian_noise fallback (LR burst, post-forward)"
    else:
        lr_burst_raw = lr_burst_clean.copy()
        noise_source = "none"
    lr_burst_highpass, highpass_source = _make_highpass(lr_burst_raw, config, use_tcforge=True)
    block_preview, block_forward_mode = _make_lr_burst(
        scene["hr_temperature_2x"],
        shifts[:1],
        config,
        forward_mode="physical_block_average",
        use_tcforge=True,
    )
    edge_margin = int(np.ceil(2.5 * config.highpass_sigma_lr_px))

    np.save(demo_dir / "shifts.npy", shifts)
    np.save(demo_dir / "lr_burst_raw.npy", lr_burst_raw)
    np.save(demo_dir / "lr_burst_highpass.npy", lr_burst_highpass)

    scene_stats = pd.DataFrame(
        [
            array_contract_row("hr_mask_2x", scene["hr_mask_2x"], role="2x synthetic binary structure truth"),
            array_contract_row("hr_temperature_2x", scene["hr_temperature_2x"], role="2x synthetic HR temperature truth [C]"),
            array_contract_row("hr_edge_map_2x", scene["hr_edge_map_2x"], role="2x contour proxy"),
            array_contract_row("hr_mask_4x", scene["hr_mask_4x"], role="4x display sanity mask"),
            array_contract_row("hr_temperature_4x", scene["hr_temperature_4x"], role="4x display sanity temperature [C]"),
        ]
    )
    forward_stats = pd.DataFrame(
        [
            array_contract_row("shifts", shifts, role="LR-to-reference alignment shifts [LR px]"),
            array_contract_row("lr_burst_raw", lr_burst_raw, role="ordinary LR temperature observations [C]"),
            array_contract_row("lr_burst_highpass", lr_burst_highpass, role="EP06-compatible structure response [C]"),
        ]
    )
    structure_only, lowfreq_only, hr_temperature_clean = _decompose_thermal_field(
        scene["hr_mask_2x"],
        scene["hr_temperature_2x"],
        config,
        use_tcforge=True,
    )
    physics_checks = _build_physics_checks(
        config,
        structure_only=structure_only,
        lowfreq_only=lowfreq_only,
        hr_temperature=hr_temperature_clean,
        lr_burst_clean=lr_burst_clean,
        lr_burst_raw=lr_burst_raw,
        mask=scene["hr_mask_2x"],
    )
    snr_budget = _build_snr_budget(config)
    noise_model_checks = _build_noise_model_checks(config, use_tcforge=True)
    demo_metrics, eval_source = _summarize_demo_metrics(demo_dir, config, use_tcforge=True)
    metadata = _build_metadata(
        config,
        scene_generation_mode=scene_generation_mode,
        forward_mode=forward_mode,
        shift_source=shift_source,
        highpass_source=highpass_source,
        tcforge_version=tcforge_version,
    )
    metadata["physics"]["noise_injection"] = noise_source
    (demo_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    (output_dir / "demo_metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")

    scene_stats.to_csv(output_dir / "scene_stats.csv", index=False)
    forward_stats.to_csv(output_dir / "forward_stats.csv", index=False)
    physics_checks.to_csv(output_dir / "physics_checks.csv", index=False)
    snr_budget.to_csv(output_dir / "snr_budget.csv", index=False)
    noise_model_checks.to_csv(output_dir / "noise_model_checks.csv", index=False)
    demo_metrics.to_csv(output_dir / "demo_metrics.csv", index=False)

    _plot_hr_scene(output_dir, scene, config)
    _plot_thermal_field_decomposition(
        output_dir,
        mask=scene["hr_mask_2x"],
        structure_only=structure_only,
        lowfreq_only=lowfreq_only,
        hr_temperature=hr_temperature_clean,
        config=config,
    )
    _plot_psf_blur_check(
        output_dir,
        hr_temperature=hr_temperature_clean,
        lr_burst_clean=lr_burst_clean,
        config=config,
        edge_margin=edge_margin,
    )
    _plot_noise_check(
        output_dir,
        lr_burst_clean=lr_burst_clean,
        lr_burst_raw=lr_burst_raw,
        config=config,
        edge_margin=edge_margin,
    )
    _plot_noise_real_vs_synthetic(
        output_dir,
        root=root,
        lr_burst_clean=lr_burst_clean,
        lr_burst_raw=lr_burst_raw,
        config=config,
    )
    _plot_snr_budget(output_dir, snr_budget, config)
    _plot_forward_highpass(
        output_dir,
        scene=scene,
        shifts=shifts,
        lr_burst_raw=lr_burst_raw,
        lr_burst_highpass=lr_burst_highpass,
        block_preview=block_preview,
        config=config,
        edge_margin=edge_margin,
    )
    _plot_dataset_overview(output_dir, demo_dir, edge_margin)
    _plot_profiles(output_dir, demo_dir, edge_margin)

    manifest = write_manifest(
        output_dir,
        version=EP07_CACHE_VERSION,
        artifacts=EP07_CACHE_ARTIFACTS,
        rebuild_command=REBUILD_COMMAND,
        extra={
            "demo_skipped": False,
            "tcforge_available": True,
            "scene_generation_mode": scene_generation_mode,
            "forward_mode": forward_mode,
            "shift_source": shift_source,
            "highpass_source": highpass_source,
            "demo_config": asdict(config),
        },
    )
    return Ep07Cache(
        output_dir=output_dir,
        demo_dir=demo_dir,
        manifest=manifest,
        demo_config=config,
        scene_generation_mode=scene_generation_mode,
        forward_mode=forward_mode,
        shift_source=shift_source,
        highpass_source=highpass_source,
        block_forward_mode=block_forward_mode,
        edge_vis_margin_lr_px=edge_margin,
        eval_source=eval_source,
        scene_stats=scene_stats,
        forward_stats=forward_stats,
        physics_checks=physics_checks,
        snr_budget=snr_budget,
        noise_model_checks=noise_model_checks,
        demo_metrics=demo_metrics,
        metadata=metadata,
        demo_skipped=False,
        tcforge_available=True,
    )


def load_ep07_cache(
    *,
    output_dir: Path | None = None,
    project_root_path: Path | None = None,
    require_complete: bool = True,
) -> Ep07Cache:
    root = project_root(project_root_path)
    output_dir = (output_dir or root / "output" / "ep07_thermal_chip_phantom").resolve()
    demo_dir = output_dir / "demo_dataset"
    manifest = load_manifest(output_dir) if (output_dir / "cache_manifest.json").exists() else {}

    if manifest.get("demo_skipped"):
        config_dict = manifest.get("demo_config") or asdict(DemoConfig())
        return Ep07Cache(
            output_dir=output_dir,
            demo_dir=demo_dir,
            manifest=manifest,
            demo_config=DemoConfig(**{k: v for k, v in config_dict.items() if k in DemoConfig.__dataclass_fields__}),
            scene_generation_mode="skipped",
            forward_mode="skipped",
            shift_source="skipped",
            highpass_source="skipped",
            block_forward_mode="skipped",
            edge_vis_margin_lr_px=0,
            eval_source="skipped",
            scene_stats=pd.DataFrame(),
            forward_stats=pd.DataFrame(),
            physics_checks=pd.DataFrame(),
            snr_budget=pd.DataFrame(),
            noise_model_checks=pd.DataFrame(),
            demo_metrics=pd.DataFrame(),
            metadata={},
            demo_skipped=True,
            tcforge_available=bool(manifest.get("tcforge_available")),
        )

    if require_complete:
        require_artifacts(output_dir, EP07_CACHE_ARTIFACTS, rebuild_command=REBUILD_COMMAND)

    config_dict = manifest.get("demo_config") or asdict(DemoConfig())
    config = DemoConfig(**{k: v for k, v in config_dict.items() if k in DemoConfig.__dataclass_fields__})
    if isinstance(config.lr_shape, list):
        config = DemoConfig(**{**asdict(config), "lr_shape": tuple(config.lr_shape)})

    metadata_path = output_dir / "demo_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    return Ep07Cache(
        output_dir=output_dir,
        demo_dir=demo_dir,
        manifest=manifest,
        demo_config=config,
        scene_generation_mode=str(manifest.get("scene_generation_mode", metadata.get("scene_generation_mode", ""))),
        forward_mode=str(manifest.get("forward_mode", metadata.get("physics", {}).get("forward_mode", ""))),
        shift_source=str(manifest.get("shift_source", metadata.get("shifts", {}).get("source", ""))),
        highpass_source=str(manifest.get("highpass_source", "")),
        block_forward_mode=str(manifest.get("block_forward_mode", "")),
        edge_vis_margin_lr_px=int(np.ceil(2.5 * config.highpass_sigma_lr_px)),
        eval_source="cache",
        scene_stats=pd.read_csv(output_dir / "scene_stats.csv"),
        forward_stats=pd.read_csv(output_dir / "forward_stats.csv"),
        physics_checks=pd.read_csv(output_dir / "physics_checks.csv"),
        snr_budget=pd.read_csv(output_dir / "snr_budget.csv"),
        noise_model_checks=pd.read_csv(output_dir / "noise_model_checks.csv"),
        demo_metrics=pd.read_csv(output_dir / "demo_metrics.csv"),
        metadata=metadata,
        demo_skipped=False,
        tcforge_available=bool(manifest.get("tcforge_available", True)),
    )
