#!/usr/bin/env python3
"""Generate a compact TCForge synthetic training pool."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
import traceback
from collections.abc import Mapping, Sequence
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TCFORGE_SRC = PROJECT_ROOT / "tcforge" / "src"
if TCFORGE_SRC.exists() and str(TCFORGE_SRC) not in sys.path:
    sys.path.insert(0, str(TCFORGE_SRC))

from tcforge import (  # noqa: E402
    add_detector_defects,
    add_noise,
    apply_drift,
    build_scene_mask_with_metadata,
    edge_map,
    fuse_burst_to_features,
    generate_lr_burst,
    load_shift_profile,
    render_temperature_field,
    save_scene_compact,
    sample_psf_parameters,
    shift_and_add,
)


DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "synthetic" / "training_pool_4x.json"
SUPPORTED_SCALES = {2, 4}
MANIFEST_FIELDS = [
    "scene_id",
    "scene_dir",
    "difficulty",
    "seed",
    "scale",
    "lr_shape",
    "hr_shape",
    "T_bg_c",
    "delta_T_c",
    "psf_sigma_lr_px",
    "noise_sigma_c",
    "drift_model",
    "rotation_deg",
    "n_frames",
]


@dataclass(frozen=True)
class ScenePlan:
    scene_index: int
    scene_id: str
    seed: int
    difficulty: str
    drift_model: str


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_path(value: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return base / path


def _parse_shape(value: str) -> tuple[int, int]:
    normalized = value.lower().replace("x", ",").replace(" ", "")
    parts = [part for part in normalized.split(",") if part]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("shape must be ROWS,COLS or ROWSxCOLS")
    rows, cols = int(parts[0]), int(parts[1])
    if rows <= 0 or cols <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return rows, cols


def _weighted_choice(rng: np.random.Generator, weights: dict[str, int | float]) -> str:
    names = list(weights)
    values = np.asarray([float(weights[name]) for name in names], dtype=np.float64)
    if len(names) == 0 or np.any(values < 0) or float(values.sum()) <= 0:
        raise ValueError("distribution weights must be non-negative and have positive sum")
    return str(rng.choice(names, p=values / values.sum()))


def _uniform_range(rng: np.random.Generator, value: Any, name: str) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        if len(values) != 2:
            raise ValueError(f"{name} range must have exactly two values")
        low, high = float(values[0]), float(values[1])
        if high < low:
            raise ValueError(f"{name} range must satisfy high >= low")
        return float(rng.uniform(low, high))
    if isinstance(value, Mapping):
        dist = str(value.get("dist", "uniform"))
        if dist == "constant":
            return float(value["value"])
        if dist == "uniform":
            low, high = float(value["low"]), float(value["high"])
            if high < low:
                raise ValueError(f"{name} uniform distribution requires high >= low")
            return float(rng.uniform(low, high))
        if dist == "lognormal":
            mean = float(value["mean"])
            sigma_factor = float(value.get("sigma_factor", value.get("sigma", 0.25)))
            if mean <= 0:
                raise ValueError(f"{name} lognormal mean must be > 0")
            if sigma_factor < 0:
                raise ValueError(f"{name} lognormal sigma_factor must be >= 0")
            return float(mean * np.exp(rng.normal(0.0, sigma_factor)))
        if dist == "choice":
            choices = list(value["values"])
            if not choices:
                raise ValueError(f"{name} choice distribution requires non-empty values")
            weights = value.get("weights")
            if weights is None:
                return float(rng.choice(choices))
            probs = np.asarray(list(weights), dtype=np.float64)
            if probs.size != len(choices) or np.any(probs < 0) or float(probs.sum()) <= 0:
                raise ValueError(f"{name} choice weights must match values and have positive sum")
            return float(rng.choice(choices, p=probs / probs.sum()))
        raise ValueError(f"unsupported distribution for {name}: {dist}")
    raise ValueError(f"{name} must be a scalar, [low, high] range, or distribution mapping")


def _range_pair(value: Any, name: str) -> tuple[float, float]:
    if isinstance(value, Mapping):
        if str(value.get("dist", "uniform")) != "uniform":
            raise ValueError(f"{name} must be a [low, high] range or uniform distribution mapping")
        low, high = float(value["low"]), float(value["high"])
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        values = list(value)
        if len(values) != 2:
            raise ValueError(f"{name} range must have exactly two values")
        low, high = float(values[0]), float(values[1])
    else:
        raise ValueError(f"{name} must be a [low, high] range")
    if low < 0 or high < low:
        raise ValueError(f"{name} must satisfy 0 <= low <= high")
    return low, high


def _require_storage_contract(config: dict[str, Any]) -> None:
    storage = dict(config.get("storage", {}))
    if storage.get("format") != "compact":
        raise ValueError("training pool storage.format must be 'compact'")
    if storage.get("save_hr_temperature", False):
        raise ValueError("training pool config must not save HR temperature")


def _validate_config(config: dict[str, Any]) -> None:
    _require_storage_contract(config)
    scale = int(config.get("scale", 0))
    lr_shape = tuple(map(int, config.get("lr_shape", [])))
    hr_shape = tuple(map(int, config.get("hr_shape", [])))
    n_frames = int(config.get("n_frames_per_scene", 0))
    if scale not in SUPPORTED_SCALES:
        raise ValueError(f"training pool config scale must be one of {sorted(SUPPORTED_SCALES)}, got {scale}")
    if len(lr_shape) != 2 or len(hr_shape) != 2:
        raise ValueError("lr_shape and hr_shape must be [rows, cols]")
    if hr_shape != (lr_shape[0] * scale, lr_shape[1] * scale):
        raise ValueError("hr_shape must equal lr_shape * scale")
    if n_frames <= 0:
        raise ValueError("n_frames_per_scene must be > 0")
    required_ranges = {
        "T_bg_c",
        "psf_sigma_lr_px",
        "noise_sigma_c",
        "low_freq_amplitude_c",
        "delta_T_c_by_difficulty",
        "rotation_deg_center",
        "rotation_jitter_deg",
    }
    missing = sorted(required_ranges - set(config.get("physics_ranges", {})))
    if missing:
        raise ValueError(f"physics_ranges missing required keys: {missing}")
    if len(config.get("obs_features_channels", [])) != 5:
        raise ValueError("obs_features_channels must list the five TCForge channels")


def _make_scene_plans(config: dict[str, Any], num_scenes: int, master_seed: int) -> list[ScenePlan]:
    rng = np.random.default_rng(master_seed)
    difficulty_distribution = dict(config["difficulty_distribution"])
    drift_distribution = dict(config["drift_distribution"])
    plans: list[ScenePlan] = []
    used_seeds: set[int] = set()
    for index in range(num_scenes):
        seed = int(rng.integers(1, np.iinfo(np.int32).max))
        while seed in used_seeds:
            seed = int(rng.integers(1, np.iinfo(np.int32).max))
        used_seeds.add(seed)
        plans.append(
            ScenePlan(
                scene_index=index,
                scene_id=f"scene_{index:04d}",
                seed=seed,
                difficulty=_weighted_choice(rng, difficulty_distribution),
                drift_model=_weighted_choice(rng, drift_distribution),
            )
        )
    return plans


def _load_or_fallback_shifts(config: dict[str, Any], *, n_frames: int, scale: int) -> tuple[np.ndarray, dict[str, Any]]:
    profile = str(config.get("shift_profile", "real_default_contour_refined"))
    try:
        shifts, metadata = load_shift_profile(profile, n_frames=n_frames, scale=scale)
        metadata = dict(metadata)
        metadata["fallback_used"] = False
        return shifts.astype(np.float32, copy=False), metadata
    except (FileNotFoundError, ValueError) as exc:
        fallback = str(config.get("shift_fallback_profile", "ideal_phase_grid"))
        if fallback == profile:
            raise
        shifts, metadata = load_shift_profile(fallback, n_frames=n_frames, scale=scale)
        metadata = dict(metadata)
        metadata.update(
            {
                "fallback_used": True,
                "requested_profile": profile,
                "fallback_profile": fallback,
                "fallback_reason": str(exc),
            }
        )
        return shifts.astype(np.float32, copy=False), metadata


def _generate_one_scene(plan: ScenePlan, config: dict[str, Any], output_dir: str) -> dict[str, Any]:
    rng = np.random.default_rng(plan.seed)
    physics = config["physics_ranges"]
    scale = int(config["scale"])
    lr_shape = tuple(map(int, config["lr_shape"]))
    hr_shape = tuple(map(int, config["hr_shape"]))
    n_frames = int(config["n_frames_per_scene"])

    t_bg_c = _uniform_range(rng, physics["T_bg_c"], "T_bg_c")
    delta_t_c = _uniform_range(
        rng,
        physics["delta_T_c_by_difficulty"][plan.difficulty],
        f"delta_T_c_by_difficulty.{plan.difficulty}",
    )
    psf_cfg = dict(config.get("psf_randomization", {}))
    if psf_cfg.get("enabled", False):
        psf_params = sample_psf_parameters(
            rng=rng,
            sigma_range=_range_pair(psf_cfg.get("sigma_range", physics["psf_sigma_lr_px"]), "psf_randomization.sigma_range"),
            elliptical_probability=float(psf_cfg.get("elliptical_probability", 0.30)),
            airy_probability=float(psf_cfg.get("airy_probability", 0.10)),
        )
        psf_sigma_lr_px = float(psf_params["psf_sigma_lr_px"])
        psf_shape = str(psf_params["psf_shape"])
        psf_sigma_y_lr_px = (
            None if psf_params.get("psf_sigma_y_lr_px") is None else float(psf_params["psf_sigma_y_lr_px"])
        )
        psf_angle_deg = float(psf_params.get("psf_angle_deg", 0.0))
    else:
        psf_sigma_lr_px = _uniform_range(rng, physics["psf_sigma_lr_px"], "psf_sigma_lr_px")
        psf_shape = "gaussian"
        psf_sigma_y_lr_px = None
        psf_angle_deg = 0.0
    noise_sigma_c = _uniform_range(rng, physics["noise_sigma_c"], "noise_sigma_c")
    low_freq_amplitude_c = _uniform_range(rng, physics["low_freq_amplitude_c"], "low_freq_amplitude_c")
    low_freq_sigma_px = float(physics.get("low_freq_sigma_px", 96.0))
    low_freq_seed = int(rng.integers(1, np.iinfo(np.int32).max))
    rotation_center = float(physics["rotation_deg_center"])
    rotation_jitter = float(physics["rotation_jitter_deg"])
    geometry_cfg = dict(config.get("geometry", {}))
    antialias = bool(geometry_cfg.get("antialias", True))
    ssaa_factor = int(geometry_cfg.get("ssaa_factor", 4))

    hr_mask, geo_meta = build_scene_mask_with_metadata(
        plan.difficulty,
        plan.seed,
        rotation_deg_center=rotation_center,
        rotation_jitter_deg=rotation_jitter,
        canvas_shape=hr_shape,
        pixel_size_um=float(config["pixel_size_um"]),
        scale=scale,
        antialias=antialias,
        ssaa_factor=ssaa_factor,
    )
    hr_temperature = render_temperature_field(
        hr_mask,
        t_bg_c=t_bg_c,
        delta_t_c=delta_t_c,
        low_freq_amplitude_c=low_freq_amplitude_c,
        low_freq_sigma_px=low_freq_sigma_px,
        seed=low_freq_seed,
    )
    hr_edge = edge_map(hr_mask >= 0.5, edge_width_px=2)

    shifts, shift_meta = _load_or_fallback_shifts(config, n_frames=n_frames, scale=scale)
    shift_jitter_std_px = float(config.get("shift_jitter_std_px", 0.0))
    if shift_jitter_std_px < 0:
        raise ValueError("shift_jitter_std_px must be >= 0")
    if shift_jitter_std_px > 0:
        shifts = (
            shifts + rng.normal(0.0, shift_jitter_std_px, size=shifts.shape).astype(np.float32)
        ).astype(np.float32, copy=False)

    lr_burst = generate_lr_burst(
        hr_temperature,
        shifts,
        forward_mode=config["forward_mode"],
        psf_sigma_lr_px=psf_sigma_lr_px,
        psf_shape=psf_shape,
        psf_sigma_y_lr_px=psf_sigma_y_lr_px,
        psf_angle_deg=psf_angle_deg,
        scale=scale,
        workers=1,  # Intentional: outer loop already uses ProcessPoolExecutor
    )
    if lr_burst.shape != (n_frames, *lr_shape):
        raise RuntimeError(f"unexpected LR burst shape: {lr_burst.shape}")
    noise_cfg = dict(config.get("noise_model", {}))
    noise_model = str(noise_cfg.get("model", "iid_gaussian"))
    noise_params_resolved = {
        "model": noise_model,
        "fpn_sigma_px": _uniform_range(rng, noise_cfg.get("fpn_sigma_px", 5.0), "noise_model.fpn_sigma_px"),
        "stripe_sigma_c": (
            None
            if noise_cfg.get("stripe_sigma_c") is None
            else _uniform_range(rng, noise_cfg.get("stripe_sigma_c"), "noise_model.stripe_sigma_c")
        ),
    }
    lr_burst = add_noise(
        lr_burst,
        noise_sigma_c=noise_sigma_c,
        seed=plan.seed + 1000,
        noise_model=noise_model,
        fpn_sigma_px=float(noise_params_resolved["fpn_sigma_px"]),
        stripe_sigma_c=(
            None
            if noise_params_resolved["stripe_sigma_c"] is None
            else float(noise_params_resolved["stripe_sigma_c"])
        ),
    )

    drift_params = dict(config.get("drift_parameters", {}))
    drift_params_resolved = {
        "amplitude_c": _uniform_range(rng, drift_params.get("amplitude_c", 0.2), "drift_parameters.amplitude_c"),
        "lowfreq_sigma_px": _uniform_range(
            rng,
            drift_params.get("lowfreq_sigma_px", 96.0),
            "drift_parameters.lowfreq_sigma_px",
        ),
        "gain_sigma": _uniform_range(rng, drift_params.get("gain_sigma", 0.01), "drift_parameters.gain_sigma"),
        "offset_sigma_c": _uniform_range(
            rng,
            drift_params.get("offset_sigma_c", 0.1),
            "drift_parameters.offset_sigma_c",
        ),
        "temporal_spatial_sigma_px": _uniform_range(
            rng,
            drift_params.get("temporal_spatial_sigma_px", 0.0),
            "drift_parameters.temporal_spatial_sigma_px",
        ),
    }
    lr_burst = apply_drift(
        lr_burst,
        model=plan.drift_model,
        seed=plan.seed + 2000,
        amplitude_c=drift_params_resolved["amplitude_c"],
        lowfreq_sigma_px=drift_params_resolved["lowfreq_sigma_px"],
        gain_sigma=drift_params_resolved["gain_sigma"],
        offset_sigma_c=drift_params_resolved["offset_sigma_c"],
        temporal_spatial_sigma_px=drift_params_resolved["temporal_spatial_sigma_px"],
    )
    defect_cfg = dict(config.get("detector_defects", {}))
    defect_params_resolved: dict[str, Any] = {"enabled": bool(defect_cfg.get("enabled", False))}
    if defect_cfg.get("enabled", False):
        defect_params_resolved.update(
            {
                "defect_rate": _uniform_range(
                    rng,
                    defect_cfg.get("defect_rate", 0.001),
                    "detector_defects.defect_rate",
                ),
                "mode": str(defect_cfg.get("mode", "offset")),
                "hot_delta_c": _uniform_range(
                    rng,
                    defect_cfg.get("hot_delta_c", 0.5),
                    "detector_defects.hot_delta_c",
                ),
                "cold_delta_c": _uniform_range(
                    rng,
                    defect_cfg.get("cold_delta_c", -0.5),
                    "detector_defects.cold_delta_c",
                ),
            }
        )
        lr_burst = add_detector_defects(
            lr_burst,
            defect_rate=float(defect_params_resolved["defect_rate"]),
            seed=plan.seed + 3000,
            mode=str(defect_params_resolved["mode"]),
            hot_delta_c=float(defect_params_resolved["hot_delta_c"]),
            cold_delta_c=float(defect_params_resolved["cold_delta_c"]),
        )
    obs_features = fuse_burst_to_features(lr_burst, shifts, output_shape=None, sigma_bg=5.0)
    if obs_features.shape != (5, *lr_shape):
        raise RuntimeError(f"unexpected obs_features shape: {obs_features.shape}")

    # Classical shift-and-add SR (optional, controlled by config)
    classical_sr_image = None
    if config.get("compute_classical_sr", False):
        classical_sr_image = shift_and_add(lr_burst, shifts, scale=scale, output_shape=hr_shape)

    metadata = {
        "scene_id": plan.scene_id,
        "scene_index": int(plan.scene_index),
        "difficulty": plan.difficulty,
        "seed": int(plan.seed),
        "dataset": config["dataset"],
        "engine": config["engine"],
        "version": config["version"],
        "scale": scale,
        "lr_shape": list(lr_shape),
        "hr_shape": list(hr_shape),
        "pixel_size_um": float(config["pixel_size_um"]),
        "spatial_resolution_um": float(config["spatial_resolution_um"]),
        "T_bg_c": float(t_bg_c),
        "delta_T_c": float(delta_t_c),
        "low_freq_amplitude_c": float(low_freq_amplitude_c),
        "low_freq_sigma_px": float(low_freq_sigma_px),
        "low_freq_seed": int(low_freq_seed),
        "psf_sigma_lr_px": float(psf_sigma_lr_px),
        "psf_shape": psf_shape,
        "psf_sigma_y_lr_px": None if psf_sigma_y_lr_px is None else float(psf_sigma_y_lr_px),
        "psf_angle_deg": float(psf_angle_deg),
        "noise_sigma_c": float(noise_sigma_c),
        "noise_model": noise_model,
        "noise_parameters": noise_params_resolved,
        "drift_model": plan.drift_model,
        "drift_parameters": drift_params_resolved,
        "detector_defects": defect_params_resolved,
        "rotation_deg": float(geo_meta["rotation_deg"]),
        "geometry_metadata": geo_meta,
        "n_frames": n_frames,
        "forward_mode": config["forward_mode"],
        "shift_profile": config.get("shift_profile", "real_default_contour_refined"),
        "shift_metadata": shift_meta,
        "shift_jitter_std_px": float(shift_jitter_std_px),
        "shift_convention": shift_meta.get("convention", "LR-to-reference alignment shift"),
        "obs_features_channels": list(config["obs_features_channels"]),
        "obs_features_resolution": config.get("storage", {}).get("obs_features_resolution", "1x"),
        "has_classical_sr": classical_sr_image is not None,
        "storage": {
            "format": "compact",
            "save_lr_burst": bool(config.get("storage", {}).get("save_lr_burst", False)),
            "save_hr_temperature": False,
            "hr_mask_semantics": "coverage" if antialias else "binary",
            "hr_mask_quantization": "uint8_png_0_255",
        },
    }

    scene_dir = Path(output_dir) / plan.scene_id
    save_scene_compact(
        scene_dir,
        hr_mask=hr_mask,
        hr_edge=hr_edge,
        obs_features=obs_features,
        shifts=shifts,
        metadata=metadata,
        classical_sr=classical_sr_image,
        lr_burst=(
            lr_burst
            if bool(config.get("storage", {}).get("save_lr_burst", False))
            else None
        ),
    )
    return {
        "scene_id": plan.scene_id,
        "scene_dir": plan.scene_id,
        "difficulty": plan.difficulty,
        "seed": int(plan.seed),
        "scale": scale,
        "lr_shape": f"{lr_shape[0]}x{lr_shape[1]}",
        "hr_shape": f"{hr_shape[0]}x{hr_shape[1]}",
        "T_bg_c": float(t_bg_c),
        "delta_T_c": float(delta_t_c),
        "psf_sigma_lr_px": float(psf_sigma_lr_px),
        "noise_sigma_c": float(noise_sigma_c),
        "drift_model": plan.drift_model,
        "rotation_deg": float(geo_meta["rotation_deg"]),
        "n_frames": n_frames,
    }


def _write_manifest(output_dir: Path, rows: list[dict[str, Any]]) -> Path:
    manifest_path = output_dir / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in sorted(rows, key=lambda item: str(item["scene_id"])):
            writer.writerow({field: row[field] for field in MANIFEST_FIELDS})
    return manifest_path


def _shape_to_manifest_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return f"{int(value[0])}x{int(value[1])}"
    raise ValueError(f"shape must be a 2-value sequence, got: {value!r}")


def _manifest_row_from_metadata(scene_dir: Path) -> dict[str, Any]:
    metadata_path = scene_dir / "metadata.json"
    metadata = _load_json(metadata_path)
    return {
        "scene_id": metadata["scene_id"],
        "scene_dir": scene_dir.name,
        "difficulty": metadata["difficulty"],
        "seed": int(metadata["seed"]),
        "scale": int(metadata["scale"]),
        "lr_shape": _shape_to_manifest_value(metadata["lr_shape"]),
        "hr_shape": _shape_to_manifest_value(metadata["hr_shape"]),
        "T_bg_c": float(metadata["T_bg_c"]),
        "delta_T_c": float(metadata["delta_T_c"]),
        "psf_sigma_lr_px": float(metadata["psf_sigma_lr_px"]),
        "noise_sigma_c": float(metadata["noise_sigma_c"]),
        "drift_model": metadata["drift_model"],
        "rotation_deg": float(metadata["rotation_deg"]),
        "n_frames": int(metadata["n_frames"]),
    }


def _estimate_memory_per_worker(config: dict[str, Any]) -> float:
    """Estimate peak memory per worker in GB (conservative upper bound)."""
    lr = config.get("lr_shape", [480, 640])
    hr = config.get("hr_shape", [1920, 2560])
    n_frames = int(config.get("n_frames_per_scene", 248))
    geometry_cfg = dict(config.get("geometry", {}))
    antialias = bool(geometry_cfg.get("antialias", True))
    ssaa_factor = int(geometry_cfg.get("ssaa_factor", 4)) if antialias else 1
    hr_bytes = hr[0] * hr[1] * 4  # float32
    ssaa_pixels = hr[0] * ssaa_factor * hr[1] * ssaa_factor
    # Incremental geometry keeps one uint8 draw canvas, one uint8 subtract
    # canvas, and one float32 rotated coverage canvas live near the peak.
    ssaa_bytes = ssaa_pixels * (1 + 1 + 4)
    lr_burst_bytes = n_frames * lr[0] * lr[1] * 4
    # aligned stack (×2 for highpass path) + overhead
    fusion_bytes = 2 * n_frames * lr[0] * lr[1] * 4
    total = hr_bytes + ssaa_bytes + lr_burst_bytes + fusion_bytes
    return total / (1024 ** 3)


def _generate_pool(config: dict[str, Any], output_dir: Path, *, num_scenes: int, workers: int, seed: int) -> Path:
    _validate_config(config)
    output_dir.mkdir(parents=True, exist_ok=True)
    plans = _make_scene_plans(config, num_scenes, seed)

    worker_count = max(1, int(workers))

    # P4: Print memory estimation so users can gauge worker count
    mem_per_worker_gb = _estimate_memory_per_worker(config)
    total_mem_gb = mem_per_worker_gb * worker_count
    print(f"Memory estimate: ~{mem_per_worker_gb:.1f} GB/worker × {worker_count} workers = ~{total_mem_gb:.1f} GB peak")

    rows: list[dict[str, Any]] = []

    # P2: Checkpoint/resume — skip scenes that already have a metadata.json
    remaining_plans: list[ScenePlan] = []
    skipped = 0
    for plan in plans:
        scene_dir = output_dir / plan.scene_id
        if (scene_dir / "metadata.json").exists():
            skipped += 1
            rows.append(_manifest_row_from_metadata(scene_dir))
        else:
            remaining_plans.append(plan)
    if skipped:
        print(f"Resuming: {skipped} scenes already exist, generating {len(remaining_plans)} remaining")

    failed: list[tuple[str, str]] = []

    if worker_count == 1:
        for plan in tqdm(remaining_plans, desc="Generating scenes"):
            try:
                rows.append(_generate_one_scene(plan, config, str(output_dir)))
            except Exception:
                tb = traceback.format_exc()
                logger.error("Scene %s failed:\n%s", plan.scene_id, tb)
                failed.append((plan.scene_id, tb))
    else:
        with ProcessPoolExecutor(max_workers=worker_count) as executor:
            future_to_plan = {
                executor.submit(_generate_one_scene, plan, config, str(output_dir)): plan
                for plan in remaining_plans
            }
            for future in tqdm(as_completed(future_to_plan), total=len(future_to_plan), desc="Generating scenes"):
                plan = future_to_plan[future]
                try:
                    rows.append(future.result())
                except Exception:
                    tb = traceback.format_exc()
                    logger.error("Scene %s failed:\n%s", plan.scene_id, tb)
                    failed.append((plan.scene_id, tb))

    if failed:
        print(f"\n⚠️  {len(failed)} scene(s) failed:")
        for scene_id, _ in failed:
            print(f"  - {scene_id}")
        fail_log = output_dir / "failed_scenes.log"
        with fail_log.open("w", encoding="utf-8") as f:
            for scene_id, tb in failed:
                f.write(f"=== {scene_id} ===\n{tb}\n")
        print(f"  Details: {fail_log}")

    return _write_manifest(output_dir, rows)


def _resolve_num_scenes(args: argparse.Namespace, config: dict[str, Any]) -> int:
    if args.pool_size is not None and args.num_scenes is not None:
        raise ValueError("Use only one of --pool-size and --num-scenes")
    if args.pool_size is not None:
        return int(args.pool_size)
    if args.num_scenes is not None:
        return int(args.num_scenes)
    return int(config["num_scenes"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        epilog=(
            "Examples:\n"
            "  uv run python scripts/generate_training_pool.py \\\n"
            "    --config configs/synthetic/training_pool_2x.json \\\n"
            "    --output-dir data/synthetic/training_pool_2x_aa \\\n"
            "    --pool-size 1000\n"
            "  uv run python scripts/generate_training_pool.py \\\n"
            "    --config configs/synthetic/training_pool_2x.json \\\n"
            "    --output-dir data/synthetic/training_pool_2x_aa \\\n"
            "    --pool-size 2000 --workers 4"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Training pool JSON config.")
    parser.add_argument(
        "--pool-size",
        type=int,
        choices=(1000, 2000),
        default=None,
        help="Preset scene count for full training pools (1000 default in config, or 2000).",
    )
    parser.add_argument(
        "--num-scenes",
        type=int,
        default=None,
        help="Override scene count with any positive integer (e.g. 5 for smoke tests). "
        "Do not combine with --pool-size.",
    )
    parser.add_argument("--output-dir", type=Path, default=None, help="Override config output_dir.")
    parser.add_argument("--workers", type=int, default=1, help="Scene-level worker processes; default is conservative.")
    parser.add_argument("--seed", type=int, default=None, help="Override config master seed.")
    parser.add_argument("--lr-shape", type=_parse_shape, default=None, help="Override LR shape as ROWS,COLS.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = _resolve_path(args.config)
    config = _load_json(config_path)
    if args.lr_shape is not None:
        config["lr_shape"] = [int(args.lr_shape[0]), int(args.lr_shape[1])]
        scale = int(config["scale"])
        config["hr_shape"] = [int(args.lr_shape[0] * scale), int(args.lr_shape[1] * scale)]
    num_scenes = _resolve_num_scenes(args, config)
    if num_scenes <= 0:
        raise ValueError("num_scenes must be > 0")
    output_dir = _resolve_path(args.output_dir if args.output_dir is not None else config["output_dir"])
    seed = int(args.seed if args.seed is not None else config.get("seed", 700700))
    manifest_path = _generate_pool(config, output_dir, num_scenes=num_scenes, workers=args.workers, seed=seed)
    print(f"Generated {num_scenes} compact scenes")
    print(f"Output: {output_dir}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
