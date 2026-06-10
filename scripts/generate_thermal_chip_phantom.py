#!/usr/bin/env python3
"""Generate ThermalChipPhantom synthetic scenes.

The script is intentionally a thin CLI around TCForge. Small local fallback
paths are kept only to make error messages and notebook demos robust while the
package source is being edited in-place.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[1]
TCFORGE_SRC = PROJECT_ROOT / "tcforge" / "src"
if TCFORGE_SRC.exists() and str(TCFORGE_SRC) not in sys.path:
    sys.path.insert(0, str(TCFORGE_SRC))

DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "synthetic" / "phantom_smoke.json"
DEFAULT_SHIFT_CONFIG = PROJECT_ROOT / "configs" / "synthetic" / "shift_profiles.json"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "data" / "synthetic" / "thermal_chip_phantom"
REQUIRED_SCENE_FILES = (
    "hr_temperature_2x.npy",
    "hr_mask_2x.npy",
    "hr_edge_map_2x.npy",
    "lr_burst_raw.npy",
    "lr_burst_highpass.npy",
    "shifts.npy",
    "metadata.json",
)


def _allow_fallback(config: dict[str, Any]) -> bool:
    return bool(config.get("allow_fallback_demo", False))


def _load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_psf_settings(config: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    profile = str(config.get("psf_profile", "explicit"))
    if profile == "ep09_provisional":
        psf_config_path = PROJECT_ROOT / str(config.get("psf_calibration_path", "configs/psf_calibration.json"))
        calibration = _load_json(psf_config_path)
        sigma = float(calibration["psf_sigma_lr_px"])
        return sigma, {
            "psf_profile": profile,
            "psf_profile_source": str(psf_config_path.relative_to(PROJECT_ROOT)),
            "psf_calibration_episode": calibration.get("calibration_episode", ""),
            "psf_calibration_status": calibration.get("status", ""),
            "psf_confidence_interval_95_lr_px": calibration.get("confidence_interval_95_lr_px", []),
            "psf_four_x_verdict": calibration.get("four_x_verdict", ""),
        }
    profiles = config.get("psf_profiles", {})
    if profile in profiles:
        entry = dict(profiles[profile])
        return float(entry["psf_sigma_lr_px"]), {
            "psf_profile": profile,
            "psf_profile_source": entry.get("source", ""),
            "psf_calibration_status": entry.get("status", ""),
        }
    return float(config["psf_sigma_lr_px"]), {
        "psf_profile": profile,
        "psf_profile_source": "config.psf_sigma_lr_px",
        "psf_calibration_status": "explicit_or_legacy",
    }


def _write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return "unknown"
    return result.stdout.strip()


def _parse_shape(value: str) -> tuple[int, int]:
    normalized = value.lower().replace("x", ",").replace(" ", "")
    parts = [p for p in normalized.split(",") if p]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("shape must be ROWS,COLS or ROWSxCOLS")
    rows, cols = (int(parts[0]), int(parts[1]))
    if rows <= 0 or cols <= 0:
        raise argparse.ArgumentTypeError("shape dimensions must be positive")
    return rows, cols


def _repeat_or_trim(shifts: np.ndarray, n_frames: int) -> np.ndarray:
    if shifts.ndim != 2 or shifts.shape[1] != 2:
        raise ValueError("shifts must have shape (N, 2)")
    if len(shifts) == 0:
        raise ValueError("shift profile produced no shifts")
    repeats = int(np.ceil(n_frames / len(shifts)))
    return np.tile(shifts, (repeats, 1))[:n_frames].astype(np.float32, copy=False)


def _boolish(value: str) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y", "t"}


def _read_csv_shift_profile(profile: dict[str, Any], n_frames: int) -> tuple[np.ndarray, dict[str, Any]]:
    source = PROJECT_ROOT / profile["source_path"]
    if not source.exists():
        raise FileNotFoundError(f"shift profile source not found: {source}")
    rows: list[dict[str, str]] = []
    with source.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            success_col = profile.get("filter_success_column")
            if success_col and success_col in row and not _boolish(row[success_col]):
                continue
            rows.append(row)
    for key in reversed(profile.get("sort_by", [])):
        rows.sort(key=lambda r: float(r.get(key, "0") or 0.0))
    dx_col, dy_col = profile["columns"]
    arr = np.asarray([[float(r[dx_col]), float(r[dy_col])] for r in rows], dtype=np.float32)
    info = {
        "source_path": profile["source_path"],
        "source_sha256": _sha256(source),
        "columns": [dx_col, dy_col],
    }
    return _repeat_or_trim(arr, n_frames), info


def _stage_grid(profile: dict[str, Any]) -> np.ndarray:
    theta = np.deg2rad(float(profile["theta_deg"]))
    pitch = float(profile["pixel_size_um"])
    shifts = []
    for y_um in profile["y_um_values"]:
        for x_um in profile["x_um_values"]:
            dx = (x_um * np.cos(theta) + y_um * np.sin(theta)) / pitch
            dy = (-x_um * np.sin(theta) + y_um * np.cos(theta)) / pitch
            shifts.append((dx, dy))
    return np.asarray(shifts, dtype=np.float32)


def _load_shifts(name: str, n_frames: int, shift_config_path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    shift_config = _load_json(shift_config_path)
    profiles = shift_config["profiles"]
    if name not in profiles:
        raise KeyError(f"unknown shift profile {name!r}; available: {sorted(profiles)}")
    profile = profiles[name]
    kind = profile["kind"]
    if kind == "csv_columns":
        shifts, info = _read_csv_shift_profile(profile, n_frames)
    elif kind == "phase_grid":
        base = np.asarray([(dx, dy) for dy in profile["dy_values"] for dx in profile["dx_values"]], dtype=np.float32)
        shifts, info = _repeat_or_trim(base, n_frames), {"source_path": None, "source_sha256": None, "columns": None}
    elif kind == "stage_coordinate_grid":
        shifts, info = _repeat_or_trim(_stage_grid(profile), n_frames), {"source_path": None, "source_sha256": None, "columns": None}
    elif kind == "jittered_profile":
        base, info = _load_shifts(profile["base_profile"], n_frames, shift_config_path)
        rng = np.random.default_rng(int(profile.get("seed", 0)))
        shifts = base + rng.normal(0.0, float(profile["jitter_sigma_px"]), size=base.shape).astype(np.float32)
    else:
        raise ValueError(f"unsupported shift profile kind: {kind}")
    info.update(
        {
            "profile": name,
            "units": shift_config.get("units", "LR pixels"),
            "convention": shift_config.get("convention", "LR-to-reference alignment shift"),
        }
    )
    return shifts.astype(np.float32, copy=False), info


def _rect(mask: np.ndarray, cy: float, cx: float, h: float, w: float, value: int = 1) -> None:
    r0 = max(0, int(round(cy - h / 2)))
    r1 = min(mask.shape[0], int(round(cy + h / 2)))
    c0 = max(0, int(round(cx - w / 2)))
    c1 = min(mask.shape[1], int(round(cx + w / 2)))
    if r1 > r0 and c1 > c0:
        mask[r0:r1, c0:c1] = value


def _fallback_scene_mask(
    difficulty: str,
    seed: int,
    *,
    hr_shape: tuple[int, int],
    pixel_size_um: float,
    scale: int,
    rotation_deg_center: float,
    rotation_jitter_deg: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    rng = np.random.default_rng(seed)
    hr_pitch_um = pixel_size_um / scale
    rows, cols = hr_shape
    cy, cx = rows / 2.0, cols / 2.0
    mask = np.zeros(hr_shape, dtype=np.uint8)
    size = {
        "easy": (360.0, 520.0, 70.0),
        "medium": (280.0, 420.0, 35.0),
        "hard": (230.0, 340.0, 22.0),
        "stress": (180.0, 260.0, 14.0),
    }.get(difficulty, (280.0, 420.0, 35.0))
    outer_h, outer_w, feature_um = size
    _rect(mask, cy, cx, outer_h / hr_pitch_um, outer_w / hr_pitch_um, 1)
    _rect(mask, cy, cx, (outer_h - 2 * feature_um) / hr_pitch_um, (outer_w - 2 * feature_um) / hr_pitch_um, 0)
    n_pins = 7 if difficulty in {"easy", "medium"} else 11
    spacing = (outer_w / (n_pins + 1)) / hr_pitch_um
    pin_w = max(1.0, feature_um / hr_pitch_um)
    pin_l = (feature_um * 2.3) / hr_pitch_um
    for idx in range(n_pins):
        x = cx - outer_w / (2 * hr_pitch_um) + spacing * (idx + 1)
        _rect(mask, cy - outer_h / (2 * hr_pitch_um) - pin_l / 2, x, pin_l, pin_w, 1)
        _rect(mask, cy + outer_h / (2 * hr_pitch_um) + pin_l / 2, x, pin_l, pin_w, 1)
    if difficulty in {"medium", "hard", "stress"}:
        trench_w = max(1.0, feature_um / (1.5 * hr_pitch_um))
        for off in (-2, -1, 0, 1, 2):
            _rect(mask, cy + off * 2.5 * trench_w, cx, trench_w, outer_w / (1.8 * hr_pitch_um), 0)
    rotation = rotation_deg_center + float(rng.uniform(-rotation_jitter_deg, rotation_jitter_deg))
    rotated = ndimage.rotate(mask, rotation, reshape=False, order=0, mode="constant", cval=0.0)
    geometry = {
        "units": "um",
        "rotation_deg": rotation,
        "min_feature_um": feature_um,
        "primitives": ["frame", "pin_array", "trenches"] if difficulty != "easy" else ["frame", "pin_array"],
        "implementation": "script_fallback",
    }
    return (rotated > 0.5).astype(np.uint8), geometry


def _build_scene_mask(config: dict[str, Any], difficulty: str, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    scale = int(config["scale"])
    lr_shape = tuple(config["lr_shape"])
    hr_shape = tuple(config.get("hr_shape", [lr_shape[0] * scale, lr_shape[1] * scale]))
    try:
        from tcforge.geometry import build_scene_mask_with_metadata  # type: ignore

        mask, geometry = build_scene_mask_with_metadata(
            difficulty,
            seed,
            rotation_deg_center=float(config["rotation_deg_center"]),
            rotation_jitter_deg=float(config["rotation_jitter_deg"]),
            canvas_shape=hr_shape,
            pixel_size_um=float(config["pixel_size_um"]),
            scale=scale,
        )
        geometry["min_feature_um"] = config["min_feature_um_by_difficulty"][difficulty]
        return np.asarray(mask, dtype=np.uint8), geometry
    except Exception as exc:
        if not _allow_fallback(config):
            raise RuntimeError("TCForge geometry generation failed; pass --allow-fallback-demo only for notebook demos") from exc
        return _fallback_scene_mask(
            difficulty,
            seed,
            hr_shape=hr_shape,
            pixel_size_um=float(config["pixel_size_um"]),
            scale=scale,
            rotation_deg_center=float(config["rotation_deg_center"]),
            rotation_jitter_deg=float(config["rotation_jitter_deg"]),
        )


def _fallback_edge_map(mask: np.ndarray) -> np.ndarray:
    eroded = ndimage.binary_erosion(mask > 0, structure=np.ones((3, 3), dtype=bool), border_value=0)
    dilated = ndimage.binary_dilation(mask > 0, structure=np.ones((3, 3), dtype=bool), border_value=0)
    return (dilated ^ eroded).astype(np.float32)


def _edge_map(mask: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    try:
        from tcforge.physics import edge_map

        return np.asarray(edge_map(mask), dtype=np.float32)
    except Exception as exc:
        if not _allow_fallback(config):
            raise RuntimeError("TCForge edge_map failed; pass --allow-fallback-demo only for notebook demos") from exc
        return _fallback_edge_map(mask)


def _temperature(mask: np.ndarray, config: dict[str, Any], difficulty: str, seed: int) -> np.ndarray:
    try:
        from tcforge.physics import render_temperature_field

        return np.asarray(
            render_temperature_field(
                mask,
                t_bg_c=float(config["T_bg_c"]),
                delta_t_c=float(config["delta_T_c_by_difficulty"][difficulty]),
                low_freq_amplitude_c=float(config["low_freq_amplitude_c"]),
                low_freq_sigma_px=max(8.0, min(mask.shape) / 12.0),
                seed=seed + 17,
            ),
            dtype=np.float32,
        )
    except Exception as exc:
        if not _allow_fallback(config):
            raise RuntimeError("TCForge temperature rendering failed; pass --allow-fallback-demo only for notebook demos") from exc
    rng = np.random.default_rng(seed + 17)
    delta_t = float(config["delta_T_c_by_difficulty"][difficulty])
    t_bg = float(config["T_bg_c"])
    amp = float(config["low_freq_amplitude_c"])
    low = rng.normal(size=mask.shape).astype(np.float32)
    sigma = max(8.0, min(mask.shape) / 12.0)
    low = ndimage.gaussian_filter(low, sigma=sigma, mode="nearest")
    low -= float(low.mean())
    max_abs = float(np.max(np.abs(low))) or 1.0
    low = (amp * low / max_abs).astype(np.float32)
    return (t_bg + delta_t * mask.astype(np.float32) + low).astype(np.float32)


def _fallback_forward(x_hr: np.ndarray, shift: np.ndarray, psf_sigma: float, *, scale: int) -> np.ndarray:
    x = np.asarray(x_hr, dtype=np.float64)
    sigma = max(0.0, float(psf_sigma) * scale)
    blurred = ndimage.gaussian_filter(x, sigma=sigma, mode="constant", cval=0.0) if sigma > 0 else x
    h_lr, w_lr = x.shape[0] // scale, x.shape[1] // scale
    dx, dy = np.asarray(shift, dtype=np.float64)
    yy = scale * (np.arange(h_lr, dtype=np.float64) + dy)
    xx = scale * (np.arange(w_lr, dtype=np.float64) + dx)
    coords = np.meshgrid(yy, xx, indexing="ij")
    return ndimage.map_coordinates(blurred, coords, order=1, mode="constant", cval=0.0, prefilter=False)


def _forward_frame(x_hr: np.ndarray, shift: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    psf_sigma, _psf_meta = _resolve_psf_settings(config)
    scale = int(config["scale"])
    try:
        from tcforge._ep06_reference.forward import forward  # type: ignore

        return np.asarray(forward(x_hr, shift, psf_sigma=psf_sigma, scale=scale), dtype=np.float32)
    except Exception:
        try:
            from tcforge.forward import forward  # type: ignore

            return np.asarray(forward(x_hr, shift, psf_sigma=psf_sigma, scale=scale), dtype=np.float32)
        except Exception:
            return _fallback_forward(x_hr, shift, psf_sigma, scale=scale).astype(np.float32)


def _highpass(frames: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    try:
        from tcforge.highpass import highpass_preprocess  # type: ignore

        return highpass_preprocess(
            frames,
            sigma_bg=float(config["highpass_sigma_lr_px"]),
            mode=str(config.get("highpass_mode", "nearest")),
        )
    except Exception as exc:
        if not _allow_fallback(config):
            raise RuntimeError("TCForge highpass preprocessing failed; pass --allow-fallback-demo only for notebook demos") from exc
        arr = np.asarray(frames, dtype=np.float32)
        sigma = float(config["highpass_sigma_lr_px"])
        mode = str(config.get("highpass_mode", "nearest"))
        if arr.ndim == 2:
            return (arr - ndimage.gaussian_filter(arr, sigma=sigma, mode=mode)).astype(np.float32)
        bg = ndimage.gaussian_filter(arr, sigma=(0.0, sigma, sigma), mode=mode)
        return (arr - bg).astype(np.float32, copy=False)


def _apply_drift(frames: np.ndarray, model: str, seed: int) -> np.ndarray:
    if model in {"none", "clean", ""}:
        return frames
    try:
        from tcforge.physics import apply_drift

        model_map = {
            "drift_scalar": "scalar_offset",
            "scalar_offset_random_walk": "scalar_offset",
            "drift_lowfreq": "lowfreq",
            "spatial_lowfreq_gaussian": "lowfreq",
            "drift_gain_offset": "gain_offset",
            "gain_plus_offset": "gain_offset",
        }
        return np.asarray(apply_drift(frames, model=model_map.get(model, model), seed=seed + 31), dtype=np.float32)
    except Exception as exc:
        if not _allow_fallback({"allow_fallback_demo": False}):
            raise RuntimeError("TCForge drift model failed") from exc
    rng = np.random.default_rng(seed + 31)
    out = frames.astype(np.float32, copy=True)
    n = out.shape[0]
    if model in {"drift_scalar", "scalar_offset_random_walk"}:
        walk = np.cumsum(rng.normal(0.0, 0.01, size=n)).astype(np.float32)
        walk -= float(walk.mean())
        max_abs = float(np.max(np.abs(walk))) or 1.0
        out += (0.08 * walk / max_abs)[:, None, None]
    elif model in {"drift_lowfreq", "spatial_lowfreq_gaussian"}:
        noise = rng.normal(size=out.shape).astype(np.float32)
        low = ndimage.gaussian_filter(noise, sigma=(0.0, max(4.0, out.shape[1] / 6), max(4.0, out.shape[2] / 6)), mode="nearest")
        low -= float(low.mean())
        max_abs = float(np.max(np.abs(low))) or 1.0
        out += 0.12 * low / max_abs
    elif model in {"drift_gain_offset", "gain_plus_offset"}:
        gain = 1.0 + rng.normal(0.0, 0.004, size=n).astype(np.float32)
        offset = rng.normal(0.0, 0.04, size=n).astype(np.float32)
        out = out * gain[:, None, None] + offset[:, None, None]
    else:
        raise ValueError(f"unsupported drift_model: {model}")
    return out.astype(np.float32, copy=False)


def _make_burst(hr_temperature: np.ndarray, shifts: np.ndarray, config: dict[str, Any], seed: int) -> tuple[np.ndarray, np.ndarray]:
    forward_mode = str(config.get("forward_mode", config.get("forward_modes", ["exact_ep06_point"])[0]))
    psf_sigma, _psf_meta = _resolve_psf_settings(config)
    try:
        from tcforge.forward import generate_lr_burst

        raw = np.asarray(
            generate_lr_burst(
                hr_temperature,
                shifts,
                forward_mode=forward_mode,
                psf_sigma_lr_px=psf_sigma,
                scale=int(config["scale"]),
            ),
            dtype=np.float32,
        )
    except Exception as exc:
        if not _allow_fallback(config):
            raise RuntimeError(
                f"TCForge LR burst generation failed for forward_mode={forward_mode!r}; "
                "pass --allow-fallback-demo only for notebook demos"
            ) from exc
        raw = np.stack([_forward_frame(hr_temperature, shift, config) for shift in shifts], axis=0).astype(np.float32)

    noise_sigma = float(config["noise_sigma_c"])
    if noise_sigma > 0:
        try:
            from tcforge.physics import add_noise

            raw = np.asarray(
                add_noise(
                    raw,
                    noise_sigma_c=noise_sigma,
                    seed=seed + 23,
                    noise_model=str(config.get("noise_model", "iid_gaussian")),
                    fpn_sigma_px=float(config.get("fpn_sigma_px", 5.0)),
                    stripe_sigma_c=(
                        None
                        if config.get("stripe_sigma_c") is None
                        else float(config.get("stripe_sigma_c"))
                    ),
                ),
                dtype=np.float32,
            )
        except Exception as exc:
            if not _allow_fallback(config):
                raise RuntimeError("TCForge noise injection failed; pass --allow-fallback-demo only for notebook demos") from exc
            rng = np.random.default_rng(seed + 23)
            raw += rng.normal(0.0, noise_sigma, size=raw.shape).astype(np.float32)
    raw = _apply_drift(raw, str(config.get("drift_model", "none")), seed)
    return raw, _highpass(raw, config)


def _scene_id(difficulty: str, index: int) -> str:
    return f"tcp_{difficulty}_{index:04d}"


def _save_scene(
    scene_dir: Path,
    *,
    hr_temperature: np.ndarray,
    mask: np.ndarray,
    edge_map: np.ndarray,
    raw: np.ndarray,
    highpass: np.ndarray,
    shifts: np.ndarray,
    metadata: dict[str, Any],
) -> str:
    scene_dir.mkdir(parents=True, exist_ok=True)
    np.save(scene_dir / "hr_temperature_2x.npy", hr_temperature.astype(np.float32, copy=False))
    np.save(scene_dir / "hr_mask_2x.npy", mask.astype(np.uint8, copy=False))
    np.save(scene_dir / "hr_edge_map_2x.npy", edge_map.astype(np.float32, copy=False))
    np.save(scene_dir / "lr_burst_raw.npy", raw.astype(np.float32, copy=False))
    np.save(scene_dir / "lr_burst_highpass.npy", highpass.astype(np.float32, copy=False))
    np.save(scene_dir / "shifts.npy", shifts.astype(np.float32, copy=False))
    _write_json(scene_dir / "metadata.json", metadata)
    return _sha256(scene_dir / "metadata.json")


def _manifest_row(metadata: dict[str, Any], scene_dir: Path, metadata_sha256: str) -> dict[str, Any]:
    physics = metadata["physics"]
    shifts = metadata["shifts"]
    geometry = metadata["geometry"]
    return {
        "scene_id": metadata["scene_id"],
        "split": metadata["split"],
        "difficulty": metadata["difficulty"],
        "scale": metadata["scale"],
        "seed": metadata["seed"],
        "forward_mode": physics["forward_mode"],
        "drift_model": physics["drift_model"],
        "min_feature_um": geometry["min_feature_um"],
        "delta_T_c": physics["delta_T_c"],
        "psf_sigma_lr_px": physics["psf_sigma_lr_px"],
        "psf_profile": physics.get("psf_profile", ""),
        "psf_calibration_status": physics.get("psf_calibration_status", ""),
        "noise_sigma_c": physics["noise_sigma_c"],
        "noise_model": physics.get("noise_model", "iid_gaussian"),
        "fpn_sigma_px": physics.get("fpn_sigma_px", ""),
        "stripe_sigma_c": physics.get("stripe_sigma_c", ""),
        "shift_profile": shifts["profile"],
        "scene_dir": str(scene_dir),
        "metadata_sha256": metadata_sha256,
    }


def generate_dataset(config: dict[str, Any], output_root: Path, shift_config_path: Path) -> list[dict[str, Any]]:
    _validate_supported_config(config)
    output_root.mkdir(parents=True, exist_ok=True)
    scenes_root = output_root / "scenes"
    n_scenes = int(config["num_scenes"])
    difficulties = list(config.get("difficulties") or [])
    if not difficulties:
        distribution = config.get("difficulty_distribution", {"easy": n_scenes})
        difficulties = [difficulty for difficulty, count in distribution.items() for _ in range(int(count))]
    difficulties = (difficulties * int(np.ceil(n_scenes / len(difficulties))))[:n_scenes]
    seeds = list(config.get("seeds") or range(1001, 1001 + n_scenes))
    seeds = (seeds * int(np.ceil(n_scenes / len(seeds))))[:n_scenes]

    manifest: list[dict[str, Any]] = []
    for index, (difficulty, seed) in enumerate(zip(difficulties, seeds), start=1):
        seed = int(seed)
        scene_id = _scene_id(str(difficulty), index)
        shifts, shift_info = _load_shifts(str(config["shift_profile"]), int(config["n_frames_per_scene"]), shift_config_path)
        mask, geometry = _build_scene_mask(config, str(difficulty), seed)
        hr_temperature = _temperature(mask, config, str(difficulty), seed)
        edge_map = _edge_map(mask, config)
        psf_sigma, psf_metadata = _resolve_psf_settings(config)
        raw, hp = _make_burst(hr_temperature, shifts, config, seed)
        metadata = {
            "schema_version": "0.1",
            "dataset": config.get("dataset", "ThermalChipPhantom"),
            "engine": config.get("engine", "TCForge"),
            "scene_id": scene_id,
            "seed": seed,
            "split": config.get("split", "test") if isinstance(config.get("split", "test"), str) else "mixed",
            "difficulty": difficulty,
            "scale": int(config["scale"]),
            "lr_shape": list(map(int, config["lr_shape"])),
            "hr_shape": list(mask.shape),
            "pixel_size_um": float(config["pixel_size_um"]),
            "spatial_resolution_um": float(config["spatial_resolution_um"]),
            "geometry": geometry,
            "physics": {
                "T_bg_c": float(config["T_bg_c"]),
                "delta_T_c": float(config["delta_T_c_by_difficulty"][difficulty]),
                "low_freq_background_c": float(config["low_freq_amplitude_c"]),
                "psf_sigma_lr_px": psf_sigma,
                **psf_metadata,
                "noise_sigma_c": float(config["noise_sigma_c"]),
                "noise_model": str(config.get("noise_model", "iid_gaussian")),
                "fpn_sigma_px": float(config.get("fpn_sigma_px", 5.0)),
                "stripe_sigma_c": (
                    None if config.get("stripe_sigma_c") is None else float(config.get("stripe_sigma_c"))
                ),
                "forward_mode": config.get("forward_mode", config.get("forward_modes", ["exact_ep06_point"])[0]),
                "highpass_sigma_lr_px": float(config["highpass_sigma_lr_px"]),
                "highpass_mode": str(config.get("highpass_mode", "nearest")),
                "drift_model": str(config.get("drift_model", "none")),
            },
            "shifts": shift_info,
            "provenance": {
                "generator_git_sha": _git_sha(),
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "config_path": str(config.get("_config_path", "")),
            },
        }
        scene_dir = scenes_root / scene_id
        metadata_sha = _save_scene(
            scene_dir,
            hr_temperature=hr_temperature,
            mask=mask,
            edge_map=edge_map,
            raw=raw,
            highpass=hp,
            shifts=shifts,
            metadata=metadata,
        )
        manifest.append(_manifest_row(metadata, scene_dir, metadata_sha))
        print(f"generated synthetic scene {scene_id}: {raw.shape[0]} synthetic LR frames, lr={tuple(raw.shape[1:])}, scene_dir={scene_dir}")

    _write_manifest(output_root, manifest, config)
    return manifest


def _validate_supported_config(config: dict[str, Any]) -> None:
    """Fail fast on benchmark/P1 keys this CLI does not yet materialize."""

    unsupported: list[str] = []
    if "drift_tracks" in config:
        unsupported.append("drift_tracks")
    if isinstance(config.get("split"), dict):
        unsupported.append("split object")
    if len(config.get("forward_modes", [])) > 1:
        unsupported.append("multiple forward_modes")
    if config.get("storage_strategy") not in (None, "full_frame"):
        unsupported.append(f"storage_strategy={config.get('storage_strategy')!r}")
    if unsupported:
        raise ValueError(
            "This generator CLI currently supports the P0 smoke dataset path only. "
            f"Unsupported benchmark/P1 keys: {', '.join(unsupported)}. "
            "Use phantom_smoke.json or implement P1 benchmark materialization before running this config."
        )


def _write_manifest(output_root: Path, rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    csv_path = output_root / "manifest.csv"
    if rows:
        with csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
    _write_json(output_root / "manifest.json", rows)
    dataset_metadata = {
        "schema_version": "0.1",
        "dataset": config.get("dataset", "ThermalChipPhantom"),
        "engine": config.get("engine", "TCForge"),
        "version": config.get("version", "0.1.0"),
        "num_scenes": len(rows),
        "required_scene_files": list(REQUIRED_SCENE_FILES),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(output_root / "dataset_metadata.json", dataset_metadata)


def _apply_overrides(config: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = dict(config)
    out["_config_path"] = str(args.config)
    if args.max_scenes is not None:
        out["num_scenes"] = int(args.max_scenes)
        if "difficulties" in out:
            out["difficulties"] = out["difficulties"][: int(args.max_scenes)]
        if "seeds" in out:
            out["seeds"] = out["seeds"][: int(args.max_scenes)]
    if args.n_frames is not None:
        out["n_frames_per_scene"] = int(args.n_frames)
    if args.lr_shape is not None:
        out["lr_shape"] = [int(args.lr_shape[0]), int(args.lr_shape[1])]
        scale = int(out["scale"])
        out["hr_shape"] = [out["lr_shape"][0] * scale, out["lr_shape"][1] * scale]
    if args.shift_profile is not None:
        out["shift_profile"] = args.shift_profile
    if args.drift_model is not None:
        out["drift_model"] = args.drift_model
    if args.allow_fallback_demo:
        out["allow_fallback_demo"] = True
    return out


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--shift-config", type=Path, default=DEFAULT_SHIFT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-scenes", type=int, default=None, help="limit scene count for smoke/demo runs")
    parser.add_argument("--n-frames", type=int, default=None, help="override frames per scene")
    parser.add_argument("--lr-shape", type=_parse_shape, default=None, help="override LR shape as ROWS,COLS or ROWSxCOLS")
    parser.add_argument("--shift-profile", default=None)
    parser.add_argument("--drift-model", default=None)
    parser.add_argument(
        "--allow-fallback-demo",
        action="store_true",
        help="allow local fallback generation only for small notebook demos; formal runs fail fast",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = _apply_overrides(_load_json(args.config), args)
    manifest = generate_dataset(config, args.output_root, args.shift_config)
    print(f"wrote manifest: {args.output_root / 'manifest.csv'} ({len(manifest)} scenes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
