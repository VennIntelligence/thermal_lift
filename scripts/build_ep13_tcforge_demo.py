"""Generate TCForge-based training demo bundle for EP13 Loss Atlas.

库模块，无 CLI 入口：由 scripts/build_ep13_cache.py 导入并调用
build_tcforge_training_demo() / save_training_demo_bundle()。
输入: configs/synthetic/training_pool_2x.json + tcforge/src 合成引擎
输出: 调用方指定目录下的 2x 训练 demo bundle（NPZ + 元数据）
关联: EP13
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_TRAINING_POOL_CONFIG = Path("configs/synthetic/training_pool_2x.json")


def _import_tcforge(project_root: Path):
    src = project_root / "tcforge" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from tcforge.forward import generate_lr_burst
    from tcforge.fusion import fuse_burst_to_features
    from tcforge.geometry import build_scene_mask_with_metadata
    from tcforge.physics import add_noise, apply_drift
    from tcforge.reconstruct import reconstruct_hr_temperature
    from tcforge.shifts import load_shift_profile

    return (
        build_scene_mask_with_metadata,
        reconstruct_hr_temperature,
        generate_lr_burst,
        fuse_burst_to_features,
        load_shift_profile,
        add_noise,
        apply_drift,
    )


def _load_shifts_with_fallback(
    project_root: Path,
    *,
    shift_profile: str,
    shift_fallback_profile: str,
    n_frames: int,
    scale: int,
    seed: int,
    shift_jitter_std_px: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    (_, _, _, _, load_shift_profile, _, _) = _import_tcforge(project_root)
    try:
        shifts, shift_meta = load_shift_profile(shift_profile, n_frames=n_frames, scale=scale)
        shift_meta = dict(shift_meta)
        shift_meta["fallback_used"] = False
    except (FileNotFoundError, ValueError) as exc:
        shifts, shift_meta = load_shift_profile(
            shift_fallback_profile,
            n_frames=n_frames,
            scale=scale,
            seed=seed,
            jitter_std_px=shift_jitter_std_px,
        )
        shift_meta = dict(shift_meta)
        shift_meta["fallback_used"] = True
        shift_meta["requested_profile"] = shift_profile
        shift_meta["fallback_reason"] = str(exc)

    if shift_jitter_std_px > 0:
        rng = np.random.default_rng(seed + 17)
        shifts = (
            shifts + rng.normal(0.0, shift_jitter_std_px, size=shifts.shape).astype(np.float32)
        ).astype(np.float32, copy=False)
        shift_meta["shift_jitter_std_px"] = float(shift_jitter_std_px)
    return shifts.astype(np.float32, copy=False), shift_meta


def build_tcforge_training_demo(
    project_root: Path,
    *,
    seed: int = 13,
    scale: int = 2,
    canvas_shape: tuple[int, int] = (256, 320),
    n_frames_demo: int = 16,
    n_frames_per_scene: int = 248,
    rotation_deg_center: float = 47.6,
    rotation_jitter_deg: float = 1.5,
    shift_profile: str = "real_default_contour_refined",
    shift_fallback_profile: str = "ideal_phase_grid",
    shift_jitter_std_px: float = 0.02,
    noise_sigma_c: float = 0.0724,
    noise_model: str = "detector_realistic",
    fpn_sigma_px: float = 5.0,
    stripe_sigma_c: float = 0.015,
    drift_model: str = "scalar_offset",
    drift_amplitude_c: float = 0.15,
    drift_lowfreq_sigma_px: float = 96.0,
) -> dict[str, Any]:
    """Build a compact TCForge scene aligned with ``training_pool_2x`` defaults."""

    (
        build_scene_mask_with_metadata,
        reconstruct_hr_temperature,
        generate_lr_burst,
        fuse_burst_to_features,
        _,
        add_noise,
        apply_drift,
    ) = _import_tcforge(project_root)

    hr_mask, scene_meta = build_scene_mask_with_metadata(
        "medium",
        seed,
        rotation_deg_center=rotation_deg_center,
        rotation_jitter_deg=rotation_jitter_deg,
        canvas_shape=canvas_shape,
        scale=scale,
    )
    hr_mask = hr_mask.astype(np.float32)
    rotation_deg = float(scene_meta.get("rotation_deg", rotation_deg_center))

    t_bg = 21.0
    delta_t = 2.0
    hr_temperature = reconstruct_hr_temperature(
        hr_mask,
        T_bg_c=t_bg,
        delta_T_c=delta_t,
        low_freq_amplitude_c=0.2,
        low_freq_sigma_px=96.0,
        seed=seed,
    ).astype(np.float32)

    shifts_full, shift_meta = _load_shifts_with_fallback(
        project_root,
        shift_profile=shift_profile,
        shift_fallback_profile=shift_fallback_profile,
        n_frames=int(n_frames_per_scene),
        scale=scale,
        seed=seed,
        shift_jitter_std_px=shift_jitter_std_px,
    )

    lr_burst_full = generate_lr_burst(
        hr_temperature,
        shifts_full,
        forward_mode="physical_block_average",
        psf_sigma_lr_px=0.5,
        scale=scale,
        workers=1,
    )
    lr_burst_full = add_noise(
        lr_burst_full,
        noise_sigma_c=noise_sigma_c,
        seed=seed + 1000,
        noise_model=noise_model,
        fpn_sigma_px=fpn_sigma_px,
        stripe_sigma_c=stripe_sigma_c,
    )
    lr_burst_full = apply_drift(
        lr_burst_full,
        model=drift_model,
        seed=seed + 2000,
        amplitude_c=drift_amplitude_c,
        lowfreq_sigma_px=drift_lowfreq_sigma_px,
    )

    demo_count = min(int(n_frames_demo), lr_burst_full.shape[0])
    lr_burst = lr_burst_full[:demo_count].astype(np.float32, copy=False)
    shifts_demo = shifts_full[:demo_count].astype(np.float32, copy=False)
    obs_features = fuse_burst_to_features(lr_burst_full, shifts_full, sigma_bg=5.0)

    return {
        "hr_mask": hr_mask,
        "hr_temperature": hr_temperature,
        "lr_burst": lr_burst,
        "lr_burst_full_count": int(lr_burst_full.shape[0]),
        "shifts_demo": shifts_demo,
        "shifts_full_count": int(n_frames_per_scene),
        "obs_features": obs_features,
        "rotation_deg": rotation_deg,
        "scale": int(scale),
        "canvas_shape": tuple(map(int, canvas_shape)),
        "scene_meta": scene_meta,
        "temperature_meta": {
            "T_bg_c": t_bg,
            "delta_T_c": delta_t,
            "low_freq_amplitude_c": 0.2,
            "low_freq_sigma_px": 96.0,
            "seed": seed,
        },
        "physics_meta": {
            "rotation_deg_center": rotation_deg_center,
            "rotation_jitter_deg": rotation_jitter_deg,
            "shift_profile": shift_profile,
            "shift_fallback_profile": shift_fallback_profile,
            "shift_meta": shift_meta,
            "noise_model": noise_model,
            "noise_sigma_c": noise_sigma_c,
            "fpn_sigma_px": fpn_sigma_px,
            "stripe_sigma_c": stripe_sigma_c,
            "drift_model": drift_model,
            "drift_amplitude_c": drift_amplitude_c,
            "drift_lowfreq_sigma_px": drift_lowfreq_sigma_px,
            "training_pool_config_ref": str(DEFAULT_TRAINING_POOL_CONFIG),
        },
        "obs_channel_names": [
            "aligned_mean",
            "aligned_median",
            "coverage",
            "variance",
            "highpass_fused_mean",
        ],
    }


def save_training_demo_bundle(output_dir: Path, demo: dict[str, Any]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / "training_demo_bundle.npz"
    np.savez_compressed(
        npz_path,
        hr_mask=demo["hr_mask"],
        hr_temperature=demo["hr_temperature"],
        lr_burst=demo["lr_burst"],
        shifts_demo=demo["shifts_demo"],
        obs_features=demo["obs_features"],
    )
    physics = demo.get("physics_meta", {})
    meta = {
        "rotation_deg": demo["rotation_deg"],
        "scale": demo["scale"],
        "canvas_shape": demo["canvas_shape"],
        "n_frames_demo": int(demo["lr_burst"].shape[0]),
        "n_frames_per_scene": demo["shifts_full_count"],
        "n_frames_train_ref": demo["shifts_full_count"],
        "obs_channel_names": demo["obs_channel_names"],
        "temperature_meta": demo["temperature_meta"],
        "physics_meta": physics,
        "scene_meta_keys": sorted(demo["scene_meta"].keys()),
        "scene_rotation_deg": demo["rotation_deg"],
        "training_pool_note": (
            "Mirrors configs/synthetic/training_pool_2x.json defaults: "
            f"{demo['shifts_full_count']} frames/scene, real refined shifts, "
            "detector_realistic noise, and per-scene drift before fusion. "
            "Compact pool scenes store hr_mask PNG + obs_features_1x.npz + metadata; "
            "LR burst is not kept on disk."
        ),
    }
    (output_dir / "training_demo_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return npz_path
