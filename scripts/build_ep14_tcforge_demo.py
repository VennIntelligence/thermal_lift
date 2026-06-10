"""Generate TCForge-based training demo bundle for EP14 4X Loss Atlas."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import zoom

DEFAULT_TRAINING_POOL_CONFIG = Path("configs/synthetic/training_pool_4x.json")


def _import_tcforge(project_root: Path):
    src = project_root / "tcforge" / "src"
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    from tcforge.classical_sr import drizzle_features
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
        drizzle_features,
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
    (_, _, _, _, load_shift_profile, _, _, _) = _import_tcforge(project_root)
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
    seed: int = 14,
    scale: int = 4,
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
    """Build a compact TCForge 4x scene aligned with ``training_pool_4x`` defaults."""

    (
        build_scene_mask_with_metadata,
        reconstruct_hr_temperature,
        generate_lr_burst,
        fuse_burst_to_features,
        _,
        add_noise,
        apply_drift,
        drizzle_features,
    ) = _import_tcforge(project_root)

    # Reconstruct 4x HR mask and temp target
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

    # Load 4x alignment shifts
    shifts_full, shift_meta = _load_shifts_with_fallback(
        project_root,
        shift_profile=shift_profile,
        shift_fallback_profile=shift_fallback_profile,
        n_frames=int(n_frames_per_scene),
        scale=scale,
        seed=seed,
        shift_jitter_std_px=shift_jitter_std_px,
    )

    # Forward physical model: generate LR burst (scale=4, LR shape is HR // 4)
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

    # 1. 1x features at LR coordinates (5 channels)
    obs_1x = fuse_burst_to_features(lr_burst_full, shifts_full, sigma_bg=5.0)

    # 2. 4x drizzle features (3 channels)
    obs_4x = drizzle_features(
        lr_burst_full,
        shifts_full,
        scale=scale,
        output_shape=canvas_shape,
        kernel="bilinear",
    )

    # 3. Upsample 1x features by scale=4 and concatenate to form 8-channel input
    obs_1x_up = zoom(obs_1x, (1, scale, scale), order=1).astype(np.float32)
    
    # Handle minor float boundary shape rounding if any
    if obs_1x_up.shape[-2:] != canvas_shape:
        obs_1x_up = obs_1x_up[:, :canvas_shape[0], :canvas_shape[1]]

    obs_features = np.concatenate([obs_4x, obs_1x_up], axis=0)

    demo_count = min(int(n_frames_demo), lr_burst_full.shape[0])
    lr_burst = lr_burst_full[:demo_count].astype(np.float32, copy=False)
    shifts_demo = shifts_full[:demo_count].astype(np.float32, copy=False)

    return {
        "hr_mask": hr_mask,
        "hr_temperature": hr_temperature,
        "lr_burst": lr_burst,
        "lr_burst_full_count": int(lr_burst_full.shape[0]),
        "shifts_demo": shifts_demo,
        "shifts_full_count": int(n_frames_per_scene),
        "obs_features": obs_features,
        "obs_1x": obs_1x,
        "obs_4x": obs_4x,
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
            "drizzle_mean_4x",
            "drizzle_coverage_4x",
            "drizzle_variance_4x",
            "aligned_mean_upsampled",
            "aligned_median_upsampled",
            "coverage_upsampled",
            "variance_upsampled",
            "highpass_fused_mean_upsampled",
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
        obs_1x=demo["obs_1x"],
        obs_4x=demo["obs_4x"],
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
            "Mirrors configs/synthetic/training_pool_4x.json defaults: "
            f"{demo['shifts_full_count']} frames/scene, scale=4, "
            "drizzle features 4x concatenated with 1x features upsampled. "
            "Compact pool scenes store hr_mask PNG + obs_features_1x.npz + obs_features_4x.npz + metadata; "
            "LR burst is not kept on disk."
        ),
    }
    (output_dir / "training_demo_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return npz_path
