from __future__ import annotations

from pathlib import Path

import numpy as np

from sr4x.dataset import ThermalSR4xDataset
from tcforge.storage import save_scene_compact


def _write_scene(scene_dir: Path, *, write_precomputed: bool = True) -> Path:
    mask = np.zeros((8, 8), dtype=np.uint8)
    mask[2:6, 2:6] = 1
    edge = np.zeros_like(mask)
    edge[2, 2:6] = 1
    obs_1x = np.zeros((5, 2, 2), dtype=np.float32)
    obs_1x[0] = 21.0
    obs_1x[2] = 1.0
    shifts = np.asarray([[0.0, 0.0], [0.25, 0.0], [0.0, 0.25], [0.25, 0.25]], dtype=np.float32)
    metadata = {
        "scene_id": scene_dir.name,
        "difficulty": "easy",
        "seed": 11,
        "scale": 4,
        "lr_shape": [2, 2],
        "hr_shape": [8, 8],
        "T_bg_c": 21.0,
        "delta_T_c": 2.0,
        "low_freq_amplitude_c": 0.0,
        "low_freq_sigma_px": 6.0,
        "low_freq_seed": 12,
        "n_frames": 4,
    }
    save_scene_compact(
        scene_dir,
        hr_mask=mask,
        hr_edge=edge,
        obs_features=obs_1x,
        shifts=shifts,
        metadata=metadata,
    )
    burst = np.stack(
        [
            np.asarray([[21.0, 22.0], [23.0, 24.0]], dtype=np.float32),
            np.asarray([[21.1, 22.1], [23.1, 24.1]], dtype=np.float32),
            np.asarray([[20.9, 21.9], [22.9, 23.9]], dtype=np.float32),
            np.asarray([[21.2, 22.2], [23.2, 24.2]], dtype=np.float32),
        ],
        axis=0,
    )
    np.save(scene_dir / "lr_burst.npy", burst.astype(np.float16))
    if write_precomputed:
        np.savez_compressed(scene_dir / "obs_features_4x.npz", obs_features=np.ones((3, 8, 8), dtype=np.float16))
        np.savez_compressed(scene_dir / "obs_features_2x_up4x.npz", obs_features=np.ones((3, 8, 8), dtype=np.float16) * 2)
        np.savez_compressed(scene_dir / "obs_features_1x_up4x.npz", obs_features=np.ones((5, 8, 8), dtype=np.float16) * 3)
    return scene_dir


def test_dataset_loads_precomputed_multiscale_features(tmp_path: Path) -> None:
    _write_scene(tmp_path / "scene_0000", write_precomputed=True)

    dataset = ThermalSR4xDataset(tmp_path, patch_size=4, patches_per_scene=2, include_multiscale=True)
    sample = dataset[0]

    assert dataset.in_channels == 11
    assert sample["obs_features"].shape == (11, 4, 4)
    assert sample["hr_target"].shape == (1, 4, 4)
    assert sample["hr_edge"].shape == (1, 4, 4)
    assert sample["coverage_4x"].shape == (1, 4, 4)
    assert sample["metadata"]["scale"] == 4


def test_dataset_can_compute_burst_augmented_features(tmp_path: Path) -> None:
    _write_scene(tmp_path / "scene_0000", write_precomputed=False)

    dataset = ThermalSR4xDataset(
        tmp_path,
        patch_size=8,
        patches_per_scene=1,
        include_multiscale=False,
        burst_augment=True,
        burst_keep_range=(0.5, 0.75),
        min_burst_frames=1,
        shift_noise_std_px=0.02,
    )
    sample = dataset[0]

    assert dataset.in_channels == 8
    assert sample["obs_features"].shape == (8, 8, 8)
    assert sample["metadata"]["burst_augmented"] is True
    assert sample["metadata"]["burst_frames_kept"] >= 1


def test_dataset_can_defer_1x_upsample(tmp_path: Path) -> None:
    _write_scene(tmp_path / "scene_0000", write_precomputed=True)

    dataset = ThermalSR4xDataset(
        tmp_path,
        patch_size=4,
        patches_per_scene=1,
        defer_1x_upsample=True,
        return_metadata=False,
    )
    sample = dataset[0]

    assert set(sample) == {"obs_features_hr", "obs_features_1x_lr", "hr_target", "hr_edge", "drizzle_mean_4x", "coverage_4x"}
    assert sample["obs_features_hr"].shape == (3, 4, 4)
    assert sample["obs_features_1x_lr"].shape == (5, 1, 1)
