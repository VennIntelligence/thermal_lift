from __future__ import annotations

from pathlib import Path

import numpy as np

from sr4x.dataset import ThermalSR4xDataset
from tcforge.storage import save_scene_compact


def _write_scene(scene_dir: Path) -> Path:
    mask = np.zeros((8, 8), dtype=np.float32)
    mask[2:6, 2:6] = 1.0
    mask[1, 2:6] = 0.5
    mask[6, 2:6] = 0.5
    mask[2:6, 1] = 0.5
    mask[2:6, 6] = 0.5
    edge = np.zeros_like(mask)
    edge[2, 2:6] = 1
    obs_1x = np.zeros((5, 2, 2), dtype=np.float32)
    obs_1x[0] = 21.0
    obs_1x[2] = 1.0
    shifts = np.asarray([[0.0, 0.0], [0.25, 0.0], [0.0, 0.25], [0.25, 0.25]], dtype=np.float32)
    burst = np.stack(
        [
            np.asarray([[21.0, 22.0], [23.0, 24.0]], dtype=np.float32),
            np.asarray([[21.1, 22.1], [23.1, 24.1]], dtype=np.float32),
            np.asarray([[20.9, 21.9], [22.9, 23.9]], dtype=np.float32),
            np.asarray([[21.2, 22.2], [23.2, 24.2]], dtype=np.float32),
        ],
        axis=0,
    )
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
        lr_burst=burst,
    )
    return scene_dir


def _write_scene_with_precomputed_drizzle(scene_dir: Path) -> tuple[Path, np.ndarray]:
    mask = np.zeros((16, 16), dtype=np.float32)
    mask[4:12, 5:11] = 1.0
    edge = np.zeros_like(mask)
    edge[4, 5:11] = 1.0
    obs_1x = np.zeros((5, 4, 4), dtype=np.float32)
    obs_1x[0] = 21.0
    obs_1x[2] = 1.0
    shifts = np.asarray([[0.0, 0.0]], dtype=np.float32)
    metadata = {
        "scene_id": scene_dir.name,
        "seed": 21,
        "scale": 4,
        "lr_shape": [4, 4],
        "hr_shape": [16, 16],
        "T_bg_c": 21.0,
        "delta_T_c": 2.0,
        "low_freq_amplitude_c": 0.0,
        "low_freq_sigma_px": 6.0,
        "low_freq_seed": 22,
        "n_frames": 1,
    }
    save_scene_compact(
        scene_dir,
        hr_mask=mask,
        hr_edge=edge,
        obs_features=obs_1x,
        shifts=shifts,
        metadata=metadata,
    )
    yy, xx = np.mgrid[:8, :8].astype(np.float32)
    obs_drz = np.stack(
        [
            yy * 10.0 + xx,
            100.0 + yy * 10.0 + xx,
            200.0 + yy * 10.0 + xx,
        ],
        axis=0,
    ).astype(np.float32)
    np.savez_compressed(scene_dir / "obs_features_2x.npz", obs_features=obs_drz)
    return scene_dir, obs_drz


def test_dataset_computes_hybrid_drizzle_from_burst_and_preserves_soft_mask(tmp_path: Path) -> None:
    _write_scene(tmp_path / "scene_0000")

    dataset = ThermalSR4xDataset(tmp_path, patch_size=8, patches_per_scene=2)
    sample = dataset[0]

    assert dataset.in_channels == 8
    assert sample["obs_features"].shape == (8, 4, 4)
    assert sample["hr_target"].shape == (1, 8, 8)
    assert sample["hr_edge"].shape == (1, 8, 8)
    assert sample["coverage"].shape == (1, 4, 4)
    assert sample["metadata"]["scale"] == 4
    target = sample["hr_target"].numpy()[0]
    assert np.any((target > 21.0) & (target < 23.0))


def test_dataset_can_compute_burst_augmented_features(tmp_path: Path) -> None:
    _write_scene(tmp_path / "scene_0000")

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
    assert sample["obs_features"].shape == (8, 4, 4)
    assert sample["metadata"]["burst_augmented"] is True
    assert sample["metadata"]["burst_frames_kept"] >= 1


def test_dataset_can_defer_1x_upsample(tmp_path: Path) -> None:
    _write_scene(tmp_path / "scene_0000")

    dataset = ThermalSR4xDataset(
        tmp_path,
        patch_size=8,
        patches_per_scene=1,
        defer_1x_upsample=True,
        return_metadata=False,
    )
    sample = dataset[0]

    assert set(sample) == {"obs_features_hr", "obs_features_1x_lr", "hr_target", "hr_edge", "drizzle_mean", "coverage"}
    assert sample["obs_features_hr"].shape == (3, 4, 4)
    assert sample["obs_features_1x_lr"].shape == (5, 2, 2)


def test_dataset_augments_loss_drizzle_channels_with_inputs(tmp_path: Path) -> None:
    _, obs_drz = _write_scene_with_precomputed_drizzle(tmp_path / "scene_0000")

    for defer_1x in (False, True):
        dataset = ThermalSR4xDataset(
            tmp_path,
            patch_size=16,
            patches_per_scene=1,
            seed=42,
            defer_1x_upsample=defer_1x,
            return_metadata=False,
        )
        sample = dataset[0]
        features_key = "obs_features_hr" if defer_1x else "obs_features"
        features = sample[features_key].numpy()

        np.testing.assert_array_equal(sample["drizzle_mean"].numpy()[0], features[0])
        np.testing.assert_array_equal(sample["coverage"].numpy()[0], features[1])
        assert not np.array_equal(sample["drizzle_mean"].numpy()[0], obs_drz[0])


def test_crop_origin_aligns_to_full_sr_scale(tmp_path: Path) -> None:
    _write_scene_with_precomputed_drizzle(tmp_path / "scene_0000")
    dataset = ThermalSR4xDataset(tmp_path, patch_size=8, patches_per_scene=64, seed=42)

    origins = [dataset._crop_origin(index, (24, 24)) for index in range(64)]

    assert all(y % dataset.scale == 0 and x % dataset.scale == 0 for y, x in origins)
