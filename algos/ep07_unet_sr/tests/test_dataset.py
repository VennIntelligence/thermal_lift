from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from tcforge.reconstruct import reconstruct_hr_temperature
from tcforge.storage import load_scene_compact, save_scene_compact
from unet_sr.dataset import ThermalSRDataset


def _write_manifest(pool: Path) -> None:
    with (pool / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
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
            ],
        )
        writer.writeheader()
        writer.writerow(
            {
                "scene_id": "scene_0000",
                "scene_dir": "scene_0000",
                "difficulty": "easy",
                "seed": 123,
                "scale": 4,
                "lr_shape": "8x10",
                "hr_shape": "32x40",
                "T_bg_c": 20.5,
                "delta_T_c": 1.7,
                "psf_sigma_lr_px": 0.23,
                "noise_sigma_c": 0.07,
                "drift_model": "none",
                "rotation_deg": 47.6,
                "n_frames": 3,
            }
        )


def _make_pool(tmp_path: Path) -> Path:
    pool = tmp_path / "pool"
    pool.mkdir()
    mask = np.zeros((32, 40), dtype=np.uint8)
    mask[8:24, 12:28] = 1
    edge = np.zeros_like(mask)
    edge[8, 12:28] = 1
    obs = np.zeros((5, 8, 10), dtype=np.float32)
    obs[0] = 20.0
    obs[2] = 1.0
    shifts = np.zeros((3, 2), dtype=np.float32)
    save_scene_compact(
        pool / "scene_0000",
        hr_mask=mask,
        hr_edge=edge,
        obs_features=obs,
        shifts=shifts,
        metadata={
            "scene_id": "scene_0000",
            "seed": 123,
            "scale": 4,
            "lr_shape": [8, 10],
            "hr_shape": [32, 40],
            "T_bg_c": 20.5,
            "delta_T_c": 1.7,
            "low_freq_amplitude_c": 0.25,
            "low_freq_sigma_px": 8.0,
            "low_freq_seed": 999,
            "difficulty": "easy",
            "psf_sigma_lr_px": 0.23,
            "noise_sigma_c": 0.07,
            "drift_model": "none",
            "rotation_deg": 47.6,
            "n_frames": 3,
        },
    )
    _write_manifest(pool)
    return pool


def test_dataset_reads_compact_scene_and_reconstructs_metadata_target(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path)
    dataset = ThermalSRDataset(pool, patch_size_hr=16, scale=4, seed=5, patches_per_scene=2)

    sample = dataset[0]

    assert len(dataset) == 2
    assert sample["obs_features"].shape == (5, 4, 4)
    assert sample["hr_target"].shape == (1, 16, 16)
    assert sample["hr_edge"].shape == (1, 16, 16)
    assert sample["hr_mask"].shape == (1, 16, 16)
    assert sample["obs_features"].dtype.is_floating_point
    assert sample["hr_mask"].dtype.is_floating_point
    assert np.isfinite(sample["hr_mask"].numpy()).all()
    assert sample["metadata"]["scale"] == 4

    scene = dataset._load_cached(0)
    y_lr, x_lr = dataset._crop_origin_lr(0, (8, 10))
    y_hr, x_hr = y_lr * 4, x_lr * 4
    loaded = load_scene_compact(pool / "scene_0000")
    expected = reconstruct_hr_temperature(
        loaded["hr_mask"],
        T_bg_c=20.5,
        delta_T_c=1.7,
        low_freq_amplitude_c=0.25,
        low_freq_sigma_px=8.0,
        seed=999,
    )
    assert np.array_equal(scene["hr_target"], expected)
    assert np.allclose(sample["hr_target"].numpy()[0], scene["hr_target"][y_hr : y_hr + 16, x_hr : x_hr + 16])


def test_dataset_default_collate_keeps_hr_mask_without_metadata(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path)
    dataset = ThermalSRDataset(
        pool,
        patch_size_hr=16,
        scale=4,
        seed=5,
        patches_per_scene=2,
        return_metadata=False,
    )
    batch = next(iter(DataLoader(dataset, batch_size=2)))

    assert "metadata" not in batch
    assert batch["hr_mask"].shape == (2, 1, 16, 16)
    assert batch["hr_mask"].dtype.is_floating_point
