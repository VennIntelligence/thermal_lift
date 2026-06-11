from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from tcforge.reconstruct import reconstruct_hr_temperature
from tcforge.storage import load_scene_compact, save_scene_compact
from unet_sr.dataset import ThermalSRDataset


def _write_manifest(pool: Path) -> None:
    _write_manifest_generic(pool, scale=4, n_frames=3)


def _make_pool(tmp_path: Path, *, scale: int = 4, with_burst: bool = False) -> Path:
    pool = tmp_path / "pool"
    pool.mkdir()
    lr_h, lr_w = 8, 10
    hr_h, hr_w = lr_h * scale, lr_w * scale
    mask = np.zeros((hr_h, hr_w), dtype=np.uint8)
    mask[hr_h // 4 : hr_h * 3 // 4, hr_w // 4 : hr_w * 3 // 4] = 1
    edge = np.zeros_like(mask)
    edge[hr_h // 4, hr_w // 4 : hr_w * 3 // 4] = 1
    obs = np.zeros((5, lr_h, lr_w), dtype=np.float32)
    obs[0] = 20.0
    obs[2] = 1.0
    n_frames = 40 if with_burst else 3
    shifts = np.random.default_rng(42).uniform(-0.5, 0.5, (n_frames, 2)).astype(np.float32)
    lr_burst = (np.ones((n_frames, lr_h, lr_w), dtype=np.float32) * 20.0
                + np.random.default_rng(42).normal(0, 0.07, (n_frames, lr_h, lr_w)).astype(np.float32)) if with_burst else None
    save_scene_compact(
        pool / "scene_0000",
        hr_mask=mask,
        hr_edge=edge,
        obs_features=obs,
        shifts=shifts,
        lr_burst=lr_burst,
        metadata={
            "scene_id": "scene_0000",
            "seed": 123,
            "scale": scale,
            "lr_shape": [lr_h, lr_w],
            "hr_shape": [hr_h, hr_w],
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
            "n_frames": n_frames,
        },
    )
    _write_manifest_generic(pool, scale=scale, n_frames=n_frames)
    return pool


def _write_manifest_generic(pool: Path, *, scale: int = 4, n_frames: int = 3) -> None:
    lr_h, lr_w = 8, 10
    with (pool / "manifest.csv").open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "scene_id", "scene_dir", "difficulty", "seed", "scale",
                "lr_shape", "hr_shape", "T_bg_c", "delta_T_c",
                "psf_sigma_lr_px", "noise_sigma_c", "drift_model",
                "rotation_deg", "n_frames",
            ],
        )
        writer.writeheader()
        writer.writerow({
            "scene_id": "scene_0000",
            "scene_dir": "scene_0000",
            "difficulty": "easy",
            "seed": 123,
            "scale": scale,
            "lr_shape": f"{lr_h}x{lr_w}",
            "hr_shape": f"{lr_h * scale}x{lr_w * scale}",
            "T_bg_c": 20.5,
            "delta_T_c": 1.7,
            "psf_sigma_lr_px": 0.23,
            "noise_sigma_c": 0.07,
            "drift_model": "none",
            "rotation_deg": 47.6,
            "n_frames": n_frames,
        })


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


def test_dataset_precomputes_loss_weights_in_sample_when_boost_enabled(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path)
    dataset = ThermalSRDataset(
        pool,
        patch_size_hr=16,
        scale=4,
        seed=5,
        patches_per_scene=2,
        return_metadata=False,
        thin_boost=6.0,
        gap_boost=4.0,
    )
    sample = dataset[0]

    assert sample["thin_weight"].shape == (1, 16, 16)
    assert sample["gap_weight"].shape == (1, 16, 16)
    assert float(sample["thin_weight"].max()) >= 1.0
    assert float(sample["gap_weight"].max()) >= 1.0

    batch = next(iter(DataLoader(dataset, batch_size=2, num_workers=0)))
    assert batch["thin_weight"].shape == (2, 1, 16, 16)
    assert batch["gap_weight"].shape == (2, 1, 16, 16)


def test_dataset_skips_loss_weights_when_boost_disabled(tmp_path: Path) -> None:
    pool = _make_pool(tmp_path)
    dataset = ThermalSRDataset(pool, patch_size_hr=16, scale=4, seed=5, patches_per_scene=1)
    sample = dataset[0]

    assert "thin_weight" not in sample
    assert "gap_weight" not in sample


# --- V9A hybrid_drizzle2x tests ---


def test_hybrid_dataset_sample_shape(tmp_path: Path) -> None:
    """V9A: hybrid_drizzle2x produces 8ch obs at 2x grid."""
    pool = _make_pool(tmp_path, scale=2, with_burst=True)
    dataset = ThermalSRDataset(
        pool, patch_size_hr=8, scale=2, seed=5,
        patches_per_scene=2, input_mode="hybrid_drizzle2x",
    )
    sample = dataset[0]

    assert sample["obs_features"].shape == (8, 8, 8)
    assert sample["hr_target"].shape == (1, 8, 8)
    assert sample["hr_edge"].shape == (1, 8, 8)
    assert sample["hr_mask"].shape == (1, 8, 8)


def test_hybrid_dataset_augment_sync(tmp_path: Path) -> None:
    """V9A: augmentation transforms obs and target consistently."""
    pool = _make_pool(tmp_path, scale=2, with_burst=True)
    dataset = ThermalSRDataset(
        pool, patch_size_hr=8, scale=2, seed=42,
        patches_per_scene=4, input_mode="hybrid_drizzle2x",
        return_metadata=False,
    )
    s0 = dataset[0]
    s1 = dataset[1]
    assert s0["obs_features"].shape[0] == 8
    assert s1["obs_features"].shape[0] == 8
    assert s0["obs_features"].shape[1:] == s0["hr_target"].shape[1:]


def test_hybrid_burst_subset_reproducible(tmp_path: Path) -> None:
    """V9A: burst subset selection is deterministic for same epoch/scene."""
    pool = _make_pool(tmp_path, scale=2, with_burst=True)
    dataset = ThermalSRDataset(
        pool, patch_size_hr=8, scale=2, seed=99,
        patches_per_scene=2, input_mode="hybrid_drizzle2x",
    )
    s1 = dataset[0]
    dataset._cache.clear()
    s2 = dataset[0]
    assert np.allclose(s1["obs_features"].numpy(), s2["obs_features"].numpy())


def test_hybrid_burst_min_frames_enforced(tmp_path: Path) -> None:
    """V9A: min_burst_frames lower bound is respected."""
    pool = _make_pool(tmp_path, scale=2, with_burst=True)
    dataset = ThermalSRDataset(
        pool, patch_size_hr=8, scale=2, seed=5,
        patches_per_scene=2, input_mode="hybrid_drizzle2x",
        min_burst_frames=35,
    )
    _ = dataset[0]


def test_lr_mode_unchanged_with_input_mode_default(tmp_path: Path) -> None:
    """V9A regression: input_mode='lr' produces identical output to legacy."""
    pool = _make_pool(tmp_path)
    ds_legacy = ThermalSRDataset(pool, patch_size_hr=16, scale=4, seed=5, patches_per_scene=2)
    ds_explicit = ThermalSRDataset(pool, patch_size_hr=16, scale=4, seed=5, patches_per_scene=2, input_mode="lr")
    s_legacy = ds_legacy[0]
    s_explicit = ds_explicit[0]

    assert np.allclose(s_legacy["obs_features"].numpy(), s_explicit["obs_features"].numpy())
    assert np.allclose(s_legacy["hr_target"].numpy(), s_explicit["hr_target"].numpy())


def test_hybrid_dataset_epoch_changes_burst(tmp_path: Path) -> None:
    """V9A: changing epoch produces different burst augmentation."""
    pool = _make_pool(tmp_path, scale=2, with_burst=True)
    dataset = ThermalSRDataset(
        pool, patch_size_hr=8, scale=2, seed=5,
        patches_per_scene=2, input_mode="hybrid_drizzle2x",
    )
    s_epoch0 = dataset[0]["obs_features"].numpy().copy()
    dataset.set_epoch(1)
    dataset._cache.clear()
    s_epoch1 = dataset[0]["obs_features"].numpy()
    assert not np.allclose(s_epoch0, s_epoch1)
