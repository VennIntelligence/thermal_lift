from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcforge.storage import COMPACT_SCENE_FILES, load_scene_compact, save_scene_compact


def _metadata() -> dict[str, object]:
    return {
        "scene_id": "scene_0042",
        "difficulty": "medium",
        "seed": 2042,
        "scale": 4,
        "lr_shape": [12, 16],
        "hr_shape": [48, 64],
        "T_bg_c": 21.3,
        "delta_T_c": 1.8,
        "low_freq_amplitude_c": 0.25,
        "low_freq_sigma_px": 12.0,
        "low_freq_seed": 9042,
        "psf_sigma_lr_px": 0.23,
        "noise_sigma_c": 0.072,
        "drift_model": "scalar_offset",
        "rotation_deg": 47.8,
        "n_frames": 5,
        "forward_mode": "physical_block_average",
        "obs_features_channels": ["aligned_mean", "aligned_median", "coverage", "variance", "highpass_fused"],
        "obs_features_resolution": "1x",
    }


def test_compact_scene_storage_round_trip(tmp_path: Path) -> None:
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[8:32, 10:40] = 1
    edge = np.zeros_like(mask)
    edge[8, 10:40] = 1
    rng = np.random.default_rng(123)
    obs_features = rng.normal(0.0, 0.2, size=(5, 12, 16)).astype(np.float32)
    shifts = rng.normal(0.0, 0.5, size=(5, 2)).astype(np.float32)

    scene_dir = save_scene_compact(
        tmp_path / "scene_0042",
        hr_mask=mask,
        hr_edge=edge,
        obs_features=obs_features,
        shifts=shifts,
        metadata=_metadata(),
    )
    loaded = load_scene_compact(scene_dir)

    for name in COMPACT_SCENE_FILES:
        assert (scene_dir / name).exists()
    assert np.array_equal(loaded["hr_mask"], mask)
    assert np.array_equal(loaded["hr_edge"], edge)
    assert loaded["obs_features"].dtype == np.float16
    assert np.max(np.abs(loaded["obs_features"].astype(np.float32) - obs_features)) < 0.01
    assert loaded["shifts"].dtype == np.float32
    assert np.allclose(loaded["shifts"], shifts)
    assert loaded["metadata"]["scale"] == 4
    for key in (
        "scale",
        "lr_shape",
        "hr_shape",
        "T_bg_c",
        "delta_T_c",
        "low_freq_amplitude_c",
        "low_freq_sigma_px",
        "low_freq_seed",
        "psf_sigma_lr_px",
        "noise_sigma_c",
        "n_frames",
        "forward_mode",
        "obs_features_channels",
        "obs_features_resolution",
    ):
        assert key in loaded["metadata"]


def test_compact_scene_storage_preserves_soft_mask_coverage_quantized(tmp_path: Path) -> None:
    mask = np.zeros((12, 16), dtype=np.float32)
    mask[2:8, 3:10] = 1.0
    mask[8, 3:10] = np.linspace(0.0, 1.0, 7, dtype=np.float32)
    edge = (mask >= 0.5).astype(np.uint8)
    obs_features = np.zeros((5, 3, 4), dtype=np.float32)
    shifts = np.zeros((5, 2), dtype=np.float32)

    scene_dir = save_scene_compact(
        tmp_path / "scene_soft",
        hr_mask=mask,
        hr_edge=edge,
        obs_features=obs_features,
        shifts=shifts,
        metadata={**_metadata(), "lr_shape": [3, 4], "hr_shape": [12, 16]},
    )
    loaded = load_scene_compact(scene_dir)

    assert loaded["hr_mask"].dtype == np.float32
    assert 0.0 <= float(loaded["hr_mask"].min()) <= float(loaded["hr_mask"].max()) <= 1.0
    assert np.max(np.abs(loaded["hr_mask"] - mask)) <= (0.5 / 255.0 + 1e-6)
    assert set(np.unique(loaded["hr_edge"]).tolist()) <= {0, 1}


def test_load_scene_compact_includes_optional_ep12_artifacts(tmp_path: Path) -> None:
    mask = np.zeros((8, 8), dtype=np.uint8)
    edge = np.zeros_like(mask)
    obs_features = np.zeros((5, 2, 2), dtype=np.float32)
    shifts = np.zeros((3, 2), dtype=np.float32)
    scene_dir = save_scene_compact(
        tmp_path / "scene_optional",
        hr_mask=mask,
        hr_edge=edge,
        obs_features=obs_features,
        shifts=shifts,
        metadata={**_metadata(), "lr_shape": [2, 2], "hr_shape": [8, 8], "n_frames": 3},
    )
    lr_burst = np.ones((3, 2, 2), dtype=np.float16)
    np.save(scene_dir / "lr_burst.npy", lr_burst)
    np.savez_compressed(scene_dir / "obs_features_4x.npz", obs_features=np.ones((3, 8, 8), dtype=np.float16))
    np.savez_compressed(scene_dir / "obs_features_2x_up4x.npz", obs_features=np.ones((3, 8, 8), dtype=np.float16) * 2)
    np.savez_compressed(scene_dir / "obs_features_1x_up4x.npz", obs_features=np.ones((5, 8, 8), dtype=np.float16) * 3)

    loaded = load_scene_compact(scene_dir)

    assert "lr_burst" in loaded
    assert loaded["lr_burst"].shape == (3, 2, 2)
    assert loaded["obs_features_4x"].shape == (3, 8, 8)
    assert loaded["obs_features_2x_up4x"].shape == (3, 8, 8)
    assert loaded["obs_features_1x_up4x"].shape == (5, 8, 8)
    assert "lr_burst.npy" not in COMPACT_SCENE_FILES
