from __future__ import annotations

from pathlib import Path

import numpy as np

from sr4x.evaluate import split_half_consistency, split_half_consistency_scene
from tcforge.storage import save_scene_compact


def test_split_half_consistency_array_baseline() -> None:
    burst = np.ones((4, 2, 2), dtype=np.float32)
    shifts = np.zeros((4, 2), dtype=np.float32)

    result = split_half_consistency(burst, shifts, scale=4, kernel="nearest")

    assert result["split_half_odd_frames"] == 2
    assert result["split_half_even_frames"] == 2
    assert result["split_half_eval_pixels"] > 0
    assert result["split_half_mae_c"] == 0.0


def test_split_half_consistency_scene_with_predictor(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scene_0000"
    mask = np.zeros((8, 8), dtype=np.uint8)
    edge = np.zeros_like(mask)
    obs = np.zeros((5, 2, 2), dtype=np.float32)
    shifts = np.zeros((4, 2), dtype=np.float32)
    save_scene_compact(
        scene_dir,
        hr_mask=mask,
        hr_edge=edge,
        obs_features=obs,
        shifts=shifts,
        metadata={
            "scene_id": "scene_0000",
            "difficulty": "easy",
            "seed": 1,
            "scale": 4,
            "lr_shape": [2, 2],
            "hr_shape": [8, 8],
            "T_bg_c": 21.0,
            "delta_T_c": 2.0,
            "n_frames": 4,
        },
    )
    np.save(scene_dir / "lr_burst.npy", np.ones((4, 2, 2), dtype=np.float16))

    result = split_half_consistency_scene(scene_dir, predictor=lambda features: features[0], kernel="nearest")

    assert result["scene_id"] == "scene_0000"
    assert result["scale"] == 4
    assert result["split_half_mae_c"] == 0.0
