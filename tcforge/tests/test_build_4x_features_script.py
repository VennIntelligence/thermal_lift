from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from tcforge.storage import save_scene_compact


def _load_builder_module():
    path = PROJECT_ROOT / "scripts" / "build_4x_features.py"
    spec = importlib.util.spec_from_file_location("build_4x_features", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("failed to load build_4x_features.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["build_4x_features"] = module
    spec.loader.exec_module(module)
    return module


def test_build_scene_features_writes_ep12_artifacts(tmp_path: Path) -> None:
    builder = _load_builder_module()
    scene_dir = tmp_path / "scene_0000"
    mask = np.zeros((8, 8), dtype=np.uint8)
    edge = np.zeros_like(mask)
    obs_features = np.zeros((5, 2, 2), dtype=np.float32)
    obs_features[0] = 1.0
    shifts = np.zeros((4, 2), dtype=np.float32)
    metadata = {
        "scene_id": "scene_0000",
        "difficulty": "easy",
        "seed": 1,
        "scale": 4,
        "lr_shape": [2, 2],
        "hr_shape": [8, 8],
        "T_bg_c": 21.0,
        "delta_T_c": 2.0,
        "n_frames": 4,
    }
    save_scene_compact(
        scene_dir,
        hr_mask=mask,
        hr_edge=edge,
        obs_features=obs_features,
        shifts=shifts,
        metadata=metadata,
    )
    np.save(scene_dir / "lr_burst.npy", np.ones((4, 2, 2), dtype=np.float16))

    result = builder.build_scene_features(scene_dir, builder.BuildOptions(kernel="nearest", force=True))

    assert result["status"] == "written"
    with np.load(scene_dir / "obs_features_4x.npz") as data:
        assert data["obs_features"].shape == (3, 8, 8)
    with np.load(scene_dir / "obs_features_2x_up4x.npz") as data:
        assert data["obs_features"].shape == (3, 8, 8)
    with np.load(scene_dir / "obs_features_1x_up4x.npz") as data:
        assert data["obs_features"].shape == (5, 8, 8)
