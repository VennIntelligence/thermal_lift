from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tcforge.manifest as manifest


def _write_dummy_scene(scene_dir: Path) -> None:
    scene_dir.mkdir(parents=True)
    np.save(scene_dir / "hr_temperature_2x.npy", np.ones((8, 10), dtype=np.float32))
    np.save(scene_dir / "hr_mask_2x.npy", np.ones((8, 10), dtype=np.uint8))
    np.save(scene_dir / "hr_edge_map_2x.npy", np.zeros((8, 10), dtype=np.float32))
    np.save(scene_dir / "lr_burst_raw.npy", np.ones((3, 4, 5), dtype=np.float32))
    np.save(scene_dir / "lr_burst_highpass.npy", np.zeros((3, 4, 5), dtype=np.float32))
    np.save(scene_dir / "shifts.npy", np.zeros((3, 2), dtype=np.float32))
    metadata = {
        "scene_id": scene_dir.name,
        "scale": 2,
        "lr_shape": [4, 5],
        "hr_shape": [8, 10],
        "physics": {"forward_mode": "exact_ep06_point", "drift_model": "none"},
    }
    (scene_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_json_metadata_write_read_round_trip(tmp_path: Path) -> None:
    payload = {"scene_id": "tcp_easy_0001", "lr_shape": [4, 5]}
    path = tmp_path / "metadata.json"

    manifest.write_json(payload, path)
    loaded = manifest.read_json(path)

    assert loaded == payload


def test_manifest_write_read_and_validation_round_trip(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scenes" / "tcp_easy_0001"
    _write_dummy_scene(scene_dir)
    rows = [
        {
            "scene_id": "tcp_easy_0001",
            "split": "test",
            "difficulty": "easy",
            "scale": 2,
            "seed": 1001,
            "lr_shape": [4, 5],
            "hr_shape": [8, 10],
            "forward_mode": "exact_ep06_point",
            "drift_model": "none",
            "min_feature_um": 40.0,
            "delta_T_c": 2.5,
            "psf_sigma_lr_px": 0.5,
            "noise_sigma_c": 0.0724,
            "shift_profile": "ideal_phase_grid",
            "scene_dir": str(scene_dir),
            "metadata_sha256": "",
        }
    ]

    manifest_path = tmp_path / "manifest.csv"
    manifest.write_manifest_csv(rows, manifest_path)
    loaded = manifest_path.read_text(encoding="utf-8")
    assert "tcp_easy_0001" in loaded

    manifest.validate_scene_manifest(rows[0])
    found = manifest.validate_file_list(
        scene_dir,
        (
            "hr_temperature_2x.npy",
            "hr_mask_2x.npy",
            "hr_edge_map_2x.npy",
            "lr_burst_raw.npy",
            "lr_burst_highpass.npy",
            "shifts.npy",
            "metadata.json",
        ),
    )
    assert len(found) == 7
