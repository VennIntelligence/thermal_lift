from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np


SRC = Path(__file__).resolve().parents[1] / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import tcforge.evaluate as evaluate


def _write_scene(scene_dir: Path) -> None:
    scene_dir.mkdir(parents=True)
    hr = np.zeros((8, 10), dtype=np.float32)
    hr[2:6, 3:7] = 2.0
    mask = (hr > 0).astype(np.uint8)
    edge = np.zeros_like(hr, dtype=np.float32)
    edge[2:6, 3] = 1.0
    raw = np.stack([hr[::2, ::2], hr[1::2, ::2]], axis=0).astype(np.float32)
    highpass = np.zeros_like(raw, dtype=np.float32)
    shifts = np.asarray([[0.0, 0.0], [0.5, 0.0]], dtype=np.float32)
    np.save(scene_dir / "hr_temperature_2x.npy", hr)
    np.save(scene_dir / "hr_mask_2x.npy", mask)
    np.save(scene_dir / "hr_edge_map_2x.npy", edge)
    np.save(scene_dir / "lr_burst_raw.npy", raw)
    np.save(scene_dir / "lr_burst_highpass.npy", highpass)
    np.save(scene_dir / "shifts.npy", shifts)
    metadata = {
        "scene_id": scene_dir.name,
        "split": "test",
        "difficulty": "easy",
        "scale": 2,
        "physics": {"forward_mode": "exact_ep06_point", "drift_model": "none"},
    }
    (scene_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")


def test_summarize_scene_returns_scene_level_contract(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scenes" / "tcp_easy_0001"
    _write_scene(scene_dir)

    summary = evaluate.summarize_scene(scene_dir)

    assert summary["scene_id"] == "tcp_easy_0001"
    assert summary["n_frames"] == 2
    assert summary["scale"] == 2
    assert summary["tcforge_hr_finite_fraction"] == 1.0
    assert summary["tcforge_raw_finite_fraction"] == 1.0


def test_evaluate_scene_adds_reconstruction_and_mask_metrics(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scenes" / "tcp_easy_0001"
    _write_scene(scene_dir)
    hr = np.load(scene_dir / "hr_temperature_2x.npy")
    mask = np.load(scene_dir / "hr_mask_2x.npy")
    np.save(scene_dir / "sr_temperature_2x.npy", hr + 0.1)
    np.save(scene_dir / "sr_mask_2x.npy", mask)

    result = evaluate.evaluate_scene(scene_dir)

    assert np.isclose(result["sr_temperature_mae_c"], 0.1, atol=1e-6)
    assert result["sr_temperature_nrmse"] > 0
    assert result["sr_mask_iou"] == 1.0
    assert result["sr_mask_boundary_f1"] == 1.0


def test_evaluate_dataset_aggregates_manifest_rows(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scenes" / "tcp_easy_0001"
    _write_scene(scene_dir)
    manifest_path = tmp_path / "manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id", "scene_dir"])
        writer.writeheader()
        writer.writerow({"scene_id": "tcp_easy_0001", "scene_dir": str(scene_dir)})

    result = evaluate.evaluate_dataset(tmp_path)

    assert len(result["scenes"]) == 1
    assert result["aggregate"]["scene_count"] == 1
    assert result["aggregate"]["n_frames_mean"] == 2.0
