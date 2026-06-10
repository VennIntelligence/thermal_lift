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
from tcforge.storage import save_scene_compact


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


def test_summarize_and_evaluate_scene_accept_compact_4x_scene(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scenes" / "scene_0042"
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[4:12, 5:15] = 1
    edge = np.zeros_like(mask)
    edge[4, 5:15] = 1
    obs_features = np.zeros((5, 4, 5), dtype=np.float32)
    obs_features[2] = 1.0
    shifts = np.zeros((3, 2), dtype=np.float32)
    metadata = {
        "scene_id": "scene_0042",
        "split": "train",
        "difficulty": "medium",
        "scale": 4,
        "lr_shape": [4, 5],
        "hr_shape": [16, 20],
        "T_bg_c": 21.0,
        "delta_T_c": 2.0,
        "low_freq_amplitude_c": 0.0,
        "low_freq_sigma_px": 6.0,
        "low_freq_seed": 44,
        "n_frames": 3,
        "forward_mode": "physical_block_average",
    }
    save_scene_compact(
        scene_dir,
        hr_mask=mask,
        hr_edge=edge,
        obs_features=obs_features,
        shifts=shifts,
        metadata=metadata,
    )
    np.save(scene_dir / "sr_temperature_4x.npy", 21.0 + 2.0 * mask.astype(np.float32))
    np.save(scene_dir / "sr_mask_4x.npy", mask)

    summary = evaluate.summarize_scene(scene_dir)
    result = evaluate.evaluate_scene(scene_dir)

    assert summary["scene_format"] == "compact_4x"
    assert summary["scale"] == 4
    assert summary["n_frames"] == 3
    assert summary["lr_rows"] == 4
    assert summary["hr_rows"] == 16
    assert summary["obs_features_channels"] == 5
    assert result["sr_temperature_mae_c"] == 0.0
    assert result["sr_mask_iou"] == 1.0


def test_compact_scene_missing_required_file_raises_clear_error(tmp_path: Path) -> None:
    scene_dir = tmp_path / "bad_scene"
    scene_dir.mkdir()
    (scene_dir / "metadata.json").write_text("{}", encoding="utf-8")

    try:
        evaluate.summarize_scene(scene_dir)
    except FileNotFoundError as exc:
        message = str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")

    assert "obs_features_1x.npz" in message


def test_split_half_drizzle_consistency_scene(tmp_path: Path) -> None:
    scene_dir = tmp_path / "scenes" / "scene_split"
    mask = np.zeros((8, 8), dtype=np.uint8)
    edge = np.zeros_like(mask)
    obs_features = np.zeros((5, 2, 2), dtype=np.float32)
    shifts = np.zeros((4, 2), dtype=np.float32)
    metadata = {
        "scene_id": "scene_split",
        "difficulty": "easy",
        "seed": 10,
        "scale": 4,
        "lr_shape": [2, 2],
        "hr_shape": [8, 8],
        "T_bg_c": 21.0,
        "delta_T_c": 2.0,
        "low_freq_amplitude_c": 0.0,
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

    result = evaluate.split_half_drizzle_consistency_scene(scene_dir, kernel="nearest")

    assert result["scene_id"] == "scene_split"
    assert result["split_half_odd_frames"] == 2
    assert result["split_half_even_frames"] == 2
    assert result["split_half_eval_pixels"] > 0
    assert result["split_half_mae_c"] == 0.0
