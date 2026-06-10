from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ep08 import data as data_mod


def test_load_real_dataset_center_crop_and_raw_control_track(monkeypatch) -> None:
    frames = np.arange(5 * 6 * 8, dtype=np.float32).reshape(5, 6, 8)
    metadata = pd.DataFrame(
        {
            "file": [f"frame_{idx}.txt" for idx in range(5)],
            "acquisition_order": np.arange(5),
        }
    )

    def fake_load_main_session_frames(**kwargs):
        assert kwargs["limit"] is None
        assert kwargs["workers"] == 2
        return frames, metadata.copy()

    def fake_load_alignment_shifts(**kwargs):
        assert kwargs["method"] == "contour_refined"
        assert len(kwargs["metadata"]) == 5
        return np.array(
            [[0.0, 0.0], [0.25, -0.5], [0.5, 0.25], [1.0, 1.0], [-1.0, -1.0]],
            dtype=np.float32,
        )

    monkeypatch.setattr(data_mod, "load_main_session_frames", fake_load_main_session_frames)
    monkeypatch.setattr(data_mod, "load_alignment_shifts", fake_load_alignment_shifts)

    bundle = data_mod.load_real_dataset(
        n_frames=3,
        patch_size=4,
        workers=2,
        alignment_method="contour_refined",
        highpass_sigma=0.0,
        track="raw_control",
    )

    assert bundle.observations.shape == (3, 4, 4)
    assert bundle.raw_control.shape == (3, 4, 4)
    assert bundle.highpass.shape == (3, 4, 4)
    assert torch.equal(bundle.observations, bundle.raw_control)
    assert bundle.shifts.shape == (3, 2)
    assert bundle.metadata["crop"] == {"top": 1, "left": 2, "height": 4, "width": 4}
    assert bundle.metadata["lr_shape"] == (4, 4)
    assert np.allclose(np.median(bundle.raw_control.numpy(), axis=(1, 2)), np.zeros(3))


def test_load_real_dataset_default_highpass_track_and_metadata(monkeypatch) -> None:
    frames = np.ones((2, 5, 7), dtype=np.float32)
    metadata = pd.DataFrame({"file": ["a.txt", "b.txt"], "acquisition_order": [10, 11]})
    calls: dict[str, object] = {}

    def fake_load_main_session_frames(**kwargs):
        calls["limit"] = kwargs["limit"]
        return frames, metadata.copy()

    def fake_load_alignment_shifts(**kwargs):
        calls["method"] = kwargs["method"]
        return np.zeros((2, 2), dtype=np.float32)

    def fake_highpass(input_frames, sigma_bg, workers=None, mode="nearest"):
        calls["sigma_bg"] = sigma_bg
        calls["mode"] = mode
        return input_frames + 2.0

    def fake_offset(input_frames, workers=None):
        return input_frames - 2.0

    monkeypatch.setattr(data_mod, "load_main_session_frames", fake_load_main_session_frames)
    monkeypatch.setattr(data_mod, "load_alignment_shifts", fake_load_alignment_shifts)
    monkeypatch.setattr(data_mod, "highpass_preprocess", fake_highpass)
    monkeypatch.setattr(data_mod, "offset_correction", fake_offset)

    bundle = data_mod.load_real_dataset()

    assert calls["limit"] is None
    assert calls["method"] == "contour_refined"
    assert calls["sigma_bg"] == 5.0
    assert calls["mode"] == "nearest"
    assert bundle.metadata["track"] == "highpass"
    assert bundle.metadata["n_frames"] == 2
    assert torch.equal(bundle.observations, bundle.highpass)
    assert torch.all(bundle.observations == 3.0)


def test_load_real_dataset_rejects_invalid_inputs() -> None:
    try:
        data_mod.load_real_dataset(n_frames=0)
    except ValueError as exc:
        assert "n_frames" in str(exc)
    else:
        raise AssertionError("expected ValueError")

    try:
        data_mod.load_real_dataset(track="stage_command")  # type: ignore[arg-type]
    except ValueError as exc:
        assert "track" in str(exc)
    else:
        raise AssertionError("expected ValueError")
