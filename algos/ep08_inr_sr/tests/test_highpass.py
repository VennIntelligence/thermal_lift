from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

EP06_SRC = Path(__file__).resolve().parents[2] / "ep06_sr_poc" / "src"
if str(EP06_SRC) not in sys.path:
    sys.path.insert(0, str(EP06_SRC))

from common import data_loader as ep06_data
from ep08.highpass import highpass_preprocess, offset_correction


def test_highpass_torch_matches_ep06_reference_for_stack() -> None:
    rng = np.random.default_rng(5)
    frames = rng.normal(size=(4, 17, 19)).astype(np.float32)

    expected = ep06_data.highpass_preprocess(frames, sigma_bg=2.0, mode="nearest")
    actual = highpass_preprocess(torch.as_tensor(frames), sigma_bg=2.0, mode="nearest").detach().cpu().numpy()

    assert np.max(np.abs(actual - expected)) < 1e-5


def test_highpass_numpy_matches_ep06_reference_for_single_frame() -> None:
    rng = np.random.default_rng(8)
    frame = rng.normal(size=(21, 23)).astype(np.float32)

    expected = ep06_data.highpass_preprocess(frame, sigma_bg=3.0, mode="nearest")
    actual = highpass_preprocess(frame, sigma_bg=3.0, mode="nearest")

    assert np.max(np.abs(actual - expected)) < 1e-5


def test_offset_correction_matches_ep06_reference_with_offsets() -> None:
    rng = np.random.default_rng(10)
    frames = rng.normal(size=(3, 8, 9)).astype(np.float32)

    expected, expected_offsets = ep06_data.offset_correction(frames, method="median", return_offsets=True)
    actual, actual_offsets = offset_correction(frames, method="median", return_offsets=True)

    assert np.max(np.abs(actual - expected)) < 1e-5
    assert np.max(np.abs(actual_offsets - expected_offsets)) < 1e-5
