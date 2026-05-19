"""Deterministic train/validation splits for EP08 micro-scan bursts."""

from __future__ import annotations

import numpy as np


def build_train_val_split(
    frames: np.ndarray | object,
    shifts: np.ndarray | object,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build a reproducible split stratified by 4x4 shift subpixel phase bins."""

    n_frames = int(len(frames))
    shift_arr = np.asarray(shifts, dtype=np.float64)
    if shift_arr.ndim != 2 or shift_arr.shape[1] != 2:
        raise ValueError("shifts must have shape (N, 2)")
    if len(shift_arr) != n_frames:
        raise ValueError("frames and shifts must have the same length")
    if not 0.0 <= float(val_ratio) <= 1.0:
        raise ValueError("val_ratio must be in [0, 1]")

    rng = np.random.default_rng(seed)
    phase = np.mod(shift_arr, 1.0)
    bins = np.floor(phase * 4.0).astype(np.int64).clip(0, 3)
    bin_id = bins[:, 1] * 4 + bins[:, 0]

    val_chunks: list[np.ndarray] = []
    for current_bin in range(16):
        members = np.flatnonzero(bin_id == current_bin)
        if len(members) < 3:
            continue
        n_val = int(np.floor(len(members) * float(val_ratio)))
        if n_val < 1 and val_ratio > 0:
            n_val = 1
        if n_val >= len(members):
            n_val = len(members) - 1
        if n_val <= 0:
            continue
        shuffled = members.copy()
        rng.shuffle(shuffled)
        val_chunks.append(np.sort(shuffled[:n_val]))

    val_indices = np.sort(np.concatenate(val_chunks)) if val_chunks else np.empty(0, dtype=np.int64)
    val_mask = np.zeros(n_frames, dtype=bool)
    val_mask[val_indices] = True
    train_indices = np.flatnonzero(~val_mask)
    return train_indices.astype(np.int64), val_indices.astype(np.int64), val_mask
