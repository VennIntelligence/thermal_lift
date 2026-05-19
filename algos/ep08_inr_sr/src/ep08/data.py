"""Real EP08 data loading for main-session INR SR experiments."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch

from ep08.highpass import highpass_preprocess, offset_correction

EP08_ROOT = Path(__file__).resolve().parents[2]
ALGOS_ROOT = EP08_ROOT.parent
EP06_SRC = ALGOS_ROOT / "ep06_sr_poc" / "src"
if str(EP06_SRC) not in sys.path:
    sys.path.insert(0, str(EP06_SRC))

from common.alignment import load_alignment_shifts  # noqa: E402
from common.data_loader import (  # noqa: E402
    DEFAULT_DATA_DIR,
    DEFAULT_FRAME_AUDIT_PATH,
    load_main_session_frames,
)

TrackName = Literal["highpass", "raw_control"]


@dataclass(slots=True)
class RealDataBundle:
    """Tensor tracks and compact metadata for EP08 real-data training."""

    observations: torch.Tensor
    shifts: torch.Tensor
    highpass: torch.Tensor
    raw_control: torch.Tensor
    metadata: dict[str, Any]


def _normalize_patch_size(patch_size: int | tuple[int, int] | None) -> tuple[int, int] | None:
    if patch_size is None:
        return None
    if isinstance(patch_size, tuple):
        if len(patch_size) != 2:
            raise ValueError("patch_size tuple must be (height, width)")
        patch = tuple(int(v) for v in patch_size)
    else:
        value = int(patch_size)
        patch = (value, value)
    if patch[0] <= 0 or patch[1] <= 0:
        raise ValueError("patch_size must be positive")
    return patch


def _center_crop(frames: np.ndarray, patch_size: int | tuple[int, int] | None) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    patch = _normalize_patch_size(patch_size)
    if patch is None:
        height, width = int(frames.shape[-2]), int(frames.shape[-1])
        return frames, (0, 0, height, width)

    patch_h, patch_w = patch
    height, width = int(frames.shape[-2]), int(frames.shape[-1])
    if patch_h > height or patch_w > width:
        raise ValueError(f"patch_size {patch} exceeds frame shape {(height, width)}")
    top = (height - patch_h) // 2
    left = (width - patch_w) // 2
    return frames[:, top : top + patch_h, left : left + patch_w], (top, left, patch_h, patch_w)


def _as_tensor(array: np.ndarray, device: str | torch.device | None = None) -> torch.Tensor:
    kwargs: dict[str, Any] = {"dtype": torch.float32}
    if device is not None:
        kwargs["device"] = torch.device(device)
    return torch.as_tensor(np.array(array, dtype=np.float32, copy=True), **kwargs)


def load_real_dataset(
    *,
    n_frames: int | None = None,
    patch_size: int | tuple[int, int] | None = None,
    workers: int | None = 1,
    alignment_method: str = "contour_refined",
    data_dir: str | Path | None = None,
    frame_audit_path: str | Path | None = None,
    highpass_sigma: float = 5.0,
    highpass_mode: str = "nearest",
    track: TrackName = "highpass",
    device: str | torch.device | None = None,
) -> RealDataBundle:
    """Load main-session frames, EP05 shifts, and EP08 tensor tracks.

    The returned shifts come from EP05 alignment products through EP06 common
    loaders. Stage command positions are intentionally not used as targets.
    """

    if track not in {"highpass", "raw_control"}:
        raise ValueError("track must be 'highpass' or 'raw_control'")
    if n_frames is not None and int(n_frames) <= 0:
        raise ValueError("n_frames must be positive when provided")

    frames, frame_metadata = load_main_session_frames(
        data_dir=data_dir,
        frame_audit_path=frame_audit_path,
        workers=workers,
        dtype=np.float32,
        limit=None if n_frames is None else int(n_frames),
    )
    shifts_np = load_alignment_shifts(
        method=alignment_method,
        metadata=frame_metadata,
        frame_audit_path=frame_audit_path,
    ).astype(np.float32, copy=False)
    if shifts_np.shape != (frames.shape[0], 2):
        raise ValueError(f"alignment shifts shape {shifts_np.shape} does not match {frames.shape[0]} frames")

    cropped, crop = _center_crop(np.asarray(frames, dtype=np.float32), patch_size)
    highpass_np = highpass_preprocess(
        cropped,
        sigma_bg=float(highpass_sigma),
        workers=workers,
        mode=highpass_mode,
    )
    raw_control_np = offset_correction(cropped, workers=workers)

    highpass_t = _as_tensor(np.asarray(highpass_np, dtype=np.float32), device)
    raw_control_t = _as_tensor(np.asarray(raw_control_np, dtype=np.float32), device)
    observations = highpass_t if track == "highpass" else raw_control_t
    shifts = _as_tensor(shifts_np, device)

    data_root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    audit_path = Path(frame_audit_path) if frame_audit_path is not None else DEFAULT_FRAME_AUDIT_PATH
    meta: dict[str, Any] = {
        "data_mode": "real",
        "track": track,
        "n_frames": int(cropped.shape[0]),
        "requested_n_frames": None if n_frames is None else int(n_frames),
        "lr_shape": tuple(int(v) for v in cropped.shape[-2:]),
        "full_lr_shape": tuple(int(v) for v in frames.shape[-2:]),
        "crop": {"top": int(crop[0]), "left": int(crop[1]), "height": int(crop[2]), "width": int(crop[3])},
        "alignment_method": str(alignment_method),
        "highpass_sigma": float(highpass_sigma),
        "highpass_mode": str(highpass_mode),
        "data_dir": str(data_root),
        "frame_audit_path": str(audit_path),
        "frame_files": frame_metadata["file"].astype(str).head(5).tolist() if "file" in frame_metadata else [],
        "shift_source": "EP05 alignment via EP06 common.alignment.load_alignment_shifts",
    }
    return RealDataBundle(
        observations=observations,
        shifts=shifts,
        highpass=highpass_t,
        raw_control=raw_control_t,
        metadata=meta,
    )


__all__ = ["RealDataBundle", "load_real_dataset"]
