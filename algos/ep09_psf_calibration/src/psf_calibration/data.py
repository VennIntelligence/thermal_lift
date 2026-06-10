"""Data loading helpers for EP09."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import PROJECT_ROOT, bootstrap_project_paths

bootstrap_project_paths()

from common.alignment import load_alignment_shifts, load_alignment_table  # noqa: E402
from common.data_loader import highpass_preprocess, load_main_session_frames, load_main_session_metadata  # noqa: E402
from thermal_core.alignment_paths import default_contour_alignment_csv  # noqa: E402
from thermal_core.io import load_frame  # noqa: E402


DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "data_raw" / "infrared_avi"
DEFAULT_FRAME_AUDIT_CSV = PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv"
DEFAULT_ALIGNMENT_CSV = default_contour_alignment_csv(project_root_path=PROJECT_ROOT)
DEFAULT_CONTOUR_SEGMENTS_CSV = PROJECT_ROOT / "output" / "ep04_global_validation" / "inputs" / "contour_segments.csv"
DEFAULT_EP06_HR_CANDIDATES = [
    PROJECT_ROOT / "output" / "ep06_sr_poc" / "map_tv_highpass.npy",
    PROJECT_ROOT / "output" / "ep06_sr_poc" / "map_tv" / "hr_highpass.npy",
    PROJECT_ROOT / "output" / "ep08_inr_sr" / "ep06_patch_baseline" / "hr_image.npy",
]


@dataclass(frozen=True)
class MainSessionInputs:
    """Main-session LR observations and EP05 shifts."""

    frames_raw: np.ndarray
    frames_highpass: np.ndarray
    metadata: pd.DataFrame
    shifts: np.ndarray


def load_main_inputs(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    frame_audit_csv: str | Path = DEFAULT_FRAME_AUDIT_CSV,
    alignment_csv: str | Path = DEFAULT_ALIGNMENT_CSV,
    alignment_method: str = "contour_refined",
    highpass_sigma: float = 5.0,
    workers: int = 1,
    limit: int | None = None,
) -> MainSessionInputs:
    """Load main-session raw/highpass frames and matching alignment shifts."""

    frames, metadata = load_main_session_frames(
        data_dir=data_dir,
        frame_audit_path=frame_audit_csv,
        workers=workers,
        dtype=np.float32,
        limit=limit,
    )
    shifts = load_alignment_shifts(
        method=alignment_method,
        metadata=metadata,
        alignment_csv=alignment_csv,
        strict=True,
    ).astype(np.float32, copy=False)
    highpass = highpass_preprocess(frames, sigma_bg=highpass_sigma, workers=workers).astype(np.float32, copy=False)
    return MainSessionInputs(
        frames_raw=frames.astype(np.float32, copy=False),
        frames_highpass=highpass,
        metadata=metadata.copy(),
        shifts=shifts,
    )


def resolve_ep06_hr(path: str | Path | None = None) -> Path:
    """Resolve the EP06 MAP-TV highpass HR input."""

    if path is not None:
        candidate = Path(path)
        if not candidate.exists():
            raise FileNotFoundError(f"EP06 HR file not found: {candidate}")
        return candidate
    for candidate in DEFAULT_EP06_HR_CANDIDATES:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Could not find an EP06 HR highpass image. Checked: "
        + ", ".join(str(p) for p in DEFAULT_EP06_HR_CANDIDATES)
    )


def load_ep06_hr(path: str | Path | None = None) -> tuple[np.ndarray, Path]:
    """Load EP06 MAP-TV HR highpass image as float32."""

    resolved = resolve_ep06_hr(path)
    image = np.load(resolved).astype(np.float32, copy=False)
    if image.ndim != 2:
        raise ValueError(f"EP06 HR image must be 2D, got shape {image.shape}")
    return image, resolved


def load_segments(path: str | Path = DEFAULT_CONTOUR_SEGMENTS_CSV) -> pd.DataFrame:
    """Load EP04 contour segment anchors."""

    csv_path = Path(path)
    if not csv_path.exists():
        raise FileNotFoundError(f"EP04 contour segment table not found: {csv_path}")
    segments = pd.read_csv(csv_path)
    required = {"segment_id", "x_px", "y_px", "nx", "ny", "tx", "ty", "abs_delta_t_c", "normal_projection"}
    missing = required - set(segments.columns)
    if missing:
        raise ValueError(f"Contour segment table is missing required columns: {sorted(missing)}")
    return segments


def resolve_reference_file(
    *,
    frame_audit_csv: str | Path = DEFAULT_FRAME_AUDIT_CSV,
    alignment_csv: str | Path = DEFAULT_ALIGNMENT_CSV,
    alignment_method: str = "contour_refined",
) -> str:
    """Return the EP05 reference frame filename when available."""

    metadata = load_main_session_metadata(frame_audit_csv)
    csv_path = Path(alignment_csv)
    if csv_path.exists():
        table = pd.read_csv(csv_path)
        refs = table.get("reference_file", pd.Series(dtype=str)).dropna().astype(str)
        if not refs.empty and refs.iloc[0] in set(metadata["file"].astype(str)):
            return str(refs.iloc[0])
    else:
        load_alignment_table(
            method=alignment_method,
            metadata=metadata,
            alignment_csv=alignment_csv,
            strict=False,
        )
    return str(metadata.iloc[len(metadata) // 2]["file"])


def load_reference_frame(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    frame_audit_csv: str | Path = DEFAULT_FRAME_AUDIT_CSV,
    alignment_csv: str | Path = DEFAULT_ALIGNMENT_CSV,
    alignment_method: str = "contour_refined",
    reference_file: str | None = None,
) -> tuple[np.ndarray, str]:
    """Load the raw temperature reference frame for ESF fitting."""

    file_name = reference_file or resolve_reference_file(
        frame_audit_csv=frame_audit_csv,
        alignment_csv=alignment_csv,
        alignment_method=alignment_method,
    )
    path = Path(data_dir) / file_name
    if not path.exists():
        raise FileNotFoundError(f"Reference frame not found: {path}")
    return np.asarray(load_frame(path), dtype=np.float32), file_name
