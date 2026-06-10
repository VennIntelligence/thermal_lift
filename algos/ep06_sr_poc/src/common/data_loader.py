"""Data loading and preprocessing utilities for EP06 SR experiments."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, zoom


PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_DATA_DIR = PROJECT_ROOT / "data" / "data_raw" / "infrared_avi"
DEFAULT_FRAME_AUDIT_PATH = PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv"
DEFAULT_CLEAN_SR_FRAME_COUNT = 248


def _boolish(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(int).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes", "y", "t"})


def _resolve_workers(workers: int | None = None, n_jobs: int | None = None) -> int:
    value = workers if workers is not None else n_jobs
    if value is None:
        return 1
    value = int(value)
    if value < 1:
        raise ValueError("workers/n_jobs must be >= 1")
    return value


def load_main_session_metadata(
    frame_audit_path: str | Path | None = None,
    *,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load EP01 main-session metadata sorted by acquisition order."""

    audit_path = Path(frame_audit_path) if frame_audit_path is not None else DEFAULT_FRAME_AUDIT_PATH
    if not audit_path.exists():
        raise FileNotFoundError(f"Frame audit CSV not found: {audit_path}")
    audit = pd.read_csv(audit_path)
    required = {"file", "acquisition_order"}
    missing = required - set(audit.columns)
    if missing:
        raise ValueError(f"Frame audit CSV is missing required columns: {sorted(missing)}")

    if "is_sr_usable" in audit.columns:
        metadata = audit[_boolish(audit["is_sr_usable"])].copy()
    elif "is_main_session" in audit.columns:
        metadata = audit[_boolish(audit["is_main_session"])].copy()
    elif "session" in audit.columns:
        main_session = 2 if audit["session"].eq(2).any() else audit.groupby("session")["file"].count().idxmax()
        metadata = audit[audit["session"].eq(main_session)].copy()
    else:
        raise ValueError("Frame audit must contain is_sr_usable, is_main_session, or session")

    metadata["acquisition_order"] = pd.to_numeric(metadata["acquisition_order"], errors="coerce")
    metadata = metadata.sort_values(["acquisition_order", "file"]).reset_index(drop=True)
    if limit is not None:
        metadata = metadata.iloc[:limit].copy()
    if "frame_index" in metadata.columns:
        metadata = metadata.drop(columns=["frame_index"])
    metadata.insert(0, "frame_index", np.arange(len(metadata), dtype=int))
    return metadata


def _load_single_frame(path: Path, dtype: np.dtype | str | None) -> np.ndarray:
    from thermal_core.io import load_frame

    arr = np.asarray(load_frame(path))
    if dtype is not None:
        arr = arr.astype(dtype, copy=False)
    return arr


def load_main_session_frames(
    data_dir: str | Path | None = None,
    frame_audit_path: str | Path | None = None,
    *,
    frame_audit_csv: str | Path | None = None,
    workers: int | None = None,
    n_jobs: int | None = None,
    dtype: np.dtype | str | None = np.float32,
    limit: int | None = None,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Load the clean main SR TXT input as ``(N, H, W)`` plus metadata."""

    audit_path = frame_audit_path if frame_audit_path is not None else frame_audit_csv
    metadata = load_main_session_metadata(audit_path, limit=limit)
    root = Path(data_dir) if data_dir is not None else DEFAULT_DATA_DIR
    if not root.exists():
        raise FileNotFoundError(f"Raw frame directory not found: {root}")

    paths = [root / str(name) for name in metadata["file"]]
    missing = [path for path in paths if not path.exists()]
    if missing:
        preview = ", ".join(str(path) for path in missing[:5])
        suffix = "" if len(missing) <= 5 else f", ... ({len(missing)} total)"
        raise FileNotFoundError(f"Missing TXT frame files: {preview}{suffix}")

    n_workers = _resolve_workers(workers, n_jobs)
    if n_workers == 1:
        frames_list = [_load_single_frame(path, dtype) for path in paths]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            frames_list = list(executor.map(lambda path: _load_single_frame(path, dtype), paths))

    shapes = {frame.shape for frame in frames_list}
    if len(shapes) != 1:
        raise ValueError(f"Main-session frames have inconsistent shapes: {sorted(shapes)}")
    return np.stack(frames_list, axis=0), metadata


def highpass_preprocess(
    frames: np.ndarray,
    sigma_bg: float = 5.0,
    *,
    workers: int | None = None,
    n_jobs: int | None = None,
    mode: str = "nearest",
) -> np.ndarray:
    """Subtract a Gaussian background from each frame to produce structure maps."""

    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 2:
        return arr - gaussian_filter(arr, sigma=sigma_bg, mode=mode)
    if arr.ndim != 3:
        raise ValueError("frames must be 2D or 3D")

    n_workers = min(_resolve_workers(workers, n_jobs), arr.shape[0])
    if n_workers == 1:
        bg = gaussian_filter(arr, sigma=(0.0, sigma_bg, sigma_bg), mode=mode)
        return (arr - bg).astype(np.float32, copy=False)

    def process(frame: np.ndarray) -> np.ndarray:
        return frame - gaussian_filter(frame, sigma=sigma_bg, mode=mode)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        result = list(executor.map(process, arr))
    return np.stack(result, axis=0).astype(np.float32, copy=False)


def offset_correction(
    frames: np.ndarray,
    *,
    method: Literal["median", "mean"] = "median",
    workers: int | None = None,
    n_jobs: int | None = None,
    return_offsets: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Remove per-frame scalar offsets for the raw-temperature control track."""

    del workers, n_jobs
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 2:
        axes: tuple[int, ...] = (0, 1)
    elif arr.ndim == 3:
        axes = (1, 2)
    else:
        raise ValueError("frames must be 2D or 3D")

    if method == "median":
        offsets = np.nanmedian(arr, axis=axes)
    elif method == "mean":
        offsets = np.nanmean(arr, axis=axes)
    else:
        raise ValueError("method must be 'median' or 'mean'")

    corrected = arr - float(offsets) if arr.ndim == 2 else arr - offsets[:, None, None]
    if return_offsets:
        return corrected.astype(np.float32, copy=False), np.asarray(offsets)
    return corrected.astype(np.float32, copy=False)


def bicubic_upsample(
    image: np.ndarray,
    *,
    scale: int = 2,
    order: int = 3,
    mode: str = "nearest",
) -> np.ndarray:
    """Upsample a 2D image or ``(N, H, W)`` stack."""

    if scale < 1:
        raise ValueError("scale must be >= 1")
    arr = np.asarray(image, dtype=np.float32)
    if arr.ndim == 2:
        factors = (scale, scale)
    elif arr.ndim == 3:
        factors = (1, scale, scale)
    else:
        raise ValueError("image must be 2D or 3D")
    return zoom(arr, zoom=factors, order=order, mode=mode, prefilter=(order > 1)).astype(np.float32, copy=False)
