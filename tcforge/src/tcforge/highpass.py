"""Highpass preprocessing matched to EP06."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
from scipy.ndimage import gaussian_filter

from ._utils import resolve_workers


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

    n_workers = min(resolve_workers(workers, n_jobs), arr.shape[0])
    if n_workers == 1:
        bg = gaussian_filter(arr, sigma=(0.0, sigma_bg, sigma_bg), mode=mode)
        return (arr - bg).astype(np.float32, copy=False)

    def process(frame: np.ndarray) -> np.ndarray:
        return frame - gaussian_filter(frame, sigma=sigma_bg, mode=mode)

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        result = list(executor.map(process, arr))
    return np.stack(result, axis=0).astype(np.float32, copy=False)
