"""EP08 highpass preprocessing and raw-control offset correction."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter


def _gaussian_kernel1d(sigma: float, *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    radius = int(4.0 * float(sigma) + 0.5)
    x = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    phi = torch.exp(-0.5 / (float(sigma) * float(sigma)) * x.square())
    return phi / phi.sum()


def _pad_for_mode(x: torch.Tensor, padding: tuple[int, int, int, int], mode: str) -> torch.Tensor:
    if mode == "nearest":
        return F.pad(x, padding, mode="replicate")
    if mode == "constant":
        return F.pad(x, padding, mode="constant", value=0.0)
    raise ValueError("mode must be 'nearest' or 'constant'")


def gaussian_filter2d_torch(image: torch.Tensor, sigma: float, *, mode: str = "nearest") -> torch.Tensor:
    """Torch Gaussian filter matching SciPy defaults for EP08 preprocessing."""

    if sigma <= 0:
        return image
    if image.ndim != 2:
        raise ValueError("image must be 2D")
    kernel = _gaussian_kernel1d(float(sigma), dtype=image.dtype, device=image.device)
    radius = kernel.numel() // 2
    x = image[None, None]
    ky = kernel.view(1, 1, -1, 1)
    kx = kernel.view(1, 1, 1, -1)
    x = F.conv2d(_pad_for_mode(x, (0, 0, radius, radius), mode), ky)
    x = F.conv2d(_pad_for_mode(x, (radius, radius, 0, 0), mode), kx)
    return x[0, 0]


def highpass_preprocess_numpy(
    frames: np.ndarray,
    sigma_bg: float = 5.0,
    *,
    workers: int | None = None,
    n_jobs: int | None = None,
    mode: str = "nearest",
) -> np.ndarray:
    """NumPy/SciPy highpass reference mirroring EP06 data_loader.py."""

    del workers, n_jobs
    arr = np.asarray(frames, dtype=np.float32)
    if arr.ndim == 2:
        return (arr - gaussian_filter(arr, sigma=sigma_bg, mode=mode)).astype(np.float32, copy=False)
    if arr.ndim != 3:
        raise ValueError("frames must be 2D or 3D")
    bg = gaussian_filter(arr, sigma=(0.0, sigma_bg, sigma_bg), mode=mode)
    return (arr - bg).astype(np.float32, copy=False)


def highpass_preprocess_torch(
    frames: torch.Tensor,
    sigma_bg: float = 5.0,
    *,
    workers: int | None = None,
    n_jobs: int | None = None,
    mode: str = "nearest",
) -> torch.Tensor:
    """Torch highpass preprocessing for 2D or ``(N, H, W)`` LR frames."""

    del workers, n_jobs
    arr = frames.to(dtype=frames.dtype if frames.is_floating_point() else torch.float32)
    if arr.ndim == 2:
        return arr - gaussian_filter2d_torch(arr, sigma_bg, mode=mode)
    if arr.ndim != 3:
        raise ValueError("frames must be 2D or 3D")
    return torch.stack([frame - gaussian_filter2d_torch(frame, sigma_bg, mode=mode) for frame in arr], dim=0)


def highpass_preprocess(
    frames: np.ndarray | torch.Tensor,
    sigma_bg: float = 5.0,
    *,
    workers: int | None = None,
    n_jobs: int | None = None,
    mode: str = "nearest",
) -> np.ndarray | torch.Tensor:
    """Subtract a Gaussian background to produce signed structure maps."""

    if torch.is_tensor(frames):
        return highpass_preprocess_torch(frames, sigma_bg=sigma_bg, workers=workers, n_jobs=n_jobs, mode=mode)
    return highpass_preprocess_numpy(frames, sigma_bg=sigma_bg, workers=workers, n_jobs=n_jobs, mode=mode)


def offset_correction_numpy(
    frames: np.ndarray,
    *,
    method: Literal["median", "mean"] = "median",
    workers: int | None = None,
    n_jobs: int | None = None,
    return_offsets: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Remove per-frame scalar offsets for raw-control data."""

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
    corrected = corrected.astype(np.float32, copy=False)
    if return_offsets:
        return corrected, np.asarray(offsets)
    return corrected


def offset_correction_torch(
    frames: torch.Tensor,
    *,
    method: Literal["median", "mean"] = "median",
    workers: int | None = None,
    n_jobs: int | None = None,
    return_offsets: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """Torch convenience wrapper for raw-control offset correction."""

    del workers, n_jobs
    arr = frames.to(dtype=frames.dtype if frames.is_floating_point() else torch.float32)
    if arr.ndim == 2:
        reduce_dims = (0, 1)
    elif arr.ndim == 3:
        reduce_dims = (1, 2)
    else:
        raise ValueError("frames must be 2D or 3D")

    if method == "median":
        offsets = torch.nanquantile(arr.reshape(-1), 0.5) if arr.ndim == 2 else torch.stack(
            [torch.nanquantile(frame.reshape(-1), 0.5) for frame in arr], dim=0
        )
    elif method == "mean":
        offsets = torch.nanmean(arr, dim=reduce_dims)
    else:
        raise ValueError("method must be 'median' or 'mean'")

    corrected = arr - offsets if arr.ndim == 2 else arr - offsets[:, None, None]
    if return_offsets:
        return corrected, offsets
    return corrected


def offset_correction(
    frames: np.ndarray | torch.Tensor,
    *,
    method: Literal["median", "mean"] = "median",
    workers: int | None = None,
    n_jobs: int | None = None,
    return_offsets: bool = False,
) -> np.ndarray | torch.Tensor | tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
    """Remove per-frame scalar offsets using EP06 median/mean semantics."""

    if torch.is_tensor(frames):
        return offset_correction_torch(
            frames,
            method=method,
            workers=workers,
            n_jobs=n_jobs,
            return_offsets=return_offsets,
        )
    return offset_correction_numpy(
        frames,
        method=method,
        workers=workers,
        n_jobs=n_jobs,
        return_offsets=return_offsets,
    )
