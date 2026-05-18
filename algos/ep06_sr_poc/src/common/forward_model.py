"""Matrix-free 2x forward model for EP06 SR.

All shifts use the EP05 convention: ``shift=(dx, dy)`` is the LR-pixel
translation that moves an observed frame into the reference coordinate system.
``forward`` predicts the original raw observation by sampling the reference HR
scene at detector positions plus that alignment shift; ``adjoint`` backprojects
LR residuals with the positive shift into the reference HR grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage


def _validate_scale(scale: int) -> int:
    scale = int(scale)
    if scale != 2:
        raise ValueError("EP06 forward model is defined for scale=2 only")
    return scale


def _sigma_hr(psf_sigma: float, scale: int) -> float:
    sigma = float(psf_sigma)
    return max(0.0, sigma * scale)


def downsample_block_average(image_hr: np.ndarray, *, scale: int = 2) -> np.ndarray:
    """Downsample HR to LR with 2x2 block averaging."""

    scale = _validate_scale(scale)
    img = np.asarray(image_hr, dtype=np.float64)
    rows = img.shape[0] // scale
    cols = img.shape[1] // scale
    cropped = img[: rows * scale, : cols * scale]
    return cropped.reshape(rows, scale, cols, scale).mean(axis=(1, 3))


def upsample_block_adjoint(image_lr: np.ndarray, *, scale: int = 2) -> np.ndarray:
    """Adjoint of block averaging: repeat LR residual with ``1/scale^2`` gain."""

    scale = _validate_scale(scale)
    lr = np.asarray(image_lr, dtype=np.float64)
    return np.repeat(np.repeat(lr, scale, axis=0), scale, axis=1) / float(scale * scale)


def _sample_reference_to_lr(image_hr: np.ndarray, shift: np.ndarray, *, scale: int) -> np.ndarray:
    h_hr, w_hr = image_hr.shape
    h_lr, w_lr = h_hr // scale, w_hr // scale
    dx, dy = np.asarray(shift, dtype=np.float64)
    yy = scale * (np.arange(h_lr, dtype=np.float64) + dy)
    xx = scale * (np.arange(w_lr, dtype=np.float64) + dx)
    coords = np.meshgrid(yy, xx, indexing="ij")
    return ndimage.map_coordinates(image_hr, coords, order=1, mode="constant", cval=0.0, prefilter=False)


def _scatter_lr_to_reference(
    image_lr: np.ndarray,
    shift: np.ndarray,
    *,
    hr_shape: tuple[int, int],
    scale: int,
) -> np.ndarray:
    lr = np.asarray(image_lr, dtype=np.float64)
    h_hr, w_hr = hr_shape
    h_lr, w_lr = lr.shape
    dx, dy = np.asarray(shift, dtype=np.float64)
    y = scale * (np.arange(h_lr, dtype=np.float64)[:, None] + dy)
    x = scale * (np.arange(w_lr, dtype=np.float64)[None, :] + dx)
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    fy = y - y0
    fx = x - x0

    out = np.zeros(hr_shape, dtype=np.float64)
    out_flat = out.ravel()
    finite = np.isfinite(lr)
    clean = np.where(finite, lr, 0.0)
    for oy in (0, 1):
        yy = y0 + oy
        wy = (1.0 - fy) if oy == 0 else fy
        valid_y = (yy >= 0) & (yy < h_hr)
        for ox in (0, 1):
            xx = x0 + ox
            wx = (1.0 - fx) if ox == 0 else fx
            valid_x = (xx >= 0) & (xx < w_hr)
            weight = wy * wx
            mask = valid_y & valid_x & finite
            if np.any(mask):
                idx = yy * w_hr + xx
                np.add.at(out_flat, idx[mask], (clean * weight)[mask])
    return out


def forward(
    x_hr: np.ndarray,
    shift: tuple[float, float] | np.ndarray,
    psf_sigma: float = 1.0,
    *,
    scale: int = 2,
    mode: str = "constant",
) -> np.ndarray:
    """Predict one raw LR observation from a reference HR image."""

    scale = _validate_scale(scale)
    x = np.asarray(x_hr, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("x_hr must be 2D")
    sigma = _sigma_hr(psf_sigma, scale)
    blurred = ndimage.gaussian_filter(x, sigma=sigma, mode=mode, cval=0.0) if sigma > 0 else x
    return _sample_reference_to_lr(blurred, np.asarray(shift, dtype=np.float64), scale=scale)


def adjoint(
    y_residual: np.ndarray,
    shift: tuple[float, float] | np.ndarray,
    psf_sigma: float = 1.0,
    *,
    hr_shape: tuple[int, int] | None = None,
    scale: int = 2,
    mode: str = "constant",
) -> np.ndarray:
    """Backproject one LR residual into the reference HR grid."""

    scale = _validate_scale(scale)
    y = np.asarray(y_residual, dtype=np.float64)
    if y.ndim != 2:
        raise ValueError("y_residual must be 2D")
    if hr_shape is None:
        hr_shape = (y.shape[0] * scale, y.shape[1] * scale)
    scattered = _scatter_lr_to_reference(y, np.asarray(shift, dtype=np.float64), hr_shape=hr_shape, scale=scale)
    sigma = _sigma_hr(psf_sigma, scale)
    return ndimage.gaussian_filter(scattered, sigma=sigma, mode=mode, cval=0.0) if sigma > 0 else scattered


@dataclass(frozen=True)
class ObservationOperator:
    hr_shape: tuple[int, int]
    lr_shape: tuple[int, int]
    shifts: np.ndarray
    psf_sigma: float = 1.0
    scale: int = 2
    mode: str = "constant"

    def forward(self, x_hr: np.ndarray, index: int) -> np.ndarray:
        return forward(x_hr, self.shifts[index], self.psf_sigma, scale=self.scale, mode=self.mode)

    def adjoint(self, y_residual: np.ndarray, index: int) -> np.ndarray:
        return adjoint(
            y_residual,
            self.shifts[index],
            self.psf_sigma,
            hr_shape=self.hr_shape,
            scale=self.scale,
            mode=self.mode,
        )

    def forward_all(self, x_hr: np.ndarray) -> np.ndarray:
        return np.stack([self.forward(x_hr, idx) for idx in range(len(self.shifts))], axis=0)

    def adjoint_sum(self, residuals: np.ndarray, *, average: bool = False) -> np.ndarray:
        arr = np.asarray(residuals)
        if arr.ndim != 3 or len(arr) != len(self.shifts):
            raise ValueError("residuals must have shape (N, H, W) matching shifts")
        total = np.zeros(self.hr_shape, dtype=np.float64)
        for idx, residual in enumerate(arr):
            total += self.adjoint(residual, idx)
        if average and len(arr):
            total /= float(len(arr))
        return total

    def __iter__(self):
        yield self.forward
        yield self.adjoint


def build_observation_operator(
    hr_shape: tuple[int, int],
    lr_shape: tuple[int, int] | None = None,
    shifts: np.ndarray | None = None,
    *,
    psf_sigma: float = 1.0,
    scale: int = 2,
    mode: str = "constant",
) -> ObservationOperator:
    """Create a functional operator without building a dense matrix."""

    scale = _validate_scale(scale)
    hr_shape = tuple(map(int, hr_shape))
    expected_lr = (hr_shape[0] // scale, hr_shape[1] // scale)
    lr_shape = expected_lr if lr_shape is None else tuple(map(int, lr_shape))
    if lr_shape != expected_lr:
        raise ValueError("lr_shape is inconsistent with hr_shape and scale")
    shift_arr = np.empty((0, 2), dtype=float) if shifts is None else np.asarray(shifts, dtype=float)
    if shift_arr.ndim != 2 or shift_arr.shape[1] != 2:
        raise ValueError("shifts must have shape (N, 2)")
    return ObservationOperator(hr_shape, lr_shape, shift_arr, psf_sigma=float(psf_sigma), scale=scale, mode=mode)
