"""Classical multi-frame super-resolution via shift-and-add (drizzle).

Places LR pixel values onto a finer HR grid using known sub-pixel shifts,
then normalises by coverage.  This is the simplest classical SR baseline —
no deconvolution, no iterative optimisation — but it correctly combines
multi-frame sub-pixel information and respects the physical forward model.

Shift convention follows TCForge standard: ``shifts[i] = [dx, dy]`` in
**LR pixel** units, representing the alignment offset that maps observed
frame *i* onto the reference frame coordinate system.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

DRIZZLE_CH_MEAN: int = 0
DRIZZLE_CH_COVERAGE: int = 1
DRIZZLE_CH_VARIANCE: int = 2
DRIZZLE_N_CHANNELS: int = 3


def _validate_burst_and_shifts(lr_burst: np.ndarray, shifts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    frames = np.asarray(lr_burst, dtype=np.float32)
    if frames.ndim != 3:
        raise ValueError("lr_burst must have shape (N, H_lr, W_lr)")
    if frames.shape[0] <= 0:
        raise ValueError("lr_burst must contain at least one frame")
    if not np.isfinite(frames).all():
        raise ValueError("lr_burst contains NaN or Inf")

    shift_arr = np.asarray(shifts, dtype=np.float32)
    if shift_arr.shape != (frames.shape[0], 2):
        raise ValueError("shifts must have shape (N, 2) matching lr_burst")
    if not np.isfinite(shift_arr).all():
        raise ValueError("shifts contain NaN or Inf")
    return frames, shift_arr


def _resolve_hr_shape(
    h_lr: int,
    w_lr: int,
    scale: int,
    output_shape: tuple[int, int] | None,
) -> tuple[int, int]:
    scale = int(scale)
    if scale <= 0:
        raise ValueError("scale must be positive")
    if output_shape is None:
        return h_lr * scale, w_lr * scale
    h_hr, w_hr = int(output_shape[0]), int(output_shape[1])
    if h_hr <= 0 or w_hr <= 0:
        raise ValueError("output_shape entries must be positive")
    return h_hr, w_hr


def drizzle_features(
    lr_burst: np.ndarray,
    shifts: np.ndarray,
    *,
    scale: int = 4,
    output_shape: tuple[int, int] | None = None,
    kernel: str = "bilinear",
) -> np.ndarray:
    """Splat an LR burst onto an HR grid as direct-observation features.

    The output has shape ``(3, H_hr, W_hr)`` with channels:

    * mean temperature at each observed HR bin
    * normalized coverage, ``scatter_weight / n_frames``
    * weighted per-bin observation variance

    This is a forward scatter-add implementation. It avoids resampling every
    LR frame onto the full HR grid with ``map_coordinates``, which is the hot
    path for 4x feature building. Shift convention matches ``shift_and_add``:
    ``shifts[i] = [dx, dy]`` maps observed frame coordinates into the
    reference coordinate system.
    """

    frames, shift_arr = _validate_burst_and_shifts(lr_burst, shifts)
    n_frames, h_lr, w_lr = frames.shape
    scale = int(scale)
    h_hr, w_hr = _resolve_hr_shape(h_lr, w_lr, scale, output_shape)
    kernel_name = str(kernel).lower()
    if kernel_name not in {"nearest", "bilinear"}:
        raise ValueError("kernel must be 'nearest' or 'bilinear'")

    yy, xx = np.mgrid[:h_lr, :w_lr]
    yy_scaled = (yy.ravel() * scale).astype(np.int32, copy=False)
    xx_scaled = (xx.ravel() * scale).astype(np.int32, copy=False)
    out_size = h_hr * w_hr
    weight_flat = np.zeros(out_size, dtype=np.float64)
    sum_flat = np.zeros(out_size, dtype=np.float64)
    sumsq_flat = np.zeros(out_size, dtype=np.float64)

    # Use chunked processing to reduce np.bincount calls and avoid massive memory allocations
    chunk_size = 32
    for chunk_start in range(0, n_frames, chunk_size):
        chunk_idx = []
        chunk_values = []
        chunk_weights = []

        for idx in range(chunk_start, min(chunk_start + chunk_size, n_frames)):
            frame = frames[idx]
            dx, dy = float(shift_arr[idx, 0]), float(shift_arr[idx, 1])
            frame_flat = frame.ravel()

            y_shift = dy * scale
            x_shift = dx * scale

            if kernel_name == "nearest":
                y_hr = yy_scaled + np.rint(y_shift).astype(np.int32)
                x_hr = xx_scaled + np.rint(x_shift).astype(np.int32)
                valid = (y_hr >= 0) & (y_hr < h_hr) & (x_hr >= 0) & (x_hr < w_hr)
                if np.any(valid):
                    flat_idx = y_hr[valid] * w_hr + x_hr[valid]
                    chunk_idx.append(flat_idx)
                    chunk_values.append(frame_flat[valid])
                    chunk_weights.append(np.ones(flat_idx.shape, dtype=np.float32))
            else: # bilinear
                y0_shift = np.floor(y_shift)
                x0_shift = np.floor(x_shift)
                fy = float(y_shift - y0_shift)
                fx = float(x_shift - x0_shift)

                y0 = yy_scaled + int(y0_shift)
                x0 = xx_scaled + int(x0_shift)

                for oy, wy in ((0, 1.0 - fy), (1, fy)):
                    if wy <= 0.0:
                        continue
                    yy_c = y0 + oy
                    valid_y = (yy_c >= 0) & (yy_c < h_hr)

                    for ox, wx in ((0, 1.0 - fx), (1, fx)):
                        w = wy * wx
                        if w <= 0.0:
                            continue
                        xx_c = x0 + ox
                        valid = valid_y & (xx_c >= 0) & (xx_c < w_hr)
                        if not np.any(valid):
                            continue

                        flat_idx = yy_c[valid] * w_hr + xx_c[valid]
                        chunk_idx.append(flat_idx)
                        chunk_values.append(frame_flat[valid])
                        chunk_weights.append(np.full(flat_idx.shape, w, dtype=np.float32))

        if chunk_idx:
            c_idx = np.concatenate(chunk_idx)
            c_vals = np.concatenate(chunk_values)
            c_wgts = np.concatenate(chunk_weights)

            weight_flat += np.bincount(c_idx, weights=c_wgts, minlength=out_size)
            c_vals_w = c_vals * c_wgts
            sum_flat += np.bincount(c_idx, weights=c_vals_w, minlength=out_size)
            sumsq_flat += np.bincount(c_idx, weights=c_vals_w * c_vals, minlength=out_size)

    observed = weight_flat > 0.0
    safe_weight = np.maximum(weight_flat, 1.0)
    mean_flat = sum_flat / safe_weight
    if np.any(observed):
        global_mean = float(sum_flat[observed].sum() / max(weight_flat[observed].sum(), 1e-12))
    else:
        global_mean = float(np.mean(frames))
    mean_flat = np.where(observed, mean_flat, global_mean)
    variance_flat = np.maximum(sumsq_flat / safe_weight - mean_flat * mean_flat, 0.0)
    variance_flat = np.where(observed, variance_flat, 0.0)
    coverage_flat = np.clip(weight_flat / float(n_frames), 0.0, 1.0)

    features = np.stack(
        [
            mean_flat.reshape(h_hr, w_hr),
            coverage_flat.reshape(h_hr, w_hr),
            variance_flat.reshape(h_hr, w_hr),
        ],
        axis=0,
    )
    return np.where(np.isfinite(features), features, 0.0).astype(np.float32, copy=False)


def phase_bin_drizzle(
    lr_burst: np.ndarray,
    shifts: np.ndarray,
    *,
    scale: int = 2,
    n_bins: int = 4,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Drizzle the burst into per-sub-pixel-phase bins.

    Routes each frame to one of ``n_bins`` cells on a ``g × g`` grid
    (``g = sqrt(n_bins)``) by its sub-pixel phase ``(frac(dy), frac(dx))``,
    then runs the standard mean-channel drizzle on each bin's frames. Returns
    ``(n_bins, H_hr, W_hr)`` float32. Empty bins are filled with the global
    burst mean, matching ``drizzle_features``' unobserved-bin convention.

    ``n_bins`` must be a perfect square.
    """

    frames, shift_arr = _validate_burst_and_shifts(lr_burst, shifts)
    n_frames, h_lr, w_lr = frames.shape
    scale = int(scale)
    n_bins = int(n_bins)
    if n_bins <= 0:
        raise ValueError("n_bins must be > 0")
    g = int(round(np.sqrt(n_bins)))
    if g * g != n_bins:
        raise ValueError("n_bins must be a perfect square")
    h_hr, w_hr = _resolve_hr_shape(h_lr, w_lr, scale, output_shape)

    # Route each frame to a phase bin: (frac(dy), frac(dx)) → (row, col) → flat.
    frac_dy = np.mod(shift_arr[:, 1], 1.0)
    frac_dx = np.mod(shift_arr[:, 0], 1.0)
    row = np.clip((frac_dy * g).astype(np.int64), 0, g - 1)
    col = np.clip((frac_dx * g).astype(np.int64), 0, g - 1)
    bin_idx = row * g + col

    global_mean = float(np.mean(frames))
    out = np.empty((n_bins, h_hr, w_hr), dtype=np.float32)
    for b in range(n_bins):
        sel = bin_idx == b
        if np.any(sel):
            mean_channel = drizzle_features(
                frames[sel], shift_arr[sel], scale=scale, output_shape=(h_hr, w_hr),
            )[DRIZZLE_CH_MEAN]
            out[b] = mean_channel
        else:
            out[b] = global_mean
    return out


def drizzle_features_4x(
    lr_burst: np.ndarray,
    shifts: np.ndarray,
    *,
    output_shape: tuple[int, int] | None = None,
    kernel: str = "bilinear",
) -> np.ndarray:
    """Convenience wrapper for 4x drizzle features."""

    return drizzle_features(lr_burst, shifts, scale=4, output_shape=output_shape, kernel=kernel)


def shift_and_add(
    lr_burst: np.ndarray,
    shifts: np.ndarray,
    *,
    scale: int = 2,
    output_shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Classical shift-and-add SR from an LR burst.

    Parameters
    ----------
    lr_burst : (N, H_lr, W_lr) float array
        Multi-frame LR observations.
    shifts : (N, 2) float array
        Per-frame ``[dx, dy]`` alignment shifts in LR pixel units.
    scale : int
        Upsampling factor (output HR grid is ``scale × LR``).
    output_shape : (H_hr, W_hr) or None
        HR output shape.  Defaults to ``(H_lr * scale, W_lr * scale)``.

    Returns
    -------
    hr_image : (H_hr, W_hr) float32 array
        Shift-and-add HR reconstruction.  Pixels with zero coverage are
        filled with the global coverage-weighted mean.
    """

    frames, shift_arr = _validate_burst_and_shifts(lr_burst, shifts)
    n_frames, h_lr, w_lr = frames.shape

    scale = int(scale)
    h_hr, w_hr = _resolve_hr_shape(h_lr, w_lr, scale, output_shape)

    # Build HR coordinate grids
    yy_hr = np.arange(h_hr, dtype=np.float32)
    xx_hr = np.arange(w_hr, dtype=np.float32)
    yy, xx = np.meshgrid(yy_hr, xx_hr, indexing="ij")

    accumulator = np.zeros((h_hr, w_hr), dtype=np.float64)
    weight = np.zeros((h_hr, w_hr), dtype=np.float64)

    for idx in range(n_frames):
        dx, dy = float(shift_arr[idx, 0]), float(shift_arr[idx, 1])

        # Map HR grid coords back to this frame's LR pixel coords:
        # HR coord → reference LR coord → observed LR coord
        src_y = yy / scale - dy
        src_x = xx / scale - dx

        # Validity mask: source coords must be within the LR frame
        valid = (
            (src_y >= 0.0)
            & (src_y <= h_lr - 1)
            & (src_x >= 0.0)
            & (src_x <= w_lr - 1)
        )

        # Bilinear interpolation from LR frame onto HR grid
        interpolated = ndimage.map_coordinates(
            frames[idx],
            (src_y, src_x),
            order=1,
            mode="constant",
            cval=0.0,
            prefilter=False,
        )

        accumulator += np.where(valid, interpolated, 0.0)
        weight += valid.astype(np.float64)

    # Normalise by coverage
    safe_weight = np.maximum(weight, 1.0)
    hr_image = (accumulator / safe_weight).astype(np.float32)

    # Fill zero-coverage pixels with global mean
    if np.any(weight == 0):
        global_mean = float(accumulator.sum() / max(safe_weight.sum(), 1.0))
        hr_image = np.where(weight > 0, hr_image, global_mean).astype(np.float32)

    return hr_image
