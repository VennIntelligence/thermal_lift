"""MAP-TGV reconstruction for EP10.

This module deliberately reuses EP06's frame validation, initialization and
matrix-free forward/adjoint helpers. The FISTA outer loop mirrors
``map_tv.reconstruct_map_tv``; the proximal operator is replaced by a CCPi-RGL
TGV denoiser. The default data-gradient path is algebraically equivalent to
EP06 but caches the per-iteration Gaussian blur to avoid per-frame repetition.
"""

from __future__ import annotations

import sys
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import pandas as pd
from scipy import ndimage


PROJECT_ROOT = Path(__file__).resolve().parents[4]
EP06_SRC = PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src"
CORE_SRC = PROJECT_ROOT / "core" / "src"
for _path in (str(CORE_SRC), str(EP06_SRC)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from common.forward_model import _sample_reference_to_lr, _scatter_lr_to_reference, _sigma_hr  # noqa: E402
from ibp.ibp import _as_frames, _as_shifts, _initial_image, _resolve_workers  # noqa: E402
from map_tv.map_tv import _data_gradient_and_loss, tv_norm  # noqa: E402


_CCPI_TGV: Callable[..., np.ndarray] | None = None
_CCPI_IMPORT_ERROR: Exception | None = None
_RUNTIME_FALLBACK_WARNED = False
_LAST_TGV_BACKEND_PROVENANCE: dict[str, object] = {
    "backend": "not_run",
    "status": "not_run",
    "requested_device": None,
    "selected_device": None,
    "candidate_devices": [],
    "error": None,
}


def _record_tgv_backend(**updates: object) -> None:
    _LAST_TGV_BACKEND_PROVENANCE.update(updates)


def get_tgv_backend_provenance() -> dict[str, object]:
    """Return backend provenance from the most recent ``tgv_denoise`` call."""

    return dict(_LAST_TGV_BACKEND_PROVENANCE)


def _load_ccpi_tgv() -> Callable[..., np.ndarray] | None:
    global _CCPI_TGV, _CCPI_IMPORT_ERROR
    if _CCPI_TGV is not None:
        return _CCPI_TGV
    if _CCPI_IMPORT_ERROR is not None:
        return None
    try:
        from ccpi.filters.regularisers import TGV as tgv_regulariser
    except Exception as exc:  # pragma: no cover - exercised only without CCPi.
        _CCPI_IMPORT_ERROR = exc
        return None
    _CCPI_TGV = tgv_regulariser
    return _CCPI_TGV


def _candidate_tgv_devices(device: str | int | None) -> list[str | int]:
    if device is None:
        device = "auto"
    if isinstance(device, str):
        text = device.strip().lower()
        if text in {"auto", ""}:
            try:
                from ccpi.filters.utils import cilregcuda  # type: ignore
            except Exception:
                cilregcuda = None
            return [0, "cpu"] if cilregcuda is not None else ["cpu"]
        if text == "cpu":
            return ["cpu"]
        if text == "gpu":
            return ["gpu", "cpu"]
        if text.isdigit():
            return [int(text), "cpu"]
        raise ValueError("tgv device must be 'auto', 'cpu', 'gpu', or a GPU index")
    return [int(device), "cpu"]


def _forward_gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grad_y = np.zeros_like(image, dtype=np.float64)
    grad_x = np.zeros_like(image, dtype=np.float64)
    grad_y[:-1, :] = image[1:, :] - image[:-1, :]
    grad_x[:, :-1] = image[:, 1:] - image[:, :-1]
    return grad_y, grad_x


def _ranges(n_items: int, n_chunks: int) -> Iterable[tuple[int, int]]:
    edges = np.linspace(0, n_items, num=n_chunks + 1, dtype=int)
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        if start < stop:
            yield int(start), int(stop)


def _cached_gradient_chunk(
    blurred_hr: np.ndarray,
    frames: np.ndarray,
    shifts: np.ndarray,
    *,
    hr_shape: tuple[int, int],
    scale: int,
) -> tuple[np.ndarray, float]:
    scatter_sum = np.zeros(hr_shape, dtype=np.float64)
    sse = 0.0
    for frame, shift in zip(frames, shifts, strict=True):
        pred = _sample_reference_to_lr(blurred_hr, shift, scale=scale)
        residual = np.where(np.isfinite(frame), pred - frame, 0.0)
        scatter_sum += _scatter_lr_to_reference(
            residual,
            shift,
            hr_shape=hr_shape,
            scale=scale,
        )
        sse += float(np.sum(residual * residual))
    return scatter_sum, sse


def _data_gradient_and_loss_cached(
    x_hr: np.ndarray,
    frames: np.ndarray,
    shifts: np.ndarray,
    *,
    psf_sigma: float,
    scale: int,
    workers: int,
) -> tuple[np.ndarray, float]:
    sigma = _sigma_hr(psf_sigma, scale)
    x = np.asarray(x_hr, dtype=np.float64)
    blurred = ndimage.gaussian_filter(x, sigma=sigma, mode="constant", cval=0.0) if sigma > 0 else x
    workers = min(max(1, workers), frames.shape[0])
    if workers == 1:
        scatter_sum, sse = _cached_gradient_chunk(
            blurred,
            frames,
            shifts,
            hr_shape=x.shape,
            scale=scale,
        )
    else:
        scatter_sum = np.zeros_like(x, dtype=np.float64)
        sse = 0.0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _cached_gradient_chunk,
                    blurred,
                    frames[start:stop],
                    shifts[start:stop],
                    hr_shape=x.shape,
                    scale=scale,
                )
                for start, stop in _ranges(frames.shape[0], workers)
            ]
            for future in futures:
                chunk_scatter, chunk_sse = future.result()
                scatter_sum += chunk_scatter
                sse += chunk_sse

    grad = ndimage.gaussian_filter(scatter_sum, sigma=sigma, mode="constant", cval=0.0) if sigma > 0 else scatter_sum
    grad /= frames.shape[0]
    residual_mse = sse / float(frames.size)
    return grad, residual_mse


def _gradient_adjoint(grad_y: np.ndarray, grad_x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(grad_y, dtype=np.float64)
    out[:-1, :] -= grad_y[:-1, :]
    out[1:, :] += grad_y[:-1, :]
    out[:, :-1] -= grad_x[:, :-1]
    out[:, 1:] += grad_x[:, :-1]
    return out


def _sym_gradient(v_y: np.ndarray, v_x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    e_yy = np.zeros_like(v_y, dtype=np.float64)
    e_xx = np.zeros_like(v_x, dtype=np.float64)
    d_x_vy = np.zeros_like(v_y, dtype=np.float64)
    d_y_vx = np.zeros_like(v_x, dtype=np.float64)

    e_yy[:-1, :] = v_y[1:, :] - v_y[:-1, :]
    e_xx[:, :-1] = v_x[:, 1:] - v_x[:, :-1]
    d_x_vy[:, :-1] = v_y[:, 1:] - v_y[:, :-1]
    d_y_vx[:-1, :] = v_x[1:, :] - v_x[:-1, :]
    e_yx = 0.5 * (d_x_vy + d_y_vx)
    return e_yy, e_xx, e_yx


def _sym_gradient_adjoint(
    q_yy: np.ndarray,
    q_xx: np.ndarray,
    q_yx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # Adjoint of (d_y v_y, d_x v_x, 0.5 * (d_x v_y + d_y v_x)).
    adj_y = _gradient_adjoint(q_yy, 0.5 * q_yx)
    adj_x = _gradient_adjoint(0.5 * q_yx, q_xx)
    return adj_y, adj_x


def _project_vector_ball(
    a: np.ndarray,
    b: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray]:
    norm = np.maximum(1.0, np.sqrt(a * a + b * b) / max(radius, 1e-12))
    return a / norm, b / norm


def _project_vector_ball_aniso(
    a: np.ndarray,
    b: np.ndarray,
    radius_a: float,
    radius_b: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Anisotropic dual projection for first-order TGV term.

    Projects onto the ellipsoidal ball {(a, b) : (a/r_a)^2 + (b/r_b)^2 <= 1}.
    """
    ra = max(radius_a, 1e-12)
    rb = max(radius_b, 1e-12)
    norm = np.maximum(1.0, np.sqrt((a / ra) ** 2 + (b / rb) ** 2))
    return a / norm, b / norm


def _project_sym_ball(
    q_yy: np.ndarray,
    q_xx: np.ndarray,
    q_yx: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    norm = np.maximum(1.0, np.sqrt(q_yy * q_yy + q_xx * q_xx + 2.0 * q_yx * q_yx) / max(radius, 1e-12))
    return q_yy / norm, q_xx / norm, q_yx / norm


def _project_sym_ball_aniso(
    q_yy: np.ndarray,
    q_xx: np.ndarray,
    q_yx: np.ndarray,
    radius_yy: float,
    radius_xx: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Anisotropic dual projection for second-order TGV term.

    Projects onto the anisotropic ball with per-component radii derived from
    directional alpha0 weights: r_yy = alpha0_y, r_xx = alpha0_x,
    r_yx = sqrt(alpha0_y * alpha0_x).
    """
    r_yy = max(radius_yy, 1e-12)
    r_xx = max(radius_xx, 1e-12)
    r_yx = max(np.sqrt(radius_yy * radius_xx), 1e-12)
    norm = np.maximum(
        1.0,
        np.sqrt((q_yy / r_yy) ** 2 + (q_xx / r_xx) ** 2 + 2.0 * (q_yx / r_yx) ** 2),
    )
    return q_yy / norm, q_xx / norm, q_yx / norm


def _tgv_denoise_fallback(
    image: np.ndarray,
    weight: float,
    *,
    alpha_ratio: float,
    max_iter: int,
    tol: float = 1e-5,
    aniso_ratio_y: float = 1.0,
) -> np.ndarray:
    """Chambolle-Pock TGV denoiser with optional anisotropic regularization.

    When ``aniso_ratio_y > 1``, the Y-direction regularization radii are
    scaled up, allowing larger gradients in Y before penalization.  This
    suppresses the horizontal stripe artifacts caused by raster-scan
    anisotropy (X-gap=1 vs Y-gap≈16).
    """

    f = np.asarray(image, dtype=np.float64)
    if weight <= 0:
        return f.copy()

    alpha1 = float(weight)
    alpha0 = float(weight) / float(alpha_ratio)
    aniso_ratio_y = float(aniso_ratio_y)
    use_aniso = abs(aniso_ratio_y - 1.0) > 1e-8

    u = f.copy()
    u_bar = u.copy()
    v_y = np.zeros_like(f, dtype=np.float64)
    v_x = np.zeros_like(f, dtype=np.float64)
    v_y_bar = v_y.copy()
    v_x_bar = v_x.copy()
    p_y = np.zeros_like(f, dtype=np.float64)
    p_x = np.zeros_like(f, dtype=np.float64)
    q_yy = np.zeros_like(f, dtype=np.float64)
    q_xx = np.zeros_like(f, dtype=np.float64)
    q_yx = np.zeros_like(f, dtype=np.float64)

    tau = 0.20
    sigma = 0.20
    theta = 1.0
    for _ in range(max(1, int(max_iter))):
        grad_y, grad_x = _forward_gradient(u_bar)
        p_y += sigma * (grad_y - v_y_bar)
        p_x += sigma * (grad_x - v_x_bar)
        if use_aniso:
            p_y, p_x = _project_vector_ball_aniso(
                p_y, p_x, alpha1 * aniso_ratio_y, alpha1,
            )
        else:
            p_y, p_x = _project_vector_ball(p_y, p_x, alpha1)

        e_yy, e_xx, e_yx = _sym_gradient(v_y_bar, v_x_bar)
        q_yy += sigma * e_yy
        q_xx += sigma * e_xx
        q_yx += sigma * e_yx
        if use_aniso:
            q_yy, q_xx, q_yx = _project_sym_ball_aniso(
                q_yy, q_xx, q_yx,
                alpha0 * aniso_ratio_y, alpha0,
            )
        else:
            q_yy, q_xx, q_yx = _project_sym_ball(q_yy, q_xx, q_yx, alpha0)

        u_old = u
        v_y_old = v_y
        v_x_old = v_x

        kt_u = _gradient_adjoint(p_y, p_x)
        u = (u - tau * kt_u + tau * f) / (1.0 + tau)

        adj_y, adj_x = _sym_gradient_adjoint(q_yy, q_xx, q_yx)
        v_y = v_y + tau * p_y - tau * adj_y
        v_x = v_x + tau * p_x - tau * adj_x

        u_bar = u + theta * (u - u_old)
        v_y_bar = v_y + theta * (v_y - v_y_old)
        v_x_bar = v_x + theta * (v_x - v_x_old)

        rel = float(np.linalg.norm(u - u_old) / max(np.linalg.norm(u_old), 1e-12))
        if rel < tol:
            break
    return u


def tgv_denoise(
    image: np.ndarray,
    weight: float,
    *,
    alpha_ratio: float = 2.0,
    max_iter: int = 80,
    device: str | int | None = "auto",
    aniso_ratio_y: float = 1.0,
) -> np.ndarray:
    """TGV proximal operator using CCPi-RGL's official CPU/CUDA backend.

    The EP10 contract maps ``weight`` directly to CCPi's TGV alpha parameters:
    ``alpha1 = weight`` and ``alpha0 = weight / alpha_ratio``. CCPi's low-level
    function also exposes ``lambdaPar``; this wrapper keeps it at 1.0 so the
    requested alpha mapping is the only regularisation strength control.

    When ``aniso_ratio_y != 1.0``, CCPi is bypassed and the local
    Chambolle-Pock solver is used with anisotropic directional weights.
    """

    requested_device = "auto" if device is None else device
    aniso_ratio_y = float(aniso_ratio_y)
    use_aniso = abs(aniso_ratio_y - 1.0) > 1e-8
    _record_tgv_backend(
        backend="not_run",
        status="started",
        requested_device=str(requested_device),
        selected_device=None,
        candidate_devices=[],
        error=None,
    )

    f = np.asarray(image, dtype=np.float64)
    if f.ndim != 2:
        raise ValueError("image must be a 2D array")
    weight = float(weight)
    if weight <= 0:
        _record_tgv_backend(backend="none", status="skipped_nonpositive_weight")
        return f.copy()
    alpha_ratio = float(alpha_ratio)
    if alpha_ratio <= 0:
        raise ValueError("alpha_ratio must be > 0")

    # When anisotropic mode is requested, CCPi does not support directional
    # weights — go directly to the local Chambolle-Pock fallback.
    if use_aniso:
        _record_tgv_backend(
            backend="fallback",
            status="aniso_forced_fallback",
            selected_device="local_chambolle_pock",
            error=None,
        )
        return _tgv_denoise_fallback(
            f, weight, alpha_ratio=alpha_ratio, max_iter=max_iter,
            aniso_ratio_y=aniso_ratio_y,
        )

    alpha1 = weight
    alpha0 = weight / alpha_ratio
    tgv_regulariser = _load_ccpi_tgv()
    if tgv_regulariser is not None:
        input_f32 = np.ascontiguousarray(f.astype(np.float32, copy=False))
        out = np.empty_like(input_f32)
        last_exc: Exception | None = None
        candidates = _candidate_tgv_devices(device)
        _record_tgv_backend(candidate_devices=[str(candidate) for candidate in candidates])
        for candidate in candidates:
            try:
                result = tgv_regulariser(
                    input_f32,
                    1.0,
                    float(alpha1),
                    float(alpha0),
                    int(max_iter),
                    12.0,
                    1e-6,
                    out=out,
                    device=candidate,
                )
                _record_tgv_backend(
                    backend="ccpi",
                    status="success",
                    selected_device=str(candidate),
                    error=None,
                )
                return np.asarray(result, dtype=np.float64)
            except Exception as exc:  # pragma: no cover - defensive runtime fallback.
                last_exc = exc
                continue

        global _RUNTIME_FALLBACK_WARNED
        if not _RUNTIME_FALLBACK_WARNED:
            warnings.warn(
                f"CCPi TGV failed at runtime; falling back to local Chambolle-Pock TGV: {last_exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            _RUNTIME_FALLBACK_WARNED = True
        _record_tgv_backend(
            backend="fallback",
            status="ccpi_runtime_failed",
            selected_device="local_chambolle_pock",
            error=str(last_exc),
        )
    else:
        _record_tgv_backend(
            backend="fallback",
            status="ccpi_import_failed",
            selected_device="local_chambolle_pock",
            error=str(_CCPI_IMPORT_ERROR) if _CCPI_IMPORT_ERROR is not None else None,
        )

    return _tgv_denoise_fallback(f, weight, alpha_ratio=alpha_ratio, max_iter=max_iter)


def _compute_coverage_map(
    shifts: np.ndarray,
    *,
    lr_shape: tuple[int, int],
    hr_shape: tuple[int, int],
    scale: int,
    psf_sigma: float,
) -> np.ndarray:
    """Precompute per-HR-pixel coverage from all frame shifts.

    Coverage counts how many frames contribute to each HR pixel via bilinear
    splatting + PSF adjoint.  Used to normalize the data gradient per-pixel
    instead of dividing uniformly by frame count, which avoids overfitting
    low-coverage regions.
    """
    ones_lr = np.ones(lr_shape, dtype=np.float64)
    coverage = np.zeros(hr_shape, dtype=np.float64)
    sigma = _sigma_hr(psf_sigma, scale)
    for shift in shifts:
        scattered = _scatter_lr_to_reference(
            ones_lr, shift, hr_shape=hr_shape, scale=scale,
        )
        coverage += scattered
    if sigma > 0:
        coverage = ndimage.gaussian_filter(coverage, sigma=sigma, mode="constant", cval=0.0)
    return coverage


def _data_gradient_and_loss_coverage(
    x_hr: np.ndarray,
    frames: np.ndarray,
    shifts: np.ndarray,
    *,
    psf_sigma: float,
    scale: int,
    workers: int,
    coverage_map: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Data gradient with per-pixel coverage normalization.

    Instead of dividing the scatter-sum gradient by ``N_frames`` uniformly,
    each HR pixel's gradient is divided by its coverage count.  This prevents
    low-coverage regions (common along Y-direction due to raster acquisition
    anisotropy) from being overfitted by sparse frame contributions.
    """
    sigma = _sigma_hr(psf_sigma, scale)
    x = np.asarray(x_hr, dtype=np.float64)
    blurred = ndimage.gaussian_filter(x, sigma=sigma, mode="constant", cval=0.0) if sigma > 0 else x
    workers = min(max(1, workers), frames.shape[0])
    if workers == 1:
        scatter_sum, sse = _cached_gradient_chunk(
            blurred, frames, shifts, hr_shape=x.shape, scale=scale,
        )
    else:
        scatter_sum = np.zeros_like(x, dtype=np.float64)
        sse = 0.0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _cached_gradient_chunk,
                    blurred,
                    frames[start:stop],
                    shifts[start:stop],
                    hr_shape=x.shape,
                    scale=scale,
                )
                for start, stop in _ranges(frames.shape[0], workers)
            ]
            for future in futures:
                chunk_scatter, chunk_sse = future.result()
                scatter_sum += chunk_scatter
                sse += chunk_sse

    grad = ndimage.gaussian_filter(scatter_sum, sigma=sigma, mode="constant", cval=0.0) if sigma > 0 else scatter_sum
    # Per-pixel normalization: divide by coverage instead of uniform N_frames.
    safe_coverage = np.maximum(coverage_map, 1.0)
    grad /= safe_coverage
    residual_mse = sse / float(frames.size)
    return grad, residual_mse


def reconstruct_map_tgv(
    frames: np.ndarray,
    shifts: np.ndarray,
    *,
    initial: str | np.ndarray = "saa",
    lambda_tv: float = 1e-3,
    alpha_ratio: float = 2.0,
    max_iter: int = 100,
    step_size: float = 1.0,
    psf_sigma: float = 1.0,
    scale: int = 2,
    workers: int | None = None,
    n_jobs: int | None = None,
    tol: float = 1e-4,
    tgv_inner_iter: int = 80,
    use_fista: bool = True,
    cached_gradient: bool = True,
    tgv_device: str | int | None = "auto",
    return_dataframe: bool = False,
    aniso_ratio_y: float = 1.0,
    coverage_weighted: bool = False,
) -> tuple[np.ndarray, list[dict[str, object]] | pd.DataFrame]:
    """Run MAP-TGV reconstruction with EP06's FISTA outer loop.

    Parameters
    ----------
    aniso_ratio_y : float
        Scale factor for Y-direction TGV regularization radii.  Values > 1
        allow larger Y-gradients before penalty, suppressing horizontal
        stripe artifacts from raster-scan anisotropy.  Default 1.0 (isotropic).
    coverage_weighted : bool
        When True, the data-fidelity gradient is normalized per-pixel by the
        coverage map instead of dividing uniformly by frame count.  This
        avoids overfitting low-coverage HR pixels.  Default False.
    """

    if scale != 2:
        raise ValueError("EP10 MAP-TGV is defined for scale=2 only")

    frames_arr = _as_frames(frames)
    shifts_arr = _as_shifts(shifts, frames_arr.shape[0])
    n_workers = _resolve_workers(workers, n_jobs)

    x = _initial_image(frames_arr, shifts_arr, initial, scale=scale, workers=n_workers)
    z = x.copy()
    t = 1.0

    lambda_tv = float(lambda_tv)
    alpha_ratio = float(alpha_ratio)
    aniso_ratio_y = float(aniso_ratio_y)
    coverage_weighted = bool(coverage_weighted)
    step_size = float(step_size)
    records: list[dict[str, object]] = []

    # Select gradient function.
    if coverage_weighted:
        lr_shape = (frames_arr.shape[1], frames_arr.shape[2])
        hr_shape = x.shape
        coverage_map = _compute_coverage_map(
            shifts_arr,
            lr_shape=lr_shape,
            hr_shape=hr_shape,
            scale=scale,
            psf_sigma=psf_sigma,
        )

        def _gradient_fn(
            z_hr: np.ndarray,
            frms: np.ndarray,
            shfts: np.ndarray,
            *,
            psf_sigma: float,
            scale: int,
            workers: int,
        ) -> tuple[np.ndarray, float]:
            return _data_gradient_and_loss_coverage(
                z_hr, frms, shfts,
                psf_sigma=psf_sigma, scale=scale, workers=workers,
                coverage_map=coverage_map,
            )

        gradient_fn = _gradient_fn
    else:
        gradient_fn = _data_gradient_and_loss_cached if cached_gradient else _data_gradient_and_loss

    for iteration in range(1, max(0, int(max_iter)) + 1):
        grad, residual_mse = gradient_fn(
            z,
            frames_arr,
            shifts_arr,
            psf_sigma=psf_sigma,
            scale=scale,
            workers=n_workers,
        )

        x_temp = z - step_size * grad
        x_new = tgv_denoise(
            x_temp,
            weight=lambda_tv * step_size,
            alpha_ratio=alpha_ratio,
            max_iter=tgv_inner_iter,
            device=tgv_device,
            aniso_ratio_y=aniso_ratio_y,
        )
        backend_info = get_tgv_backend_provenance()

        denom = float(np.linalg.norm(x))
        rel_update = float(np.linalg.norm(x_new - x) / max(denom, 1e-12))
        tv_proxy = tv_norm(x_new) / float(x_new.size)
        objective_proxy = 0.5 * float(residual_mse) + lambda_tv * tv_proxy
        stopped = rel_update < tol

        records.append(
            {
                "iteration": iteration,
                "residual_mse": float(residual_mse),
                "tv_proxy": float(tv_proxy),
                "objective_proxy": float(objective_proxy),
                "relative_update": rel_update,
                "lambda_tv": lambda_tv,
                "alpha_ratio": alpha_ratio,
                "aniso_ratio_y": aniso_ratio_y,
                "coverage_weighted": coverage_weighted,
                "alpha1": float(lambda_tv * step_size),
                "alpha0": float(lambda_tv * step_size / alpha_ratio),
                "step_size": step_size,
                "psf_sigma": float(psf_sigma),
                "cached_gradient": bool(cached_gradient),
                "tgv_device": str(tgv_device),
                "tgv_backend": str(backend_info["backend"]),
                "tgv_backend_status": str(backend_info["status"]),
                "tgv_backend_device": str(backend_info["selected_device"]),
                "tgv_backend_error": str(backend_info["error"]) if backend_info["error"] is not None else "",
                "stopped": bool(stopped),
            }
        )

        if stopped:
            x = x_new
            break

        if use_fista:
            t_new = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * t * t))
            z = x_new + ((t - 1.0) / t_new) * (x_new - x)
            t = float(t_new)
        else:
            z = x_new
        x = x_new

    if return_dataframe:
        return x, pd.DataFrame.from_records(records)
    return x, records
