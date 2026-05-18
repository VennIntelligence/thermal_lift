"""MAP reconstruction with total-variation regularization for EP06.

This module uses the same shift convention as ``ibp``: shifts move LR frames
into the reference coordinate system; the forward model predicts raw frames by
using the inverse scene displacement, and the adjoint back-projects residuals
with ``+shift``.

No scikit-image dependency is used. The TV proximal step is a small
Chambolle-Pock/Chambolle-style projection solver for isotropic TV denoising.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Iterable

import numpy as np
import pandas as pd

from ibp.ibp import (
    _as_frames,
    _as_shifts,
    _initial_image,
    _resolve_workers,
    adjoint,
    forward,
)


def _ranges(n_items: int, n_chunks: int) -> Iterable[tuple[int, int]]:
    edges = np.linspace(0, n_items, num=n_chunks + 1, dtype=int)
    for start, stop in zip(edges[:-1], edges[1:], strict=True):
        if start < stop:
            yield int(start), int(stop)


def _forward_gradient(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grad_y = np.zeros_like(image, dtype=np.float64)
    grad_x = np.zeros_like(image, dtype=np.float64)
    grad_y[:-1, :] = image[1:, :] - image[:-1, :]
    grad_x[:, :-1] = image[:, 1:] - image[:, :-1]
    return grad_y, grad_x


def _gradient_adjoint(grad_y: np.ndarray, grad_x: np.ndarray) -> np.ndarray:
    out = np.zeros_like(grad_y, dtype=np.float64)

    out[:-1, :] -= grad_y[:-1, :]
    out[1:, :] += grad_y[:-1, :]
    out[:, :-1] -= grad_x[:, :-1]
    out[:, 1:] += grad_x[:, :-1]

    return out


def tv_norm(image: np.ndarray, eps: float = 1e-8) -> float:
    """Isotropic TV norm used for convergence reporting."""

    grad_y, grad_x = _forward_gradient(np.asarray(image, dtype=np.float64))
    return float(np.sum(np.sqrt(grad_y * grad_y + grad_x * grad_x + eps * eps)))


def tv_denoise_chambolle(
    image: np.ndarray,
    weight: float,
    *,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> np.ndarray:
    """Proximal operator for ``weight * TV`` without scikit-image.

    Solves ``min_u 0.5 * ||u - image||_2^2 + weight * TV(u)`` with a
    first-order primal-dual projection iteration.
    """

    f = np.asarray(image, dtype=np.float64)
    weight = float(weight)
    if weight <= 0:
        return f.copy()

    u = f.copy()
    u_bar = u.copy()
    p_y = np.zeros_like(f, dtype=np.float64)
    p_x = np.zeros_like(f, dtype=np.float64)

    tau = 0.25
    sigma = 0.25
    theta = 1.0

    for _ in range(max(1, int(max_iter))):
        grad_y, grad_x = _forward_gradient(u_bar)
        p_y += sigma * grad_y
        p_x += sigma * grad_x

        norm = np.maximum(1.0, np.sqrt(p_y * p_y + p_x * p_x) / weight)
        p_y /= norm
        p_x /= norm

        u_old = u
        kt_p = _gradient_adjoint(p_y, p_x)
        u = (u - tau * kt_p + tau * f) / (1.0 + tau)
        u_bar = u + theta * (u - u_old)

        rel = float(np.linalg.norm(u - u_old) / max(np.linalg.norm(u_old), 1e-12))
        if rel < tol:
            break

    return u


def _gradient_chunk(
    x_hr: np.ndarray,
    frames: np.ndarray,
    shifts: np.ndarray,
    *,
    psf_sigma: float,
    scale: int,
) -> tuple[np.ndarray, float]:
    grad = np.zeros_like(x_hr, dtype=np.float64)
    sse = 0.0
    for frame, shift in zip(frames, shifts, strict=True):
        pred = forward(x_hr, shift, psf_sigma=psf_sigma, scale=scale)
        residual = np.where(np.isfinite(frame), pred - frame, 0.0)
        grad += adjoint(
            residual,
            shift,
            psf_sigma=psf_sigma,
            hr_shape=x_hr.shape,
            scale=scale,
        )
        sse += float(np.sum(residual * residual))
    return grad, sse


def _data_gradient_and_loss(
    x_hr: np.ndarray,
    frames: np.ndarray,
    shifts: np.ndarray,
    *,
    psf_sigma: float,
    scale: int,
    workers: int,
) -> tuple[np.ndarray, float]:
    workers = min(max(1, workers), frames.shape[0])
    if workers == 1:
        grad, sse = _gradient_chunk(
            x_hr,
            frames,
            shifts,
            psf_sigma=psf_sigma,
            scale=scale,
        )
    else:
        grad = np.zeros_like(x_hr, dtype=np.float64)
        sse = 0.0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(
                    _gradient_chunk,
                    x_hr,
                    frames[start:stop],
                    shifts[start:stop],
                    psf_sigma=psf_sigma,
                    scale=scale,
                )
                for start, stop in _ranges(frames.shape[0], workers)
            ]
            for future in futures:
                chunk_grad, chunk_sse = future.result()
                grad += chunk_grad
                sse += chunk_sse

    grad /= frames.shape[0]
    residual_mse = sse / float(frames.size)
    return grad, residual_mse


def reconstruct_map_tv(
    frames: np.ndarray,
    shifts: np.ndarray,
    *,
    initial: str | np.ndarray = "saa",
    lambda_tv: float = 1e-3,
    max_iter: int = 100,
    step_size: float = 1.0,
    psf_sigma: float = 1.0,
    scale: int = 2,
    workers: int | None = None,
    n_jobs: int | None = None,
    tol: float = 1e-4,
    tv_inner_iter: int = 30,
    use_fista: bool = True,
    return_dataframe: bool = False,
) -> tuple[np.ndarray, list[dict[str, float | int | bool]] | pd.DataFrame]:
    """Run MAP-TV reconstruction.

    Returns ``(image, records)`` by default. Set ``return_dataframe=True`` to
    receive a pandas ``DataFrame``.
    """

    if scale != 2:
        raise ValueError("EP06 MAP-TV is defined for scale=2 only")

    frames_arr = _as_frames(frames)
    shifts_arr = _as_shifts(shifts, frames_arr.shape[0])
    n_workers = _resolve_workers(workers, n_jobs)

    x = _initial_image(frames_arr, shifts_arr, initial, scale=scale, workers=n_workers)
    z = x.copy()
    t = 1.0

    lambda_tv = float(lambda_tv)
    step_size = float(step_size)
    records: list[dict[str, float | int | bool]] = []

    for iteration in range(1, max(0, int(max_iter)) + 1):
        grad, residual_mse = _data_gradient_and_loss(
            z,
            frames_arr,
            shifts_arr,
            psf_sigma=psf_sigma,
            scale=scale,
            workers=n_workers,
        )

        x_temp = z - step_size * grad
        x_new = tv_denoise_chambolle(
            x_temp,
            weight=lambda_tv * step_size,
            max_iter=tv_inner_iter,
        )

        denom = float(np.linalg.norm(x))
        rel_update = float(np.linalg.norm(x_new - x) / max(denom, 1e-12))
        tv_value = tv_norm(x_new) / float(x_new.size)
        objective = 0.5 * float(residual_mse) + lambda_tv * tv_value
        stopped = rel_update < tol

        records.append(
            {
                "iteration": iteration,
                "residual_mse": float(residual_mse),
                "tv": float(tv_value),
                "objective": float(objective),
                "relative_update": rel_update,
                "lambda_tv": lambda_tv,
                "step_size": step_size,
                "psf_sigma": float(psf_sigma),
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


map_tv_reconstruct = reconstruct_map_tv
reconstruct = reconstruct_map_tv
