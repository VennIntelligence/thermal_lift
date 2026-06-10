#!/usr/bin/env python3
"""Run EP15 M4 GPU MAP-TV deconvolution anchor.

M4 is a classical baseline-to-beat for later 4x/5x neural methods.  It uses
the clean 248-frame main session, contour-refined shifts, a detector-aperture
box forward model, and a PSF sigma scan over the M3 credible range.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy import ndimage
from scipy.signal import find_peaks
from scipy.signal.windows import tukey
from tqdm import tqdm


SCRIPT_PATH = Path(__file__).resolve()
ALGO_ROOT = SCRIPT_PATH.parents[1]
PROJECT_ROOT = SCRIPT_PATH.parents[3]
EP06_SRC = PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src"
EP06_SCRIPTS = PROJECT_ROOT / "algos" / "ep06_sr_poc" / "scripts"
EP07_SRC = PROJECT_ROOT / "algos" / "ep07_unet_sr" / "src"
TCFORGE_SRC = PROJECT_ROOT / "tcforge" / "src"

for path in (ALGO_ROOT / "src", EP06_SRC, EP06_SCRIPTS, EP07_SRC, TCFORGE_SRC, PROJECT_ROOT / "core" / "src"):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

from thermal_core.alignment_paths import default_contour_alignment_csv  # noqa: E402

from common.alignment import load_alignment_shifts  # noqa: E402
from common.data_loader import (  # noqa: E402
    bicubic_upsample,
    highpass_preprocess,
    load_main_session_frames,
    load_main_session_metadata,
)
from run_saa import forward_observation  # noqa: E402
from thermal_core.plotting import COLORMAPS, FIGURE_SIZES, get_method_style, savefig_academic, setup_academic_style  # noqa: E402


EXPECTED_CLEAN_SR_FRAMES = 248
PIXEL_SIZE_UM = 10.0
NOISE_FLOOR_C = 0.0724
DEFAULT_PSF_SIGMAS = (0.2, 0.3, 0.4, 0.5)
DEFAULT_LAMBDAS = (3e-4, 1e-3, 3e-3)
DEFAULT_EP07_CHECKPOINT = PROJECT_ROOT / "algos" / "ep07_unet_sr" / "outputs" / "ep07_v6_physics" / "model_final.pt"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "ep15_info_limit" / "m4_deconv_anchor"
ZIGZAG_ROI_FRACTION = 1.0 / 6.0
ZIGZAG_ROI_CENTER_YX = (0.5, 0.5)


@dataclass(frozen=True)
class GridDecision:
    scale: int
    source: str


@dataclass(frozen=True)
class Reconstruction:
    image: np.ndarray
    zero_coverage_pct: float


@dataclass(frozen=True)
class MapTVResult:
    image: np.ndarray
    convergence: pd.DataFrame
    elapsed_sec: float


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _read_json(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _to_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return _rel(value)
    if isinstance(value, np.generic):
        return _to_jsonable(value.item())
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _to_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_jsonable(v) for v in value]
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_to_jsonable(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def parse_float_list(text: str) -> list[float]:
    values = [float(item.strip()) for item in str(text).split(",") if item.strip()]
    if not values:
        raise ValueError("Expected at least one numeric value")
    return values


def load_grid_decision(grid_json: Path, m1_summary_json: Path) -> GridDecision:
    if grid_json.exists():
        data = _read_json(grid_json)
        return GridDecision(scale=int(data.get("grid_scale", data.get("scale", 5))), source=_rel(grid_json))
    if m1_summary_json.exists():
        data = _read_json(m1_summary_json)
        return GridDecision(scale=int(data.get("grid_scale", data.get("scale", 5))), source=_rel(m1_summary_json))
    return GridDecision(scale=5, source="default")


def resolve_device(requested: str) -> torch.device:
    request = str(requested).strip().lower()
    if request.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False")
        device = torch.device(request)
        _ = torch.empty(1, device=device)
        name = torch.cuda.get_device_name(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        print(
            f"Using {device} ({name}); free={free_bytes / 2**30:.2f} GiB / total={total_bytes / 2**30:.2f} GiB"
        )
        return device
    print(f"Using device: {request}")
    return torch.device(request)


def validate_inputs(frames: np.ndarray, metadata: pd.DataFrame, shifts: np.ndarray, *, allow_limit: bool) -> None:
    if not allow_limit and len(metadata) != EXPECTED_CLEAN_SR_FRAMES:
        raise ValueError(f"Expected {EXPECTED_CLEAN_SR_FRAMES} clean SR frames; got {len(metadata)}")
    if frames.ndim != 3:
        raise ValueError(f"Expected frames shape (N,H,W); got {frames.shape}")
    if frames.shape[1:] != (480, 640):
        raise ValueError(f"Expected detector frame shape (480,640); got {frames.shape[1:]}")
    if shifts.shape != (len(metadata), 2):
        raise ValueError(f"Expected shifts shape ({len(metadata)},2); got {shifts.shape}")


def _gaussian_kernel1d(sigma: float, *, device: torch.device, dtype: torch.dtype, truncate: float = 4.0) -> torch.Tensor:
    sigma = float(sigma)
    if sigma <= 0:
        return torch.ones(1, device=device, dtype=dtype)
    radius = max(1, int(math.ceil(truncate * sigma)))
    x = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (x / sigma) ** 2)
    return kernel / kernel.sum()


def _gaussian_blur2d(x: torch.Tensor, kernel_1d: torch.Tensor) -> torch.Tensor:
    if kernel_1d.numel() <= 1:
        return x
    radius = kernel_1d.numel() // 2
    kx = kernel_1d.view(1, 1, 1, -1)
    ky = kernel_1d.view(1, 1, -1, 1)
    out = F.pad(x, (radius, radius, 0, 0), mode="replicate")
    out = F.conv2d(out, kx)
    out = F.pad(out, (0, 0, radius, radius), mode="replicate")
    return F.conv2d(out, ky)


class BatchForwardModel:
    """GPU batch observation model with shifts, Gaussian PSF, and detector box."""

    def __init__(
        self,
        shifts: np.ndarray,
        *,
        scale: int,
        psf_sigma: float,
        device: torch.device | str,
        use_box: bool = True,
        chunk_size: int = 32,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        shifts_arr = np.asarray(shifts, dtype=np.float32)
        if shifts_arr.ndim != 2 or shifts_arr.shape[1] != 2:
            raise ValueError(f"shifts must have shape (N,2); got {shifts_arr.shape}")
        self.shifts = torch.as_tensor(shifts_arr, device=device, dtype=dtype)
        self.scale = int(scale)
        self.psf_sigma = float(psf_sigma)
        self.device = torch.device(device)
        self.use_box = bool(use_box)
        self.chunk_size = max(1, int(chunk_size))
        self.dtype = dtype
        self.kernel = _gaussian_kernel1d(self.psf_sigma * self.scale, device=self.device, dtype=dtype)
        self._base_grid: torch.Tensor | None = None
        self._base_grid_shape: tuple[int, int] | None = None

    @property
    def n_frames(self) -> int:
        return int(self.shifts.shape[0])

    def _base_grid_for(self, rows: int, cols: int) -> torch.Tensor:
        shape = (int(rows), int(cols))
        if self._base_grid is not None and self._base_grid_shape == shape:
            return self._base_grid
        y = torch.linspace(-1.0, 1.0, rows, device=self.device, dtype=self.dtype)
        x = torch.linspace(-1.0, 1.0, cols, device=self.device, dtype=self.dtype)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        self._base_grid = torch.stack((xx, yy), dim=-1).contiguous()
        self._base_grid_shape = shape
        return self._base_grid

    def _grid(self, shifts: torch.Tensor, *, rows: int, cols: int, sign: float) -> torch.Tensor:
        base = self._base_grid_for(rows, cols)
        denom_x = max(cols - 1, 1)
        denom_y = max(rows - 1, 1)
        tx = sign * 2.0 * shifts[:, 0] * self.scale / float(denom_x)
        ty = sign * 2.0 * shifts[:, 1] * self.scale / float(denom_y)
        translation = torch.stack((tx, ty), dim=1).view(-1, 1, 1, 2)
        return base.unsqueeze(0) + translation

    @torch.no_grad()
    def forward(self, hr: torch.Tensor) -> torch.Tensor:
        """Predict all LR observations from one HR image.

        ``hr`` must be ``[1,1,H,W]``.  The returned tensor is ``[N,1,h,w]``.
        """

        if hr.ndim != 4 or hr.shape[0] != 1 or hr.shape[1] != 1:
            raise ValueError(f"hr must have shape [1,1,H,W]; got {tuple(hr.shape)}")
        hr = hr.to(device=self.device, dtype=self.dtype)
        rows, cols = int(hr.shape[-2]), int(hr.shape[-1])
        out: list[torch.Tensor] = []
        for start in range(0, self.n_frames, self.chunk_size):
            stop = min(self.n_frames, start + self.chunk_size)
            shifts = self.shifts[start:stop]
            grid = self._grid(shifts, rows=rows, cols=cols, sign=+1.0)
            shifted = F.grid_sample(
                hr.expand(stop - start, -1, -1, -1),
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            blurred = _gaussian_blur2d(shifted, self.kernel)
            if self.use_box:
                lr = F.avg_pool2d(blurred, kernel_size=self.scale, stride=self.scale)
            else:
                lr = blurred[..., :: self.scale, :: self.scale]
            out.append(lr)
        return torch.cat(out, dim=0)

    @torch.no_grad()
    def adjoint(self, residuals: torch.Tensor) -> torch.Tensor:
        """Backproject LR residuals into one HR gradient image."""

        if residuals.ndim != 4 or residuals.shape[1] != 1:
            raise ValueError(f"residuals must have shape [N,1,h,w]; got {tuple(residuals.shape)}")
        if residuals.shape[0] != self.n_frames:
            raise ValueError(f"Expected {self.n_frames} residual frames; got {residuals.shape[0]}")
        residuals = residuals.to(device=self.device, dtype=self.dtype)
        hr_rows = int(residuals.shape[-2]) * self.scale
        hr_cols = int(residuals.shape[-1]) * self.scale
        total = torch.zeros((1, 1, hr_rows, hr_cols), device=self.device, dtype=self.dtype)
        for start in range(0, self.n_frames, self.chunk_size):
            stop = min(self.n_frames, start + self.chunk_size)
            chunk = residuals[start:stop]
            up = chunk.repeat_interleave(self.scale, dim=-2).repeat_interleave(self.scale, dim=-1)
            if self.use_box:
                up = up / float(self.scale * self.scale)
            blurred = _gaussian_blur2d(up, self.kernel)
            grid = self._grid(self.shifts[start:stop], rows=hr_rows, cols=hr_cols, sign=-1.0)
            back = F.grid_sample(
                blurred,
                grid,
                mode="bilinear",
                padding_mode="border",
                align_corners=True,
            )
            total += back.sum(dim=0, keepdim=True)
        return total


def tv_value_torch(image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    gx = image[..., :, 1:] - image[..., :, :-1]
    gy = image[..., 1:, :] - image[..., :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    return torch.sqrt(gx * gx + gy * gy + eps * eps).sum()


def tv_gradient_torch(image: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    gx = image[..., :, 1:] - image[..., :, :-1]
    gy = image[..., 1:, :] - image[..., :-1, :]
    gx = F.pad(gx, (0, 1, 0, 0))
    gy = F.pad(gy, (0, 0, 0, 1))
    norm = torch.sqrt(gx * gx + gy * gy + eps * eps)
    px = gx / norm
    py = gy / norm
    div = torch.zeros_like(image)
    div[..., :, :-1] += px[..., :, :-1]
    div[..., :, 1:] -= px[..., :, :-1]
    div[..., :-1, :] += py[..., :-1, :]
    div[..., 1:, :] -= py[..., :-1, :]
    return -div


def run_forward_smoke(
    *,
    shifts: np.ndarray,
    scale: int,
    psf_sigma: float,
    device: torch.device,
    use_box: bool,
    chunk_size: int,
    tolerance: float,
) -> dict[str, float | bool]:
    rng = np.random.default_rng(1504)
    truth = rng.normal(0.0, 1.0, size=(96, 128)).astype(np.float32)
    truth = ndimage.gaussian_filter(truth, sigma=1.0, mode="nearest").astype(np.float32)
    sample_shifts = np.asarray(shifts[: min(8, len(shifts))], dtype=np.float32)
    model = BatchForwardModel(
        sample_shifts,
        scale=scale,
        psf_sigma=psf_sigma,
        device=device,
        use_box=use_box,
        chunk_size=chunk_size,
    )
    hr = torch.from_numpy(truth[None, None]).to(device)
    pred = model.forward(hr).detach().cpu().numpy()[:, 0]
    if not use_box:
        return {
            "forward_smoke_max_abs": None,
            "forward_smoke_mean_abs": None,
            "forward_smoke_pass": True,
            "forward_smoke_reference": "skipped_ep06_box_reference_for_no_box_model",
        }

    ref = np.stack([forward_observation(truth, shift, scale=scale, psf_sigma=psf_sigma) for shift in sample_shifts])
    diff = np.abs(pred - ref)
    max_abs = float(np.max(diff))
    mean_abs = float(np.mean(diff))
    ok = bool(max_abs < tolerance)
    if not ok:
        raise AssertionError(f"forward smoke max_abs={max_abs:.6g} exceeds tolerance={tolerance:.6g}")
    return {
        "forward_smoke_max_abs": max_abs,
        "forward_smoke_mean_abs": mean_abs,
        "forward_smoke_pass": ok,
        "forward_smoke_reference": "ep06_run_saa.forward_observation",
    }


def _splat_frame(
    accum: np.ndarray,
    weight_sum: np.ndarray,
    frame: np.ndarray,
    *,
    dx_px: float,
    dy_px: float,
    scale: int,
    y_base: np.ndarray,
    x_base: np.ndarray,
) -> None:
    hr_rows, hr_cols = accum.shape
    y = y_base + float(dy_px) * scale
    x = x_base + float(dx_px) * scale
    y0 = np.floor(y).astype(np.int64)
    x0 = np.floor(x).astype(np.int64)
    fy = (y - y0).astype(np.float32, copy=False)
    fx = (x - x0).astype(np.float32, copy=False)

    for y_idx, wy in ((y0, 1.0 - fy), (y0 + 1, fy)):
        valid_y = (y_idx >= 0) & (y_idx < hr_rows) & (wy > 0.0)
        if not bool(valid_y.any()):
            continue
        yy = y_idx[valid_y]
        wy_valid = wy[valid_y].astype(np.float32, copy=False)
        for x_idx, wx in ((x0, 1.0 - fx), (x0 + 1, fx)):
            valid_x = (x_idx >= 0) & (x_idx < hr_cols) & (wx > 0.0)
            if not bool(valid_x.any()):
                continue
            xx = x_idx[valid_x]
            wx_valid = wx[valid_x].astype(np.float32, copy=False)
            weights = wy_valid[:, None] * wx_valid[None, :]
            values = frame[np.ix_(valid_y, valid_x)].astype(np.float32, copy=False)
            target = np.ix_(yy, xx)
            accum[target] += values * weights
            weight_sum[target] += weights


def bilinear_drizzle(frames: np.ndarray, shifts: np.ndarray, *, scale: int, desc: str) -> Reconstruction:
    if frames.ndim != 3:
        raise ValueError("frames must have shape (N,H,W)")
    if shifts.shape != (frames.shape[0], 2):
        raise ValueError(f"shifts shape {shifts.shape} does not match frames {frames.shape}")
    _, rows, cols = frames.shape
    accum = np.zeros((rows * scale, cols * scale), dtype=np.float32)
    weight_sum = np.zeros_like(accum)
    y_base = np.arange(rows, dtype=np.float64) * scale
    x_base = np.arange(cols, dtype=np.float64) * scale
    for frame, (dx_px, dy_px) in tqdm(zip(frames, shifts, strict=True), total=len(frames), desc=desc):
        _splat_frame(
            accum,
            weight_sum,
            frame,
            dx_px=float(dx_px),
            dy_px=float(dy_px),
            scale=scale,
            y_base=y_base,
            x_base=x_base,
        )
    covered = weight_sum > 1e-6
    image = np.empty_like(accum)
    image[covered] = accum[covered] / weight_sum[covered]
    image[~covered] = float(np.nanmean(frames))
    return Reconstruction(image=image, zero_coverage_pct=100.0 * float(1.0 - covered.mean()))


def gradient_magnitude(image: np.ndarray) -> np.ndarray:
    gx = ndimage.sobel(image, axis=1, mode="nearest")
    gy = ndimage.sobel(image, axis=0, mode="nearest")
    return np.hypot(gx, gy).astype(np.float32)


def artifact_score(image: np.ndarray) -> float:
    arr = np.asarray(image, dtype=np.float32)
    if not np.isfinite(arr).all():
        return float("inf")
    high_freq = arr - ndimage.gaussian_filter(arr, sigma=1.0, mode="nearest")
    lap = ndimage.laplace(arr, mode="nearest")
    base = float(np.std(arr))
    if base <= 1e-12:
        return 0.0
    return float((np.std(high_freq) + 0.25 * np.std(lap)) / base)


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    lhs = np.asarray(a, dtype=np.float32)
    rhs = np.asarray(b, dtype=np.float32)
    denom = float(np.std(lhs) + np.std(rhs))
    return float(np.sqrt(np.mean((lhs - rhs) ** 2)) / max(denom, 1e-6))


def run_map_tv_gpu(
    frames: np.ndarray,
    shifts: np.ndarray,
    init_hr: np.ndarray,
    *,
    scale: int,
    psf_sigma: float,
    lambda_tv: float,
    max_iter: int,
    step_size: float,
    tol: float,
    device: torch.device,
    use_box: bool,
    chunk_size: int,
    track: str,
    progress: bool = True,
) -> MapTVResult:
    start = time.perf_counter()
    model = BatchForwardModel(
        shifts,
        scale=scale,
        psf_sigma=psf_sigma,
        device=device,
        use_box=use_box,
        chunk_size=chunk_size,
    )
    frames_t = torch.from_numpy(np.asarray(frames, dtype=np.float32)[:, None]).to(device)
    x = torch.from_numpy(np.asarray(init_hr, dtype=np.float32)[None, None]).to(device).clone()
    y = x.clone()
    t = 1.0
    rows: list[dict[str, float | int | str | bool]] = []
    prev_objective = float("inf")

    iterator = range(1, int(max_iter) + 1)
    if progress:
        iterator = tqdm(iterator, desc=f"MAP-TV {track} sigma={psf_sigma:g} lambda={lambda_tv:g}")  # type: ignore[assignment]

    for iteration in iterator:
        pred = model.forward(y)
        residual = pred - frames_t
        data_rmse_t = torch.sqrt(torch.mean(residual * residual))
        data_grad = model.adjoint(residual) / float(max(1, len(frames)))
        tv_grad = tv_gradient_torch(y)
        grad = data_grad + float(lambda_tv) * tv_grad
        x_new = y - float(step_size) * grad
        if not bool(torch.isfinite(x_new).all()):
            raise FloatingPointError(f"Non-finite MAP-TV iterate at iteration {iteration}, track={track}")

        tv_t = tv_value_torch(x_new)
        objective_t = 0.5 * data_rmse_t * data_rmse_t + float(lambda_tv) * tv_t / float(x_new.numel())
        objective = float(objective_t.detach().cpu())
        if objective > prev_objective * 1.05 and iteration > 4:
            # Monotone restart keeps the accelerated loop from running away on
            # poorly scaled sigma/lambda combinations.
            y = x.clone()
            t = 1.0
            pred = model.forward(y)
            residual = pred - frames_t
            data_rmse_t = torch.sqrt(torch.mean(residual * residual))
            data_grad = model.adjoint(residual) / float(max(1, len(frames)))
            grad = data_grad + float(lambda_tv) * tv_gradient_torch(y)
            x_new = y - float(step_size) * grad
            tv_t = tv_value_torch(x_new)
            objective_t = 0.5 * data_rmse_t * data_rmse_t + float(lambda_tv) * tv_t / float(x_new.numel())
            objective = float(objective_t.detach().cpu())
            restarted = True
        else:
            restarted = False

        rel_update = float((torch.linalg.vector_norm(x_new - x) / torch.clamp(torch.linalg.vector_norm(x), min=1e-12)).cpu())
        rows.append(
            {
                "track": track,
                "iteration": int(iteration),
                "psf_sigma_lr_px": float(psf_sigma),
                "lambda_tv": float(lambda_tv),
                "data_rmse": float(data_rmse_t.detach().cpu()),
                "tv_value": float(tv_t.detach().cpu()),
                "objective": float(objective),
                "relative_update": rel_update,
                "fista_restarted": restarted,
            }
        )

        t_new = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        y = x_new + ((t - 1.0) / t_new) * (x_new - x)
        x = x_new
        t = t_new
        prev_objective = min(prev_objective, objective)
        if rel_update < tol:
            break

    image = x.detach().cpu().numpy()[0, 0].astype(np.float32, copy=False)
    del frames_t, x, y, model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return MapTVResult(image=image, convergence=pd.DataFrame(rows), elapsed_sec=float(time.perf_counter() - start))


def select_lambda_for_sigma(
    frames: np.ndarray,
    shifts: np.ndarray,
    *,
    scale: int,
    psf_sigma: float,
    lambdas: list[float],
    max_iter: int,
    step_size: float,
    tol: float,
    device: torch.device,
    use_box: bool,
    chunk_size: int,
    artifact_weight: float,
    std_weight: float,
) -> tuple[float, pd.DataFrame]:
    even = np.arange(len(frames)) % 2 == 0
    odd = ~even
    init_a = bilinear_drizzle(frames[even], shifts[even], scale=scale, desc=f"split A init sigma={psf_sigma:g}").image
    init_b = bilinear_drizzle(frames[odd], shifts[odd], scale=scale, desc=f"split B init sigma={psf_sigma:g}").image
    init_std = 0.5 * (float(np.std(init_a)) + float(np.std(init_b)))
    rows: list[dict[str, Any]] = []

    for lambda_tv in lambdas:
        a = run_map_tv_gpu(
            frames[even],
            shifts[even],
            init_a,
            scale=scale,
            psf_sigma=psf_sigma,
            lambda_tv=lambda_tv,
            max_iter=max_iter,
            step_size=step_size,
            tol=tol,
            device=device,
            use_box=use_box,
            chunk_size=chunk_size,
            track=f"select_sigma{psf_sigma:g}_lambda{lambda_tv:g}_a",
        )
        b = run_map_tv_gpu(
            frames[odd],
            shifts[odd],
            init_b,
            scale=scale,
            psf_sigma=psf_sigma,
            lambda_tv=lambda_tv,
            max_iter=max_iter,
            step_size=step_size,
            tol=tol,
            device=device,
            use_box=use_box,
            chunk_size=chunk_size,
            track=f"select_sigma{psf_sigma:g}_lambda{lambda_tv:g}_b",
        )
        consistency = nrmse(a.image, b.image)
        artifacts = 0.5 * (artifact_score(a.image) + artifact_score(b.image))
        mean_grad = 0.5 * (float(np.mean(gradient_magnitude(a.image))) + float(np.mean(gradient_magnitude(b.image))))
        mean_std = 0.5 * (float(np.std(a.image)) + float(np.std(b.image)))
        std_excess = max(0.0, mean_std / max(init_std, 1e-12) - 1.0)
        rows.append(
            {
                "psf_sigma_lr_px": float(psf_sigma),
                "lambda_tv": float(lambda_tv),
                "split_half_nrmse": consistency,
                "artifact_score": artifacts,
                "mean_gradient": mean_grad,
                "std": mean_std,
                "init_drizzle_std": init_std,
                "std_excess_vs_drizzle": std_excess,
                "selection_proxy": consistency + artifact_weight * artifacts + std_weight * std_excess,
                "n_split_a": int(even.sum()),
                "n_split_b": int(odd.sum()),
                "iterations_a": int(len(a.convergence)),
                "iterations_b": int(len(b.convergence)),
                "elapsed_sec_a": a.elapsed_sec,
                "elapsed_sec_b": b.elapsed_sec,
                "selected_within_sigma": False,
                "selected_global": False,
            }
        )
    table = pd.DataFrame(rows)
    best_idx = int(table["selection_proxy"].idxmin())
    table.loc[best_idx, "selected_within_sigma"] = True
    return float(table.loc[best_idx, "lambda_tv"]), table


def crop_bounds_fraction(
    shape: tuple[int, int],
    *,
    fraction: float,
    y_frac: float = 0.5,
    x_frac: float = 0.5,
) -> tuple[int, int, int, int]:
    rows, cols = shape
    crop_rows = max(1, int(round(rows * float(fraction))))
    crop_cols = max(1, int(round(cols * float(fraction))))
    cy = int(round(rows * float(y_frac)))
    cx = int(round(cols * float(x_frac)))
    y0 = min(max(0, cy - crop_rows // 2), rows - crop_rows)
    x0 = min(max(0, cx - crop_cols // 2), cols - crop_cols)
    return y0, y0 + crop_rows, x0, x0 + crop_cols


def center_fraction_crop(image: np.ndarray, fraction: float) -> np.ndarray:
    y0, y1, x0, x1 = crop_bounds_fraction(image.shape, fraction=fraction)
    return image[y0:y1, x0:x1]


def crop_at_fraction(
    image: np.ndarray,
    *,
    fraction: float,
    y_frac: float = 0.5,
    x_frac: float = 0.5,
) -> np.ndarray:
    y0, y1, x0, x1 = crop_bounds_fraction(image.shape, fraction=fraction, y_frac=y_frac, x_frac=x_frac)
    return image[y0:y1, x0:x1]


def display_crop(
    image: np.ndarray,
    *,
    center_fraction: float,
    zoom: float,
    roi_fraction: float | None = None,
    roi_y_frac: float = 0.5,
    roi_x_frac: float = 0.5,
) -> np.ndarray:
    if roi_fraction is None:
        crop = center_fraction_crop(np.asarray(image), center_fraction)
    else:
        crop = crop_at_fraction(
            np.asarray(image),
            fraction=roi_fraction,
            y_frac=roi_y_frac,
            x_frac=roi_x_frac,
        )
    return ndimage.zoom(crop, zoom=float(zoom), order=1).astype(np.float32, copy=False)


def temperature_limits(images: list[np.ndarray]) -> tuple[float, float]:
    finite = [image[np.isfinite(image)].ravel() for image in images if np.isfinite(image).any()]
    if not finite:
        return 0.0, 1.0
    values = np.concatenate(finite)
    return float(np.percentile(values, 1.0)), float(np.percentile(values, 99.0))


def shared_abs_limit(images: list[np.ndarray], percentile: float = 99.0) -> float:
    finite = [np.abs(image[np.isfinite(image)]).ravel() for image in images if np.isfinite(image).any()]
    if not finite:
        return 1.0
    return max(float(np.percentile(np.concatenate(finite), percentile)), 1e-6)


def highpass_image(image: np.ndarray, *, sigma: float = 5.0) -> np.ndarray:
    return (np.asarray(image, dtype=np.float32) - ndimage.gaussian_filter(image, sigma=sigma, mode="nearest")).astype(
        np.float32,
        copy=False,
    )


def save_panel_figure(
    arrays: dict[str, np.ndarray],
    output_path: Path,
    *,
    mode: str,
    zoom: float,
    center_fraction: float,
    roi_fraction: float | None = None,
    roi_y_frac: float = 0.5,
    roi_x_frac: float = 0.5,
) -> None:
    setup_academic_style()
    panels = [
        (
            name,
            display_crop(
                image,
                center_fraction=center_fraction,
                zoom=zoom,
                roi_fraction=roi_fraction,
                roi_y_frac=roi_y_frac,
                roi_x_frac=roi_x_frac,
            ),
        )
        for name, image in arrays.items()
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(min(13.2, 3.0 * len(panels)), 3.2), squeeze=False)
    if mode == "temperature":
        vmin, vmax = temperature_limits(list(arrays.values()))
        cmap = COLORMAPS["temperature"]
        label = "Temperature [deg C]"
    elif mode == "highpass":
        vmax = shared_abs_limit([image for _, image in panels])
        vmin = -vmax
        cmap = COLORMAPS["residual_diff"]
        label = "Highpass response [deg C]"
    else:
        raise ValueError(f"Unknown mode: {mode}")
    for ax, (title, image) in zip(axes.ravel(), panels, strict=True):
        im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.025, pad=0.02).set_label(label)
    savefig_academic(fig, output_path)


def load_ep07_temperature(
    raw_frames: np.ndarray,
    shifts: np.ndarray,
    *,
    checkpoint: Path,
    target_scale: int,
    device: torch.device,
    highpass_sigma: float,
    patch_size_hr: int,
    overlap: int,
) -> np.ndarray:
    from unet_sr.inference import infer_from_burst
    from unet_sr.model import ThermalSRUNet

    if not checkpoint.exists():
        fallback = PROJECT_ROOT / "output" / "ep12_4x_benchmark" / "ep07x2up_vs_ep12" / "ep07_2x_x2up_temp.npy"
        if fallback.exists():
            image4 = np.load(fallback).astype(np.float32, copy=False)
            return ndimage.zoom(image4, zoom=target_scale / 4.0, order=3, mode="nearest").astype(np.float32, copy=False)
        raise FileNotFoundError(f"EP07 checkpoint not found: {checkpoint}")

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = dict(ckpt.get("config") or {})
    config_path = checkpoint.parent / "config.json"
    if config_path.exists():
        cfg = {**json.loads(config_path.read_text(encoding="utf-8")), **cfg}
    residual = bool(cfg.get("residual", False))
    ep07_scale = int(cfg.get("scale", 2))
    if ep07_scale != 2:
        raise ValueError(f"Expected EP07 scale=2; got {ep07_scale}")
    model = ThermalSRUNet(
        in_channels=int(cfg.get("in_channels", 6 if residual else 5)),
        out_channels=int(cfg.get("out_channels", 1)),
        base_channels=int(cfg.get("base_channels", 64)),
        scale=1 if residual else ep07_scale,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    ep07_2x = infer_from_burst(
        model,
        raw_frames,
        shifts,
        scale=2,
        patch_size_hr=int(patch_size_hr or cfg.get("patch_size_hr", 256)),
        overlap=int(overlap),
        device=str(device),
        sigma_bg=highpass_sigma,
        residual=residual,
    ).astype(np.float32, copy=False)
    return ndimage.zoom(ep07_2x, zoom=target_scale / 2.0, order=3, mode="nearest").astype(np.float32, copy=False)


def crop_for_frc(image: np.ndarray, *, scale: int, crop_lr_px: int) -> np.ndarray:
    crop = int(crop_lr_px * scale)
    if crop <= 0:
        return np.asarray(image, dtype=np.float32)
    if image.shape[0] <= 2 * crop or image.shape[1] <= 2 * crop:
        raise ValueError(f"crop {crop} too large for image shape {image.shape}")
    return np.asarray(image[crop:-crop, crop:-crop], dtype=np.float32)


def frc_curve(
    image_a: np.ndarray,
    image_b: np.ndarray,
    *,
    scale: int,
    crop_lr_px: int,
    tukey_alpha: float,
) -> pd.DataFrame:
    a = crop_for_frc(image_a, scale=scale, crop_lr_px=crop_lr_px)
    b = crop_for_frc(image_b, scale=scale, crop_lr_px=crop_lr_px)
    if a.shape != b.shape:
        raise ValueError(f"FRC inputs have different shapes: {a.shape} vs {b.shape}")
    a = a - float(np.nanmean(a))
    b = b - float(np.nanmean(b))
    win_y = tukey(a.shape[0], alpha=tukey_alpha).astype(np.float32)
    win_x = tukey(a.shape[1], alpha=tukey_alpha).astype(np.float32)
    window = win_y[:, None] * win_x[None, :]

    fa = np.fft.fft2(a * window)
    fb = np.fft.fft2(b * window)
    cross = fa * np.conj(fb)
    power_a = np.abs(fa) ** 2
    power_b = np.abs(fb) ** 2

    hr_pitch_um = PIXEL_SIZE_UM / scale
    rows, cols = a.shape
    image_size = min(rows, cols)
    df = 1.0 / (image_size * hr_pitch_um)
    fy = np.fft.fftfreq(rows, d=hr_pitch_um)
    fx = np.fft.fftfreq(cols, d=hr_pitch_um)
    radial_frequency = np.hypot(fy[:, None], fx[None, :])
    ring_index = np.floor(radial_frequency / df + 1e-12).astype(np.int32)
    max_frequency = 0.5 / hr_pitch_um
    valid = radial_frequency <= max_frequency + 1e-12
    flat_ring = ring_index[valid].ravel()
    max_ring = int(flat_ring.max())

    numerator = np.bincount(flat_ring, weights=np.real(cross[valid]).ravel(), minlength=max_ring + 1)
    denom_a = np.bincount(flat_ring, weights=power_a[valid].ravel(), minlength=max_ring + 1)
    denom_b = np.bincount(flat_ring, weights=power_b[valid].ravel(), minlength=max_ring + 1)
    n_ring = np.bincount(flat_ring, minlength=max_ring + 1).astype(int)
    denominator = np.sqrt(np.maximum(denom_a * denom_b, 0.0))
    frc = np.divide(numerator, denominator, out=np.full_like(numerator, np.nan), where=denominator > 0)
    frequencies = np.arange(max_ring + 1, dtype=float) * df
    periods = np.divide(1.0, frequencies, out=np.full_like(frequencies, np.inf), where=frequencies > 0)

    sqrt_n = np.sqrt(np.maximum(n_ring.astype(float), 1.0))
    half_bit = (0.2071 + 1.9102 / sqrt_n) / (1.2071 + 0.9102 / sqrt_n)
    return pd.DataFrame(
        {
            "frequency_um_inv": frequencies,
            "period_um": periods,
            "frc": frc,
            "threshold_1_7": np.full_like(frequencies, 1.0 / 7.0, dtype=float),
            "threshold_half_bit": half_bit,
            "n_ring_pixels": n_ring,
        }
    )


def find_cutoff_period(curve: pd.DataFrame, threshold_column: str = "threshold_1_7") -> float:
    valid = curve[
        (curve["frequency_um_inv"] > 0)
        & np.isfinite(curve["frc"])
        & np.isfinite(curve[threshold_column])
    ].copy()
    if valid.empty:
        return float("nan")
    below = valid["frc"].to_numpy(dtype=float) < valid[threshold_column].to_numpy(dtype=float)
    if bool(below.any()):
        return float(valid.iloc[int(np.argmax(below))]["period_um"])
    return float(valid.iloc[-1]["period_um"])


def plot_frc_verification(curves: dict[str, pd.DataFrame], output_path: Path) -> None:
    setup_academic_style()
    fig, ax = plt.subplots(figsize=FIGURE_SIZES["one_half_col"])
    for idx, (name, curve) in enumerate(curves.items()):
        style = get_method_style(idx)
        plot = curve[np.isfinite(curve["period_um"]) & (curve["period_um"] >= 4.0)].copy()
        ax.plot(
            plot["period_um"],
            plot["frc"],
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=1.2,
            label=name,
        )
    first = next(iter(curves.values()))
    plot = first[np.isfinite(first["period_um"]) & (first["period_um"] >= 4.0)].copy()
    ax.plot(plot["period_um"], plot["threshold_half_bit"], color="#666666", linewidth=0.9, label="half-bit")
    ax.axhline(1.0 / 7.0, color="#444444", linestyle="--", linewidth=0.9, label="1/7")
    ax.axvline(12.0, color="#222222", linestyle=":", linewidth=0.9, label="12 um")
    ax.axvline(17.03, color="#777777", linestyle="-.", linewidth=0.9, label="M2 cutoff")
    ax.set_xlabel("Spatial period [um]")
    ax.set_ylabel("FRC")
    ax.set_title("M4 Split-Half FRC Verification")
    ax.set_xlim(30.0, 4.0)
    ax.set_ylim(-0.12, 1.04)
    ax.grid(axis="y", alpha=0.25)
    ax.legend(loc="upper right", fontsize=6.5)
    savefig_academic(fig, output_path)


def sample_line(image: np.ndarray, y0: float, x0: float, y1: float, x1: float, *, n_samples: int | None = None) -> np.ndarray:
    length = float(np.hypot(y1 - y0, x1 - x0))
    n = int(n_samples or max(16, round(length) + 1))
    ys = np.linspace(y0, y1, n)
    xs = np.linspace(x0, x1, n)
    return ndimage.map_coordinates(np.asarray(image, dtype=np.float32), [ys, xs], order=1, mode="nearest")


def line_signal_for_dark_trace(profile: np.ndarray) -> tuple[np.ndarray, float]:
    baseline = float(np.percentile(profile, 90.0))
    signal = baseline - np.asarray(profile, dtype=np.float32)
    return signal.astype(np.float32, copy=False), baseline


def profile_metrics(signal: np.ndarray, *, pitch_um: float, min_spacing_um: float = 8.0) -> dict[str, float | bool]:
    sig = ndimage.gaussian_filter1d(np.asarray(signal, dtype=np.float32), sigma=1.0, mode="nearest")
    base = float(np.percentile(sig, 10.0))
    height = float(np.max(sig) - base)
    if height <= 1e-8:
        return {
            "fwhm_um": float("nan"),
            "dip_ratio_formula": float("nan"),
            "dip_depth": float("nan"),
            "lines_separated": False,
            "n_peaks": 0,
        }
    peaks, props = find_peaks(sig, prominence=0.10 * height, distance=max(2, int(round(min_spacing_um / pitch_um))))
    if peaks.size == 0:
        peak_idx = int(np.argmax(sig))
        peaks = np.asarray([peak_idx], dtype=int)
    primary = int(peaks[np.argmax(sig[peaks])])
    half = base + 0.5 * (float(sig[primary]) - base)
    left = primary
    while left > 0 and sig[left] > half:
        left -= 1
    right = primary
    while right < sig.size - 1 and sig[right] > half:
        right += 1
    fwhm_um = float(max(0, right - left) * pitch_um)

    if peaks.size >= 2:
        ranked = peaks[np.argsort(sig[peaks])[-2:]]
        p0, p1 = sorted(int(p) for p in ranked)
        valley = float(np.min(sig[p0:p1 + 1]))
        peak_ref = min(float(sig[p0]), float(sig[p1]))
        dip_ratio = (valley - base) / max(peak_ref - base, 1e-8)
        dip_ratio = float(np.clip(dip_ratio, 0.0, 1.5))
        dip_depth = float(1.0 - np.clip(dip_ratio, 0.0, 1.0))
        lines_separated = bool(dip_depth >= 0.25 and (p1 - p0) * pitch_um >= min_spacing_um)
    else:
        dip_ratio = float("nan")
        dip_depth = float("nan")
        lines_separated = False
    return {
        "fwhm_um": fwhm_um,
        "dip_ratio_formula": dip_ratio,
        "dip_depth": dip_depth,
        "lines_separated": lines_separated,
        "n_peaks": int(peaks.size),
    }


def zigzag_profile_specs(image_shape: tuple[int, int]) -> list[dict[str, float | str]]:
    y0, y1, x0, x1 = crop_bounds_fraction(
        image_shape,
        fraction=ZIGZAG_ROI_FRACTION,
        y_frac=ZIGZAG_ROI_CENTER_YX[0],
        x_frac=ZIGZAG_ROI_CENTER_YX[1],
    )
    roi_h = y1 - y0
    roi_w = x1 - x0
    return [
        {
            "profile_id": "zigzag_upper_left",
            "y0": y0 + 0.24 * roi_h,
            "x0": x0 + 0.02 * roi_w,
            "y1": y0 + 0.24 * roi_h,
            "x1": x0 + 0.44 * roi_w,
        },
        {
            "profile_id": "zigzag_mid_left",
            "y0": y0 + 0.36 * roi_h,
            "x0": x0 + 0.03 * roi_w,
            "y1": y0 + 0.36 * roi_h,
            "x1": x0 + 0.46 * roi_w,
        },
        {
            "profile_id": "zigzag_lower_left",
            "y0": y0 + 0.49 * roi_h,
            "x0": x0 + 0.04 * roi_w,
            "y1": y0 + 0.49 * roi_h,
            "x1": x0 + 0.48 * roi_w,
        },
    ]


def analyze_zigzag_profiles(
    drizzle_temp: np.ndarray,
    map_tv_temp: np.ndarray,
    *,
    scale: int,
    output_csv: Path,
    output_png: Path,
) -> pd.DataFrame:
    pitch = PIXEL_SIZE_UM / scale
    rows: list[dict[str, Any]] = []
    setup_academic_style()
    specs = zigzag_profile_specs(map_tv_temp.shape)
    fig, axes = plt.subplots(len(specs), 1, figsize=(7.2, 2.2 * len(specs)), squeeze=False)
    for ax, spec in zip(axes.ravel(), specs, strict=True):
        y0 = float(spec["y0"])
        x0 = float(spec["x0"])
        y1 = float(spec["y1"])
        x1 = float(spec["x1"])
        drizzle_profile = sample_line(drizzle_temp, y0, x0, y1, x1)
        map_profile = sample_line(map_tv_temp, y0, x0, y1, x1, n_samples=drizzle_profile.size)
        d_signal, _ = line_signal_for_dark_trace(drizzle_profile)
        m_signal, _ = line_signal_for_dark_trace(map_profile)
        distance_um = np.arange(drizzle_profile.size, dtype=float) * pitch
        ax.plot(distance_um, d_signal, color="#4C72B0", linewidth=1.1, label="bare drizzle")
        ax.plot(distance_um, m_signal, color="#C44E52", linewidth=1.1, label="MAP-TV")
        ax.set_title(str(spec["profile_id"]).replace("_", " "))
        ax.set_xlabel("Distance along profile [um]")
        ax.set_ylabel("Inverted line contrast [deg C]")
        ax.grid(axis="y", alpha=0.25)
        ax.legend(loc="upper right", fontsize=7)
        for method, signal in (("bare_drizzle", d_signal), ("map_tv", m_signal)):
            metric = profile_metrics(signal, pitch_um=pitch)
            rows.append(
                {
                    "profile_id": spec["profile_id"],
                    "method": method,
                    "y0_hr_px": y0,
                    "x0_hr_px": x0,
                    "y1_hr_px": y1,
                    "x1_hr_px": x1,
                    **metric,
                }
            )
    fig.tight_layout()
    savefig_academic(fig, output_png)
    table = pd.DataFrame(rows)
    table.to_csv(output_csv, index=False)
    return table


def plot_convergence(convergence: pd.DataFrame, output_path: Path) -> None:
    setup_academic_style()
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.0), squeeze=False)
    full = convergence[convergence["track"].astype(str).str.startswith("full_")].copy()
    if full.empty:
        full = convergence.copy()
    for idx, (track, group) in enumerate(full.groupby("track", sort=False)):
        style = get_method_style(idx)
        label = str(track).replace("full_", "")
        axes[0, 0].plot(group["iteration"], group["data_rmse"], color=style["color"], linestyle=style["linestyle"], label=label)
        axes[0, 1].plot(group["iteration"], group["objective"], color=style["color"], linestyle=style["linestyle"], label=label)
        axes[0, 2].semilogy(
            group["iteration"],
            np.maximum(group["relative_update"].to_numpy(dtype=float), 1e-12),
            color=style["color"],
            linestyle=style["linestyle"],
            label=label,
        )
    axes[0, 0].set_title("Data RMSE")
    axes[0, 1].set_title("Objective proxy")
    axes[0, 2].set_title("Relative update")
    for ax in axes.ravel():
        ax.set_xlabel("Iteration")
        ax.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel("deg C")
    axes[0, 2].set_ylabel("L2 ratio")
    axes[0, 1].legend(loc="best", fontsize=6.5)
    fig.tight_layout()
    savefig_academic(fig, output_path)


def summarize_zigzag(table: pd.DataFrame) -> dict[str, float | int]:
    pivot = table.pivot(index="profile_id", columns="method")
    fwhm_before = pivot["fwhm_um"]["bare_drizzle"].to_numpy(dtype=float)
    fwhm_after = pivot["fwhm_um"]["map_tv"].to_numpy(dtype=float)
    dip_before = pivot["dip_depth"]["bare_drizzle"].to_numpy(dtype=float)
    dip_after = pivot["dip_depth"]["map_tv"].to_numpy(dtype=float)
    fwhm_before_median = float(np.nanmedian(fwhm_before))
    fwhm_after_median = float(np.nanmedian(fwhm_after))
    dip_before_median = float(np.nanmedian(dip_before))
    dip_after_median = float(np.nanmedian(dip_after))
    return {
        "fwhm_before_median_um": fwhm_before_median,
        "fwhm_after_median_um": fwhm_after_median,
        "fwhm_narrowing_by_medians_um": float(fwhm_before_median - fwhm_after_median),
        "fwhm_delta_median_um": float(np.nanmedian(fwhm_after - fwhm_before)),
        "dip_depth_before_median": dip_before_median,
        "dip_depth_after_median": dip_after_median,
        "dip_depth_delta_by_medians": float(dip_after_median - dip_before_median),
        "dip_depth_delta_median": float(np.nanmedian(dip_after - dip_before)),
        "profiles_total": int(table["profile_id"].nunique()),
        "profiles_map_tv_separated": int(table[(table["method"].eq("map_tv")) & (table["lines_separated"].astype(bool))]["profile_id"].nunique()),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    start_all = time.perf_counter()
    setup_academic_style()
    torch.backends.cudnn.benchmark = True
    device = resolve_device(args.device)
    output_dir = args.output_dir.resolve()
    if args.smoke and output_dir == DEFAULT_OUTPUT_DIR.resolve():
        output_dir = output_dir / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)

    grid = load_grid_decision(args.grid_decision_json, args.m1_summary_json)
    scale = int(args.scale or grid.scale)
    if scale < 2:
        raise ValueError("scale must be >=2")
    psf_sigmas = parse_float_list(args.psf_sigmas)
    lambdas = parse_float_list(args.lambda_grid)
    max_iter = int(args.smoke_iters if args.smoke else args.max_iter)
    lambda_select_iter = int(args.smoke_iters if args.smoke else args.lambda_select_iter)
    if args.smoke:
        psf_sigmas = psf_sigmas[:1]
        lambdas = lambdas[:1]

    frame_limit = int(args.smoke_frames) if args.smoke else args.limit
    print("Loading clean main-session frames and contour-refined shifts")
    raw_frames, metadata = load_main_session_frames(
        args.data_dir,
        args.frame_audit_csv,
        workers=args.workers,
        dtype=np.float32,
        limit=frame_limit,
    )
    if frame_limit is None:
        shifts = load_alignment_shifts("contour_refined", metadata=metadata, alignment_csv=args.alignment_csv).astype(np.float32)
    else:
        full_metadata = load_main_session_metadata(args.frame_audit_csv)
        full_shifts = load_alignment_shifts("contour_refined", metadata=full_metadata, alignment_csv=args.alignment_csv).astype(
            np.float32
        )
        shifts = full_shifts[: len(metadata)]
    validate_inputs(raw_frames, metadata, shifts, allow_limit=args.smoke or args.limit is not None)

    highpass_frames = highpass_preprocess(raw_frames, sigma_bg=args.highpass_sigma, workers=args.workers).astype(np.float32)
    smoke = run_forward_smoke(
        shifts=shifts,
        scale=scale,
        psf_sigma=float(psf_sigmas[0]),
        device=device,
        use_box=not args.no_box,
        chunk_size=args.chunk_size,
        tolerance=args.forward_smoke_tol,
    )
    _write_json(output_dir / "forward_smoke.json", smoke)
    if smoke["forward_smoke_max_abs"] is None:
        print("Forward smoke passed: EP06 box reference skipped for --no-box model")
    else:
        print(f"Forward smoke passed: max_abs={smoke['forward_smoke_max_abs']:.3g}")

    bare_hp = bilinear_drizzle(highpass_frames, shifts, scale=scale, desc="bare drizzle highpass")
    init_full = bare_hp.image
    np.save(output_dir / "bare_drizzle_highpass.npy", init_full.astype(np.float32, copy=False))

    selection_tables: list[pd.DataFrame] = []
    convergence_tables: list[pd.DataFrame] = []
    full_results: dict[float, MapTVResult] = {}
    full_rows: list[dict[str, Any]] = []

    for sigma in psf_sigmas:
        selected_lambda, table = select_lambda_for_sigma(
            highpass_frames,
            shifts,
            scale=scale,
            psf_sigma=float(sigma),
            lambdas=lambdas,
            max_iter=lambda_select_iter,
            step_size=args.step_size,
            tol=args.tol,
            device=device,
            use_box=not args.no_box,
            chunk_size=args.chunk_size,
            artifact_weight=args.selection_artifact_weight,
            std_weight=args.selection_std_weight,
        )
        selection_tables.append(table)
        print(f"sigma={sigma:g}: selected lambda={selected_lambda:g}")
        result = run_map_tv_gpu(
            highpass_frames,
            shifts,
            init_full,
            scale=scale,
            psf_sigma=float(sigma),
            lambda_tv=selected_lambda,
            max_iter=max_iter,
            step_size=args.step_size,
            tol=args.tol,
            device=device,
            use_box=not args.no_box,
            chunk_size=args.chunk_size,
            track=f"full_sigma{sigma:g}_lambda{selected_lambda:g}",
        )
        convergence_tables.append(result.convergence)
        full_results[float(sigma)] = result
        np.save(output_dir / f"map_tv_sigma{sigma:g}_lambda{selected_lambda:g}.npy", result.image)
        full_rows.append(
            {
                "psf_sigma_lr_px": float(sigma),
                "lambda_tv": float(selected_lambda),
                "full_iterations": int(len(result.convergence)),
                "full_elapsed_sec": result.elapsed_sec,
                "full_final_data_rmse": float(result.convergence["data_rmse"].iloc[-1]),
                "full_final_objective": float(result.convergence["objective"].iloc[-1]),
                "full_final_relative_update": float(result.convergence["relative_update"].iloc[-1]),
            }
        )

    selection = pd.concat(selection_tables, ignore_index=True)
    best_idx = int(selection["selection_proxy"].idxmin())
    selection.loc[best_idx, "selected_global"] = True
    best_sigma = float(selection.loc[best_idx, "psf_sigma_lr_px"])
    best_lambda = float(selection.loc[best_idx, "lambda_tv"])
    selection = selection.merge(pd.DataFrame(full_rows), on=["psf_sigma_lr_px", "lambda_tv"], how="left")
    selection.to_csv(output_dir / "parameter_selection.csv", index=False)

    all_convergence = pd.concat(convergence_tables, ignore_index=True)
    all_convergence.to_csv(output_dir / "convergence_curves.csv", index=False)
    plot_convergence(all_convergence, output_dir / "convergence_curves.png")

    best_image = full_results[best_sigma].image
    np.save(output_dir / "map_tv_best.npy", best_image.astype(np.float32, copy=False))

    print("Building display arms")
    bare_temp = bilinear_drizzle(raw_frames, shifts, scale=scale, desc="bare drizzle temperature")
    bicubic_temp = bicubic_upsample(np.nanmean(raw_frames, axis=0), scale=scale)
    lowfreq_base = bicubic_upsample(
        ndimage.gaussian_filter(np.nanmean(raw_frames, axis=0), sigma=args.highpass_sigma, mode="nearest"),
        scale=scale,
    )
    map_tv_temp = (lowfreq_base + best_image).astype(np.float32, copy=False)
    ep07_temp = load_ep07_temperature(
        raw_frames,
        shifts,
        checkpoint=args.ep07_checkpoint,
        target_scale=scale,
        device=device,
        highpass_sigma=args.highpass_sigma,
        patch_size_hr=args.ep07_patch_size_hr,
        overlap=args.ep07_overlap,
    )
    arms_temp = {
        "Bicubic": bicubic_temp.astype(np.float32, copy=False),
        "Bare drizzle": bare_temp.image.astype(np.float32, copy=False),
        "MAP-TV": map_tv_temp,
        "EP07 v6 x2.5up": ep07_temp,
    }
    arms_hp = {name: highpass_image(image, sigma=args.figure_highpass_sigma) for name, image in arms_temp.items()}
    for name, image in arms_temp.items():
        stem = name.lower().replace(" ", "_").replace(".", "p")
        np.save(output_dir / f"{stem}_temperature.npy", image.astype(np.float32, copy=False))
        np.save(output_dir / f"{stem}_highpass.npy", arms_hp[name].astype(np.float32, copy=False))

    save_panel_figure(
        arms_temp,
        output_dir / "four_arm_comparison.png",
        mode="temperature",
        zoom=args.center_zoom,
        center_fraction=args.center_fraction,
    )
    save_panel_figure(
        arms_hp,
        output_dir / "four_arm_highpass.png",
        mode="highpass",
        zoom=args.roi_zoom,
        center_fraction=args.center_fraction,
        roi_fraction=ZIGZAG_ROI_FRACTION,
        roi_y_frac=ZIGZAG_ROI_CENTER_YX[0],
        roi_x_frac=ZIGZAG_ROI_CENTER_YX[1],
    )

    zigzag = analyze_zigzag_profiles(
        bare_temp.image,
        map_tv_temp,
        scale=scale,
        output_csv=output_dir / "zigzag_profile_metrics.csv",
        output_png=output_dir / "zigzag_profiles.png",
    )
    zigzag_summary = summarize_zigzag(zigzag)

    print("Running split-half FRC verification for best MAP-TV")
    even = np.arange(len(highpass_frames)) % 2 == 0
    odd = ~even
    bare_a = bilinear_drizzle(highpass_frames[even], shifts[even], scale=scale, desc="FRC bare A").image
    bare_b = bilinear_drizzle(highpass_frames[odd], shifts[odd], scale=scale, desc="FRC bare B").image
    init_a = bare_a
    init_b = bare_b
    map_a = run_map_tv_gpu(
        highpass_frames[even],
        shifts[even],
        init_a,
        scale=scale,
        psf_sigma=best_sigma,
        lambda_tv=best_lambda,
        max_iter=max_iter,
        step_size=args.step_size,
        tol=args.tol,
        device=device,
        use_box=not args.no_box,
        chunk_size=args.chunk_size,
        track=f"frc_best_a_sigma{best_sigma:g}_lambda{best_lambda:g}",
    )
    map_b = run_map_tv_gpu(
        highpass_frames[odd],
        shifts[odd],
        init_b,
        scale=scale,
        psf_sigma=best_sigma,
        lambda_tv=best_lambda,
        max_iter=max_iter,
        step_size=args.step_size,
        tol=args.tol,
        device=device,
        use_box=not args.no_box,
        chunk_size=args.chunk_size,
        track=f"frc_best_b_sigma{best_sigma:g}_lambda{best_lambda:g}",
    )
    frc_bare = frc_curve(bare_a, bare_b, scale=scale, crop_lr_px=args.frc_crop_lr_px, tukey_alpha=args.frc_tukey_alpha)
    frc_map = frc_curve(map_a.image, map_b.image, scale=scale, crop_lr_px=args.frc_crop_lr_px, tukey_alpha=args.frc_tukey_alpha)
    frc_bare.insert(0, "method", "bare_drizzle")
    frc_map.insert(0, "method", "map_tv")
    pd.concat([frc_bare, frc_map], ignore_index=True).to_csv(output_dir / "frc_verification.csv", index=False)
    plot_frc_verification(
        {
            "bare drizzle": frc_bare.drop(columns=["method"]),
            "MAP-TV": frc_map.drop(columns=["method"]),
        },
        output_dir / "frc_verification.png",
    )
    frc_summary = {
        "bare_cutoff_period_um": find_cutoff_period(frc_bare),
        "map_tv_cutoff_period_um": find_cutoff_period(frc_map),
    }
    for period in (20.0, 16.0, 14.0, 12.0, 10.0):
        target = 1.0 / period
        for name, curve in (("bare", frc_bare), ("map_tv", frc_map)):
            valid = np.isfinite(curve["frequency_um_inv"]) & np.isfinite(curve["frc"])
            order = np.argsort(curve.loc[valid, "frequency_um_inv"].to_numpy(dtype=float))
            freq = curve.loc[valid, "frequency_um_inv"].to_numpy(dtype=float)[order]
            vals = curve.loc[valid, "frc"].to_numpy(dtype=float)[order]
            frc_summary[f"{name}_frc_at_{period:g}um"] = float(np.interp(target, freq, vals, left=np.nan, right=np.nan))

    summary = {
        "task": "EP15 M4 MAP-TV deconvolution anchor",
        "run_mode": "smoke" if args.smoke else "full",
        "n_clean_sr_frames": int(len(raw_frames)),
        "frame_shape_lr": list(raw_frames.shape[1:]),
        "scale": int(scale),
        "hr_pitch_um": PIXEL_SIZE_UM / scale,
        "grid_decision_source": grid.source,
        "alignment_method": "contour_refined",
        "forward_model": "shift -> Gaussian PSF -> detector box avg_pool2d -> LR" if not args.no_box else "shift -> Gaussian PSF -> point sample",
        "psf_sigmas_lr_px": [float(v) for v in psf_sigmas],
        "lambda_grid": [float(v) for v in lambdas],
        "max_iter": int(max_iter),
        "lambda_select_iter": int(lambda_select_iter),
        "step_size": float(args.step_size),
        "tol": float(args.tol),
        "chunk_size": int(args.chunk_size),
        "device": str(device),
        "selected_psf_sigma_lr_px": best_sigma,
        "selected_lambda_tv": best_lambda,
        "best_selection_proxy": float(selection.loc[best_idx, "selection_proxy"]),
        "zigzag": zigzag_summary,
        "frc": frc_summary,
        "zero_coverage_pct": {
            "bare_highpass": bare_hp.zero_coverage_pct,
            "bare_temperature": bare_temp.zero_coverage_pct,
        },
        "forward_smoke": smoke,
        "outputs": {
            "four_arm_comparison": output_dir / "four_arm_comparison.png",
            "four_arm_highpass": output_dir / "four_arm_highpass.png",
            "zigzag_profiles": output_dir / "zigzag_profiles.png",
            "zigzag_profile_metrics": output_dir / "zigzag_profile_metrics.csv",
            "convergence_curves": output_dir / "convergence_curves.png",
            "convergence_curves_csv": output_dir / "convergence_curves.csv",
            "frc_verification": output_dir / "frc_verification.png",
            "frc_verification_csv": output_dir / "frc_verification.csv",
            "parameter_selection": output_dir / "parameter_selection.csv",
            "map_tv_best": output_dir / "map_tv_best.npy",
            "forward_smoke": output_dir / "forward_smoke.json",
            "m4_summary": output_dir / "m4_summary.json",
            "per_sigma_map_tv": [
                output_dir / f"map_tv_sigma{row['psf_sigma_lr_px']:g}_lambda{row['lambda_tv']:g}.npy"
                for row in full_rows
            ],
            "display_arrays": [
                output_dir / "bicubic_temperature.npy",
                output_dir / "bare_drizzle_temperature.npy",
                output_dir / "map-tv_temperature.npy",
                output_dir / "ep07_v6_x2p5up_temperature.npy",
                output_dir / "bicubic_highpass.npy",
                output_dir / "bare_drizzle_highpass.npy",
                output_dir / "map-tv_highpass.npy",
                output_dir / "ep07_v6_x2p5up_highpass.npy",
            ],
        },
        "elapsed_sec": float(time.perf_counter() - start_all),
    }
    _write_json(output_dir / "m4_summary.json", summary)
    print(f"Selected sigma={best_sigma:g}, lambda={best_lambda:g}")
    print(
        "Zigzag FWHM median: "
        f"{zigzag_summary['fwhm_before_median_um']:.2f} -> {zigzag_summary['fwhm_after_median_um']:.2f} um; "
        "dip depth median: "
        f"{zigzag_summary['dip_depth_before_median']:.3f} -> {zigzag_summary['dip_depth_after_median']:.3f}"
    )
    print(f"Saved M4 outputs to {_rel(output_dir)}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=PROJECT_ROOT / "data" / "data_raw" / "infrared_avi")
    parser.add_argument("--frame-audit-csv", type=Path, default=PROJECT_ROOT / "output" / "ep01_data_processing" / "frame_audit.csv")
    parser.add_argument("--alignment-csv", type=Path, default=default_contour_alignment_csv(project_root_path=PROJECT_ROOT))
    parser.add_argument("--grid-decision-json", type=Path, default=PROJECT_ROOT / "output" / "ep15_info_limit" / "m1_phase_structure" / "grid_decision.json")
    parser.add_argument("--m1-summary-json", type=Path, default=PROJECT_ROOT / "output" / "ep15_info_limit" / "m1_phase_structure" / "m1_phase_structure_summary.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--ep07-checkpoint", type=Path, default=DEFAULT_EP07_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--scale", type=int, default=None)
    parser.add_argument("--psf-sigmas", default=",".join(str(v) for v in DEFAULT_PSF_SIGMAS))
    parser.add_argument("--lambda-grid", default=",".join(str(v) for v in DEFAULT_LAMBDAS))
    parser.add_argument("--max-iter", type=int, default=150)
    parser.add_argument("--lambda-select-iter", type=int, default=150)
    parser.add_argument("--step-size", type=float, default=1.0)
    parser.add_argument("--tol", type=float, default=1e-5)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--no-box", action="store_true", help="Disable detector box integration and point-sample after PSF.")
    parser.add_argument("--highpass-sigma", type=float, default=5.0)
    parser.add_argument("--figure-highpass-sigma", type=float, default=5.0)
    parser.add_argument("--selection-artifact-weight", type=float, default=0.05)
    parser.add_argument("--selection-std-weight", type=float, default=0.08)
    parser.add_argument("--forward-smoke-tol", type=float, default=1e-4)
    parser.add_argument("--center-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--center-zoom", type=float, default=3.0)
    parser.add_argument("--roi-zoom", type=float, default=5.0)
    parser.add_argument("--ep07-patch-size-hr", type=int, default=256)
    parser.add_argument("--ep07-overlap", type=int, default=128)
    parser.add_argument("--frc-crop-lr-px", type=int, default=16)
    parser.add_argument("--frc-tukey-alpha", type=float, default=0.25)
    parser.add_argument("--smoke", action="store_true", help="Run a 32-frame, 5-iteration reduced pipeline under output_dir/smoke.")
    parser.add_argument("--smoke-frames", type=int, default=32)
    parser.add_argument("--smoke-iters", type=int, default=5)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
