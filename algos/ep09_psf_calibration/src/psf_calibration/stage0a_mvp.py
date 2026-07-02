"""Stage 0a MVP: bounded real-frame operator calibration diagnostics.

This is intentionally not the full Solver V2 joint optimizer.  It fixes most
variables, builds one deterministic SAA estimate from the clean real burst, and
then sweeps a small PSF/shift-refinement grid to measure whether the
band-limited DC residual floor can move when the operator is less misspecified.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from scipy import ndimage

from .data import DEFAULT_ALIGNMENT_CSV, DEFAULT_DATA_DIR, DEFAULT_FRAME_AUDIT_CSV
from .utils import (
    OUTPUT_DIR,
    crop_slices,
    default_workers,
    deterministic_split,
    ensure_dir,
    relative,
    write_json,
)

from .utils import bootstrap_project_paths

bootstrap_project_paths()

from common.alignment import load_alignment_shifts  # noqa: E402
from common.data_loader import highpass_preprocess, load_main_session_frames  # noqa: E402
from saa.saa import reconstruct_saa  # noqa: E402
from thermal_core.displacement import coordinate_to_shift  # noqa: E402


DEFAULT_STAGE_CONFIG = Path(__file__).resolve().parents[4] / "configs" / "stage_calibration.json"
DEFAULT_STAGE0A_OUTPUT_DIR = OUTPUT_DIR / "stage0a_mvp"
BASELINE_SIGMA_LR_PX = 0.5


@dataclass(frozen=True)
class Stage0aInputs:
    """Loaded real frames plus the initial alignment operator."""

    frames_raw: np.ndarray
    frames_highpass: np.ndarray
    metadata: pd.DataFrame
    shifts: np.ndarray
    shift_source: str
    detector_pitch_um: float
    spatial_resolution_um: float


@dataclass(frozen=True)
class PsfCandidate:
    """Small-grid PSF candidate in LR detector-pixel units."""

    sigma_x_lr_px: float
    sigma_y_lr_px: float
    angle_deg: float

    @property
    def anisotropy_ratio(self) -> float:
        lo = max(min(self.sigma_x_lr_px, self.sigma_y_lr_px), 1e-12)
        hi = max(self.sigma_x_lr_px, self.sigma_y_lr_px)
        return float(hi / lo)

    @property
    def label(self) -> str:
        return f"sx{self.sigma_x_lr_px:.3f}_sy{self.sigma_y_lr_px:.3f}_a{self.angle_deg:.1f}"


@dataclass(frozen=True)
class FrameScore:
    """Per-frame residual score and optional bounded shift correction."""

    frame_index: int
    file: str
    split: str
    refine_mode: str
    candidate_label: str
    sigma_x_lr_px: float
    sigma_y_lr_px: float
    angle_deg: float
    initial_dx_px: float
    initial_dy_px: float
    delta_dx_px: float
    delta_dy_px: float
    refined_dx_px: float
    refined_dy_px: float
    band_mse: float
    band_rms: float
    full_mse: float
    full_rms: float


@dataclass(frozen=True)
class Stage0aResult:
    """Stage 0a MVP outputs."""

    sweep: pd.DataFrame
    frame_scores: pd.DataFrame
    shift_refinements: pd.DataFrame
    summary: dict


def _read_stage_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text(encoding="utf-8"))
    pitch = float(cfg.get("pixel_size_um", np.nan))
    resolution = float(cfg.get("current_spatial_resolution_um", np.nan))
    if not np.isfinite(pitch) or not np.isfinite(resolution):
        raise ValueError(f"stage config missing finite pixel_size_um/current_spatial_resolution_um: {path}")
    if abs(pitch - 20.0) > 1e-6 or abs(resolution - 20.0) > 1e-6:
        raise ValueError(
            "Stage 0a must run under the corrected 20 um detector pitch and 20 um spatial "
            f"resolution contract, got pixel_size_um={pitch:g}, current_spatial_resolution_um={resolution:g}"
        )
    return cfg


def _stage_prior_shifts(metadata: pd.DataFrame, *, stage_config: dict, reference_file: str | None = None) -> np.ndarray:
    """Return stage-command prior shifts relative to a reference frame.

    This is only a fallback/init path.  It deliberately mirrors the existing
    EP05 stage-prior convention and is never treated as alignment truth.
    """

    if reference_file is not None and reference_file in set(metadata["file"].astype(str)):
        ref_row = metadata[metadata["file"].astype(str).eq(reference_file)].iloc[0]
    else:
        ref_row = metadata.iloc[len(metadata) // 2]
    dx, dy_math = coordinate_to_shift(
        metadata["X"].to_numpy(dtype=float) - float(ref_row["X"]),
        metadata["Y"].to_numpy(dtype=float) - float(ref_row["Y"]),
        theta_deg=float(stage_config["theta_deg"]),
        pixel_size_um=float(stage_config["pixel_size_um"]),
    )
    # Existing EP05 diagnostics use image-x sign flipped relative to the math
    # delta, while dy_math is already in the image-row convention used by the
    # SAA/forward code paths.
    return np.column_stack([-dx, dy_math]).astype(np.float32, copy=False)


def load_stage0a_inputs(
    *,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    frame_audit_csv: str | Path = DEFAULT_FRAME_AUDIT_CSV,
    alignment_csv: str | Path = DEFAULT_ALIGNMENT_CSV,
    alignment_method: str = "contour_refined",
    alignment_source: str = "auto",
    stage_config_path: str | Path = DEFAULT_STAGE_CONFIG,
    highpass_sigma: float = 5.0,
    workers: int = default_workers(),
    limit: int | None = None,
) -> Stage0aInputs:
    """Load clean real frames and shifts for Stage 0a.

    ``alignment_source='auto'`` prefers contour-refined EP04/EP05 shifts and
    falls back to the stage-command prior only when the alignment artifact is
    unavailable.  The fallback is explicitly labeled in the returned result.
    """

    if alignment_source not in {"auto", "contour", "stage_prior"}:
        raise ValueError("alignment_source must be 'auto', 'contour', or 'stage_prior'")
    stage_config = _read_stage_config(stage_config_path)
    frames, metadata = load_main_session_frames(
        data_dir=data_dir,
        frame_audit_path=frame_audit_csv,
        workers=workers,
        dtype=np.float32,
        limit=limit,
    )
    if len(metadata) != 248 and limit is None:
        raise ValueError(f"Stage 0a expected 248 clean SR frames, got {len(metadata)}")

    shift_source = ""
    if alignment_source in {"auto", "contour"}:
        try:
            shifts = load_alignment_shifts(
                method=alignment_method,
                metadata=metadata,
                alignment_csv=alignment_csv,
                strict=True,
            ).astype(np.float32, copy=False)
            shift_source = f"alignment:{alignment_method}"
        except Exception:
            if alignment_source == "contour":
                raise
            shifts = _stage_prior_shifts(metadata, stage_config=stage_config)
            shift_source = "stage_prior_fallback_not_alignment_truth"
    else:
        shifts = _stage_prior_shifts(metadata, stage_config=stage_config)
        shift_source = "stage_prior_not_alignment_truth"

    hp = highpass_preprocess(frames, sigma_bg=highpass_sigma, workers=workers).astype(np.float32, copy=False)
    return Stage0aInputs(
        frames_raw=frames.astype(np.float32, copy=False),
        frames_highpass=hp,
        metadata=metadata.copy(),
        shifts=shifts,
        shift_source=shift_source,
        detector_pitch_um=float(stage_config["pixel_size_um"]),
        spatial_resolution_um=float(stage_config["current_spatial_resolution_um"]),
    )


def parse_float_list(values: str | Iterable[float]) -> list[float]:
    """Parse CLI comma-separated floats or normalize an existing iterable."""

    if isinstance(values, str):
        out = [float(item.strip()) for item in values.split(",") if item.strip()]
    else:
        out = [float(item) for item in values]
    if not out:
        raise ValueError("float list must not be empty")
    return out


def build_psf_candidates(
    *,
    sigmas: Iterable[float],
    anisotropy_ratios: Iterable[float],
    angles: Iterable[float],
) -> list[PsfCandidate]:
    """Build a de-duplicated small candidate grid."""

    candidates: dict[tuple[float, float, float], PsfCandidate] = {}
    for sigma in parse_float_list(sigmas):
        if sigma < 0:
            raise ValueError("PSF sigma values must be non-negative")
        for ratio in parse_float_list(anisotropy_ratios):
            if ratio < 1.0:
                raise ValueError("anisotropy ratios must be >= 1")
            if abs(ratio - 1.0) < 1e-12:
                key = (round(sigma, 8), round(sigma, 8), 0.0)
                candidates[key] = PsfCandidate(float(sigma), float(sigma), 0.0)
                continue
            for angle in parse_float_list(angles):
                sx = float(sigma)
                sy = float(sigma) * float(ratio)
                key = (round(sx, 8), round(sy, 8), round(float(angle), 8))
                candidates[key] = PsfCandidate(sx, sy, float(angle))
    return list(candidates.values())


def shift_refine_offsets(radius: float, step: float) -> np.ndarray:
    """Return a bounded ``(dx, dy)`` correction grid in LR pixels."""

    radius = float(radius)
    step = float(step)
    if radius < 0:
        raise ValueError("shift_refine_radius must be >= 0")
    if radius == 0:
        return np.zeros((1, 2), dtype=np.float32)
    if step <= 0:
        raise ValueError("shift_refine_step must be > 0 when radius > 0")
    vals = np.arange(-radius, radius + 0.5 * step, step, dtype=np.float32)
    vals = vals[np.abs(vals) <= radius + 1e-6]
    yy, xx = np.meshgrid(vals, vals, indexing="ij")
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32, copy=False)


def _anisotropic_kernel(
    candidate: PsfCandidate,
    *,
    scale: int,
    truncate: float = 4.0,
) -> np.ndarray | None:
    sx = float(candidate.sigma_x_lr_px) * int(scale)
    sy = float(candidate.sigma_y_lr_px) * int(scale)
    if sx <= 0 and sy <= 0:
        return None
    sx = max(sx, 1e-6)
    sy = max(sy, 1e-6)
    radius = max(1, int(np.ceil(max(sx, sy, 1.0) * float(truncate))))
    axis = np.arange(-radius, radius + 1, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    theta = np.deg2rad(float(candidate.angle_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    xr = xx * cos_t + yy * sin_t
    yr = -xx * sin_t + yy * cos_t
    kernel = np.exp(-0.5 * ((xr / sx) ** 2 + (yr / sy) ** 2))
    total = float(kernel.sum())
    if total <= 0:
        raise ValueError(f"PSF kernel has zero mass for {candidate}")
    return (kernel / total).astype(np.float32, copy=False)


def blur_hr_anisotropic(hr_image: np.ndarray, candidate: PsfCandidate, *, scale: int = 2) -> np.ndarray:
    """Blur an HR estimate with an anisotropic Gaussian PSF candidate."""

    x = np.asarray(hr_image, dtype=np.float32)
    kernel = _anisotropic_kernel(candidate, scale=scale)
    if kernel is None:
        return x
    return ndimage.convolve(x, kernel, mode="constant", cval=0.0).astype(np.float32, copy=False)


def forward_block_average_shifted(blurred_hr: np.ndarray, shift: np.ndarray, *, scale: int = 2) -> np.ndarray:
    """Block-average a pre-blurred HR image at one shifted LR detector grid."""

    image = np.asarray(blurred_hr, dtype=np.float32)
    if image.ndim != 2:
        raise ValueError("blurred_hr must be 2D")
    scale = int(scale)
    if scale <= 0:
        raise ValueError("scale must be positive")
    h_hr, w_hr = image.shape
    h_lr, w_lr = h_hr // scale, w_hr // scale
    dx, dy = np.asarray(shift, dtype=np.float64)
    out = np.zeros((h_lr, w_lr), dtype=np.float32)
    base_y = scale * (np.arange(h_lr, dtype=np.float64) + dy)
    base_x = scale * (np.arange(w_lr, dtype=np.float64) + dx)
    for oy in range(scale):
        yy = base_y + float(oy)
        for ox in range(scale):
            xx = base_x + float(ox)
            coords = np.meshgrid(yy, xx, indexing="ij")
            out += ndimage.map_coordinates(
                image,
                coords,
                order=1,
                mode="constant",
                cval=0.0,
                prefilter=False,
            ).astype(np.float32, copy=False)
    out /= float(scale * scale)
    return out


def highpass_lr(image: np.ndarray, sigma_lr_px: float) -> np.ndarray:
    """High-pass filter one LR residual with reflect padding, matching EP07 DC monitoring."""

    arr = np.asarray(image, dtype=np.float32)
    sigma = float(sigma_lr_px)
    if sigma <= 0:
        return arr
    return (arr - ndimage.gaussian_filter(arr, sigma=sigma, mode="reflect")).astype(np.float32, copy=False)


def _mse_or_huber(residual: np.ndarray, *, huber_delta: float) -> float:
    r = np.asarray(residual, dtype=np.float64)
    valid = np.isfinite(r)
    if not np.any(valid):
        return float("nan")
    rv = r[valid]
    delta = float(huber_delta)
    if delta <= 0:
        return float(np.mean(rv * rv))
    a = np.abs(rv)
    loss = np.where(a <= delta, 0.5 * rv * rv, delta * (a - 0.5 * delta))
    return float(2.0 * np.mean(loss))


def _score_shift(
    blurred_hr: np.ndarray,
    frame: np.ndarray,
    shift: np.ndarray,
    *,
    scale: int,
    crop_margin: int,
    band_sigma_lr_px: float,
    huber_delta: float,
) -> tuple[float, float]:
    pred = forward_block_average_shifted(blurred_hr, shift, scale=scale)
    residual_full = pred - np.asarray(frame, dtype=np.float32)
    sl_y, sl_x = crop_slices(tuple(residual_full.shape), crop_margin)
    full_mse = _mse_or_huber(residual_full[sl_y, sl_x], huber_delta=huber_delta)
    residual_band = highpass_lr(residual_full, band_sigma_lr_px)
    band_mse = _mse_or_huber(residual_band[sl_y, sl_x], huber_delta=huber_delta)
    return band_mse, full_mse


def _score_one_frame(
    *,
    frame_idx: int,
    split: str,
    refine_mode: str,
    blurred_hr: np.ndarray,
    frame: np.ndarray,
    initial_shift: np.ndarray,
    file_name: str,
    candidate: PsfCandidate,
    offsets: np.ndarray,
    scale: int,
    crop_margin: int,
    band_sigma_lr_px: float,
    huber_delta: float,
) -> FrameScore:
    best: tuple[float, float, np.ndarray] | None = None
    for delta in offsets:
        shift = np.asarray(initial_shift, dtype=np.float32) + np.asarray(delta, dtype=np.float32)
        band_mse, full_mse = _score_shift(
            blurred_hr,
            frame,
            shift,
            scale=scale,
            crop_margin=crop_margin,
            band_sigma_lr_px=band_sigma_lr_px,
            huber_delta=huber_delta,
        )
        if best is None or band_mse < best[0]:
            best = (band_mse, full_mse, shift)
    if best is None:
        raise RuntimeError("empty shift-refinement offset grid")
    band_mse, full_mse, refined_shift = best
    delta = refined_shift - np.asarray(initial_shift, dtype=np.float32)
    return FrameScore(
        frame_index=int(frame_idx),
        file=str(file_name),
        split=split,
        refine_mode=refine_mode,
        candidate_label=candidate.label,
        sigma_x_lr_px=float(candidate.sigma_x_lr_px),
        sigma_y_lr_px=float(candidate.sigma_y_lr_px),
        angle_deg=float(candidate.angle_deg),
        initial_dx_px=float(initial_shift[0]),
        initial_dy_px=float(initial_shift[1]),
        delta_dx_px=float(delta[0]),
        delta_dy_px=float(delta[1]),
        refined_dx_px=float(refined_shift[0]),
        refined_dy_px=float(refined_shift[1]),
        band_mse=float(band_mse),
        band_rms=float(np.sqrt(max(band_mse, 0.0))),
        full_mse=float(full_mse),
        full_rms=float(np.sqrt(max(full_mse, 0.0))),
    )


def score_candidate(
    candidate: PsfCandidate,
    *,
    refine_mode: str,
    hr_image: np.ndarray,
    frames_raw: np.ndarray,
    metadata: pd.DataFrame,
    shifts: np.ndarray,
    split_indices: dict[str, np.ndarray],
    offsets: np.ndarray,
    scale: int,
    crop_margin: int,
    band_sigma_lr_px: float,
    huber_delta: float,
    workers: int,
) -> tuple[list[dict], list[dict]]:
    """Score one PSF candidate across deterministic train/validation frames."""

    blurred = blur_hr_anisotropic(hr_image, candidate, scale=scale)
    tasks = []
    for split, indices in split_indices.items():
        for frame_idx in indices:
            tasks.append(
                dict(
                    frame_idx=int(frame_idx),
                    split=split,
                    refine_mode=refine_mode,
                    blurred_hr=blurred,
                    frame=frames_raw[int(frame_idx)],
                    initial_shift=shifts[int(frame_idx)],
                    file_name=str(metadata.iloc[int(frame_idx)]["file"]),
                    candidate=candidate,
                    offsets=offsets,
                    scale=scale,
                    crop_margin=crop_margin,
                    band_sigma_lr_px=band_sigma_lr_px,
                    huber_delta=huber_delta,
                )
            )
    n_workers = min(max(1, int(workers)), max(1, len(tasks)))
    if n_workers == 1:
        scores = [_score_one_frame(**task) for task in tasks]
    else:
        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            scores = list(executor.map(lambda task: _score_one_frame(**task), tasks))

    frame_rows = [score.__dict__ for score in scores]
    frame_df = pd.DataFrame(frame_rows)
    summary_rows: list[dict] = []
    for split, part in frame_df.groupby("split", sort=False):
        delta_norm = np.hypot(part["delta_dx_px"].to_numpy(dtype=float), part["delta_dy_px"].to_numpy(dtype=float))
        summary_rows.append(
            {
                "candidate_label": candidate.label,
                "split": split,
                "refine_mode": refine_mode,
                "sigma_x_lr_px": float(candidate.sigma_x_lr_px),
                "sigma_y_lr_px": float(candidate.sigma_y_lr_px),
                "angle_deg": float(candidate.angle_deg),
                "anisotropy_ratio": float(candidate.anisotropy_ratio),
                "n_frames": int(len(part)),
                "band_mse_mean": float(part["band_mse"].mean()),
                "band_rms_mean": float(part["band_rms"].mean()),
                "band_rms_from_mean_mse": float(np.sqrt(max(float(part["band_mse"].mean()), 0.0))),
                "band_rms_median": float(part["band_rms"].median()),
                "full_rms_from_mean_mse": float(np.sqrt(max(float(part["full_mse"].mean()), 0.0))),
                "shift_delta_norm_mean_px": float(delta_norm.mean()) if len(delta_norm) else 0.0,
                "shift_delta_norm_p95_px": float(np.percentile(delta_norm, 95)) if len(delta_norm) else 0.0,
            }
        )
    return summary_rows, frame_rows


def _limited_split(
    n_frames: int,
    *,
    val_stride: int,
    max_train_score: int | None,
    max_val_score: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    return deterministic_split(
        n_frames,
        val_stride=val_stride,
        max_train=max_train_score,
        max_val=max_val_score,
    )


def _best_train_candidate(sweep: pd.DataFrame, *, refine_mode: str = "bounded") -> pd.Series:
    train = sweep[sweep["split"].eq("train") & sweep["refine_mode"].eq(refine_mode)].sort_values(
        ["band_mse_mean", "sigma_x_lr_px", "sigma_y_lr_px"]
    )
    if train.empty:
        raise ValueError(f"sweep has no train rows for refine_mode={refine_mode!r}")
    return train.iloc[0]


def _baseline_candidate() -> PsfCandidate:
    return PsfCandidate(BASELINE_SIGMA_LR_PX, BASELINE_SIGMA_LR_PX, 0.0)


def _summary_for_candidate(sweep: pd.DataFrame, candidate: PsfCandidate, split: str, *, refine_mode: str) -> dict:
    rows = sweep[
        sweep["split"].eq(split)
        & sweep["refine_mode"].eq(refine_mode)
        & np.isclose(sweep["sigma_x_lr_px"], candidate.sigma_x_lr_px)
        & np.isclose(sweep["sigma_y_lr_px"], candidate.sigma_y_lr_px)
        & np.isclose(sweep["angle_deg"], candidate.angle_deg)
    ]
    if rows.empty:
        return {}
    row = rows.iloc[0]
    return {str(k): (v.item() if isinstance(v, np.generic) else v) for k, v in row.to_dict().items()}


def run_stage0a_mvp(
    *,
    output_dir: str | Path = DEFAULT_STAGE0A_OUTPUT_DIR,
    data_dir: str | Path = DEFAULT_DATA_DIR,
    frame_audit_csv: str | Path = DEFAULT_FRAME_AUDIT_CSV,
    alignment_csv: str | Path = DEFAULT_ALIGNMENT_CSV,
    alignment_method: str = "contour_refined",
    alignment_source: str = "auto",
    stage_config_path: str | Path = DEFAULT_STAGE_CONFIG,
    highpass_sigma: float = 5.0,
    band_sigma_lr_px: float = 5.0,
    sigmas: Iterable[float] = (0.2, 0.3, 0.4, 0.5),
    anisotropy_ratios: Iterable[float] = (1.0, 1.5),
    angles: Iterable[float] = (0.0, 45.0, 90.0, 135.0),
    shift_refine_radius: float = 0.10,
    shift_refine_step: float = 0.10,
    val_stride: int = 5,
    max_train_score: int | None = 48,
    max_val_score: int | None = 32,
    crop_margin: int = 8,
    workers: int = default_workers(),
    limit: int | None = None,
    scale: int = 2,
    huber_delta: float = 0.0,
    save_xhat: bool = False,
) -> Stage0aResult:
    """Run the bounded Stage 0a MVP diagnostic.

    The full 248 clean burst is loaded and used for the fixed SAA estimate when
    ``limit`` is not set.  The residual sweep itself is intentionally
    sub-sampled by default; set ``max_train_score=None`` and
    ``max_val_score=None`` from Python, or use the CLI's ``--use-all-score``,
    for a slower full-score pass.
    """

    start = time.perf_counter()
    out_dir = ensure_dir(output_dir)
    inputs = load_stage0a_inputs(
        data_dir=data_dir,
        frame_audit_csv=frame_audit_csv,
        alignment_csv=alignment_csv,
        alignment_method=alignment_method,
        alignment_source=alignment_source,
        stage_config_path=stage_config_path,
        highpass_sigma=highpass_sigma,
        workers=workers,
        limit=limit,
    )
    train_idx, val_idx = _limited_split(
        len(inputs.metadata),
        val_stride=val_stride,
        max_train_score=max_train_score,
        max_val_score=max_val_score,
    )
    split_indices = {"train": train_idx, "val": val_idx}
    xhat = reconstruct_saa(
        inputs.frames_highpass,
        inputs.shifts,
        scale=scale,
        workers=workers,
        fill_missing=True,
    ).astype(np.float32, copy=False)
    candidates = build_psf_candidates(sigmas=sigmas, anisotropy_ratios=anisotropy_ratios, angles=angles)
    baseline = _baseline_candidate()
    if not any(
        np.isclose(c.sigma_x_lr_px, baseline.sigma_x_lr_px)
        and np.isclose(c.sigma_y_lr_px, baseline.sigma_y_lr_px)
        and np.isclose(c.angle_deg, baseline.angle_deg)
        for c in candidates
    ):
        candidates.insert(0, baseline)
    offsets = shift_refine_offsets(shift_refine_radius, shift_refine_step)
    zero_offsets = shift_refine_offsets(0.0, shift_refine_step)

    sweep_rows: list[dict] = []
    frame_rows: list[dict] = []
    baseline_summary, baseline_frames = score_candidate(
        baseline,
        refine_mode="none",
        hr_image=xhat,
        frames_raw=inputs.frames_raw,
        metadata=inputs.metadata,
        shifts=inputs.shifts,
        split_indices=split_indices,
        offsets=zero_offsets,
        scale=scale,
        crop_margin=crop_margin,
        band_sigma_lr_px=band_sigma_lr_px,
        huber_delta=huber_delta,
        workers=workers,
    )
    sweep_rows.extend(baseline_summary)
    frame_rows.extend(baseline_frames)
    for candidate in candidates:
        summary_part, frame_part = score_candidate(
            candidate,
            refine_mode="bounded",
            hr_image=xhat,
            frames_raw=inputs.frames_raw,
            metadata=inputs.metadata,
            shifts=inputs.shifts,
            split_indices=split_indices,
            offsets=offsets,
            scale=scale,
            crop_margin=crop_margin,
            band_sigma_lr_px=band_sigma_lr_px,
            huber_delta=huber_delta,
            workers=workers,
        )
        sweep_rows.extend(summary_part)
        frame_rows.extend(frame_part)

    sweep = pd.DataFrame(sweep_rows).sort_values(["split", "band_mse_mean", "candidate_label"]).reset_index(drop=True)
    frame_scores = pd.DataFrame(frame_rows).sort_values(["split", "frame_index", "refine_mode", "candidate_label"]).reset_index(drop=True)
    best_row = _best_train_candidate(sweep, refine_mode="bounded")
    best = PsfCandidate(
        float(best_row["sigma_x_lr_px"]),
        float(best_row["sigma_y_lr_px"]),
        float(best_row["angle_deg"]),
    )
    best_refinements = frame_scores[
        frame_scores["refine_mode"].eq("bounded")
        &
        np.isclose(frame_scores["sigma_x_lr_px"], best.sigma_x_lr_px)
        & np.isclose(frame_scores["sigma_y_lr_px"], best.sigma_y_lr_px)
        & np.isclose(frame_scores["angle_deg"], best.angle_deg)
    ].copy()
    baseline_train = _summary_for_candidate(sweep, baseline, "train", refine_mode="none")
    baseline_val = _summary_for_candidate(sweep, baseline, "val", refine_mode="none")
    best_train = _summary_for_candidate(sweep, best, "train", refine_mode="bounded")
    best_val = _summary_for_candidate(sweep, best, "val", refine_mode="bounded")
    baseline_val_mse = float(baseline_val.get("band_mse_mean", np.nan))
    best_val_mse = float(best_val.get("band_mse_mean", np.nan))
    val_improvement_pct = (
        100.0 * (baseline_val_mse - best_val_mse) / baseline_val_mse
        if np.isfinite(baseline_val_mse) and baseline_val_mse > 0 and np.isfinite(best_val_mse)
        else float("nan")
    )

    summary = {
        "stage": "solver_v2_stage0a_mvp",
        "purpose": "bounded real 248-frame operator calibration scaffold; not full joint high-dimensional optimization",
        "n_loaded_frames": int(len(inputs.metadata)),
        "n_train_score_frames": int(len(train_idx)),
        "n_val_score_frames": int(len(val_idx)),
        "shift_source": inputs.shift_source,
        "alignment_method": alignment_method,
        "detector_pitch_um": float(inputs.detector_pitch_um),
        "current_spatial_resolution_um": float(inputs.spatial_resolution_um),
        "pitch_contract": "20um_per_detector_pixel; previous 10um scale is not used",
        "xhat": {
            "method": "SAA over loaded clean highpass frames",
            "highpass_sigma_lr_px": float(highpass_sigma),
            "scale": int(scale),
            "shape": list(map(int, xhat.shape)),
        },
        "scoring": {
            "band_sigma_lr_px": float(band_sigma_lr_px),
            "crop_margin_lr_px": int(crop_margin),
            "huber_delta": float(huber_delta),
            "shift_refine_radius_lr_px": float(shift_refine_radius),
            "shift_refine_step_lr_px": float(shift_refine_step),
            "n_shift_offsets": int(len(offsets)),
            "selection_metric": "train band_mse_mean",
            "full_residual_note": "full residual is reported for debugging only because xhat is highpass by default",
        },
        "baseline_sigma_0p5_iso_no_refine": {
            "train": baseline_train,
            "val": baseline_val,
        },
        "best_train_selected": {
            "candidate": best.__dict__,
            "train": best_train,
            "val": best_val,
        },
        "dc_resid_floor_probe": {
            "baseline_val_band_rms_from_mean_mse": float(baseline_val.get("band_rms_from_mean_mse", np.nan)),
            "best_val_band_rms_from_mean_mse": float(best_val.get("band_rms_from_mean_mse", np.nan)),
            "val_band_mse_improvement_pct": float(val_improvement_pct),
            "interpretation": "diagnostic floor movement only; not a resolution or temperature-metrology claim",
        },
        "outputs": {
            "sweep_csv": relative(out_dir / "stage0a_model_sweep.csv"),
            "frame_scores_csv": relative(out_dir / "stage0a_frame_scores.csv"),
            "shift_refinements_csv": relative(out_dir / "stage0a_best_shift_refinements.csv"),
            "summary_json": relative(out_dir / "stage0a_summary.json"),
        },
        "elapsed_sec": float(time.perf_counter() - start),
    }

    sweep.to_csv(out_dir / "stage0a_model_sweep.csv", index=False)
    frame_scores.to_csv(out_dir / "stage0a_frame_scores.csv", index=False)
    best_refinements.to_csv(out_dir / "stage0a_best_shift_refinements.csv", index=False)
    if save_xhat:
        np.save(out_dir / "stage0a_saa_xhat_highpass.npy", xhat.astype(np.float32, copy=False))
        summary["outputs"]["xhat_npy"] = relative(out_dir / "stage0a_saa_xhat_highpass.npy")
    write_json(out_dir / "stage0a_summary.json", summary)
    return Stage0aResult(
        sweep=sweep,
        frame_scores=frame_scores,
        shift_refinements=best_refinements,
        summary=summary,
    )
