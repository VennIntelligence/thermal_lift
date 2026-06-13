"""Shared helpers for the tracked EP07 V9 review reproduction pipeline."""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tcforge.classical_sr import drizzle_features
from tcforge.highpass import highpass_preprocess

from unet_sr.dataset import HYBRID_DRIZZLE_MEAN_CHANNEL
from unet_sr.inference import infer_from_burst
from unet_sr.model import ThermalSRUNet
from unet_sr.real_eval import (
    _import_ep06_common,
    _load_real_eval_cache,
    _temperature_limits,
    pearson_finite,
    zoom_center,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ALGO_ROOT = PROJECT_ROOT / "algos" / "ep07_unet_sr"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "output" / "ep07_v9_review"
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_DIR / "cache"
DEFAULT_RUN_ROOT = ALGO_ROOT / "outputs"
DEFAULT_TGV_PATH = PROJECT_ROOT / "output" / "ep10_tgv_sr" / "best_hr_temperature.npy"

FINE_ROW_SLICE = slice(384, 518)
FINE_COL_SLICE = slice(478, 674)
CENTER_FRACTION = 1.0 / 3.0
ZOOM = 3.0
HIGHPASS_SIGMA = 5.0

DEFAULT_EVAL_RUNS = [
    "ep07_v8_1a_loss_cooldown",
    "ep07_v9b_fwd_consistency",
    "ep07_v9d_fwd_fullband",
    "ep07_v9a_hybrid_drizzle",
    "ep07_v9c_hybrid_legal_fwd",
]

DEFAULT_V9A_STEPS = [
    5000,
    10000,
    15000,
    20000,
    25000,
    30000,
    40000,
    45000,
    50000,
    55000,
    60000,
]


@dataclass(frozen=True)
class CheckpointSpec:
    """A single checkpoint to render or score."""

    label: str
    run_dir: Path
    step: int
    input_mode: str


@dataclass(frozen=True)
class ReferenceImages:
    raw_control: np.ndarray
    drizzle_mean: np.ndarray
    tgv: np.ndarray


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.grid": False,
        }
    )


def project_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    return p if p.is_absolute() else (PROJECT_ROOT / p).resolve()


def ensure_output_dirs(output_dir: Path, cache_dir: Path | None = None) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)


def parse_int_list(text: str) -> list[int]:
    values = [item.strip() for item in text.split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one comma-separated integer")
    try:
        return [int(item) for item in values]
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid integer list: {text}") from exc


def fine_window(image: np.ndarray) -> np.ndarray:
    """Return the fixed fine-zigzag window on the raw 2x grid."""

    return np.asarray(image)[FINE_ROW_SLICE, FINE_COL_SLICE]


def highpass_fine(image: np.ndarray, *, sigma_bg: float = HIGHPASS_SIGMA) -> np.ndarray:
    return fine_window(highpass_preprocess(image, sigma_bg=sigma_bg))


def lattice_score(hp_win: np.ndarray) -> float:
    x = np.asarray(hp_win, dtype=np.float64)
    x = x - np.nanmean(x)
    f = np.abs(np.fft.fftshift(np.fft.fft2(np.nan_to_num(x)))) ** 2
    rows, cols = x.shape
    fy = np.fft.fftshift(np.fft.fftfreq(rows))[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(cols))[None, :]
    band = (np.abs(fy) > 0.35) | (np.abs(fx) > 0.35)
    return float(f[band].sum() / max(float(f.sum()), 1e-12))


def sharp_p95(temp_win: np.ndarray) -> float:
    gy, gx = np.gradient(np.asarray(temp_win, dtype=np.float64))
    return float(np.nanpercentile(np.hypot(gy, gx), 95.0))


def metric_row(
    name: str,
    temp: np.ndarray,
    *,
    drizzle_hp_fine: np.ndarray,
    tgv_hp_fine: np.ndarray | None,
    step: int | None = None,
    sigma_bg: float = HIGHPASS_SIGMA,
) -> dict[str, float | int | str | None]:
    hp_f = highpass_fine(temp, sigma_bg=sigma_bg)
    return {
        "name": name,
        "step": step,
        "hp_corr_input": pearson_finite(hp_f, drizzle_hp_fine),
        "hp_corr_tgv": pearson_finite(hp_f, tgv_hp_fine) if tgv_hp_fine is not None else float("nan"),
        "sharp_p95": sharp_p95(fine_window(temp)),
        "lattice_score": lattice_score(hp_f),
    }


def load_real_inputs(
    *,
    frame_limit: int = 248,
    alignment_method: str = "contour_refined",
    tgv_path: Path = DEFAULT_TGV_PATH,
) -> tuple[np.ndarray, np.ndarray, ReferenceImages]:
    # This file is imported as module name "common" when scripts are executed
    # from scripts/v9_review/.  EP06 also exposes a package named "common";
    # temporarily unshadow it while real_eval imports common.alignment.
    v9_common_module = sys.modules.get("common")
    if v9_common_module is sys.modules.get(__name__):
        sys.modules.pop("common", None)
    try:
        raw_frames, shifts = _load_real_eval_cache(frame_limit, alignment_method)
        _, _, bicubic_upsample = _import_ep06_common()
    finally:
        if v9_common_module is not None:
            sys.modules["common"] = v9_common_module
    raw_control = bicubic_upsample(np.nanmean(raw_frames, axis=0), scale=2)
    drizzle_mean = drizzle_features(raw_frames, shifts, scale=2, kernel="bilinear")[0]
    tgv = np.load(tgv_path).astype(np.float32, copy=False)
    refs = ReferenceImages(
        raw_control=raw_control.astype(np.float32, copy=False),
        drizzle_mean=drizzle_mean.astype(np.float32, copy=False),
        tgv=tgv,
    )
    return raw_frames, shifts, refs


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def _cache_label(label: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", label).strip("_")


def checkpoint_cache_path(cache_dir: Path, spec: CheckpointSpec) -> Path:
    return cache_dir / f"{_cache_label(spec.label)}_step{spec.step}_temperature.npy"


def infer_checkpoint_cached(
    spec: CheckpointSpec,
    *,
    cache_dir: Path,
    raw_frames: np.ndarray,
    shifts: np.ndarray,
    device: str = "cpu",
    force: bool = False,
    patch_size_hr: int = 256,
    overlap: int = 128,
    scale: int = 2,
    sigma_bg: float = HIGHPASS_SIGMA,
) -> np.ndarray:
    """Run one checkpoint through real-data inference, using an npy cache."""

    cache_path = checkpoint_cache_path(cache_dir, spec)
    if cache_path.exists() and not force:
        return np.load(cache_path).astype(np.float32, copy=False)

    import torch

    resolved_device = resolve_device(device)
    ckpt_path = spec.run_dir / f"checkpoint_step_{spec.step:06d}.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model_scale = 1 if spec.input_mode == "hybrid_drizzle2x" else int(cfg["scale"])
    # V10 residual-over-observation: model emits delta and the drizzle mean base
    # (hybrid channel 5) must be re-added at inference, exactly as real_eval does.
    # Without this the harness scores the bare residual field, not drizzle + delta.
    residual_mode = str(cfg.get("residual_mode", "none"))
    residual_channel = HYBRID_DRIZZLE_MEAN_CHANNEL if residual_mode == "drizzle2x" else None
    model = ThermalSRUNet(
        in_channels=int(cfg["in_channels"]),
        out_channels=int(cfg["out_channels"]),
        base_channels=int(cfg["base_channels"]),
        scale=model_scale,
        hr_upsampler=str(cfg["hr_upsampler"]),
        hr_res_blocks=int(cfg["hr_res_blocks"]),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    pred = infer_from_burst(
        model,
        raw_frames,
        shifts,
        scale=scale,
        patch_size_hr=patch_size_hr,
        overlap=overlap,
        device=resolved_device,
        residual=bool(cfg.get("residual", False)),
        sigma_bg=sigma_bg,
        input_mode=spec.input_mode,
        residual_channel=residual_channel,
    ).astype(np.float32, copy=False)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, pred)
    return pred


def default_v9a_specs(run_root: Path = DEFAULT_RUN_ROOT) -> list[CheckpointSpec]:
    run_dir = run_root / "ep07_v9a_hybrid_drizzle"
    return [
        CheckpointSpec(label=f"v9a_{step // 1000}k", run_dir=run_dir, step=step, input_mode="hybrid_drizzle2x")
        for step in DEFAULT_V9A_STEPS
    ]


def default_panel_specs(run_root: Path = DEFAULT_RUN_ROOT) -> list[CheckpointSpec]:
    return [
        CheckpointSpec(
            label="v8_1a_60k",
            run_dir=run_root / "ep07_v8_1a_loss_cooldown",
            step=60000,
            input_mode="lr",
        ),
        CheckpointSpec(
            label="v9a_10k",
            run_dir=run_root / "ep07_v9a_hybrid_drizzle",
            step=10000,
            input_mode="hybrid_drizzle2x",
        ),
        CheckpointSpec(
            label="v9a_60k",
            run_dir=run_root / "ep07_v9a_hybrid_drizzle",
            step=60000,
            input_mode="hybrid_drizzle2x",
        ),
    ]


def parse_checkpoint_spec(text: str, *, run_root: Path = DEFAULT_RUN_ROOT) -> CheckpointSpec:
    """Parse LABEL=RUN_DIR:STEP:INPUT_MODE or RUN_DIR:STEP:INPUT_MODE."""

    label: str | None = None
    rhs = text
    if "=" in text:
        label, rhs = text.split("=", 1)
        label = label.strip()
    parts = [part.strip() for part in rhs.split(":")]
    if len(parts) != 3:
        raise argparse.ArgumentTypeError(
            "checkpoint spec must be LABEL=RUN_DIR:STEP:INPUT_MODE or RUN_DIR:STEP:INPUT_MODE"
        )
    run_text, step_text, input_mode = parts
    try:
        step = int(step_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid checkpoint step: {step_text}") from exc
    run_dir = Path(run_text).expanduser()
    if not run_dir.is_absolute():
        candidate = run_root / run_dir
        run_dir = candidate if candidate.exists() else project_path(run_dir)
    if label is None or not label:
        label = f"{run_dir.name}_{step // 1000}k"
    return CheckpointSpec(label=label, run_dir=run_dir, step=step, input_mode=input_mode)


def save_temperature_panel(path: Path, image: np.ndarray, *, title: str) -> Path:
    configure_matplotlib()
    z = zoom_center(image, center_fraction=CENTER_FRACTION, zoom=ZOOM)
    vmin, vmax = _temperature_limits(image)
    fig, ax = plt.subplots(figsize=(4.1, 3.0))
    im = ax.imshow(z, cmap="inferno", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Temperature [deg C]")
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_fine_panel(path: Path, panels: list[tuple[str, np.ndarray]], *, title: str, ncols: int = 3) -> Path:
    configure_matplotlib()
    n = len(panels)
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.4 * nrows), squeeze=False)
    for ax, (name, image) in zip(axes.ravel(), panels):
        vmin, vmax = _temperature_limits(image)
        ax.imshow(fine_window(image), cmap="inferno", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(name, fontsize=12)
        ax.set_xticks([])
        ax.set_yticks([])
    for ax in axes.ravel()[n:]:
        ax.axis("off")
    fig.suptitle(title, fontsize=14)
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


def save_pareto_scatter(path: Path, df: pd.DataFrame) -> Path:
    configure_matplotlib()
    sub = df.dropna(subset=["step"])
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(sub.hp_corr_input, sub.sharp_p95, c=sub.step, cmap="viridis", s=60, zorder=3)
    for _, row in sub.iterrows():
        ax.annotate(
            f"{int(row.step) // 1000}K",
            (row.hp_corr_input, row.sharp_p95),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=9,
        )
    for ref_name, marker in [("input_drizzle", "*"), ("tgv", "P")]:
        ref = df[df.name == ref_name]
        if ref.empty:
            continue
        row = ref.iloc[0]
        ax.scatter(row.hp_corr_input, row.sharp_p95, marker=marker, s=180, color="crimson", zorder=4)
        ax.annotate(
            ref_name,
            (row.hp_corr_input, row.sharp_p95),
            textcoords="offset points",
            xytext=(6, -12),
            fontsize=10,
            color="crimson",
        )
    ax.set_xlabel("Fine-window highpass corr vs drizzle input (fidelity)")
    ax.set_ylabel("Fine-window P95 |gradient| (sharpness proxy)")
    ax.set_title("V9 training-time Pareto: fidelity vs sharpness (fine zigzag window)")
    fig.colorbar(sc, ax=ax).set_label("training step")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path
