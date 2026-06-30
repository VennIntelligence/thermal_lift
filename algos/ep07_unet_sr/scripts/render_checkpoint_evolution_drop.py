"""Render small checkpoint-evolution drops for local/remote review.

This script is intentionally a drop-pack builder, not a training/evaluation
harness.  It renders a few selected checkpoints to compact PNG/NPZ artifacts
that can be copied to an analysis host inbox without syncing full output trees
or checkpoint weights.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from matplotlib import pyplot as plt
from tcforge.highpass import highpass_preprocess

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "algos" / "ep06_sr_poc" / "src"))

from unet_sr.dataset import HYBRID_DRIZZLE_MEAN_CHANNEL  # noqa: E402
from unet_sr.inference import infer_from_burst  # noqa: E402
from unet_sr.model import ThermalSRUNet  # noqa: E402
from unet_sr.real_eval import (  # noqa: E402
    _load_real_eval_cache,
    _shared_vmax,
    infer_solver_from_burst,
    save_ep11_temperature_figure,
    zoom_center,
)
from unet_sr.unroll import UnrolledSolver  # noqa: E402


DEFAULT_V10 = [
    "checkpoint_step_005000.pt",
    "checkpoint_step_010000.pt",
    "checkpoint_step_020000.pt",
    "checkpoint_step_030000.pt",
    "checkpoint_step_040000.pt",
]
DEFAULT_SOLVER = [
    "solver_step_002500.pt",
    "solver_step_005000.pt",
    "solver_step_010000.pt",
    "solver_step_015000.pt",
    "solver_step_020000.pt",
]


@dataclass(frozen=True)
class ArtifactRecord:
    run_id: str
    method: str
    checkpoint: str
    step: int
    temperature_png: str
    highpass_png: str
    temperature_npz: str
    mean_c: float
    std_c: float
    min_c: float
    max_c: float
    render_seconds: float


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _step_from_checkpoint(path: Path, ckpt: dict[str, Any]) -> int:
    if "step" in ckpt and int(ckpt["step"]) > 0:
        return int(ckpt["step"])
    match = re.search(r"(?:step_?|_)(0*\d+)", path.stem)
    return int(match.group(1)) if match else 0


def _load_checkpoint(path: Path) -> tuple[dict[str, Any], dict[str, torch.Tensor], int]:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"unsupported checkpoint format: {path}")
    cfg = dict(ckpt.get("config") or {})
    return cfg, ckpt["model_state_dict"], _step_from_checkpoint(path, ckpt)


def _build_unet(cfg: dict[str, Any], state: dict[str, torch.Tensor]) -> ThermalSRUNet:
    input_mode = str(cfg.get("input_mode", "lr"))
    residual = bool(cfg.get("residual", False))
    residual_mode = str(cfg.get("residual_mode", "none"))
    model_scale = 1 if (residual or input_mode == "hybrid_drizzle2x" or residual_mode == "drizzle2x") else int(cfg.get("scale", 2))
    model = ThermalSRUNet(
        in_channels=int(cfg.get("in_channels", 5)),
        out_channels=int(cfg.get("out_channels", 1)),
        base_channels=int(cfg.get("base_channels", 64)),
        scale=model_scale,
        hr_upsampler=str(cfg.get("hr_upsampler", "bilinear")),
        hr_res_blocks=int(cfg.get("hr_res_blocks", 0)),
    )
    model.load_state_dict(state)
    model.eval()
    return model


def _build_solver(cfg: dict[str, Any], state: dict[str, torch.Tensor]) -> UnrolledSolver:
    cond_channels = 5 if bool(cfg.get("solver_no_drizzle", False)) else int(cfg.get("in_channels", 9))
    solver = UnrolledSolver(
        n_steps=int(cfg.get("unroll_steps", 4)),
        cond_channels=cond_channels,
        base_channels=int(cfg.get("base_channels", 64)),
        scale=int(cfg.get("scale", 2)),
        share_weights=bool(cfg.get("solver_share_weights", True)),
        band_highpass_sigma_lr_px=float(cfg.get("solver_band_sigma", 5.0)),
        huber_delta=float(cfg.get("solver_huber_delta", 0.0)),
        eta_init=float(cfg.get("solver_eta_init", 0.5)),
        learn_eta=bool(cfg.get("solver_learn_eta", False)),
        prox_use_se=bool(cfg.get("solver_prox_use_se", True)),
        prox_norm=str(cfg.get("solver_prox_norm", "group")),
    )
    solver.load_state_dict(state)
    solver.eval()
    return solver


def _save_highpass_png(image: np.ndarray, output_path: Path, *, title: str, center_fraction: float, zoom: float) -> Path:
    hp_zoom = zoom_center(image, center_fraction=center_fraction, zoom=zoom)
    vmax = _shared_vmax([hp_zoom])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(4.1, 3.0), squeeze=True)
    im = ax.imshow(hp_zoom, cmap="coolwarm", vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04).set_label("Highpass [deg C]")
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return output_path


def _stats(image: np.ndarray) -> dict[str, float]:
    finite = image[np.isfinite(image)]
    return {
        "mean_c": float(np.mean(finite)),
        "std_c": float(np.std(finite)),
        "min_c": float(np.min(finite)),
        "max_c": float(np.max(finite)),
    }


def _render_unet(
    checkpoint: Path,
    out_dir: Path,
    raw_frames: np.ndarray,
    shifts: np.ndarray,
    *,
    device: str,
    center_fraction: float,
    zoom: float,
    overlap: int,
) -> ArtifactRecord:
    cfg, state, step = _load_checkpoint(checkpoint)
    model = _build_unet(cfg, state)
    residual_mode = str(cfg.get("residual_mode", "none"))
    residual_channel = HYBRID_DRIZZLE_MEAN_CHANNEL if residual_mode == "drizzle2x" else None
    t0 = time.monotonic()
    temp = infer_from_burst(
        model,
        raw_frames,
        shifts,
        scale=int(cfg.get("scale", 2)),
        patch_size_hr=int(cfg.get("patch_size_hr", 256)),
        overlap=overlap,
        device=device,
        sigma_bg=float(cfg.get("highpass_sigma", 5.0)),
        residual=bool(cfg.get("residual", False)),
        input_mode=str(cfg.get("input_mode", "lr")),
        residual_channel=residual_channel,
    ).astype(np.float32, copy=False)
    elapsed = time.monotonic() - t0
    return _save_record(
        run_id=checkpoint.parent.name,
        method="V10",
        checkpoint=checkpoint,
        step=step,
        temp=temp,
        out_dir=out_dir,
        scale=int(cfg.get("scale", 2)),
        sigma_bg=float(cfg.get("highpass_sigma", 5.0)),
        center_fraction=center_fraction,
        zoom=zoom,
        elapsed=elapsed,
    )


def _render_solver(
    checkpoint: Path,
    out_dir: Path,
    raw_frames: np.ndarray,
    shifts: np.ndarray,
    *,
    device: str,
    center_fraction: float,
    zoom: float,
    overlap: int,
    solver_patch_size_hr: int,
) -> ArtifactRecord:
    cfg, state, step = _load_checkpoint(checkpoint)
    solver = _build_solver(cfg, state)
    eval_cfg = dict(cfg)
    eval_cfg["patch_size_hr"] = int(solver_patch_size_hr)
    t0 = time.monotonic()
    temp = infer_solver_from_burst(
        solver,
        raw_frames,
        shifts,
        training_config=SimpleNamespace(**eval_cfg),
        patch_size_hr=int(solver_patch_size_hr),
        overlap=overlap,
        device=device,
    ).astype(np.float32, copy=False)
    elapsed = time.monotonic() - t0
    return _save_record(
        run_id=checkpoint.parent.name,
        method="Solver",
        checkpoint=checkpoint,
        step=step,
        temp=temp,
        out_dir=out_dir,
        scale=int(cfg.get("scale", 2)),
        sigma_bg=float(cfg.get("highpass_sigma", 5.0)),
        center_fraction=center_fraction,
        zoom=zoom,
        elapsed=elapsed,
    )


def _save_record(
    *,
    run_id: str,
    method: str,
    checkpoint: Path,
    step: int,
    temp: np.ndarray,
    out_dir: Path,
    scale: int,
    sigma_bg: float,
    center_fraction: float,
    zoom: float,
    elapsed: float,
) -> ArtifactRecord:
    run_out = out_dir / run_id
    stem = f"{method.lower()}_step{step:06d}"
    npz_path = run_out / f"{stem}_temperature_c.npz"
    temp_png = run_out / f"{stem}_center_zoom{int(zoom)}x_temperature.png"
    hp_png = run_out / f"{stem}_center_zoom{int(zoom)}x_highpass.png"
    run_out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(npz_path, temperature_c=temp)
    vmin, vmax = float(np.percentile(temp[np.isfinite(temp)], 1.0)), float(np.percentile(temp[np.isfinite(temp)], 99.0))
    save_ep11_temperature_figure(
        temp,
        temp_png,
        zoom=zoom,
        center_fraction=center_fraction,
        step=step,
        scale=scale,
        method_label=method,
        vmin=vmin,
        vmax=vmax,
    )
    hp = highpass_preprocess(temp, sigma_bg=sigma_bg)
    _save_highpass_png(
        hp,
        hp_png,
        title=f"{method} {scale}x @ EP07 step {step} (highpass)",
        center_fraction=center_fraction,
        zoom=zoom,
    )
    vals = _stats(temp)
    return ArtifactRecord(
        run_id=run_id,
        method=method,
        checkpoint=_rel(checkpoint),
        step=step,
        temperature_png=_rel(temp_png),
        highpass_png=_rel(hp_png),
        temperature_npz=_rel(npz_path),
        render_seconds=float(elapsed),
        **vals,
    )


def _write_readme(out_dir: Path, records: list[ArtifactRecord], args: argparse.Namespace) -> None:
    lines = [
        "# Checkpoint Evolution Drop",
        "",
        "Small local-to-analysis-host drop package. This is not a full output-tree sync.",
        "",
        f"- Source commit: `{_git_sha()}`",
        f"- Frame limit: `{args.frame_limit}` clean main-session frames",
        f"- Alignment method: `{args.alignment_method}`",
        f"- Center fraction / display zoom: `{args.center_fraction}` / `{args.zoom}`",
        f"- V10 overlap: `{args.overlap}`",
        f"- Solver render patch / overlap: `{args.solver_patch_size_hr}` / `{args.solver_overlap}`",
        "",
        "## Records",
        "",
        "| run | method | step | temperature PNG | highpass PNG | data | mean C | render s |",
        "|---|---:|---:|---|---|---|---:|---:|",
    ]
    for rec in records:
        lines.append(
            f"| {rec.run_id} | {rec.method} | {rec.step} | `{rec.temperature_png}` | "
            f"`{rec.highpass_png}` | `{rec.temperature_npz}` | {rec.mean_c:.4f} | {rec.render_seconds:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Boundaries",
            "",
            "- PNGs are center-ROI visual diagnostics for checkpoint evolution.",
            "- `.npz` arrays contain rendered full-frame temperature maps in deg C.",
            "- No checkpoint weights or raw data are included.",
            "- Solver real-data rendering uses the configured scalar Gaussian PSF because real data has no synthetic per-scene PSF metadata.",
        ]
    )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _git_sha() -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True).strip()
    except Exception:
        return "unknown"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "output" / "remote_drop" / "20260627_checkpoint_evolution")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--frame-limit", type=int, default=248)
    parser.add_argument("--alignment-method", default="contour_refined")
    parser.add_argument("--center-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--zoom", type=float, default=3.0)
    parser.add_argument("--overlap", type=int, default=128)
    parser.add_argument("--solver-overlap", type=int, default=128)
    parser.add_argument("--solver-patch-size-hr", type=int, default=256)
    parser.add_argument("--skip-v10", action="store_true")
    parser.add_argument("--skip-solver", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    raw_frames, shifts = _load_real_eval_cache(args.frame_limit, args.alignment_method)
    records: list[ArtifactRecord] = []

    if not args.skip_v10:
        run_dir = PROJECT_ROOT / "algos" / "ep07_unet_sr" / "outputs" / "v10_v4_acl027"
        for name in DEFAULT_V10:
            ckpt = run_dir / name
            print(f"render V10 {ckpt}")
            records.append(
                _render_unet(
                    ckpt,
                    args.output_dir,
                    raw_frames,
                    shifts,
                    device=args.device,
                    center_fraction=args.center_fraction,
                    zoom=args.zoom,
                    overlap=args.overlap,
                )
            )

    if not args.skip_solver:
        run_dir = PROJECT_ROOT / "algos" / "ep07_unet_sr" / "outputs" / "solver_v4_acl027"
        for name in DEFAULT_SOLVER:
            ckpt = run_dir / name
            print(f"render solver {ckpt}")
            records.append(
                _render_solver(
                    ckpt,
                    args.output_dir,
                    raw_frames,
                    shifts,
                    device=args.device,
                    center_fraction=args.center_fraction,
                    zoom=args.zoom,
                    overlap=args.solver_overlap,
                    solver_patch_size_hr=args.solver_patch_size_hr,
                )
            )

    manifest = {
        "source_commit": _git_sha(),
        "args": vars(args) | {"output_dir": _rel(args.output_dir)},
        "records": [asdict(r) for r in records],
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    _write_readme(args.output_dir, records, args)
    print(f"wrote {_rel(args.output_dir)}")


if __name__ == "__main__":
    main()
