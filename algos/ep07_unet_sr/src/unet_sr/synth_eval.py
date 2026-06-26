"""Held-out synthetic GT evaluation for EP07 training.

The synthetic pool has a clean ground-truth temperature + structure mask, so —
unlike ``real_eval`` (no GT → only GT-free proxies) — here we can measure the
fidelity we actually care about on v4 data:

- ``psnr``               : recoverable-band accuracy vs GT (fixed data-range, dB).
- ``region_rmse``        : temperature error *inside* the chip body (°C). The
                           isothermal-level fidelity the old edge metrics ignored.
- ``boundary_f1``        : defect/edge preservation — a filled-in hole drops
                           recall, a hallucinated edge/beading drops precision.
- ``out_of_band_ratio``  : hallucination energy above the pitch cutoff.

The eval scenes are a fixed tail slice of the training pool (``holdout_role=
"eval"``), excluded from the training sampler, so there is no leakage. The model
forward differs by training path (plain UNet vs unrolled solver), so the caller
supplies a ``forward_fn(batch, device) -> pred`` closure.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .dataset import ThermalSRDataset
from .metrics import boundary_f1, out_of_band_ratio, psnr, region_rmse

ForwardFn = Callable[[dict[str, Any], torch.device], torch.Tensor]


@dataclass(frozen=True)
class SynthEvalConfig:
    enabled: bool = True
    every: int = 0              # 0 -> training save_every
    holdout_tail: int = 0       # tail scenes held out for GT eval (0 disables)
    patches_per_scene: int = 2
    max_patches: int = 128
    batch_size: int = 8
    seed: int = 12345
    psnr_data_range: float = 5.0  # fixed °C span -> PSNR is a stable RMSE-in-dB, no flat-patch inf


def build_eval_loader(
    training_config: Any,
    eval_config: SynthEvalConfig,
    *,
    input_mode: str,
    provide_burst: bool,
    solver_m_frames: int,
    solver_no_drizzle: bool,
) -> DataLoader:
    """Held-out (tail-slice) eval loader matching the training input contract."""

    dataset = ThermalSRDataset(
        training_config.training_pool_dir,
        patch_size_hr=training_config.patch_size_hr,
        scale=training_config.scale,
        seed=eval_config.seed,
        patches_per_scene=eval_config.patches_per_scene,
        max_scene_cache=8,
        input_mode=input_mode,
        return_metadata=False,
        boundary_boost=0.0,          # eval needs no loss-weight maps
        provide_burst=provide_burst,
        solver_m_frames=solver_m_frames,
        solver_no_drizzle=solver_no_drizzle,
        holdout_tail=eval_config.holdout_tail,
        holdout_role="eval",
    )
    return DataLoader(dataset, batch_size=eval_config.batch_size, shuffle=False, num_workers=0)


def _aggregate(values: list[float]) -> float:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.mean(finite)) if finite else float("nan")


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    forward_fn: ForwardFn,
    *,
    scale: int,
    device: torch.device,
    max_patches: int,
    psnr_data_range: float,
) -> tuple[dict[str, float], int]:
    was_training = model.training
    model.eval()
    acc: dict[str, list[float]] = {"psnr": [], "region_rmse": [], "boundary_f1": [], "out_of_band_ratio": []}
    seen = 0
    for batch in loader:
        pred = forward_fn(batch, device)
        pred_np = pred.detach().float().cpu().numpy()
        target_np = batch["hr_target"].detach().float().cpu().numpy()
        mask_np = batch["hr_mask"].detach().float().cpu().numpy()
        for b in range(pred_np.shape[0]):
            p, t, mk = pred_np[b, 0], target_np[b, 0], mask_np[b, 0]
            acc["psnr"].append(psnr(p, t, data_range=psnr_data_range))
            acc["region_rmse"].append(region_rmse(p, t, mk))
            acc["boundary_f1"].append(boundary_f1(p, mk)["f1"])
            acc["out_of_band_ratio"].append(out_of_band_ratio(p, scale=scale))
            seen += 1
            if seen >= max_patches:
                break
        if seen >= max_patches:
            break
    if was_training:
        model.train()
    return {key: _aggregate(vals) for key, vals in acc.items()}, seen


def maybe_log_synth_eval(
    writer: SummaryWriter | None,
    *,
    model: torch.nn.Module,
    loader: DataLoader | None,
    forward_fn: ForwardFn,
    eval_config: SynthEvalConfig,
    training_config: Any,
    step: int,
    device: torch.device,
) -> dict[str, float] | None:
    """Run the held-out synthetic GT eval on cadence and log eval_synth/* scalars."""

    if writer is None or loader is None or not eval_config.enabled or eval_config.holdout_tail <= 0:
        return None
    every = eval_config.every if eval_config.every > 0 else int(training_config.save_every)
    if every <= 0 or step % every != 0:
        return None
    metrics, seen = evaluate(
        model, loader, forward_fn,
        scale=int(training_config.scale), device=device,
        max_patches=eval_config.max_patches, psnr_data_range=eval_config.psnr_data_range,
    )
    for key, val in metrics.items():
        writer.add_scalar(f"eval_synth/{key}", val, step)
    writer.add_scalar("eval_synth/n_patches", float(seen), step)
    writer.flush()
    return metrics
