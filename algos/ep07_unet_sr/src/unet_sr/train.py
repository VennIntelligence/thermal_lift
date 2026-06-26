"""Training CLI for EP07v2 compact-scene UNet SR."""

from __future__ import annotations

import json
import os
import random
import time
from dataclasses import asdict
from itertools import count
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import TrainingConfig, config_from_args
from .dataset import HYBRID_DRIZZLE_MEAN_CHANNEL, ThermalSRDataset, SceneInterleavedSampler
from .losses import ContourSRLoss, ThermalSRLoss
from .model import ThermalSRUNet
from .real_eval import RealEvalConfig, maybe_log_real_eval
from .synth_eval import SynthEvalConfig, build_eval_loader, maybe_log_synth_eval


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(value: str) -> torch.device:
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but unavailable; falling back to CPU.")
        return torch.device("cpu")
    if device.type == "cuda":
        index = 0 if device.index is None else int(device.index)
        torch.cuda.set_device(index)
        device = torch.device(f"cuda:{index}")
    return device


def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def _setup_ddp() -> tuple[int, int]:
    """Initialize DDP from torchrun environment variables."""
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl")
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    torch.cuda.set_device(local_rank)
    return local_rank, world_size


def _is_ddp() -> bool:
    return dist.is_available() and dist.is_initialized()


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    unwrapped = getattr(model, "module", model)
    return getattr(unwrapped, "_orig_mod", unwrapped)


def _to_device_tensor(tensor: torch.Tensor, *, device: torch.device, channels_last: bool) -> torch.Tensor:
    out = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
    if channels_last and out.ndim == 4:
        out = out.contiguous(memory_format=torch.channels_last)
    return out


def _delta_l1_penalty(delta: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    delta_f = delta.float()
    return delta_f.abs().mean(), delta_f.detach().mean(), delta_f.detach().std(unbiased=False)


def _save_checkpoint(
    path: Path,
    *,
    step: int,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainingConfig,
    scaler: GradScaler | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "model_state_dict": _unwrap_model(model).state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler is not None else {},
            "config": asdict(config),
        },
        path,
    )


def _log_pred_vs_target(
    writer: SummaryWriter,
    pred: torch.Tensor,
    target: torch.Tensor,
    step: int,
) -> None:
    """Log first sample in batch: pred, target, |error| as side-by-side images."""
    with torch.no_grad():
        p = pred[0, 0].detach().float().cpu()
        t = target[0, 0].detach().float().cpu()
        err = (p - t).abs()

        # Normalize each to [0, 1] for visualization
        def _norm(x: torch.Tensor) -> torch.Tensor:
            lo, hi = x.min(), x.max()
            if hi - lo < 1e-8:
                return torch.zeros_like(x)
            return (x - lo) / (hi - lo)

        p_img = _norm(p).unsqueeze(0)   # (1, H, W)
        t_img = _norm(t).unsqueeze(0)
        e_img = _norm(err).unsqueeze(0)

        writer.add_image("visual/prediction", p_img, step)
        writer.add_image("visual/target", t_img, step)
        writer.add_image("visual/abs_error", e_img, step)


def train(config: TrainingConfig) -> Path:
    config.validate()
    _set_seed(config.seed)
    device = _resolve_device(config.device)
    ddp = "LOCAL_RANK" in os.environ and device.type == "cuda"
    if ddp:
        local_rank, world_size = _setup_ddp()
        device = torch.device(f"cuda:{local_rank}")
    else:
        local_rank, world_size = 0, 1
    is_main = local_rank == 0
    use_amp = config.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    output_dir = Path(config.output_dir)
    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    real_eval_cfg = RealEvalConfig(
        enabled=config.real_eval_enabled,
        every=config.real_eval_every,
        frame_limit=config.real_eval_frame_limit,
        alignment_method=config.real_eval_alignment_method,
        baseline_hr=config.real_eval_baseline_hr,
        center_fraction=config.real_eval_center_fraction,
        zoom=config.real_eval_zoom,
        overlap=config.real_eval_overlap,
        highpass_sigma=config.highpass_sigma,
        output_dir=str(output_dir),
    )
    if is_main and real_eval_cfg.enabled:
        print(
            "Real-data eval enabled (EP11-style, not TCForge): "
            f"every={real_eval_cfg.every or config.save_every} steps, "
            f"frames={real_eval_cfg.frame_limit}, "
            f"SR={config.scale}x, display zoom={real_eval_cfg.zoom:g}x center ROI"
        )

    # --- TensorBoard writer (rank-0 only) ---
    writer: SummaryWriter | None = None
    if is_main:
        tb_dir = Path(config.tb_log_dir)
        tb_dir.mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(log_dir=str(tb_dir))
        print(f"TensorBoard logs → {tb_dir}")

    dataset = ThermalSRDataset(
        config.training_pool_dir,
        patch_size_hr=config.patch_size_hr,
        scale=config.scale,
        seed=config.seed,
        patches_per_scene=config.patches_per_scene,
        max_scene_cache=config.max_scene_cache,
        residual=config.residual,
        input_mode=config.input_mode,
        return_metadata=False,
        boundary_boost=config.boundary_boost,
        boundary_tau_px=config.boundary_tau_px,
        holdout_tail=config.synth_eval_holdout,  # exclude the GT-eval tail from training (no leakage)
        holdout_role="train",
    )
    if is_main and config.boundary_boost > 0.0:
        print(
            "Boundary-emphasis loss weights precomputed in DataLoader workers "
            f"(boundary_boost={config.boundary_boost:g}, tau_px={config.boundary_tau_px:g})"
        )
    sampler = SceneInterleavedSampler(
        n_scenes=len(dataset.scene_paths),
        patches_per_scene=dataset.patches_per_scene,
        scenes_per_bucket=config.scenes_per_bucket,
        patches_per_fetch=config.patches_per_fetch,
        seed=config.seed,
        rank=local_rank,
        world_size=world_size,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=_worker_init_fn if config.num_workers > 0 else None,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )

    model_scale = 1 if (config.residual or config.input_mode == "hybrid_drizzle2x") else config.scale
    model = ThermalSRUNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_channels=config.base_channels,
        scale=model_scale,
        hr_upsampler=config.hr_upsampler,
        hr_res_blocks=config.hr_res_blocks,
    ).to(device)
    if config.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
        if is_main:
            print("torch.compile enabled")
    if ddp:
        model = DDP(model, device_ids=[local_rank])
    if config.loss_type == "contour_sr":
        criterion = ContourSRLoss(
            highpass_weight=config.highpass_loss_weight,
            highpass_sigma=config.highpass_sigma,
            edge_weight=config.edge_loss_weight,
            ssim_weight=config.ssim_loss_weight,
            mse_weight=config.mse_loss_weight,
            structure_boost=config.structure_boost,
            edge_coarse_weight=config.edge_coarse_weight,
            grad_vector_weight=config.grad_vector_weight,
            flatness_weight=config.flatness_weight,
            flatness_tau=config.flatness_tau,
            laplacian_weight=config.laplacian_weight,
            forward_model_weight=config.forward_model_weight,
            forward_model_psf_sigma=config.forward_model_psf_sigma,
            forward_model_scale=config.scale,
            forward_model_band=config.forward_model_band,
            forward_model_band_sigma=config.forward_model_band_sigma,
        )
    else:
        criterion = ThermalSRLoss(edge_weight=config.edge_loss_weight, ssim_weight=config.ssim_loss_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.total_steps))
    if config.lr_warmup_steps > 0:
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=config.lr_warmup_steps,
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[config.lr_warmup_steps],
        )
    else:
        scheduler = cosine_scheduler
    scaler = GradScaler(enabled=use_amp)

    # --- Resume from checkpoint ---
    start_step = 0
    if config.resume_from:
        ckpt_path = Path(config.resume_from)
        if not ckpt_path.exists():
            raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        _unwrap_model(model).load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt and use_amp:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_step = int(ckpt.get("step", 0))
        if is_main:
            print(f"Resumed from {ckpt_path} at step {start_step}")

    # Log model parameter count and mode
    if is_main:
        n_params = sum(p.numel() for p in _unwrap_model(model).parameters())
        print(f"Model parameters: {n_params:,}")
        print(f"Loss function: {config.loss_type}")
        print(f"HR upsampler: {config.hr_upsampler} (hr_res_blocks={config.hr_res_blocks})")
        if config.residual:
            print(f"Residual mode: {config.in_channels}ch@{config.scale}x input → model(scale=1) → residual + classical_sr")
        if config.input_mode == "hybrid_drizzle2x":
            print(f"Hybrid drizzle 2x mode: {config.in_channels}ch@2x input (5ch fused↑2x + {config.phase_bin_channels}ch phase-bin drizzle@2x) → model(scale=1) → direct predict")
        if config.residual_mode == "drizzle2x":
            print(
                "Residual-over-observation mode: "
                f"pred = hybrid ch{HYBRID_DRIZZLE_MEAN_CHANNEL} phase-bin anchor + model delta; "
                f"L1(delta) weight={config.residual_penalty_weight:g}"
            )

    # --- Held-out synthetic GT eval (PSNR / region-RMSE / defect boundary-F1 / out-of-band) ---
    synth_eval_cfg = SynthEvalConfig(
        enabled=config.synth_eval_enabled, every=config.synth_eval_every,
        holdout_tail=config.synth_eval_holdout, patches_per_scene=config.synth_eval_patches_per_scene,
        max_patches=config.synth_eval_max_patches, batch_size=config.batch_size,
    )
    synth_loader = None
    if is_main and synth_eval_cfg.enabled and synth_eval_cfg.holdout_tail > 0:
        synth_loader = build_eval_loader(
            config, synth_eval_cfg, input_mode=config.input_mode, provide_burst=False,
            solver_m_frames=config.solver_m_frames, solver_no_drizzle=config.solver_no_drizzle,
        )
        print(f"Synthetic held-out eval: {synth_eval_cfg.holdout_tail} tail scenes, "
              f"<= {synth_eval_cfg.max_patches} patches every "
              f"{synth_eval_cfg.every or config.save_every} steps")

    def _unet_forward(batch: dict, dev: torch.device) -> torch.Tensor:
        obs_e = _to_device_tensor(batch["obs_features"], device=dev, channels_last=config.channels_last)
        pred_e = _unwrap_model(model)(obs_e)
        if config.residual:
            pred_e = obs_e[:, -1:, :, :] + pred_e
        elif config.residual_mode == "drizzle2x":
            pred_e = obs_e[:, HYBRID_DRIZZLE_MEAN_CHANNEL : HYBRID_DRIZZLE_MEAN_CHANNEL + 1, :, :] + pred_e
        return pred_e

    model.train()
    progress = tqdm(
        total=config.total_steps,
        initial=start_step,
        desc="Training",
        dynamic_ncols=True,
        disable=not is_main,
    )
    step = start_step
    ema_loss: float | None = None
    ema_alpha = 0.02  # ~50-step smoothing window
    t_step_start = time.monotonic()
    try:
        for epoch in count():
            dataset.set_epoch(epoch)
            sampler.set_epoch(epoch)

            for batch in loader:
                step += 1
                obs = _to_device_tensor(batch["obs_features"], device=device, channels_last=config.channels_last)
                target = _to_device_tensor(batch["hr_target"], device=device, channels_last=config.channels_last)
                boundary_weight = None
                if isinstance(criterion, ContourSRLoss) and "boundary_weight" in batch:
                    boundary_weight = _to_device_tensor(
                        batch["boundary_weight"], device=device, channels_last=config.channels_last,
                    )

                delta_stats: tuple[torch.Tensor, torch.Tensor] | None = None
                with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    pred = model(obs)
                    if config.residual:
                        # Residual skip: add classical_sr (last input channel) to model output
                        classical_sr_batch = obs[:, -1:, :, :]
                        pred = classical_sr_batch + pred
                    elif config.residual_mode == "drizzle2x":
                        delta = pred
                        pred = obs[:, HYBRID_DRIZZLE_MEAN_CHANNEL:HYBRID_DRIZZLE_MEAN_CHANNEL + 1, :, :] + delta
                        residual_penalty, delta_mean, delta_std = _delta_l1_penalty(delta)
                        delta_stats = (delta_mean, delta_std)
                    if pred.shape != target.shape:
                        raise RuntimeError(f"model output shape {pred.shape} does not match target {target.shape}")
                    if isinstance(criterion, ContourSRLoss):
                        lr_obs = None
                        lr_mean = None
                        if config.forward_model_weight > 0:
                            if "lr_obs" in batch:
                                lr_obs = _to_device_tensor(
                                    batch["lr_obs"], device=device, channels_last=config.channels_last,
                                )
                            else:
                                lr_mean = obs[:, 0:1, :, :]  # ch0 = aligned_mean on the native LR input grid
                        losses = criterion(
                            pred,
                            target,
                            lr_observation=lr_mean,
                            lr_obs=lr_obs,
                            boundary_weight=boundary_weight,
                        )
                    else:
                        edge = _to_device_tensor(batch["hr_edge"], device=device, channels_last=config.channels_last)
                        losses = criterion(pred, target, edge_mask=edge)
                    if config.residual_mode == "drizzle2x":
                        losses = dict(losses)
                        losses["residual_penalty"] = residual_penalty
                        losses["total"] = losses["total"] + config.residual_penalty_weight * residual_penalty
                    total = losses["total"]
                if not torch.isfinite(total):
                    raise FloatingPointError(f"non-finite loss at step {step}: {losses}")

                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(model.parameters(), 1.0).item()
                scaler_scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_was_skipped = use_amp and scaler.get_scale() < scaler_scale_before
                optimizer.zero_grad(set_to_none=True)
                if not optimizer_was_skipped:
                    scheduler.step()

                # --- Timing ---
                t_step_end = time.monotonic()
                step_time_ms = (t_step_end - t_step_start) * 1000.0
                t_step_start = t_step_end

                # --- EMA loss (updated every step, logged at log_every) ---
                loss_val = losses["total"].item()
                if ema_loss is None:
                    ema_loss = loss_val
                else:
                    ema_loss = ema_alpha * loss_val + (1.0 - ema_alpha) * ema_loss

                progress.update(1)
                if is_main and (step == 1 or step % config.log_every == 0):
                    lr_value = scheduler.get_last_lr()[0]
                    loss_parts = " ".join(f"{k}={v.item():.6f}" for k, v in losses.items())
                    print(f"step={step} {loss_parts} lr={lr_value:.6g}")

                    # --- TensorBoard scalars ---
                    if writer is not None:
                        for loss_key, loss_val in losses.items():
                            writer.add_scalar(f"loss/{loss_key}", loss_val.item(), step)
                        writer.add_scalar("train/learning_rate", lr_value, step)
                        writer.add_scalar("train/grad_norm", grad_norm, step)
                        writer.add_scalar("train/step_time_ms", step_time_ms, step)
                        # Smoothed loss for trend monitoring
                        if ema_loss is not None:
                            writer.add_scalar("loss/total_ema50", ema_loss, step)
                        if use_amp:
                            writer.add_scalar("train/amp_scaler_scale", scaler.get_scale(), step)
                        if delta_stats is not None:
                            delta_mean, delta_std = delta_stats
                            writer.add_scalar("residual/delta_mean", delta_mean.item(), step)
                            writer.add_scalar("residual/delta_std", delta_std.item(), step)

                # --- TensorBoard images ---
                if (
                    is_main
                    and writer is not None
                    and config.tb_image_every > 0
                    and step % config.tb_image_every == 0
                ):
                    _log_pred_vs_target(writer, pred, target, step)

                if is_main and step % config.save_every == 0:
                    _save_checkpoint(
                        output_dir / f"checkpoint_step_{step:06d}.pt",
                        step=step,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        config=config,
                        scaler=scaler,
                    )
                    eval_metrics = maybe_log_real_eval(
                        writer,
                        model=_unwrap_model(model),
                        config=real_eval_cfg,
                        training_config=config,
                        step=step,
                        device=device,
                    )
                    if eval_metrics is not None and is_main:
                        print(
                            "eval_real "
                            + " ".join(f"{key}={value:.6g}" for key, value in eval_metrics.items())
                        )
                    synth_metrics = maybe_log_synth_eval(
                        writer,
                        model=_unwrap_model(model),
                        loader=synth_loader,
                        forward_fn=_unet_forward,
                        eval_config=synth_eval_cfg,
                        training_config=config,
                        step=step,
                        device=device,
                    )
                    if synth_metrics is not None and is_main:
                        print(
                            "eval_synth "
                            + " ".join(f"{key}={value:.4g}" for key, value in synth_metrics.items())
                        )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                if step >= config.total_steps:
                    break

            if step >= config.total_steps:
                break
    finally:
        progress.close()

    final_path = output_dir / "model_final.pt"
    if is_main:
        _save_checkpoint(
            final_path,
            step=config.total_steps,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
            scaler=scaler,
        )
        print(f"Saved final checkpoint: {final_path}")
        maybe_log_real_eval(
            writer,
            model=_unwrap_model(model),
            config=real_eval_cfg,
            training_config=config,
            step=config.total_steps,
            device=device,
        )
        if device.type == "cuda":
            torch.cuda.empty_cache()
        if writer is not None:
            writer.close()
    elif writer is not None:
        writer.close()
    if ddp:
        dist.barrier(device_ids=[local_rank])
        dist.destroy_process_group()
    return final_path


def main(argv: list[str] | None = None) -> None:
    train(config_from_args(argv))


if __name__ == "__main__":
    main()
