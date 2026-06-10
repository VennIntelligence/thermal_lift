"""Training CLI for EP12 drizzle-informed 4x SR."""

from __future__ import annotations

import ctypes
import json
import random
import time
from dataclasses import asdict
from itertools import count
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


def _trim_memory() -> None:
    """Ask glibc to return free pages to the OS.

    Persistent DataLoader workers (and the main process) accumulate
    freed-but-not-returned heap pages.  Calling ``malloc_trim(0)``
    periodically keeps RSS from growing without bound.
    """
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass  # non-glibc or non-Linux; skip silently

from .config import TrainingConfig, config_from_args
from .dataset import SceneInterleavedSampler, ThermalSR4xDataset
from .losses import ThermalSR4xLoss
from .model import ThermalSR4xUNet
from .real_eval import RealEvalConfig, maybe_log_real_eval


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
        return torch.device(f"cuda:{index}")
    return device


def _worker_init_fn(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % 2**32
    random.seed(worker_seed + worker_id)
    np.random.seed(worker_seed + worker_id)


def _split_model_output(output: torch.Tensor | tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor | None]:
    if isinstance(output, tuple):
        if len(output) != 2:
            raise ValueError("model output tuple must be (prediction, log_var)")
        return output[0], output[1]
    return output, None


def _unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def _to_device_tensor(tensor: torch.Tensor, *, device: torch.device, channels_last: bool) -> torch.Tensor:
    out = tensor.to(device=device, dtype=torch.float32, non_blocking=True)
    if channels_last and out.ndim == 4:
        out = out.contiguous(memory_format=torch.channels_last)
    return out


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


def _log_pred_vs_target(writer: SummaryWriter, pred: torch.Tensor, target: torch.Tensor, step: int) -> None:
    with torch.no_grad():
        p = pred[0, 0].detach().float().cpu()
        t = target[0, 0].detach().float().cpu()
        err = (p - t).abs()

        def _norm(x: torch.Tensor) -> torch.Tensor:
            lo, hi = x.min(), x.max()
            if hi - lo < 1e-8:
                return torch.zeros_like(x)
            return (x - lo) / (hi - lo)

        writer.add_image("visual/prediction", _norm(p).unsqueeze(0), step)
        writer.add_image("visual/target", _norm(t).unsqueeze(0), step)
        writer.add_image("visual/abs_error", _norm(err).unsqueeze(0), step)


def train(config: TrainingConfig) -> Path:
    config.validate()
    _set_seed(config.seed)
    device = _resolve_device(config.device)
    use_amp = config.amp and device.type == "cuda"
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "config.json").write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    writer: SummaryWriter | None = None
    tb_dir = Path(config.tb_log_dir)
    tb_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(tb_dir))
    print(f"TensorBoard logs -> {tb_dir}")

    dataset = ThermalSR4xDataset(
        config.training_pool_dir,
        patch_size=config.patch_size,
        scale=config.scale,
        drizzle_scale=config.drizzle_scale,
        seed=config.seed,
        patches_per_scene=config.patches_per_scene,
        max_scene_cache=config.max_scene_cache,
        include_multiscale=config.include_multiscale,
        burst_augment=config.burst_augment,
        burst_keep_range=(config.burst_keep_min, config.burst_keep_max),
        min_burst_frames=config.min_burst_frames,
        shift_noise_std_px=config.shift_noise_std_px,
        drizzle_kernel=config.drizzle_kernel,
        defer_1x_upsample=config.defer_1x_upsample,
        return_metadata=False,
    )
    sampler = SceneInterleavedSampler(
        n_scenes=len(dataset.scene_paths),
        patches_per_scene=dataset.patches_per_scene,
        scenes_per_bucket=config.scenes_per_bucket,
        patches_per_fetch=config.patches_per_fetch,
        seed=config.seed,
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

    model_scale = config.scale // config.drizzle_scale
    model = ThermalSR4xUNet(
        in_channels=config.in_channels,
        out_channels=config.out_channels,
        base_channels=config.base_channels,
        scale=model_scale,
        depth=config.unet_depth,
        dilated_bottleneck=config.dilated_bottleneck,
        predict_log_variance=config.predict_log_variance,
        min_log_variance=config.min_log_variance,
        max_log_variance=config.max_log_variance,
    ).to(device)
    if config.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    if config.compile_model and hasattr(torch, "compile"):
        model = torch.compile(model)
        print("torch.compile enabled")

    criterion = ThermalSR4xLoss(
        sigma_lf=config.sigma_lf,
        lf_weight=config.lf_loss_weight,
        hf_weight=config.hf_loss_weight,
        edge_weight=config.edge_loss_weight,
        forward_weight=config.forward_loss_weight,
        nll_weight=config.nll_loss_weight,
        coverage_gain=config.coverage_loss_gain,
        edge_coarse_weight=config.edge_coarse_weight,
        hf_detail_weight=config.hf_detail_weight,
        hf_detail_gain=config.hf_detail_gain,
        scale=config.scale,
        psf_sigma_lr_px=config.psf_sigma_lr_px,
        min_log_variance=config.min_log_variance,
        max_log_variance=config.max_log_variance,
    )
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
        print(f"Resumed from {ckpt_path} at step {start_step}")

    n_params = sum(p.numel() for p in _unwrap_model(model).parameters())
    print(f"Model parameters: {n_params:,}  (model_scale={model_scale}, drizzle_scale={config.drizzle_scale})")

    real_eval_cfg = RealEvalConfig(
        enabled=config.real_eval_enabled,
        every=config.real_eval_every,
        frame_limit=config.real_eval_frame_limit,
        alignment_method=config.real_eval_alignment_method,
        baseline_hr=config.real_eval_baseline_hr,
        center_fraction=config.real_eval_center_fraction,
        zoom=config.real_eval_zoom,
        overlap=config.real_eval_overlap,
        output_dir=str(output_dir),
    )
    if real_eval_cfg.enabled:
        print(
            f"Real-data eval: "
            f"every={real_eval_cfg.every or config.save_every} steps, "
            f"frames={real_eval_cfg.frame_limit}, "
            f"SR={config.scale}x (drizzle={config.drizzle_scale}x), display zoom={real_eval_cfg.zoom:g}x center ROI"
        )

    model.train()
    progress = tqdm(total=config.total_steps, initial=start_step, desc="Training", dynamic_ncols=True)
    step = start_step
    ema_loss: float | None = None
    ema_alpha = 0.02
    t_step_start = time.monotonic()
    try:
        for epoch in count():
            dataset.set_epoch(epoch)
            sampler.set_epoch(epoch)
            for batch in loader:
                step += 1
                target = _to_device_tensor(batch["hr_target"], device=device, channels_last=config.channels_last)
                edge = _to_device_tensor(batch["hr_edge"], device=device, channels_last=config.channels_last)
                drizzle = _to_device_tensor(batch["drizzle_mean"], device=device, channels_last=config.channels_last)
                coverage = _to_device_tensor(batch["coverage"], device=device, channels_last=config.channels_last)
                if config.defer_1x_upsample:
                    obs_hr = _to_device_tensor(batch["obs_features_hr"], device=device, channels_last=config.channels_last)
                    obs_lr = _to_device_tensor(batch["obs_features_1x_lr"], device=device, channels_last=config.channels_last)
                    obs_lr_up = F.interpolate(obs_lr, size=obs_hr.shape[-2:], mode="bilinear", align_corners=False)
                    obs = torch.cat([obs_hr, obs_lr_up], dim=1)
                    if config.channels_last:
                        obs = obs.contiguous(memory_format=torch.channels_last)
                else:
                    obs = _to_device_tensor(batch["obs_features"], device=device, channels_last=config.channels_last)

                with autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                    output = model(obs)
                    pred, log_var = _split_model_output(output)
                    if pred.shape != target.shape:
                        raise RuntimeError(f"model output shape {pred.shape} does not match target {target.shape}")
                    # Upsample drizzle and coverage to match pred resolution
                    if model_scale > 1:
                        drizzle = F.interpolate(drizzle, scale_factor=model_scale, mode="bilinear", align_corners=False)
                        coverage = F.interpolate(coverage, scale_factor=model_scale, mode="nearest")
                    losses = criterion(
                        pred,
                        target,
                        edge_mask=edge,
                        coverage_4x=coverage,
                        drizzle_mean_4x=drizzle,
                        log_var=log_var,
                    )
                    total = losses["total"]
                if not torch.isfinite(total):
                    raise FloatingPointError(f"non-finite loss at step {step}: {losses}")

                scaler.scale(total).backward()
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(model.parameters(), 1.0).item()
                scale_before = scaler.get_scale()
                scaler.step(optimizer)
                scaler.update()
                optimizer_was_skipped = use_amp and scaler.get_scale() < scale_before
                optimizer.zero_grad(set_to_none=True)
                if not optimizer_was_skipped:
                    scheduler.step()

                t_step_end = time.monotonic()
                step_time_ms = (t_step_end - t_step_start) * 1000.0
                t_step_start = t_step_end

                # ---- Free computation graph BEFORE logging ----
                loss_val = float(total.detach().item())
                loss_items = {k: float(v.detach().item()) for k, v in losses.items()}
                ema_loss = loss_val if ema_loss is None else ema_alpha * loss_val + (1.0 - ema_alpha) * ema_loss
                # Detach pred for TensorBoard logging, then release graph
                pred_detached = pred.detach()
                del total, losses, output, pred, log_var
                # obs/target/edge/drizzle/coverage are leaf tensors (no grad), safe to keep

                progress.update(1)
                if step == 1 or step % config.log_every == 0:
                    lr_value = scheduler.get_last_lr()[0]
                    parts = [f"step={step}", f"total={loss_items['total']:.6f}"]
                    parts.extend(f"{name}={value:.6f}" for name, value in loss_items.items() if name != "total")
                    parts.append(f"lr={lr_value:.6g}")
                    print(" ".join(parts))
                    if writer is not None:
                        for name, value in loss_items.items():
                            writer.add_scalar(f"loss/{name}", value, step)
                        writer.add_scalar("train/learning_rate", lr_value, step)
                        writer.add_scalar("train/grad_norm", grad_norm, step)
                        writer.add_scalar("train/step_time_ms", step_time_ms, step)
                        writer.add_scalar("loss/total_ema50", ema_loss, step)
                        if use_amp:
                            writer.add_scalar("train/amp_scaler_scale", scaler.get_scale(), step)

                if writer is not None and step % config.tb_image_every == 0:
                    _log_pred_vs_target(writer, pred_detached, target, step)
                del pred_detached  # release last reference

                # Periodic memory maintenance: flush TB buffers and trim heap
                if step % config.save_every == 0:
                    if writer is not None:
                        writer.flush()
                    _trim_memory()

                if step % config.save_every == 0:
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
                    if eval_metrics is not None:
                        print(
                            "eval_real "
                            + " ".join(f"{key}={value:.6g}" for key, value in eval_metrics.items())
                        )
                if step >= config.total_steps:
                    break
            if step >= config.total_steps:
                break
    finally:
        progress.close()
        if writer is not None:
            writer.close()

    final_path = output_dir / "model_final.pt"
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
    return final_path


def main(argv: list[str] | None = None) -> None:
    train(config_from_args(argv))


if __name__ == "__main__":
    main()
