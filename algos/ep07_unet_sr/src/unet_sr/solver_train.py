"""Training entry point for the K-step physics-constrained unrolled SR solver (ACL-024).

Separate from train.py (the plain-UNet path) to keep the solver path isolated and reviewable.
Reuses the dataset (with provide_burst=True), the ContourSRLoss structure/band terms, and the
Gate-A-certified forward operator. Runs in fp32 (double-backward through the autograd DC step is
more stable without AMP; revisit once it's working).

Loss = band-aware structure supervision on the GT target (ContourSRLoss) + a terminal hard
data-consistency term ||A x_K - y||^2 in the SR band, with an ~8 LR-px patch-edge rim masked out
(A(patch) zero-pads outside the patch — validated, only the rim differs). The structure prior can
be late-annealed (0->1 over N steps) so DC dominates early and the prior fills the null space
later — directly targeting the fidelity cliff (FM-1).

Usage (see docs/REMOTE_ORDERS.md):
    uv run python -m unet_sr.solver_train --training-pool-dir data/synthetic/pool_2x_v3_5k \
        --input-mode hybrid_drizzle2x --scale 2 --unroll-steps 4 --total-steps 20000 \
        --batch-size 4 --patch-size-hr 256 --output-dir outputs/solver_v1
"""

from __future__ import annotations

import time
import warnings
from itertools import count
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .config import TrainingConfig, config_from_args
from .dataset import HYBRID_DRIZZLE_MEAN_CHANNEL, SceneInterleavedSampler, ThermalSRDataset
from .forward_torch import ScenePSF, _highpass, forward_burst
from .losses import ContourSRLoss
from .real_eval import RealEvalConfig, maybe_log_solver_real_eval
from .synth_eval import SynthEvalConfig, build_eval_loader, maybe_log_synth_eval
from .train import _log_pred_vs_target, _to_device_tensor, _worker_init_fn
from .unroll import UnrolledSolver


def _compile_solver_prox(solver: UnrolledSolver) -> None:
    """Compile only the learned prox UNet(s), not the physics/autograd DC path."""

    if not hasattr(torch, "compile"):
        print("torch.compile requested but this PyTorch build has no torch.compile; continuing uncompiled")
        return
    if solver.share_weights:
        solver.prox = torch.compile(solver.prox)  # type: ignore[assignment]
    else:
        solver.prox = torch.nn.ModuleList(torch.compile(module) for module in solver.prox)  # type: ignore[arg-type,assignment]
    print("torch.compile enabled for solver prox network(s); physics DC path remains eager")


def _checkpoint_state_dict(solver: UnrolledSolver) -> dict[str, torch.Tensor]:
    """Return a checkpoint-compatible state dict even when child modules are compiled."""

    state = solver.state_dict()
    return {key.replace("._orig_mod.", "."): value for key, value in state.items()}


def _save_solver_checkpoint(
    path: Path,
    *,
    step: int,
    solver: UnrolledSolver,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    config: TrainingConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "step": int(step),
            "model_state_dict": _checkpoint_state_dict(solver),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "config": vars(config),
        },
        path,
    )


def _load_resume_checkpoint(path_text: str, device: torch.device) -> tuple[Path, dict[str, Any], int]:
    ckpt_path = Path(path_text)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"resume checkpoint not found: {ckpt_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if not isinstance(ckpt, dict) or "model_state_dict" not in ckpt:
        raise ValueError(f"unsupported solver checkpoint format: {ckpt_path}")
    return ckpt_path, ckpt, int(ckpt.get("step", 0))


def _fast_forward_scheduler(scheduler: torch.optim.lr_scheduler.LRScheduler, steps: int) -> None:
    """Approximate scheduler state for legacy solver checkpoints without scheduler_state_dict."""

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="Detected call of `lr_scheduler.step\\(\\)` before")
        for _ in range(max(0, int(steps))):
            scheduler.step()


def build_scene_psf(batch: dict, device: torch.device) -> ScenePSF:
    """Per-sample PSF from the batched scene metadata (gaussian/elliptical/airy)."""
    return ScenePSF(
        sigma_lr_px=batch["psf_sigma_lr_px"].to(device),
        shape=list(batch["psf_shape"]),
        sigma_y_lr_px=[float(v) for v in batch["psf_sigma_y_lr_px"]],
        angle_deg=batch["psf_angle_deg"].to(device),
    )


def edge_mask(h: int, w: int, rim: int, device: torch.device) -> torch.Tensor:
    """(1,1,h,w) mask with a `rim`-px border of zeros (the DC term ignores the patch edge)."""
    m = torch.zeros(1, 1, h, w, device=device)
    if rim > 0:
        m[..., rim:-rim, rim:-rim] = 1.0
    else:
        m[...] = 1.0
    return m


def terminal_dc_loss(pred, burst, shifts, psf, scale, band_sigma, mask, huber_delta):
    """Hard data-consistency on the final estimate, band-limited + edge-masked."""
    r = forward_burst(pred, shifts, psf, scale) - burst
    if band_sigma > 0:
        r = _highpass(r, band_sigma)
    r = r * mask
    if huber_delta and huber_delta > 0:
        a = r.abs()
        return torch.where(a <= huber_delta, 0.5 * r * r, huber_delta * (a - 0.5 * huber_delta)).mean()
    return (r * r).mean()


def build_solver(config: TrainingConfig, device: torch.device, cond_channels: int) -> UnrolledSolver:
    return UnrolledSolver(
        n_steps=config.unroll_steps,
        cond_channels=cond_channels,  # 8 (hybrid) or 5 (no-drizzle lean path)
        base_channels=config.base_channels,
        scale=config.scale,
        share_weights=config.solver_share_weights,
        band_highpass_sigma_lr_px=config.solver_band_sigma,
        huber_delta=config.solver_huber_delta,
        eta_init=config.solver_eta_init,
        learn_eta=config.solver_learn_eta,
        prox_use_se=config.solver_prox_use_se,
        prox_norm=config.solver_prox_norm,
        prox_highpass_residual=config.solver_prox_highpass_residual,
        prox_highpass_sigma_hr=config.solver_prox_highpass_sigma_hr,
        fusion=config.solver_fusion,
        fusion_channels=config.solver_fusion_channels,
        fusion_frame_chunk=config.solver_fusion_frame_chunk,
    ).to(device)


def build_criterion(config: TrainingConfig) -> ContourSRLoss:
    return ContourSRLoss(
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
        forward_model_weight=0.0,  # hard DC replaces the (falsified) soft forward-model loss
    )


def train(config: TrainingConfig) -> Path:
    if config.unroll_steps <= 0:
        raise ValueError("solver_train requires --unroll-steps > 0 (use train.py for the plain UNet)")
    device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
    torch.manual_seed(config.seed)
    if device.type == "cuda":
        # The solver runs fp32 (no AMP) for double-backward stability — which makes cuDNN
        # autotuning + TF32 essential: otherwise the K prox-UNet convs (the dominant cost) run
        # as unaccelerated full-fp32 on the 5090. TF32 keeps fp32 storage (double-backward safe,
        # unlike fp16) but uses the tensor cores; shapes are fixed here so benchmark pays off.
        # (Gate A certification uses a separate fp64 path and is unaffected.)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tb_dir = Path(config.tb_log_dir or (output_dir / "tb_logs"))
    tb_dir.mkdir(parents=True, exist_ok=True)
    resume_ckpt: dict[str, Any] | None = None
    resume_path: Path | None = None
    start_step = 0
    if config.resume_from:
        resume_path, resume_ckpt, start_step = _load_resume_checkpoint(config.resume_from, device)
    real_eval_cfg = RealEvalConfig(
        enabled=config.real_eval_enabled,
        every=config.real_eval_every,
        frame_limit=config.real_eval_frame_limit,
        alignment_method=config.real_eval_alignment_method,
        baseline_hr=config.real_eval_baseline_hr,
        center_fraction=config.real_eval_center_fraction,
        zoom=config.real_eval_zoom,
        overlap=config.real_eval_overlap,
        tile_batch=config.real_eval_tile_batch,
        solver_mode=config.real_eval_solver_mode,
        solver_halo_hr=config.real_eval_solver_halo_hr,
        highpass_sigma=config.highpass_sigma,
        output_dir=str(output_dir),
        save_png=True,
    )
    if real_eval_cfg.enabled:
        print(
            "Real-data solver eval: "
            f"every={real_eval_cfg.every or config.save_every} steps, "
            f"frames={real_eval_cfg.frame_limit}, "
            f"DC frames={config.solver_m_frames}, "
            f"SR={config.scale}x, display zoom={real_eval_cfg.zoom:g}x center ROI"
        )

    dataset = ThermalSRDataset(
        config.training_pool_dir,
        patch_size_hr=config.patch_size_hr,
        scale=config.scale,
        seed=config.seed,
        patches_per_scene=config.patches_per_scene,
        max_scene_cache=config.max_scene_cache,
        input_mode="hybrid_drizzle2x",
        return_metadata=False,
        boundary_boost=config.boundary_boost,  # geometry-agnostic edge emphasis (chip/hole/crack/notch)
        boundary_tau_px=config.boundary_tau_px,
        holdout_tail=config.synth_eval_holdout,  # exclude the GT-eval tail from training (no leakage)
        holdout_role="train",
        provide_burst=True,
        solver_m_frames=config.solver_m_frames,
        solver_no_drizzle=config.solver_no_drizzle,
        phasebin_ontf=config.solver_phasebin_ontf,
        phase_bin_channels=config.phase_bin_channels,
        # Stage 1a operator DR — training only; the synth_eval loader below never jitters,
        # so held-out metrics stay comparable across DR and no-DR arms.
        dc_shift_jitter_std_px=config.solver_dc_shift_jitter_std_px,
        dc_psf_sigma_jitter_frac=config.solver_dc_psf_sigma_jitter_frac,
        dc_psf_angle_jitter_deg=config.solver_dc_psf_angle_jitter_deg,
    )
    # cond/warm-start channels: lean path uses 5ch upsampled fused + aligned_mean (ch0);
    # hybrid path uses 9ch obs and warm-starts from the first phase-bin channel (ch5) by default,
    # or the smooth aligned-mean (ch0) when --solver-warmstart aligned_mean (de-waffled, ACL-032).
    cond_channels = 5 if config.solver_no_drizzle else config.in_channels
    mean_ch = 0 if (config.solver_no_drizzle or config.solver_warmstart == "aligned_mean") \
        else HYBRID_DRIZZLE_MEAN_CHANNEL
    sampler = SceneInterleavedSampler(
        n_scenes=len(dataset.scene_paths),
        patches_per_scene=dataset.patches_per_scene,
        scenes_per_bucket=config.scenes_per_bucket,
        patches_per_fetch=config.patches_per_fetch,
        seed=config.seed,
        rank=0,
        world_size=1,
        num_workers=config.num_workers,
        batch_size=config.batch_size,
    )
    loader = DataLoader(
        dataset, batch_size=config.batch_size, sampler=sampler, num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
        worker_init_fn=_worker_init_fn if config.num_workers > 0 else None,
        persistent_workers=config.num_workers > 0,
        prefetch_factor=config.prefetch_factor if config.num_workers > 0 else None,
    )

    solver = build_solver(config, device, cond_channels)
    if resume_ckpt is not None:
        solver.load_state_dict(resume_ckpt["model_state_dict"])
    if config.compile_model:
        _compile_solver_prox(solver)
    criterion = build_criterion(config)
    n_params = sum(p.numel() for p in solver.parameters())
    warmstart_label = "aligned_mean(ch0)" if mean_ch == 0 else f"phasebin(ch{mean_ch})"
    dr_label = "off" if (
        config.solver_dc_shift_jitter_std_px == 0
        and config.solver_dc_psf_sigma_jitter_frac == 0
        and config.solver_dc_psf_angle_jitter_deg == 0
    ) else (f"shift±{config.solver_dc_shift_jitter_std_px:g}px "
            f"psfσ±{config.solver_dc_psf_sigma_jitter_frac:.0%} "
            f"angle±{config.solver_dc_psf_angle_jitter_deg:g}°")
    fusion_label = "off" if config.solver_fusion == "none" else (
        f"{config.solver_fusion}(E={config.solver_fusion_channels}, "
        f"+{3 * config.solver_fusion_channels}ch, chunk={config.solver_fusion_frame_chunk})")
    print(f"UnrolledSolver: K={config.unroll_steps} steps, M={config.solver_m_frames} frames, "
          f"{n_params:,} params, cond={cond_channels}ch, no_drizzle={config.solver_no_drizzle}, "
          f"warmstart x0={warmstart_label}, band_sigma={config.solver_band_sigma}, "
          f"operator_DR={dr_label}, fusion={fusion_label}, device={device}")

    optimizer = torch.optim.AdamW(solver.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.total_steps))
    if config.lr_warmup_steps > 0:
        warmup = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-6, end_factor=1.0,
                                                   total_iters=config.lr_warmup_steps)
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[config.lr_warmup_steps])
    else:
        scheduler = cosine
    if resume_ckpt is not None:
        if "optimizer_state_dict" in resume_ckpt:
            optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        else:
            print(
                "Resume checkpoint has no optimizer_state_dict; continuing as a warm restart "
                "with fresh AdamW moments."
            )
        if "scheduler_state_dict" in resume_ckpt:
            scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        else:
            _fast_forward_scheduler(scheduler, start_step)
            print(f"Resume checkpoint has no scheduler_state_dict; fast-forwarded scheduler to step {start_step}.")
        print(f"Resumed solver from {resume_path} at step {start_step}")
    writer_kwargs: dict[str, Any] = {"log_dir": str(tb_dir)}
    if start_step > 0:
        writer_kwargs["purge_step"] = start_step + 1
    writer = SummaryWriter(**writer_kwargs)
    print(f"TensorBoard logs → {tb_dir}  (tensorboard --logdir {tb_dir})")

    p1 = config.patch_size_hr // config.scale
    mask = edge_mask(p1, p1, config.solver_dc_rim_lr_px, device)

    # Held-out synthetic GT eval (PSNR / region-RMSE / defect boundary-F1 / out-of-band).
    synth_eval_cfg = SynthEvalConfig(
        enabled=config.synth_eval_enabled, every=config.synth_eval_every,
        holdout_tail=config.synth_eval_holdout, patches_per_scene=config.synth_eval_patches_per_scene,
        max_patches=config.synth_eval_max_patches, batch_size=config.batch_size,
    )
    synth_loader = None
    if synth_eval_cfg.enabled and synth_eval_cfg.holdout_tail > 0:
        synth_loader = build_eval_loader(
            config, synth_eval_cfg, input_mode="hybrid_drizzle2x", provide_burst=True,
            solver_m_frames=config.solver_m_frames, solver_no_drizzle=config.solver_no_drizzle,
            phasebin_ontf=config.solver_phasebin_ontf, phase_bin_channels=config.phase_bin_channels,
        )
        print(f"Synthetic held-out eval: {synth_eval_cfg.holdout_tail} tail scenes, "
              f"<= {synth_eval_cfg.max_patches} patches every "
              f"{synth_eval_cfg.every or config.save_every} steps")

    def _solver_forward(batch: dict, dev: torch.device) -> torch.Tensor:
        obs_e = _to_device_tensor(batch["obs_features"], device=dev, channels_last=False)
        x0_e = obs_e[:, mean_ch : mean_ch + 1]
        return solver(x0_e, batch["lr_burst_patch"].to(dev), batch["burst_shifts"].to(dev),
                      build_scene_psf(batch, dev), obs_e, frame_mask=mask)

    solver.train()
    step = start_step
    t0 = time.monotonic()
    # disable=None auto-disables the bar on a non-TTY (e.g. nohup > log) so the periodic
    # progress.write() lines stay clean; on a terminal you get the live bar + ETA + postfix.
    progress = tqdm(total=config.total_steps, initial=start_step, desc="Solver", dynamic_ncols=True, disable=None)
    if step >= config.total_steps:
        progress.close()
        final = output_dir / "solver_final.pt"
        _save_solver_checkpoint(
            final,
            step=step,
            solver=solver,
            optimizer=optimizer,
            scheduler=scheduler,
            config=config,
        )
        print(f"resume step {step} already reached total_steps={config.total_steps}; saved {final}")
        writer.close()
        return final
    for epoch in count():
        dataset.set_epoch(epoch)
        sampler.set_epoch(epoch)
        for batch in loader:
            step += 1
            obs = _to_device_tensor(batch["obs_features"], device=device, channels_last=False)
            target = _to_device_tensor(batch["hr_target"], device=device, channels_last=False)
            burst = batch["lr_burst_patch"].to(device)
            shifts = batch["burst_shifts"].to(device)
            psf = build_scene_psf(batch, device)
            x0 = obs[:, mean_ch : mean_ch + 1]
            boundary = batch["boundary_weight"].to(device) if "boundary_weight" in batch else None

            pred = solver(x0, burst, shifts, psf, obs, frame_mask=mask)  # (1,1,h,w) broadcasts over B,M
            losses = criterion(pred, target, lr_observation=None, lr_obs=None, boundary_weight=boundary)
            # DC is now enforced ARCHITECTURALLY (end-on-DC + frozen eta, ACL-026). The soft DC loss
            # term is the falsified soft forward anchoring (ACL-017/019) and is optional. When its
            # weight is 0, `dc` is only a monitor, so avoid paying an extra A(x) every train step.
            # Logging/checkpoint eval still compute it periodically.
            dc = None
            if config.solver_dc_weight > 0:
                dc = terminal_dc_loss(pred, burst, shifts, psf, config.scale, config.solver_band_sigma,
                                      mask, config.solver_huber_delta)
            anneal = 1.0 if config.solver_prior_anneal_steps <= 0 else min(1.0, step / config.solver_prior_anneal_steps)
            total = anneal * losses["total"]
            if dc is not None:
                total = total + config.solver_dc_weight * dc

            if not torch.isfinite(total):
                dc_msg = "not-computed" if dc is None else str(dc)
                raise FloatingPointError(f"non-finite loss at step {step}: total={total}, dc={dc_msg}")
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(solver.parameters(), 1.0).item()
            optimizer.step()
            scheduler.step()
            progress.update(1)

            if step == 1 or step % config.log_every == 0:
                if dc is None:
                    with torch.no_grad():
                        dc_log = terminal_dc_loss(
                            pred, burst, shifts, psf, config.scale, config.solver_band_sigma,
                            mask, config.solver_huber_delta,
                        ).detach()
                else:
                    dc_log = dc.detach()
                eta = float(torch.nn.functional.softplus(solver.eta_raw.detach()).mean())
                resumed_steps = max(1, step - start_step)
                ms = (time.monotonic() - t0) * 1000.0 / resumed_steps
                progress.set_postfix_str(
                    f"total={float(total.detach()):.4f} dc={float(dc_log):.5f} eta={eta:.3f} gnorm={grad_norm:.2f}")
                # Break out every ContourSRLoss component (mse/highpass/edge/ssim/grad_vector, plus
                # flatness/laplacian when enabled). The criterion returns them all; solver_train used
                # to collapse them into the single loss/struct scalar, so the loss BALANCE was
                # invisible in TensorBoard/nohup — exactly what's needed to tune it (ACL-032).
                comp = {k: float(v.detach()) for k, v in losses.items() if k != "total"}
                comp_str = " ".join(f"{k}={val:.4f}" for k, val in comp.items())
                progress.write(
                    f"step={step} total={float(total.detach()):.5f} struct={float(losses['total'].detach()):.5f} "
                    f"dc={float(dc_log):.6f} {comp_str} anneal={anneal:.2f} eta={eta:.3f} gnorm={grad_norm:.2f} "
                    f"lr={scheduler.get_last_lr()[0]:.2e} {ms:.0f}ms/step")
                writer.add_scalar("loss/total", float(total.detach()), step)  # grand total = struct*anneal + w*dc
                writer.add_scalar("loss/struct", float(losses["total"].detach()), step)
                writer.add_scalar("loss/dc", float(dc_log), step)
                for _name, _val in comp.items():
                    writer.add_scalar(f"loss/{_name}", _val, step)
                writer.add_scalar("train/anneal", anneal, step)
                writer.add_scalar("train/eta", eta, step)
                writer.add_scalar("train/grad_norm", grad_norm, step)
                writer.add_scalar("train/lr", scheduler.get_last_lr()[0], step)
                writer.add_scalar("train/ms_per_step", ms, step)
            if config.tb_image_every > 0 and step % config.tb_image_every == 0:
                _log_pred_vs_target(writer, pred, target, step)
            if step % config.save_every == 0:
                ckpt = output_dir / f"solver_step_{step:06d}.pt"
                _save_solver_checkpoint(
                    ckpt,
                    step=step,
                    solver=solver,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    config=config,
                )
                progress.write(f"saved {ckpt}")
                real_metrics = maybe_log_solver_real_eval(
                    writer,
                    solver=solver,
                    config=real_eval_cfg,
                    training_config=config,
                    step=step,
                    device=device,
                )
                if real_metrics is not None:
                    progress.write(
                        "eval_real "
                        + " ".join(f"{key}={value:.6g}" for key, value in real_metrics.items())
                    )
                if device.type == "cuda":
                    torch.cuda.empty_cache()
            if synth_loader is not None:
                sm = maybe_log_synth_eval(
                    writer, model=solver, loader=synth_loader, forward_fn=_solver_forward,
                    eval_config=synth_eval_cfg, training_config=config, step=step, device=device)
                if sm is not None:
                    progress.write("eval_synth " + " ".join(f"{k}={v:.4g}" for k, v in sm.items()))
            if step >= config.total_steps:
                break
        if step >= config.total_steps:
            break

    progress.close()
    final = output_dir / "solver_final.pt"
    _save_solver_checkpoint(
        final,
        step=step,
        solver=solver,
        optimizer=optimizer,
        scheduler=scheduler,
        config=config,
    )
    print(f"saved {final}")
    maybe_log_solver_real_eval(
        writer,
        solver=solver,
        config=real_eval_cfg,
        training_config=config,
        step=step,
        device=device,
    )
    writer.close()
    return final


def main(argv: list[str] | None = None) -> None:
    train(config_from_args(argv))


if __name__ == "__main__":
    main()
