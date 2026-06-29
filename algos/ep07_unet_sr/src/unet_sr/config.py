"""Training configuration for EP07v2 UNet SR."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    training_pool_dir: str
    output_dir: str = "outputs/ep07_unet_sr"
    scale: int = 4
    in_channels: int = 5
    phase_bin_channels: int = 4  # hybrid drizzle input = 5 fused↑2x + this many precomputed phase-bin ch
    out_channels: int = 1
    base_channels: int = 64
    hr_upsampler: str = "bilinear"
    hr_res_blocks: int = 0
    patch_size_hr: int = 256
    batch_size: int = 4
    num_workers: int = 4
    total_steps: int = 50_000
    lr: float = 2e-4
    weight_decay: float = 1e-4
    edge_loss_weight: float = 0.05
    ssim_loss_weight: float = 0.15
    loss_type: str = "contour_sr"
    highpass_loss_weight: float = 1.0
    highpass_sigma: float = 5.0
    structure_boost: float = 4.0
    mse_loss_weight: float = 0.2
    edge_coarse_weight: float = 0.25
    grad_vector_weight: float = 0.3
    boundary_boost: float = 4.0
    boundary_tau_px: float = 2.5
    flatness_weight: float = 0.0
    flatness_tau: float = 0.25
    laplacian_weight: float = 0.0
    forward_model_weight: float = 0.0
    forward_model_psf_sigma: float = 0.5
    forward_model_band: str = "full"
    forward_model_band_sigma: float = 5.0
    input_mode: str = "lr"
    lr_warmup_steps: int = 500
    log_every: int = 50
    save_every: int = 1_000
    seed: int = 42
    device: str = "cuda"
    patches_per_scene: int = 64
    max_scene_cache: int = 0  # 0 = auto (= scenes_per_bucket)
    amp: bool = True
    compile_model: bool = False
    channels_last: bool = False
    tb_log_dir: str = ""
    tb_image_every: int = 0
    residual: bool = False
    residual_mode: str = "none"
    residual_penalty_weight: float = 0.0
    resume_from: str = ""
    prefetch_factor: int = 4
    scenes_per_bucket: int = 0  # 0 = auto (16)
    patches_per_fetch: int = 0  # 0 = auto (batch_size // 8)
    real_eval_enabled: bool = True
    real_eval_every: int = 0
    real_eval_frame_limit: int = 248
    real_eval_baseline_hr: str = ""
    real_eval_alignment_method: str = "contour_refined"
    real_eval_center_fraction: float = 1.0 / 3.0
    real_eval_zoom: float = 3.0
    real_eval_overlap: int = 128
    real_eval_tile_batch: int = 16
    # --- Held-out synthetic GT eval (PSNR / region-RMSE / defect boundary-F1 /
    # out-of-band). Eval scenes are a fixed tail of the pool, excluded from
    # training (no leakage). synth_eval_holdout=0 disables it. ---
    synth_eval_enabled: bool = True
    synth_eval_every: int = 0
    synth_eval_holdout: int = 0
    synth_eval_patches_per_scene: int = 2
    synth_eval_max_patches: int = 128
    # --- Physics-constrained unrolled solver (unroll_steps=0 keeps the plain UNet) ---
    unroll_steps: int = 0
    solver_m_frames: int = 16
    solver_band_sigma: float = 5.0
    solver_huber_delta: float = 0.0
    solver_share_weights: bool = True
    solver_eta_init: float = 0.5
    solver_learn_eta: bool = False  # False freezes eta (a learnable eta let the optimizer bypass the DC step, ACL-026)
    solver_dc_rim_lr_px: int = 8
    solver_dc_weight: float = 0.1
    solver_prior_anneal_steps: int = 0
    solver_no_drizzle: bool = False
    # Warm-start source for x0 in the hybrid (9ch) path. "phasebin" seeds x0 from the first
    # phase-bin drizzle channel (ch5) — the historical default, but it carries the phase-bin
    # coverage waffle (a 2-HR-px = 1-pitch checkerboard on flat background; ACL-032). "aligned_mean"
    # seeds x0 from the smooth fused aligned-mean (ch0) instead while KEEPING all 9 cond channels,
    # so the prox still sees the phase bins but the warm-start is de-waffled. No effect under
    # --solver-no-drizzle (that path already warm-starts from ch0).
    solver_warmstart: str = "phasebin"

    def validate(self) -> None:
        if self.device == "cpu" and self.amp:
            print("AMP requested on CPU; disabling AMP.")
            self.amp = False
        if self.residual_mode not in ("none", "drizzle2x"):
            raise ValueError("residual_mode must be 'none' or 'drizzle2x'")
        if self.residual_penalty_weight < 0:
            raise ValueError("residual_penalty_weight must be >= 0")
        if self.residual_mode == "drizzle2x":
            if self.input_mode != "hybrid_drizzle2x":
                raise ValueError("residual_mode='drizzle2x' requires input_mode='hybrid_drizzle2x'")
            if self.scale != 2:
                raise ValueError("residual_mode='drizzle2x' requires --scale 2")
            if self.residual:
                raise ValueError("residual_mode='drizzle2x' cannot be combined with --residual")
            if self.forward_model_weight > 0:
                raise ValueError("residual_mode='drizzle2x' is mutually exclusive with forward_model_weight > 0")
        if self.residual and self.input_mode == "hybrid_drizzle2x":
            raise ValueError("residual and hybrid_drizzle2x modes are mutually exclusive")
        if self.residual and self.in_channels == 5:
            self.in_channels = 6
            print("Residual mode: auto-set in_channels 5 → 6 (5 fused + 1 classical_sr)")
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if self.in_channels <= 0 or self.out_channels <= 0:
            raise ValueError("channel counts must be positive")
        if self.hr_upsampler not in ("bilinear", "pixelshuffle"):
            raise ValueError("hr_upsampler must be 'bilinear' or 'pixelshuffle'")
        if self.hr_res_blocks < 0:
            raise ValueError("hr_res_blocks must be >= 0")
        if self.patch_size_hr <= 0:
            raise ValueError("patch_size_hr must be positive")
        effective_scale = 1 if (self.residual or self.input_mode == "hybrid_drizzle2x") else self.scale
        if self.patch_size_hr % effective_scale != 0:
            raise ValueError("patch_size_hr must be divisible by scale")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.num_workers < 0:
            raise ValueError("num_workers must be >= 0")
        if self.total_steps <= 0:
            raise ValueError("total_steps must be positive")
        if self.lr <= 0:
            raise ValueError("lr must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be >= 0")
        if self.edge_loss_weight < 0:
            raise ValueError("edge_loss_weight must be >= 0")
        if self.ssim_loss_weight < 0:
            raise ValueError("ssim_loss_weight must be >= 0")
        if self.log_every <= 0 or self.save_every <= 0:
            raise ValueError("log_every and save_every must be positive")
        if self.loss_type not in ("thermal_sr", "contour_sr"):
            raise ValueError("loss_type must be 'thermal_sr' or 'contour_sr'")
        if self.highpass_loss_weight < 0:
            raise ValueError("highpass_loss_weight must be >= 0")
        if self.highpass_sigma < 0:
            raise ValueError("highpass_sigma must be >= 0")
        if self.structure_boost < 0:
            raise ValueError("structure_boost must be >= 0")
        if self.mse_loss_weight < 0:
            raise ValueError("mse_loss_weight must be >= 0")
        if self.edge_coarse_weight < 0:
            raise ValueError("edge_coarse_weight must be >= 0")
        if self.grad_vector_weight < 0:
            raise ValueError("grad_vector_weight must be >= 0")
        if not (0.0 <= self.boundary_boost < 10.0):
            raise ValueError("boundary_boost must satisfy 0 <= boundary_boost < 10")
        if self.boundary_tau_px <= 0:
            raise ValueError("boundary_tau_px must be > 0")
        if self.flatness_weight < 0:
            raise ValueError("flatness_weight must be >= 0")
        if self.flatness_tau <= 0:
            raise ValueError("flatness_tau must be > 0")
        if self.laplacian_weight < 0:
            raise ValueError("laplacian_weight must be >= 0")
        if self.forward_model_weight < 0:
            raise ValueError("forward_model_weight must be >= 0")
        if self.forward_model_psf_sigma < 0:
            raise ValueError("forward_model_psf_sigma must be >= 0")
        if self.forward_model_band not in ("full", "highpass"):
            raise ValueError("forward_model_band must be 'full' or 'highpass'")
        if self.forward_model_band_sigma < 0:
            raise ValueError("forward_model_band_sigma must be >= 0")
        if self.input_mode not in ("lr", "hybrid_drizzle2x"):
            raise ValueError("input_mode must be 'lr' or 'hybrid_drizzle2x'")
        if self.unroll_steps < 0:
            raise ValueError("unroll_steps must be >= 0")
        if self.unroll_steps > 0:
            if self.input_mode != "hybrid_drizzle2x":
                raise ValueError("unroll_steps>0 (unrolled solver) requires --input-mode hybrid_drizzle2x")
            if self.scale != 2:
                raise ValueError("unroll_steps>0 requires --scale 2")
            if self.solver_m_frames < 1:
                raise ValueError("solver_m_frames must be >= 1")
            if self.solver_warmstart not in ("phasebin", "aligned_mean"):
                raise ValueError("solver_warmstart must be 'phasebin' or 'aligned_mean'")
        if self.input_mode == "hybrid_drizzle2x" and self.forward_model_weight > 0 and self.scale != 2:
            raise ValueError(
                "input_mode='hybrid_drizzle2x' with forward_model_weight > 0 requires --scale 2 "
                "so the dataset can supply a legal 1x lr_obs patch for 2x→1x forward consistency."
            )
        if self.input_mode == "hybrid_drizzle2x":
            hybrid_ch = 5 + int(self.phase_bin_channels)
            if self.in_channels == 5:
                self.in_channels = hybrid_ch
                print(f"Hybrid drizzle 2x mode: auto-set in_channels 5 → {hybrid_ch} "
                      f"(5 fused↑2x + {self.phase_bin_channels} precomputed phase-bin drizzle@2x)")
        if self.lr_warmup_steps < 0:
            raise ValueError("lr_warmup_steps must be >= 0")
        if self.patches_per_scene <= 0:
            raise ValueError("patches_per_scene must be positive")
        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")

        # --- Auto-compute DataLoader cache parameters ---
        # With worker-scene affinity the sampler guarantees each worker only
        # touches its own scene partition, so:
        #   patches_per_fetch  → controls batch diversity (~8 scenes/batch)
        #   scenes_per_bucket  → controls per-worker cache size (16 = 0.8 GB)
        #   max_scene_cache    → must equal scenes_per_bucket for 100% hit
        auto_parts: list[str] = []
        if self.patches_per_fetch <= 0:
            self.patches_per_fetch = max(1, self.batch_size // 8)
            auto_parts.append(f"patches_per_fetch={self.patches_per_fetch}")
        if self.scenes_per_bucket <= 0:
            self.scenes_per_bucket = 16
            auto_parts.append(f"scenes_per_bucket={self.scenes_per_bucket}")
        if self.max_scene_cache <= 0:
            self.max_scene_cache = self.scenes_per_bucket
            auto_parts.append(f"max_scene_cache={self.max_scene_cache}")
        if auto_parts:
            print(f"Auto-tuned cache params: {', '.join(auto_parts)}")

        if self.max_scene_cache <= 0:
            raise ValueError("max_scene_cache must be positive")
        if self.scenes_per_bucket <= 0:
            raise ValueError("scenes_per_bucket must be positive")
        if self.patches_per_fetch <= 0:
            raise ValueError("patches_per_fetch must be positive")
        if self.real_eval_every < 0:
            raise ValueError("real_eval_every must be >= 0")
        if self.real_eval_frame_limit <= 0:
            raise ValueError("real_eval_frame_limit must be positive")
        if not (0.0 < self.real_eval_center_fraction <= 1.0):
            raise ValueError("real_eval_center_fraction must be in (0, 1]")
        if self.real_eval_zoom <= 0:
            raise ValueError("real_eval_zoom must be positive")
        if self.real_eval_overlap < 0 or self.real_eval_overlap >= self.patch_size_hr:
            raise ValueError("real_eval_overlap must satisfy 0 <= overlap < patch_size_hr")
        if self.real_eval_tile_batch <= 0:
            raise ValueError("real_eval_tile_batch must be positive")
        if self.synth_eval_every < 0:
            raise ValueError("synth_eval_every must be >= 0")
        if self.synth_eval_holdout < 0:
            raise ValueError("synth_eval_holdout must be >= 0")
        if self.synth_eval_patches_per_scene <= 0:
            raise ValueError("synth_eval_patches_per_scene must be > 0")
        if self.synth_eval_max_patches <= 0:
            raise ValueError("synth_eval_max_patches must be > 0")
        if not self.tb_log_dir:
            self.tb_log_dir = str(Path(self.output_dir) / "tb_logs")
        if self.tb_image_every < 0:
            raise ValueError("tb_image_every must be >= 0 (0 disables TCForge batch images)")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train EP07v2 compact-scene UNet thermal SR.")
    parser.add_argument("--training-pool-dir", required=True, help="Directory containing manifest.csv and scene_* dirs.")
    parser.add_argument("--output-dir", default=TrainingConfig.output_dir, help="Directory for checkpoints and logs.")
    parser.add_argument("--total-steps", type=int, default=TrainingConfig.total_steps)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--patch-size-hr", type=int, default=TrainingConfig.patch_size_hr)
    parser.add_argument("--num-workers", type=int, default=TrainingConfig.num_workers)
    parser.add_argument("--device", default=TrainingConfig.device)
    parser.add_argument("--scale", type=int, default=TrainingConfig.scale)
    parser.add_argument("--base-channels", type=int, default=TrainingConfig.base_channels)
    parser.add_argument(
        "--hr-upsampler",
        default=TrainingConfig.hr_upsampler,
        choices=["bilinear", "pixelshuffle"],
        help="Final HR upsampler head. Use pixelshuffle for V8_1B; default preserves legacy bilinear.",
    )
    parser.add_argument(
        "--hr-res-blocks",
        type=int,
        default=TrainingConfig.hr_res_blocks,
        help="Number of no-norm HR residual refine blocks after PixelShuffle (default: 0).",
    )
    parser.add_argument("--lr", type=float, default=TrainingConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainingConfig.weight_decay)
    parser.add_argument("--edge-loss-weight", type=float, default=TrainingConfig.edge_loss_weight)
    parser.add_argument("--ssim-loss-weight", type=float, default=TrainingConfig.ssim_loss_weight)
    parser.add_argument(
        "--loss-type",
        default=TrainingConfig.loss_type,
        choices=["thermal_sr", "contour_sr"],
        help="Loss function type: 'contour_sr' (default, structure-focused) or 'thermal_sr' (legacy MSE-based).",
    )
    parser.add_argument("--highpass-loss-weight", type=float, default=TrainingConfig.highpass_loss_weight)
    parser.add_argument("--highpass-sigma", type=float, default=TrainingConfig.highpass_sigma,
                        help="Gaussian sigma for highpass filter in contour_sr loss (default: 5.0).")
    parser.add_argument("--structure-boost", type=float, default=TrainingConfig.structure_boost,
                        help="Gradient-based structure weight boost: edge pixels get "
                             "(1 + structure_boost)x weight in highpass loss (default: 4.0).")
    parser.add_argument("--mse-loss-weight", type=float, default=TrainingConfig.mse_loss_weight,
                        help="Weight on raw MSE loss to anchor DC/low-freq and protect gaps (default: 0.2).")
    parser.add_argument("--edge-coarse-weight", type=float, default=TrainingConfig.edge_coarse_weight,
                        help="Weight on 2x-downsampled Sobel edge loss for connectivity (default: 0.25).")
    parser.add_argument("--grad-vector-weight", type=float, default=TrainingConfig.grad_vector_weight,
                        help="Full Sobel gradient vector (gx,gy) matching loss: catches thickening, "
                             "merging, disconnection via direction+magnitude (default: 0.3).")
    parser.add_argument("--boundary-boost", type=float, default=TrainingConfig.boundary_boost,
                        help="Geometry-agnostic boundary emphasis: pixels within boundary_tau_px of any "
                             "mask edge (chip outline, hole rim, crack wall, notch) get up to "
                             "(1 + boundary_boost)x weight in highpass/grad-vector loss. Replaces the old "
                             "thin/gap line priors (default: 4.0; use 0.0 to disable).")
    parser.add_argument("--boundary-tau-px", type=float, default=TrainingConfig.boundary_tau_px,
                        help="Gaussian falloff (HR px) of the boundary-emphasis weight (default: 2.5).")
    parser.add_argument("--flatness-weight", type=float, default=TrainingConfig.flatness_weight,
                        help="Isothermal-flatness loss: penalises |grad(pred)| where the GT is flat "
                             "(interiors + background). Encodes the near-isothermal prior "
                             "(default: 0.0, disabled; enable for v4 data).")
    parser.add_argument("--flatness-tau", type=float, default=TrainingConfig.flatness_tau,
                        help="Normalised target-gradient scale below which flatness is enforced (default: 0.25).")
    parser.add_argument("--laplacian-weight", type=float, default=TrainingConfig.laplacian_weight,
                        help="Asymmetric Laplacian sharpness loss: penalises pred being blurrier "
                             "than target, preventing thin-line thickening (default: 0.0, disabled).")
    parser.add_argument("--forward-model-weight", type=float, default=TrainingConfig.forward_model_weight,
                        help="PSF forward-model consistency loss: HR pred blurred by PSF and "
                             "downsampled must match LR observation (default: 0.0, disabled).")
    parser.add_argument("--forward-model-psf-sigma", type=float, default=TrainingConfig.forward_model_psf_sigma,
                        help="PSF Gaussian sigma at LR pixel scale for forward model loss (default: 0.5).")
    parser.add_argument(
        "--forward-model-band",
        default=TrainingConfig.forward_model_band,
        choices=["full", "highpass"],
        help="Forward model frequency band: 'full' (legacy) or 'highpass' (subtract σ blur, "
             "avoids low-freq gradient conflict with structure losses).",
    )
    parser.add_argument(
        "--forward-model-band-sigma",
        type=float,
        default=TrainingConfig.forward_model_band_sigma,
        help="Gaussian sigma (LR px) for highpass band in forward model (default: 5.0, "
             "matches pipeline σ_bg=5 convention).",
    )
    parser.add_argument(
        "--input-mode",
        default=TrainingConfig.input_mode,
        choices=["lr", "hybrid_drizzle2x"],
        help="Input mode: 'lr' (5ch 1x obs, default) or 'hybrid_drizzle2x' "
             "(9ch 2x: 5ch fused↑2x + 4ch precomputed phase-bin drizzle@2x). "
             "Hybrid requires training pool with phase_bin_drizzle_2x.npy.",
    )
    parser.add_argument("--lr-warmup-steps", type=int, default=TrainingConfig.lr_warmup_steps,
                        help="Linear LR warmup from 0 to target LR over this many steps (default: 500).")
    parser.add_argument("--log-every", type=int, default=TrainingConfig.log_every)
    parser.add_argument("--save-every", type=int, default=TrainingConfig.save_every)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--patches-per-scene", type=int, default=TrainingConfig.patches_per_scene)
    parser.add_argument("--max-scene-cache", type=int, default=TrainingConfig.max_scene_cache,
                        help="LRU cache size per worker (0 = auto → scenes_per_bucket). "
                             "With worker-scene affinity, should equal scenes_per_bucket.")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=TrainingConfig.amp,
        help="Enable CUDA AMP mixed precision. Automatically disabled on CPU.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        dest="compile_model",
        default=TrainingConfig.compile_model,
        help="Enable torch.compile before DDP wrapping.",
    )
    parser.add_argument(
        "--channels-last",
        action=argparse.BooleanOptionalAction,
        default=TrainingConfig.channels_last,
        help="Use NHWC/channels-last tensors for CUDA convolution throughput experiments.",
    )
    parser.add_argument(
        "--tb-log-dir",
        default=TrainingConfig.tb_log_dir,
        help="TensorBoard log directory. Defaults to {output_dir}/tb_logs.",
    )
    parser.add_argument(
        "--tb-image-every",
        type=int,
        default=TrainingConfig.tb_image_every,
        help="Log TCForge synthetic batch pred/target images every N steps (0 = disabled).",
    )
    parser.add_argument(
        "--residual",
        action="store_true",
        default=TrainingConfig.residual,
        help="Residual refinement mode: 6ch@2x input (5 fused upsampled + classical_sr), model learns residual.",
    )
    parser.add_argument(
        "--residual-mode",
        default=TrainingConfig.residual_mode,
        choices=["none", "drizzle2x"],
        help="Observation residual parameterization. 'drizzle2x' predicts a delta over hybrid ch5 "
             "(first phase-bin channel @2x) and requires --input-mode hybrid_drizzle2x --scale 2.",
    )
    parser.add_argument(
        "--residual-penalty-weight",
        type=float,
        default=TrainingConfig.residual_penalty_weight,
        help="L1 penalty weight on the model delta when --residual-mode drizzle2x is enabled.",
    )
    parser.add_argument(
        "--resume",
        dest="resume_from",
        default=TrainingConfig.resume_from,
        metavar="PATH",
        help="Resume training from a checkpoint .pt file (restores model, optimizer, scheduler, step).",
    )
    parser.add_argument(
        "--prefetch-factor",
        type=int,
        default=TrainingConfig.prefetch_factor,
        help="DataLoader prefetch queue depth per worker (default: 4).",
    )
    parser.add_argument(
        "--scenes-per-bucket",
        type=int,
        default=TrainingConfig.scenes_per_bucket,
        help="Scenes interleaved per sampling bucket (0 = auto → 16). "
             "Controls per-worker cache size; rarely needs manual tuning.",
    )
    parser.add_argument(
        "--patches-per-fetch",
        type=int,
        default=TrainingConfig.patches_per_fetch,
        help="Consecutive patches per scene in round-robin (0 = auto → batch_size//8). "
             "Controls scenes-per-batch diversity.",
    )
    parser.add_argument(
        "--no-real-eval",
        action="store_false",
        dest="real_eval_enabled",
        default=TrainingConfig.real_eval_enabled,
        help="Disable real-data EP11-style TensorBoard eval at checkpoint steps.",
    )
    parser.add_argument(
        "--real-eval-every",
        type=int,
        default=TrainingConfig.real_eval_every,
        help="Run real-data eval every N steps (0 = save_every).",
    )
    parser.add_argument(
        "--real-eval-frame-limit",
        type=int,
        default=TrainingConfig.real_eval_frame_limit,
        help="Number of clean main-session frames for checkpoint eval (default: 248, same as EP11).",
    )
    parser.add_argument(
        "--real-eval-baseline-hr",
        default=TrainingConfig.real_eval_baseline_hr,
        help="Optional TGV highpass baseline .npy for side-by-side TensorBoard panels.",
    )
    parser.add_argument(
        "--real-eval-alignment-method",
        default=TrainingConfig.real_eval_alignment_method,
        help="Alignment method passed to EP04 shifts loader during real eval.",
    )
    parser.add_argument(
        "--real-eval-center-fraction",
        type=float,
        default=TrainingConfig.real_eval_center_fraction,
        help="Center ROI fraction for eval_real TensorBoard images.",
    )
    parser.add_argument(
        "--real-eval-zoom",
        type=float,
        default=TrainingConfig.real_eval_zoom,
        help="Display zoom on center ROI (default 3.0 = EP11 center_zoom3x; SR scale stays --scale).",
    )
    parser.add_argument(
        "--real-eval-overlap",
        type=int,
        default=TrainingConfig.real_eval_overlap,
        help="Tiled inference overlap for real eval.",
    )
    parser.add_argument(
        "--real-eval-tile-batch",
        type=int,
        default=TrainingConfig.real_eval_tile_batch,
        help="Number of full-frame real-eval tiles evaluated per solver batch (default: 16).",
    )
    # --- Held-out synthetic GT eval ---
    parser.add_argument("--synth-eval", action=argparse.BooleanOptionalAction,
                        default=TrainingConfig.synth_eval_enabled, dest="synth_eval_enabled",
                        help="Held-out synthetic GT eval (PSNR/region-RMSE/boundary-F1/out-of-band). Default ON.")
    parser.add_argument("--synth-eval-every", type=int, default=TrainingConfig.synth_eval_every,
                        help="Steps between synthetic GT evals (0 = use --save-every).")
    parser.add_argument("--synth-eval-holdout", type=int, default=TrainingConfig.synth_eval_holdout,
                        help="Tail scenes held out from training for GT eval. 0 disables; "
                             "~200 recommended for the 5k v4 pool.")
    parser.add_argument("--synth-eval-patches-per-scene", type=int,
                        default=TrainingConfig.synth_eval_patches_per_scene,
                        help="Deterministic eval patches drawn per held-out scene (default 2).")
    parser.add_argument("--synth-eval-max-patches", type=int, default=TrainingConfig.synth_eval_max_patches,
                        help="Cap on total eval patches per eval pass (default 128).")
    # --- Unrolled solver (used by solver_train.py / test_gate_c_smoke.py) ---
    parser.add_argument("--unroll-steps", type=int, default=TrainingConfig.unroll_steps,
                        help="K unroll iterations (0 = plain UNet; >0 enables the physics-constrained solver).")
    parser.add_argument("--solver-m-frames", type=int, default=TrainingConfig.solver_m_frames,
                        help="Number of burst frames fed to the DC term per sample (fixed; default 16).")
    parser.add_argument("--solver-band-sigma", type=float, default=TrainingConfig.solver_band_sigma,
                        help="Highpass sigma (LR px) for the band-limited DC term (rejects drift; default 5.0).")
    parser.add_argument("--solver-huber-delta", type=float, default=TrainingConfig.solver_huber_delta,
                        help="Huber delta for a robust DC term (0 = plain L2; defects/stripe noise).")
    parser.add_argument("--solver-share-weights", action=argparse.BooleanOptionalAction,
                        default=TrainingConfig.solver_share_weights, help="Share the prox UNet across unroll steps.")
    parser.add_argument("--solver-eta-init", type=float, default=TrainingConfig.solver_eta_init,
                        help="DC step size (frozen by default; see --solver-learn-eta).")
    parser.add_argument("--solver-learn-eta", action=argparse.BooleanOptionalAction,
                        default=TrainingConfig.solver_learn_eta,
                        help="Learn the per-step DC step size eta. Default OFF (frozen): a learnable eta "
                             "let the optimizer drive the DC step toward 0 and bypass the constraint (ACL-026).")
    parser.add_argument("--solver-dc-rim-lr-px", type=int, default=TrainingConfig.solver_dc_rim_lr_px,
                        help="LR-px rim masked out of the DC term (patch-edge zero-padding artifact; default 8).")
    parser.add_argument("--solver-dc-weight", type=float, default=TrainingConfig.solver_dc_weight,
                        help="Weight of the terminal data-consistency loss term (default 0.1).")
    parser.add_argument("--solver-prior-anneal-steps", type=int, default=TrainingConfig.solver_prior_anneal_steps,
                        help="Linearly ramp the structure-prior loss weight from 0->1 over N steps (0 = off; "
                             "fights the fidelity cliff by letting DC dominate early).")
    parser.add_argument("--solver-no-drizzle", action="store_true", default=TrainingConfig.solver_no_drizzle,
                        help="Lean solver input: drop the drizzle entirely (no on-the-fly cost, no precomputed "
                             "variants/disk). Warm-start from upsampled aligned_mean; the DC term carries the "
                             "multi-frame SR signal. cond becomes 5ch (vs 9ch hybrid).")
    parser.add_argument("--solver-warmstart", choices=("phasebin", "aligned_mean"),
                        default=TrainingConfig.solver_warmstart,
                        help="Hybrid (9ch) warm-start source for x0: 'phasebin' = first phase-bin drizzle "
                             "channel (ch5, default, carries the 2px coverage waffle); 'aligned_mean' = smooth "
                             "fused aligned-mean (ch0), de-waffled, keeps all 9 cond channels (ACL-032).")
    return parser


def config_from_args(argv: list[str] | None = None) -> TrainingConfig:
    args = build_arg_parser().parse_args(argv)
    cfg = TrainingConfig(
        training_pool_dir=args.training_pool_dir,
        output_dir=args.output_dir,
        scale=args.scale,
        base_channels=args.base_channels,
        hr_upsampler=args.hr_upsampler,
        hr_res_blocks=args.hr_res_blocks,
        patch_size_hr=args.patch_size_hr,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        total_steps=args.total_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        edge_loss_weight=args.edge_loss_weight,
        ssim_loss_weight=args.ssim_loss_weight,
        loss_type=args.loss_type,
        highpass_loss_weight=args.highpass_loss_weight,
        highpass_sigma=args.highpass_sigma,
        structure_boost=args.structure_boost,
        mse_loss_weight=args.mse_loss_weight,
        edge_coarse_weight=args.edge_coarse_weight,
        grad_vector_weight=args.grad_vector_weight,
        boundary_boost=args.boundary_boost,
        boundary_tau_px=args.boundary_tau_px,
        flatness_weight=args.flatness_weight,
        flatness_tau=args.flatness_tau,
        laplacian_weight=args.laplacian_weight,
        forward_model_weight=args.forward_model_weight,
        forward_model_psf_sigma=args.forward_model_psf_sigma,
        forward_model_band=args.forward_model_band,
        forward_model_band_sigma=args.forward_model_band_sigma,
        input_mode=args.input_mode,
        lr_warmup_steps=args.lr_warmup_steps,
        log_every=args.log_every,
        save_every=args.save_every,
        seed=args.seed,
        device=args.device,
        patches_per_scene=args.patches_per_scene,
        max_scene_cache=args.max_scene_cache,
        amp=args.amp,
        compile_model=args.compile_model,
        channels_last=args.channels_last,
        tb_log_dir=args.tb_log_dir,
        tb_image_every=args.tb_image_every,
        residual=args.residual,
        residual_mode=args.residual_mode,
        residual_penalty_weight=args.residual_penalty_weight,
        resume_from=args.resume_from,
        prefetch_factor=args.prefetch_factor,
        scenes_per_bucket=args.scenes_per_bucket,
        patches_per_fetch=args.patches_per_fetch,
        real_eval_enabled=args.real_eval_enabled,
        real_eval_every=args.real_eval_every,
        real_eval_frame_limit=args.real_eval_frame_limit,
        real_eval_baseline_hr=args.real_eval_baseline_hr,
        real_eval_alignment_method=args.real_eval_alignment_method,
        real_eval_center_fraction=args.real_eval_center_fraction,
        real_eval_zoom=args.real_eval_zoom,
        real_eval_overlap=args.real_eval_overlap,
        real_eval_tile_batch=args.real_eval_tile_batch,
        synth_eval_enabled=args.synth_eval_enabled,
        synth_eval_every=args.synth_eval_every,
        synth_eval_holdout=args.synth_eval_holdout,
        synth_eval_patches_per_scene=args.synth_eval_patches_per_scene,
        synth_eval_max_patches=args.synth_eval_max_patches,
        unroll_steps=args.unroll_steps,
        solver_m_frames=args.solver_m_frames,
        solver_band_sigma=args.solver_band_sigma,
        solver_huber_delta=args.solver_huber_delta,
        solver_share_weights=args.solver_share_weights,
        solver_eta_init=args.solver_eta_init,
        solver_learn_eta=args.solver_learn_eta,
        solver_dc_rim_lr_px=args.solver_dc_rim_lr_px,
        solver_dc_weight=args.solver_dc_weight,
        solver_prior_anneal_steps=args.solver_prior_anneal_steps,
        solver_no_drizzle=args.solver_no_drizzle,
        solver_warmstart=args.solver_warmstart,
    )
    cfg.validate()
    return cfg
