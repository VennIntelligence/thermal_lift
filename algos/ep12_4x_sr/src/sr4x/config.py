"""Training configuration for EP12 drizzle-informed 4x SR."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    training_pool_dir: str
    output_dir: str = "outputs/ep12_4x_sr"
    scale: int = 4
    drizzle_scale: int = 2
    in_channels: int = 8
    out_channels: int = 1
    base_channels: int = 48
    unet_depth: int = 4
    dilated_bottleneck: bool = True
    predict_log_variance: bool = True
    patch_size: int = 256
    batch_size: int = 4
    num_workers: int = 4
    total_steps: int = 80_000
    lr: float = 2e-4
    weight_decay: float = 1e-4
    lf_loss_weight: float = 1.0
    hf_loss_weight: float = 0.3
    edge_loss_weight: float = 0.1
    forward_loss_weight: float = 0.2
    nll_loss_weight: float = 0.05
    coverage_loss_gain: float = 4.0
    edge_coarse_weight: float = 0.25
    hf_detail_weight: float = 0.3
    hf_detail_gain: float = 4.0
    lr_warmup_steps: int = 500
    sigma_lf: float = 8.0
    psf_sigma_lr_px: float = 0.25
    min_log_variance: float = -8.0
    max_log_variance: float = 4.0
    log_every: int = 50
    save_every: int = 2_000
    seed: int = 42
    device: str = "cuda"
    patches_per_scene: int = 64
    max_scene_cache: int = 0  # 0 = auto (= scenes_per_bucket for 100% cache hit)
    prefetch_factor: int = 4
    scenes_per_bucket: int = 16
    patches_per_fetch: int = 0  # 0 = auto (batch_size // 8)
    amp: bool = True
    compile_model: bool = False
    channels_last: bool = False
    tb_log_dir: str = ""
    tb_image_every: int = 0
    resume_from: str = ""
    include_multiscale: bool = False
    defer_1x_upsample: bool = False
    burst_augment: bool = True
    burst_keep_min: float = 0.6
    burst_keep_max: float = 1.0
    min_burst_frames: int = 30
    shift_noise_std_px: float = 0.05
    drizzle_kernel: str = "bilinear"
    real_eval_enabled: bool = True
    real_eval_every: int = 0
    real_eval_frame_limit: int = 248
    real_eval_alignment_method: str = "contour_refined"
    real_eval_baseline_hr: str = ""
    real_eval_center_fraction: float = 1.0 / 3.0
    real_eval_zoom: float = 3.0
    real_eval_overlap: int = 64

    def validate(self) -> None:
        if self.device == "cpu" and self.amp:
            print("AMP requested on CPU; disabling AMP.")
            self.amp = False
        if self.scale <= 0:
            raise ValueError("scale must be positive")
        if self.drizzle_scale <= 0 or self.drizzle_scale > self.scale:
            raise ValueError("drizzle_scale must be positive and <= scale")
        if self.scale % self.drizzle_scale != 0:
            raise ValueError("scale must be divisible by drizzle_scale")
        if self.include_multiscale and self.drizzle_scale != self.scale:
            print("WARNING: include_multiscale not supported when drizzle_scale != scale; disabling.")
            self.include_multiscale = False
        if self.in_channels < 3:
            raise ValueError("in_channels must include at least drizzle mean/coverage/variance")
        expected_channels = 11 if self.include_multiscale else 8
        if self.in_channels != expected_channels:
            print(
                f"Input channel contract: auto-set in_channels {self.in_channels} -> "
                f"{expected_channels} ({'4x+2x+1x' if self.include_multiscale else '4x+1x'} features)."
            )
            self.in_channels = expected_channels
        if self.out_channels != 1:
            raise ValueError("EP12 losses expect a single temperature output channel")
        if self.base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if self.unet_depth < 3 or self.unet_depth > 5:
            raise ValueError("unet_depth must be between 3 and 5")
        if self.patch_size <= 0 or self.patch_size % self.scale != 0:
            raise ValueError("patch_size must be positive and divisible by scale")
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
        for name in (
            "lf_loss_weight",
            "hf_loss_weight",
            "edge_loss_weight",
            "forward_loss_weight",
            "nll_loss_weight",
            "coverage_loss_gain",
            "edge_coarse_weight",
            "hf_detail_weight",
            "hf_detail_gain",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.lr_warmup_steps < 0:
            raise ValueError("lr_warmup_steps must be >= 0")
        if self.sigma_lf <= 0:
            raise ValueError("sigma_lf must be positive")
        if self.psf_sigma_lr_px < 0:
            raise ValueError("psf_sigma_lr_px must be >= 0")
        if self.min_log_variance >= self.max_log_variance:
            raise ValueError("min_log_variance must be < max_log_variance")
        if self.nll_loss_weight > 0 and not self.predict_log_variance:
            print("nll_loss_weight > 0 requires predict_log_variance; disabling NLL.")
            self.nll_loss_weight = 0.0
        if self.log_every <= 0 or self.save_every <= 0:
            raise ValueError("log_every and save_every must be positive")
        if self.patches_per_scene <= 0:
            raise ValueError("patches_per_scene must be positive")
        if self.prefetch_factor <= 0:
            raise ValueError("prefetch_factor must be positive")
        if self.scenes_per_bucket <= 0:
            raise ValueError("scenes_per_bucket must be positive")

        # --- Auto-compute DataLoader cache parameters ---
        # With worker-scene affinity the sampler guarantees each worker only
        # touches its own scene partition, so:
        #   patches_per_fetch  → controls batch diversity (~8 scenes/batch)
        #   scenes_per_bucket  → controls per-worker cache size
        #   max_scene_cache    → must equal scenes_per_bucket for 100% hit
        auto_parts: list[str] = []
        if self.patches_per_fetch <= 0:
            self.patches_per_fetch = max(1, self.batch_size // 8)
            auto_parts.append(f"patches_per_fetch={self.patches_per_fetch}")
        if self.max_scene_cache <= 0:
            self.max_scene_cache = self.scenes_per_bucket
            auto_parts.append(f"max_scene_cache={self.max_scene_cache}")
        if auto_parts:
            print(f"Auto-tuned cache params: {', '.join(auto_parts)}")

        if self.max_scene_cache <= 0:
            raise ValueError("max_scene_cache must be positive")
        if self.patches_per_fetch <= 0:
            raise ValueError("patches_per_fetch must be positive")
        if not (0.0 < self.burst_keep_min <= self.burst_keep_max <= 1.0):
            raise ValueError("burst_keep_min/max must satisfy 0 < min <= max <= 1")
        if self.min_burst_frames <= 0:
            raise ValueError("min_burst_frames must be positive")
        if self.shift_noise_std_px < 0:
            raise ValueError("shift_noise_std_px must be >= 0")
        if self.drizzle_kernel not in {"nearest", "bilinear"}:
            raise ValueError("drizzle_kernel must be 'nearest' or 'bilinear'")
        if not self.tb_log_dir:
            self.tb_log_dir = str(Path(self.output_dir) / "tb_logs")
        if self.tb_image_every <= 0:
            self.tb_image_every = self.save_every


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train EP12 drizzle-informed 4x thermal SR.")
    parser.add_argument("--training-pool-dir", required=True, help="Directory containing manifest.csv and scene dirs.")
    parser.add_argument("--output-dir", default=TrainingConfig.output_dir)
    parser.add_argument("--scale", type=int, default=TrainingConfig.scale)
    parser.add_argument("--drizzle-scale", type=int, default=TrainingConfig.drizzle_scale,
                        help="Drizzle accumulation scale (2=no checkerboard, 4=legacy same-grid).")
    parser.add_argument("--in-channels", type=int, default=TrainingConfig.in_channels)
    parser.add_argument("--base-channels", type=int, default=TrainingConfig.base_channels)
    parser.add_argument("--unet-depth", type=int, default=TrainingConfig.unet_depth)
    parser.add_argument("--dilated-bottleneck", action=argparse.BooleanOptionalAction, default=TrainingConfig.dilated_bottleneck)
    parser.add_argument("--predict-log-variance", action=argparse.BooleanOptionalAction, default=TrainingConfig.predict_log_variance)
    parser.add_argument("--patch-size", type=int, default=TrainingConfig.patch_size)
    parser.add_argument("--batch-size", type=int, default=TrainingConfig.batch_size)
    parser.add_argument("--num-workers", type=int, default=TrainingConfig.num_workers)
    parser.add_argument("--total-steps", type=int, default=TrainingConfig.total_steps)
    parser.add_argument("--lr", type=float, default=TrainingConfig.lr)
    parser.add_argument("--weight-decay", type=float, default=TrainingConfig.weight_decay)
    parser.add_argument("--lf-loss-weight", type=float, default=TrainingConfig.lf_loss_weight)
    parser.add_argument("--hf-loss-weight", type=float, default=TrainingConfig.hf_loss_weight)
    parser.add_argument("--edge-loss-weight", type=float, default=TrainingConfig.edge_loss_weight)
    parser.add_argument("--forward-loss-weight", type=float, default=TrainingConfig.forward_loss_weight)
    parser.add_argument("--nll-loss-weight", type=float, default=TrainingConfig.nll_loss_weight)
    parser.add_argument("--coverage-loss-gain", type=float, default=TrainingConfig.coverage_loss_gain)
    parser.add_argument("--edge-coarse-weight", type=float, default=TrainingConfig.edge_coarse_weight,
                        help="Weight on 2x-downsampled coarse-scale Sobel edge loss (default: 0.25).")
    parser.add_argument("--hf-detail-weight", type=float, default=TrainingConfig.hf_detail_weight,
                        help="Weight on inverse-coverage-weighted HF detail L1 loss (default: 0.3).")
    parser.add_argument("--hf-detail-gain", type=float, default=TrainingConfig.hf_detail_gain,
                        help="Gain for inverse coverage weighting in HF detail loss (default: 4.0).")
    parser.add_argument("--lr-warmup-steps", type=int, default=TrainingConfig.lr_warmup_steps,
                        help="Linear LR warmup from 0 to target LR over this many steps (default: 500).")
    parser.add_argument("--sigma-lf", type=float, default=TrainingConfig.sigma_lf)
    parser.add_argument("--psf-sigma-lr-px", type=float, default=TrainingConfig.psf_sigma_lr_px)
    parser.add_argument("--min-log-variance", type=float, default=TrainingConfig.min_log_variance)
    parser.add_argument("--max-log-variance", type=float, default=TrainingConfig.max_log_variance)
    parser.add_argument("--log-every", type=int, default=TrainingConfig.log_every)
    parser.add_argument("--save-every", type=int, default=TrainingConfig.save_every)
    parser.add_argument("--seed", type=int, default=TrainingConfig.seed)
    parser.add_argument("--device", default=TrainingConfig.device)
    parser.add_argument("--patches-per-scene", type=int, default=TrainingConfig.patches_per_scene)
    parser.add_argument("--max-scene-cache", type=int, default=TrainingConfig.max_scene_cache,
                        help="LRU cache size per worker (0 = auto → scenes_per_bucket). "
                             "With worker-scene affinity, should equal scenes_per_bucket.")
    parser.add_argument("--prefetch-factor", type=int, default=TrainingConfig.prefetch_factor)
    parser.add_argument("--scenes-per-bucket", type=int, default=TrainingConfig.scenes_per_bucket)
    parser.add_argument("--patches-per-fetch", type=int, default=TrainingConfig.patches_per_fetch,
                        help="Consecutive patches per scene in round-robin (0 = auto → batch_size//8). "
                             "Controls scenes-per-batch diversity.")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=TrainingConfig.amp)
    parser.add_argument("--compile", action="store_true", dest="compile_model", default=TrainingConfig.compile_model)
    parser.add_argument("--channels-last", action=argparse.BooleanOptionalAction, default=TrainingConfig.channels_last)
    parser.add_argument("--tb-log-dir", default=TrainingConfig.tb_log_dir)
    parser.add_argument("--tb-image-every", type=int, default=TrainingConfig.tb_image_every)
    parser.add_argument("--resume", dest="resume_from", default=TrainingConfig.resume_from, metavar="PATH")
    parser.add_argument("--include-multiscale", action="store_true", default=TrainingConfig.include_multiscale)
    parser.add_argument(
        "--defer-1x-upsample",
        action=argparse.BooleanOptionalAction,
        default=TrainingConfig.defer_1x_upsample,
        help="Return 1x feature patches separately and upsample them on GPU in the training loop.",
    )
    parser.add_argument("--burst-augment", action=argparse.BooleanOptionalAction, default=TrainingConfig.burst_augment)
    parser.add_argument("--burst-keep-min", type=float, default=TrainingConfig.burst_keep_min)
    parser.add_argument("--burst-keep-max", type=float, default=TrainingConfig.burst_keep_max)
    parser.add_argument("--min-burst-frames", type=int, default=TrainingConfig.min_burst_frames)
    parser.add_argument("--shift-noise-std-px", type=float, default=TrainingConfig.shift_noise_std_px)
    parser.add_argument("--drizzle-kernel", choices=["nearest", "bilinear"], default=TrainingConfig.drizzle_kernel)
    parser.add_argument(
        "--no-real-eval",
        action="store_false",
        dest="real_eval_enabled",
        default=TrainingConfig.real_eval_enabled,
        help="Disable real-data TensorBoard eval at checkpoint steps.",
    )
    parser.add_argument("--real-eval-every", type=int, default=TrainingConfig.real_eval_every,
                        help="Run real-data eval every N steps (0 = save_every).")
    parser.add_argument("--real-eval-frame-limit", type=int, default=TrainingConfig.real_eval_frame_limit)
    parser.add_argument("--real-eval-alignment-method", default=TrainingConfig.real_eval_alignment_method)
    parser.add_argument("--real-eval-baseline-hr", default=TrainingConfig.real_eval_baseline_hr,
                        help="Optional TGV highpass baseline .npy for side-by-side TensorBoard panels.")
    parser.add_argument("--real-eval-center-fraction", type=float, default=TrainingConfig.real_eval_center_fraction)
    parser.add_argument("--real-eval-zoom", type=float, default=TrainingConfig.real_eval_zoom)
    parser.add_argument("--real-eval-overlap", type=int, default=TrainingConfig.real_eval_overlap)
    return parser


def config_from_args(argv: list[str] | None = None) -> TrainingConfig:
    args = build_arg_parser().parse_args(argv)
    cfg = TrainingConfig(
        training_pool_dir=args.training_pool_dir,
        output_dir=args.output_dir,
        scale=args.scale,
        drizzle_scale=args.drizzle_scale,
        in_channels=args.in_channels,
        base_channels=args.base_channels,
        unet_depth=args.unet_depth,
        dilated_bottleneck=args.dilated_bottleneck,
        predict_log_variance=args.predict_log_variance,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        total_steps=args.total_steps,
        lr=args.lr,
        weight_decay=args.weight_decay,
        lf_loss_weight=args.lf_loss_weight,
        hf_loss_weight=args.hf_loss_weight,
        edge_loss_weight=args.edge_loss_weight,
        forward_loss_weight=args.forward_loss_weight,
        nll_loss_weight=args.nll_loss_weight,
        coverage_loss_gain=args.coverage_loss_gain,
        edge_coarse_weight=args.edge_coarse_weight,
        hf_detail_weight=args.hf_detail_weight,
        hf_detail_gain=args.hf_detail_gain,
        lr_warmup_steps=args.lr_warmup_steps,
        sigma_lf=args.sigma_lf,
        psf_sigma_lr_px=args.psf_sigma_lr_px,
        min_log_variance=args.min_log_variance,
        max_log_variance=args.max_log_variance,
        log_every=args.log_every,
        save_every=args.save_every,
        seed=args.seed,
        device=args.device,
        patches_per_scene=args.patches_per_scene,
        max_scene_cache=args.max_scene_cache,
        prefetch_factor=args.prefetch_factor,
        scenes_per_bucket=args.scenes_per_bucket,
        patches_per_fetch=args.patches_per_fetch,
        amp=args.amp,
        compile_model=args.compile_model,
        channels_last=args.channels_last,
        tb_log_dir=args.tb_log_dir,
        tb_image_every=args.tb_image_every,
        resume_from=args.resume_from,
        include_multiscale=args.include_multiscale,
        defer_1x_upsample=args.defer_1x_upsample,
        burst_augment=args.burst_augment,
        burst_keep_min=args.burst_keep_min,
        burst_keep_max=args.burst_keep_max,
        min_burst_frames=args.min_burst_frames,
        shift_noise_std_px=args.shift_noise_std_px,
        drizzle_kernel=args.drizzle_kernel,
        real_eval_enabled=args.real_eval_enabled,
        real_eval_every=args.real_eval_every,
        real_eval_frame_limit=args.real_eval_frame_limit,
        real_eval_alignment_method=args.real_eval_alignment_method,
        real_eval_baseline_hr=args.real_eval_baseline_hr,
        real_eval_center_fraction=args.real_eval_center_fraction,
        real_eval_zoom=args.real_eval_zoom,
        real_eval_overlap=args.real_eval_overlap,
    )
    cfg.validate()
    return cfg
