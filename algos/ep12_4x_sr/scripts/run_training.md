# EP12 Drizzle-Informed 4x SR Training

## Guarded baseline (with burst augmentation, LR warmup, multi-scale edge, HF detail)

```bash
cd algos/ep12_4x_sr

CUDA_VISIBLE_DEVICES=0 uv run python -m sr4x.train \
    --training-pool-dir ../../data/synthetic/training_pool_4x_aa_2000 \
    --output-dir outputs/ep12_hybrid_v2_guarded \
    --total-steps 80000 \
    --batch-size 40 \
    --patch-size 256 \
    --num-workers 4 \
    --save-every 2000 \
    --log-every 50 \
    --amp \
    --device cuda \
    --burst-augment \
    --scenes-per-bucket 80 \
    --patches-per-fetch 8 \
    --max-scene-cache 80 \
    --defer-1x-upsample \
    --lr-warmup-steps 500 \
    --hf-detail-weight 0.3 \
    --hf-detail-gain 4.0 \
    --edge-coarse-weight 0.25
```

## Key parameters

### Loss weights

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lf-loss-weight` | 1.0 | LF (Gaussian-blurred) L1 -- anchors smooth temperature field |
| `--hf-loss-weight` | 0.3 | HF (highpass) L1 weighted by coverage (high-cov = high weight) |
| `--hf-detail-weight` | 0.3 | HF L1 weighted by **inverse** coverage (low-cov = high weight) |
| `--hf-detail-gain` | 4.0 | Gain for inverse coverage weighting |
| `--edge-loss-weight` | 0.1 | Multi-scale Sobel edge L1 |
| `--edge-coarse-weight` | 0.25 | Weight on 2x-downsampled coarse-scale edge loss |
| `--forward-loss-weight` | 0.2 | PSF-aware forward consistency |
| `--nll-loss-weight` | 0.05 | Heteroscedastic NLL (uncertainty head) |
| `--coverage-loss-gain` | 4.0 | Gain for coverage weighting in HF and NLL losses |
| `--burst-augment` | on | Rebuild drizzle from perturbed `lr_burst.npy + shifts.npy` subsets each epoch |
| `--no-burst-augment` | off | Compatibility switch for legacy fixed-drizzle pools |

### Training schedule

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--lr-warmup-steps` | 500 | LR linear warmup 0 -> 2e-4, then cosine decay to 0 |
| `--lr` | 2e-4 | Peak learning rate |
| `--total-steps` | 80000 | Total training steps |

## What changed from the original EP12

1. **Hybrid 2x drizzle input** (`drizzle_scale=2`): the Dataset computes 2x drizzle
   from `lr_burst.npy + shifts.npy` on demand. The old same-grid `obs_features_4x.npz`
   precompute step is no longer part of the main training path.

2. **Burst augmentation default-on** (`burst_augment=True`): each epoch trains on
   a perturbed frame subset and shift-noise draw, reducing dependence on one fixed
   detector-axis coverage map.

3. **HF detail loss** (`hf_detail_weight=0.3`): inverse-coverage-weighted HF L1.
   Low-coverage pixels (where fine structures typically live) get **higher** weight,
   forcing the network to preserve detail where drizzle data is sparse.

4. **Multi-scale edge** (`edge_coarse_weight=0.25`): 2x-downsampled Sobel edge loss
   in addition to fine-scale. Thin-structure breaks are proportionally larger at
   coarser resolution, making the edge loss more sensitive to fine-line discontinuities.

5. **LR warmup** (`lr_warmup_steps=500`): prevents early gradient instability
   from conflicting loss terms.

## Resume from checkpoint

```bash
CUDA_VISIBLE_DEVICES=0 uv run python -m sr4x.train \
    --training-pool-dir ../../data/synthetic/training_pool_4x_aa_2000 \
    --output-dir outputs/ep12_hybrid_v2_guarded \
    --total-steps 80000 \
    --batch-size 40 \
    --patch-size 256 \
    --num-workers 4 \
    --save-every 2000 \
    --log-every 50 \
    --amp \
    --device cuda \
    --burst-augment \
    --scenes-per-bucket 80 \
    --patches-per-fetch 8 \
    --max-scene-cache 80 \
    --defer-1x-upsample \
    --lr-warmup-steps 500 \
    --hf-detail-weight 0.3 \
    --hf-detail-gain 4.0 \
    --edge-coarse-weight 0.25 \
    --resume outputs/ep12_hybrid_v2_guarded/checkpoint_step_020000.pt
```

## TensorBoard

```bash
tensorboard --logdir outputs/ep12_hybrid_v2_guarded/tb_logs
```

Key curves:
- `loss/total_ema50` -- smoothed total loss
- `loss/hf_detail` -- inverse-coverage HF detail loss (new)
- `loss/edge` -- now includes coarse-scale component
- `train/learning_rate` -- should show warmup ramp then cosine decay
- `loss/forward` -- forward consistency (physics constraint)
