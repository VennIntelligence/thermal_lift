# EP13 — UNet SR Loss Atlas

## Goal

Visual tutorial for EP07 training: TCForge-based training input pipeline + ContourSRLoss term-by-term breakdown.

## Build

```bash
uv run python scripts/build_ep13_cache.py --force
uv run python scripts/build_notebook.py notebooks/ep13_loss_atlas --execute
```

## Data source

- Geometry/temperature/burst/fusion: `tcforge` (`build_scene_mask_with_metadata`, `reconstruct_hr_temperature`, `generate_lr_burst`, `fuse_burst_to_features`)
- Demo aligns with `configs/synthetic/training_pool_2x.json` defaults: **248 frames/scene**, EP05 refined shifts, `detector_realistic` noise, fixed demo drift (`scalar_offset`)
- Demo shows 16 LR frames for figures; obs_features fused from full 248-frame burst
- Loss patch: TCForge center crop with synthetic ringing on pred (teaching only)

## Configurable generator knobs

See `configs/synthetic/training_pool_2x.json` and `scripts/generate_training_pool.py`:

- `n_frames_per_scene` (default 248)
- `physics_ranges.rotation_deg_center` / `rotation_jitter_deg`
- `shift_profile` / `shift_jitter_std_px`
- `noise_model` / `drift_distribution`

## Status

- TCForge-integrated pipeline figures (00-07) + loss atlas figures (08-16)
