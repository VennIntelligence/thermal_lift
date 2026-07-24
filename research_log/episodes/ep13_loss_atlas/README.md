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

## 最终结论

- 教学图集交付完成：TCForge 训练输入管线图（00–07）+ ContourSRLoss 逐项分解图（08–16），demo 与 `configs/synthetic/training_pool_2x.json` 默认口径对齐（248 帧/景、EP05 refined shifts、detector_realistic 噪声）。（出处: 本 README Status / Data source 节）
- 所记录的 loss 已换代：ContourSRLoss 的 thin/gap 线先验于 ACL-027 被"几何无关 boundary 权重 + 等温 flatness"重设计取代，评测同时改用 held-out 合成 GT；ACL-024 之后主线整体转入 unrolled solver。（出处: `research_log/algorithm_changelog.md` ACL-024/027）
- 本图集定位为 UNet 时代（项目阶段 I）的历史训练管线教程；其展示的 2x 合成池此后演化到 v6→v9 代（点保真修复与归因见 ACL-065→072），阅读时不应把图集中的配方当作最终生产配置。（出处: `docs/publication_figures/GALLERY.md` 阶段时间线；changelog 速览 #11）
