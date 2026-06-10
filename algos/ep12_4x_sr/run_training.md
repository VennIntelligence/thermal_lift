# EP12 4x SR — 训练指南

## 方案概述

**Hybrid Progressive**: Dataset 从 `lr_burst.npy + shifts.npy` 按需计算 2x drizzle（无棋盘伪影），再由 UNet + PixelShuffle 2x 上采样到 4x 输出。当前主流程不需要预生成 `obs_features_4x.npz`。

```
248 LR frames (480×640) ──┬── 2x Drizzle ──→ 3ch @ 960×1280 (coverage 均匀 ✅)
                          │
                          └── 1x Fusion  ──→ 5ch @ 480×640 → 上采样到 960×1280
                                                    ↓
                                          Concat 8ch @ 960×1280
                                                    ↓
                                          UNet (depth=4, base=48)
                                                    ↓
                                          PixelShuffle 2x
                                                    ↓
                                          1ch 温度场 @ 1920×2560 (4x)
```

## 环境准备

```bash
cd algos/ep12_4x_sr
uv sync
uv pip install -e ../../core
uv pip install -e ../../tcforge
```

## 训练命令

### 标准训练（推荐）

```bash
cd /home/ujs/mycode/thermal_lift/algos/ep12_4x_sr

CUDA_VISIBLE_DEVICES=0 uv run python -m sr4x.train \
    --training-pool-dir /home/ujs/mycode/thermal_lift/data/synthetic/training_pool_4x_aa_2000 \
    --output-dir outputs/ep12_hybrid_v1 \
    --device cuda:0 \
    --scale 4 \
    --drizzle-scale 2 \
    --compile \
    --num-workers 8 \
    --batch-size 4 \
    --total-steps 80000 \
    --save-every 2000 \
    --amp
```

### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--scale` | 4 | 总 SR 倍率 (LR→HR) |
| `--drizzle-scale` | 2 | Drizzle 累积分辨率。**2 = 无棋盘**，4 = 旧版 same-grid |
| `--compile` | off | 启用 `torch.compile` 加速训练 |
| `--num-workers` | 4 | DataLoader worker 数量 |
| `--batch-size` | 4 | 训练 batch size |
| `--patch-size` | 256 | HR 输出 patch 尺寸 (4x grid) |
| `--total-steps` | 80000 | 总训练步数 |
| `--save-every` | 2000 | 每 N 步保存 checkpoint + 真实数据评估 |
| `--amp` | on | 混合精度训练 |
| `--defer-1x-upsample` | off | 将 1x→2x 上采样移到 GPU（减轻 CPU worker 负担） |

### Worker 缓存参数（自动调优）

以下参数控制 DataLoader 的 worker-scene affinity 和缓存策略，默认值经过自动调优，通常不需要手动设置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--scenes-per-bucket` | 16 | 每个 bucket 的 scene 数（控制 per-worker cache 大小） |
| `--max-scene-cache` | 0 (auto) | 每 worker LRU 缓存容量。0 = 自动设为 `scenes_per_bucket`，保证 ~100% 缓存命中 |
| `--patches-per-fetch` | 0 (auto) | 每 scene 连续取 patch 数。0 = 自动设为 `batch_size // 8`，控制 batch 多样性 |
| `--prefetch-factor` | 4 | DataLoader 每 worker 预取队列深度 |

### 真实数据评估参数

每次保存 checkpoint 时自动在 248 帧真实数据上推演 4x SR，结果写入 TensorBoard。

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--no-real-eval` | off | 禁用真实数据评估 |
| `--real-eval-every` | 0 | 评估频率 (0 = 跟随 save_every) |
| `--real-eval-frame-limit` | 248 | 使用的 clean 帧数 |
| `--real-eval-zoom` | 3.0 | 中心 ROI 放大倍率 |
| `--real-eval-overlap` | 64 | 推理 tiled overlap |

### 从 checkpoint 恢复训练

```bash
uv run python -m sr4x.train \
    --training-pool-dir /home/ujs/mycode/thermal_lift/data/synthetic/training_pool_4x_aa_2000 \
    --output-dir outputs/ep12_hybrid_v1 \
    --compile \
    --num-workers 8 \
    --resume outputs/ep12_hybrid_v1/checkpoint_step_010000.pt
```

## 监控训练

```bash
# 启动 TensorBoard
uv run tensorboard --logdir outputs/ep12_hybrid_v1/tb_logs --port 6006
```

TensorBoard 中可看到：

- **loss/**: 各 loss 分量 (total, lf_l1, hf_l1, edge, forward, nll, hf_detail)
- **eval_real/**: 真实数据评估
  - `temperature_center_zoom`: 温度场 inferno colormap（中心 3x 放大）
  - `highpass_center_zoom`: 高通滤波结构可视化
  - `artifact_score`: 伪影评分（越低越好）
  - `raw_control_corr`: 与 bicubic 上采样的 Pearson 相关

## 模型架构变化（相比旧版）

| 特性 | 旧版 EP12 | 新版 Hybrid |
|------|----------|-------------|
| Drizzle 分辨率 | 4x (棋盘 ❌) | 2x (均匀 ✅) |
| 模型输入 | 8ch @ 4x grid | 8ch @ 2x grid |
| 模型输出 | 1ch @ 4x (same-grid) | 1ch @ 4x (PixelShuffle 2x) |
| UNet scale | 1 | 2 |
| 上采样方式 | 无 | PixelShuffle (可学习) |
| Forward consistency | pred→4x pool→1x | pred→2x pool→1x (经 upsample 后) |
