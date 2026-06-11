# EP07 UNet SR 训练指南

## 基础训练命令

```bash
cd algos/ep07_unet_sr

CUDA_VISIBLE_DEVICES=1 uv run python -m unet_sr.train \
    --training-pool-dir ../../data/synthetic/training_pool_2x \
    --output-dir outputs/ep07_run \
    --scale 2 \
    --total-steps 40000 \
    --batch-size 80 \
    --patch-size-hr 256 \
    --num-workers 4 \
    --save-every 2000 \
    --log-every 20 \
    --amp \
    --device cuda \
    --lr-warmup-steps 500
```

> **缓存参数已自动优化**：`scenes_per_bucket`、`patches_per_fetch`、`max_scene_cache`
> 均由 `batch_size` 自动计算，无需手动指定。详见下方"数据加载架构"一节。

## 数据加载架构

### Worker-Scene 亲和 + 自动调参

Sampler 内置 **worker-scene 亲和机制**：将 scene 按 `num_workers` 分区，
每个 DataLoader worker 只处理自己分区内的 scene，LRU 缓存命中率从 ~33% 提升至 ~86%+。

**自动计算的参数**（当 CLI 不指定时，默认值 = 0 触发自动计算）：

| 参数 | 自动值 | 含义 |
|------|--------|------|
| `patches_per_fetch` | `max(1, batch_size // 8)` | 每次从同一 scene 取多少 patch 再切换，保证每 batch 覆盖 ~8 个 scene |
| `scenes_per_bucket` | `16` | 每个采样桶的 scene 数，决定 worker 缓存大小 |
| `max_scene_cache` | `= scenes_per_bucket` | 每 worker 的 LRU 缓存容量，与桶大小匹配保证桶内 100% 命中 |

**内存估算**（2x pool，residual mode，~51.6 MB/scene）：

| `num_workers` | 单 worker 缓存 | 总缓存 RAM |
|---|---|---|
| 2 | 0.8 GB | 1.6 GB |
| 4 | 0.8 GB | 3.3 GB |
| 8 | 0.8 GB | 6.6 GB |

> 用户只需调整 `--num-workers` 来平衡内存与加载速度。如有特殊需求，
> 仍可通过 `--scenes-per-bucket 32 --max-scene-cache 32` 手动覆盖。

## 损失函数：ContourSRLoss（梯度加权 + 梯度向量匹配）

### 设计原理

使用 **target 温度场的 Sobel 梯度** 作为 highpass L1 的连续结构权重，配合
**梯度向量匹配 loss** 约束结构形态。

核心思想：现有 edge loss 只比较梯度**幅值** `|∇pred| - |∇target|`，当边缘膨胀
导致梯度方向偏转但幅值不变时会漏检。梯度向量 loss 比较完整 `(gx, gy)` 向量差，
是幅值对比的**严格超集**，一次性覆盖四种结构缺陷：

| 缺陷类型 | 梯度幅值 loss | 梯度向量 loss |
|----------|:---:|:---:|
| 膨胀（方向变、幅值不变） | ❌ 漏检 | ✅ 捕获 |
| 粘连（梯度消失） | ✅ | ✅ |
| 断连（梯度消失） | ✅ | ✅ |
| 幻觉（假梯度出现） | ✅ | ✅ |

### 权重图计算

```python
target_edges = sobel_edges(target)                    # Sobel 梯度幅度
edge_norm = target_edges / target_edges.amax(...)     # 归一化到 [0, 1]
weight_map = 1.0 + structure_boost * edge_norm        # 默认 = 1.0 + 4.0 * edge_norm
hp_loss = (hp_error * weight_map).mean()              # 直接 mean，不用 weight_sum
```

### 参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--highpass-loss-weight` | 1.0 | 主导项：高通滤波 L1（梯度加权） |
| `--mse-loss-weight` | 0.2 | MSE，锚定直流分量 + 保护 gap |
| `--ssim-loss-weight` | 0.15 | 结构相似性损失 |
| `--edge-loss-weight` | 0.05 | 多尺度 Sobel 边缘幅值损失 |
| `--grad-vector-weight` | 0.3 | 梯度向量 (gx,gy) 匹配损失（防止膨胀/粘连/断连） |
| `--structure-boost` | 4.0 | 边缘像素权重倍数（边缘=5×，背景=1×） |
| `--edge-coarse-weight` | 0.25 | 2x 降采样 Sobel 权重（细线连通性） |
| `--thin-boost` | 6.0 | ≤3 HR px 细结构在 highpass/grad_vector 中的温和加权；`1.0` 表示关闭 |
| `--gap-boost` | 4.0 | ≤3 HR px 窄缝背景在 MSE/highpass 中的温和加权；`1.0` 表示关闭 |
| `--forward-model-band` | `full` | 前向一致性频段：`full`（全频，默认，向后兼容）或 `highpass`（带限，V9B） |
| `--forward-model-band-sigma` | 5.0 | highpass 带的高斯 σ（LR px），与 `highpass_sigma` 约定一致 |
| `--input-mode` | `lr` | 输入模式：`lr`（5ch 1x obs，默认）或 `hybrid_drizzle2x`（8ch 2x，V9A） |

> **默认关闭、显式启用**: `--laplacian-weight`（默认 0）、`--forward-model-weight`（默认 0）。
> v7 纯 grad-vector 配置不启用；v8 复现实验沿用上一轮效果较满意的 hybrid 配置，
> 显式设为 `0.1 / 0.1`。V9B 在 v8.1a 基线上只加 `--forward-model-weight 0.1 --forward-model-band highpass`。

### Total Loss 公式

$$\mathcal{L}_{total} = 0.2 \cdot \mathrm{MSE} + 1.0 \cdot \mathrm{HP}_{grad} + 0.05 \cdot \mathrm{Edge}_{mag} + 0.15 \cdot \mathrm{SSIM} + 0.3 \cdot \mathrm{GradVec}$$

## 推荐训练命令（v7 梯度向量匹配）

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m unet_sr.train \
    --training-pool-dir ../../data/synthetic/training_pool_2x \
    --output-dir outputs/ep07_v7_gradvec \
    --scale 2 \
    --total-steps 40000 \
    --batch-size 96 \
    --patch-size-hr 256 \
    --num-workers 15 \
    --save-every 2000 \
    --log-every 50 \
    --amp \
    --device cuda \
    --lr-warmup-steps 500 \
    --grad-vector-weight 0.3
```

## 推荐训练命令（v8 AA 覆盖率训练池 + 细结构/窄缝加权）

沿用上一轮视觉效果较满意的 hybrid 主干
`grad_vector=0.3 + laplacian=0.1 + forward_model=0.1`，本轮只切换到
AA coverage 训练池，并追加温和的 thin/gap boost。`real_eval` 默认开启：
248 帧 clean 主 session、center zoom3x，与 EP11 口径一致。

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m unet_sr.train \
    --training-pool-dir ../../data/synthetic/training_pool_2x_aa \
    --output-dir outputs/ep07_v8_aa \
    --scale 2 \
    --total-steps 60000 \
    --batch-size 96 \
    --patch-size-hr 256 \
    --num-workers 15 \
    --save-every 2000 \
    --log-every 50 \
    --amp \
    --device cuda \
    --lr-warmup-steps 500 \
    --grad-vector-weight 0.3 \
    --laplacian-weight 0.1 \
    --forward-model-weight 0.1 \
    --forward-model-psf-sigma 0.5 \
    --thin-boost 6 \
    --gap-boost 4
```

## 推荐训练命令（V8_1A / V8_1B 并行归因实验）

V8_1A 和 V8_1B 使用同一套 conservative loss，仅改变 final HR head：

| 实验 | 目的 | 差异 |
|---|---|---|
| `V8_1A` | 判断 v8 亮边/膨胀是否主要来自 loss 过热 | 保留旧 `bilinear` head，只降低结构权重 |
| `V8_1B` | 判断 final upsampler/head 是否贡献 2x 相位网格 | `PixelShuffle + ICNR + 1` 个无归一化 HR residual block |

### V8_1A — loss cooldown，保留 bilinear head

```bash
cd algos/ep07_unet_sr

CUDA_VISIBLE_DEVICES=0 uv run python -m unet_sr.train \
    --training-pool-dir ../../data/synthetic/training_pool_2x_aa \
    --output-dir outputs/ep07_v8_1a_loss_cooldown \
    --scale 2 \
    --hr-upsampler bilinear \
    --hr-res-blocks 0 \
    --total-steps 60000 \
    --batch-size 96 \
    --patch-size-hr 256 \
    --num-workers 8 \
    --save-every 10000 \
    --log-every 100 \
    --amp \
    --compile \
    --device cuda \
    --lr-warmup-steps 500 \
    --mse-loss-weight 0.3 \
    --highpass-loss-weight 0.8 \
    --structure-boost 2.0 \
    --edge-loss-weight 0.05 \
    --ssim-loss-weight 0.15 \
    --grad-vector-weight 0.15 \
    --laplacian-weight 0.0 \
    --forward-model-weight 0.0 \
    --thin-boost 3.0 \
    --gap-boost 2.0
```

### V8_1B — PixelShuffle head，同一套 loss

```bash
cd algos/ep07_unet_sr

CUDA_VISIBLE_DEVICES=1 uv run python -m unet_sr.train \
    --training-pool-dir ../../data/synthetic/training_pool_2x_aa \
    --output-dir outputs/ep07_v8_1b_pixelshuffle \
    --scale 2 \
    --hr-upsampler pixelshuffle \
    --hr-res-blocks 1 \
    --total-steps 60000 \
    --batch-size 96 \
    --patch-size-hr 256 \
    --num-workers 8 \
    --save-every 10000 \
    --log-every 100 \
    --amp \
    --compile \
    --device cuda \
    --lr-warmup-steps 500 \
    --mse-loss-weight 0.3 \
    --highpass-loss-weight 0.8 \
    --structure-boost 2.0 \
    --edge-loss-weight 0.05 \
    --ssim-loss-weight 0.15 \
    --grad-vector-weight 0.15 \
    --laplacian-weight 0.0 \
    --forward-model-weight 0.0 \
    --thin-boost 3.0 \
    --gap-boost 2.0
```

## 推荐训练命令（V9A / V9B 双槽并行归因实验）

V9 相对 v8.1a 基线**每臂只改一个因子**（见 ACL-016 / ACL-017）：

| 实验 | 槽位 | 单因子改动 | 训练池 |
|---|---|---|---|
| `V9B` | GPU 1 | 加 highpass-band forward consistency | 旧 `training_pool_2x_aa` |
| `V9A` | GPU 0 | hybrid 2x drizzle 输入（8ch @ 2x 网格） | 新 `training_pool_2x_aa_burst`（含 `lr_burst`） |

**共享 v8.1a loss 壳**（两臂除下表差异外保持一致）：

```text
--hr-upsampler bilinear --hr-res-blocks 0
--mse-loss-weight 0.3 --highpass-loss-weight 0.8 --structure-boost 2.0
--grad-vector-weight 0.15 --laplacian-weight 0.0
--thin-boost 3.0 --gap-boost 2.0
--edge-loss-weight 0.05 --ssim-loss-weight 0.15 --edge-coarse-weight 0.25
--forward-model-weight 0.0
--batch-size 128 --lr 2e-4 --total-steps 60000 --seed 42 --amp --compile
```

> hybrid 模式下 UNet 全程在 256×256（2x 网格）上运行，显存约为 v8.1a 的 ~4×；
> OOM 时先把 `--batch-size` 降到 64。`input_mode=hybrid_drizzle2x` 与
> `forward_model_weight > 0` 互斥（obs 第 0 通道是上采 mean，不是合法 1x LR 观测）。

### V9B — highpass-band forward consistency（GPU 1，不依赖新池）

相对 v8.1a **唯一差异**：`--forward-model-weight 0.1 --forward-model-band highpass`。

```bash
cd algos/ep07_unet_sr

# smoke（200 步，48 帧 real_eval）
CUDA_VISIBLE_DEVICES=1 uv run python -m unet_sr.train \
    --training-pool-dir ../../data/synthetic/training_pool_2x_aa \
    --output-dir outputs/ep07_v9b_smoke \
    --scale 2 --total-steps 200 --save-every 100 \
    --real-eval-frame-limit 48 \
    --mse-loss-weight 0.3 --highpass-loss-weight 0.8 --structure-boost 2.0 \
    --grad-vector-weight 0.15 --laplacian-weight 0.0 \
    --thin-boost 3.0 --gap-boost 2.0 \
    --edge-loss-weight 0.05 --ssim-loss-weight 0.15 --edge-coarse-weight 0.25 \
    --forward-model-weight 0.1 --forward-model-psf-sigma 0.5 \
    --forward-model-band highpass --forward-model-band-sigma 5.0 \
    --amp --compile --device cuda

# 全量
CUDA_VISIBLE_DEVICES=1 uv run python -m unet_sr.train \
    --training-pool-dir ../../data/synthetic/training_pool_2x_aa \
    --output-dir outputs/ep07_v9b_fwd_consistency \
    --scale 2 --total-steps 60000 --batch-size 128 \
    --mse-loss-weight 0.3 --highpass-loss-weight 0.8 --structure-boost 2.0 \
    --grad-vector-weight 0.15 --laplacian-weight 0.0 \
    --thin-boost 3.0 --gap-boost 2.0 \
    --edge-loss-weight 0.05 --ssim-loss-weight 0.15 --edge-coarse-weight 0.25 \
    --forward-model-weight 0.1 --forward-model-psf-sigma 0.5 \
    --forward-model-band highpass --forward-model-band-sigma 5.0 \
    --amp --compile --device cuda
```

### V9A — 训练池重生成（CPU，与 V9B 并行）

旧 `training_pool_2x_aa` 无 `lr_burst.npy`；V9A 必须用 scatter `drizzle_features` 保证 train/infer parity。
配置：`configs/synthetic/training_pool_2x_burst.json`（`save_lr_burst: true`，`compute_classical_sr: false`）。

```bash
# 项目根目录 — mini 池（smoke / 单元测试，8 scenes）
uv run python scripts/generate_training_pool.py \
    --config configs/synthetic/training_pool_2x_burst.json \
    --output-dir data/synthetic/training_pool_2x_aa_burst_mini8 \
    --pool-size 8 --workers 4

# 全量 1000 scenes（~152 GB lr_burst float16；生成前确认 df -h 余量 ≥ 250 GB）
uv run python scripts/generate_training_pool.py \
    --config configs/synthetic/training_pool_2x_burst.json \
    --output-dir data/synthetic/training_pool_2x_aa_burst \
    --pool-size 1000 --workers 14
```

### V9A — hybrid drizzle 训练（GPU 0，等 burst 池就绪）

相对 v8.1a **唯一差异**：`--input-mode hybrid_drizzle2x` + burst 训练池；`forward_model_weight` 保持 0。

```bash
cd algos/ep07_unet_sr

# smoke（mini 池）
CUDA_VISIBLE_DEVICES=0 uv run python -m unet_sr.train \
    --training-pool-dir ../../data/synthetic/training_pool_2x_aa_burst_mini8 \
    --output-dir outputs/ep07_v9a_smoke \
    --input-mode hybrid_drizzle2x \
    --scale 2 --total-steps 200 --save-every 100 \
    --real-eval-frame-limit 48 \
    --mse-loss-weight 0.3 --highpass-loss-weight 0.8 --structure-boost 2.0 \
    --grad-vector-weight 0.15 --laplacian-weight 0.0 \
    --thin-boost 3.0 --gap-boost 2.0 \
    --edge-loss-weight 0.05 --ssim-loss-weight 0.15 --edge-coarse-weight 0.25 \
    --amp --compile --device cuda

# 全量
CUDA_VISIBLE_DEVICES=0 uv run python -m unet_sr.train \
    --training-pool-dir ../../data/synthetic/training_pool_2x_aa_burst \
    --output-dir outputs/ep07_v9a_hybrid_drizzle \
    --input-mode hybrid_drizzle2x \
    --scale 2 --total-steps 60000 --batch-size 128 \
    --mse-loss-weight 0.3 --highpass-loss-weight 0.8 --structure-boost 2.0 \
    --grad-vector-weight 0.15 --laplacian-weight 0.0 \
    --thin-boost 3.0 --gap-boost 2.0 \
    --edge-loss-weight 0.05 --ssim-loss-weight 0.15 --edge-coarse-weight 0.25 \
    --amp --compile --device cuda
```

**验收对照**（详见 ACL-016 / ACL-017）：V9A 盯中心最细线与锯齿；V9B 盯 40K→60K 的
`artifact_score` / `raw_control_corr` 漂移是否压平。最终 checkpoint 在 40–60K 区间按
proxy + 视觉联合选优，不默认取 60K。

## 从检查点恢复

```bash
CUDA_VISIBLE_DEVICES=1 uv run python -m unet_sr.train \
    --training-pool-dir ../../data/synthetic/training_pool_2x \
    --output-dir outputs/ep07_run \
    --scale 2 \
    --total-steps 40000 \
    --batch-size 80 \
    --patch-size-hr 256 \
    --num-workers 4 \
    --save-every 2000 \
    --log-every 20 \
    --amp \
    --device cuda \
    --lr-warmup-steps 500 \
    --resume outputs/ep07_run/checkpoint_step_010000.pt
```

## TensorBoard 监控

```bash
tensorboard --logdir outputs/ep07_v7_gradvec/tb_logs
```

**Checkpoint 推演（`--save-every 2000`）做什么、不做什么：**

| 行为 | 说明 |
|------|------|
| ✅ **会做** | 在 **248 帧 clean 主 session 真实数据** 上跑 tiled inference，生成与 EP11 相同的 **center 1/3 ROI + 3× 显示放大** 温度 sanity 图 |
| ✅ **会做** | 同 ROI 的 highpass 对比（UNet vs TGV baseline，若 baseline 存在） |
| ❌ **默认不做** | **不会**在 checkpoint 时推演 TCForge 合成训练集 patch（需显式 `--tb-image-every N`） |

重点关注曲线：
- `loss/total_ema50` — 平滑总损失，应稳步下降
- `loss/highpass` — 主导项，梯度加权后应比旧版更稳定
- `loss/grad_vector` — 梯度向量匹配，应持续下降；若与 highpass 同步下降说明无冲突
- `loss/mse` — 应持续下降，保证温度场整体质量
- `train/grad_norm` — 应比 v6 更稳定（v6 在 12~263 间波动，v7 预期显著收窄）
- `eval_real/temperature_center_zoom` — EP11 同款中心 ROI 3× 显示放大温度 sanity view
