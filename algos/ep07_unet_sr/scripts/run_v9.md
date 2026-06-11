# EP07 V9 训练启动指令（V9A / V9B / V9C / V9D）

> **Canonical 文档**：V9 双槽归因实验的复制粘贴启动命令以本文件为准。  
> 变更记录见 `research_log/algorithm_changelog.md` 的 ACL-016（V9A）、ACL-017（V9B）、ACL-019（V9C）。
> 通用训练说明（DataLoader 缓存、ContourSRLoss 原理）见同目录 `run_training.md`。

---

## 参数写法原则

**不必把所有 CLI 参数都展开。** 只写与 `TrainingConfig`（`src/unet_sr/config.py`）默认值不同的项；其余走代码默认或自动推导。

| 类别 | 处理方式 |
|------|----------|
| v8.1a conservative loss 壳 | **必须显式写出**（均不同于 CLI 默认） |
| V9 单因子改动 | **必须显式写出** |
| 与 v8.1a 相同、且等于 CLI 默认 | **省略**（如 `--hr-upsampler bilinear`、`--seed 42`、`--real-eval-*`） |
| `max_scene_cache` / `scenes_per_bucket` / `patches_per_fetch` | **省略**（`0` 时由 `batch_size` 自动算，128 → 16/16/16） |

### v8.1a conservative loss 壳（相对 CLI 默认的差异）

以下参数在 V9A/V9B/V9C/V9D 中**保持一致**，且均**不同于** `TrainingConfig` 默认值，故必须在命令里写出：

```text
--scale 2
--batch-size 128
--num-workers 8
--total-steps 60000
--save-every 5000
--log-every 100
--compile
--mse-loss-weight 0.3
--highpass-loss-weight 0.8
--structure-boost 2.0
--grad-vector-weight 0.15
--thin-boost 3.0
--gap-boost 2.0
```

以下 v8.1a 使用值与 CLI 默认相同，**可省略**：

`hr_upsampler=bilinear`, `hr_res_blocks=0`, `base_channels=64`, `patch_size_hr=256`,  
`lr=2e-4`, `weight_decay=1e-4`, `lr_warmup_steps=500`, `seed=42`,  
`edge_loss_weight=0.05`, `ssim_loss_weight=0.15`, `edge_coarse_weight=0.25`,  
`highpass_sigma=5.0`, `laplacian_weight=0.0`, `forward_model_weight=0.0`（V9A）,  
`amp=true`, `device=cuda`, 全部 `real_eval_*` 默认口径（248 帧 / contour_refined / center 1/3 / zoom3x / overlap 128）。

### V9 输入 / 锚定消融

| 实验 | 额外参数 |
|------|----------|
| **V9A** | `--input-mode hybrid_drizzle2x` + burst 训练池路径 |
| **V9B** | `--forward-model-weight 0.1 --forward-model-band highpass`（旧 `training_pool_2x_aa`） |
| **V9C** | `--input-mode hybrid_drizzle2x --forward-model-weight 0.1 --forward-model-band highpass` + burst 训练池路径 |
| **V9D** | `--forward-model-weight 0.1 --forward-model-band full`（旧 `training_pool_2x_aa`） |

> V9A 仍保持 `forward_model_weight=0` 作为纯输入消融；V9C 才启用 hybrid+forward。
> hybrid 下 `obs[:, 0:1]` 是上采 mean，不是合法 1x 观测；V9C 的 forward anchor
> 来自 dataset 单独返回的 `lr_obs`（原始 1x aligned_mean crop，偶数 2x origin，同步增广）。

---

## V9A 前置 ①：burst 训练池生成

旧 `training_pool_2x_aa` 无 `lr_burst.npy`；V9A 训练/推理必须统一用 scatter `drizzle_features`。

配置：`configs/synthetic/training_pool_2x_burst.json`（`save_lr_burst: true`，`compute_classical_sr: false`）。

```bash
# 项目根目录 — mini 池（smoke / 单元测试）
uv run python scripts/generate_training_pool.py \
  --config configs/synthetic/training_pool_2x_burst.json \
  --output-dir data/synthetic/training_pool_2x_aa_burst_mini8 \
  --pool-size 8 --workers 4

# 全量 1000 scenes（~152 GB lr_burst float16；生成前 df -h 确认余量 ≥ 250 GB）
uv run python scripts/generate_training_pool.py \
  --config configs/synthetic/training_pool_2x_burst.json \
  --output-dir data/synthetic/training_pool_2x_aa_burst \
  --pool-size 1000 --workers 14
```

## V9A 前置 ②：池侧预计算 drizzle 变体（ACL-018，必须）

DataLoader 内现场 drizzle（~2.7 s/scene/epoch）+ float32 burst 缓存（~305 MB/scene）
会导致首 batch 160 s+、主机 RAM 爆掉 worker 被 OOM kill。训练前必须为池预生成
`drizzle_variants_2x.npy`（每 scene K=4 个固定增广变体，float16，~30 MB/scene）：

```bash
# 项目根目录；2000 scenes × 4 变体，14 workers 约 25 min；磁盘 ~59 GB
# 可断点续跑：已完成的 scene 自动跳过
uv run python scripts/precompute_drizzle_variants.py \
  --pool-dir data/synthetic/training_pool_2x_aa_burst \
  --num-variants 4 --workers 14
```

变体定义：variant 0 = 全帧无噪声（canonical，与推理口径一致）；variant 1–3 =
随机抽 60–100% burst 帧 + shifts 加 σ=0.05 px 高斯噪声（与原 `_select_burst`
增广分布一致）。训练时每 epoch 按 (seed, epoch, scene) 确定性抽一个变体，
不再读取 `lr_burst.npy`。无变体文件时 dataset 自动 fallback 到现场 drizzle
（仅限 mini 池 smoke；全量训练禁止走 fallback）。

---

## V9A — hybrid 2x drizzle 输入（GPU 0）

hybrid 下 UNet 全程在 256×256（2x 网格）上运行，显存约为 v8.1a 的 ~4×；OOM 时先把 `--batch-size 128` 改为 `64`。

### smoke（mini 池，~200 步）

```bash
cd algos/ep07_unet_sr

CUDA_VISIBLE_DEVICES=0 uv run python -m unet_sr.train \
  --training-pool-dir ../../data/synthetic/training_pool_2x_aa_burst_mini8 \
  --output-dir outputs/ep07_v9a_smoke \
  --input-mode hybrid_drizzle2x \
  --scale 2 \
  --batch-size 128 \
  --num-workers 8 \
  --total-steps 200 \
  --save-every 100 \
  --log-every 50 \
  --compile \
  --mse-loss-weight 0.3 \
  --highpass-loss-weight 0.8 \
  --structure-boost 2.0 \
  --grad-vector-weight 0.15 \
  --thin-boost 3.0 \
  --gap-boost 2.0 \
  --real-eval-frame-limit 48
```

### 全量（60K）

```bash
cd algos/ep07_unet_sr

CUDA_VISIBLE_DEVICES=0 uv run python -m unet_sr.train \
  --training-pool-dir ../../data/synthetic/training_pool_2x_aa_burst \
  --output-dir outputs/ep07_v9a_hybrid_drizzle \
  --input-mode hybrid_drizzle2x \
  --scale 2 \
  --batch-size 128 \
  --num-workers 8 \
  --total-steps 60000 \
  --save-every 5000 \
  --log-every 100 \
  --compile \
  --mse-loss-weight 0.3 \
  --highpass-loss-weight 0.8 \
  --structure-boost 2.0 \
  --grad-vector-weight 0.15 \
  --thin-boost 3.0 \
  --gap-boost 2.0
```

---

## V9B — highpass-band forward consistency（GPU 1）

不依赖 burst 池；数据池沿用旧 `training_pool_2x_aa`。

### smoke（200 步）

```bash
cd algos/ep07_unet_sr

CUDA_VISIBLE_DEVICES=1 uv run python -m unet_sr.train \
  --training-pool-dir ../../data/synthetic/training_pool_2x_aa \
  --output-dir outputs/ep07_v9b_smoke \
  --scale 2 \
  --batch-size 128 \
  --num-workers 8 \
  --total-steps 200 \
  --save-every 100 \
  --log-every 50 \
  --compile \
  --mse-loss-weight 0.3 \
  --highpass-loss-weight 0.8 \
  --structure-boost 2.0 \
  --grad-vector-weight 0.15 \
  --thin-boost 3.0 \
  --gap-boost 2.0 \
  --forward-model-weight 0.1 \
  --forward-model-band highpass \
  --real-eval-frame-limit 48
```

### 全量（60K）

```bash
cd algos/ep07_unet_sr

CUDA_VISIBLE_DEVICES=1 uv run python -m unet_sr.train \
  --training-pool-dir ../../data/synthetic/training_pool_2x_aa \
  --output-dir outputs/ep07_v9b_fwd_consistency \
  --scale 2 \
  --batch-size 128 \
  --num-workers 8 \
  --total-steps 60000 \
  --save-every 5000 \
  --log-every 100 \
  --compile \
  --mse-loss-weight 0.3 \
  --highpass-loss-weight 0.8 \
  --structure-boost 2.0 \
  --grad-vector-weight 0.15 \
  --thin-boost 3.0 \
  --gap-boost 2.0 \
  --forward-model-weight 0.1 \
  --forward-model-band highpass
```

---

## V9C — hybrid 输入 + 合法 1x highpass forward anchor（GPU 1）

V9C 使用 V9A 的 hybrid drizzle 输入，但 forward consistency 不再读取 hybrid
`obs[:, 0:1]`。dataset 会从原始 1x `aligned_mean` 裁出 `lr_obs`
（shape = `1×128×128` for `patch_size_hr=256`），训练循环自动传给 loss；
loss 内部固定用 2x→1x 下采样倍率。

### smoke（mini 池，~200 步）

```bash
cd algos/ep07_unet_sr

CUDA_VISIBLE_DEVICES=1 uv run python -m unet_sr.train \
  --training-pool-dir ../../data/synthetic/training_pool_2x_aa_burst_mini8 \
  --output-dir outputs/ep07_v9c_smoke \
  --input-mode hybrid_drizzle2x \
  --scale 2 \
  --batch-size 128 \
  --num-workers 8 \
  --total-steps 200 \
  --save-every 100 \
  --log-every 50 \
  --compile \
  --mse-loss-weight 0.3 \
  --highpass-loss-weight 0.8 \
  --structure-boost 2.0 \
  --grad-vector-weight 0.15 \
  --thin-boost 3.0 \
  --gap-boost 2.0 \
  --forward-model-weight 0.1 \
  --forward-model-band highpass \
  --real-eval-frame-limit 48
```

### 全量（60K）

```bash
cd algos/ep07_unet_sr

CUDA_VISIBLE_DEVICES=1 uv run python -m unet_sr.train \
  --training-pool-dir ../../data/synthetic/training_pool_2x_aa_burst \
  --output-dir outputs/ep07_v9c_hybrid_legal_fwd \
  --input-mode hybrid_drizzle2x \
  --scale 2 \
  --batch-size 128 \
  --num-workers 8 \
  --total-steps 60000 \
  --save-every 5000 \
  --log-every 100 \
  --compile \
  --mse-loss-weight 0.3 \
  --highpass-loss-weight 0.8 \
  --structure-boost 2.0 \
  --grad-vector-weight 0.15 \
  --thin-boost 3.0 \
  --gap-boost 2.0 \
  --forward-model-weight 0.1 \
  --forward-model-band highpass
```

---

## V9D — full-band forward consistency（GPU 1）

V9D 保持旧 1x statistics 输入，只把 V9B 的 highpass-band forward anchor
改成 full-band anchor；用于判断 V9B 失败是否来自 band 限制。

### smoke（200 步）

```bash
cd algos/ep07_unet_sr

CUDA_VISIBLE_DEVICES=1 uv run python -m unet_sr.train \
  --training-pool-dir ../../data/synthetic/training_pool_2x_aa \
  --output-dir outputs/ep07_v9d_smoke \
  --scale 2 \
  --batch-size 128 \
  --num-workers 8 \
  --total-steps 200 \
  --save-every 100 \
  --log-every 50 \
  --compile \
  --mse-loss-weight 0.3 \
  --highpass-loss-weight 0.8 \
  --structure-boost 2.0 \
  --grad-vector-weight 0.15 \
  --thin-boost 3.0 \
  --gap-boost 2.0 \
  --forward-model-weight 0.1 \
  --forward-model-band full \
  --real-eval-frame-limit 48
```

### 全量（60K）

```bash
cd algos/ep07_unet_sr

CUDA_VISIBLE_DEVICES=1 uv run python -m unet_sr.train \
  --training-pool-dir ../../data/synthetic/training_pool_2x_aa \
  --output-dir outputs/ep07_v9d_fwd_full \
  --scale 2 \
  --batch-size 128 \
  --num-workers 8 \
  --total-steps 60000 \
  --save-every 5000 \
  --log-every 100 \
  --compile \
  --mse-loss-weight 0.3 \
  --highpass-loss-weight 0.8 \
  --structure-boost 2.0 \
  --grad-vector-weight 0.15 \
  --thin-boost 3.0 \
  --gap-boost 2.0 \
  --forward-model-weight 0.1 \
  --forward-model-band full
```

---

## 验收与 checkpoint 选择

| 实验 | 成功信号 | 对照（v8.1a @ 60K） |
|------|----------|---------------------|
| V9A | 中心最细 zigzag 线改善；锯齿减轻；`raw_control_corr` 上升 | corr 0.689 |
| V9B | 40K→60K `artifact_score` / `raw_control_corr` 漂移压平 | artifact 0.643，corr 0.689 |
| V9C | 在 V9A 输入收益上，合法 1x anchor 不引入 blur/漂移；若压平后期漂移则支持「hybrid 下锚可见」 | 对比 V9A/V9B |
| V9D | full-band anchor 是否比 V9B highpass 更能压漂移，或是否重现 ACL-005 低频冲突 | 对比 V9B |

最终 checkpoint **不默认取 60K**；在 40–60K 区间按 proxy + 视觉联合选优。

若 V9B/V9C/V9D 出现 ACL-005 式震荡，先把 `--forward-model-weight` 降到 `0.05` 重跑（只动这一旋钮）。
