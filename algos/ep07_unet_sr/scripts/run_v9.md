# EP07 V9 训练启动指令（V9A / V9B）

> **Canonical 文档**：V9 双槽归因实验的复制粘贴启动命令以本文件为准。  
> 变更记录见 `research_log/algorithm_changelog.md` 的 ACL-016（V9A）、ACL-017（V9B）。  
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

以下参数在 V9A/V9B 两臂中**保持一致**，且均**不同于** `TrainingConfig` 默认值，故必须在命令里写出：

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

### 单因子归因（两臂相对 v8.1a 的唯一差异）

| 实验 | 额外参数 |
|------|----------|
| **V9A** | `--input-mode hybrid_drizzle2x` + burst 训练池路径 |
| **V9B** | `--forward-model-weight 0.1 --forward-model-band highpass`（旧 `training_pool_2x_aa`） |

> V9A 禁止 `--forward-model-weight > 0`（hybrid 下 obs 第 0 通道是上采 mean，不是合法 1x LR 观测）。

---

## V9A 前置：burst 训练池生成

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

## 验收与 checkpoint 选择

| 实验 | 成功信号 | 对照（v8.1a @ 60K） |
|------|----------|---------------------|
| V9A | 中心最细 zigzag 线改善；锯齿减轻；`raw_control_corr` 上升 | corr 0.689 |
| V9B | 40K→60K `artifact_score` / `raw_control_corr` 漂移压平 | artifact 0.643，corr 0.689 |

最终 checkpoint **不默认取 60K**；在 40–60K 区间按 proxy + 视觉联合选优。

若 V9B 出现 ACL-005 式震荡，先把 `--forward-model-weight` 降到 `0.05` 重跑（只动这一旋钮）。
