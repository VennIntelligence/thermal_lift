# 4x 合成训练池生成指南

本文是 **v8 抗锯齿 TCForge 4x 训练池** 的唯一执行文档。Windows 和 Linux 都使用同一个新入口：

```bash
uv run python scripts/generate_training_pool.py
```

旧入口 `scripts/generate_thermal_chip_phantom.py`、旧 drizzle 预计算脚本和 `obs_features_4x.npz` same-grid 路线已下线。

## 前提

- 已安装 Git 和 uv。
- 磁盘：1000 scenes 仅 `lr_burst.npy` 约 190 GB；2000 scenes 至少按 380 GB 起算，另需预留 manifest、mask、1x features 和日志空间。
- 内存：4x + v8 SSAA 约 1.3-1.5 GB/worker。`--workers 16` 建议至少预留 32 GB 以上系统内存。

生成脚本通过 `sys.path` 加载 `tcforge/src`，不需要单独安装 tcforge venv。

默认真实位移 CSV 已纳入 Git：

- 配置：`configs/alignment/paths.json`
- 数据：`configs/alignment/contour_alignment_results.csv`

如需临时覆盖，可设置：

```powershell
$env:TCFORGE_REAL_SHIFT_CSV = "D:\path\to\custom.csv"
```

Linux/bash 等价：

```bash
export TCFORGE_REAL_SHIFT_CSV=/path/to/custom.csv
```

## Smoke 测试

先生成 2 个 scene，确认真实位移和 v8 AA 都接通：

```bash
uv run python scripts/generate_training_pool.py \
  --config configs/synthetic/training_pool_4x.json \
  --output-dir data/synthetic/training_pool_4x_aa_smoke \
  --num-scenes 2 \
  --workers 1
```

检查 `data/synthetic/training_pool_4x_aa_smoke/scene_0000/metadata.json`：

| 字段 | 期望 |
|---|---|
| `shift_metadata.fallback_used` | `false`，表示加载了真实 contour-refined 位移 |
| `geometry_metadata.mask_semantics` | `"coverage"`，表示 v8 抗锯齿 soft mask |
| `storage.hr_mask_semantics` | `"coverage"` |

每个 scene 的主文件应包括：

```text
hr_mask_4x.png
hr_edge_4x.png
obs_features_1x.npz
shifts.npy
metadata.json
lr_burst.npy
```

## 全量 2000 场景

本地 16 workers 生成 2000 个 scene：

```bash
uv run python scripts/generate_training_pool.py \
  --config configs/synthetic/training_pool_4x.json \
  --output-dir data/synthetic/training_pool_4x_aa_2000 \
  --pool-size 2000 \
  --workers 16
```

后台运行并记录日志：

```bash
mkdir -p output/logs
nohup uv run python scripts/generate_training_pool.py \
  --config configs/synthetic/training_pool_4x.json \
  --output-dir data/synthetic/training_pool_4x_aa_2000 \
  --pool-size 2000 \
  --workers 16 \
  > output/logs/training_pool_4x_aa_2000.log 2>&1 &
```

生成器支持断点续跑：已有 `metadata.json` 的 scene 会自动跳过；失败 scene 写入 `failed_scenes.log`。

## 与 EP12 Hybrid 训练的关系

当前 EP12 4x Hybrid 训练不再使用旧 `obs_features_4x.npz` same-grid drizzle 文件，也不要求先跑第二段 drizzle 预计算。

训练时 Dataset 会读取：

- `obs_features_1x.npz`
- `lr_burst.npy`
- `shifts.npy`
- `hr_mask_4x.png`
- `metadata.json`

然后在加载 scene 时从 `lr_burst.npy + shifts.npy` 按需计算 **2x drizzle features**，与 1x features 拼成 8 通道 2x 输入，再由 UNet + PixelShuffle 输出 4x 温度场。

训练池路径示例：

```bash
cd algos/ep12_4x_sr
CUDA_VISIBLE_DEVICES=0 uv run python -m sr4x.train \
  --training-pool-dir ../../data/synthetic/training_pool_4x_aa_2000 \
  --output-dir outputs/ep12_hybrid_v2 \
  --scale 4 \
  --drizzle-scale 2 \
  --device cuda:0 \
  --num-workers 8 \
  --batch-size 4 \
  --total-steps 80000 \
  --save-every 2000 \
  --amp
```

## 不再运行

以下旧路径不是当前 4x AA + Hybrid 主流程的一部分：

```text
scripts/generate_thermal_chip_phantom.py
scripts/smoke_test_thermal_chip_phantom.py
scripts/build_4x_features.py
scripts/precompute_drizzle_2x.py
```

如果后续确实需要为性能做离线 2x drizzle cache，应新增一个明确命名的新脚本，并同步更新本文件。
