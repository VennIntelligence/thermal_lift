# EP12 4x SR

Hybrid drizzle-informed 4x thermal restoration model.

## 角色定位

**4x SR 主线（学习型）**。在 EP07 2x UNet 之上探索 4x 输出网格：Dataset 按需从
`lr_burst.npy + shifts.npy` 计算 2x drizzle 特征，UNet + PixelShuffle 2x 输出 4x
温度场。项目叙事中 4x 只是「表现/正则化网格」，采纳与否受 EP15 M4 MAP-TV 锚点
与 EP07 2x x2up 的三组门槛约束（见下方 Adoption boundary）。

## 目录构成

| 路径 | 职责 |
|------|------|
| `src/sr4x/config.py` | `TrainingConfig` dataclass + 全部训练 CLI 参数定义 |
| `src/sr4x/dataset.py` | 训练池 Dataset（按需 2x drizzle、burst augmentation、worker-scene 缓存） |
| `src/sr4x/model.py` | UNet + PixelShuffle 2x 架构 |
| `src/sr4x/losses.py` | LF/HF/edge/forward/NLL/HF-detail 多分量 loss |
| `src/sr4x/train.py` | 训练主入口（`python -m sr4x.train`） |
| `src/sr4x/inference.py` | tiled 推理（被 real_eval 与 ep12_4x_benchmark 复用） |
| `src/sr4x/real_eval.py` | 每次存 checkpoint 时对 248 帧真实数据自动评估并写 TensorBoard |
| `src/sr4x/evaluate.py` | 评估辅助函数 |
| `scripts/evaluate_split_half.py` | 训练池 split-half drizzle 一致性评估 |
| `run_training.md` | **中文训练指南**（Hybrid Progressive 方案、参数表、TensorBoard 说明） |
| `scripts/run_training.md` | **英文 guarded-baseline 配方**（含 hf-detail/edge-coarse/warmup 的具体命令）；与根目录版内容不同，两份都保留 |
| `tests/` | config/dataset/evaluate/model+losses/train-smoke 测试 |

## 环境安装

```bash
cd algos/ep12_4x_sr
uv sync
```

`thermal-core` 与 `tcforge` 已在 `[tool.uv.sources]` 中声明为 editable 路径依赖，
`uv sync` 会自动装入（`run_training.md` 中的手动 `uv pip install -e ../../core
../../tcforge` 步骤是历史写法，现已非必需）。主要依赖：torch>=2.2、tensorboard、
numpy、scipy、pandas、tqdm、pytest。

## 运行方法

训练（参数已核实自 `src/sr4x/config.py`，仅 `--training-pool-dir` 必填）：

```bash
cd algos/ep12_4x_sr
CUDA_VISIBLE_DEVICES=0 uv run python -m sr4x.train \
    --training-pool-dir ../../data/synthetic/training_pool_4x_aa_2000 \
    --output-dir outputs/ep12_hybrid_v2_guarded \
    --scale 4 --drizzle-scale 2 \
    --burst-augment --compile --amp \
    --batch-size 4 --num-workers 8 \
    --total-steps 80000 --save-every 2000
```

从 checkpoint 恢复：追加 `--resume outputs/<run>/checkpoint_step_010000.pt`。
完整参数表（loss 权重、worker 缓存、real-eval 选项）见 `run_training.md` 与
`scripts/run_training.md`。

训练池 split-half 一致性评估：

```bash
uv run python scripts/evaluate_split_half.py \
  --pool-dir ../../data/synthetic/training_pool_4x_aa_2000 \
  --scale 4 --kernel bilinear
```

测试（冒烟）：

```bash
uv run pytest
```

## 关键输出

- 训练产物写到 `--output-dir`（惯例 `outputs/<run_name>/`，Git 忽略）：`checkpoint_step_XXXXXX.pt`、`tb_logs/`（TensorBoard：loss 分量 + `eval_real/` 真实数据评估图与 artifact score）
- `evaluate_split_half.py` 默认只把聚合结果打印到 stdout；传 `--output-json` / `--output-csv` 才写文件
- 真实数据 4x 对照评测产物见姊妹项目 `algos/ep12_4x_benchmark/` → `output/ep12_4x_benchmark/`

## 相关文档

- Episode 记录：`research_log/episodes/ep12_4x_benchmark/README.md`（评测侧）；信息上限门槛来自 `research_log/episodes/ep15_info_limit/README.md`
- 算法变更日志：`research_log/algorithm_changelog.md` — **ACL-013**（4x v8 AA 训练池入口统一 + soft mask 接入）、**ACL-014**（修复 drizzle/coverage 错位并默认启用 burst augmentation）
- 训练池生成：根目录 `scripts/generate_training_pool.py` + `configs/synthetic/training_pool_4x.json`（详见 `tcforge/README.md`）

## Current route

```text
248 LR frames + shifts
  -> 2x drizzle features (computed by Dataset from lr_burst.npy)
  -> concat with 1x fused features upsampled to 2x
  -> UNet on 2x grid
  -> PixelShuffle 2x
  -> 4x temperature field
```

Training expects v8 AA compact scene directories from `scripts/generate_training_pool.py`:

- `hr_mask_4x.png`: soft coverage mask, loaded as `[0, 1]`
- `hr_edge_4x.png`
- `obs_features_1x.npz`: 5 LR channels
- `lr_burst.npy`: 248 LR frames, required for on-demand 2x drizzle
- `shifts.npy`
- `metadata.json`

`obs_features_4x.npz` is not part of the current training contract.
Training enables burst augmentation by default so each epoch rebuilds drizzle
features from a perturbed frame subset. Use `--no-burst-augment` only for
legacy pools that intentionally rely on fixed precomputed drizzle features.

Adoption boundary: 4x output is a presentation/regularization grid, not
evidence for new 10-14 um information. A 4x checkpoint is only useful if it
beats the M4 MAP-TV anchor on FRC/zigzag gates and is not worse than EP07 2x
x2up on artifact score and contour quality.

Smoke command:

```bash
cd algos/ep12_4x_sr
uv run pytest
```
