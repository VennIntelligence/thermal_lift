# EP07v2 UNet Thermal SR

Regression-style UNet for the EP07v2 4x thermal super-resolution POC. The model learns from compact TCForge synthetic scenes: 1x fused observation features go in, reconstructed 4x Celsius temperature patches come out.

This is an independent UV project under `algos/ep07_unet_sr/`; do not run it from the repository root virtualenv.

## 角色定位

**项目主线算法**，champion 模型（depb9v6 等）出自本项目。包含两代路径：

1. **Plain UNet 回归**（`unet_sr.train`，上文英文部分描述的 4x 路径，ACL-001–023 时代）——现为遗留路径。
2. **K-step unrolled physics solver**（`unet_sr.solver_train`，ACL-024 起的当前主线）——UNet prox + 硬数据一致性（DC）步的展开求解器，探测器 pitch 重标定（ACL-023，20 µm）后主线目标为 **2x** contour-level SR。

注：本 README 顶部的 "4x thermal super-resolution POC" 描述是 UNet 时代的口径；solver 主线自 ACL-023/024 起以 2x 池（如 `pool_2x_v3_5k`、v6/v7/v8 系列池）为默认输入。

## 目录构成

| 路径 | 职责 |
|---|---|
| `src/unet_sr/config.py` | `TrainingConfig` 数据类 + 全量 CLI 定义（`config_from_args`），`train` 与 `solver_train` 共用同一参数面 |
| `src/unet_sr/train.py` | plain UNet 训练入口（遗留路径）|
| `src/unet_sr/solver_train.py` | unrolled solver 训练入口（主线，ACL-024 起）|
| `src/unet_sr/unroll.py` | `UnrolledSolver`（prox UNet + autograd DC 步）|
| `src/unet_sr/model.py` | `ThermalSRUNet` 网络结构 |
| `src/unet_sr/forward_torch.py` | Gate-A 认证的 PyTorch 前向算子（shift + PSF + 下采样，向量化，ACL-033/036）|
| `src/unet_sr/losses.py` | `ContourSRLoss`（highpass/边界/梯度向量等结构项，ACL-027）|
| `src/unet_sr/dataset.py` | `ThermalSRDataset`（compact TCForge 场景池读取）+ `SceneInterleavedSampler` |
| `src/unet_sr/fusion.py` | 多帧 burst → 1x 观测特征融合 |
| `src/unet_sr/inference.py` | `infer_full_frame` / `infer_from_burst` 分块推理 |
| `src/unet_sr/real_eval.py` | 训练过程中对真实 248 clean 帧的周期性评测出图（ACL-029/038/050）|
| `src/unet_sr/synth_eval.py` | held-out 合成 GT 评测（ACL-027）|
| `src/unet_sr/metrics.py`、`mask_weights.py` | 指标与结构权重图 |
| `scripts/` | probe / 评测 / 绘图脚本（见下文简表）|
| `tests/` | 单元与 smoke 测试（含 Gate B/C）|
| `clean.sh` | 交互式删除 `outputs/` 下最近一轮训练产物 |

## Environment

```bash
cd algos/ep07_unet_sr
uv sync
```

`tcforge` is installed from `../../tcforge` as an editable path dependency. CPU smoke tests are supported. Formal training should use a CUDA PyTorch build when available; if CUDA is unavailable, pass `--device cpu`.

## Generate A Smoke Pool

Run this from the repository root:

```bash
uv run python scripts/generate_training_pool.py \
  --config configs/synthetic/training_pool_4x.json \
  --num-scenes 5 \
  --output-dir /tmp/smoke_test_pool \
  --workers 1
```

The pool stores compact scenes only:

```text
scene_0000/
├── hr_mask_4x.png
├── hr_edge_4x.png
├── obs_features_1x.npz
├── shifts.npy
└── metadata.json
```

It does not store the raw 248-frame LR burst or the HR temperature field. `ThermalSRDataset` reads `obs_features_1x.npz` and reconstructs the HR target from `hr_mask_4x.png` plus `metadata.json` fields such as `T_bg_c`, `delta_T_c`, `low_freq_amplitude_c`, `low_freq_sigma_px`, and `low_freq_seed`.

## Training

```bash
# Single-GPU AMP training
cd algos/ep07_unet_sr
uv run python -m unet_sr.train \
  --training-pool-dir /path/to/pool \
  --output-dir outputs/run1 \
  --total-steps 50000 \
  --device cuda \
  --amp
```

```bash
# Two-GPU DDP + AMP training
cd algos/ep07_unet_sr
uv run torchrun --nproc_per_node=2 -m unet_sr.train \
  --training-pool-dir /path/to/pool \
  --output-dir outputs/run1 \
  --total-steps 50000 \
  --device cuda \
  --amp
```

```bash
# CPU fallback. AMP and DDP are disabled on CPU.
cd algos/ep07_unet_sr
uv run python -m unet_sr.train \
  --training-pool-dir /tmp/smoke_test_pool \
  --output-dir outputs/smoke \
  --total-steps 10 \
  --batch-size 2 \
  --patch-size-hr 64 \
  --num-workers 0 \
  --device cpu
```

Expected smoke behavior: the run prints finite `total`, `mse`, and `edge` losses and writes `outputs/smoke/model_final.pt`.

## Solver 训练（主线路径）

unrolled solver 与 UNet 共用 `config.py` 的 CLI；solver 专属旋钮以 `--solver-*` 与 `--unroll-steps` 为前缀（完整清单见 `uv run python -m unet_sr.solver_train --help`）。典型命令（源自 `solver_train.py` 模块 docstring）：

```bash
cd algos/ep07_unet_sr
uv run python -m unet_sr.solver_train \
  --training-pool-dir ../../data/synthetic/pool_2x_v3_5k \
  --input-mode hybrid_drizzle2x --scale 2 --unroll-steps 4 \
  --total-steps 20000 --batch-size 4 --patch-size-hr 256 \
  --output-dir outputs/solver_v1
```

要点：

- solver 路径固定 fp32（DC 步的 double-backward 在 AMP 下不稳定），`--amp` 只对 UNet 路径生效。
- 常用 solver 旋钮：`--solver-dc-weight`、`--solver-prior-anneal-steps`（先 DC 后 prior 的退火，ACL-025/026）、`--solver-share-weights`、`--solver-prox-no-se` / `--solver-prox-norm`（架构消融，ACL-040/041）、`--solver-phasebin-ontf` + `--phase-bin-channels`（on-the-fly phase-bin drizzle，ACL-043）、`--solver-dc-*-jitter-*`（算子误差鲁棒性，ACL-060）。
- `--compile` 只编译 prox 子网（ACL-031）；支持完整 resume 与 TensorBoard 续写（ACL-034）。
- 历史各版本（v1–v10、depb9 系列）的确切配方见 `research_log/algorithm_changelog.md` 对应 ACL 条目及 `scripts/run_*.md` runbook。

## Inference

```python
import torch
from unet_sr.inference import infer_full_frame, infer_from_burst
from unet_sr.model import ThermalSRUNet

model = ThermalSRUNet(scale=4)
checkpoint = torch.load("outputs/smoke/model_final.pt", map_location="cpu")
model.load_state_dict(checkpoint["model_state_dict"])

hr = infer_full_frame(model, obs_features, scale=4, patch_size_hr=256, overlap=32, device="cpu")
hr_from_burst = infer_from_burst(model, lr_burst, shifts, scale=4, device="cpu")
```

`infer_full_frame` tiles in LR coordinates and blends predicted HR patches in HR coordinates. `infer_from_burst` first calls `tcforge.fusion.fuse_burst_to_features()`.

## Tests

```bash
cd algos/ep07_unet_sr
uv run pytest -q
```

## Scripts 简表

`scripts/` 下均为分析 / 评测 / 绘图辅助脚本（各自带 argparse，用 `--help` 查看参数），不承担训练：

| 脚本 | 用途 |
|---|---|
| `solver_regression_suite.py` | checkpoint 回归评测套件（可对已保存输出打分，无需 checkpoint）|
| `run_stage2b_synth_benchmark.py` | Stage 2b 合成基准：经典（TGV/MAP-TV）vs 神经 solver 同池对比（ACL-054/060）|
| `probe_dot_retention.py` / `probe_dot_retention_gt.py` / `eval_arms_dot_probe.py` | 小暗点缺陷保真探针（真实域 P0 / GT 池 / 批量多臂，ACL-063 起）|
| `analyze_dc_residual_confidence.py` | DC 残差能否暴露点抹除位置的审计（ACL-075）|
| `bench_forward_fast.py` | 向量化前向算子 vs 认证循环的等价性验证与基准（ACL-033）|
| `extract_checkpoint_metrics.py` / `plot_checkpoint_selection.py` / `plot_drift_trajectories_paper.py` / `render_checkpoint_evolution_drop.py` | TensorBoard 指标提取、checkpoint 选择轨迹与论文图 |
| `compare_regression_distributions.py` | 回归套件指标分布的好/坏 case 对比 |
| `profile_training_bottleneck.py` | 训练瓶颈 profile（数据加载 vs GPU 计算）|
| `v9_review/` | v9 代复盘工具（fusion baseline、pareto sweep、TB 提取、对比面板）|
| `run_training.md` / `run_v9.md` / `run_v10.md` / `run_v10_highlam.md` | 历史训练 runbook（文档，非脚本）|

## 关键输出

- 训练产物写到本目录 `outputs/<run名>/`（不入 Git）：`model_step*.pt` / `model_final.pt` checkpoint、`tb_logs/` TensorBoard 日志、real-eval / synth-eval PNG 面板。
- probe / 评测脚本产物按各脚本 `--output-dir` 参数写出（多为项目根 `tmp/` 或本目录 `outputs/`）。
- 训练池由项目根 `scripts/generate_training_pool.py` 生成，存放在 `data/synthetic/`（不入 Git）。
- 机器间交付走 `remote_inbox/`（严禁 git add）。

## 相关文档

- 远程训练指挥与任务包：`docs/REMOTE_ORDERS.md`
- 算法变更日志：`research_log/algorithm_changelog.md` —— EP07 是 ACL 主体：ACL-001–023（UNet 时代）、ACL-024 起（solver 主线）、champion 判决线见 ACL-064/070/074/076/077/079/080
- Episode 记录：`research_log/episodes/ep07_solver_boundary_artifact/`、`ep07_thermal_chip_phantom/`、`ep07_solver_v8_k4_fullhalo_eval_archive/`、`ep07_unet_sr_task1_audit.md`
- 正式报告：`paper/reports/ep07_thermal_chip_phantom/`、`paper/reports/ep07_v9_attribution/`

## Git And Data Rules

Training pools, checkpoints, logs, `outputs/`, `.venv/`, and generated data must not be committed. Source code and tests are the reproducible part.

## Current Limits

This is synthetic-only pretraining. Applying it to the real 248-frame clean SR set still requires alignment quality gates and feature validation. A visually sharper 4x output is not evidence for 5 um temperature metrology or true optical resolution recovery.
