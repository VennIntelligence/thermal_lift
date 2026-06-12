# EP07 V10 residual-over-observation 启动指令

> **状态**: 本文件只记录可复制命令。当前 Codex 任务不启动 GPU；GPU smoke/full 由用户手动执行。  
> **变更记录**: `research_log/algorithm_changelog.md` ACL-020。  
> **核心单因子**: `--residual-mode drizzle2x`，即 `pred = hybrid drizzle mean ch5 + model_delta`，并用 `--residual-penalty-weight` 显式惩罚 `mean(abs(delta))`。

## 参数原则

V10 沿用 V9A 的 hybrid 2x drizzle 输入和 v8.1a conservative loss 壳，但做三处改变：

| 项 | V9A | V10 |
|---|---|---|
| 输出参数化 | direct predict | `drizzle ch5 + delta` |
| 观测约束 | 无显式残差约束 | `residual_penalty_weight * mean(abs(delta))` |
| 训练步数 | 60K, 后期会侵蚀保真 | 25K 完整 cosine sweep，避免靠中途 checkpoint 选优 |

固定参数：

```text
--input-mode hybrid_drizzle2x
--residual-mode drizzle2x
--scale 2
--batch-size 128
--num-workers 8
--total-steps 25000
--save-every 2500
--log-every 100
--compile
--mse-loss-weight 0.3
--highpass-loss-weight 0.8
--structure-boost 2.0
--grad-vector-weight 0.15
--thin-boost 3.0
--gap-boost 2.0
```

V10 与 `--forward-model-weight > 0` 互斥；不要把 V9C 的 forward anchor 混进同一实验。

## Smoke

先用 mini burst 池做 200 step smoke。若只想检查配置和 CPU 代码路径，可把 `CUDA_VISIBLE_DEVICES=0` 改成 `CUDA_VISIBLE_DEVICES=` 并加 `--device cpu --compile` 去掉；正式 smoke 建议由用户手动在 GPU 0 启动。

```bash
cd algos/ep07_unet_sr

CUDA_VISIBLE_DEVICES=0 uv run python -m unet_sr.train \
  --training-pool-dir ../../data/synthetic/training_pool_2x_aa_burst_mini8 \
  --output-dir outputs/ep07_v10_smoke \
  --input-mode hybrid_drizzle2x \
  --residual-mode drizzle2x \
  --residual-penalty-weight 0.05 \
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

Smoke 后看 TensorBoard 中：

| Scalar | 期望 |
|---|---|
| `loss/residual_penalty` | 有限，量级可用于标定 lambda；TensorBoard 中该 scalar 是未乘 lambda 的 `mean(abs(delta))` |
| `residual/delta_mean` | 不应快速发散到大 DC offset |
| `residual/delta_std` | 应随结构学习增长但保持有限 |

lambda 初始目标：按 TensorBoard 标量 `loss/residual_penalty / loss/total` 标定三档，使训练初期比例约为 10% / 20% / 30%。注意当前实现中 `loss/residual_penalty` 是未乘 lambda 的 delta L1；真正加到 `loss/total` 的项是 `lambda * loss/residual_penalty`。

已完成的 lambda=0.05 smoke (`outputs/ep07_v10_smoke/tb_logs`) 最近 3 个记录：

| 项 | last3 |
|---|---|
| `loss/total` | 0.14084 / 0.16753 / 0.15847 |
| `loss/residual_penalty` | 0.03107 / 0.04057 / 0.04373 |
| ratio | 22.06% / 24.22% / 27.60% |

三点均值 ratio = 24.62%，高于 20% 中档目标；按 `lambda_target = 0.05 * target_ratio / 0.24624` 缩放，Full Sweep 采用 `0.0203 / 0.0406 / 0.0609`。

## Full Lambda Sweep

标定后三档使用 `0.0203 / 0.0406 / 0.0609`。这不是物理常数；若后续更长 smoke 的 loss 量级明显漂移，按上面的 10-30% 原则重新缩放。

### G1: GPU 0 先启动低/中 lambda

```bash
cd algos/ep07_unet_sr

CUDA_VISIBLE_DEVICES=0 uv run python -m unet_sr.train \
  --training-pool-dir ../../data/synthetic/training_pool_2x_aa_burst \
  --output-dir outputs/ep07_v10_residual_lam00203 \
  --input-mode hybrid_drizzle2x \
  --residual-mode drizzle2x \
  --residual-penalty-weight 0.0203 \
  --scale 2 \
  --batch-size 128 \
  --num-workers 8 \
  --total-steps 25000 \
  --save-every 2500 \
  --log-every 100 \
  --compile \
  --mse-loss-weight 0.3 \
  --highpass-loss-weight 0.8 \
  --structure-boost 2.0 \
  --grad-vector-weight 0.15 \
  --thin-boost 3.0 \
  --gap-boost 2.0
```

```bash
cd algos/ep07_unet_sr

CUDA_VISIBLE_DEVICES=0 uv run python -m unet_sr.train \
  --training-pool-dir ../../data/synthetic/training_pool_2x_aa_burst \
  --output-dir outputs/ep07_v10_residual_lam00406 \
  --input-mode hybrid_drizzle2x \
  --residual-mode drizzle2x \
  --residual-penalty-weight 0.0406 \
  --scale 2 \
  --batch-size 128 \
  --num-workers 8 \
  --total-steps 25000 \
  --save-every 2500 \
  --log-every 100 \
  --compile \
  --mse-loss-weight 0.3 \
  --highpass-loss-weight 0.8 \
  --structure-boost 2.0 \
  --grad-vector-weight 0.15 \
  --thin-boost 3.0 \
  --gap-boost 2.0
```

### G2: V9C 自然结束后 GPU 1 启动高 lambda

```bash
cd algos/ep07_unet_sr

CUDA_VISIBLE_DEVICES=1 uv run python -m unet_sr.train \
  --training-pool-dir ../../data/synthetic/training_pool_2x_aa_burst \
  --output-dir outputs/ep07_v10_residual_lam00609 \
  --input-mode hybrid_drizzle2x \
  --residual-mode drizzle2x \
  --residual-penalty-weight 0.0609 \
  --scale 2 \
  --batch-size 128 \
  --num-workers 8 \
  --total-steps 25000 \
  --save-every 2500 \
  --log-every 100 \
  --compile \
  --mse-loss-weight 0.3 \
  --highpass-loss-weight 0.8 \
  --structure-boost 2.0 \
  --grad-vector-weight 0.15 \
  --thin-boost 3.0 \
  --gap-boost 2.0
```

## Post-run Diagnostics

每臂完成后，用 tracked V9 review 管线追加评估：

```bash
cd algos/ep07_unet_sr

uv run python scripts/v9_review/extract_tb_metrics.py \
  --output-csv ../../output/ep07_v9_review/ep07_eval_real_metrics.csv

uv run python scripts/v9_review/run_pareto_sweep.py \
  --run-dir outputs/ep07_v10_residual_lam00406 \
  --checkpoints 2500 5000 7500 10000 12500 15000 17500 20000 22500 25000 \
  --device cpu
```

成功判据：某个 lambda 在 `(hp_corr_input, sharp_p95)` 平面上支配 TGV `(0.960, 0.96)`，且视觉上无新增格纹/振铃；否则 Claim 3 升级为负结果。
