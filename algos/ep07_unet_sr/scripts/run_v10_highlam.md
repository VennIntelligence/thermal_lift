# V10 高-λ 残差扫描 — GPU 后台交接提示词

> **用途**: 本文件**整体**就是交给「具备长等待能力的后台 GPU 系统/智能体」的提示词。
> 复制全文给该智能体即可。它需要自主完成：环境自检 → 4 臂训练 → 评估 → 回报判定。
> **变更记录**: `research_log/algorithm_changelog.md` ACL-020（含本轮动机：V10 评估 bug 修正 + λ 区间过弱）。
> **前置阅读（该智能体必读）**: `AGENTS.md`、ACL-016/017/019/020、`algos/ep07_unet_sr/scripts/run_v10.md`。

---

## 0. 你的任务（一句话）

在 **batch_size=128、25K 步、完整 cosine 退火** 下，对 V10 残差参数化 `pred = drizzle_mean(ch5) + delta`（惩罚 `λ·mean|delta|`）做**高-λ 扫描**，把「保真↑/锐度↓」折中曲线从 V9A 区域一直探到 drizzle base，判定是否存在某个 λ 的 checkpoint **严格支配经典 TGV** 工作点。

## 1. 背景：为什么是高-λ（务必理解，否则会重蹈覆辙）

- V10 之前用 **λ=0.02/0.05/0.15、bs=64** 跑过一轮。其 fine-window 评估曾因一个 bug（评估时未把 drizzle base 通道加回，只评了裸残差 delta）被误判为「Claim 4 灾难性失败、hp_corr≈0.46、lattice 超标 16×」。**该判定已作废。**
- 修正评估后（base 加回）三档 λ 的真实表现几乎重合：`hp_corr_input≈0.88–0.91`、`sharp_p95≈1.2–1.37`、`lattice≈0.017–0.024`——即落在 **V9A 后期同一折中区**：比 TGV 锐，但保真不及 TGV，**不支配 TGV**。
- 根因：`λ·mean|delta| / total_loss` 在 0.02–0.15 区间只有 **~1–4%**，残差惩罚几乎没起约束作用（`run_v10.md` 原标定公式把「未乘 λ 的 penalty/total≈24.6%」误当成损失占比）。**要真正把输出拉回 drizzle base 附近，需要大得多的 λ。**
- 早期训练标量经验值：`loss/residual_penalty`(未乘 λ 的 `mean|delta|`)≈0.04，`loss/total`≈0.15。则惩罚损失占比 ≈ `λ·0.04/0.15`：

  | λ | 早期惩罚占比（约） |
  |---|---|
  | 0.4 | ~11% |
  | 0.8 | ~21% |
  | 1.6 | ~43% |
  | 3.2 | ~85% |

## 2. 科学问题与成功判据

随 λ 增大，工作点应从 V9A 区 `(hp_in≈0.88, sharp≈1.3)` 沿曲线移向 drizzle base `(1.000, 0.503)`。问题：**这条曲线是否穿过 TGV 的右上方（同时更保真且更锐）？**

中心细线窗口指标（`hp_corr_input`=保真↑，`sharp_p95`=锐度↑，`lattice_score`=格纹↓）参照点：

| 对象 | hp_corr_input | sharp_p95 | lattice | 备注 |
|---|---|---|---|---|
| drizzle 输入（观测域上限） | 1.000 | 0.503 | 0.0015 | 软但不幻觉 |
| **EP10 TGV（要被支配的目标）** | **0.960** | **0.959** | **0.0169** | 经典参照，非 GT |
| 零训练 fusion（TGV+0.1·V9A60δ） | 0.963 | 0.970 | 0.0134 | 已知能微微支配 TGV 的后处理前沿 |
| 旧 V10 三档 @25K（修正后） | 0.88–0.91 | 1.2–1.37 | 0.017–0.024 | 本轮要超越的起点 |

**成功（Claim 4 正结果）**：某 checkpoint 同时满足
`hp_corr_input ≥ 0.960` **且** `sharp_p95 ≥ 0.960` **且** `lattice_score ≤ 0.0169`，**且** 视觉门控通过（中心细线窗口无新增格纹/振铃）。
**失败（Claim 4 干净反证）**：曲线全程位于 TGV 左上或右下，无点同时达标 → 升级为「即使显式残差控制也无法越过经典前沿」，并报告最接近的点与曲线形状。

## 3. 硬约束（违反即作废，必须逐条遵守）

1. **batch_size=128**（不是 64）——消除上一轮 bs 混杂。
2. **25K 步 + `--save-every 2500`**，完整 cosine（不靠中途 early-stop 选点）。
3. **单因子**：`--residual-mode drizzle2x --input-mode hybrid_drizzle2x --scale 2`，`--forward-model-weight 0`（默认即 0），**不要**加旧 `--residual`。
4. **评估必须用已修复的 harness**：`scripts/v9_review/common.py` 须把 `residual_channel=5` 透传（2026-06-13 已修复）。**自检**：评估后任一 V10 缓存 `output/ep07_v9_review/.../*.npy` 的均值应 ≈ **23°C（室温温度图）**；若 ≈ **0**，说明 base 没加回、bug 复活，立即 `git pull`/核对 `common.py` 再重评。`real_eval.py` 漂移路径本就正确。

## 4. 环境自检（开跑前）

```bash
cd <REPO>/algos/ep07_unet_sr
uv sync                       # 复原 venv
uv pip install -e ../../core  # 若 core 未装
python -c "import tcforge, unet_sr"   # 应无报错
nvidia-smi                    # 确认可见 GPU
```

训练池与 drizzle 变体（V10 hybrid 输入依赖，ACL-018）：

```bash
ls ../../data/synthetic/training_pool_2x_aa_burst/drizzle_variants_meta.json
ls ../../data/synthetic/training_pool_2x_aa_burst/scene_0000/drizzle_variants_2x.npy
# 若以上缺失（如换了新机器），从仓库根重新预计算（~25 min, +59GB）：
#   cd <REPO> && uv run python scripts/precompute_drizzle_variants.py \
#     --pool-dir data/synthetic/training_pool_2x_aa_burst --num-variants 4 --workers 14
```

预算：单臂 25K ≈ **3.5–4.25 h**（历史值）。4 臂串行单 GPU ≈ 16 h；多 GPU 可并行。

## 5. （可选但推荐）λ 标定 smoke

在**真实训练池**上跑 300 步，读 TB 确认占比，必要时按损失量级重缩放 λ：

```bash
cd <REPO>/algos/ep07_unet_sr
CUDA_VISIBLE_DEVICES=0 uv run python -m unet_sr.train \
  --training-pool-dir ../../data/synthetic/training_pool_2x_aa_burst \
  --output-dir outputs/ep07_v10_hl_calib --input-mode hybrid_drizzle2x \
  --residual-mode drizzle2x --residual-penalty-weight 0.8 --scale 2 \
  --batch-size 128 --num-workers 8 --total-steps 300 --save-every 300 \
  --log-every 50 --compile --mse-loss-weight 0.3 --highpass-loss-weight 0.8 \
  --structure-boost 2.0 --grad-vector-weight 0.15 --thin-boost 3.0 --gap-boost 2.0 \
  --real-eval-frame-limit 48
```

看 TB（`outputs/ep07_v10_hl_calib/tb_logs`）：取 `loss/residual_penalty`(未乘 λ) 与 `loss/total` 末几步均值，
令早期占比 `λ·penalty/total`。**目标四档分别落在 ~10% / 20% / 40% / 80%**。
若该池 `penalty/total` 明显偏离 0.04/0.15，则按 `λ = 目标占比 · (total/penalty)` 重算四档（保持四档覆盖 ~10%→~85%）。

## 6. Phase 1 — 四臂训练（核心，4 × 25K）

默认四档 **λ ∈ {0.4, 0.8, 1.6, 3.2}**（标定后如需则替换）。逐臂模板（`<LAM>`/`<TAG>` 成对替换：0.4→040 / 0.8→080 / 1.6→160 / 3.2→320；`<GPU>` 选可用卡）：

```bash
cd <REPO>/algos/ep07_unet_sr
CUDA_VISIBLE_DEVICES=<GPU> uv run python -m unet_sr.train \
  --training-pool-dir ../../data/synthetic/training_pool_2x_aa_burst \
  --output-dir outputs/ep07_v10_resid_hl_lam<TAG> \
  --input-mode hybrid_drizzle2x \
  --residual-mode drizzle2x \
  --residual-penalty-weight <LAM> \
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

训练中每 `save_every` 会自动记录 `eval_real/*` 漂移指标到 TB（这条路径残差处理正确）。
**监控**：`residual/delta_std` 应随 λ 增大而下降；`loss/total` 不应发散/出现 NaN；如某臂 NaN，降 `--highpass-loss-weight` 到 0.6 重启该臂并记录。

## 7. 每臂完成后的评估（CPU，分钟级）

```bash
cd <REPO>/algos/ep07_unet_sr

# 7.1 漂移曲线（real_eval，路径已正确）
uv run python scripts/v9_review/extract_tb_metrics.py \
  --output-csv ../../output/ep07_v9_review/ep07_eval_real_metrics.csv

# 7.2 fine-window Pareto（务必用修复后的 common.py；<TAG> 同上）
CUDA_VISIBLE_DEVICES= uv run python scripts/v9_review/run_pareto_sweep.py \
  --device cpu --force \
  --output-dir output/ep07_v9_review/v10_highlam \
  --checkpoint "v10hl_lam<TAG>_5k=ep07_v10_resid_hl_lam<TAG>:5000:hybrid_drizzle2x" \
  --checkpoint "v10hl_lam<TAG>_10k=ep07_v10_resid_hl_lam<TAG>:10000:hybrid_drizzle2x" \
  --checkpoint "v10hl_lam<TAG>_15k=ep07_v10_resid_hl_lam<TAG>:15000:hybrid_drizzle2x" \
  --checkpoint "v10hl_lam<TAG>_20k=ep07_v10_resid_hl_lam<TAG>:20000:hybrid_drizzle2x" \
  --checkpoint "v10hl_lam<TAG>_25k=ep07_v10_resid_hl_lam<TAG>:25000:hybrid_drizzle2x"
```

> ⚠️ `--output-dir` 用**相对仓库根的路径**（如 `output/ep07_v9_review/v10_highlam`），脚本会拼到 `PROJECT_ROOT`；**不要**写 `../../output/...`（会跳出仓库）。
> **自检**：评估完成后确认 `output/ep07_v9_review/cache/v10hl_*_temperature.npy` 均值 ≈ 23（见硬约束 4）。

## 8. 判定逻辑

1. 汇总四臂 `v9a_pareto_metrics.csv`，连同 TGV/drizzle/fusion 参照点画到一张 (hp_corr_input, sharp_p95) 平面。
2. 是否有 checkpoint 落在 TGV 右上且 `lattice ≤ 0.0169`？
   - **是** → 取该点做视觉门控（中心细线窗口，无新增格纹/振铃）→ 通过则 **Claim 4 正结果**。
   - **否** → 报告最接近 TGV 的点、曲线随 λ 的走向（是否朝 drizzle 单调回软）→ **Claim 4 干净反证**。

## 9. Phase 2 — 可选精化（用剩余预算，总预算 4–8 × 25K）

- 若某 λ 最接近支配 TGV，在其两侧二分加 1–2 档（25K）。
- 给最佳臂加第二个 seed（25K）验证稳定性。
- 仅当 Phase 1 出现「擦边支配」才值得；否则把预算留给统一 harness T1/T2 重跑（另见 `docs/paper/00_status_and_plan.md` 状态板）。

## 10. 回报模板（训练完成后给主线）

```
## V10 高-λ sweep 结果
- 实际 λ 四档 / bs / 步数 / 每臂耗时：
- 标定 smoke 占比（如跑）：
- 每臂 fine-window Pareto（5K..25K）hp_corr_input / sharp_p95 / lattice / hp_corr_tgv 表：
- 每臂漂移端点 artifact_score / raw_control_corr：
- 是否有 checkpoint 支配 TGV(0.960, 0.959, ≤0.0169)？最佳点坐标：
- 残差自检：缓存 npy 均值≈23？(是/否)
- 判定：Claim 4 正结果 / 干净反证 + 决定性数字
- 异常（NaN/发散/降权重重启等）：
- 产物路径：output/ep07_v9_review/v10_highlam/、ep07_eval_real_metrics.csv
```

完成后由主线回填 ACL-020、`reports/ep07_v9_attribution/` Claim 4 节、`docs/paper/07_experiments.md` §6.2/§6.6。
