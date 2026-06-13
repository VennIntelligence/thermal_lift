# V10 高-λ 残差扫描 — GPU 后台交接提示词

> **用途**: 本文件**整体**就是交给「具备长等待能力的后台 GPU 系统/智能体」的提示词。
> 复制全文给该智能体即可。它需要自主完成：环境自检 → 4 臂训练 → 评估 → 回报判定。
> **变更记录**: `research_log/algorithm_changelog.md` ACL-020（含本轮动机：V10 评估 bug 修正 + λ 区间过弱）。
> **前置阅读（该智能体必读）**: `AGENTS.md`、ACL-016/017/019/020、`algos/ep07_unet_sr/scripts/run_v10.md`。

---

> ## ✅ READY TO LAUNCH（2026-06-13 标定后定稿，务必先读 — 覆盖 §1/§5/§6 旧 λ）
>
> **最终 spec**：4 臂 **λ ∈ {0.2, 0.5, 1.2, 3.0}**、**bs=128**（已确认未降）、**patch=192**（3090 OOM 被迫从 256 降；全臂统一，记可比性 caveat）、25K 步、`--save-every 2500`。判据见 §2（平衡版，**不追求"打败 TGV"**）。
>
> 1. **成功判据（§0/§2/§8/§10）**：学习输出（含 V10）在中心细线窗口携带的高频内容**最多**（v10 lattice 0.024 > TGV 0.017 > drizzle 0.0015），无 GT 不可验证——**TGV 并非 grain 最多者**。目标 = **保真 hp_corr_input 高 + grain(lattice) ≤ TGV(0.0169) + 梳齿清晰连续**（底稿 `docs/paper/reframe_c4_claim3.md`）。
> 2. **λ 区间定为 {0.2, 0.5, 1.2, 3.0}（几何跨越，由推理定，非标定）**：旧 V10（patch256/bs64）λ≤0.15 收敛仍是自由漂移（hp_corr≈0.88、lattice≈0.024）⇒ 下限取 0.2（刚过"过弱"）；几何跨到 3.0（强绑定/近 drizzle 软态锚）。中间 0.5/1.2 覆盖过渡。**收敛曲线由 25K 的 2.5K 间隔 checkpoint + 事后 fine-window 评估来读**，Phase 2 在 grain 跨过 0.0169 的两臂之间二分细化。
> 3. **300 步标定不可靠（仅供参考）**：LR 才到峰值 0.6%，`mean|delta|` 非单调（λ=0.1→0.021, 0.2→0.040, 0.4→0.032, 0.8→0.007）、全部 sub-noise。**不要用它选 λ**；"占比%"亦自限制不可外推。最终 λ 已按 2 定，凭收敛 checkpoint 判读。
> 4. **bs/patch 已定**：bs=128 **已确认未降** ✓（bs 混杂已消除）；patch=192 全臂统一（现有对照臂旧 V10/V9A 为 patch=256，记 caveat；若有 ≥32GB 卡可改 patch=256 提升可比性，但需重测一致）。

---

## 0. 你的任务（一句话）

在 **batch_size=128、25K 步、完整 cosine 退火** 下，对 V10 残差参数化 `pred = drizzle_mean(ch5) + delta`（惩罚 `λ·mean|delta|`）做**残差强度扫描**，把「保真↑/锐度↓」折中曲线从 V9A 区域探到 drizzle base，判定残差约束能否产出「**锐而不 grain、保真不掉**」的工作点（**不是**"打败 TGV"——判据见 §2、HOLD 横幅、`docs/paper/reframe_c4_claim3.md`）。

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

  > ⚠️ **上表已作废**（见 READY 横幅 2/3）：①「占比%」自限制、不能线性外推；②300 步标定 LR 仅 0.6% peak、`mean|delta|` 非单调，不可用于选 λ。**最终 λ 已定为 {0.2, 0.5, 1.2, 3.0}**（由旧 V10 收敛行为推理），凭 25K checkpoint 判读收敛曲线。

## 2. 科学问题与成功判据（2026-06-13 重写 — 替换旧"支配 TGV"判据）

> **判据改写的依据**：主线复核（`docs/paper/reframe_c4_claim3.md`）发现，原"在 (hp_corr_input, sharp_p95) 上支配 TGV"方向不成立——这两个轴**不度量轮廓连续性**，`sharp_p95` 还会被珠串/颗粒推高；而且 **TGV 并非 grain 最多者**，学习输出（含 V10）的高频 `lattice` 反而最高且无 GT 不可验证。所以本轮**不追求"打败 TGV"**。

**真正要回答的问题**：残差约束能否产出一个「**锐而不 grain、且保真不掉**」的工作点——即把旧 V10 过多的 grain（`lattice≈0.024`）压下去，同时保留中心梳齿的清晰/连续，且 `hp_corr_input` 不比 V9A 折中区更低。

中心细线窗口参照点（`hp_corr_input`=保真↑，`sharp_p95`=锐度↑但**不可单用**，`lattice`=grain/HF↓）：

| 对象 | hp_corr_input | sharp_p95 | lattice | 目视 |
|---|---|---|---|---|
| drizzle（观测，软上限） | 1.000 | 0.503 | 0.0015 | 连续但糊 |
| **EP10 TGV（经典锚，非 GT）** | **0.960** | **0.959** | **0.0169** | 解析梳齿，有 TV-staircase 珠串 |
| 旧 V10 三档 @25K（修正后） | 0.88–0.91 | 1.2–1.37 | 0.017–0.024 | 梳齿锐，但 grain 最多（要改善的起点） |

**有价值的工作点（不等于"打败 TGV"）**：某 checkpoint 满足
`hp_corr_input ≥ 0.92`（保真高于旧 V10 折中区、趋向 drizzle）**且** `lattice ≤ 0.0169`（grain 不高于 TGV）**且** 视觉门控：中心梳齿清晰**连续**、无新增格纹/振铃。→ 说明显式残差约束能在"锐+保真+低 grain"间找到比自由漂移更好的点。
**若全程做不到**（grain 始终 > TGV，或压低 grain 必然把保真/锐度也压回 drizzle 软态）：记为干净结论"显式残差控制无法在该网络/先验下同时拿到锐度与低 grain"，报告 (hp_corr_input, lattice, sharp_p95) 三维曲线随 λ 的走向。
**红线**：任何结论都**不得**写成"学习支配/打败 TGV"，也不得把高 `sharp_p95` 单独当作成功；锐度必并报 `lattice` + 视觉，且"目视轮廓更可辨"只作 task-level 偏好、非保真证据。

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

## 5. λ 标定 smoke（✅ 已完成 2026-06-13；结论：300 步不可靠，最终 λ 见 READY 横幅 / §6）

> 已在 patch=192/bs=128 真实池跑过 4 档 300 步：`mean|delta|`(°C)：λ0.1→0.021, 0.2→0.040, 0.4→0.032, 0.8→0.007。
> **非单调、全部 sub-noise、LR 仅 0.6% peak ⇒ 不可用于选 λ**。最终 λ={0.2,0.5,1.2,3.0} 由旧 V10 收敛行为推理（横幅 2）。下方命令仅留作复现参考。

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

> ⚠️ 上述「按占比% 标定」方法**已证伪**（占比自限制 + 300 步太早）。**不要再用它选 λ**；最终 λ 已定（§6/横幅），靠 25K 收敛 checkpoint 判读 fidelity-grain 曲线。

## 6. Phase 1 — 四臂训练（核心，4 × 25K）

**最终四档 λ ∈ {0.2, 0.5, 1.2, 3.0}**（见 READY 横幅，几何跨越自由→强绑定）。逐臂模板（`<LAM>`/`<TAG>` 成对替换：0.2→020 / 0.5→050 / 1.2→120 / 3.0→300；`<GPU>` 选可用卡；双 GPU 可两两并行）：

```bash
cd <REPO>/algos/ep07_unet_sr
CUDA_VISIBLE_DEVICES=<GPU> uv run python -m unet_sr.train \
  --training-pool-dir ../../data/synthetic/training_pool_2x_aa_burst \
  --output-dir outputs/ep07_v10_resid_hl_lam<TAG> \
  --input-mode hybrid_drizzle2x \
  --residual-mode drizzle2x \
  --residual-penalty-weight <LAM> \
  --scale 2 \
  --patch-size-hr 192 \
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

1. 汇总四臂 `v9a_pareto_metrics.csv`，连同 TGV/drizzle/旧 V10 参照点，画 (hp_corr_input, lattice) 与 (hp_corr_input, sharp_p95) 两张图；**lattice 必须作为独立第三轴呈现**，不要只看 (hp_corr_input, sharp_p95)。
2. 是否有 checkpoint 满足 `hp_corr_input ≥ 0.92` **且** `lattice ≤ 0.0169` **且** 视觉门控（梳齿清晰连续、无新增格纹/振铃）？
   - **是** → 说明残差约束拿到了「锐+保真+低 grain」的有价值工作点（**仍不写成"打败 TGV"**）。
   - **否** → 报告 (hp_corr_input, lattice, sharp_p95) 随 λ 的三维走向、最佳折中点坐标 → 干净结论「显式残差控制无法同时拿到锐度与低 grain」。

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
- 是否有 checkpoint 满足 hp_corr_input≥0.92 且 lattice≤0.0169 且视觉连续？最佳折中点坐标 (hp_corr_input, sharp_p95, lattice)：
- 残差自检：缓存 npy 均值≈23？(是/否)
- 判定：有价值工作点 / 干净结论（不写"打败TGV"）+ 决定性三维数字
- 异常（NaN/发散/降权重重启等）：
- 产物路径：output/ep07_v9_review/v10_highlam/、ep07_eval_real_metrics.csv
```

完成后由主线回填 ACL-020、`reports/ep07_v9_attribution/` Claim 4 节、`docs/paper/07_experiments.md` §6.2/§6.6。
