# EP07 V9 系列研究发现与下一步行动计划

> **日期**: 2026-06-12
> **状态**: V9A/V9B/V9D 已完成 60K；V9C 在 GPU 1 训练中（~20K/60K，预计今晚 20:30–21:30 完成）；GPU 0 空闲
> **用途**: 本文档为接手智能体提供完整上下文——V9 系列归因实验的定量发现、机制解读、以及已规划的下一步 CPU/GPU 工作。执行提示词见 `tmp/codex_next_move_prompt.md`。
> **前置阅读**: `AGENTS.md`（项目规范）、`research_log/algorithm_changelog.md` ACL-015～ACL-019、`algos/ep07_unet_sr/scripts/run_v9.md`

---

## 1. TL;DR — 三个核心发现

1. **相位信息瓶颈已证实并被打通**：v8.1 时代 5 个 1x 统计输入通道在进网络前坍缩了 248 帧的亚像素相位信息（ACL-015 推断）。V9A 的 hybrid 2x drizzle 输入让中心最细 zigzag「梯子」结构的信息真正进入网络——早期 checkpoint（10K–20K）能以 highpass corr ≈ 0.97 透传它。
2. **合成结构先验会在训练后期主动吃掉输入里的真实细节**：V9A 在 30K 处发生「保真悬崖」（中心窗口 highpass corr 0.974→0.906 后焊死），换来的锐度超过 TGV 的部分恰好是幻觉过冲（饱和对比、粗细线交融）。loss 侧 forward 锚定（V9B highpass band / V9D full band）对此完全无效——漂移方向位于 forward 算子零空间。
3. **UNet 整条训练时间轴的 fidelity–sharpness Pareto 前沿被经典 TGV 支配**：TGV 同时做到保真 0.960、锐度 0.96；最保真的 UNet checkpoint（20K，0.974）锐度只有 0.62，而锐度超过 TGV 的所有 checkpoint（30K+）保真全部塌到 0.906。视觉选优的 25K 在两个轴上被 TGV 严格支配。**这本身是可发表的核心 claim**。

---

## 2. 实验矩阵与状态

V9 是双槽（输入 × 锚定）单因子归因设计，loss 壳统一为 v8.1a conservative loss（见 `algos/ep07_unet_sr/scripts/run_v9.md`）：

| 实验 | 输入 | forward 锚定 | 状态 | 输出目录（`algos/ep07_unet_sr/outputs/`） |
|------|------|--------------|------|------|
| v8.1a | 1x 统计 5ch | 无 | 完成（对照基线） | `ep07_v8_1a_loss_cooldown` |
| V9A | **hybrid drizzle 2x 8ch** | 无 | **完成 60K**（35K 中断后 bs=64 续跑） | `ep07_v9a_hybrid_drizzle` |
| V9B | 1x 统计 5ch | highpass band, w=0.1 | 完成，已回填 ACL-017 | `ep07_v9b_fwd_consistency` |
| V9C | hybrid drizzle 2x 8ch | 合法 1x lr_obs highpass, w=0.1 | **训练中** ~20K/60K, bs=64, GPU 1 | `ep07_v9c_hybrid_legal_fwd` |
| V9D | 1x 统计 5ch | full band, w=0.1 | **完成 60K**（无独立 ACL 条目，结果已补入 ACL-017） | `ep07_v9d_fwd_fullband` |

hybrid 输入通道顺序（`src/unet_sr/dataset.py::_build_hybrid_obs` / `src/unet_sr/inference.py::infer_from_burst`）：

```
ch 0–4: 1x fused 统计（aligned_mean / median / coverage / variance / highpass）双线性上采到 2x
ch 5:   drizzle mean @2x   ← 观测域最保真的通道（本文档的关键对象）
ch 6:   drizzle coverage @2x
ch 7:   drizzle variance @2x
```

注意 `tcforge/src/tcforge/classical_sr.py::drizzle_features` 对未观测 bin 的填充：mean 填全局均值、coverage=0、variance=0——coverage 通道可识别空洞。

## 3. 全局漂移指标（real_eval 248 帧, contour_refined, EP11 同口径）

`artifact_score`（越小越好）/ `raw_control_corr`（越大越好），完整数据：`output/ep07_v9_review/ep07_eval_real_metrics.csv`（由 `algos/ep07_unet_sr/scripts/v9_review/extract_tb_metrics.py` 从 tb_logs 提取）：

| step | v8.1a | V9A | V9B | V9D | V9C |
|------|-------|-----|-----|-----|-----|
| 10K | 0.390 / 0.756 | 0.446 / 0.719 | 0.369 / 0.758 | 0.379 / 0.758 | 0.516 / 0.714 |
| 20K | 0.476 / 0.729 | 0.514 / 0.702 | 0.486 / 0.735 | 0.575 / 0.642 | 0.689 / 0.655 |
| 30K | 0.602 / 0.703 | 0.660 / 0.663 | 0.611 / 0.709 | 0.615 / 0.694 | — |
| 40K | 0.627 / 0.698 | 0.656 / 0.665 | 0.640 / 0.697 | 0.672 / 0.681 | — |
| 60K | 0.643 / 0.689 | 0.646 / 0.669 | 0.655 / 0.688 | 0.677 / 0.677 | — |

要点：

- **V9A 是唯一漂移压平的臂**（30K→60K artifact −0.014 / corr +0.007，其余臂单调恶化），但平台位置 corr 0.669 低于 v8.1a 的 0.689 → run_v9.md「corr 上升」验收在 60K 不达成。
- **V9D（full band）比 V9B（highpass band）更差**且 1K–28K 剧烈震荡（如 20K artifact 0.575 后回弹），复现 ACL-005 的全频低通梯度冲突。**V9B+V9D 合并结论：loss 侧 forward 锚定路线无论 band 都已关闭**。
- **V9C 早期证据偏负面**：20K 时 artifact 0.689（V9A 同期仅 0.514），漂得比 V9A 更快。注意 V9C 全程 bs=64、V9A 前 35K 为 bs=128，臂间不完全可比，结论以 60K 完整曲线为准。

## 4. 中心细线窗口诊断（本次新增的关键证据）

### 4.1 方法

诊断脚本已迁移到 `algos/ep07_unet_sr/scripts/v9_review/`（tracked），默认输出与 checkpoint npy cache 位于 `output/ep07_v9_review/`：

- 窗口定义：2x 网格全幅 960×1280 → center-1/3 crop（rows 320:640, cols 427:853）→ 窗口 rows 384:518, cols 478:674，即中心两条细线「梯子」及其周边粗 zigzag。
- 指标（`algos/ep07_unet_sr/scripts/v9_review/run_pareto_sweep.py`，结果 `output/ep07_v9_review/v9a_pareto_metrics.csv`）：
  - `hp_corr_input`：窗口内 highpass(σ=5) 对 **drizzle 2x mean 输入通道**的 Pearson corr —— **保真轴**（网络是否保留了输入里的观测信息）
  - `hp_corr_tgv`：对 EP10 TGV（`output/ep10_tgv_sr/best_hr_temperature.npy`）的 highpass corr —— 保真 proxy 交叉验证
  - `sharp_p95`：窗口温度图 P95 梯度幅值 —— **锐度 proxy**（⚠️ 振铃/假边也会推高，AGENTS 硬教训 8）
  - `lattice_score`：highpass 窗口中 |f|>0.35 cyc/px 频段能量占比 —— 捕捉 drizzle 格纹/棋盘伪影

### 4.2 定量结果

| 对象 | hp_corr_input | hp_corr_tgv | sharp_p95 | lattice_score |
|------|--------------|-------------|-----------|---------------|
| drizzle 输入通道 | 1.000 | 0.960 | 0.503 | 0.0015 |
| **EP10 TGV** | **0.960** | 1.000 | **0.959** | 0.0169 |
| V9A 5K | 0.968 | 0.945 | 0.677 | 0.0009 |
| V9A 10K | 0.970 | 0.953 | 0.683 | 0.0010 |
| V9A 15K | 0.936 | 0.937 | 1.017 | 0.0028 |
| **V9A 20K** | **0.974** | 0.944 | 0.615 | 0.0009 |
| V9A 25K | 0.935 | 0.931 | 0.831 | 0.0062 |
| V9A 30K | 0.908 | 0.908 | 1.147 | 0.0121 |
| V9A 40K–60K | 0.906±0.001 | 0.908 | 1.21–1.25 | 0.015 |

对照：v8.1a 60K 在同窗口 hp_corr_tgv=0.936 / hp_corr_input=0.926 —— V9A 60K（0.935/0.925）与之无差别，hybrid 输入的早期增益在 60K 被完全抹平。

### 4.3 图表证据

| 图 | 路径 | 说明 |
|----|------|------|
| 输入 vs 输出 vs 经典对照面板 | `../output/ep07_v9_review/fine_zigzag_final_panel.png` | drizzle 输入、TGV、v8.1a 60K、V9A 10K/60K 同口径中心裁剪 |
| 训练时间轴 Pareto 散点 | `../output/ep07_v9_review/v9a_pareto_scatter.png` | 保真×锐度，TGV/drizzle 参考点标红 |
| checkpoint 演化条带 | `../output/ep07_v9_review/v9a_checkpoint_strip.png` | 5K→60K 中心窗口视觉演化（raw 2x 网格无插值） |
| 训练期 eval 序列 | `../algos/ep07_unet_sr/outputs/ep07_v9a_hybrid_drizzle/eval_real/` | 每 5K 步的 center_zoom3x 温度图（35K 缺失，中断所致） |

![Pareto 散点](../output/ep07_v9_review/v9a_pareto_scatter.png)

![checkpoint 演化条带](../output/ep07_v9_review/v9a_checkpoint_strip.png)

![输入输出对照面板](../output/ep07_v9_review/fine_zigzag_final_panel.png)

### 4.4 视觉判读（与用户共同确认）

- **drizzle 输入通道**：除模糊外保真最好——中心梯子内部条纹部分可分辨，无幻觉。这是「观测域上限」参考点。
- **V9A 5K–10K**：透传中心结构，但平坦区有编织状/棋盘状网格纹理（drizzle lattice 泄漏，网络尚未学会压制）。
- **V9A 20K**：最保真（0.974）且格纹已压下去，但整体最软。
- **V9A 25K**：开始变锐，中心开始涂抹；在保真和锐度两轴上**均被 TGV 严格支配**。
- **V9A 30K+**：保真悬崖。粗 zigzag 线极锐利、对比饱和，中心粗细线交融成团块——「超过 TGV 的锐度」全部来自幻觉过冲。

## 5. 机制解读（论文 claim 草案）

1. **Claim 1 — 相位信息瓶颈**：多帧亚像素 SR 的学习方法必须在输入端保留相位信息（2x 网格 drizzle 通道），1x 网格统计输入使网络上限退化为单帧锐化。证据：v8.1 A/B 中心模糊对 loss/head 不变 + V9A 早期 0.97 透传。
2. **Claim 2 — 锚定的零空间失效**：1x forward consistency（任何 band）无法抑制真实数据漂移，因为幻觉方向位于 shift→PSF→下采样算子的零空间。证据：V9B/V9D 漂移曲线与无锚 v8.1a 重合，`loss/forward_model` 躺平地板的同时 artifact 持续上爬；V9C（hybrid 下合法 1x 锚）预计补全第三臂。
3. **Claim 3 — 先验侵蚀与时间轴 Pareto 被经典方法支配**：合成结构先验随训练步数用观测保真换取锐度，整条前沿不及 TGV 的 (0.960, 0.96) 工作点；学习方法唯一占优的锐度区间与可测量的观测去相关（幻觉）重合。
4. **Claim 4（待实验，V10）**：把 fidelity–sharpness 权衡从「训练时长」这个失控旋钮变成显式设计参数（残差幅度惩罚 λ），并检验学习方法能否在受控条件下越过经典前沿。

## 6. 对 POC 交付的含义

- 客户面板建议：bicubic | drizzle | TGV | UNet-20K 四列 + 中心 inset。**TGV 做中心细结构主证据**；UNet 的价值点是全局粗轮廓最干净（大块结构边缘比 TGV 的台阶伪影顺滑）。
- UNet checkpoint 选择：**20K**（保真最高、格纹已压制）。不取 60K，符合 ACL-017「artifact/corr 降级为 checkpoint 选择器」决定。
- 与 AGENTS.md 主线一致：contour-level 可视化增益 + 质量门控 + 可复现实验记录；不声明计量级温度恢复。

## 7. 下一步行动（已写成执行提示词 `tmp/codex_next_move_prompt.md`）

### CPU 批（快，先做）

| # | 任务 | 产出 |
|---|------|------|
| C1 | `algos/ep07_unet_sr/scripts/v9_review/` tracked 脚本已承接 V9 review，输出统一到 `output/ep07_v9_review/` | 可复现的论文图管线 |
| C2 | 零训练融合 baseline：`fused = (1−λ)·drizzle_mean + λ·UNet`（λ∈[0,1] 扫描，附 TGV 锚版本），同窗口指标 + Pareto 叠加图 | V10 的对照前沿；可能直接给出比任何 checkpoint 更好的工作点 |
| C3 | V10 代码实现（残差参数化 + λ 惩罚，规格见提示词）+ pytest + ACL-020 条目 + `run_v10.md` | GPU 实验就绪 |
| C4 | V9C 完成后（~今晚）：指标提取、fine-window 诊断、ACL-019 回填 | 2×2 矩阵闭合 |
| C5 | `reports/ep07_v9_attribution/` 报告骨架（claim 1–3 + 图引用 + split-half FRC TODO） | 论文素材登记 |

### GPU 批（慢，CPU 批完成后）

- **V10 λ 扫描**：GPU 0 现在空闲，smoke 通过后立刻可上第一臂；V9C 完成后 GPU 1 加入第二臂。设计：`total_steps=25000` 完整 cosine 退火（修正「中 LR 抓 checkpoint」的不principled 之处），λ 三档（具体取值由 smoke 的 loss 量级标定，目标是惩罚项初期占总 loss 10–30%）。
- 每臂完成后：real_eval 漂移曲线 + fine-window Pareto 叠加 + ACL-020 回填。
- 成功判据：某个 λ 的工作点在 (hp_corr_input, sharp_p95) 平面上**支配 TGV (0.960, 0.96)**，且无新增格纹/振铃 → Claim 4 正结果；全部失败 → Claim 3 升级为「即使输出网格锚定也无法越过经典前沿」。

## 8. 风险与 caveat

- `sharp_p95` 是锐度 proxy：振铃、假边、饱和对比都会推高它，**不能单独作为 SR 成功证据**（AGENTS 硬教训 8）。报告/论文必须配 split-half FRC（EP15 口径）做收敛证据。
- `hp_corr_input` 以 drizzle 输入为参照，对「忠实复制输入的模糊」也给高分——它度量保真不度量增强；两轴必须一起看。
- TGV 的锐度里含台阶伪影（其 lattice_score 0.017 是所有对象里最高的）；TGV 不是 ground truth，只是当前最优经典参照。
- V9A 35K 中断续跑（bs 128→64）使 30K 前后的训练动力学不完全连续；30K 悬崖与此重合，**不能完全排除 batch size 变化的混杂**——V10 用恒定 bs 重跑可顺带检验。
- `output/` 不入 Git：V9 review 图、CSV 与 checkpoint npy cache 需要通过 tracked CLI 从训练产物和经典参照重新生成。

## 9. 文件索引

| 类别 | 路径 |
|------|------|
| 训练代码 | `algos/ep07_unet_sr/src/unet_sr/`（dataset/model/losses/train/inference/real_eval） |
| V9 启动命令 | `algos/ep07_unet_sr/scripts/run_v9.md` |
| 变更日志 | `research_log/algorithm_changelog.md`（ACL-015～019 已回填 V9A/V9B/V9D 结果） |
| 训练产物 | `algos/ep07_unet_sr/outputs/ep07_v9{a,b,c,d}_*/`（checkpoints + tb_logs + eval_real PNG） |
| V9 review 复现脚本 | `algos/ep07_unet_sr/scripts/v9_review/{common,extract_tb_metrics,run_pareto_sweep,render_comparison_panels}.py` |
| V9 review 输出 | `output/ep07_v9_review/{ep07_eval_real_metrics,v9a_pareto_metrics,fine_zigzag_panel_metrics}.csv`、`*.png`、`cache/*_temperature.npy` |
| 经典方法参照 | `output/ep10_tgv_sr/best_hr_temperature.npy`、`output/ep10_drizzle/drizzle_pf*_hr.npy` |
| 真实数据加载口径 | `unet_sr/real_eval.py::_load_real_eval_cache`（248 帧 + contour_refined 位移，经 `algos/ep06_sr_poc/src/common/`） |
| 执行提示词 | `tmp/codex_next_move_prompt.md` |
