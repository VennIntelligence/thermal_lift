# 第 6 章 实验与消融（中文打磨稿）

> 本文是论文 **§6（对应英文权威稿 `docs/paper/07_experiments.md`）** 的中文打磨稿。
> **终表铁律**：T1/T2 的每个数字都来自**同一个统一 harness 的单次重跑**
> （`algos/ep11_dl_benchmark/scripts/run_unified_harness_t1_t2.py`，产物 `output/ep11_unified_harness/`）。
> TensorBoard `eval_real/*` 值与 EP11/harness 值**用不同的 artifact 尺度，绝不混入同一表/图**；
> 轨迹图（§6.2）可用 TB-scale（图内一致、图注声明）。

## 6.1 主对比（T1 + F5）

终表 T1 全部数字来自统一 harness（`t1_metrics.csv`，manifest 同目录）。本表 artifact 用 EP11/common.metrics
harness 尺度；TB-scale `eval_real/*` artifact 只用于 §6.2 的臂内轨迹图，**不混入本表**。
所有 hybrid/V10 臂的缓存均值已过 23 °C 自检（V10 λ=1.2@15K 均值 23.288 °C），确认 residual-over-drizzle 推理
**把 channel 5 基加回来了**、而不是报告一个近零的 delta 场。

| 臂 | step | split NRMSE↓ | artifact↓ | corr↑ | FRC16↑ | FRC14↑ | FRC12↑ | lattice↓ | sharp P95↑ | FWHM µm↓ | dip↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicubic | — | 0.047 | 1.388 | 1.000 | 0.951 | 0.945 | 0.941 | 0.001 | 0.344 | 60 | 1.000 |
| drizzle | — | 0.024 | 1.138 | 0.771 | 0.439 | 0.564 | 0.765 | 0.001 | 0.481 | 45 | 0.970 |
| MAP-TV (5x) | — | 0.024 | 2.333 | 0.438 | 0.965 | 0.955 | 0.948 | 0.002 | 0.352 | 42 | 1.000 |
| TGV | — | 0.031 | 0.695 | 0.741 | 0.479 | 0.556 | 0.610 | 0.011 | 0.999 | 40 | 0.973 |
| v6 hot loss | 8K | 0.051 | 1.789 | 0.774 | 0.752 | 0.761 | 0.217 | 0.001 | 0.656 | 40 | 1.000 |
| v8.1a conservative | 15K | 0.073 | 1.943 | 0.758 | 0.712 | 0.660 | 0.429 | 0.001 | 0.864 | 40 | 1.000 |
| v9b band anchor | 11K | 0.054 | 1.766 | 0.777 | 0.744 | 0.741 | 0.197 | 0.001 | 0.698 | 40 | 1.000 |
| v9d full anchor | 7K | 0.054 | 1.726 | 0.771 | 0.744 | 0.708 | 0.150 | 0.001 | 0.697 | 40 | 1.000 |
| V9A hybrid | 10K | 0.054 | 1.762 | 0.719 | 0.945 | 0.870 | 0.190 | 0.001 | 0.683 | 40 | 1.000 |
| V9C hybrid legal anchor | 5K | 0.064 | 1.669 | 0.718 | 0.891 | 0.769 | 0.195 | 0.001 | 0.766 | 40 | 1.000 |
| V10 residual λ=1.2 | 15K | 0.041 | 2.726 | 0.711 | 0.986 | 0.984 | 0.979 | 0.013 | 0.968 | 40 | 1.000 |

注：MAP-TV 是预计算的 EP15 5x 去卷积锚，故其 `output_grid_scale` 在 CSV 中显式标为 5（不可与 2x 臂逐格隐式混同）。
TGV 的 split/FRC 列**目前**复用 EP16 同子集/同 shifts 的 drizzle 代理（Task E 将补真值）；TGV 的 artifact/corr/zigzag
列是在真实 TGV 图上测得。zigzag FWHM 在 2x（5 µm 步进）下量化饱和（多臂同为 40 µm），区分度低，不应过度解读。

**裁决句（2026-06-13 reframe，见 `reframe_c4_claim3.md`）**：在该无 GT 区**不存在可认证的单一赢家**。统一 harness
是**把证据拆开**而非评出赢家：TGV 在 2x 锚口径下 artifact 最低、contour 宽度强，但带已知的 TV 阶梯/beading caveat；
drizzle 与部分学习臂在 raw-control corr 或 FRC 式一致性上更高，但要么偏软、要么携带不可验证的学习高频。高-λ V10
fine-window sweep 表明 residual-over-observation 使「保真–锐度–grain」权衡**可调**：λ=1.2@15K 达到锐度 ≈ TGV
而 fine-window grain 更低（`lattice` 0.014 < TGV 0.0169），但观测保真更低（`hp_corr_input` 0.922 < TGV 0.960）。
因此 F5（`output/paper_figures/fig05_main_visual.{png,pdf}`）作为**双域 task-level 视觉门控**报告，**不是**保真或分辨率证据。

## 6.2 零空间漂移（F3）——全文核心负机制图

各臂 (artifact, corr) vs step 轨迹 + forward-loss inset。**本表为 TB-scale `eval_real/*`，图内一致、不与 T1/T2 混表。**

| 臂（1x 输入） | 锚 | artifact 10K→60K | corr 10K→60K | forward-loss 行为 |
|---|---|---|---|---|
| hot loss (v6) | full 0.1（hybrid cfg） | 0.339@2K → 0.883 | 0.773 → 0.648 | —（被混淆，漂移参照） |
| conservative (v8.1a) | none | 0.390 → 0.643 | 0.756 → 0.689 | n/a |
| conservative (v8.1b, PixelShuffle) | none | 0.413 → 0.709 | 0.747 → 0.667 | n/a（失败头，控制） |
| conservative (v9b) | 窄带 0.1 | 0.369 → 0.655 | 0.758 → 0.688 | **自 10K 贴底 0.004–0.009** |
| conservative (v9d) | 全带 0.1 | 0.379 → 0.677 | 0.758 → 0.677 | 1–28K 振荡（如 20K 处 0.575/0.642）后稳定 |
| hybrid (v9c) | 合法 1x 0.1 | 0.516 → 0.695 | 0.714 → 0.669 | 合法锚，漂移未压平 |

关键实测事实：v9b 的 40K→60K 漂移（artifact +0.0145 / corr −0.0082）与无锚的 v8.1a（+0.016 / −0.009）**几乎重合**——
锚定什么都没改变，其 forward loss 全程贴底。loss 侧旋钮如今覆盖四种锚变体——none(v8.1a)、窄带(v9b)、全带(v9d)、
以及 hybrid 输入下的*合法* 1x 锚(v9c)——全部收敛到同一 ≈0.65–0.70 artifact / ≈0.67–0.69 corr 平台。
**每个 loss 侧变体的漂移曲线近乎一致 ⇒ 漂移是先验驱动、驻留在零空间；loss 侧锚定路线已彻底关闭**
（v9d 击破「带太窄」反驳；v9c 击破「hybrid 输入下锚不合法」反驳）。

## 6.3 输入模式消融（T2 + F6）

矩阵（input × anchor）：{1x 统计, hybrid drizzle} × {none, 窄带, 全带, 合法}。全部臂训练到 60K：
1x×none(v8.1a)、1x×band(v9b)、1x×full(v9d)、hybrid×none(v9a)、hybrid×legal-band(v9c)；第五个学习变体 V10
在 hybrid 输入上叠加 residual-over-observation 参数化 + 可调惩罚 λ（即 Claim-4 旋钮，§6.1）。
读数：选定 checkpoint 的 harness 指标 + 中心细线/边缘阶梯视觉裁剪 + 臂内轨迹。T2 行来自 `t2_metrics.csv`：

| 臂 | 输入 | 锚/参数化 | step | split↓ | artifact↓ | corr↑ | FRC16↑ | FRC14↑ | FRC12↑ | lattice↓ | sharp P95↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| v8.1a | 1x stats | none | 15K | 0.073 | 1.943 | 0.758 | 0.712 | 0.660 | 0.429 | 0.001 | 0.864 |
| v8.1b | 1x stats | none + PixelShuffle 控制 | 5K | 0.050 | 1.782 | 0.739 | 0.734 | 0.692 | 0.392 | 0.001 | 0.682 |
| v9b | 1x stats | 窄带 0.1 | 11K | 0.054 | 1.766 | 0.777 | 0.744 | 0.741 | 0.197 | 0.001 | 0.698 |
| v9d | 1x stats | 全带 0.1 | 7K | 0.054 | 1.726 | 0.771 | 0.744 | 0.708 | 0.150 | 0.001 | 0.697 |
| V9A | hybrid drizzle2x | none | 10K | 0.054 | 1.762 | 0.719 | 0.945 | 0.870 | 0.190 | 0.001 | 0.683 |
| V9C | hybrid drizzle2x | 合法 1x 锚 | 5K | 0.064 | 1.669 | 0.718 | 0.891 | 0.769 | 0.195 | 0.001 | 0.766 |
| V10 | hybrid drizzle2x | residual λ=1.2 | 15K | 0.041 | 2.726 | 0.711 | 0.986 | 0.984 | 0.979 | 0.013 | 0.968 |

两臂归因已立（loss-cooldown vs PixelShuffle）：最细线模糊对 loss 温度与 HR 头都不变 → **输入信息瓶颈**。
V9A 早期证据（fine-window 诊断 step 5K/10K）：中心细线比 1x 统计输入能暴露更多相位结构，但最终 harness 指标显示
**这份输入证据不是免费的保真收益**。锚轴在两种输入下都为 null：无论 1x 输入锚（v9b/v9d）还是 hybrid 输入下的合法锚
（v9c）都未压平漂移（§6.2）。V10 通过把输出参数化为 residual-over-observation 改变了权衡，但其选定行仍是一个
sharp/高-FRC、较低-corr/高-artifact 的工作点。**结论：输入通路能暴露亚像素证据，而 loss 侧锚定无法让合成先验漂移
变得可观测；没有任何 T2 行能认证学习保真。**

## 6.4 帧数预算（EP16 经典 CPU 臂，已完成）

EP16 完成了 drizzle 与 TGV 的经典 CPU 子集矩阵：N = {31, 62, 124, 248}，相位分层抽样，17 个 frame-budget 行全部成功
（`output/ep16_budget_robustness/frame_budget.csv`）。**EP16 表是推理期稳定性研究，不属于统一 T1 harness。**

drizzle 呈现最清晰的信息预算趋势：raw-control corr 从 N=31 的 0.747±0.032 升到 N=248 的 0.771，split-half NRMSE
从 0.0715 降到 0.0306，artifact 从 1.649 降到 1.145，FRC@16 µm 从 0.109 升到 0.479。**多数 raw-control 增益在
N=62 即出现**；FRC/split-half proxy 随更多相位覆盖继续改善。

TGV 在同一矩阵全程 artifact 低于 drizzle（N=31 的 0.946±0.003 到 N=248 的 0.708），但 raw-control corr 非单调
（0.728±0.053、0.754±0.009、0.735±0.013、0.741）。TGV 的 split/FRC 列用同子集/同 shift 的 drizzle 代理（为把过夜
预算控制在 17 次全量 TGV 重建内）；TGV 的 artifact/raw-control/zigzag 列在真实 TGV HR 上测得。MAP-TV 与学习臂留作后续，
不作主文门控。

## 6.5 鲁棒性（EP16 经典 CPU 臂，已完成）

EP16 完成 shift 扰动与对齐源消融的经典矩阵（`shift_robustness.csv` 20 行全成功；`alignment_source.csv` 4 行全成功）。
shift 扰动是对**实测** contour-refined shift 加高斯噪声的压力测试，**不得解读为对齐误差真值**。

drizzle 在 σ = 0 → 0.2 px 下 raw-control corr 几乎不变（0.771 → 0.770），但 artifact 恶化 1.145 → 1.434、
FRC@16 µm 降 0.479 → 0.340。TGV 的 raw-control corr 在该代理下也稳定（0.741 → 0.744），而共享 FRC 代理随同子集/同 shift
同样下降。**结论是指标特异的**：raw-control 一致性在小扰动下稳定，覆盖/FRC 式 proxy 更敏感。

对齐源消融更强：把 command-prior shift 换成 contour-refined shift，drizzle 的 raw-control corr 0.662 → 0.771、
FRC@16 µm 0.0166 → 0.479；TGV 的 raw-control corr 在同样替换下 0.642 → 0.741。**这是数据驱动对齐精修有效的端到端证据。
command shift 仍是先验、不是真值。**

## 6.6 选点协议实战（F4）

逐臂 Pareto 散点 + TGV 参照点（0.695, 0.916，TB-scale）；选定 checkpoint（v6@8K, v8.1a@15K, v8.1b@5K, v9b@11K）
对比 60K 端点；视觉 panel 确认选择。**金句：端点上报（领域默认做法）会让每个臂交出其最差的 checkpoint。**

## 6.7 负结果（保留，一小节）

PixelShuffle HR 头（条纹、proxy 更差）；4x 网络（无真实增益，与 MTF 界一致——4x Nyquist MTF ≤ 0.042）；
loss 侧锚定对零空间漂移无效（§6.2，v9b+v9d+v9c 已闭合）；渲染 AVI 作 SR 输入（8-bit、67% 重复——audit 排除，仅作方向旁证）。

> 待办（§6）：① F5 图注精修，并决定迁 LaTeX 前是否加第二个 held-out fine-window 检查（见 Task E E2）；
> ② T2 的 F6 视觉裁剪行与正文交叉引用对齐。
