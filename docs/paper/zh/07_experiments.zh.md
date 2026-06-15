# 第 5 章 实验与消融

本节在同一无真值评估框架下比较经典与学习方法，回答三个递进问题：(i) 无 GT 下谁更好？没有可认证的赢家（§5.1）；(ii) 学习方法为何持续漂移？合成先验侵蚀活在观测算子的零空间内，loss 侧锚定对此盲目（§5.2）；(iii) 输入通路与残差参数化能否改变权衡？能调节，但不能认证保真（§5.3）。全部终表数字来自统一 harness 单次重跑（`output/ep11_unified_harness/`），确保相同 ROI 与指标管线；轨迹图（§5.2）使用 TensorBoard `eval_real/*` 尺度，图注声明。

## 5.1 无 GT 下不存在可认证的赢家

表 T1 汇总经典方法与学习方法在统一口径下的指标。加粗标出每列最优——**加粗散落在不同方法上，没有任何一行全面胜出**。

**表 T1. 主对比**（加粗 = 该列最优 ↑最高/↓最低；FRC@{14,12} µm 见附录 S-T1）

| 方法 | step | split↓ | artifact↓ | corr↑ | FRC@16↑ | lattice↓ | sharp↑ | FWHM↓ | dip↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicubic | — | 0.047 | 1.388 | **1.000**† | 0.951 | **0.001** | 0.344 | 60 | **1.000** |
| drizzle | — | **0.024** | 1.138 | 0.771 | 0.439 | **0.001** | 0.481 | 45 | 0.970 |
| MAP-TV ($5\times$) | — | **0.024** | 2.333 | 0.438 | 0.965 | 0.002 | 0.352 | 42 | **1.000** |
| TGV | — | 0.032 | **0.695** | 0.741 | 0.975 | 0.011 | **0.999** | **40** | 0.973 |
| | | | | | | | | | |
| HotLoss | 8K | 0.051 | 1.789 | 0.774 | 0.752 | **0.001** | 0.656 | **40** | **1.000** |
| Stats | 15K | 0.073 | 1.943 | 0.758 | 0.712 | **0.001** | 0.864 | **40** | **1.000** |
| Stats+HP-FC | 11K | 0.054 | 1.766 | 0.777 | 0.744 | **0.001** | 0.698 | **40** | **1.000** |
| Stats+Full-FC | 7K | 0.054 | 1.726 | 0.771 | 0.744 | **0.001** | 0.697 | **40** | **1.000** |
| Hybrid | 10K | 0.054 | 1.762 | 0.719 | 0.945 | **0.001** | 0.683 | **40** | **1.000** |
| Hybrid+Native-FC | 5K | 0.064 | 1.669 | 0.718 | 0.891 | **0.001** | 0.766 | **40** | **1.000** |
| Hybrid+ResObs $\lambda\!=\!1.2$ | 15K | 0.041 | 2.726 | 0.711 | **0.986** | 0.013 | 0.968 | **40** | **1.000** |

> **表注.** ① 加粗 = 该列最优（↑最大/↓最小）。② 经典与学习方法间以空行分组（LaTeX 迁移时改 `\midrule`）。③ †bicubic corr = 1.000 为平凡值（恒等上采样，未注入信息）。④ TGV 的 split/FRC 为相位分层实际 half-set 重建（seed 42，A/B = 124/124），非 drizzle 代理。⑤ MAP-TV 为 $5\times$ 去卷积基准，`output_grid_scale` 与 $2\times$ 方法不同。⑥ Zigzag FWHM 在 $2\times$（5 µm 步进）下量化饱和，多方法同为 40 µm，区分度低。⑦ 学习变体 checkpoint 按 proxy Pareto 选取（附录 C.5），非训练端点——端点上报会交出漂移最严重的 checkpoint。⑧ Hybrid / Hybrid+ResObs 缓存均值已过 23 °C 自检，确认 residual-over-drizzle 推理正确加回 channel 5 基。

散落的加粗直接印证核心裁决。TGV 在保真 proxy 上领先（artifact 最低 0.695、锐度最高 0.999），但在最细对角轮廓上留有 TV-staircase 珠串——这是观测本身不具有的伪影。Hybrid+ResObs 在频率一致性上领先（FRC@16 = 0.986）且锐度接近 TGV（0.968 vs 0.999），但 artifact 为全表最高（2.726）、corr 最低之一（0.711），携带不可验证的高频 grain。Drizzle 的 split-half 一致性最优，但频率恢复最弱（FRC@16 = 0.439）。统一 harness 的作用是将证据按维度拆开，而非评出赢家。

图 F5 将这种互补失效做成双域、双 ROI 的视觉对照。

![图 F5：主视觉对比——两个固定几何 ROI × 双域。列依次为 drizzle、TGV、Hybrid late endpoint (60K)、Hybrid+ResObs $\lambda\!=\!1.2$ @15K；上两行为中心 zigzag ROI 的温度域与 highpass 域，下两行为预声明的 held-out ROI2（右上外围 plate-edge/zigzag-branch）的同一对域。每行色标独立。温度行检查「是否只是描边」，highpass 行承载锐度/grain 差异。中心 ROI：drizzle 偏软、TGV 轻微 TV-staircase、Hybrid 60K 边缘过度增厚、Hybrid+ResObs 锐度≈TGV 但 grain 更低。F5 是 task-level 视觉门控，不是保真或分辨率证据。](../../../output/paper_figures/fig05_combined_visual.png)

四列读法：drizzle 偏软但最忠于观测；TGV 轮廓最锐但有 TV-staircase 珠串；Hybrid 60K 端点边缘过度增厚（漂移后果）；Hybrid+ResObs 锐度与 TGV 相当但仍携带不可验证的高频内容。目视偏好是 task-level 主观判断，不构成保真证据。第二验证 ROI（F5 下两行）的 lattice 排序与中心 ROI 一致（drizzle < TGV < Hybrid+ResObs < Hybrid-60K），但 sharp_p95 与 zigzag profile 排序跨 ROI 不一致——因此 F5 只作为 held-out 视觉/proxy 审计，不升级为方法胜负判定（附录 D.6）。

## 5.2 核心发现：漂移活在零空间

图 F3 是全文的核心机制结果。五个 $1\times$ 输入变体在训练过程中的 proxy 轨迹揭示了一条统一的失效机制：合成先验侵蚀沿观测算子零空间方向累积，loss 侧锚定对此盲目。

![图 F3：零空间漂移轨迹。三 panel 共享 step 横轴，覆盖五个 $1\times$ 输入变体：(a) artifact $\uparrow$、(b) corr $\downarrow$ vs 训练步数；(c) forward-consistency loss（log $y$），Stats+HP-FC / Stats+Full-FC 自约 10K 起贴底于 0.004–0.009 灰带。Canonical checkpoint $\circ$，60K 端点 $\times$。](../../../output/paper_figures/fig03_nullspace_drift.png)

| 变体（$1\times$ 输入） | 锚 | artifact 10K $\to$ 60K | corr 10K $\to$ 60K | forward-loss 行为 |
|---|---|---|---|---|
| HotLoss | full 0.1 | $0.339 \to 0.883$ | $0.773 \to 0.648$ | 被混淆（漂移参照） |
| Stats | none | $0.390 \to 0.643$ | $0.756 \to 0.689$ | — |
| Stats+PixelShuffle | none | $0.413 \to 0.709$ | $0.747 \to 0.667$ | —（失败头，控制） |
| Stats+HP-FC | HP-FC 0.1 | $0.369 \to 0.655$ | $0.758 \to 0.688$ | 自 10K 贴底 0.004–0.009 |
| Stats+Full-FC | Full-FC 0.1 | $0.379 \to 0.677$ | $0.758 \to 0.677$ | 1–28K 振荡后稳定 |

两个实测事实构成零空间判据。**第一，漂移平行**：四种锚定变体（none / HP-FC / Full-FC / Hybrid+Native-FC）的漂移曲线近乎一致，均收敛到 artifact $\approx 0.65$–$0.70$ / corr $\approx 0.67$–$0.69$ 平台——例如 Stats+HP-FC 在 40K→60K 的增量（artifact $+0.015$ / corr $-0.008$）与无锚 Stats（$+0.016$ / $-0.009$）几乎重合。**第二，forward loss 贴底**：HP-FC 与 Full-FC 的观测域 forward-consistency loss 自 10K 起全程贴底于 0.004–0.009，说明漂移分量落在观测算子 $A$ 的零/近零空间内——$A$ 看不到的方向上，有限权重的锚定提供零恢复曲率。

这与理论预测一致：对 $A$ 的精确或 $\epsilon$-零空间内的漂移分量，有界频带的观测侧锚无法制造非消失的恢复力（附录 A.2，命题 P1–P3）。Hybrid 输入变体的轨迹因 proxy 跨输入模式不可横比，移至附录（图 S-F3）。

## 5.3 消融：输入通路与观测锚定

表 T2 按 $\{\text{1x stats},\ \text{hybrid drizzle}\} \times \{\text{none},\ \text{band},\ \text{full},\ \text{legal}\}$ 矩阵隔离两个因素的贡献，全部变体训练至 60K step。

**表 T2. 输入×锚定消融矩阵**（加粗 = 该列最优；FRC@{14,12} µm 见附录 S-T1）

| 变体 | 输入 | 锚/参数化 | step | split↓ | artifact↓ | corr↑ | FRC@16↑ | lattice↓ | sharp↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| Stats | $1\times$ stats | none | 15K | 0.073 | 1.943 | 0.758 | 0.712 | **0.001** | 0.864 |
| Stats+PixelShuffle | $1\times$ stats | PixelShuffle 控制 | 5K | 0.050 | 1.782 | 0.739 | 0.734 | **0.001** | 0.682 |
| Stats+HP-FC | $1\times$ stats | HP-FC 0.1 | 11K | 0.054 | 1.766 | **0.777** | 0.744 | **0.001** | 0.698 |
| Stats+Full-FC | $1\times$ stats | Full-FC 0.1 | 7K | 0.054 | 1.726 | 0.771 | 0.744 | **0.001** | 0.697 |
| | | | | | | | | | |
| Hybrid | hybrid $2\times$ | none | 10K | 0.054 | 1.762 | 0.719 | 0.945 | **0.001** | 0.683 |
| Hybrid+Native-FC | hybrid $2\times$ | native $1\times$ FC | 5K | 0.064 | **1.669** | 0.718 | 0.891 | **0.001** | 0.766 |
| Hybrid+ResObs | hybrid $2\times$ | ResObs $\lambda\!=\!1.2$ | 15K | **0.041** | 2.726 | 0.711 | **0.986** | 0.013 | **0.968** |

> **表注.** ① 加粗 = 该列最优。② $1\times$ 与 hybrid 输入间以空行分组；两组的 proxy 数值不可跨输入模式直接比较。③ Stats+PixelShuffle 仅作 HR 头归因对照（条纹伪影，附录 D.4）。④ Hybrid+ResObs 的 split↓ 最优与 artifact 最高并存，反映残差参数化压低 split-half 不一致的同时拉高了高频内容。

消融矩阵揭示两条清晰的归因线。

**锚轴为空。** 在两种输入下，所有 loss 侧锚定变体（Stats+HP-FC / Stats+Full-FC / Hybrid+Native-FC）的漂移曲线均与无锚基线重合（§5.2），forward loss 贴底。有限权重的观测侧锚定无法让先验漂移变得可观测——与命题 P1–P3 一致。

**输入通路有信号。** Hybrid 输入使 FRC@16 从 $1\times$ 输入的 0.712–0.744 跃升至 0.945（无锚 Hybrid）乃至 0.986（Hybrid+ResObs），表明亚像素相位通过 drizzle 通道直接注入了网络。但这份输入证据不是免费的保真收益：Hybrid 组的 corr 系统低于 $1\times$ 组（0.711–0.719 vs 0.758–0.777），Hybrid+ResObs 的 artifact（2.726）为全表最高。

**残差参数化使权衡可调。** Hybrid+ResObs 的 $\lambda$ 旋钮将保真–锐度–grain 三维权衡显式化。$\lambda = 1.2$@15K 在频率一致性上大幅领先，锐度接近 TGV，但以更高 artifact 和更低观测保真为代价。中心细线窗口诊断（附录 D.0）中该工作点的 lattice（0.014）低于 TGV（0.017），但该口径依赖窗口选择——第二验证窗重选出 $\lambda = 0.1$、排序部分反转。无论哪个工作点，观测保真均低于 TGV，因此不构成可认证支配。

结论：输入通路能暴露亚像素证据，而 loss 侧锚定无法让合成先验漂移变得可观测。T2 中没有任何一行能认证学习保真。

## 5.4 帧数预算、鲁棒性与负结果

经典方法帧数预算消融（$N \in \{31, 62, 124, 248\}$）显示 drizzle 主要增益在 $N = 62$ 即出现（corr $0.747 \to 0.764$），之后边际递减；shift 扰动压力测试（$\sigma$ 至 0.2 px）下 raw-control corr 稳定但 FRC proxy 更敏感。Contour-refined 对齐精修相对 command prior 提升 drizzle corr 从 0.662 至 0.771——这是数据驱动对齐精修有效的端到端证据。完整曲线与数值表见附录 D.5。

负结果如实上报：PixelShuffle HR 头 \citep{shi2016realtime} 引入条纹伪影，proxy 全程劣于 bilinear 头（仅作 head 归因对照保留）；$4\times$ 网络无真实增益，与 MTF 界一致（$4\times$ Nyquist 处 MTF $\le 0.042$，附录 A.1）；有界 loss 侧锚定对零空间漂移无效，已由三个锚定变体闭合验证（§5.2）。详见附录 D.4。
