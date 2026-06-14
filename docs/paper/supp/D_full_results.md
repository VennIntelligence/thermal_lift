# 补充材料 D —— 完整实验结果

## D.1 统一 benchmark 全表

统一 harness 已于 2026-06-14 完成，入口脚本为
`algos/ep11_dl_benchmark/scripts/run_unified_harness_t1_t2.py`，产物位于
`output/ep11_unified_harness/`。完整列（runtime、source、cache path、temperature sanity、FRC@20/10
等）保存在 `all_arm_metrics.csv`；主文 T1/T2 分别使用 `t1_metrics.csv` 与 `t2_metrics.csv`。
所有 artifact 数字均为 EP11/common.metrics harness scale，不能与 TensorBoard `eval_real/*`
artifact 混表；`tb_vs_harness_scale_check.csv` 给出三条对照（如 v9b@11K: TB 0.3385 vs harness
1.7662）。

### D.1.1 T1 主表精选列

| 方法 | step | split↓ | artifact↓ | corr↑ | FRC16↑ | FRC14↑ | FRC12↑ | lattice↓ | sharp↑ | FWHM µm↓ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| bicubic | — | 0.047 | 1.388 | 1.000 | 0.951 | 0.945 | 0.941 | 0.001 | 0.344 | 60 |
| drizzle | — | 0.023 | 1.138 | 0.771 | 0.439 | 0.564 | 0.765 | 0.001 | 0.481 | 45 |
| MAP-TV (5x) | — | 0.024 | 2.333 | 0.438 | 0.965 | 0.955 | 0.947 | 0.002 | 0.352 | 42 |
| TGV | — | 0.031 | 0.695 | 0.741 | 0.479 | 0.556 | 0.610 | 0.011 | 0.999 | 40 |
| v6 hot loss | 8K | 0.051 | 1.789 | 0.774 | 0.752 | 0.761 | 0.217 | 0.001 | 0.656 | 40 |
| v8.1a | 15K | 0.073 | 1.943 | 0.758 | 0.711 | 0.660 | 0.429 | 0.001 | 0.864 | 40 |
| v9b band | 11K | 0.054 | 1.766 | 0.777 | 0.744 | 0.741 | 0.197 | 0.001 | 0.698 | 40 |
| v9d full | 7K | 0.054 | 1.726 | 0.771 | 0.744 | 0.708 | 0.150 | 0.001 | 0.697 | 40 |
| V9A hybrid | 10K | 0.054 | 1.762 | 0.719 | 0.945 | 0.870 | 0.190 | 0.001 | 0.683 | 40 |
| V9C legal hybrid | 5K | 0.064 | 1.669 | 0.718 | 0.891 | 0.769 | 0.195 | 0.001 | 0.766 | 40 |
| V10 λ=1.2 | 15K | 0.041 | 2.726 | 0.711 | 0.986 | 0.984 | 0.979 | 0.013 | 0.968 | 40 |

边界说明：MAP-TV 是预计算 5x anchor；TGV 的 split/FRC 列复用 EP16 同子集/同 shifts 的
drizzle proxy，TGV 自身列为 artifact/corr/zigzag。V10/V9A/V9C 的缓存均值约 23°C，说明
hybrid/V10 推理没有漏加 drizzle base。

### D.1.2 T2 input × anchor 精选列

| 方法 | input | anchor/参数化 | step | split↓ | artifact↓ | corr↑ | FRC16↑ | FRC14↑ | FRC12↑ | sharp↑ |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v8.1a | 1x stats | none | 15K | 0.073 | 1.943 | 0.758 | 0.711 | 0.660 | 0.429 | 0.864 |
| v8.1b | 1x stats | PixelShuffle control | 5K | 0.050 | 1.782 | 0.739 | 0.734 | 0.692 | 0.392 | 0.682 |
| v9b | 1x stats | band-limited 0.1 | 11K | 0.054 | 1.766 | 0.777 | 0.744 | 0.741 | 0.197 | 0.698 |
| v9d | 1x stats | full-band 0.1 | 7K | 0.054 | 1.726 | 0.771 | 0.744 | 0.708 | 0.150 | 0.697 |
| V9A | hybrid drizzle2x | none | 10K | 0.054 | 1.762 | 0.719 | 0.945 | 0.870 | 0.190 | 0.683 |
| V9C | hybrid drizzle2x | legal 1x anchor | 5K | 0.064 | 1.669 | 0.718 | 0.891 | 0.769 | 0.195 | 0.766 |
| V10 | hybrid drizzle2x | residual λ=1.2 | 15K | 0.041 | 2.726 | 0.711 | 0.986 | 0.984 | 0.979 | 0.968 |

读数：输入端引入 drizzle2x 证据能改变 fine-window 结构权衡，但不是自动保真胜利；loss-side
anchor 在 1x/full/band/legal-hybrid 变体中均未压平真实数据漂移。V10 把 sharp/FRC 推高，同时
artifact 与 corr 变差；它是可调 trade-off，不是可认证赢家。

---

## D.2 FRC 完整档案

本节归档主文 §5.1 中 FRC 分析的全部数值结果，方法学细节见 A.4。

### D.2.1 主曲线与 cutoff

$1/7$ cutoff 为 **17.03 µm**（3-seed 均值曲线），逐 seed 分别为 16.17/16.17/17.03 µm（std = 0.50 µm），half-bit 判据给出相同值。控制组 cutoff 为：bicubic 正控 13.58 µm（未通过预期）、漂移控制 26.20 µm；zero-coverage 均值 27.2%、最大 36.2%。频带全表与控制组详细分析见 A.4.3 和 A.4.4。

### D.2.2 MAP-TV 前后 split-half 对照

以 MAP-TV（$\sigma = 0.2$, $\lambda = 10^{-3}$）对 bare drizzle 做去卷积后，split-half FRC 在 20/16/14/12/10 µm 周期段分别从 0.319/0.088/0.053/0.575/0.893 提升至 0.976/0.965/0.955/0.947/0.934。这反映的是 split-half 一致性的改善，而非光学分辨率本身的提升——bare cutoff 仍为 17.03 µm，与 M2 一致。

### D.2.3 Zigzag 轮廓指标

MAP-TV 锚在 zigzag 剖面上的结构收益有限：中位 FWHM 从 114 µm 缩窄至 100 µm，dip 从 0.929 微升至 0.934，3/3 剖面保持分离但改善不均（2 条变宽、1 条显著变窄）。论文措辞限定为 "limited contour enhancement"。

---

## D.3 漂移演化档案

### D.3.1 全局漂移指标

下表记录了各臂在训练期间真实数据 eval（TB-scale，248 帧 contour\_refined）上的 artifact（$\downarrow$）与 raw\_control\_corr（$\uparrow$）轨迹：

| step | v8.1a | V9A | V9B | V9D | V9C |
|---|---|---|---|---|---|
| 10K | 0.390 / 0.756 | 0.446 / 0.719 | 0.369 / 0.758 | 0.379 / 0.758 | 0.516 / 0.714 |
| 20K | 0.476 / 0.729 | 0.514 / 0.702 | 0.486 / 0.735 | 0.575 / 0.642 | 0.689 / 0.655 |
| 30K | 0.602 / 0.703 | 0.660 / 0.663 | 0.611 / 0.709 | 0.615 / 0.694 | 0.686 / 0.675 |
| 40K | 0.627 / 0.698 | 0.656 / 0.665 | 0.640 / 0.697 | 0.672 / 0.681 | 0.688 / 0.672 |
| 60K | 0.643 / 0.689 | 0.646 / 0.669 | 0.655 / 0.688 | 0.677 / 0.677 | 0.695 / 0.669 |

从轨迹形状看，V9A 是唯一在 30K→60K 段漂移趋平的臂（变化量 −0.014/+0.007，其余单调恶化）。V9D 比 V9B 更差且 1K–28K 段震荡剧烈。V9C 在 30K 后没有恢复，60K 端点为 0.695/0.669，说明 hybrid 输入下的合法 1x anchor 也不能压平漂移。需要强调的是，hybrid 臂（V9A/V9C）与 1x 臂的 proxy 不可跨列横比（见 A.3.2），此处只看各臂轨迹形状。

### D.3.2 Fine-window 训练时间轴

为更精细地诊断训练动态，在 2x 网格全幅 $960 \times 1280$ 的中心细线窗口（rows 384:518, cols 478:674——中心两条细 zigzag 及周边粗线）上，跟踪四个指标：$\text{hp\_corr\_input}$（highpass 域对 drizzle 输入通道的 Pearson，衡量保真性）、$\text{hp\_corr\_tgv}$（同口径对 TGV 的 Pearson，交叉验证）、$\text{sharp\_p95}$（温度图 P95 梯度幅值，衡量锐度）、$\text{lattice\_score}$（highpass 域 $|f| > 0.35$ cyc/px 频段能量占比，捕捉格纹伪影）。

两个参照点为：drizzle 输入通道 $(1.000, 0.503)$——观测域上限（模糊但零幻觉）；TGV $(0.960, 0.959)$——经典前沿工作点。

V9A 的训练时间轴揭示了一个关键现象——**30K 保真悬崖**。V9A 10K 的 $\text{hp\_corr\_input}$ 为 0.970，20K 达峰 0.974，此后断崖下跌：30K 降至 0.908 并在 40K–60K 焊死于 $0.906 \pm 0.001$。同时，锐度 $\text{sharp\_p95}$ 在 30K 后超过 TGV（1.147 vs 0.959），并在 40K–60K 继续攀升至 1.21–1.25——锐度超过 TGV 的区间恰好与去相关重合，指示过冲为幻觉驱动。对照 v8.1a 60K 同窗口为 0.926/0.936，V9A 60K（0.925/0.935）与之无差别——hybrid 早期增益被训练后期完全抹平。

需注意 V9A 的 35K 中断（batch size 从 128 切换至 64）与悬崖时间重合，混杂尚未排除（见 C.3）。

Fine-window 是局部诊断窗口，依赖 TGV 参照，独立性低于 FRC/proxy 对——仅用于归因与选型，不替代全局评估协议。

---

## D.4 负结果档案

本节完整记录四项负结果，为主文 §6.7 提供数据背书。每项按「现象 → 数字 → 结论边界」结构组织。

### D.4.1 PixelShuffle HR 头

v8.1b 使用 pixelshuffle HR 头替代默认的 bilinear 头（其余与 v8.1a 相同），出现了中等边框间条纹状亮色伪影，锯齿未改善，中心细线模糊程度与 bilinear 头相同。TB-scale 数据显示 artifact 全程高于 v8.1a（0.413→0.709 vs 0.390→0.643），corr 全程更低（0.747→0.667 vs 0.758→0.689）。

结论边界：head 归因失败——中心细线瓶颈不在解码头。该结果与 v8.1a 的 loss-cooldown 对照共同构成主文 §6.3 的两臂归因：细线模糊对 loss 温度和 head 均不敏感，指向输入信息瓶颈。

### D.4.2 4x 网络

EP12 4x 网络在所有指标上均劣于 EP07 2x 上采样方案：raw-control highpass Pearson 0.223 vs 0.389，artifact 0.535 vs 0.472。

该实验结果与 A.1.3 的理论预测互证：4x Nyquist MTF $\le 0.042$（$\sigma = 0.2$），$\sigma \ge 0.35$ 时接近零；相位 bin 在 3x/4x 网格出现 collapse（见 B.2.3）。4x 无真实增益且理论上不应有——这是「先第一性原理后实验」工作流的展示案例。

### D.4.3 Forward 锚定（V9B band + V9D full + V9C legal hybrid）

这些锚定臂的漂移曲线与无锚 v8.1a 几乎重合。V9B 的 forward loss 自 10K 步起贴底于 0.004–0.009，同期 artifact 持续上爬。V9B 40K→60K 漂移 +0.0145/−0.0082 vs v8.1a +0.016/−0.009，几乎完全重合。V9D 60K（0.677/0.677）劣于 V9B（0.655/0.688），且 1K–28K 段剧烈震荡——复现了全频低通梯度冲突的已知问题。

V9C 是最后一条反对意见检验：在 hybrid 输入含 2x drizzle 证据的条件下，forward anchor 不再偷用上采样 mean，而是通过数据管线携带合法的 1x aligned-mean patch。结果仍为 0.516/0.714 @10K → 0.695/0.669 @60K，未压平后期漂移。

结论边界：无论频带选择和输入证据路径如何，1x 观测域锚定对真实漂移不可见——机制即 A.2.3 Proposition 1。V9D 关闭了「band 太窄」的反对意见；V9C 关闭了「hybrid 第 0 通道不是合法 1x 观测」的反对意见。loss-side anchoring route 在 band/full/legal-hybrid 三种变体下正式关闭。

### D.4.4 AVI 作 SR 输入

8-bit 渲染视频、约 67% 重复帧、无温度矩阵（见 B.1.3）——直接排除。AVI 仅用于方向一致性检验。该负结果的意义在于：「能拿到的数据」不等于「能进重建的数据」，审计应先于算法。

---

## D.5 帧预算与鲁棒性

本节报告 EP16 的三组经典臂实验：E1 帧数预算、E2 shift 扰动、E3 对齐源消融。共 37 个 unique run 全部成功。注意这是推理期稳定性研究，不属于统一 harness。

### D.5.1 帧数预算

帧数预算实验使用 phase-stratified 子集（$N \in \{31, 62, 124, 248\}$，seeds 101/202/303），drizzle 和 TGV 两臂：

| 方法 | $N$ | corr | split-half NRMSE | artifact | FRC@16 µm | zigzag FWHM (µm) |
|---|---:|---:|---:|---:|---:|---:|
| drizzle | 31 | 0.747±0.032 | 0.0715±0.0012 | 1.649±0.041 | 0.109±0.065 | 56.7±20.2 |
| drizzle | 62 | 0.772±0.016 | 0.0514±0.0034 | 1.536±0.044 | 0.248±0.070 | 56.7±20.2 |
| drizzle | 124 | 0.770±0.010 | 0.0361±0.0010 | 1.341±0.038 | 0.332±0.018 | 45.0 |
| drizzle | 248 | 0.771 | 0.0306 | 1.145 | 0.479 | 45.0 |
| TGV | 31 | 0.728±0.053 | — | 0.946±0.003 | — | 42.5±3.5 |
| TGV | 62 | 0.754±0.009 | — | 0.851±0.051 | — | 40.0 |
| TGV | 124 | 0.735±0.013 | — | 0.747±0.011 | — | 40.0 |
| TGV | 248 | 0.741 | — | 0.708 | — | 40.0 |

TGV 的 split-half/FRC 列使用同子集 drizzle proxy（预算考虑），以 "—" 标注。

Drizzle corr 的增益大半在 $N = 62$ 前到位（0.747→0.772），之后趋平；split-half、artifact 和 FRC 随 $N$ 单调改善，受益于相位覆盖的累积。TGV artifact 全程低于 drizzle（0.946→0.708），但 corr 呈非单调变化（0.728→0.754→0.735→0.741）。

### D.5.2 Shift 扰动

扰动实验在 $\sigma \in \{0, 0.05, 0.1, 0.2\}$ px（seeds 401–403）下测试对齐误差的影响。Drizzle corr 从 0.771 到 0.770 几乎不变，而 artifact 从 1.145 升至 1.434、FRC@16 从 0.479 降至 0.340。TGV corr 同样稳定（0.741→0.744）。

结论：鲁棒性是 metric-specific 的——raw-control corr 对 $\le 0.2$ px 扰动几乎不敏感，而 coverage/FRC 类指标敏感。这是压力测试口径，非真实对齐误差估计。

### D.5.3 对齐源消融

将 command\_prior 与 contour\_refined 两种对齐源的端到端效果作比：drizzle corr 从 0.662 升至 0.771（+0.109），FRC@16 从 0.0166 升至 0.479；TGV corr 从 0.642 升至 0.741（+0.099）。

数据驱动对齐对两个经典臂都带来约 +0.10 corr 的端到端增益，FRC@16 从几乎为零到 0.479——这是 B.2 对齐链投入的最终回报证据。

---

## D.6 视觉 gate panel

已有 EP11 四臂（v6、v8.1a、v8.1b、v9b）的 checkpoint 选择 panel：每臂 canonical + 60K 对照的温度域三联图，标注 proxy 值。v8.1b 行保留条纹伪影证据。主文 F5 已补 V9A late 60K 与 V10 λ=1.2@15K 的双域视觉 gate：`output/paper_figures/fig05_main_visual.{png,pdf}`。V9C 的 step 序列 panel 可作为 supp 选图补充，不是主文硬门槛。

---

## D.7 零训练融合 baseline

本节建立一个无需额外训练的简单 baseline：对经典重建（锚）与学习臂输出做事后线性融合，测试学习臂的增益是否超出后处理可得的范围。

### D.7.1 方法

融合公式为 $\text{fused}(\lambda) = (1-\lambda) \cdot \text{anchor} + \lambda \cdot \hat{x}_{\text{unet}}$，$\lambda \in \{0, 0.1, \ldots, 1.0\}$。锚选择 drizzle 2x mean（248 帧 contour\_refined）或 TGV。UNet 预测使用 V9A 20K（最保真）和 V9A 60K（最锐）。评估指标为 D.3.2 的 fine-window 四指标。

### D.7.2 结果

TGV 锚 × V9A-60K 的融合中，存在严格支配 TGV 工作点的区间。$\lambda = 0.2$ 时四指标为 hp\_corr\_input = 0.963（+0.003）、hp\_corr\_tgv = 0.995、sharp\_p95 = 0.968（+0.009）、lattice = 0.0108（−36%），同时改善了保真、锐度和格纹三个维度。$\lambda = 0.1$ 和 $\lambda = 0.3$ 同样支配 TGV。

相比之下，drizzle 锚 × V9A 的任意 $\lambda$ 均不支配 drizzle 工作点（保真极高但锐度始终不足）。

### D.7.3 结论与边界

第一，存在事后线性组合可严格支配 TGV 工作点——零训练、零 GPU、推理期一次加权即可。第二，V10 的成功判据因此从「越过 TGV」抬高为「越过融合前沿」。若 V10 所有 $\lambda$ 臂均不及融合 baseline，则 Claim 4 将收敛为「学习贡献可被事后融合替代」的诚实结论。

V10 高-λ sweep 的最佳工作点为 $\lambda = 1.2$ @15K：hp\_corr\_input = 0.922、sharp\_p95 = 0.987、lattice = 0.0141。它满足“锐而不 grain”的工作点判据（lattice 低于 TGV 0.0169，sharp 约等于 TGV），但保真仍低于 TGV（0.922 < 0.960），因此不构成可认证支配。需注意 fine-window 为局部口径且依赖 TGV 参照（非 GT），最优 $\lambda$ 在同一窗口上选出而无独立验证窗——终稿若引用最优 $\lambda$ 须附 selection-on-test 的 caveat；第二验证窗仍是可选加固项。
