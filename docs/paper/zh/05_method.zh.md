# 第 4 章 方法（中文打磨稿）

> 本文是论文 **§4（对应英文权威稿 `docs/paper/05_method.md`）** 的中文打磨稿。
> 本章结构必须让 §6 的消融矩阵（输入模式 × 观测锚定）**直接读出来**。

本章给出（i）经典锚定重建、（ii）物理匹配的合成训练平台、（iii）学习臂及其输入与锚定变体——
组织方式使 §6 的 input × anchor 矩阵能从本节结构直接对应。

## 4.1 经典锚

**Drizzle。** 把 248 帧对齐后做通量守恒的亚像素散射到 2x 网格；**最小假设**融合 baseline；它会暴露任何细网格方法都
必须管理的覆盖/lattice 伪影（5x 诊断网格上平均零覆盖率约 27%）。

**MAP-TV 去卷积锚。** 在全部帧上做 GPU 批量前向模型（shift → Gaussian PSF → 探测器 box → 下采样）；FISTA + 平滑 TV
梯度、150 次迭代；σ 与 λ 由 split-half NRMSE + artifact/std proxy 在 σ ∈ {0.2,…,0.5} × λ ∈ {3e-4, 1e-3, 3e-3}
上选出（选定 σ = 0.2 LR px、λ = 1e-3）。这是**采纳门控（acceptance gate）**：一个学习方法只有**同时**在 FRC band
一致性**和** contour-profile 指标上胜过该锚，才会被采纳。

**各向异性 coverage-weighted TGV。** raster 采集使数据约束各向异性（行内 X gap 1 vs 行间 Y gap ≈ 16），且双线性散射
把权重集中在固定 HR 行上，在各向同性 TGV 下产生水平条纹伪影。两处改动消除它：带 Y 轴正则比 1.5 的椭圆对偶球投影，
以及用 per-pixel 帧覆盖（而非帧数）归一化的数据梯度。效果：artifact score 3.870 → 0.695（−82%），
raw-control corr 0.902 → 0.916。**这是务实的经典交付物。**

## 4.2 物理匹配的合成平台（TCForge）

训练数据用**实测前向模型**合成，而非对抗退化：芯片状场景几何以 4× SSAA 覆盖抗锯齿渲染（软 [0,1] 覆盖掩膜，
使 HR 目标不含阶梯边、亚像素线携带覆盖缩放的幅值）；温度渲染 T_bg + ΔT · coverage；可信 σ 区间内的 Gaussian PSF；
探测器 box 积分；实测噪声底；以及重放真实 raster command lattice 的位移分布。Pool：1000 个场景（2x AA），
外加一个 burst pool（按帧存 LR stack + shift，每场景预算 K = 4 个 drizzle 变体用于输入模式训练；帧子集与 shift-noise
增广烘焙进变体；变体 0 是与推理匹配的 canonical 全 burst / 无噪配置）。

## 4.3 学习臂

**骨干（各臂固定）。** 朴素 UNet（base 64）+ 双线性 HR 头——PixelShuffle 头经归因测试后被否决（它加入条纹伪影、
在不减混叠的同时劣化 proxy；作为负结果报告）。**架构刻意保持不变**；实验变量是**输入信息通路**与**观测锚定**。

**Contour 取向 loss（conservative 配置）。** MSE 0.3 + 高通结构 0.8（σ = 5）+ SSIM 0.15 + 梯度向量匹配 0.15
（完整 (gx, gy) L1，捕捉只测幅值的边缘 loss 漏掉的膨胀/畸变）+ edge 0.05，外加温和的细结构（×3）与间隙（×2）加权；
早期的 hot 配置（结构 ×4、thin ×6）作为**漂移放大参照臂**保留。权重演化史（skeleton boost 30 → 振铃、loss cooldown A/B）
压成一段动机叙述。

**输入模式。**
- *1x 统计输入（baseline）*：1x 网格上 5 通道——对齐后的 mean / median / coverage / variance / 高通。这是常规特征化方式；
  我们证明它在网络看到之前就**坍塌**了 burst 的亚像素相位信息。
- *注入证据的 hybrid 输入*：从对齐 burst 在 2x 网格上渲染 3 个 drizzle 通道（散射 mean / coverage / 散射 variance），
  与 5 个上采样的 1x 通道拼接；网络以 scale 1 在 2x 网格上运行。**亚像素证据作为数据进入，而非作为学习先验。**【V9A】

**观测锚定（loss 侧）。**
- none / 窄带 forward consistency（重投影残差的高通带，权重 0.1）/ 全带 forward consistency（权重 0.1）——
  均把预测经实测算子重投影后与持有的 1x 观测比较。
- *hybrid 输入下的合法锚*：hybrid 输入的 channel 0 是上采样的 mean（**不是**合法的 1x 观测），故锚消费经数据管线单独
  携带的原始 1x 对齐-mean patch（偶数原点裁剪、增广同步）。【V9C】
- *residual-over-observation 输出（V10）*：网络在 2x hybrid 网格上预测 δ，最终输出 = `drizzle_mean_ch5 + δ`；
  一个 L1 残差惩罚 λ 控制输出能离观测域 drizzle 基多远。

**Checkpoint 选点（方法的一部分，非事后）。** 逐臂把 proxy 对（artifact score ↓、raw-control corr ↑）归一化到 [0,1]，
取最接近理想点的 3 个 step（≥ 5K step 间隔），始终携带末端 step 作漂移参照，并用视觉 panel（温度图，不只高通）把关
机械选择。**60K 端点永不作默认交付。**

> 待办（§4）：① 加 loss 公式块与选点规则 5 行伪代码（公式细节放 supp C，正文只放总式）；
> ② 把 V9A/V9C/V10 实现细节压成迁 LaTeX 前正文可读的一段。
