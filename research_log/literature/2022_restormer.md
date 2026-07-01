# Restormer — Efficient Transformer for High-Resolution Image Restoration

- **论文**: arXiv:2111.09881 · *Restormer: Efficient Transformer for High-Resolution Image Restoration* · Zamir 等 (IIAI/MBZUAI/Google) · **CVPR 2022**
- **学习日期**: 2026-06-30 · **为什么读**: 用户点名"很强的论文"，想看它的结构对我们 prox 有无借鉴
- **关联**: [BSRT 笔记](2022_bsrt_burst_sr.md) · [EP07 solver 诊断](../episodes/ep07_solver_v8_k4_fullhalo_eval_archive/diagnosis_20260630/DIAGNOSIS.md)（关键：我们的 GroupNorm/SE 全局算子造成 extent-shift 伪影）

---

## 1. 忠实摘要（核对自 PDF）

| 项 | Restormer |
|---|---|
| 任务 | **通用图像复原**（不是 SR）：去雨 / 运动去模糊 / 去散焦 / 去噪（高斯灰度+彩色+真实）。16 个基准 |
| 输入/输出 | **单图**，单次前向；预测**残差** Î = I + R。无多帧、无对齐模块 |
| 架构 | 4 级 U 形多尺度 **Transformer**；下/上采样用 pixel-unshuffle/shuffle；LayerNorm；bias-free conv |
| 深度 | 每级 block=[4,6,6,8]，head=[1,2,4,8]，通道=[48,96,192,384]，+4 block 全分辨率 refinement |
| 参数/算力 | **≈26.1M / ≈141 GFLOPs@256²**（比 IPT ~115M 少 4.4×；比 SwinIR 快 13×、FLOPs 少 3.14×） |
| 损失 | **仅 L1**。无 GAN/感知/扩散 → 纯**保真**复原 |
| 关键结果 | 真实去噪 SIDD **40.02** / DND **40.03**（首个双双破 40）；去雨平均 +1.05dB；GoPro 去模糊 32.92 |

**两大创新**：
- **MDTA（Multi-Dconv head Transposed Attention）**：把自注意力放在**通道维**而非空间维。把 Q、K reshape 后点积得到 **Ĉ×Ĉ 的"通道协方差"注意力图**（而非 (HW)×(HW) 的巨型空间图）→ **复杂度对图像尺寸线性**（普通空间注意力是 O(W²H²)）。Q/K/V 先经 1×1 逐点卷积（跨通道混合）+ 3×3 深度卷积（逐通道空间上下文）。
- **GDFN（Gated-Dconv FFN）**：两条 1×1→3×3-dconv 支路逐元素相乘、一条用 GELU 门控，控制哪些特征往下传；扩张比降到 γ=2.66 保持算力。

**待核实**：总参数量正文未给单一数字（≈26.1M 由消融表 Table 8 + "比 IPT 少 4.4×"推得）；无专门 Limitations 节。

---

## 2. 输入模式：和我们一样是"单图先验"

Restormer 是**单图、单次前向、预测残差**的复原器——**没有多帧融合、没有对齐**。所以它扮演的角色就是我们 unrolled solver 里 **prox（去噪/先验）那一块**。这和 BSRT（多帧全进网络、学对齐+融合）是**互补的两类工作**：
- BSRT = 怎么把多帧信息**喂进/融合进**网络（对齐+融合）。
- Restormer = 单张图里**一个高效的先验/复原 block** 怎么搭（MDTA+GDFN）。

对我们：**MDTA+GDFN 是"prox block 升级"的候选**，因为它在**纯保真 L1**下做到全局上下文且算力线性——正是我们想要的（长程 PSF/微扫描结构 + 不幻觉）。

---

## 3. ⚠️ 关键警示：MDTA 仍是"全局空间归约"，对我们有 EP07 同款风险

不要天真地把 MDTA 当成"尺度无关的全局上下文"来救 EP07 的网格/絮状。**MDTA 的通道协方差是对所有空间位置求和**（Σ over HW 的 q·k）→ 这**仍是一次全局空间归约**，和我们诊断出的 **SE 全局池化 / GroupNorm 全局归一化同类**——都让输出依赖"解多大一块、里面有什么"，因此**同样会有 extent-shift**（192-tile vs 整帧推理不一致 → 网格/絮状的根源，见 EP07 诊断 §3）。

结论与可操作判据：
- **有用的部分**：GDFN 的门控 + 深度卷积、bias-free conv、残差、L1 保真——这些是**局部、尺度无关**的，可以借鉴。
- **要小心的部分**：任何**全局归约**（MDTA 的通道注意力、SE、GroupNorm）若要用，**必须"训练即推理尺度"或窗口化**，否则会复刻 EP07 的伪影。可直接用 `diagnosis_20260630/diag_extent.py` 的远场扰动测试验证：换上 MDTA block 后远场耦合是否仍 ≫0。
- **替代**：若只想要"线性复杂度的长程上下文"，**窗口化注意力 / 局部 Mamba 扫描**比"对整帧求和"的通道协方差更 extent-friendly。

---

## 4. 对我们的启示（一句话）

Restormer 给的不是"上 transformer"，而是**两个具体可借鉴件**（GDFN 门控前馈 + L1 纯保真）和**一个要避坑的点**（任何全局归约都要按 EP07 的 extent 教训处理）。它本身是单图先验，**不解决我们的多帧/对齐问题**（那是 BSRT 的地盘）。
