# 2024–2026 图像复原 / 超分 SOTA 综述（面向"保真优先的热红外多帧 SR"）

- **整理日期**: 2026-06-30 · **方法**: 3 路并行子代理 + 对抗式交叉核对（部分含二级子代理）。⚠️ = 单源/未评审预印本/年份未完全确认。
- **关联**: [BSRT](2022_bsrt_burst_sr.md) · [Restormer](2022_restormer.md) · [EP07 诊断](../episodes/ep07_solver_v8_k4_fullhalo_eval_archive/diagnosis_20260630/DIAGNOSIS.md) · `AGENTS.md` 硬教训 #6（先用 MTF/SNR 定边界）

---

## 0. 一句话定位

整个领域已**分叉成两条轨**：
- **保真轨（fidelity）**：transformer / CNN / state-space，按 PSNR/SSIM 优化——**计量/检测安全**。
- **感知-生成轨（perceptual）**：diffusion + GAN 先验，**合成**看着合理的细节——逼真分高，但**会编结构**。

对我们这种**低 SNR、PSF 受限、平坦内容、硬性禁止幻觉**的 LWIR 检测任务：**头条上所有令人兴奋的东西（StableSR/SUPIR/SeeSR/一步扩散热潮）都在错的一边。** 我们的 unrolled hard-DC solver **正好站在每个子领域的对的一边**。下面是证据和该怎么用。

---

## 1. 保真 vs 感知：这是对我们最重要的一条轴

理论根基：**Perception–Distortion Tradeoff**（Blau & Michaeli, CVPR 2018）——感知逼真度越过某界，失真**必然**上升。计量任务要的是**最小失真端**。

**指标纪律（关键，且可直接采纳）**：
- **保真/失真（可作训练目标）**：PSNR、SSIM、RMSE。
- **感知-有参考（奖励特征域逼真、容忍像素偏差）**：LPIPS、**DISTS**（对纹理容忍 → **对"编出来的纹理"是盲的**）。
- **无参考"逼真度"（最易被幻觉骗）**：NIQE、MUSIQ、MANIQA、CLIPIQA——**幻觉图能比真实图分还高**。生成式 SR 专门刷这些。

**最强的"幻觉看不见"证据（我们引用首选）**：
- **Hallucination Score**（arXiv:2507.14367, 2025 ⚠️）：证明生成式 SR 的幻觉**同时正交于** PSNR/SSIM/LPIPS **和** NR 指标——**标准 SR 指标在原理上检测不到编造的细节**。
- **PRISM**（arXiv:2603.14151, MIT, 2026 ⚠️）：科学图像复原，明确警告 SR **会幻觉出亚细胞结构**，"**更多复原不一定更好**"——强度定量要的是准确强度，不是好看。
- **STAR / Flux-Error**（arXiv:2507.16385, 2025 ⚠️）：天文 SR 引入**物理"通量误差"指标**和 **Flux-Invariant SR**——把保真定义为**守恒量一致性**。**若我们有守恒量（辐射/温度能量），这是绝佳模板。**
- **BayesDL-SIM**（Nat. Commun. 2025）：GAN 高频增益"部分来自不可靠的幻觉"，用**认知不确定度图**标记幻觉。

---

## 2. 四大家族快照（2024–2026）

**(a) Diffusion / 生成先验——发文最多，多为"感知/会编"**

| 方法 | 年/会 | 思路 | 取向 |
|---|---|---|---|
| StableSR | IJCV'24 | 冻结 SD2.1 + CFW 保真模块 | 感知(+保真旋钮) |
| DiffBIR | ECCV'24 | 先去退化再用 ControlNet 重生细节 | 生成 |
| **SUPIR** | CVPR'24 | SDXL 2.6B + LLaVA caption + 20M 图 | **最激进，臭名昭著爱编** |
| **ResShift / SinSR** | NeurIPS'23 / CVPR'24 | 残差移位、**无 T2I 先验** | **最偏保真的扩散** |
| OSEDiff / TSD-SR / FluxSR | NeurIPS'24 / CVPR'25 | 一步扩散 | 感知（OSEDiff 已知会幻觉） |
| PiSA-SR / OFTSR / CTSR | CVPR'25 | **可调保真↔感知旋钮**（免重训） | 可调 |

要点：这些方法之间**绝对 PSNR 只差 ~1–2 dB**，肉眼差别全在"编多少细节"。**任何建在自然图 T2I 先验上的（SD/SDXL/FLUX）都会合成"自然照片般但不在我们数据里"的结构 → 对 LWIR 是正确性失败。**

**(b) State-Space / Mamba——保真，效率挑战者**：MambaIR(ECCV'24) → **MambaIRv2**(CVPR'25, 单扫描省 ~50% 算力) / MaIR(CVPR'25, 0.88M tiny) / **EAMamba**(ICCV'25, 省 31–89% FLOPs)。**线性复杂度，但 selective scan 在小 patch 上对 GPU 不友好，优势只在大输入(>~1280px)显现**——我们若整帧解才划算。

**(c) Transformer/CNN 谱系——经典 SR & 复原的保真前沿**：SwinIR→Restormer→**HAT**→**DRCT-L(28.70, 大模型领跑)**/**ATD(28.17, 自适应 token 字典近线性全局)**；2025+ 有 PFT(28.20@~20M)、ATD-ext(TPAMI'26)。还有 **NAFNet**（无注意力 CNN，真实去噪/去模糊反超 transformer）、**FFTformer**（频域注意力，去模糊强）。

**(d) GAN 真实世界 SR——感知；不再是头条但仍是工具**：Real-ESRGAN/BSRGAN 仍是实用基线；对我们最相关的是**抗伪影分支** **LDL**(CVPR'22 惩罚 GAN 伪影) / **DeSRA**(ICML'23 事后检测移除伪影)。另：*Does Diffusion Beat GAN in SR?*(2024) 发现**同架构同算力下 GAN 与扩散相当甚至更好**——"扩散赢"部分是算力混淆。

---

## 3. "我们网络是不是太简单了"——SOTA 给的答案：**别追排行榜 dB**

- **保真 SOTA 已平台期**：经典 SR(Urban100×4) 五年总共才涨 ~1.25 dB，**近 18 个月 <0.1 dB**；真实去噪 SIDD 也卡在 ~40.0–40.4 dB。换 HAT/DRCT/Mamba 这种更花的 backbone，在自然图上只换**零点几 dB**，在我们**低纹理平坦热图上更少**。
- **2025 真正的信号是"混合"，不是新范式**：**NTIRE 2025 去噪冠军是 Restormer(transformer)+NAFNet(CNN) 混合**，前三名都是 transformer+CNN ensemble。没有单一架构通吃。
- **结论**：我们紧凑的 unrolled UNet 是**站得住的选择**；"太简单"基本是个**误判方向**——差距在**对齐 / 前向模型 / 正则 / 不确定度**，不在 backbone 容量。架构购物对我们是低 ROI。

---

## 4. 与我们最相关的四个小众（精炼版）

**A. Burst / 多帧 SR（BSRT 之后）**：2023–25 主流从"可变形卷积+transformer"转向 **Mamba/SSM 融合**（QMambaBSR CVPR'25；BurstMamba/MCA+MS-SSM 2025 ⚠️）。但 **SyntheticBurst 上 2022 的 BSRT-Large 43.62 仍压过所有已评审后继**（QMambaBSR 43.12 等都更低；更高的 44.51/49.22 是未评审预印本 ⚠️）。**关键洞察**：整族都靠"跨帧真·新亚像素采样"，对齐质量主导保真；**这族纯前馈、无数据一致性项——我们的 unrolled solver 已有它们缺的东西**。QMambaBSR 的洞察"亚像素内容空间一致、噪声不一致"对低 SNR LWIR 直接对口。

**B. Deep Unfolding / 物理约束（最像我们）**：统一范式 = **保留硬数据一致性步（梯度/FFT 闭式/CG）+ 只升级先验**（USRNet/DGUNet 谱系）。2024–26 进展：**Learned Proximal Networks**(ICLR'24, 学**精确近端算子**，给 PnP 收敛保证)；**PnP 在"先验分布不匹配"下收敛**(2026 ⚠️，正是真实部署情形)；**Poisson/KL-Bregman 数据项的镜像下降 DEQ**(2025 ⚠️，匹配**光子受限/低 SNR**噪声)；**FFT 正规算子近似**把数据一致性解做成 FFT(大规模/3D)。**我们的架构就是主流范式，升级方向是"先验 block"和"收敛鲁棒性"，不是重设计。**

> **B-补充 · Diffusion 当先验做反问题**：DDRM/ΠGDM/**DDNM**(值域-零空间分解：测得的分量解析钉死到数据，先验只填零空间)/DDS/ReSample/DAPS。**核心可迁移思想 = 值域/零空间原则**：凡前向算子测得的，钉死到数据；只让学习先验碰**真正没测到的零空间**，并把零空间内容**视为可疑**。⚠️ **InverseBench(2025)**：14 个 PnP-扩散求解器在 OOD 下**塌向先验、编出假结构**（如假黑洞环）、**用不上好的初始化**（输给物理基线）。→ **扩散先验不建议进我们的检测主线**；要更强先验就在现有 hard-DC 环里换**学习近端/去噪器**，不是后验采样。

**C. 热红外 / LWIR SR**：小众，绕着 **PBVS TISR 挑战赛**（CVPR workshop；T1 单图 ×8，T2 RGB 引导 ×8/×16）。
- **能用的**：长程上下文（IR 频谱低频主导，全局结构>局部纹理）；**频域/小波监督**（分离热对比低频与弱高频）；**真实/物理退化建模**（真实 PSF/散焦，闭最大 sim-to-real gap）；**物理/温度一致性损失**（PCNet 2026 ⚠️ 的热传导+温度一致性；Real-IISR 的温度-强度单调一致）。
- **会失败的**：**GAN 在平坦场幻觉假纹理**；**RGB 引导"纹理拷贝/渗透"**（可见边被当结构拷进来，热图强度常与结构边不对齐 → 更糟）；**bicubic 训练→真实崩**；**PSNR/SSIM 抓不到幻觉**。
- **微扫描多帧**：物理最诚实的一支，但**几乎不在深度 PBVS 主线里**（多为经典 TV+交替梯度，Applied Optics 2026 ⚠️）。**我们的微扫描路线是采"真·新亚像素信息"、差异化且物理诚实的。**

**D. 科学/显微保真 SR**：根基警告——**Antun(PNAS'20)** 深度重建不稳定、小真实结构会漏；**Bhadra/Anastasio(TMI'21)** 幻觉=先验填的**零空间假结构**，提出**测量空间幻觉图**(无需 GT)。五条收敛的保真策略：① 值域/零空间数据一致性；② **物理/系统嵌入**重建(把 PSF/前向模型写进 loss/架构)；③ **校准的不确定度**(BayesDL 异方差+认知；**SURE 校准的 conformal**无需 GT)；④ **幻觉检测/审计**(sFRC 扫描傅里叶环相关，定位过全局指标的幻觉)；⑤ 理论(*Looks Too Good To Be True* 证明 uncertainty–perception tradeoff → 计量就该选保真)。

---

## 5. 对 thermal_lift 的结论与行动（按性价比排序）

**保留（被每个子领域共识背书，别拿去换排行榜）**：unrolled **硬数据一致性** + **物理嵌入** + **保真优先于逼真**。

**高性价比、低风险的添加（建议先做）**：
1. **真实前向退化建模**：合成训练用**真实 LWIR PSF(σ≈0.5px) + 传感器/固定图案噪声(噪声底 0.0724°C)**，不是 bicubic。（对照 `research_log/synthetic_data_realism.md`，确认已覆盖到什么程度。）
2. **用已知微扫描偏移替代/约束学习对齐**：BSRT 把大半容量花在学对齐去抗误配；我们 stage command + EP04 给了**已知亚像素偏移**，直接用可**移除一个主要幻觉源**（误 warp）。
3. **频域/小波 + 温度一致性正则**：贴合"分片平滑+硬边界"的内容；温度-强度单调/守恒量(辐射能量)一致性当物理正则。
4. **保真审计与不确定度（评测侧，便宜且可发表）**：① **测量空间幻觉审计**（输出经前向算子回投，比对其测量分量，无需 GT，给检测任务"无编造细节"背书）；② **守恒量检查**（STAR Flux-Error 的热辐射类比：SR 前后区域/总测量能量是否守恒）；③ **逐像素校准不确定度图**（异方差头或 SURE-conformal），高认知不确定度=低 SNR 分布漂移下的实用幻觉探测器。**这些 PSNR 都给不了。**

**可考虑的"先验 block"升级（保持保真前提）**：
5. **精确学习近端(LPN)** 替换当前 prox → 收敛保证 + 更强先验，不破坏 hard-DC。
6. **高效全局上下文**（Mamba 局部扫描 / 窗口注意力 / Restormer 风格通道注意力）——**但必须"训练即推理尺度"或窗口化**，否则复刻 EP07 的 extent-shift 网格/絮状（见 [Restormer 笔记](2022_restormer.md) §3 与 EP07 诊断）。换上后用 `diag_extent.py` 远场扰动测试验证耦合是否归零。
7. **若为散粒噪声主导**：把 L2 数据项换成 **Poisson/KL-Bregman** 数据项（镜像下降 DEQ 思路）。

**避免（明确的坑）**：扩散/GAN 自然图先验（幻觉，且对 PSNR/LPIPS/NR 指标都不可见）；RGB 跨谱引导的纹理渗透；为零点几 dB 去换更大 backbone。

---

## 6. Sources（精选，⚠️=单源/未评审/年份待确）
- 轴线/幻觉：Perception–Distortion (1711.06077) · Hallucination Score 2507.14367 ⚠️ · PRISM 2603.14151 ⚠️ · STAR/Flux 2507.16385 ⚠️ · BayesDL nat.commun s41467-025-60093-w · Bhadra/Anastasio TMI'21 (PMC8673588) · Antun PNAS'20 (PMC7720232)
- 扩散/感知 SR：StableSR 2305.07015 · DiffBIR 2308.15070 · SUPIR 2401.13627 · ResShift 2307.12348 · SinSR 2311.14760 · OSEDiff 2406.08177 · PiSA-SR 2412.03017
- Mamba：MambaIR 2402.15648 · MambaIRv2 2411.15269 · MaIR 2412.20066 · EAMamba 2506.22246
- Transformer/CNN：Restormer 2111.09881 · NAFNet 2204.04676 · HAT 2309.05239 · DRCT 2404.00722 · ATD 2401.08209 · FFTformer 2211.12250 · NTIRE'25 去噪 2504.12276 · CascadedGaze TMLR'24 2401.15235
- GAN/抗伪影：Real-ESRGAN 2107.10833 · LDL 2203.09195 · DeSRA 2307.02457 · Does Diffusion Beat GAN 2405.17261
- Burst：BSRT 2204.08332 · BIPNet 2110.03680 · Burstormer 2304.01194 · GMTNet(CVPR'23) · RBSR 2306.17595 · BurstM 2409.15384 · QMambaBSR 2408.08665 · BurstMamba 2503.19634 ⚠️
- Unfolding/扩散先验：USRNet 2003.10428 · DGUNet 2204.13348 · LPN 2310.14344 · PnP-mismatch 2601.09831 ⚠️ · DDNM 2212.00490 · DDS 2303.05754 · ReSample 2307.08123 · DAPS 2407.01521 · InverseBench 2503.11043 ⚠️ · Hallucination Index(Med.Phys'24) 2407.12780
- 热红外：DifIISR(CVPR'25) 2503.01187 · SwinFuSR(PBVS'24) 2404.14533 · PCNet 2601.03526 ⚠️ · Real-IISR 2603.04745 ⚠️ · IRSRMamba 2405.09873 ⚠️ · PBVS TISR challenge (pbvs-workshop.github.io) · microscan MWIR TV+AG (Applied Optics'26) ⚠️
- 科学保真：Trustworthy SR/Generative-Pseudoinverse 2505.12375 · sFRC 2603.04673 ⚠️ · Looks Too Good To Be True 2405.16475 · Self-supervised conformal 2502.05127
