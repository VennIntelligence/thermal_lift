# Supplementary A — 理论与推导（中文草稿）

> **角色**: technical appendix 的理论块，支撑主文 §3（forward model / claim boundary）与 §5（无 GT 评估协议）。
> **语言**: 按 2026-06-12 决策先以中文成稿；迁 LaTeX 时翻译为英文（项目约定 supp 终稿英文）。
> **占位规范**: 未落地内容标 ⬜ 并注明「等什么」；数字全部带仓库出处，TB-scale 与 EP11-harness-scale 绝不混表。
> **对应清单**: `10_writing_handover.md` §3.A（A.1–A.5 全部 ★ 条目）。

---

## A.1 MTF/SNR 可行性分析（对应主文 §3.3 末段、§1 Q1 边界）

### A.1.1 系统 MTF 闭式

光学 PSF 建模为各向同性 Gaussian（σ 单位 LR px，1 LR px = 10 µm）：

- **Gaussian PSF MTF**：MTF_G(f) = exp(−2π²σ²f²)，f 单位 cyc/LR px（`core/src/thermal_core/ep03.py` L61–68）。
- **10 µm 像素孔径 MTF**：MTF_det(f) = |sinc(W·f)|，W = 10 µm，f 单位 µm⁻¹（`algos/ep15_info_limit/scripts/run_m3_sigma_arbitration.py` L545–551）。
- **系统 MTF**（EP15 M3 的 FRC 形状拟合用其平方）：MTF(f)² = [exp(−2π²σ_µm²f²)·|sinc(10f)|]²，σ_µm = σ_LR × 10 µm（`run_m3_sigma_arbitration.py` L868）。

**口径注意**：EP03 正式 MTF 数值表**仅含 Gaussian PSF 项、未乘 detector sinc**（`reports/ep03_theoretical_limits/theoretical_limits_report.md` §3）；EP15 M3 拟合用的是含 sinc 的系统 MTF²。supp 终稿两处都要写明各自口径，不能混用同一张表。

### A.1.2 网格与 Nyquist

| 量 | 数值 | 出处 |
|---|---|---|
| Detector pitch | 10 µm/px（FOV 6.4×4.8 mm） | `theoretical_limits_report.md` L20–22 |
| 已校准空间分辨率 | 20 µm | 同上 L22 |
| 2x 输出采样 / Nyquist 周期 | 5 µm / **10 µm** | 同上 L30–38 |
| 4x 输出采样 / Nyquist 周期 | 2.5 µm / 5 µm | 同上 |
| 1x/2x/4x Nyquist 频率 | 0.5 / 1.0 / 2.0 cyc/detector px | 同上 L45；`ep03.py` L158 |

### A.1.3 MTF 数值表（Gaussian-only 口径，已核对）

| 网格 | f (cyc/px) | σ=0.20 | σ=0.35 | σ=0.50 |
|---|---:|---:|---:|---:|
| 1x Nyquist | 0.5 | 0.821 | 0.546 | 0.291 |
| **2x Nyquist** | **1.0** | **0.454** | **0.089** | **0.007** |
| **4x Nyquist** | **2.0** | **0.042** | 6.3×10⁻⁵ | ~3×10⁻⁹ |

出处：`theoretical_limits_report.md` L47–51；CSV `output/ep03_theoretical_limits/mtf_psf_attenuation.csv`（output 不入 Git，由 `scripts/build_ep03_cache.py` 重建）。

### A.1.4 有效 SNR 判据

**判据**：SNR_eff = ΔT · MTF(f, σ) / σ_n，σ_n = 0.0724 °C（smooth adjacent-coordinate MAE；`configs/noise_floor.json`、`theoretical_limits_report.md` L80）。

**实测对比度 ΔT**（`theoretical_limits_report.md` L84–90）：

| 标度 | ΔT (°C) | 输入 SNR |
|---|---:|---:|
| 噪声底 | 0.0724 | 1.0 |
| 3× 噪声门 | 0.2172 | 3.0 |
| 名义边缘 | 0.70 | 9.7 |
| 内轮廓中位 | 1.9385 | 26.8 |
| 外轮廓中位 | 2.4901 | 34.4 |

**Nyquist 处 SNR_eff**（`theoretical_limits_report.md` L66–70）：

| 对比度 | 2x（σ=0.20/0.35/0.50） | 4x（σ=0.20/0.35/0.50） |
|---|---|---|
| 名义 0.7 °C | 4.39 / 0.86 / 0.07 | 0.41 / 0.001 / ~0 |
| 内轮廓 1.94 °C | 12.16 / 2.39 / 0.19 | 1.14 / 0.002 / ~0 |
| 外轮廓 2.49 °C | 15.62 / 3.06 / 0.25 | 1.46 / 0.002 / ~0 |

风险带门控（`ep03.py` L207–214）：SNR_eff ≥ 5 observable；≥ 3 borderline；≥ 1 weak；否则 noise-dominated。

### A.1.5 判决与互证

- **2x 条件可行**：乐观 σ 端有可用频率余量；保守 σ=0.5 时高频强衰减 → POC 必须定位在 contour-level（`theoretical_limits_report.md` L55–56）。
- **4x 出界**：除最乐观 σ=0.2 外 SNR_eff 全线 noise-dominated；4x 网格仅作 contour oversampling/可视化（L56–57, L159–168）。
- **与 EP12 实证互证**：4x 网络无真实增益（supp D.4.2），与 MTF ≤ 0.042 界一致——理论界先于实验设定了风险预期。
- **边界声明**：SNR_eff 过门是**必要条件，非 SR 成功证明**；判据忽略 alignment 误差、热漂移与模型失配（L72–76）。

### A.1.6 CRB/ESF 定位界（辅助证据）

| ΔT | σ_PSF | 1 帧 CRB | 16 帧 CRB |
|---:|---:|---:|---:|
| 0.7 °C | 0.5 px | 0.1415 px | 0.0344 px |
| 0.7 °C | 1.0 px | 0.1947 px | 0.0487 px |
| 2.0 °C | 0.5 px | 0.0495 px | 0.0120 px |
| 2.0 °C | 1.0 px | 0.0682 px | 0.0170 px |

出处：`theoretical_limits_report.md` L108–115。口径：CRB 是乐观理论下界，不替代真实 alignment 误差（L117–121, L153）；EP04 实测 A-class split-half ~0.027 px（supp B.2）与 16 帧 CRB 同量级，作 consistency check。

**A.1 资产依赖**：`reports/ep03_theoretical_limits/theoretical_limits_report.md`（权威）；`core/src/thermal_core/ep03.py`（公式实现）；`output/ep03_theoretical_limits/{mtf_psf_attenuation.csv, mtf_snr_recoverability.csv, mtf_psf_frequency_response.png, mtf_snr_recoverability_heatmap.png}`（重建：`uv run python scripts/build_ep03_cache.py`）。
**A.1 待回填**：无——素材齐全。⬜ 仅终稿排版时决定是否将 MTF 频响曲线重绘为 supp 图（S 系列，CVPR 风格）。

---

## A.2 观测算子与零空间（对应主文 §3.4、§5.3）

### A.2.1 离散观测算子

主文记号：y_k = D·B·H·S_{t_k}·x + n_k，x 为 2x 网格 HR 图像。各因子：

- **S_{t_k}**（亚像素平移）：以 contour-refined 位移 t_k 对 HR 网格重采样；
- **H**（光学 PSF）：Gaussian 卷积，σ ∈ [0.2, 0.5] LR px（supp A.5.3）；
- **B**（探测器孔径）：10 µm box 积分；
- **D**（降采）：HR→LR 网格抽取（scale=2）；
- **n_k**：噪声底 0.0724 °C。

### A.2.2 代码实现对照（写终稿时必须如实交代的 gap）

仓库内有两套实现，离散化细节不同：

| 论文符号 | EP06 `forward()`（`algos/ep06_sr_poc/src/common/forward_model.py`） | EP07 训练 loss 链（`algos/ep07_unet_sr/src/unet_sr/losses.py::forward_model_loss`） |
|---|---|---|
| S | `map_coordinates` 双线性，`order=1, prefilter=False` | 隐含于 obs 构造侧（对齐已完成，loss 不再显式 shift） |
| H | `gaussian_filter`，σ_hr = σ_lr × scale | `gaussian_blur_2d`，σ_hr = σ_lr × scale（CLI 默认 σ_lr=0.5） |
| B | **未显式实现**（被 PSF 吸收近似） | `F.avg_pool2d(kernel=s, stride=s)` 块均值近似 B·D |
| D | LR 采样坐标隐式给出 | 同上 avg_pool 显式完成 |

EP07 链还支持带限版本：band="highpass" 时对两侧同减 Gaussian(σ_band=5 LR px) 低通后再 MSE（`losses.py`）。**终稿措辞**：零空间论证绑定的是「训练时实际生效的算子」即 EP07 链（blur → block-average → 可选带限），EP06 矩阵式 forward/adjoint 用于经典重建（MAP-TV/TGV）。两者共享 H 与 B·D 的核心低通-折叠结构，零空间论证对两者同样成立。

### A.2.3 零空间刻画

A = D·B·H·S 的零空间来自两个机制的叠加：

1. **带限衰减**：H、B 都是低通——Gaussian MTF 在 2x Nyquist 处最低只剩 0.007（σ=0.5，A.1.3），加上 box sinc，HR 网格上超过截止的高频分量经 A 映射后衰减到噪声以下，**数值上**落入 ε-零空间；
2. **混叠折叠**：D 把 HR 频谱折叠到 LR Nyquist 内，存在整族高频扰动 δx 使折叠后各别名分量相消，**严格** A·δx = 0。

直观结论：「细于截止周期的结构改动」与「别名相消组合」对所有 1x 观测不可见。多帧相位多样性缩小但不消灭该零空间（25 个相位 bin 的覆盖见 supp A.4.1/B.2.5）。

### A.2.4 Proposition 1（观测域 loss 对零空间恒盲）

**命题**：设 δx_null 满足 A·δx_null = 0，L(x̂) = ℓ(A·x̂, y) 为任意只经 A 接触预测的观测域损失（ℓ 可微）。则对任意 x̂：L(x̂ + δx_null) = ℓ(A·x̂ + A·δx_null, y) = ℓ(A·x̂, y) = L(x̂)，且 ∇L 沿 δx_null 方向分量恒为 0。∎

**推论**：把 forward-consistency 权重加大、换 band（highpass/full）、换残差范数，都不改变对 δx_null 的不可见性——这是 §6.2 中 V9B/V9D 与无锚 v8.1a 漂移曲线重合的机制解释（数字见 supp D.4.3）。

### A.2.5 两曲线诊断的充分条件

**陈述**：若训练过程中 (i) forward loss 已贴其噪声地板（再无可下降空间），且 (ii) 真实数据 proxy（artifact/corr，supp A.3）持续单调漂移，则漂移方向的观测域投影 ≈ 0，即漂移主要发生在 A 的（ε-）零空间内。

**实测实例**（TB-scale）：v9b 的 `loss/forward_model` 自 10K 步起平坦于 0.004–0.009，同期 artifact 0.37→0.65 单调上爬（`research_log/algorithm_changelog.md` ACL-017；曲线 CSV `output/ep11_dl_benchmark/checkpoint_selection/forward_loss_curves.csv`）。

### A.2.6 ⬜ 零空间投影的直接测量（stretch goal）

**做法**（已规划未执行）：取同臂相邻 checkpoint 预测差 δx̂ = x̂_(k+1) − x̂_k，用 EP06 算子计算 range 分量 δx_range = A†A·δx̂（A† 用共轭梯度近似），报告 ‖δx_range‖/‖δx̂‖ 随训练步数的变化。若该比值随漂移期趋近 0，则把 §5.3 从「两曲线间接诊断」升级为「直接测量」。
**状态**：⬜ 待跑——GPU-light（每对 checkpoint 数分钟），可在 V9C/V10 训练间隙执行；等执行后回填数字与图。

**A.2 资产依赖**：`algos/ep06_sr_poc/src/common/forward_model.py`；`algos/ep07_unet_sr/src/unet_sr/losses.py`；`docs/paper/04_problem_forward_model.md` §3.4；`output/ep11_dl_benchmark/checkpoint_selection/forward_loss_curves.csv`。
**A.2 待回填**：⬜ A.2.6 投影实验（等 GPU 空隙 + 实现脚本）；⬜ Proposition 记号与主文 §3.4/§5.3 终稿统一（迁 LaTeX 时）。

---

## A.3 proxy 反相关的构造性论证（对应主文 §5.2）

### A.3.1 指标精确定义

**共享预处理**：SR 输出温度图 → highpass 域 u = x̂ − G_{σ_bg}(x̂)，σ_bg = 5 HR px（`tcforge/src/tcforge/highpass.py::highpass_preprocess`）；控制图 c = highpass(bicubic↑2(nanmean(248 帧 raw)))（`algos/ep07_unet_sr/src/unet_sr/real_eval.py`）。

**TB-scale artifact score**（训练期 eval、F3/F4 轨迹图口径）：

```
high_freq = u − G_{σ=1}(u);  lap = ∇²u
artifact = (std(high_freq) + 0.25·std(lap)) / std(u)
```

即「u 内部更高频能量占比」（`real_eval.py`）。

**EP11-harness artifact score**（横评表口径，`algos/ep06_sr_poc/src/common/metrics.py`）：

```
ringing    = median(|∇²u|) / median(|∇u|)
blockiness = median(scale 边界差) / median(内部差)
artifact   = ringing + 0.25·blockiness (+ 5·overshoot，EP11 调用未传 lr_img 故为 0)
```

**raw-control corr**（两套口径同一实现）：corr = Pearson(u, c)，全图 finite 像素（`real_eval.py::pearson_finite`；EP11 `_pearson_finite` 相同）。

**红线**：两套 artifact 定义数值不可比，绝不混在同一表/图（主文 §6 头注）；EP10 TGV 报告里的 artifact 是温度域、控制为单帧 bicubic，跨算法引用时需注明（`algos/ep10_tgv_sr/`）。

### A.3.2 共享残差结构

两个 proxy 都是**同一张 highpass 图 u** 的泛函：artifact 度量 u 的内部高频能量，corr 度量 u 与固定观测锚 c 的线性一致性。它们不是独立证据源——这是「反相关 by construction」论证的前提。

### A.3.3 一阶符号相反

考虑「合成先验风格化」扰动方向 δu：在结构边缘处把响应增亮/增宽（典型漂移形态，见 D.3 视觉档案）。则：

1. δu 集中于边缘高频 → std(high_freq)、|∇²u| 增大快于分母 std(u)、|∇u| → **artifact 一阶上升**（两套定义同号）；
2. c 固定且 δu 不源自观测（零空间方向，A.2.4），u 在 c 的正交补内增能量 → Pearson 分母 ‖u‖ 增大而分子 ⟨u,c⟩ 近似不变 → **corr 一阶下降**。

故沿该轴二者符号恒反；联合最大化不可行。

### A.3.4 三条推论（主文 §5.2 的依据）

1. proxy 对是**漂移温度计 + 选点准则**，不是两个可联合优化的分数；
2. **跨 input-mode 不可横比**：证据注入输入（hybrid drizzle）合法携带更多高频，artifact 基线天然更高（`docs/next_move_plan.md` §3 V9A vs v8.1a 对照）；
3. **看轨迹不看端点**：单点数值无意义，训练时间轴上的走向才是信息（F3/F4）。

**A.3 资产依赖**：`algos/ep07_unet_sr/src/unet_sr/real_eval.py`；`algos/ep06_sr_poc/src/common/metrics.py`；`tcforge/src/tcforge/highpass.py`；`docs/paper/06_evaluation_protocol.md` §5.2。
**A.3 待回填**：无——可成稿。⬜ 终稿可选：用一组数值扰动实验（对 u 加参数化边缘增亮）画出两 proxy 的符号相反响应曲线作 supp 小图（CPU 几分钟，非必需）。

---

## A.4 FRC 方法学（对应主文 §5.1）

### A.4.1 phase-stratified split-half 构造

全部实现在 `algos/ep15_info_limit/scripts/run_m2_frc.py`：

| 步骤 | 设置 |
|---|---|
| 输入 | 248 clean 帧（脚本强制校验），`offset_correction(method="median")` |
| 相位 bin | stage command (X,Y) → `coordinate_to_shift` → (dx,dy)；phase = mod(shift,1)，bin = floor(phase·5)，5×5=**25 bins**；每 bin 7–13 帧（`command_phase_bin_counts.csv`） |
| 分半 | 每 bin 内随机置换、奇偶交替分给 A/B，约束 |#A−#B| ≤ 1 → 相位分布在两半中保持 |
| 重建 | A/B 各自 bilinear drizzle 到 **5x 诊断网格**（hr_pitch 2 µm），空 bin 填全局均值 |
| 重复 | seeds {42, 123, 456}，主曲线 = 3 条曲线逐 ring nanmean |
| 窗口 | Tukey α=0.25，边缘 crop 16 LR px |

### A.4.2 FRC 公式与判据

FRC(ring) = Σ Re(F_A·F̄_B) / √(Σ|F_A|²·Σ|F_B|²)，ring 宽 1 个频率 bin（df = 1/(min(H,W)·2 µm)）。

| 判据 | 公式 |
|---|---|
| 1/7 | 常数 0.142857 |
| half-bit | (0.2071 + 1.9102/√n_ring)/(1.2071 + 0.9102/√n_ring)，n_ring 为环内像素数 |

cutoff = 第一个 FRC < 阈值的 ring 对应周期（`find_cutoff`）。

### A.4.3 主结果

- **1/7 cutoff = 17.03 µm**（3-seed 均值曲线；half-bit 判据给出相同值）；逐 seed 16.17/16.17/17.03，std **0.50 µm**（`frc_repeats.csv`、`frc_summary.json`）。
- 主张口径：**只主张 17.0 µm**——超过 20 µm 分辨率的相干信息存在，但低于 11–14 µm 的理论期望。

### A.4.4 四个控制组：构造与失效模式

| 控制组 | 构造 | 预期 | 实测 | 失效模式说明 |
|---|---|---|---|---|
| 正控制 bicubic | 单帧 bicubic↑ 两副本各加 N(0, 0.0724) | cutoff 应明显差于 main | cutoff **13.58 µm**（不差于 main） | 单帧插值的平滑谱在低噪声下自相关偏高 → 正控制**未通过**，如实披露 |
| 负控制 shift-shuffle | A 图 vs「B 帧 + 置换 B shifts」drizzle | 高频 FRC 应塌缩 | 8–12 µm 中位 FRC **0.504** | 置换后仍共享场景低频与网格结构 → 负控制**部分失效** |
| 漂移控制 acquisition-half | 按采集序前半 vs 后半各自重建 | 热漂移应拉低一致性 | cutoff 退化到 **26.20 µm** | 行为符合预期：时间间隔放大漂移 → 支撑「分层 split 必须保相位与时间混合」 |
| zero-coverage 统计 | 每重建的空 bin 占比 | — | 均值 **27.2%**，最大 36.2% | 5x 网格欠覆盖 → 高频 ring 信噪受 coverage/lattice 污染 |

### A.4.5 10–12 µm 反弹的处理（红线）

band 表（`frc_band_table.csv` 全表）：

| 周期 (µm) | main | bicubic 正控 | shuffle 负控 | drift 控 |
|---:|---:|---:|---:|---:|
| 20 | 0.348 | 0.952 | −0.153 | 0.125 |
| 16 | 0.138 | 0.691 | −0.012 | 0.117 |
| 14 | 0.098 | 0.232 | −0.134 | 0.082 |
| 12 | 0.593 | 0.002 | −0.095 | 0.578 |
| 11 | 0.877 | −0.003 | 0.477 | 0.832 |
| 10 | 0.935 | 0.012 | 0.906 | 0.887 |
| 9 | 0.816 | 0.003 | 0.422 | 0.802 |
| 8 | 0.545 | −0.023 | 0.390 | 0.345 |

判读：10–12 µm 高 FRC 同时出现在负控制（10 µm 0.906）与漂移控制（0.887）中 → 反弹由 coverage/lattice + 漂移驱动，**不作分辨率证据**，仅作风险标注（`01_outline.md` 禁写边界）。

### A.4.6 MAP-TV 前后 split-half 对照（与 D.2 共享）

20/16/14/12/10 µm：bare drizzle 0.319/0.088/0.053/0.575/0.893 → MAP-TV 0.976/0.965/0.955/0.947/0.934（`output/ep15_info_limit/m4_deconv_anchor/`）。口径：提升的是 split-half **一致性/稳定性**，不是光学分辨率证明。

**A.4 资产依赖**：`algos/ep15_info_limit/scripts/run_m2_frc.py`；`output/ep15_info_limit/m2_frc/{frc_curve.csv/.png, frc_controls.csv/.png, frc_band_table.csv, frc_repeats.csv, frc_summary.json, command_phase_bin_counts.csv}`；F2 主图资产同源。
**A.4 待回填**：无——可成稿。⬜ 终稿排版：F2 重绘脚本（`scripts/paper_figures/fig02_frc.py`，待写，见 E.2）。

---

## A.5 标定不确定度传播（对应主文 §3.3）

### A.5.1 θ 不确定度 → 位移误差（闭式）

坐标→位移映射（`core/src/thermal_core/displacement.py`）：dx = (X cosθ + Y sinθ)/10，dy = (−X sinθ + Y cosθ)/10（µm→px）。对 θ 求导，纯 X 步进 X=40 µm、δθ=0.1°=1.745 mrad：

- δdx ≈ |X sinθ|·δθ/10 ≈ 0.0052 px；δdy ≈ |X cosθ|·δθ/10 ≈ 0.0047 px；合成 ≈ **0.007 px**。
- 位移**幅值** |s| = X/10 = 4.0 px 与 θ 一阶无关——θ 误差表现为方向误差，不是步长误差。
- 结论：θ=47.6°±0.1°（`configs/stage_calibration.json`）的传播误差比 alignment 残差（Chamfer 0.134 px，supp B.2）低一个量级，不是误差预算主项。

### A.5.2 AVI 独立方向验证的角色与边界

gradient-NCC 合并估计 θ = 47.14°，中位 47.11°，95% CI [46.36°, 47.92°] 覆盖 47.6°（16 个 AVI；`reports/ep02_displacement_calibration/calibration_report.md` L73–80）。X/Y 子组系统差 ~3°（X 48.70° vs Y 45.63°）→ 只作 consistency check，**不替换配置**；AVI 为 8-bit 渲染视频，不具标定精度（`AGENTS.md` 硬教训 13）。

### A.5.3 PSF σ 区间的影响范围

三路标定分歧（`reports/ep09_psf_calibration/psf_calibration_report.md`）：

| 路线 | σ (LR px) | 95% CI | N | 角色 |
|---|---:|---|---:|---|
| A forward 残差 | 0.226 | [0.208, 0.240] | 248 | primary |
| B ESF 拟合 | 1.129 | [1.041, 1.215] | 33 | cross-check |
| C joint hold-out | 0.119 | n/a | 32 | cross-check |

Route spread 1.01 px > 0.05 px 容差 → EP09 gate **不通过**，σ 状态 provisional（`configs/psf_calibration.json`）。EP15 M3 仲裁（`research_log/episodes/ep15_info_limit/README.md` M3）：

- 多边缘 ESF：外边框中位 σ_total 1.015 / 内部强边缘 0.747 / 最陡边缘 0.888 / 最锐单边缘上界 **0.546 LR px**；
- FRC 形状拟合（系统 MTF²，12–80 µm 带）：σ=0.2 最优（MSE 0.044, corr 0.931），0.3→1.0 单调变差；
- **采纳区间 σ ∈ [0.2, 0.5] LR px**；所有 PSF 依赖计算要么扫区间要么显式报告 σ。

传播后果：2x Nyquist MTF 在区间内变化 0.454→0.007（65 倍，A.1.3）→ 反卷积强度必须把 σ 当扫描参数（M4 σ 网格 {0.2,0.3,0.4,0.5}，选 0.2；supp C.4.3）。

### A.5.4 ESF 表观宽度分解（M3 核心发现）

σ_total² = σ_PSF² + w_edge²（`run_m3_sigma_arbitration.py` L849）：Route B 的 1.129 px 是 PSF ⊗ 热/几何边缘宽度的表观值——解释外边框 1.015 px 需要 w_edge ≈ 0.855 LR px。**含义**：边缘表观宽度不能直接当 PSF；同时它给反卷积激进度设了物理上限（去除光学 PSF 也除不掉热扩散边缘宽度），与 §7 Limitations 第 5 条一致。

### A.5.5 噪声底

σ_n = 0.0724 °C，smooth 区域相邻坐标 MAE 定义（`configs/noise_floor.json`、`core/src/thermal_core/ep01.py::NOISE_FLOOR_C`）。角色：A.1.4 SNR 判据分母、A.4.4 正控制噪声注入、TCForge 噪声模型均值（supp C.1.4）。

### A.5.6 误差预算汇总

| 误差源 | 量级 | 对结论的影响 | 出处 |
|---|---|---|---|
| θ ±0.1° | ~0.007 px @40 µm | 可忽略（< alignment 残差 5%） | A.5.1 |
| alignment 残差 | Chamfer 0.134 px（refined） | 主要几何误差项；E3 消融显示其改善端到端 corr +0.11（D.5.3） | supp B.2 |
| PSF σ 区间 | [0.2, 0.5] px → 2x MTF ×65 跨度 | 反卷积/合成都按区间扫描 | A.5.3 |
| 噪声底 | 0.0724 °C | 可恢复对比度下限 | A.5.5 |
| 热漂移 | session 内首尾 −0.60 °C；跨 session 中位 2.91 °C | session 门控 + 分层 split 设计动机 | supp B.1/B.3 |

**A.5 资产依赖**：`configs/stage_calibration.json`、`configs/psf_calibration.json`、`configs/noise_floor.json`；`core/src/thermal_core/displacement.py`；`reports/ep02_displacement_calibration/calibration_report.md`；`reports/ep09_psf_calibration/`；`output/ep15_info_limit/m3_sigma/sigma_summary.json`（重建：`run_m3_sigma_arbitration.py`）；S-F2（PSF 三路证据链图，资产在 `output/ep09_psf_calibration/` + `m3_sigma/`，⬜ 组合排版待做）。
**A.5 待回填**：无——可成稿。
