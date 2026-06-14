# 补充材料 A —— 理论与推导

## A.1 系统 MTF 与 SR 可行性分析

本节从系统的调制传递函数（MTF）和信噪比（SNR）出发，建立 2x 与 4x 超分辨率的理论可行性边界，为主文 §3.3 和 §1 的 claim 边界提供定量支撑。

### A.1.1 系统 MTF 模型

本系统的光学 PSF 建模为各向同性 Gaussian，标准差 $\sigma$ 以 LR 像素为单位（1 LR px = 10 µm）。对应的 MTF 为

$$\mathrm{MTF}_{\mathrm{PSF}}(f) = \exp\!\bigl(-2\pi^2 \sigma^2 f^2\bigr)$$

其中 $f$ 为空间频率（cyc/LR px）。探测器孔径的 MTF 为

$$\mathrm{MTF}_{\mathrm{det}}(f) = \bigl|\mathrm{sinc}(W f)\bigr|, \quad W = 10\;\mu\mathrm{m}$$

系统总 MTF 为二者之积。EP15 M3 的 FRC 形状拟合使用了 $\mathrm{MTF}^2$ 形式，即

$$\mathrm{MTF}(f)^2 = \bigl[\exp\!\bigl(-2\pi^2\sigma_{\mu\mathrm{m}}^2 f^2\bigr) \cdot \bigl|\mathrm{sinc}(10f)\bigr|\bigr]^2, \quad \sigma_{\mu\mathrm{m}} = \sigma_{\mathrm{LR}} \times 10\;\mu\mathrm{m}$$

需要注意的是，主文 §3.3 引用的 MTF 数值表仅含 Gaussian PSF 项，未乘 detector sinc；EP15 M3 拟合使用的则是含 sinc 的系统 MTF²。下文分别标明各自口径。

当前系统的已校准空间分辨率为 20 µm，探测器采样间距（pitch）为 10 µm/px，对应 FOV 6.4×4.8 mm。在 1x/2x/4x 输出网格上，Nyquist 频率分别为 0.5、1.0、2.0 cyc/detector px。下表给出 Gaussian-only 口径下各网格 Nyquist 处的 MTF 值：

| 网格 | $f$ (cyc/px) | $\sigma=0.20$ | $\sigma=0.35$ | $\sigma=0.50$ |
|---|---:|---:|---:|---:|
| 1x Nyquist | 0.5 | 0.821 | 0.546 | 0.291 |
| 2x Nyquist | 1.0 | 0.454 | 0.089 | 0.007 |
| 4x Nyquist | 2.0 | 0.042 | $6.3\times10^{-5}$ | ${\sim}3\times10^{-9}$ |

### A.1.2 有效 SNR 判据

为判断某一空间频率处的信息是否可恢复，定义有效 SNR 为

$$\mathrm{SNR}_{\mathrm{eff}}(f) = \frac{\Delta T \cdot \mathrm{MTF}(f,\sigma)}{\sigma_n}$$

其中 $\Delta T$ 为热对比度，$\sigma_n = 0.0724\;°\mathrm{C}$ 为探测器噪声底（由平滑区域相邻坐标 MAE 测定）。实测对比度覆盖从噪声底到外轮廓中位 2.49 °C 的宽范围：名义边缘 0.70 °C（输入 SNR = 9.7）、内轮廓中位 1.94 °C（SNR = 26.8）、外轮廓中位 2.49 °C（SNR = 34.4）。

将上述对比度代入各网格 Nyquist 处 MTF，得到有效 SNR 表：

| 对比度 | 2x（$\sigma$=0.20 / 0.35 / 0.50） | 4x（$\sigma$=0.20 / 0.35 / 0.50） |
|---|---|---|
| 名义 0.7 °C | 4.39 / 0.86 / 0.07 | 0.41 / 0.001 / ~0 |
| 内轮廓 1.94 °C | 12.16 / 2.39 / 0.19 | 1.14 / 0.002 / ~0 |
| 外轮廓 2.49 °C | 15.62 / 3.06 / 0.25 | 1.46 / 0.002 / ~0 |

风险带定义为：$\mathrm{SNR}_{\mathrm{eff}} \ge 5$ 为 observable，$\ge 3$ 为 borderline，$\ge 1$ 为 weak，其余为 noise-dominated。

### A.1.3 可行性判定

**2x 在条件下可行。** 在乐观 $\sigma$ 端（$\sigma \le 0.35$），内/外轮廓的 $\mathrm{SNR}_{\mathrm{eff}}$ 处于 observable 到 borderline 区间，存在可用的频率余量。但当 $\sigma = 0.5$ 时，2x Nyquist 处 MTF 仅剩 0.007，高频信号强烈衰减至噪声以下。因此本 POC 将目标定位在 contour-level 增强，而非计量级温度恢复。

**4x 出界。** 除最乐观的 $\sigma = 0.2$ 在外轮廓对比度下勉强达到 weak（1.46），其余条件下 $\mathrm{SNR}_{\mathrm{eff}}$ 全线 noise-dominated。4x 网格仅作 contour oversampling 和可视化用途。这一理论预测与后续 4x 网络实验的负结果互证（见 D.4.2）——理论界先于实验设定了风险预期。

需要强调的是，$\mathrm{SNR}_{\mathrm{eff}}$ 过门是必要条件而非充分证明：该判据忽略了对齐误差、热漂移与模型失配等因素。

作为辅助证据，Cramér-Rao 下界（CRB）给出了边缘定位精度的理论极限。例如在 $\Delta T = 0.7\;°\mathrm{C}$、$\sigma_{\mathrm{PSF}} = 0.5\;\mathrm{px}$ 条件下，单帧 CRB 为 0.1415 px，16 帧平均后降至 0.0344 px。EP04 实测 A-class split-half 约 0.027 px 与 16 帧 CRB 同量级，二者作为一致性检验互相支撑。

---

## A.2 观测算子与零空间

### A.2.1 前向模型

主文 §3.4 定义的离散观测模型为

$$y_k = D \cdot B \cdot H \cdot S_{t_k} \cdot x + n_k$$

其中 $x$ 为 2x 网格上的 HR 图像，$S_{t_k}$ 以 contour-refined 位移 $t_k$ 对 HR 网格作亚像素平移重采样，$H$ 为 Gaussian PSF 卷积（$\sigma \in [0.2, 0.5]$ LR px，见 A.5.3），$B$ 为 10 µm 探测器孔径 box 积分，$D$ 为 HR→LR 降采（scale=2），$n_k$ 为噪声底 0.0724 °C。

仓库中有两套该模型的实现：EP06 的矩阵式 forward/adjoint 用于经典重建（MAP-TV/TGV），EP07 训练 loss 链（blur → block-average → 可选带限）用于深度学习。两者共享 $H$ 与 $B \cdot D$ 的核心低通-折叠结构。下文的零空间论证对两者同样成立。

### A.2.2 零空间刻画

令 $A = D \cdot B \cdot H \cdot S$，其零空间源于两个机制的叠加。

第一，**带限衰减**。$H$ 和 $B$ 均为低通算子——如 A.1.1 所示，Gaussian MTF 在 2x Nyquist 处最低仅剩 0.007（$\sigma = 0.5$），叠加 box sinc 后更低。HR 网格上超过截止频率的高频分量经 $A$ 映射后衰减至噪声以下，在数值意义上落入 $\epsilon$-零空间。

第二，**混叠折叠**。降采算子 $D$ 将 HR 频谱折叠进 LR Nyquist 带宽，存在整族高频扰动 $\delta x$ 使得折叠后各别名分量相消，严格满足 $A \cdot \delta x = 0$。

直观地说，「细于截止周期的结构改动」与「别名相消组合」对所有 1x 观测均不可见。多帧相位多样性可缩小但不消灭该零空间（25 个相位 bin 的覆盖分析见 A.4.1 和 B.2.4）。

### A.2.3 零空间盲区命题

**Proposition 1.** 设 $\delta x_{\mathrm{null}}$ 满足 $A \cdot \delta x_{\mathrm{null}} = 0$，$\mathcal{L}(\hat{x}) = \ell(A\hat{x}, y)$ 为任意仅经 $A$ 接触预测的观测域损失（$\ell$ 可微）。则对任意 $\hat{x}$：

$$\mathcal{L}(\hat{x} + \delta x_{\mathrm{null}}) = \ell(A\hat{x} + A\delta x_{\mathrm{null}},\; y) = \ell(A\hat{x},\; y) = \mathcal{L}(\hat{x})$$

且 $\nabla\mathcal{L}$ 沿 $\delta x_{\mathrm{null}}$ 方向的分量恒为零。$\square$

此命题的推论是：加大 forward-consistency 权重、更换频带（highpass 或 full-band）、更换残差范数，均不改变损失对 $\delta x_{\mathrm{null}}$ 的不可见性。这正是主文 §6.2 中 V9B/V9D 与无锚 v8.1a 漂移曲线重合的机制解释（数字见 D.4.3）。

**两曲线诊断的充分条件。** 若训练过程中 (i) forward loss 已贴近其噪声地板而无法继续下降，且 (ii) 真实数据上的 proxy 指标（artifact 与 corr，定义见 A.3）持续单调漂移，则可推断漂移方向的观测域投影约为零——即漂移主要发生在 $A$ 的 $\epsilon$-零空间内。在 v9b 实验中，forward loss 自 10K 步起平坦于 0.004–0.009，而同期 artifact 从 0.37 单调上爬至 0.65，恰好满足该充分条件。

**零空间投影的直接测量**（待补充）。计划取同臂相邻 checkpoint 的预测差 $\delta\hat{x} = \hat{x}_{k+1} - \hat{x}_k$，计算 range 分量 $\delta x_{\mathrm{range}} = A^\dagger A \cdot \delta\hat{x}$（$A^\dagger$ 用共轭梯度近似），报告 $\|\delta x_{\mathrm{range}}\| / \|\delta\hat{x}\|$ 随训练步数的变化。若该比值在漂移期趋近零，则可将 §5.3 的间接诊断升级为直接测量。

---

## A.3 Proxy 反相关的构造性论证

主文 §5.2 使用 artifact score 与 raw-control correlation 构成的 proxy 对来监测训练漂移。本节论证二者为何存在构造性的反相关关系。

### A.3.1 指标定义

两个 proxy 共享同一预处理：将 SR 输出温度图变换到 highpass 域 $u = \hat{x} - G_{\sigma_{\mathrm{bg}}}(\hat{x})$（$\sigma_{\mathrm{bg}} = 5$ HR px），控制图 $c$ 为 248 帧原始数据均值的 bicubic 上采样经同一 highpass 变换。

TB-scale artifact score（用于训练期 eval 和主文 F3/F4 轨迹图）定义为

$$\mathrm{artifact} = \frac{\mathrm{std}(u - G_{\sigma=1}(u)) + 0.25 \cdot \mathrm{std}(\nabla^2 u)}{\mathrm{std}(u)}$$

即 $u$ 内部更高频能量的占比。EP11-harness 口径使用不同定义（ringing + 0.25×blockiness），数值与 TB-scale 不可比，不得混入同一表格。

Raw-control correlation 定义为 $\mathrm{corr} = \mathrm{Pearson}(u, c)$，取全图 finite 像素。

### A.3.2 构造性论证

两个 proxy 都是同一张 highpass 图 $u$ 的泛函——artifact 度量 $u$ 的内部高频能量分布，corr 度量 $u$ 与固定观测锚 $c$ 之间的线性一致性。它们并非独立的证据源，这是反相关论证的前提。

考虑一个典型的「合成先验风格化」扰动方向 $\delta u$：在结构边缘处将响应增亮或增宽（这是 D.3 中可观测到的漂移形态）。沿此方向，$\delta u$ 集中于边缘高频，$\mathrm{std}(\text{high\_freq})$ 和 $|\nabla^2 u|$ 的增速快于分母 $\mathrm{std}(u)$，因此 artifact 一阶上升。与此同时，由于 $c$ 固定且 $\delta u$ 不源自观测（属零空间方向，见 A.2.3），$u$ 在 $c$ 的正交补空间增加能量，Pearson 相关系数的分母 $\|u\|$ 增大而分子 $\langle u, c \rangle$ 近似不变，故 corr 一阶下降。

沿该扰动轴，两指标的一阶响应符号恒反，联合最大化不可行。由此得到三条推论。第一，proxy 对是漂移温度计与选点准则，不是两个可联合优化的分数。第二，跨 input-mode 的横向比较无效：hybrid drizzle 输入合法携带更多高频能量，其 artifact 基线天然更高。第三，应关注训练轨迹的走向而非端点数值。

---

## A.4 FRC 方法学

本节详述主文 §5.1 中 Fourier Ring Correlation（FRC）的实验方案与全部控制组结果。

### A.4.1 分层分半构造

FRC 的核心思路是将 248 帧 clean set 分为两个子集 A、B，各自独立重建后计算频域相关。为保证分半不引入系统性偏差，采用相位分层（phase-stratified）设计：将每帧的 stage command 经坐标变换映射为亚像素相位 $\phi = \mathrm{mod}(\text{shift}, 1)$，按 $5 \times 5 = 25$ 个 bin 分层（每 bin 7–13 帧），在每个 bin 内随机置换后奇偶交替分配给 A/B（约束 $|N_A - N_B| \le 1$）。A 和 B 各自以 bilinear drizzle 重建到 5x 诊断网格（hr\_pitch = 2 µm），空 bin 填全局均值。使用 seeds {42, 123, 456} 重复三次，主曲线取逐 ring nanmean。窗口函数为 Tukey $\alpha = 0.25$，边缘裁剪 16 LR px。

FRC 按环状积分计算：

$$\mathrm{FRC}(r) = \frac{\sum_{|\mathbf{f}| \in r} \mathrm{Re}\bigl(F_A(\mathbf{f}) \cdot \overline{F_B(\mathbf{f})}\bigr)}{\sqrt{\sum_{|\mathbf{f}| \in r} |F_A(\mathbf{f})|^2 \cdot \sum_{|\mathbf{f}| \in r} |F_B(\mathbf{f})|^2}}$$

cutoff 判据采用 $1/7$ 常数阈值（0.142857）和 half-bit 曲线，取第一个 FRC 低于阈值的 ring 对应周期。

### A.4.2 主结果与控制组

$1/7$ cutoff 为 **17.03 µm**（3-seed 均值曲线），逐 seed 分别为 16.17/16.17/17.03 µm（std = 0.50 µm）；half-bit 判据给出相同值。本文仅主张 17.0 µm：超过 20 µm 分辨率的相干信息确实存在，但低于 11–14 µm 的理论期望。

四个控制组的设计与结果如下表。正控制（单帧 bicubic 加噪）的 cutoff 为 13.58 µm，未达到「明显差于 main」的预期，原因是单帧插值的平滑频谱在低噪声下自相关偏高；负控制（shift-shuffle）在 8–12 µm 段中位 FRC 仍有 0.504，因为置换后仍共享场景低频与网格结构，部分失效。漂移控制（前/后半按采集序分别重建）cutoff 退化至 26.20 µm，符合预期——时间间隔放大了热漂移的影响。zero-coverage 统计显示空 bin 均值 27.2%、最大 36.2%，表明 5x 网格存在欠覆盖。

### A.4.3 10–12 µm 反弹

在 10–12 µm 周期段，FRC 出现异常高值，需要谨慎判读。关键频带的对比数据如下：

| 周期 (µm) | main | bicubic 正控 | shuffle 负控 | drift 控 |
|---:|---:|---:|---:|---:|
| 16 | 0.138 | 0.691 | −0.012 | 0.117 |
| 12 | 0.593 | 0.002 | −0.095 | 0.578 |
| 10 | 0.935 | 0.012 | 0.906 | 0.887 |
| 8 | 0.545 | −0.023 | 0.390 | 0.345 |

10–12 µm 的高 FRC 同时出现在负控制（10 µm 处 0.906）和漂移控制（0.887）中，说明该反弹由 coverage/lattice 结构与漂移驱动，而非真实分辨率信息。因此不作分辨率证据使用。

### A.4.4 MAP-TV 前后对照

以 MAP-TV（$\sigma = 0.2$, $\lambda = 10^{-3}$）对 drizzle 输出做去卷积后，split-half FRC 在 20–10 µm 段从 0.319/0.088/0.053/0.575/0.893 提升至 0.976/0.965/0.955/0.947/0.934。该提升反映的是分半一致性（稳定性）的改善，而非光学分辨率本身的提升。

---

## A.5 标定不确定度传播

### A.5.1 旋转角 $\theta$ 的误差传播

坐标到位移的映射为

$$d_x = \frac{X\cos\theta + Y\sin\theta}{10}, \quad d_y = \frac{-X\sin\theta + Y\cos\theta}{10}$$

（单位 µm→px）。对 $\theta$ 求导，以最大步进 $X = 40$ µm、$\delta\theta = 0.1° = 1.745$ mrad 为例：$\delta d_x \approx 0.0052$ px，$\delta d_y \approx 0.0047$ px，合成约 **0.007 px**。位移幅值 $|s| = X/10 = 4.0$ px 与 $\theta$ 一阶无关——$\theta$ 误差表现为方向偏差而非步长误差。该传播量比 alignment 残差（Chamfer 0.134 px，见 B.2.2）低一个量级，不是误差预算的主项。

AVI 的独立方向验证（gradient-NCC 合并估计 $\theta = 47.14°$，95% CI [46.36°, 47.92°]）覆盖了标定值 47.6°，但 X/Y 子组存在约 3° 系统差异（X 48.70° vs Y 45.63°），且 AVI 为 8-bit 渲染视频，仅作一致性检验，不替换配置。

### A.5.2 PSF $\sigma$ 区间

三条独立标定路线给出了不一致的结果：Route A（forward 残差）$\sigma = 0.226$ px [0.208, 0.240]；Route B（ESF 拟合）$\sigma = 1.129$ px [1.041, 1.215]；Route C（joint hold-out）$\sigma = 0.119$ px。三者 spread 达 1.01 px，远超 0.05 px 容差。

EP15 M3 仲裁澄清了分歧的物理根源：Route B 的 1.129 px 实际上是 PSF 与热/几何边缘宽度的卷积表观值，满足 $\sigma_{\mathrm{total}}^2 = \sigma_{\mathrm{PSF}}^2 + w_{\mathrm{edge}}^2$。这意味着边缘表观宽度不能直接等同于 PSF，同时也为反卷积的激进度设定了物理上限——即使完全去除光学 PSF，热扩散造成的边缘宽度仍然存在。综合 FRC 形状拟合（$\sigma = 0.2$ 最优）与多边缘 ESF 上界（0.546 px），最终采纳区间为 $\sigma \in [0.2, 0.5]$ LR px。

该区间的传播后果显著：2x Nyquist 处 MTF 在区间内变化 65 倍（0.454→0.007），因此所有依赖 PSF 的计算均须把 $\sigma$ 作为扫描参数。

### A.5.3 误差预算汇总

| 误差源 | 量级 | 对结论的影响 |
|---|---|---|
| $\theta$ ±0.1° | ~0.007 px @40 µm | 可忽略（不足 alignment 残差的 5%） |
| alignment 残差 | Chamfer 0.134 px（refined） | 主要几何误差项；E3 消融显示改善端到端 corr +0.11（见 D.5.3） |
| PSF $\sigma$ 区间 | [0.2, 0.5] px → 2x MTF 65× 跨度 | 反卷积与合成均按区间扫描 |
| 噪声底 | 0.0724 °C | 可恢复对比度的下限 |
| 热漂移 | session 内 −0.60 °C；跨 session 中位 2.91 °C | session 门控 + 分层 split 的设计动机（见 B.1、B.3） |
