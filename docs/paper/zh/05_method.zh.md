# 第 4 章 方法（中文打磨稿）

> 本文是论文 §4 的中文打磨稿（唯一事实来源）。
> 本章结构使 §5 的 input $\times$ anchor 消融矩阵（T2）可从本节小节对应直接读出。
> 实现细节（参数全表、训练配置对照、权重图机制等）放 supp C；正文只给必要的设计动机与总式。

![图 F-method：方法主图 — TCForge 合成管线、网络/输入模式、观测锚定与选点协议](../../../paper/figures/fig_tcforge_pipeline.pdf)

> **图 F-method（方法主图）**：(a) TCForge 物理匹配合成平台（场景几何 → 温度场 → 前向退化 → LR burst 训练对）；(b) ThermalLift-UNet 骨干与两种输入模式（1× stats 5ch vs hybrid drizzle 8ch）；(c) contour 取向 loss 及四种观测锚定变体（none / HP-FC / Full-FC / Native-FC），构成 T2 消融矩阵的两个因素；(d) proxy Pareto checkpoint 选点协议。终稿资产：`paper/figures/fig_tcforge_pipeline.pdf`（脚本：`scripts/make_tcforge_pipeline_figure.py`；位图备份：`fig_tcforge_pipeline.png`）。

## 4.1 方法概览

本章给出三个组件。第一，以三种经典重建方法作为门控基准：一个学习方法只有同时在 §3.3 定义的 FRC 频带一致性与 zigzag profile 指标上不输经典基准才被采纳（采纳规则见附录 C.5）。第二，构建一个参数全部源自实测的物理匹配合成训练平台 TCForge，使学习方法在合成域上训练、在真实域上部署。第三，在固定网络骨干下系统变换输入信息通路与观测锚定方式，形成 $\{\text{1x stats},\;\text{hybrid drizzle}\} \times \{\text{none},\;\text{band},\;\text{full},\;\text{legal}\}$ 矩阵，以隔离亚像素相位的送达路径与 loss 侧锚定对零空间漂移的约束能力。

## 4.2 经典重建基准

### 4.2.1 Drizzle

将 248 帧按 contour-refined 位移对齐后，以通量守恒的亚像素散射 \citep{fruchter2002drizzle} 投射到 $2\times$ 网格（pixfrac = 0.7，square kernel），coverage $< 1.0$ 的像素置 NaN。Drizzle 是最小假设融合基线，同时暴露任何细网格方法都须管理的覆盖与 lattice 伪影——在 $5\times$ 诊断网格上平均零覆盖率达 27%。

### 4.2.2 MAP-TV 去卷积基准

在全部 248 帧上构建 §3.1 中的前向模型（shift $\to$ Gaussian PSF $\to$ 10 µm detector box $\to$ 下采样），以 GPU FISTA \citep{beck2009fast} 配合平滑 TV 梯度优化 150 次迭代。参数在 $\sigma_\mathrm{PSF} \in \{0.2, 0.3, 0.4, 0.5\}\;\mathrm{LR\;px}$ 与 $\lambda_\mathrm{TV} \in \{3\!\times\!10^{-4},\;10^{-3},\;3\!\times\!10^{-3}\}$ 网格上，按 split-half proxy 组合选出 $\sigma = 0.2$、$\lambda = 10^{-3}$。MAP-TV 承担验收门控角色（附录 C.5）：一个学习方法只有同时在 FRC 频带一致性与 zigzag profile 指标上不输该基准，才会被采纳。

### 4.2.3 各向异性 coverage-weighted TGV

Raster 采集导致数据约束各向异性（行内 X 相邻帧间隔 1，行间 Y 间隔约 16），标准各向同性 TGV \citep{bredies2010tgv} 在此条件下产生水平条纹伪影。我们对 FISTA 外层的 TGV 近端算子做两项适配。

第一，各向异性对偶投影。Chambolle-Pock \citep{chambolle2011firstorder} 内层的一阶对偶球由标准球改为椭圆

$$\bigl\{(a,b): (a/r_a)^2 + (b/r_b)^2 \le 1\bigr\}, \quad r_a = \alpha_1 \cdot r_y,\; r_b = \alpha_1,\; r_y = 1.5,$$

使 Y 方向正则强度弱于 X 方向，补偿 Y 方向数据约束不足。

第二，coverage 加权数据梯度。每个 HR 像素的数据保真梯度除以其预计算的帧覆盖率（bilinear splat + PSF adjoint），而非均匀除以帧数 $K$。这修正了 bilinear scatter 把权重集中在固定 HR 行导致的条纹。两项改动将 artifact score 从 3.870 降至 0.695（$-82\%$），raw-control correlation 从 0.902 提升至 0.916。

## 4.3 物理匹配合成平台 TCForge

训练数据通过实测前向模型合成，而非学习退化分布。合成管线包含四个阶段。

场景几何以 8 层随机原语叠加构造 IC 布局（大块、柱、L 形走线、框、细部、via、pin），整体旋转角围绕实测 $\theta = 47.6°$ 做 $U(-1.5°, +1.5°)$ 抖动。为消除 HR 目标中的阶梯锯齿，默认启用 $4\times$ SSAA：在 $4\times$ 画布上绘制旋转后做 $4\times4$ 均值降采，得到 $\text{coverage} \in [0,1]$ 的连续值。温度场渲染为

$$T = T_\mathrm{bg} + \Delta T \cdot \text{coverage} + A_\mathrm{drift} \cdot \text{smooth\_noise},$$

其中 $\Delta T \sim U(0.5, 5.0)\;°\mathrm{C}$ 覆盖实测对比度范围，PSF $\sigma \sim U(0.15, 0.55)$ 覆盖可信区间（附录 A.5.2），噪声遵循均值等于实测 0.0724 °C 的 lognormal 分布。前向退化依序执行 Gaussian PSF $\to$ 分块双线性均值降采（scale 2），亚像素位移重放实测 contour-refined 248 帧 profile 并叠加 $\sigma_j = 0.02$ px jitter。

当前训练池包含 1000 个场景（$480\times640 \to 960\times1280$），hybrid 输入训练池另存完整 LR burst 并预计算 $K=4$ 组 drizzle 变体（$k=0$ 为 canonical 全帧无噪配置，$k=1\text{--}3$ 保留 60–100\% 帧并叠加 $N(0, 0.05\;\text{px})$ shift 噪声）。训练时以确定性采样选取变体编号，将帧子集与 shift 增广烘焙入预计算中，避免 dataloader 阻塞（细节见 supp C.1）。

## 4.4 学习方法族

为避免将实验内部 run id 带入论文表述，本文将学习方法族记为 ThermalLift-UNet，并在主文中使用按消融因子可读的短名：Stats、Stats+HP-FC、Stats+Full-FC、Hybrid、Hybrid+Native-FC 与 Hybrid+ResObs。内部训练编号仅在补充材料的可复现性映射中保留。

### 4.4.1 网络骨干

所有学习变体共享同一骨干，架构选择刻意冻结，实验变量仅为输入通路与观测锚定。骨干采用标准 UNet \citep{ronneberger2015unet}（base channels $c = 64$）：编码路径 3 层 MaxPool + ConvBlock（每块 $2\times 3\times3$ Conv + GroupNorm \citep{wu2018group} + SiLU + SE attention \citep{hu2018squeeze}），通道数 $c, 2c, 4c, 8c$；解码路径双线性上采样 + skip concat + ConvBlock。HR 输出头默认采用双线性模式（bilinear $\times$ scale $\to$ $3\times3$ Conv）；消融变体 Stats+PixelShuffle 另测 PixelShuffle 头 \citep{shi2016realtime}（ICNR 初始化 \citep{aitken2017checkerboard}），后者引入条纹伪影且 proxy 全程更差，仅作 head 归因对照保留（§5.4）。

### 4.4.2 Contour 取向损失

总损失为

$$\mathcal{L} = w_m \mathcal{L}_\mathrm{MSE} + w_{hp} \mathcal{L}_\mathrm{HP} + w_e \mathcal{L}_\mathrm{Edge} + w_s (1 - \mathrm{SSIM}) + w_{gv} \mathcal{L}_\mathrm{GV} + w_{fm} \mathcal{L}_\mathrm{FM} + w_r \mathcal{L}_R,$$

各项含义与设计动机如下。$\mathcal{L}_\mathrm{HP} = \|HP(\hat{x}) - HP(x)\|_1$ 为 highpass 结构损失（$HP(x) = x - G_{\sigma=5}(x)$），乘以 structure/thin/gap 三种权重图以在结构关键区域加权。$\mathcal{L}_\mathrm{GV} = \|g_{x,\hat{x}} - g_{x,x}\|_1 + \|g_{y,\hat{x}} - g_{y,x}\|_1$ 保留梯度完整向量信息，捕捉幅值型 Edge loss 看不到的边缘膨胀与畸变。$\mathcal{L}_\mathrm{FM}$ 为 forward consistency 项（§4.4.4 详述）。$\mathcal{L}_R = \text{mean}(|\hat{x} - x_\mathrm{input}|)$ 为残差惩罚（仅 Hybrid+ResObs 使用，§4.4.5）。

主线使用 conservative 权重壳（自 Stats 起冻结，后续学习变体沿用）：$w_m = 0.3$、$w_{hp} = 0.8$（structure boost 2.0、thin boost 3.0、gap boost 2.0）、$w_{gv} = 0.15$、$w_s = 0.15$、$w_e = 0.05$。消融变体 HotLoss 使用更激进的 hot 壳（$w_{hp} = 1.0$、structure boost 4.0），保留为漂移放大参照。权重演化的完整历史见 supp C.2。

### 4.4.3 输入模式

输入模式是消融矩阵的第一个因素。

1x 统计输入（baseline）。在 $1\times$ 网格上构造 5 通道：对齐后的 mean / median / coverage / variance / highpass fused，经网络以 scale = 2 上采样到 HR。这是常规特征化方式，但我们的消融实验（§5.3）表明它在输入编码阶段即坍塌了 burst 的亚像素相位信息。

Hybrid drizzle 输入。在 $2\times$ 网格上融合两类信息：通道 0--4 为上述 1x 统计经双线性上采到 $2\times$，通道 5--7 为在 $2\times$ 网格上渲染的 drizzle mean / coverage / variance。网络此时以等效 scale = 1 在 $2\times$ 网格上运行。亚像素证据由此作为数据直接进入网络，而非作为需要学习的隐式先验。

### 4.4.4 观测锚定变体

观测锚定是消融矩阵的第二个因素。所有锚定变体共享同一机制：将网络预测 $\hat{x}$ 经 §3.1 中的实测前向算子 $A = D \cdot B \cdot H \cdot S$ 重投影到 $1\times$ 观测域，与持有的 $1\times$ 观测 $y_\mathrm{obs}$ 比较。

| 变体 | 代表方法 | 锚定方式 | 描述 |
|---|---|---|---|
| none | Stats / Hybrid | $w_{fm} = 0$ | 无观测锚定 |
| HP-FC | Stats+HP-FC | $w_{fm} = 0.1$，highpass | 只在 highpass 频带内计算 forward consistency |
| Full-FC | Stats+Full-FC | $w_{fm} = 0.1$，full-band | 对全频谱计算 forward consistency |
| Native-FC | Hybrid+Native-FC | $w_{fm} = 0.1$，highpass | hybrid 输入下，锚消费经数据管线单独携带的原生 $1\times$ aligned-mean patch，而非 hybrid 通道 0 的上采样均值 |

该矩阵的全部填充变体——Stats、Stats+HP-FC、Stats+Full-FC（1x 输入）与 Hybrid、Hybrid+Native-FC（hybrid 输入）——均训练至 60K step，§5.2 的漂移轨迹证明所有 loss 侧锚定变体的漂移曲线近乎一致。

### 4.4.5 Residual-over-observation 参数化 (Hybrid+ResObs)

在 hybrid 输入上叠加输出参数化：网络预测残差 $\delta$，最终输出为

$$\hat{x} = x_\mathrm{drizzle}^\mathrm{ch5} + \delta,$$

其中 $x_\mathrm{drizzle}^\mathrm{ch5}$ 是输入的 drizzle mean 通道。一个 L1 残差惩罚

$$\mathcal{L}_R = w_r \cdot \text{mean}(|\delta|)$$

控制输出能离观测域 drizzle 基多远。权重 $w_r$ 以 $\lambda$ 参数化：通过 $\lambda \in \{0.2, 0.5, 1.2, 3.0\}$ 扫描，在 fine-window 保真/锐度/grain 三维 proxy 上选择工作点（§5.1 报告 $\lambda = 1.2$ @ 15K step 的结果）。Hybrid+ResObs 使权衡可调，但其选定行仍是一个高锐度、较低 corr、高 artifact 的工作点。

## 4.5 Checkpoint 选点协议

训练端点（领域默认做法）会让每个变体交出其最差的 checkpoint——这是在零空间漂移下的直接后果。选点因此是方法的组成部分，而非事后处理。

对每个变体的全部 eval checkpoint，首先将两个 proxy 归一化到 $[0,1]$：

$$a_\mathrm{norm} = \frac{\text{artifact} - \min}{\max - \min}, \quad c_\mathrm{norm} = \frac{\max_\mathrm{corr} - \text{corr}}{\max_\mathrm{corr} - \min_\mathrm{corr}},$$

计算到理想点 $(0, 0)$ 的距离 $d = \sqrt{a_\mathrm{norm}^2 + c_\mathrm{norm}^2}$，按 $d$ 升序取 $\le 3$ 个候选（5K window 去重），60K 端点若未入选则强制追加为漂移参照。Rank-1 为 canonical 候选。每个候选须通过温度域三联 panel 的视觉门控。选点规则的完整流程见 supp C.5。

> 待办（§4）：① 方法主图已引用 `paper/figures/fig_tcforge_pipeline.pdf`，终稿核对 (a)–(d) 子图标注与 §4.3–§4.5 一致；② 压页至 1.0 页（与 §3 共占约 2 页）。
