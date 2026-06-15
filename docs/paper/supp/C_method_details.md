# 补充材料 C —— 方法实现细节

## C.1 TCForge 物理匹配合成平台

本节详述主文 §4.2 中合成数据平台 TCForge 的实现，其目标是生成与真实 LWIR 微扫描在物理特性上匹配的训练数据。

### C.1.1 场景几何生成

TCForge 通过 8 层随机原语叠加构造 IC 布局的几何 mask：依次放置大块、中宽柱、L 形走线（25% 粗 / 75% 细）、框结构、细部与引脚阵列、圆形 via、对边 pin，最后做减通道处理。所有层均由 seed 确定性控制。特征尺度由难度参数决定：easy 的 major/minor/line 为 80/60/45 µm，medium 为 55/42/30 µm，hard 为 35/28/22 µm，stress 为 22/16/12 µm。场景整体旋转角为 $\theta = 47.6° + U(-1.5°, +1.5°)$，围绕实测 stage 角度抖动以匹配真实几何取向。

为消除阶梯锯齿对训练的污染，默认启用 4×SSAA 抗锯齿：在 4 倍画布上绘制后旋转，再做 $4 \times 4$ 块均值降采，得到 $\text{coverage} \in [0, 1]$ 的连续值。这样 HR 目标不含阶梯锯齿，亚像素细线携带 coverage 缩放后的幅度。

### C.1.2 温度渲染与物理随机化

温度场渲染公式为

$$T = T_{\mathrm{bg}} + \Delta T \cdot \text{coverage} + A \cdot \text{smooth\_noise}$$

关键物理参数的随机化分布经过仔细设计以覆盖实测范围：$\Delta T \sim U(0.5, 5.0)\;°\mathrm{C}$（覆盖实测内/外轮廓对比度 1.94/2.49 °C）；PSF $\sigma \sim U(0.15, 0.55)$ LR px（覆盖采纳区间 [0.2, 0.5]，见 A.5.2），其中 30% 使用椭圆 PSF、10% 使用 Airy 函数替代 Gaussian；噪声遵循 lognormal 分布，均值等于实测噪声底 0.0724 °C；漂移幅度 $\sim U(0, 0.3)\;°\mathrm{C}$；亚像素位移使用实测 contour\_refined 的 248 帧 profile 重放并叠加 $\sigma = 0.02$ px 的 jitter。

Forward 退化模式默认采用 physical\_block\_average：对 HR 图先做 PSF blur，再对各 shift 执行分块双线性均值降采（scale 2 或 4）。

### C.1.3 训练池

当前构建了两个训练池。training\_pool\_2x\_aa 包含 1000 个 scene（难度分布 easy 200/medium 400/hard 300/stress 100），每 scene 248 帧，LR $480 \times 640$ → HR $960 \times 1280$，用于 1x 统计输入的各训练变体：Stats、Stats+PixelShuffle、Stats+HP-FC、Stats+Full-FC（run ids: v8.1a/v8.1b/v9b/v9d）。training\_pool\_2x\_aa\_burst 在此基础上保存了完整的 LR burst（约 152 GB），用于 hybrid 输入的各变体：Hybrid、Hybrid+Native-FC、Hybrid+ResObs（run ids: V9A/V9C/V10）。

为提高训练效率，还预计算了 $K = 4$ 组 drizzle 变体：$k = 0$ 使用全部 248 帧且无 shift 噪声（canonical，与推理一致）；$k = 1 \ldots 3$ 保留 60–100% 帧（$\ge 30$）并叠加 $N(0, 0.05\;\text{px})$ 的 shift 噪声。训练时按 $[\text{seed}, \text{epoch}, \text{scene}, \text{0xBEEF}]$ 确定性抽样变体编号，将帧子集与 shift 噪声增广烘焙入预计算结果中，避免训练期间现场 drizzle 阻塞 dataloader。

---

## C.2 网络与损失

### C.2.1 网络架构

ThermalSRUNet 采用编码-解码结构：编码路径为 3 层 MaxPool + ConvBlock（每块 $2 \times 3 \times 3$ Conv + GroupNorm + SiLU + SE attention），通道数按 $c, 2c, 4c, 8c$ 递增（$c = 64$）；解码路径使用双线性上采样 + skip concat + ConvBlock。HR 输出头默认采用 bilinear 模式（双线性 $\times$ scale → $3 \times 3$ Conv），Stats+PixelShuffle（v8.1b）消融变体另测试了 pixelshuffle 模式（ICNR 初始化 sub-pixel conv + PixelShuffle + HRResBlock），后者产生条纹伪影且 proxy 全程更差（见 D.4.1），仅作为 head 归因对照保留。

在 hybrid 输入模式下，网络在 2x 网格上以等效 scale=1 运行，因为输入已经是 2x 分辨率。

### C.2.2 损失函数

ContourSRLoss 的总损失为

$$\mathcal{L} = w_m \cdot \text{MSE} + w_{hp} \cdot \text{HP} + w_e \cdot \text{Edge} + w_s \cdot (1 - \text{SSIM}) + w_{gv} \cdot \text{GV} + w_{lap} \cdot \text{Lap} + w_{fm} \cdot \text{FM} + w_r \cdot R$$

其中各项含义如下。MSE 为标准均方误差，可乘 gap\_weight。HP 为 highpass 结构损失 $\|HP(\hat{x}) - HP(x)\|_1$，预处理 $HP(x) = x - G_{\sigma=5}(x)$，乘以 structure/thin/gap 三种权重图。Edge 为梯度幅值损失 $\||\nabla\hat{x}| - |\nabla x|\|_1$ 加 0.25 倍半分辨率版本。SSIM 使用 $11 \times 11$ Gaussian 窗。GV（grad-vector）为 $\|g_{x,p} - g_{x,t}\|_1 + \|g_{y,p} - g_{y,t}\|_1$，保留梯度的完整向量信息以捕捉边缘膨胀和畸变——这是幅值型 Edge loss 看不到的方向偏差。Lap 为 $\text{ReLU}(|\Delta x| - |\Delta\hat{x}|)$ 的均值，只罚「比目标钝」。FM 为 forward consistency $\text{MSE}(\text{AvgPool}_s(G_{\sigma s}(\hat{x})),\; y_{\text{obs}})$，可选 full-band 或 highpass 模式。R 为残差惩罚 $w_r \cdot \text{mean}(|\hat{x} - x_{\text{input}}|)$（Hybrid+ResObs / V10 使用）。

权重图机制使损失在结构关键区域有针对性地加强：structure\_boost 按梯度强度对 HP/GV 权重图加权；thin\_boost 在宽度 $\le 3$ HR px 的细结构上倍增；gap\_boost 在两侧有结构的窄缝背景上倍增。

保守权重预设（conservative preset，自 Stats / v8.1a 起冻结，FC 与 hybrid 系列沿用）设定为 $w_m = 0.3$、$w_{hp} = 0.8$、structure\_boost = 2.0、$w_{gv} = 0.15$、thin\_boost = 3.0、gap\_boost = 2.0。HotLoss（v6）使用更激进的权重预设（$w_{hp} = 1.0$、structure\_boost = 4.0 等），叠加 Lap 0.1 和 full-band FM 0.1，保留为漂移放大参照变体。

### C.2.3 Hybrid 8 通道输入

Hybrid、Hybrid+Native-FC、Hybrid+ResObs（run ids: V9A/V9C/V10）使用 hybrid 8 通道输入，在 2x 网格上融合 1x 统计特征与 2x drizzle 证据。通道 0–4 为 1x fused 统计（aligned mean/median/coverage/variance/highpass fused）经双线性上采到 2x；通道 5 为 drizzle mean @2x，是观测域证据的主通道（也是 Hybrid+ResObs 残差惩罚的基准）；通道 6 为 drizzle coverage @2x（空洞可识别：未观测 bin 以全局均值填充，coverage = 0）；通道 7 为 drizzle variance @2x。

Hybrid+Native-FC（V9C）的合法锚配套值得说明：hybrid 的通道 0 是上采样均值而非合法的 1x 观测，因此 forward 锚需另走数据管线携带原生 1x aligned-mean patch（偶数 2x origin 裁剪、增广同步）。

---

## C.3 训练配置对照

七个变体的训练配置共享以下设定：scale=2、base\_channels=64、patch\_size\_hr=256、total\_steps=60000、lr=$2 \times 10^{-4}$、AMP、compile、edge/ssim/coarse 权重 0.05/0.15/0.25、highpass\_sigma=5、real\_eval 使用 248 帧 contour\_refined + center-1/3 + zoom3。

各变体的差异字段构成一个 $\{\text{input}\} \times \{\text{anchor}\}$ 消融矩阵：

| 字段 | HotLoss | Stats | Stats+PixelShuffle | Stats+HP-FC | Stats+Full-FC | Hybrid | Hybrid+Native-FC |
|---|---|---|---|---|---|---|---|
| run id | v6 | v8.1a | v8.1b | v9b | v9d | V9A | V9C |
| 训练池 | 2x | 2x\_aa | 2x\_aa | 2x\_aa | 2x\_aa | 2x\_aa\_burst | 2x\_aa\_burst |
| input / in\_ch | lr / 5 | lr / 5 | lr / 5 | lr / 5 | lr / 5 | hybrid / 8 | hybrid / 8 |
| HR 头 | bilinear | bilinear | pixelshuffle | bilinear | bilinear | bilinear | bilinear |
| loss 权重预设 | hot+Lap | cons. | cons. | cons. | cons. | cons. | cons. |
| forward 锚 | full 0.1 | 无 | 无 | hp 0.1 | full 0.1 | 无 | legal-1x hp 0.1 |
| batch\_size | 128 | 128 | 128 | 128 | 128 | 64 | 64 |

消融矩阵的直读方式：$\{1x, \text{hybrid}\} \times \{\text{none}, \text{HP-FC}, \text{Full-FC}, \text{Native-FC}\}$ = Stats / Stats+HP-FC / Stats+Full-FC / Hybrid / Hybrid+Native-FC（HotLoss 和 Stats+PixelShuffle 分别为 loss 温度和 head 的额外归因变体）。

可比性说明：Hybrid（V9A）前 35K 步使用 batch\_size=128，之后与 Hybrid+Native-FC（V9C）全程 batch\_size=64——Hybrid 与 Hybrid+Native-FC 两个变体与 1x 各变体的训练动力学不完全同条件。Hybrid（V9A）30K 保真悬崖与 batch size 切换时间重合，混杂未排除。

---

## C.4 经典方法

### C.4.1 Drizzle

Drizzle 使用 STScI drizzle 库实现，pixfrac 默认 0.7（sweep \{1.0, 0.8, 0.7, 0.6, 0.5\}），kernel 为 square，输出网格 $(H \times 2, W \times 2)$。coverage $< 1.0$ 的像素置 NaN。TCForge 训练侧的 scatter 使用 bilinear kernel，两种 kernel 的差异在 D.7 的 fine-window 指标中以 drizzle 输入通道参照点的形式体现。

### C.4.2 各向异性 TGV

TGV 重建采用外层 FISTA 框架 $x_{k+1} = \text{TGV\_prox}(z_k - \eta \nabla D(z_k))$，针对本数据集的 raster 各向异性（见 B.3.1）做了两项关键设计。

第一，各向异性对偶投影。TGV 内层（Chambolle-Pock 路径）的一阶对偶球改为椭圆 $\{(a,b) : (a/r_a)^2 + (b/r_b)^2 \le 1\}$，其中 $r_a = \alpha_1 \cdot r_y$、$r_b = \alpha_1$（$r_y = 1.5$）；二阶对称张量同构造。这使得 Y 方向正则强度弱于 X 方向，补偿 Y 方向数据约束的不足。

第二，coverage 加权数据梯度。每个 HR 像素的数据保真梯度除以预计算的 coverage（由 bilinear splat + PSF adjoint 得到），而非统一除以帧数 $N$。这修正了 bilinear scatter 把权重集中在固定 HR 行上所导致的水平条纹。

参数网格为 $\lambda_{\text{tv}} \in \{3 \times 10^{-4}, 10^{-3}, 3 \times 10^{-3}\} \times \sigma_{\text{PSF}} \in \{0.18, 0.50\}$，$\alpha_{\text{ratio}} = 2.0$（$\alpha_1 = \lambda$, $\alpha_0 = \lambda/2$），外层 100 iter、内层 80 iter。最优参数下 artifact 从 3.870 降至 0.695（−82%），raw-control corr 从 0.902 升至 0.916，耗时 30.8 min CPU。

### C.4.3 MAP-TV 去卷积基准

MAP-TV 基准用于采纳判定（C.5.0 采纳规则）：学习方法必须在 FRC 频带一致性与 zigzag profile 指标上同时不输 MAP-TV 才可采纳。其 forward 模型为 shift → Gaussian PSF（$\sigma$ LR px）→ avg\_pool box，使用 GPU FISTA + smoothed-TV 梯度优化，150 iter。

参数网格为 $\sigma \in \{0.2, 0.3, 0.4, 0.5\} \times \lambda \in \{3 \times 10^{-4}, 10^{-3}, 3 \times 10^{-3}\}$。选择规则为每个 $\sigma$ 内按 split-half proxy（split\_half\_nrmse + 0.05×artifact + 0.08×std\_excess）取最小，再跨 $\sigma$ 选全局最优，结果为 $\sigma = 0.2$、$\lambda = 10^{-3}$，耗时 4563 s GPU。

---

## C.5 Checkpoint 选择协议

训练期间的 checkpoint 选择对学习变体的最终输出质量至关重要。本节详述主文 §4.5 的选择规则与采纳判定（主文 §5.1 表注⑦引用本节）。

### C.5.0 采纳规则（Adoption rule）

一个学习方法被采纳，当且仅当在其选定 checkpoint 上同时满足：
（a）FRC 频带一致性 ≥ MAP-TV 基准；（b）zigzag FWHM/dip 不差于基准；（c）proxy 轨迹无 post-selection 漂移悬崖；
（d）双域视觉 panel 均通过。否则经典基准仍是交付物。正是这条保守规则，让诚实的负结果（PixelShuffle 头、4x 网络、
loss 侧锚定）与阳性结果得以并列报告——拒绝一个方法和采纳一个方法用的是同一把尺子。

### C.5.1 选择流程

对每个变体的所有 eval checkpoint，首先归一化两个 proxy：$a_{\text{norm}} = (\text{artifact} - \min) / (\max - \min)$，$c_{\text{norm}} = (\max_{\text{corr}} - \text{corr}) / (\max_{\text{corr}} - \min_{\text{corr}})$（corr 反向）。计算到理想点 $(0, 0)$ 的距离 $d = \sqrt{a_{\text{norm}}^2 + c_{\text{norm}}^2}$，按 $d$ 升序遍历。为避免选中相邻 checkpoint 的冗余候选，采用 5K 窗口去重：每个 checkpoint 归属 window = step // 5000，同窗只保留最先入选者，最多保留 3 个候选。60K 步若未入选则强制追加为漂移参照。Rank-1 为 canonical 候选。最终每个候选需通过温度域三联 panel 的视觉审查，人工确认后定稿。

### C.5.2 已执行选点

EP11 四个变体的选点结果如下：HotLoss（v6）canonical 8K（artifact/corr = 0.330/0.774），Stats（v8.1a）canonical 15K（0.392/0.758），Stats+PixelShuffle（v8.1b）canonical 5K（0.370/0.739），Stats+HP-FC（v9b）canonical 11K（0.339/0.777）。TGV Pareto 参考点为 (0.695, 0.916)。各变体 60K 端点的 artifact 均显著恶化——按端点上报会让每个变体交出最差 checkpoint，这正是选择协议存在的意义。

统一 harness 后的补充选点为：Hybrid（V9A）canonical 10K（harness artifact/corr = 1.762/0.719），Hybrid+Native-FC（V9C）canonical 5K（1.669/0.718），Stats+Full-FC（V9D）canonical 7K（1.726/0.771），Hybrid+ResObs（V10）采用高-λ sweep 的工作点 λ=1.2@15K（2.726/0.711；fine-window 为 hp\_corr\_input 0.922、sharp\_p95 0.987、lattice 0.0141）。Hybrid（V9A）60K 不作为交付 checkpoint，仅作为 F5 late-drift visual control。需强调 hybrid 变体与 1x 变体的 proxy 不可跨列横比（见 A.3.2）；最终 T1/T2 横评只使用 `output/ep11_unified_harness/` 的同一 harness scale。

---

## C.6 结构级指标与视觉审查

本节详述主文实验中使用的结构级评估指标与视觉审查流程。

### C.6.1 Zigzag 走线 profile 指标

以客户关心的中心走线为测量对象。在固定横截面上，对每条 profile 测表观 FWHM（full width at half maximum）与谷深（dip），报告 median FWHM/dip 及 per-profile 分布。当 per-profile 结果混杂时如实报告为「混杂」，不取巧。需要注意，$2\times$ 网格下量化步长为 5 µm，FWHM 可能饱和（主文 §5.1 多个方法同为 40 µm），区分度低时不应过度解读。

### C.6.2 双域视觉 panel

双域 panel 由两张并列图组成。第一张为高通结构图（边缘证据），使用 $HP(x) = x - G_{\sigma=5}(x)$，红/蓝分别表示相对局部背景的正/负有符号响应，白色表示无变化。第二张为普通温度图（inferno colormap，1–99 百分位裁剪），专门用于捕捉「只有边缘变亮但温度场整体失真」这类被高通图美化掉的失效模式。两图并列能让审查者同时看到高频结构证据和低频温度合理性。

### C.6.3 对齐基准一致性

重建不得劣化已通过质量筛选的锚点上的 hold-out 轮廓一致性。该检验由 EP04 对齐质量评估承担：若某方法在这些锚点上的 split-half 轮廓一致性低于对照，该方法在该区域不予采纳。
