# Phase 4 调研报告：多帧红外微扫描增强方法

> 目标：围绕当前 EP04 pipeline 的三个痛点——覆盖率低、细小结构漏检、缺少全图增强结果——调研可落地的多帧红外增强、低对比度热边缘检测、组合 pipeline 与无 GT 评价方法。以下判断基于你给出的数据条件：16 帧同视场 LWIR float 温度矩阵、σ≈1 px PSF、NCC 位移约 0.1 px RMS、法线相位覆盖约 0.35–0.44 px、噪声底 0.0724°C、不能跨 session/repeat 混合。

---

## 0. 结论先行

### 0.1 最推荐路线

**不要把主线押在 4× 全图 SR 或纯 deep learning 上。**

最有可能成功的组合是：

```text
16 帧同 session 温度矩阵
  ↓
鲁棒配准：highpass NCC / phase correlation + 小范围位移再优化
  ↓
物理 forward model 重建：PSF + 像素积分 + 位移不确定性 + TV/TGV 正则
  ↓
输出 1.5× / 2× 增强温度场 + 不确定度图
  ↓
多尺度 edge / ridge / line candidate 检测
  ↓
对候选边界继续使用 multi-frame erf ESF 或 ridge model 做亚像素定位
  ↓
split-half / FRC / forward-consistency / edge-MTF 自检
```

### 0.2 增强倍率判断

| 输出目标          | 工程可行性 | 判断                                                |
| ------------- | ----: | ------------------------------------------------- |
| **1.5× 全图增强** |     高 | 最适合作为论文主结果。可望让低对比度内部结构更清楚，并提高 edge candidate 覆盖率。 |
| **2× 全图增强**   |     中 | 可以尝试，但应明确是“2× 输出网格 + 有限频带增强”，不是光学分辨率翻倍。           |
| **3× 全图增强**   |     低 | 可能产生好看的图，但科学可信度风险较高。                              |
| **4× 全图 SR**  |    很低 | 不建议作为主张。当前 MTF、位移相位覆盖和 0.1 px 配准误差都不支持稳定 4×。      |

这个判断主要来自你们自己的 MTF 和位移约束：0.3 cyc/px 仍有约 17% 信号，16 帧可降噪约 4×，所以 1.5–2× 有物理空间；但 0.4–0.5 cyc/px 已经极弱，且法线相位覆盖不足，4× 不应作为目标。

---

## 1. 问题 1：多帧红外图像增强方法

### 1.1 推荐使用的观测模型

建议把每条 16 帧 scanline 或每个同 session raster group 写成显式物理模型：

[
y_k = D , H_{\sigma} , W_{\delta_k} x + \beta_k + n_k
]

其中：

| 符号             | 含义                           |
| -------------- | ---------------------------- |
| (x)            | 待重建的高采样温度场，单位仍为 °C           |
| (y_k)          | 第 (k) 帧原始 640×480 float 温度矩阵 |
| (W_{\delta_k}) | 第 (k) 帧相对位移                  |
| (H_\sigma)     | 光学 PSF + 像素积分，初值可设 σ=1.0 px  |
| (D)            | 从高采样网格到原始像素网格的采样 / 积分        |
| (\beta_k)      | 帧内温度 offset 或慢漂移项            |
| (n_k)          | 近似 Gaussian 温度噪声             |

关键点是：**(\delta_k) 不能当作已知真值**。你的 NCC 位移约 0.1 px RMS，因此更合理的写法是：

[
\delta_k = \delta^{NCC}*k + \Delta \delta_k, \quad
\Delta \delta_k \sim \mathcal{N}(0, \Sigma*\delta)
]

即把 NCC 位移作为 prior，而不是硬约束。

---

### 1.2 多帧反卷积：可用，但不宜单独作为主方法

经典多帧重建思想很早就明确了：如果多帧之间存在已知且足够准确的相对位移，可以利用 imaging process 的知识通过类似 back-projection 的方式提高分辨率；Irani & Peleg 属于这一类经典代表。([ScienceDirect][1])

但你的数据里，**位移不是精确已知**，这会让单纯 Wiener / Richardson-Lucy / blind deconvolution 变得危险：

| 方法                            | 优点                       | 主要风险                                         | 对本项目建议                   |
| ----------------------------- | ------------------------ | -------------------------------------------- | ------------------------ |
| Multi-frame Wiener            | 简单、可解释、适合 Gaussian noise | 对 PSF 和位移误差敏感；容易低估不确定性                       | 可作为 mild deblur baseline |
| Richardson-Lucy               | 边缘增强明显                   | 假设 Poisson 更自然；温度矩阵不是 photon count；易 ringing | 不建议作为主方法                 |
| Blind deconvolution           | 可同时估 PSF                 | ill-posed；容易把位移误差解释成纹理                       | 只估低维 σ，不做完全 blind        |
| MAP multi-frame deconvolution | 可纳入 PSF、位移、不确定度、正则项      | 实现复杂，需要调参                                    | **推荐主线**                 |

Hardie 等人的经典工作把红外和可见图像的 multi-frame SR 放到 MAP 框架中，并联合估计 registration parameters 与 HR image；这点与你们“位移不可靠”的处境非常贴近。([eCommons][2]) Pickup 等人的 Bayesian multi-frame SR 进一步强调：可以对未知 registration parameters 做 marginalization，而不是只取一个点估计；这对 0.1 px RMS 位移误差尤其重要。([NeurIPS Papers][3])

Farsiu 等人的 robust SR 方法也值得重点看：它用 L1 data term 和 bilateral-TV 类 prior，目标是对 motion / blur estimation error 更鲁棒，并保持 sharp edge。

**推荐实现形式：**

[
\min_{x,\Delta\delta,\beta,\sigma}
\sum_k \rho \left(
D H_{\sigma} W_{\delta^{NCC}_k+\Delta\delta_k}x + \beta_k - y_k
\right)

* \lambda_1 TV(x)
* \lambda_2 TGV(x)
* \sum_k |\Delta\delta_k|*{\Sigma*\delta^{-1}}^2
* \frac{(\sigma-1.0)^2}{\tau_\sigma^2}
  ]

其中：

* (\rho)：Huber 或 L1，抗 outlier；
* (TV/TGV)：边缘保持正则；
* (\Delta\delta_k)：只允许在 NCC 位移附近小范围修正；
* (\sigma)：可在 0.8–1.2 px 范围内 grid search 或弱优化；
* 输出倍率优先设为 1.5×，再尝试 2×。

---

### 1.3 多帧超分辨率：IBP / POCS / Papoulis-Gerchberg 可做 baseline，但不是最稳主线

经典 multi-frame SR 包括：

| 类别                                 | 代表方法                        | 适合做什么                   |
| ---------------------------------- | --------------------------- | ----------------------- |
| IBP                                | Iterative Back-Projection   | 最直观 baseline，检查多帧是否真有增益 |
| POCS                               | Projection Onto Convex Sets | 加入强度范围、数据一致性等约束         |
| Papoulis-Gerchberg                 | 频域带限重建                      | 适合测试频域先验，但对条件敏感         |
| Drizzle / nonuniform interpolation | 天文图像常见思想                    | 适合先做相位覆盖可视化             |

POCS、IBP、频域类方法的共同问题是：**它们需要较好的位移相位覆盖和配准精度**。Ur & Gross 的 nonuniform interpolation SR 思想强调，多帧样本需要避免“几乎重合”的采样，否则合并信息有限。([Tel Aviv University][4]) 综述和后续比较也指出，POCS 简单但可能收敛慢、解不唯一，IBP 直观但也不提供唯一解。([パターン認識国立キー研究所][5])

对你们的数据，最大问题是：

```text
法线方向相位覆盖只有 0.35–0.44 px
NCC 位移误差约 0.1 px RMS
stage command 位移不可信
```

这意味着：

* **单条 16 帧固定 Y scanline 不足以支持 4× SR**；
* **2× 可以尝试，但 conditioning 不强**；
* 如果要做 2D 全图增强，应优先利用同 session 的 raster 内部相位，而不是只依赖一条 scanline；
* 若只能用固定 Y 的 16 帧，建议定位为“1.5–2× edge-aware enhancement”，不要称为 full optical SR。

---

### 1.4 正则化重建：这是最适合当前项目的主路线

红外 SR 的难点与可见光不同：低对比度、纹理少、传感器噪声、固定图案噪声、热扩散和复杂退化都会让普通自然图像 SR 假设失效。红外 SR 综述也强调，IR degradation 往往比简单 Gaussian blur / downsampling 更复杂。([arXiv][6])

#### 推荐正则项优先级

| 正则项                                 | 推荐程度 | 原因                         |
| ----------------------------------- | ---: | -------------------------- |
| Huber-TV                            |    高 | 边缘保持，抑制噪声，容易实现             |
| TGV                                 |    高 | 比 TV 更少 staircasing，更适合温度场 |
| Bilateral-TV / BTV                  |    高 | 对边缘友好，Farsiu SR 体系中常见      |
| Hessian / second-order sparse prior |    中 | 对平滑热场有意义                   |
| Nonlocal / BM3D / PnP               |    中 | 可试验，但需防止幻觉纹理               |
| 完全 blind sparse coding              |    低 | 调参多，不稳定                    |
| GAN / perceptual loss               |   很低 | 不适合科学温度场                   |

Cascarano 等人的 thermal image TV SR 工作值得重点参考：它面向单帧或多帧 thermal images，强调保持 radiometric content，并且不需要训练数据或外部先验。([MDPI][7]) 另一类红外 SR 工作使用高阶稀疏 TV、PnP、RED 等框架，可以作为 advanced baseline，但这些方法往往参数更多、计算更重。([MDPI][8])

**本项目最推荐：**

```text
Primary: Huber data fidelity + TV/TGV + shift prior + PSF prior
Secondary: robust L1/BTV SR
Experimental: PnP/RED with conservative denoiser
```

---

### 1.5 2023–2026 最新方法：DL 很多，但要谨慎使用

近年红外 / 热像 SR 的 deep learning 方法增长很快。PBVS 2025 thermal SR challenge 已经使用 large-scale thermal benchmark，参赛方法以 transformer / hybrid architecture 为主，目标倍率甚至到 x8 / x16。([CVF Open Access][9]) 但这些 benchmark 方法多面向视觉质量或 RGB-guided thermal SR，不一定保留真实温度物理量。

值得关注的方向：

| 方向                             | 代表工作 / 思想                                                                                     | 对本项目价值                                         |
| ------------------------------ | --------------------------------------------------------------------------------------------- | ---------------------------------------------- |
| Edge-enhanced transformer      | TESR 用 edge extraction 辅助红外 SR，目标是改善边缘恢复与 ringing 问题。([Nature][10])                           | 可借鉴“边缘辅助分支”，但不建议直接当科学重建主线                      |
| Efficient IR SR                | EIRSR 等工作利用红外图像相邻相关性、结构 prior 做高效 SR。([Nature][11])                                           | 可作为对比或轻量学习 baseline                            |
| Physics-guided IR SR           | ThesIS 强调 thermal distribution 保真和 high-frequency ambiguity。([AAAI Publications][12])         | 思路很 relevant：把物理一致性写进 loss                     |
| Zero-shot / no-GT thermal SR   | 红外 NDT zero-shot SR 工作针对热扩散导致的低分辨率和低敏感度问题。([arXiv][13])                                       | 可作为实验支线，尤其适合无 HR GT 场景                         |
| Self-supervised micro-scanning | RIMO 结合红外 micro-scanning optical system 和 self-supervised blur / HR estimation。([Nature][14]) | 与你们“微扫描 + 无 GT”相近，但硬件/位移条件不同                   |
| Deep Image Prior               | DIP 说明随机初始化 CNN 本身可作为 image prior，用于 denoising / SR / inpainting，无需训练集。([arXiv][15])          | 可做 zero-shot regularizer，但必须受 forward model 约束 |

**建议定位：**

* 主论文方法：**physics-based MAP / TV / shift-uncertainty reconstruction**。
* DL 方法：放在 supplementary 或 exploratory comparison。
* 不建议用 GAN / perceptual SR 作为主图，因为它可能“看起来更清楚”，但温度场和边界坐标不可信。

---

## 2. 问题 2：低对比度热边缘检测方法

### 2.1 为什么 Otsu 必然漏掉中心十字形 / L 形结构

Otsu 是全局阈值方法，适合双峰灰度分布或强对比轮廓。你的样品内部细结构有几个不利条件：

1. 对比度低；
2. 结构尺度接近或低于 PSF 宽度；
3. 背景温度缓慢变化；
4. 外边界强信号会主导全局阈值；
5. 细线结构可能更像 ridge / valley，而不是闭合 contour。

因此，后续不应从 “Otsu contour” 开始，而应从 **multi-frame enhanced candidate map** 开始。

---

### 2.2 推荐的多帧增强边缘检测流程

```text
16 帧温度矩阵
  ↓
同 session 内 robust registration
  ↓
native-grid aligned mean / median / trimmed mean
  ↓
noise map + split-half consistency map
  ↓
background removal / local contrast normalization
  ↓
multi-scale edge + ridge detection
  ↓
candidate consensus filtering
  ↓
subpixel localization
```

建议生成三类图：

| 图                          | 作用             |
| -------------------------- | -------------- |
| aligned mean map           | 提升 SNR，展示整体温度场 |
| band-pass / DoG / LoG map  | 强化局部热结构        |
| edge-ridge probability map | 作为后续亚像素定位候选    |

---

### 2.3 热像专用 / 相关边缘检测方法

Fabijańska 2012 面向 heat-emitting specimens 的 subpixel edge detection，用 Gaussian function 重建粗边附近的 gradient，以确定 subpixel position；你们 Phase 3 已实现相关思路，仍可作为 thermal edge baseline。([EuDML][16])

后续可补充以下类别：

| 方法类别                               | 适合结构             | 对本项目建议                     |
| ---------------------------------- | ---------------- | -------------------------- |
| Gaussian gradient / local gradient | 热边界、模糊 step edge | 作为 ESF 拟合前的候选检测            |
| LoG / DoG zero-crossing            | 模糊边缘、低对比细结构      | 推荐加入 multi-scale candidate |
| Morphological top-hat              | 细线、局部亮/暗结构       | 对中心十字 / L 形尤其值得试           |
| Structure tensor                   | 线状结构方向估计         | 用于 ridge / line confidence |
| Steger / Frangi ridge detector     | 细线、中心线           | 用于非闭合热细线                   |
| Active contour / graph cut         | 连续内部轮廓           | 作为闭合结构 fallback            |

红外增强方向的 multi-scale top-hat / adaptive cutoff threshold 方法也值得尝试，其目标正是改善红外图像低清晰度和细节增强问题。([Nature][17]) LWIR unsupervised segmentation 综述也把 thresholding、edge-based、region-based、clustering、texture analysis 等方法作为热像分割的主要类别；这说明仅靠 Otsu 阈值过窄。([The Science and Information Organization][18])

---

### 2.4 亚像素边缘检测 + 定位方法

#### A. 继续保留当前 erf ESF 模型

对真正的 step-like 热边界，当前的联合 erf ESF 拟合仍然是最物理、最可信的方法。它直接对应：

```text
真实热台阶
  ↓
PSF / 像素积分模糊
  ↓
观测 ESF
```

建议把它从“只处理 Otsu 检出的 contour”扩展为：

```text
任何 multi-scale candidate edge
  ↓
判断是否 step-like
  ↓
若是，则进入 multi-frame erf ESF fitting
```

#### B. Zernike moments

Ghosal & Mehrotra 的 Zernike moment 方法把理想 2D step edge 建模为背景、step size、边缘距离和方向四个参数，可实现 subpixel edge detection。([ScienceDirect][19]) 后续 Sobel-Zernike 方法先用 Sobel 找候选点，再用 Zernike moment 精定位，目标是保持精度同时提高速度。([ScienceDirect][20])

对你们的数据，Zernike 的用途更适合是：

```text
candidate localization baseline / second opinion
```

而不是主定位模型。原因是 thermal edge 已被 σ≈1 px PSF 平滑，理想 step assumption 不如 erf ESF 物理。

#### C. Partial area effect

Partial area effect 方法可同时估计 edge position、orientation、curvature、contrast，并在 noisy images 中达到 subpixel accuracy。([ScienceDirect][21])

但对你们的 LWIR 数据要注意：partial area effect 更像“像素面积混合模型”，而你们主要退化来自光学 PSF + 像素积分 + 热扩散。因此它可以作为候选定位 baseline，但主模型应仍是 PSF-aware ESF / ridge fitting。

#### D. Ridge / line model

中心十字形、L 形、细线不一定是 step edge。它们可能更像：

* narrow hot ridge；
* narrow cold valley；
* 两条相邻 edge 的合成；
* PSF-blurred line source。

因此建议新增一个模型：

[
T(s) = T_0 + A \exp \left( -\frac{(s-s_0)^2}{2\sigma_{line}^2} \right)
]

或者双边模型：

[
T(s) = T_0 + A \left[
\operatorname{erf}\frac{s-s_1}{\sqrt{2}\sigma}
----------------------------------------------

\operatorname{erf}\frac{s-s_2}{\sqrt{2}\sigma}
\right]
]

这样可以覆盖：

| 结构                  | 推荐模型                               |
| ------------------- | ---------------------------------- |
| 外轮廓 / 内轮廓 step edge | erf ESF                            |
| 细热线 / 冷线            | Gaussian ridge / valley            |
| 有宽度的细条              | double-erf                         |
| 十字交叉点               | 2D Gaussian ridge / junction model |

---

## 3. 问题 3：推荐 Pipeline 架构

## 3.1 主 pipeline

```text
输入:
  16 帧同 session / 同 scanline 或同 raster group 的 float 温度矩阵

Step 0. 数据分组与质量控制
  - 严禁跨 session 混合
  - 严禁跨 repeat 混合
  - 保留原始 °C 浮点值
  - 坏点 / 饱和 / 异常帧 mask

Step 1. 帧间配准
  - highpass NCC / phase correlation 初估
  - 局部 ROI NCC 估计 shift covariance
  - 拟合低阶 motion model：translation / affine / local correction
  - outlier frame rejection
  - 得到 δ_NCC 和 Σδ

Step 2. Native-grid 融合 baseline
  - aligned mean
  - aligned median
  - trimmed mean
  - odd/even split-half mean
  - 输出 baseline denoised map

Step 3. 1.5× / 2× MAP 重建
  - forward model: shift + PSF + pixel integration + downsample
  - Huber / L1 data fidelity
  - TV / TGV / BTV 正则
  - 位移 prior: δ = δ_NCC + Δδ
  - PSF prior: σ≈1.0 px
  - 输出 enhanced temperature map + residual map + uncertainty map

Step 4. 多尺度结构候选检测
  - DoG / LoG / top-hat
  - adaptive Canny / hysteresis
  - structure tensor
  - ridge / valley detector
  - odd/even consistency filtering

Step 5. 亚像素定位
  - step-like candidate → multi-frame erf ESF fitting
  - ridge-like candidate → Gaussian ridge / double-erf fitting
  - junction candidate → local 2D parametric fitting
  - 输出 coordinate + uncertainty

Step 6. 验证
  - forward consistency
  - split-half map agreement
  - FRC / local FRC
  - edge-MTF / ESF σ
  - boundary repeatability
  - coverage gain
```

---

## 3.2 每个步骤的候选方法

| Step               | 经典方法                                                       | 新近 / 红外相关方法                                                          | 推荐选择                                                  |
| ------------------ | ---------------------------------------------------------- | -------------------------------------------------------------------- | ----------------------------------------------------- |
| 配准                 | NCC、phase correlation、Lucas-Kanade、Hardie MAP registration | self-supervised feature alignment、RIMO-like micro-scanning alignment | **NCC + covariance + 小范围 MAP refine**                 |
| 初始融合               | average、median、drizzle、nonuniform interpolation            | robust temporal fusion                                               | **aligned trimmed mean + split-half mean**            |
| SR / deconvolution | IBP、POCS、Papoulis-Gerchberg、Wiener、MAP SR                  | thermal TV SR、PnP/RED、physics-guided IR SR                           | **MAP + TV/TGV + shift prior**                        |
| 低对比增强              | DoG、LoG、top-hat、Canny                                      | adaptive multi-scale morphology、edge-enhanced IR SR                  | **multi-scale edge/ridge probability map**            |
| 亚像素定位              | Zernike、partial area、Gaussian gradient、ESF fitting         | local gradient thermal edge、Canny-Zernike variants                   | **erf ESF + ridge model 双路线**                         |
| 评价                 | residual、split-half、edge width、MTF                         | FRC / rFRC、no-reference thermal IQA                                  | **forward consistency + split-half + FRC + edge-MTF** |

---

## 3.3 推荐实验矩阵

建议不要一次上复杂模型，而是按以下矩阵推进：

| 实验                             | 目标                 | 成功标准                                      |
| ------------------------------ | ------------------ | ----------------------------------------- |
| E1: Native aligned mean        | 验证 16 帧降噪是否显著揭示细结构 | 中心十字 / L 形在 split-half 中同时出现              |
| E2: 1.5× MAP-TV                | 主候选方法              | residual 接近噪声底，edge-MTF 改善，无明显 ringing    |
| E3: 2× MAP-TV/TGV              | 测试上限               | split-half FRC 支持更高 cutoff                |
| E4: robust L1-BTV SR           | 对比 Farsiu 类鲁棒方法    | 对位移扰动不敏感                                  |
| E5: IBP / POCS baseline        | 经典 baseline        | 证明 naive SR 不如 MAP                        |
| E6: DIP / zero-shot DL         | 探索性                | 必须通过 forward consistency 和 split-half     |
| E7: multi-scale edge candidate | 提升覆盖率              | 内轮廓 / 细结构 candidate 数上升，false positive 可控 |

---

## 3.4 预期增强倍率

以下是针对你们这批数据的工程判断，不是文献直接给出的通用结论。

| 倍率       |       成功概率 | 可能结果                                                          | 主要限制                         |
| -------- | ---------: | ------------------------------------------------------------- | ---------------------------- |
| **1.5×** | 高，约 70–85% | 温度场更平滑，低对比结构更清晰，edge candidate 覆盖率提升                          | 依赖配准质量和正则强度                  |
| **2×**   | 中，约 45–65% | 对 10–20 µm apparent thermal structures 有帮助；可生成论文 before/after | 0.1 px shift error 会限制真实高频恢复 |
| **3×**   | 低，约 15–25% | 视觉上可能更锐，但定量可信度不足                                              | MTF、相位覆盖、噪声都不支持              |
| **4×**   |    很低，<10% | 不建议作为科学主张                                                     | 0.5 cyc/px MTF 约 0.7%，基本不可恢复 |

建议论文表述：

> We reconstruct a moderately oversampled apparent temperature field at 1.5×–2× grid density, constrained by a measured PSF and validated by split-half consistency, rather than claiming true 4× optical super-resolution.

---

## 3.5 风险评估与 fallback

| 风险                  | 表现                         | 影响           | Fallback                                     |
| ------------------- | -------------------------- | ------------ | -------------------------------------------- |
| 位移误差过大              | SR 后出现 ghost / double edge | 全图增强失败       | 降到 native aligned fusion + edge localization |
| 相位覆盖不足              | 2× 没有真实新增信息                | SR 只是插值      | 主结果用 1.5×，报告 FRC cutoff                      |
| PSF σ 不准            | 过锐或过平滑                     | MTF claim 不稳 | σ=0.8–1.2 px sensitivity sweep               |
| TV staircasing      | 温度场出现块状假结构                 | 影响细结构判断      | 换 Huber-TV / TGV / BTV                       |
| RL / deblur ringing | 边缘附近振铃                     | false edge   | early stopping，或不使用 RL                       |
| DL hallucination    | 生成不存在纹理                    | 科学风险高        | 只作 supplementary，不作主结论                       |
| 细结构不是可分辨实体          | 十字 / L 形只是热扩散 signature    | 机械尺寸解释错误     | 称为 apparent thermal structure，不声称机械 5 µm 分辨  |

---

## 4. 问题 4：无 ground truth 时的评价指标

### 4.1 不建议使用 PSNR / SSIM

没有 HR ground truth 时，PSNR / SSIM 不适用。即使拿某个插值图当 reference，也只是在评价算法是否接近插值，而不是是否更真实。

更适合本项目的是四类 self-consistency 指标：

```text
A. forward consistency
B. split-half consistency
C. frequency-domain resolution estimate
D. task-level edge localization stability
```

---

### 4.2 Forward consistency

对增强结果 (x)，用同一个 forward model 回投到每一帧：

[
\hat{y}*k = D H*{\sigma} W_{\delta_k} x + \beta_k
]

然后检查：

| 指标                               | 含义                        |
| -------------------------------- | ------------------------- |
| residual RMS                     | 是否接近 0.0724°C 噪声底         |
| residual spatial autocorrelation | 是否还有未解释结构                 |
| residual vs edge distance        | 是否边缘处系统性偏差                |
| residual histogram               | 是否接近 Gaussian / Laplacian |
| per-frame residual               | 是否某些帧配准失败                 |

如果增强图真的解释了多帧观测，residual 应该接近噪声且没有明显结构；如果 residual 里还残留十字 / L 形，那说明重建没捕捉到真实信号。

---

### 4.3 Split-half validation

这是非常适合你们当前数据的自检方法。

推荐做法：

```text
16 帧
  ↓
分成 A/B 两组：奇偶帧，或保持 phase coverage 平衡的两组
  ↓
分别重建 x_A, x_B
  ↓
比较：
  - 温度差图
  - edge candidate overlap
  - boundary coordinate difference
  - FRC cutoff
  - local confidence map
```

需要注意：不要简单奇偶分组后破坏位移相位覆盖。更稳的是先根据估计位移相位聚类，再让 A/B 两组都覆盖类似 phase range。

Fourier Ring Correlation/Fourier Shell Correlation 原本就是用两个独立噪声图像估计有效分辨率的方法，可以直接用于 split-half 重建；相关文献也说明 FRC 可从图像本身估计 effective resolution / effective PSF，并监控 deconvolution 过程。([Nature][22])

---

### 4.4 FRC / local FRC

推荐输出：

| 指标                               | 报告方式                           |
| -------------------------------- | ------------------------------ |
| global FRC cutoff                | 全图或 ROI 的有效截止频率                |
| local FRC / rolling FRC          | 不同区域的局部分辨率                     |
| before vs after FRC              | native aligned mean 与 MAP 重建比较 |
| odd/even FRC                     | split-half 一致性                 |
| FRC-derived effective resolution | 转换成 µm，而不是只给 px                |

这比“看起来更清楚”可靠得多。

---

### 4.5 Edge sharpness / MTF improvement

可以用外轮廓和高 SNR 内轮廓作为 natural slanted-edge target：

1. 在原始单帧、aligned mean、1.5× MAP、2× MAP 上分别取强边；
2. 沿法线采样 ESF；
3. 拟合 erf，得到 σ 或 FWHM；
4. 对 ESF 求导得到 LSF；
5. Fourier transform 得到 MTF；
6. 报告 MTF50、MTF10、ESF σ、overshoot。

Slanted-edge / SFR 方法本来就是 ISO 12233 中常用的成像系统锐度测量方式；MTF 表示 contrast 随空间频率的变化，edge-derived MTF 的优点是目标小、重复性较好、对采样相位不那么敏感。([Imatest][23])

需要警惕：

```text
MTF improvement ≠ 真实光学分辨率突破
```

如果 deconvolution 太激进，MTF 会看似提升，但 ringing / overshoot 也会上升。因此必须同时报告：

* ESF overshoot；
* residual consistency；
* split-half edge coordinate difference；
* shift / PSF sensitivity。

---

### 4.6 任务级指标：覆盖率提升

当前 EP04 已有很好的 QA 体系。建议新增“增强后检测覆盖率”指标：

| 指标                       | 定义                                         |
| ------------------------ | ------------------------------------------ |
| candidate segment count  | 增强后检测出的候选边段数                               |
| passed segment count     | 通过 split-half / CRB / PSF sensitivity 的边段数 |
| inner contour pass rate  | 内轮廓通过率是否从 19.6% 提升                         |
| fine structure recall    | 中心十字 / L 形 candidate 是否稳定出现                |
| false positive proxy     | odd/even 不一致 candidate 比例                  |
| coordinate repeatability | 增强前后边界定位 split-half error                  |
| CRB ratio                | 是否仍接近理论下界                                  |
| sensitivity to σ         | σ=0.8/1.0/1.2 px 时坐标变化                     |

最终论文可以报告：

```text
Before:
  Otsu → 466 segments → 120 passed, pass rate 25.8%

After:
  multi-frame enhancement + multi-scale candidate
  → N candidates
  → M passed
  → inner contour pass rate improved from 19.6% to ...
  → fine structure candidates validated by split-half
```

---

## 5. 推荐优先级

### Priority 1：立刻做

```text
P1. aligned mean / median / trimmed mean baseline
P2. split-half aligned mean
P3. DoG / LoG / top-hat candidate map
P4. edge-ridge probability map
P5. 把当前 erf ESF fitting 从 Otsu contour 扩展到 candidate edge
```

理由：实现成本低，最可能马上提升覆盖率，并直接解决“Otsu 漏检细结构”的问题。

---

### Priority 2：主论文方法

```text
P6. 1.5× MAP-TV/TGV reconstruction
P7. shift uncertainty prior
P8. σ sensitivity sweep
P9. forward consistency residual
P10. FRC + edge-MTF validation
```

理由：这是最稳的“全图增强 + 定量验证”组合。

---

### Priority 3：探索性方法

```text
P11. robust L1/BTV SR
P12. PnP / RED conservative denoiser
P13. DIP / zero-shot thermal SR
P14. edge-enhanced transformer as external comparison
```

理由：可增加论文深度，但不应成为主结论依赖。

---

## 6. 最终推荐方案

### 6.1 论文主张建议

建议把目标从：

```text
4× thermal super-resolution
```

调整为：

```text
uncertainty-aware multi-frame enhancement and super-localization
of apparent thermal structures
```

或者：

```text
moderately oversampled, physically constrained reconstruction
for thermal boundary and fine-structure localization
```

这样既保留创新性，又不被 4× SR 的物理限制拖累。

---

### 6.2 最可能成功的主方法名称

可以命名为类似：

```text
Uncertainty-Aware Multi-Frame Thermal Reconstruction, UA-MFTR
```

核心特点：

1. 使用 float temperature matrix，不做 8-bit 图像增强；
2. 使用 measured PSF；
3. 显式建模 subpixel shift uncertainty；
4. 使用 TV/TGV 保持温度场物理平滑性；
5. 输出 enhanced temperature field，而不仅是 sparse edge points；
6. 用 split-half、FRC、forward residual、edge-MTF 无 GT 验证。

---

### 6.3 最小可发表结果包

```text
Figure 1. 原始单帧 / aligned mean / 1.5× MAP / 2× MAP before-after
Figure 2. 中心十字形、L 形结构 zoom-in
Figure 3. edge-ridge probability map 与 Otsu contour 对比
Figure 4. enhanced candidate → passed boundary map
Figure 5. split-half difference map
Figure 6. FRC curve before/after
Figure 7. ESF / MTF improvement on strong edges
Table 1. 覆盖率提升：outer / inner / fine structures
Table 2. 定位重复性：split-half / CRB ratio / σ sensitivity
Table 3. ablation：mean fusion vs IBP vs MAP-TV vs MAP-TGV
```

---

## 7. 文献与方法 shortlist

| 主题                                | 重点文献 / 方法                                       | 为什么相关                                                                       |
| --------------------------------- | ----------------------------------------------- | --------------------------------------------------------------------------- |
| Classic multi-frame SR            | Irani & Peleg back-projection SR                | 说明多帧位移 + imaging model 可提高分辨率，但依赖准确位移。([ScienceDirect][1])                  |
| MAP SR + registration             | Hardie et al. MAP joint registration / HR image | 与红外 FPA undersampling 和联合配准非常相关。([eCommons][2])                             |
| Bayesian registration uncertainty | Pickup et al. Bayesian SR                       | 适合你们的位移不确定场景。([NeurIPS Papers][3])                                          |
| Robust SR                         | Farsiu L1 + BTV                                 | 对 motion / blur error 更鲁棒，适合做主 baseline。                                    |
| Thermal TV SR                     | Cascarano et al. thermal TV SR                  | 保持 radiometric content，不依赖训练数据。([MDPI][7])                                  |
| IR sparse / RED SR                | HOGS4 / OGSTV / RED / PnP                       | 可作为 advanced regularization comparison。([MDPI][8])                          |
| IR SR survey                      | 2025 infrared SR survey                         | 总结红外 SR 的低对比、噪声和退化模型问题。([arXiv][6])                                         |
| Edge-enhanced DL SR               | TESR                                            | 可借鉴 edge auxiliary branch，但需防 hallucination。([Nature][10])                  |
| Physics-guided IR SR              | ThesIS                                          | 与 thermal physics + high-frequency ambiguity 高度相关。([AAAI Publications][12]) |
| Self-supervised micro-scanning    | RIMO                                            | 与 micro-scanning + no HR GT 接近，但硬件条件不同。([Nature][14])                       |
| Thermal edge detection            | Fabijańska Gaussian gradient                    | 已实现，可继续作为 thermal edge baseline。([EuDML][16])                               |
| Partial area effect               | Trujillo-Pino et al.                            | 可作为 subpixel edge localization baseline。([ScienceDirect][21])               |
| Zernike moments                   | Ghosal & Mehrotra                               | 经典 subpixel edge method，适合作 comparison。([ScienceDirect][19])                |
| FRC validation                    | Fourier ring correlation                        | 无 GT 分辨率估计，非常适合 split-half。([Nature][22])                                   |
| Edge-MTF                          | Slanted-edge SFR / MTF                          | 可评价 before/after sharpness，但需防 ringing。([Imatest][23])                      |

[1]: https://www.sciencedirect.com/science/article/pii/104996529190045L "Improving resolution by image registration - ScienceDirect"
[2]: https://ecommons.udayton.edu/ece_fac_pub/14/ "
\"Joint MAP Registration and High Resolution Image Estimation Using a Se\" by Russell C. Hardie, Kenneth J. Barnard et al.
"
[3]: https://papers.nips.cc/paper/3037-bayesian-image-super-resolution-continued "Bayesian Image Super-resolution, Continued"
[4]: https://cris.tau.ac.il/en/publications/improved-resolution-from-subpixel-shifted-pictures/ "
        Improved resolution from subpixel shifted pictures
      \-  Tel Aviv University"
[5]: https://nlpr.ia.ac.cn/2010papers/%E5%BC%80%E6%94%BE%E8%AF%BE%E9%A2%98/%E5%9B%BD%E9%99%85%E5%88%8A%E7%89%A9/Xuelong%20Li%2C%20A%20multi-frame%20image%20super-resolution%20method.pdf?utm_source=chatgpt.com "A multi-frame image super-resolution method"
[6]: https://arxiv.org/html/2212.12322v5 "Infrared Image Super-Resolution: A Systematic Review and Future Trends"
[7]: https://www.mdpi.com/2072-4292/12/10/1642 "Super-Resolution of Thermal Images Using an Automatic Total Variation Based Method | MDPI"
[8]: https://www.mdpi.com/1424-8220/19/23/5139 "Infrared Image Super-Resolution Reconstruction Based on Quaternion and High-Order Overlapping Group Sparse Total Variation"
[9]: https://openaccess.thecvf.com/content/CVPR2025W/PBVS/html/Rivadeneira_Thermal_Image_Super-Resolution_Challenge_Results_-_PBVS_2025_CVPRW_2025_paper.html "CVPR 2025 Open Access Repository"
[10]: https://www.nature.com/articles/s41598-024-66302-8 "Edge-enhanced infrared image super-resolution reconstruction model under transformer | Scientific Reports"
[11]: https://www.nature.com/articles/s41598-025-16698-8 "Infrared image super resolution with structure prior from uncooled infrared readout circuit | Scientific Reports"
[12]: https://ojs.aaai.org/index.php/AAAI/article/view/38381 "
		Thermal-Physics Guided Infrared Image Super-Resolution with Dynamic High-Frequency Amplification
							\| Proceedings of the AAAI Conference on Artificial Intelligence
			"
[13]: https://arxiv.org/abs/2509.10902 "[2509.10902] Real-Time Super-Resolution Imaging System Based on Zero-Shot Learning for Infrared Non-Destructive Testing"
[14]: https://www.nature.com/articles/s41598-025-09834-x "Research on self-supervised super resolution restoration algorithm based on reflective micro-scanning optical system | Scientific Reports"
[15]: https://arxiv.org/abs/1711.10925 "[1711.10925] Deep Image Prior"
[16]: https://eudml.org/doc/244059 "EUDML  |  A survey of subpixel edge detection methods for images of heat-emitting metal specimens"
[17]: https://www.nature.com/articles/s41598-024-66423-0 "Advanced enhancement technique for infrared images of wind turbine blades utilizing adaptive difference multi-scale top-hat transformation | Scientific Reports"
[18]: https://thesai.org/Downloads/Volume14No6/Paper_138-Review_of_Unsupervised_Segmentation_Techniques.pdf "Review of Unsupervised Segmentation Techniques on Long Wave Infrared Images"
[19]: https://www.sciencedirect.com/science/article/pii/003132039390038X?utm_source=chatgpt.com "Orthogonal moment operators for subpixel edge detection"
[20]: https://www.sciencedirect.com/science/article/abs/pii/003132039390038X?utm_source=chatgpt.com "Orthogonal moment operators for subpixel edge detection"
[21]: https://www.sciencedirect.com/science/article/abs/pii/S0262885612001850 "Accurate subpixel edge location based on partial area effect - ScienceDirect"
[22]: https://www.nature.com/articles/s41467-019-11024-z "Fourier ring correlation simplifies image restoration in fluorescence microscopy | Nature Communications"
[23]: https://www.imatest.com/imaging/validating_slanted_edge/ "Validating the Imatest slanted-edge calculation | Imatest"

