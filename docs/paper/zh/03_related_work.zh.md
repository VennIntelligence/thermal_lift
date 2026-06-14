# 第 2 章 相关工作（中文打磨稿）

> 本文是论文 **§2（对应英文权威稿 `docs/paper/03_related_work.md`）** 的中文打磨稿。
> 引用 key 沿用 `\citep/\citet`（natbib），与 `paper/aaai/refs.bib` 对齐；翻译时**保留所有 key 不变**。
> 每桶落点统一为一句对比：**无 GT 工业场景下没人量化 hallucination，我们量化它并给出机制。**

## 经典多帧超分辨（MFSR）

配准-融合式 MFSR 可追溯到频域表述 \citep{tsai1984multiframe} 与空域迭代回投影 \citep{irani1991improving}；
鲁棒/快速变体 \citep{farsiu2004fast} 确立了我们采用的 shift–PSF–decimation 观测模型。Drizzle
\citep{fruchter2002drizzle} 做通量守恒的亚像素散射到更细网格，是我们**最小假设**的融合 baseline。带 total variation
与 total generalized variation 的正则化反演 \citep{rudin1992nonlinear,bredies2010tgv} 仍是本场景最强的经典族；
我们贡献了一个**各向异性、coverage-weighted 的 TGV 变体**以应对 raster 扫描各向异性（§4）。带显式 PSF 与孔径积分的
MAP 表述 \citep{elad1997restoration,hardie1997joint} 支撑了我们的去卷积锚。**我们使用这些方法的角色不同**：
它们是学习方法的**锚与采纳门控（anchors and acceptance gates）**，不是被打倒的竞争对手。

## 深度 burst 超分辨

RGB/RAW 的深度 burst SR 包括 DBSR \citep{bhat2021deep}、NTIRE burst-SR 挑战 \citep{bhat2021ntire}、
BIPNet \citep{dudhane2022burst} 与 Burstormer \citep{dudhane2023burstormer}；卫星多帧工作如 PROBA-V 上的
HighRes-net 与 RAMS \citep{deudon2020highres,salvetti2020rams}，是少见的**实测**多帧 benchmark（尽管其 HR 真值并不完美）。
这些方法要么假设配对 HR 监督、要么假设部署域上的代理配对数据。**我们的设定彻底移除了该假设**，并展示由此产生的失效模式
（零空间漂移）与能在其下存活的东西。在架构上，我们的重建网络刻意保持朴素（带 contour 取向 loss 的 UNet）：
**架构不是研究对象，信息通路才是。**

## 热成像超分辨

热成像 SISR/SR 主要由 PBVS 热成像 SR 挑战 \citep{rivadeneira2020thermal,rivadeneira2023thermal} 与专用的
CNN/GAN 热成像 SR 网络 \citep{chudasama2020therisurnet,rivadeneira2020novel} 推动。它们大多在**渲染后的 8-bit 图像**
上工作、用合成退化、并对来自更高端相机的 hold-out HR 热成像做全参考评估。**我们在三处不同**：使用原始温度矩阵
（而非渲染视频——我们证明后者不可用作数值 SR 输入）、带实测先验的微扫描 burst、以及全程无 HR 真值。

## 合成到真实的训练与退化建模

真实世界 SR 工作通过合成退化来弥合域差 \citep{bellkligler2019blind,wang2021real,zhang2021designing}。
我们的合成平台是**物理匹配**而非对抗匹配：场景几何用亚像素覆盖抗锯齿渲染、实测 PSF 区间、探测器 box 积分、
实测噪声底、以及真实 raster 位移分布。**残余的分布差，恰恰是我们在真实数据上用漂移分析量化的对象**——
物理匹配能减小但不能消除先验驱动的风格化。

## 无 ground truth 的评估

Split-half 一致性与 Fourier Ring/Shell Correlation 在冷冻电镜与超分辨显微中是标准做法
\citep{vanheel2005fourier,nieuwenhuizen2013measuring,banterle2013fourier}，却很少被移植到学习式 SR。
我们采用带显式正/负/漂移控制组的相位分层 split-half FRC，并**诚实报告控制组**（含一个被标注为覆盖/lattice 伪影
而非光学的高频回弹）。无参考 IQA \citep{mittal2013making} 在此不够用——因为「看起来更锐」是失效模式，不是目标。

## 逆问题中的数据一致性与零空间分解

把重建分解为前向算子的 range 与 null-space 分量，在学习式逆问题中已有先例
\citep{schwab2019deep,chen2021equivariant,ulyanov2018deep,heckel2019deep}，数据一致性层在 MRI 重建中也是标准件
\citep{schlemper2018deep}。**我们的贡献是实证与诊断性的**：在一个实测工业系统上，我们展示**真实数据上的训练期漂移
几乎完全落在零空间**——观测侧 loss 贴在其底部，真实数据 artifact proxy 却劣化——并指出实用补救是输入侧证据注入与
选点协议，而非更强的一致性惩罚。我们把它打包成一个可复用的诊断（forward-loss floor 对 proxy 轨迹），服务于无 GT 部署。

> 待办（§2）：补全 `refs.bib`；核验零空间网络引用清单；若显微 SR 文献存在 FRC-for-SR 先例则补入；
> 决定 EP12 的 4x 负结果是否值得引一篇 4x 热成像 SR 主张文献作对照。
