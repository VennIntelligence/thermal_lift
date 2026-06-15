# 第 2 章 相关工作（中文打磨稿）

> 本文是论文 §2（唯一事实来源，英文散文稿已删除） 的中文打磨稿。
> 引用 key 沿用 `\citep/\citet`（natbib），与 `paper/aaai/refs.bib` 对齐；翻译时保留所有 key 不变。

多帧超分辨（MFSR）从频域表述 \citep{tsai1984multiframe} 与空域迭代回投影 \citep{irani1991improving}
发展至鲁棒快速变体 \citep{farsiu2004fast}，形成了以 shift–PSF–decimation 为核心的观测模型；
Drizzle \citep{fruchter2002drizzle} 以通量守恒的亚像素散射提供最小假设融合，
TV/TGV 正则化反演 \citep{rudin1992nonlinear,bredies2010tgv} 与显式 PSF 的 MAP 表述
\citep{elad1997restoration,hardie1997joint} 则给出了经典族中最强的重建质量。
深度 burst SR 将上述流程端到端化：DBSR \citep{bhat2021deep}、BIPNet \citep{dudhane2022burst}、
Burstormer \citep{dudhane2023burstormer} 在 RGB/RAW 上取得显著增益；
卫星场景中 HighRes-net 与 RAMS \citep{deudon2020highres,salvetti2020rams} 是少见的实测多帧 benchmark。
热成像 SR 主要由 PBVS 挑战 \citep{rivadeneira2020thermal,rivadeneira2023thermal} 与专用 CNN/GAN
\citep{chudasama2020therisurnet,rivadeneira2020novel} 推动，但多在渲染后 8-bit 图像上用合成退化训练，
并依赖更高端相机的 hold-out HR 做全参考评估。上述三条线索共享一个前提：训练或评估阶段存在配对真值（或其代理）。
当部署场景——如本文所面对的工业热成像 raster 扫描——根本无法获取 HR 真值时，
该前提失效，由此引出两个未被充分讨论的问题：如何在无 GT 条件下评估重建质量，
以及学习式方法在此条件下会产生何种系统性失效。真实世界 SR 通过合成退化弥合域差 \citep{bellkligler2019blind,wang2021real,zhang2021designing}，但物理匹配的合成管线只能缩小、不能消除先验驱动的分布偏移。

对于无 GT 评估，split-half 一致性与 Fourier Ring Correlation（FRC）在冷冻电镜与超分辨显微中已是标准工具
\citep{vanheel2005fourier,nieuwenhuizen2013measuring,banterle2013fourier}，
却很少被移植到学习式 SR；无参考 IQA \citep{mittal2013making} 则无法区分真实锐化与幻觉式锐化。
对于失效机制，逆问题文献已将重建分解为前向算子的 range 与 null-space 分量 \citep{schwab2019deep,chen2021equivariant,ulyanov2018deep,heckel2019deep}，数据一致性层在 MRI 重建中亦为标准件 \citep{schlemper2018deep}，
但尚无工作在真实无 GT 的工业成像系统上实证量化学习式 SR 的零空间漂移并给出诊断手段。本文正是填补这一空白。
