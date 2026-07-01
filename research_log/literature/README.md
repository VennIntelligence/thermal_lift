# Literature Notes — 外部论文学习 & 对比本项目

每读一篇外部论文，在此目录写一条独立笔记：忠实摘要 + **与 thermal_lift 的对比** + **对我们的启示/候选实验**。
目标不是复述论文，而是"这篇对我们 2× contour-level LWIR 微扫描 SR 有什么用"。

## 索引

| 文件 | 论文 | 一句话 | 对我们的核心启示 |
|---|---|---|---|
| [2022_bsrt_burst_sr.md](2022_bsrt_burst_sr.md) | BSRT (arXiv:2204.08332, CVPRW'22 NTIRE 冠军) | 多帧 burst 4× **保真**SR：学习对齐(光流+可变形卷积)+Swin 融合 | 4× 靠**多帧亚像素融合(物理)+MTF余量**，非幻觉；**对齐是杠杆**；我们把多帧外包给了固定融合+物理DC，学习部分只看一张聚合图 |
| [2022_restormer.md](2022_restormer.md) | Restormer (arXiv:2111.09881, CVPR'22) | **单图**通用复原 transformer：MDTA 通道注意力(对分辨率线性) + GDFN 门控前馈，纯 L1 保真 | 是"prox/先验 block"候选；但 **MDTA 仍是全局空间归约 → 同 EP07 的 SE/GroupNorm extent-shift 风险**，用前必须"训练即推理尺度"或窗口化 |
| [2025_2026_restoration_sota.md](2025_2026_restoration_sota.md) | 2024–2026 SOTA 综述 (3 路子代理交叉核对) | 领域分叉为**保真轨 vs 感知-生成轨**；保真 SOTA 已平台期；2025 信号=混合+高效全局+幻觉审计 | 头条扩散/GAN 都在"会编"的错的一边，**对 PSNR/SSIM/LPIPS/NR 指标都不可见**；我们 unrolled hard-DC 站对边；高性价比加：真实PSF退化、已知微扫描偏移、频域/温度一致性、**测量空间幻觉审计+守恒量+不确定度** |
| [2026_info_budget_and_why_phone_4x.md](2026_info_budget_and_why_phone_4x.md) | 信息预算（离线模拟，真实常数） | 多帧亚像素信号真实但只 **+1~+1.5 dB @2×，噪声封顶**；手机 4× 余量=**可见光衍射余量+混叠采样+高SNR** 的物理馈赠 | "压成一帧"确实吃掉 ~1.5 dB，但瓶颈是 **prox 容量不是输入**(→ACL-042 D-E)；我们抠 1–3 dB 不是管线差，是 **LWIR 衍射受限+光学近Nyquist+低SNR** 只剩这么多；抬天花板靠采集/硬件不靠网络 |

## 约定
- 命名：`<年份>_<短名>.md`。
- 每条含：① 忠实摘要(带数字/出处) ② 与本项目对比表 ③ 启示 ④ 候选实验 ⑤ 待核实项。
- 物理/scale 结论必须回扣 `AGENTS.md` 的物理常数与硬教训(#6 先用 MTF/SNR 定边界)。
