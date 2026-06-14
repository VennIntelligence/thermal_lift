# 论文正文 — 中文打磨稿（zh/）

> **角色**：本目录是论文**正文全 7 章的中文打磨稿**，供用户 + 会话内 AI 人机共写、逐节进修。
> **与英文权威稿的关系**：英文稿 `docs/paper/02–08_*.md` 仍是迁 LaTeX 前的单一事实来源；本目录是「先中文成稿、
> 进修后再译英回填」的工作稿（与 supp A–E 的中文优先工作流一致）。**数字/引用 key/红队边界三者与英文稿严格一致。**
> **生成日期**：2026-06-14（Task D harness 落地后；§3.2 跨 session 数字已就地修正为 audit 权威值 2.91/4.16 °C）。

## 章节 → 文件映射

| 章 | 标题 | 中文打磨稿 | 英文权威稿 | 状态 |
|---|---|---|---|---|
| §1 | 引言 | `02_introduction.zh.md` | `02_introduction.md` | ✅ 全文打磨稿 |
| §2 | 相关工作 | `03_related_work.zh.md` | `03_related_work.md` | ✅ 全文（保留 `\citep` key） |
| §3 | 问题设定与标定化观测模型 | `04_problem_forward_model.zh.md` | `04_problem_forward_model.md` | ✅ 全文打磨稿 |
| §4 | 方法 | `05_method.zh.md` | `05_method.md` | ✅ 全文打磨稿 |
| §5 | 无 GT 评估协议 | `06_evaluation_protocol.zh.md` | `06_evaluation_protocol.md` | ✅ 全文打磨稿 |
| §6 | 实验与消融 | `07_experiments.zh.md` | `07_experiments.md` | ✅ 全文（含 T1/T2/F3 表） |
| §7 | 限制与结论 | `08_limitations_conclusion.zh.md` | `08_limitations_conclusion.md` | ✅ 全文打磨稿 |

## §0 标题与摘要（中文草稿）

**候选标题**：Trust but Verify：面向无 Ground Truth 热成像芯片检测的「证据有界」Burst 超分辨
（*Trust but Verify: Evidence-Bounded Burst Super-Resolution for Thermal Chip Inspection without Ground Truth*）

**摘要（≤ 200 词，AAAI 不设小节；待 §6 数字与图最终定稿后压缩成英文）**：

> 工业 LWIR 芯片检测产出原始温度矩阵，却在任何阶段都没有配对的高分辨率热成像真值，使得常规 SR 主张无法验证。
> 我们在一个参数全部实测的微扫描系统（电动台-像素角 47.6°±0.1°、PSF σ 仲裁到 0.2–0.5 LR px、噪声底 0.0724 °C、
> raster 采集结构、248 帧 clean 主 session）上研究**有界的 2x contour-level burst-SR**。三个发现构成全文：
> （i）**信息存在**——相位分层 split-half FRC 为相干信息定界（1/7 cutoff 17.0 µm），细网格 MAP-TV 去卷积把它转化为
> 有限但真实的 contour 增益（zigzag median FWHM 114→100 µm）；
> （ii）**信息送达**——在 1x 统计输入下，学习模型对最细结构的失效对 loss 设计不变（亚像素相位在网络看到前即坍塌），
> 把它经 2x drizzle 通道注入会改变 fine-window 权衡，但其本身不认证更高保真；
> （iii）**信息忠实**——合成先验训练的网络沿观测算子**零空间**漂移：forward-consistency loss 贴底，真实数据 artifact
> proxy 却单调劣化；漂移对观测锚定不可见，须由证据注入、residual-over-observation 参数化与选点协议设界。
> 由此得到一套无 GT 评估协议与一个诚实裁决：**该区不存在可认证的单一赢家**。经典锚定 TGV/MAP-TV 在可验证 proxy 上
> 最忠于观测，却在最细 contour 上印上轻微 TV 阶梯；注入证据的学习重建目视最锐，却携带最多不可验证高频并在观测保真上漂移。
> 我们报告可验证保真 proxy、一个结构 grain proxy + 双域视觉 panel、以及一个显式 task-level 的 contour 可辨偏好——
> 三者都不支持任何计量级主张。

## 进修须知（接手者必读）

1. **改论点/数字先改本目录，再同步英文稿**；迁 LaTeX 时以「进修后的中文 + 英文稿」合并译定。
2. **红队边界具否决权**（`01_outline.md` 末节「Claims to avoid」）：不得写「打败 TGV / 更干净 / 更保真 / 计量级」；
   sharp_p95 必与 lattice + 视觉并报；contour legibility 只作 task-level 主观；TGV 必带 TV-staircase caveat。
3. **两套尺度不混表**：T1/T2 全用 EP11/harness scale；TB-scale `eval_real/*` 仅 §6.2 轨迹图用、图注声明。
4. **仍待落地的写作侧缺口**（不挡进修）：① §2 `refs.bib` 与 `\citep` key 对齐；② F0 teaser / F1 合成图引用；
   ③ Task E 完成后 T1 的 TGV 真值 split/FRC、F5 第二 ROI、D.7 第二验证窗回填本目录对应处。
