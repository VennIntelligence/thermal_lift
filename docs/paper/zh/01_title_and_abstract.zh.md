# 标题与摘要（中文打磨稿 + 英文权威稿）

> 本文件对应论文的 标题 / Abstract。英文为权威投稿稿，中文为打磨/校对稿。
> 叙事重心（2026-06-14 锁定）：「细节更多却在漂移」= 学习式 SR 在无 GT 工业场景下沿观测算子零空间漂移；
> 三问框架（Q1 信息存在 / Q2 信息送达 / Q3 信息忠实）为支撑，结论边界为 "no GT-certifiable dominant method"。
> 注意：标题不再用 "Evidence-Bounded" 当卖点（避免暗示"交付了一个被认证的有界最优解"）；
> 卖点是 零空间幻觉诊断 + 诚实裁决（no GT-certifiable dominant method）。

---

## 标题（Title）

Detailed but Drifting: Null-Space Hallucination in Ground-Truth-Free Thermal Burst Super-Resolution

中文直译（仅供内部对照，不投稿）：
《细节更多，却在漂移：无真值热成像多帧超分辨中的零空间幻觉》

> 用词说明：
>
> - Detailed（而非 *Sharper*）— 描述"表观结构/细节增多"，不暗示真实分辨率提升；与 "but Drifting" 对应本文的保真性问题。
> - Drifting — 直接对应本文贡献专名 null-space drift；比 "Hallucinating" 更适合描述训练过程中的方向性漂移。
> - Null-Space Hallucination — 在副标题中明确机制：新增细节主要位于观测算子零空间。
> - Ground-Truth-Free（中文：无真值）— 限定整个问题设定：采集时与任何时刻都没有 HR 热成像真值。

---

## Abstract（英文权威稿）

Long-wave infrared (LWIR) thermography resolves sub-degree temperature signatures of defects and process variations in semiconductor inspection. In this modality, a more detailed-looking reconstruction does not imply genuinely higher resolution: diffraction bounds the recoverable bandwidth, high-end thermal optics are costly, and paired high-resolution ground truth is unavailable at any stage. We study a calibrated 2× micro-scan burst super-resolution system around three questions: whether the multi-frame observation carries recoverable high-frequency information, whether learned models actually receive it, and whether newly added detail remains anchored to the observations.

We identify a failure mode we term *null-space drift*. The imaging chain is modeled as an observation operator mapping a latent high-resolution temperature field to recorded low-resolution frames; its null space consists of high-frequency structures that alter the reconstruction while leaving observations nearly unchanged. On real data, networks trained on physics-matched synthetic bursts produce new detail whose growth is consistent with this null or near-null space: forward-consistency error remains pinned at the noise floor while ground-truth-free artifact and correlation diagnostics steadily degrade. We further show that for drift in the exact or $\epsilon$-null space, any bounded observation-side anchoring—finite weights, arbitrary bounded frequency bands—is structurally insensitive; effective remedies must come from priors, input-side evidence, or checkpoint-selection protocols rather than stronger consistency penalties.

Under ground-truth-free evaluation combining phase-stratified split-half consistency, observation-side drift diagnostics, and pre-specified Pareto checkpoint selection (detailed in the supplement), classical variational reconstruction is most faithful to verifiable observations, while evidence-injected learning yields richer apparent contour detail alongside more unverifiable high-frequency content—no single method is certifiably dominant. This work characterizes null-space hallucination in ground-truth-free thermal super-resolution and provides an actionable framework for balancing detail and fidelity when high-resolution ground truth is lacking.

---

## 摘要（中文打磨稿）

长波红外（LWIR）热成像能够测出缺陷与工艺偏差引起的亚摄氏度温度差异，在半导体检测中有重要应用。在这一场景下，重建图像的表观细节更多并不等于真实分辨率的提升，因此不能仅凭视觉观感判断超分辨是否成功。这一问题还受到物理与数据条件的多重约束，衍射会抑制光学通带外的信息，高端热成像光学成本高昂，配对的高分辨率热真值也无法获得。本文研究一个经过标定的 2× 微扫描多帧超分辨系统，围绕三个问题展开：观测序列是否携带可恢复的高频信息、学习模型是否真正接收到这些信息、其新增细节是否锚定于观测。
本文识别出一种我们称为零空间漂移的现象。我们把成像过程建模为一个观测算子，它将假想的高分辨率温度场映射为实际记录的低分辨率帧。这一算子存在零空间，即一类会改变重建图像、却几乎不改变任何已记录观测的高频结构。我们发现，在物理匹配的合成数据上训练的网络迁移到真实数据后，其新增细节主要表现为与这一零/近零空间一致的漂移。把重建结果重新投影回低分辨率观测域，前向一致性误差始终停在噪声与模型误差决定的下界附近；在算子层面，我们进一步证明：对落入精确或 $\epsilon$-零空间的漂移，有界的观测侧锚定（有限权重、任意有界频带）不敏感，补救只能来自先验、输入证据或选点协议。与此同时，在真实数据上不依赖真值的伪影与相关性诊断却持续劣化。这说明网络生成了更多表观高频细节，但它们并未受到观测的充分约束。
基于无真值的评估手段——包括受控的 split-half 一致性、观测侧漂移诊断与 Pareto 选点（详见附录）——经典变分重建对可验证的观测最为保真，而证据注入式学习重建给出表观细节更丰富的可见轮廓，却同时带来更多无法验证的高频内容。本文据此揭示并刻画了无真值热成像超分辨中的零空间幻觉，并给出可操作的诊断框架，为这类系统在缺乏高分辨率热真值时如何权衡细节与保真提供了依据。

---

## 待办（标题/摘要）

- [ ] 摘要英文稿已与中文打磨稿对齐（2026-06-14）；定档 venue 后再按版式压缩。
- [x] 标题已锁定（旧 `01_outline.md` working title 列表已随该文件删除，本文件为唯一标题来源）。