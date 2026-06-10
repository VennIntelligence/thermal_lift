# %% [markdown]
# ## 5. EP06 Handoff Table / EP06 交接决策表
#
# EP05 的职责是决定是否进入 2x contour-level SR POC，以及 EP06 从哪种对齐策略起步。这里不把 back-projection residual 或 Tenengrad 单独作为 SR 成功证据。
#
# 前四节分别给出四类证据：displacement capacity 说明主 session 是否有可见微扫描运动；phase capacity 说明 2x 相位采样是否够用并暴露 3x/4x 风险；alignment comparison 说明哪种位移估计在 held-out 轮廓上更稳；overlay evidence 说明对齐收益是否在人眼可读的轮廓堆叠中可见。它们合在一起，才构成 EP06 的入口条件。
#
# 这里仍保持保守表述：EP05 支持“启动 2x contour-level SR POC”，不支持“已经证明 2x SR 成功”。stage command 的角色也保持为 prior/control，因为命令坐标没有测量热像中的真实局部位移，不能替代数据驱动的 alignment quality gate。

# %% [markdown]
# | 条目 | Item | 决策 | Decision | 原因 / Reason |
# |---|---|---|---|---|
# | EP06 推荐 alignment | EP06 recommended alignment | 使用 data-driven NCC init + contour refinement gate | Use data-driven NCC init plus contour refinement gate | held-out contour Chamfer 最低，同时保留数据约束的局部 anchor |
# | phase prior | Phase prior | 保留 filename affine 与 NCC init | Keep filename affine and NCC init | 二者都提供连续 2x phase coverage，并且在局部 refinement 前仍有用 |
# | 对照组 | Control groups | no alignment、stage prior、filename affine、NCC init | No alignment, stage prior, filename affine, NCC init | 这些对照用于区分显示增益、先验增益和真实数据驱动 alignment gain |
# | 高倍率状态 | High-magnification status | 3x/4x 只作风险诊断，不作可行性声明 | Treat 3x/4x as risk diagnostics only | occupancy 不能替代 PSF/SNR、forward consistency 和 split-half 证据；contour refined 高倍率 phase collapse 尤其不能证明 4x |
# | overlay 用途 | Overlay use | 只作 visual sanity appendix | Use overlay only as a visual sanity appendix | filename affine 多组更优，scanline_y20 contour 更优；overlay 不是 SR metric |
# | 主要失败风险 | Primary failure risks | 热漂移、局部轮廓歧义、PSF/SNR 上限 | Thermal drift, local contour ambiguity, PSF/SNR ceiling | 这些风险可能产生视觉上合理但 held-out contour 不稳定的 stack |
# | 验收指标 | Acceptance metrics | split-half consistency、held-out contour Chamfer、phase-bin coverage、visual contour gain | Split-half consistency, held-out contour Chamfer, phase-bin coverage, visual contour gain | back-projection residual 或 Tenengrad 单独不足以证明 SR 成功 |

# %% [markdown]
# 本节汇总的交接决策表是对位移容量（Displacement Capacity）、相位容量（Phase Capacity）、配准精度（Alignment Comparison）和图像重叠证据（Overlay Evidence）的系统性提炼，旨在确立进入 2x 轮廓级超分辨率 POC 阶段（EP06）的基准配置与准入边界。
# 决策表中的各项条款并非独立的实验结论，而是为后续重建建立的多维度质量控制门限：
# 1. **配准推荐方案**：采用数据驱动的归一化互相关（Data-driven NCC）初始化，并引入轮廓精细化门控（Contour Refinement Gate），以获得最低的 held-out 轮廓 Chamfer 距离，同时引入局部定位锚点作为物理约束。
# 2. **对照组设计**：必须包含未对齐（no alignment）、名义命令先验（stage prior）、文件名仿射变换（filename affine）及 NCC 初始对齐，以严格区分超分辨率重建中的真实对齐增益与纯粹的缩放或插值带来的伪增益。
# 3. **高倍率（3x/4x）约束**：将 3x/4x 超分辨率重建归类为风险诊断状态，明确在当前点扩散函数（PSF）及信噪比限制下其相位覆盖容易出现塌陷（Phase Collapse），不能盲目将 2x contour 级别的局部可见度增益外推到更高倍率。
# 4. **验收指标体系**：摒弃单一的 back-projection 残差或 Tenengrad 锐度指标，采用以 split-half 一致性、held-out 轮廓 Chamfer 距离、phase-bin 覆盖度及视觉轮廓可解释性增益组成的综合门控指标。
# 综上所述，当前数据与配准分析支持启动 2x contour-level SR POC 阶段；后续重建工作必须在上述多维度验收指标下进行严谨评估，避免落入以单一指标代表分辨率提升的认知误区。
