# %% [markdown]
# ## 4. EP06 Handoff Table / EP06 交接决策表
#
# EP05 的职责是决定是否进入 2x contour-level SR POC，以及 EP06 从哪种对齐策略起步。这里不把 back-projection residual 或 Tenengrad 单独作为 SR 成功证据。
#
# 前三节分别给出三类证据：phase capacity 说明 2x 相位采样是否够用；alignment comparison 说明哪种位移估计在 held-out 轮廓上更稳；overlay evidence 说明对齐收益是否在人眼可读的轮廓堆叠中可见。三者合在一起，才构成 EP06 的入口条件。
#
# 这里仍保持保守表述：EP05 支持“启动 2x contour-level SR POC”，不支持“已经证明 2x SR 成功”。stage command 的角色也保持为 prior/control，因为命令坐标没有测量热像中的真实局部位移，不能替代数据驱动的 alignment quality gate。

# %% [markdown]
# | 条目 | Item | 决策 | Decision | 原因 / Reason |
# |---|---|---|---|---|
# | EP06 推荐 alignment | EP06 recommended alignment | 使用 data-driven NCC init + contour refinement gate | Use data-driven NCC init plus contour refinement gate | held-out contour Chamfer 最低，同时保留数据约束的局部 anchor |
# | phase prior | Phase prior | 保留 filename affine 与 NCC init | Keep filename affine and NCC init | 二者都提供连续 2x phase coverage，并且在局部 refinement 前仍有用 |
# | 对照组 | Control groups | no alignment、stage prior、filename affine、NCC init | No alignment, stage prior, filename affine, NCC init | 这些对照用于区分显示增益、先验增益和真实数据驱动 alignment gain |
# | 主要失败风险 | Primary failure risks | 热漂移、局部轮廓歧义、PSF/SNR 上限 | Thermal drift, local contour ambiguity, PSF/SNR ceiling | 这些风险可能产生视觉上合理但 held-out contour 不稳定的 stack |
# | 验收指标 | Acceptance metrics | split-half consistency、held-out contour Chamfer、phase-bin coverage、visual contour gain | Split-half consistency, held-out contour Chamfer, phase-bin coverage, visual contour gain | back-projection residual 或 Tenengrad 单独不足以证明 SR 成功 |

# %% [markdown]
# > **数据说明**: 结论表汇总 EP06 推荐方法、保留对照组、主要失败风险和验收指标。它是前面 phase、alignment、overlay 三类证据的 handoff，而不是新的实验结果。
# > **怎么读**: `decision` 列给出 EP06 应采用或保留的选择，`reason` 列说明为什么这个选择符合当前证据边界。推荐方法、对照组、风险和验收指标需要一起读，避免只拿“推荐 alignment”去扩大成最终 SR 成功声明。
# > **正常/异常理解**: 正常 handoff 应同时包含主方法和 control groups，因为没有 no alignment/stage/filename 对照，就无法区分真实 alignment gain 与显示或先验带来的变化。若 EP06 只看单一 sharpness 指标，或只用 stage command 当真值，会重新落入“展示倍率不是 SR 证据”和“stage command 不是 ground truth”的风险。
# > **核心发现**: 数据支持启动 EP06 2x contour-level SR POC；对齐主线应为 data-driven contour/NCC refinement，stage/filename 继续作为 prior 和对照。EP06 的验收必须围绕 split-half consistency、held-out contour Chamfer、phase-bin coverage 和可解释的视觉轮廓增益共同判断。
