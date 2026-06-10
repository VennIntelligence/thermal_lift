# %% [markdown]
# ## Step 4 — 内轮廓通过率与失败原因
#
# 内轮廓对应芯片内部结构/形状，是客户关心的 SR 目标区域。本节解释为什么它们在 localization-only gate 下通过率较低，并明确这些区域不是被放弃的区域。

# %%
from thermal_core.ep04 import failure_reason_table

inner_reason_table = failure_reason_table(inner_results, contour="inner")
display(
    inner_reason_table.assign(
        share_of_failed_rows=lambda df: (100.0 * df["share_of_failed_rows"]).round(1)
    )
)

# %% [markdown]
# ### 📊 内部芯片结构定位失效机制定量分析
#
# 定量分析了芯片内部轮廓在质量门控中的多维度失效原因分布（包含互相关不可靠、法线相位覆盖不足、边缘拟合宽度超限、折半定位偏差过大等指标）：
# 1. **多指标耦合判定**：由于 Row-level 门控判定具有多标签并发性，同一段对齐失败可同时触发多个门限阈值限制，因而统计占比之和超过 100%。
# 2. **主导失效机制**：数据表明，内部结构的失效主要由边缘物理宽度超限（`sigma_out_of_range`）和定位可重复度偏低（`split_half_high`）主导，反映了热传导导致的非单阶跃边缘模糊特征对局域定位模型的负面作用。
#
# **💡 算法决策**：失效机制分析证实了内部结构不适宜作为强几何约束真值。但这绝不意味着放弃该区域。后续的超分辨率重构算法（EP06/EP10）应将这些通过率低的内部结构定位为超分辨率的“待增强区”，而在几何配准阶段则完全依赖外部强锚点进行刚性/仿射漂移补偿，防止失效定位引入几何形变。

# %%
show_fig("inner_failure_reasons.png")

# %% [markdown]
# Figure 4: Inner contour failure reasons. Gate failures are grouped by quality label and dominant rejection mechanism.

# %% [markdown]
# ### 📊 内部结构特征定位通过率与门控瓶颈分析
#
# 该柱状图汇总了内轮廓各质量先验特征段的通过率及主要失效因素构成：
# 1. **通过率趋势**：符合物理规律，高对比度的先验特征段（A类、B类）在内轮廓中仍然保持着最高的相对通过率。
# 2. **瓶颈判定**：局部边缘模型的失配（如拟合宽度异常与奇偶帧拆分不一致）依然是限制内轮廓亚像素估计的主要瓶颈。
#
# **💡 算法决策**：定位门控瓶颈为后续超分辨率评价提供了“分层质量门限”依据。在进行 2x contour-level 超分辨率效果评估时，应区分强定位区与普通增强区，只用通过门控的强锚点评估几何配准残差，而在全体内轮廓区域上使用图像轮廓陡峭度（如 ESF 宽度收缩量）和结构相似度进行超分辨率效果的综合评估。
