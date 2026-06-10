# %% [markdown]
# ## Step 1 — 加载 EP04 Anchor Gate 结果
#
# 本节展示外轮廓与内轮廓的 `segment x scanline` 质量门控表。数值由 `scripts/build_ep04_cache.py` 预生成；Notebook 只读取 CSV/JSON 缓存。

# %%
from thermal_core.ep04 import combined_anchor_summary_table

anchor_summary_table = combined_anchor_summary_table(outer_segment_summary, inner_segment_summary)
display(anchor_summary_table)

# %% [markdown]
# > **表格说明**：上表是 EP04 localization anchor 的英文汇总表，统计单位为 `segment × scanline`，不是 SR 输入帧数。
# > **数据分布**：Outer/Inner 分别汇总外轮廓和内轮廓的 segment 数、通过率、split-half、CRB ratio 和 SNR。
# > **核心发现**：这些数值用于判断哪些局部轮廓段可作为 alignment anchor 候选，不能作为 248 帧 clean SR 输入全集的成败结论。

# %%
# 对照的中文表
anchor_summary_table_zh = anchor_summary_table.copy()
metric_translation = {
    "Total segments": "总片段数",
    "A-class segments": "A类片段数",
    "Passed anchor segments": "通过的定位锚点数",
    "Anchor pass rate": "定位通过率",
    "A-class anchor pass rate": "A类定位通过率",
    "Passed median split-half [px]": "通过的半折(split-half)中位数 [px]",
    "Passed median CRB ratio": "通过的 CRB 比例中位数",
    "Median input SNR": "输入信噪比(SNR)中位数",
}
anchor_summary_table_zh["Metric"] = anchor_summary_table_zh["Metric"].map(metric_translation).fillna(anchor_summary_table_zh["Metric"])
anchor_summary_table_zh = anchor_summary_table_zh.rename(columns={
    "Metric": "指标",
    "Outer": "外轮廓",
    "Inner": "内轮廓",
    "Combined": "合并",
})
display(anchor_summary_table_zh)

# %% [markdown]
# ### 📊 定位锚点性能与质量门控统计
#
# 该表格对比了外轮廓与内轮廓在质量门控各个维度下的汇总统计：
# 1. **定位通过率（Anchor Pass Rate）**：外轮廓由于具有更高的热对比度与更为清晰的几何拓扑，其定位通过率显著高于内部轮廓。内轮廓通过率较低并不表示内部结构不存在，而是因为内部结构更受热扩散及局部复杂热场演化的影响，在定位稳定性（Split-half）和几何陡峭度（ESF fitted width）上难以达到强对齐锚点的严苛要求。
# 2. **CRB 比例中位数**：通过门控的锚点，其折半误差（Split-half）与 Cramér-Rao 理论下界的比值（CRB Ratio）表现出良好的物理一致性，证明其定位偏差已接近理论物理限度。
#
# **💡 算法决策**：数据统计验证了本算法的门控筛选规则能有效区分稳定几何结构与高风险热漂移。后续 2x contour-level 重建算法应优先使用通过门控的强锚点作为 alignment anchor 输入，而将未通过门控但包含物理结构的内轮廓段作为可见性增强目标，以防低置信度 localization row 破坏整网格的清晰度。

# %%
global_table = pd.concat(
    [
        pd.Series(outer_global_summary, name="outer").rename_axis("metric"),
        pd.Series(inner_global_summary, name="inner").rename_axis("metric"),
    ],
    axis=1,
).reset_index()
display(global_table)

# %% [markdown]
# ### ⚙️ 全局对齐批处理元数据审计
#
# 批处理元数据记录了本次定位锚点评估的物理输入参数、文件源及计算耗时。
#
# **💡 算法决策**：审计核对确认外轮廓与内轮廓运行于完全一致的物理标定旋转角（$\theta = 47.6^\circ$）、相同的物理像元间距（$10.0\,\mu\text{m/pixel}$）以及同一噪声基底。一致的输入基准确保了内外轮廓性能评估在时空维度上的可比性，防止由于标定不一致引入人为对齐伪影。
