# %% [markdown]
# ## Step 3 — Anchor Coverage Map
#
# 本节把 segment gate 映射回温度参考帧，并把通过 segment 在各条 X scanline 上的支持情况拆成单独图。目标是告诉 EP06 哪些 contour/scanline 可以作为 alignment anchor，哪些空间区域缺锚点。

# %%
show_fig("anchor_coverage_map.png")

# %% [markdown]
# Figure 2: Anchor coverage map. Passed and rejected localization anchors are overlaid on the reference thermal frame.

# %% [markdown]
# ### 🗺️ 物理对齐锚点的空间覆盖地图
#
# 在低分辨率温度背景图像上叠加了外轮廓与内轮廓定位锚点（Alignment Anchor）的通过/拒绝分布：
# 1. **空间分布拓扑**：外边框处表现出高度连续的合格锚点分布（蓝色标记），表明这些区域的几何边界明确、对比度高，可为亚像素对齐提供稳定的物理锚定。
# 2. **内部稀疏特征**：内部芯片结构的合格锚点较为稀疏，说明芯片内部的热辐射结构受到了较大的低通热平滑作用，导致局部定位重复度下降。
#
# **💡 算法决策**：该覆盖地图直接指导了后续超分辨率重构中感兴趣区域（ROI）的选择和配准策略。配准计算将优先提取地图中蓝色锚点密集区域的平移残差，而在灰色未通过门控区域则采取弱约束或几何投影插值以保证重建的拓扑连贯性。

# %%
show_fig("anchor_scanline_support.png")

# %% [markdown]
# Figure 3: Anchor scanline support. Row-level anchor support is summarized across physical scanlines.

# %% [markdown]
# ### 📊 扫描线（Scanline）对锚点对齐的连续性支持评估
#
# 统计了已通过 Segment 级门控的锚点在各个不同物理 Y 坐标扫描线（Scanline）上的 Row-level 门控实际通过率：
# 1. **连续支持性特征**：彩色柱体（实际通过数量）与浅灰色柱体（被评估数量）高度吻合的扫描线，代表该物理坐标处的几何特征对多帧配准提供了最稳健、最连续的支持。
# 2. **局域退化扫描线**：部分扫描线上通过率显著下降，代表该轴线位置受局部热流漂移或无结构热场影响严重，容易在配准中引入异常值。
#
# **💡 算法决策**：此统计用于排除或降权异常扫描线。在后续重建（EP06）与几何配准（EP05）中，算法将基于此分布选择稳定支持的扫描线作为主要的配准源（Alignment Inputs），并将通过率较低的扫描线降权或用作独立的交叉验证测试集（Held-out Testlines）。

# %%
coverage_table = (
    pd.concat(
        [
            outer_results.assign(contour="outer"),
            inner_results.assign(contour="inner"),
        ],
        ignore_index=True,
    )
    .assign(pass_bool=lambda df: df["pass_fail"].astype(str).str.lower().isin(["true", "1", "yes"]))
    .groupby(["contour", "scanline_y_um"], dropna=False)
    .agg(
        evaluated_rows=("segment_id", "count"),
        passed_rows=("pass_bool", "sum"),
        row_pass_rate=("pass_bool", "mean"),
    )
    .reset_index()
)
display(coverage_table.assign(row_pass_rate=lambda df: (100.0 * df["row_pass_rate"]).round(1)))

# %% [markdown]
# ### 📊 基于扫描线坐标的对齐定量性能分析
#
# 定量汇总了各个物理 $Y$ 坐标扫描线上的 Row-level 评估帧数、通过门控帧数以及局部通过率。数据展示了物理扫描线在不同物理空间截面上几何特征质量的非均一性。
#
# **💡 算法决策**：空间上的非均一性是系统固有的热流不平衡和扫描时序延迟的物理体现。后续算法必须对不同物理 $Y$ 坐标截面处的亚像素位移估计赋予不同的置信度权重（Confidence Weights），避免将局部不良截面的估计残差外推至全局重建中。
