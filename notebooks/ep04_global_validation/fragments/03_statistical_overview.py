# %% [markdown]
# ## Step 2 — 全局 Segment 质量分布
#
# 本节按 outer/inner 与 EP03 质量标签统计 precision、CRB ratio、SNR 和 pass/fail。这里的 precision 指 split-half localization repeatability，不是最终形状重建精度。

# %%
quality_distribution = segment_quality_distribution_table(outer_segment_summary, inner_segment_summary)
display(
    quality_distribution.assign(
        pass_rate=lambda df: (100.0 * df["pass_rate"]).round(1),
        median_split_half_px=lambda df: df["median_split_half_px"].round(4),
        median_crb_ratio=lambda df: df["median_crb_ratio"].round(2),
        median_snr=lambda df: df["median_snr"].round(1),
        median_phase_coverage_px=lambda df: df["median_phase_coverage_px"].round(3),
    )
)

# %% [markdown]
# > **数据说明**: 表格按轮廓类型和质量标签聚合 segment 级结果，`pass_rate` 表示 anchor gate 通过率。
# > **读法**: `quality_label` 来自 EP03 的热对比、法向投影和曲率等先验；`pass_rate` 来自 EP04 的实际多帧 localization gate。读表时应先看同一 contour 内 A/B/C/D 的趋势，再比较 outer 与 inner 的差异。
# > **正常/异常理解**: 正常情况下 A/B 标签比 C/D 更容易通过，但不会 100% 通过，因为 A/B 只是候选质量，不保证 NCC 轨迹、ESF 拟合和 split-half 都稳定。`curvature_proxy`、`normal_projection` 这类 proxy 是几何/热像代理量，不是显微镜标注；异常高或低只能提示该段是否适合定位，不能证明真实结构形状。
# > **对本 Episode 的意义**: EP03 的质量标签是有用先验，但不能替代 EP04 的数据驱动质量门控；EP06 应同时保留外轮廓稳定 anchor 和内轮廓目标区域。

# %%
fig = plot_global_segment_quality_distribution(outer_segment_summary, inner_segment_summary)
save_fig(fig, "global_segment_quality_distribution.png")

# %% [markdown]
# > **图表说明**: 四联图分别展示 split-half precision、CRB ratio、SNR 和 pass/fail 数量，外轮廓/内轮廓分组显示。
# > **读法**: split-half 面板越靠近 0 越好，表示奇偶帧独立估计的边缘位置更接近；CRB ratio 越低表示实测重复性越接近噪声理论下限，但过度解读单个点没有意义；SNR 面板只说明热边缘对比；pass/fail 面板显示这些条件综合后的门控结果。
# > **正常/异常理解**: 正常模式是 pass 段在 split-half 和 CRB ratio 上更集中，fail 段更分散。若 fail 段 SNR 不低，通常说明问题在 NCC peak、相位覆盖、ESF 宽度、split-half 或拟合稳定性；不能把它解释为结构不存在。NCC 是红外局部纹理的相关性，不是位移真值，也不是光学真值。
# > **对本 Episode 的意义**: EP04 的核心产物是可审计的 anchor quality distribution；它支持后续 alignment gate 设计，不直接给出 SR 形状恢复结论。
