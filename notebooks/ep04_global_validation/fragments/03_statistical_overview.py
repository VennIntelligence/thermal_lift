# %% [markdown]
# ## Step 2 — 全局 Segment 质量分布
#
# 本节按 outer/inner 与 EP03 质量标签统计 precision、CRB ratio、SNR 和 pass/fail。这里的 precision 指 split-half localization repeatability，不是最终形状重建精度。

# %%
from thermal_core.ep04 import segment_quality_distribution_table

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
# ### 📊 基于先验特征的定位锚点通过率评估
#
# 定量分析了外轮廓与内轮廓在不同质量等级标签（A/B/C/D 类，基于热对比度、法线方向及局部曲率等物理先验特征划分）下的质量门控表现：
# 1. **先验标签与通过率的相关性**：A类和B类等高置信度先验片段在实际数据中的通过率（Pass Rate）明显高于C类和D类。这证明了物理先验对实际图像特征质量的强相关指导性。
# 2. **几何定位稳定性**：通过质量门控的特征段，其折半定位偏差的中位数显著低于 0.05 像素，展现出高水平的物理定位重复度（Repeatability）。
#
# **💡 算法决策**：实验数据证明，基于 EP03 构建的局部特征质量标签在反映实际图像配准稳定性方面具有高度一致性。因此，后续算法将优先利用A类和B类强特征段作为对齐计算的锚点，对低质量特征进行屏蔽。

# %%
show_fig("global_segment_quality_distribution.png")

# %% [markdown]
# Figure 1: Global segment quality distribution. Segment precision, CRB ratio, SNR, and pass counts are summarized by contour type.

# %% [markdown]
# ### 📊 定位精度与理论极限误差的分布特性
#
# 四联图表汇总分析了定位重复精度（Split-half Precision）、CRB 比例中位数、信噪比及通过门控锚点数在内外轮廓中的二维分布情况：
# 1. **分布集中度**：通过门控的段其定位偏差极其靠近 Cramér-Rao 理论下界（CRB Ratio 集中于 1.0 附近），说明其空间随机抖动主要受到热探测器随机噪声的限制。
# 2. **长尾长振幅失效**：未通过门控的段则在精度分布上呈现明显的长尾，意味着其估计值已被温漂、光栅扫描换行延迟等模型外误差所主导。
#
# **💡 算法决策**：为了确保超分辨率图像对齐的数值稳定性，算法决策必须在空域对齐中排除处于长尾长振幅区间的低置信度 `segment × scanline` localization rows。门控剔除能够收窄配准偏差的尾部，将全局亚像素运动残差控制在可接受范围内。
