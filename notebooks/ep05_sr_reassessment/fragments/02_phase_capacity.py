# %% [markdown]
# ## 2. Multi-Scale Phase Coverage Risk
#
# 这里检查每种对齐方法落入 2x、3x、4x sub-pixel phase bin 的覆盖情况。对 EP06 而言，关键不是宣称更高倍率已经成立，而是确认 2x phase diversity 是否足够支撑 contour-level POC，并把 3x/4x 的 phase collapse 风险提前暴露出来。
#
# 微扫描 SR 依赖一个基本前提：多帧之间的相对位移不能全都落在同一个像素相位上。以 2x SR 为例，一个 LR 像素会被拆成 `2 x 2 = 4` 个 sub-pixel phase bin；如果 255 帧只重复采到同一个 bin，那么它们主要提供降噪和平均，不提供 2x 网格所需的相位多样性。
#
# 这里的 phase capacity 只回答“采样几何是否足够”，不回答“SR 是否已经成功”。3x/4x 即使占满 bin，也只是 occupancy 诊断；尤其 contour refined 是局部吸附后的结果，高倍率 phase collapse 不能被解读成 4x 可行性证据。

# %%
multi_scale_phase_table = multi_scale_phase_coverage_table(summary_json, outputs["holdout_scores"])
display(multi_scale_phase_table.round({"entropy_fraction": 3}))

# %% [markdown]
# ### 📈 多尺度空间相位覆盖度（Occupancy）对比分析
#
# 评估了不同对齐策略估计的亚像素位移在 2x、3x、4x 超分辨率采样网格相位格（Phase Bins）中的空间分布覆盖度及信息熵占比（Entropy Fraction）：
# 1. **低倍率网格充盈度**：在 2x 尺度下，除不对齐（No alignment）策略外，所有位移估计方法均达成了 $100\%$ 的相位覆盖（占满全部 4 个相位区间），且信息熵比例接近 1.0，表明相位分布极其均匀。
# 2. **高倍率相位坍缩风险**：向 3x 和 4x 网格延伸时，数据驱动的轮廓精细对齐（Contour refined）在亚像素估计上呈现出明显的“相位坍缩”（Phase Collapse）特征（仅占据了极少数相位格子）。这是局部精细搜索将平移吸附在强梯度阶跃处造成的物理局限。
#
# **💡 算法决策**：多尺度相位容量分析揭示了超分辨率重构的几何容量限制。2x 重建网格在相位覆盖上展现出极高的物理充盈度，支持启动 2x 轮廓级 SR POC；而 3x 和 4x 尺度存在高度的相位中空风险，在算法设计中必须予以规避，以防引入插值伪影。

# %% [markdown]
# ### 2.1 2x Phase-Bin Capacity

# %%
phase_table = phase_capacity_table(outputs["phase_summary_2x"])
display(phase_table.round({"entropy_fraction": 3, "expected_count": 2}))

# %% [markdown]
# ### 📊 2x 网格相位容量定量分析
#
# 汇总了各配准策略在 2x 子网格相位格中的样本计数。统计数据证实，采用数据驱动对齐（NCC init）或名义电机先验，四个半像素相位格中的样本帧数均在 60 帧左右，展现出高度平衡的几何采样特征。
#
# **💡 算法决策**：主 Session 的 255 帧对于 2x 超分辨率重建在几何相位上是极其充沛且均衡的。超分辨率重建流程应选用 NCC init 与 轮廓精细化对齐（Contour refined）组合位移场作为重构的几何配准基线，以避免因局部相位缺失引入的几何失真。

# %%
show_fig(CAPACITY_DIR / "phase_bin_coverage_2x.png")

# %% [markdown]
# Figure 4: Two-times phase-bin coverage. Sub-pixel phase occupancy is shown for each alignment strategy.

# %% [markdown]
# ### 📊 2x 网格亚像素相位条形图诊断
#
# 横向堆叠条形图直观呈现了 255 帧物理帧在 2x 子网格四个空间相位的覆盖占比，右侧给出了对应的覆盖状态与信息熵指标。
#
# **💡 算法决策**：条形图证实了各数据驱动方法未在 2x 尺度上破坏空间采样多样性。该条形图诊断构成了配准质量评估的一部分，如果任何对齐参数调整导致 2x 出现空相位格（Bad Bins），算法必须自动触发预警并回退平移估计，防止逆求解发散。

# %%
phase_distribution_table = fractional_phase_distribution_table(outputs["holdout_scores"], scale=4)
display(
    phase_distribution_table.round(
        {
            "frac_x_p10": 3,
            "frac_x_median": 3,
            "frac_x_p90": 3,
            "frac_y_p10": 3,
            "frac_y_median": 3,
            "frac_y_p90": 3,
        }
    )
)

# %% [markdown]
# ### 📊 高分辨率网格小数相位分布与覆盖矩阵
#
# 展示了各对齐方法估计位移的小数部分（Fractional Phase）在 4x 奈奎斯特子网格下的 $4 \times 4$ 样本频数分布矩阵。
# 1. **采样孔洞量化**：Contour refined 算法在 4x 覆盖矩阵上存在多处频数为零的格点，证实了物理采样相位的局部空缺。
# 2. **连续对齐先验**：NCC init 与 Filename affine 在 4x 空间上分布相对连续，适合为超分辨率逆向求解提供稳健的初始平移引导。
#
# **💡 算法决策**：小数相位分布矩阵直接否定了利用当前数据集推进 4x 超分辨率的可行性。后续的 EP06 算法决策应当将物理重建目标锁定于采样容量完整覆盖的 2x 轮廓增强，而将 4x 置于高风险警告级别。
