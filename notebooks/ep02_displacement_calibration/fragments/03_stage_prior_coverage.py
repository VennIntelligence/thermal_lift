# %% [markdown]
# ## 2. Stage Prior Coverage in Detector Coordinates
#
# `configs/stage_calibration.json` defines a coordinate prior: stage X/Y commands are mapped to detector-space shifts by theta and the 10 um/pixel sampling pitch. This prior is useful for coverage, initialization, and regularization; it is not the alignment truth.
#
# 这里的“coverage”不是说画面里所有结构都已经对齐正确，而是说 stage command 在 detector 坐标中提供了哪些候选采样相位。
# 对 2x SR 来说，理想情况是样本不要都落在同一个整数像素相位附近；半像素 phase bin 都有样本，后续重建才有机会利用多帧互补信息。

# %%
from thermal_core.ep02 import add_stage_prior, phase2_table, stage_prior_summary

show_fig("ep02_stage_prior_coverage.png")

# %% [markdown]
# Figure 2: Stage prior coverage. Commanded stage coordinates are mapped into detector-space displacement phases.

# %% [markdown]
# ### 🗺️ 探测器空间物理位移先验分布
#
# 将主 Session 内的扫描指令坐标 $(x_{\text{um}}, y_{\text{um}})$ 映射至探测器像素平移空间 $(\Delta x, \Delta y)$，能够直观评估采样点的空间相位覆盖完整度：
# 1. **位移先验云图**：指令坐标经过旋转角 $\theta = 47.6^\circ$ 和采样间距 $10.0\,\mu\text{m/pixel}$ 变换后，在像素空间呈倾斜的光栅网格分布，物理覆盖范围在 $\pm 4$ 像素左右。
# 2. **2x 半像素相位 bin 覆盖**：将这些先验位移投影到 $2 \times 2$ 超分辨率网格的半像素子网格相位区间中。结果表明四个子相位（即 $(0,0)$、$(0.5,0)$、$(0,0.5)$、$(0.5,0.5)$ 邻域）均获得了充足的物理采样点覆盖。
#
# **💡 算法决策**：半像素相位的全面物理覆盖是实现 2x 亚像素超分辨率重建的几何先决条件。位移先验分布证明了原始光栅扫描在硬件设计上具备提供空间采样互补信息的能力，可作为后续重构算法的初始估计与优化边界。

# %%
prior_stats = stage_prior_summary(
    frame_audit,
    theta_deg=REFERENCE_THETA_DEG,
    pixel_size_um=PIXEL_SIZE_UM,
)
main = frame_audit[frame_audit["session"].eq(2)]
phase2_bins = phase2_table(
    add_stage_prior(main, theta_deg=REFERENCE_THETA_DEG, pixel_size_um=PIXEL_SIZE_UM)
)
display(prior_stats)
display(phase2_bins)

# %% [markdown]
# ### 📊 亚像素相位覆盖定量统计
#
# 统计数据汇总了名义平移的范围及四个半像素子网格中的采样帧数。数据证实，全部四个 Phase Bins 的采样帧数分布较为均匀，未出现严重的相位偏置或漏采样。
#
# **💡 算法决策**：这组定量统计验证了步进电机控制轨道的几何完整性。但在后续重建中，绝不能将上述名义位移直接作为重构插值的真值。必须将这些名义位移作为初始先验，交由数据驱动的对齐算法进行亚像素微调与精化。
