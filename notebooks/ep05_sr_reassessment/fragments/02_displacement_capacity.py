# %% [markdown]
# ## 1. Alignment and Displacement Capacity Preface
#
# EP05 先检查主 session 是否真的提供了可见位移和二维相位覆盖，再讨论具体的对齐方法。这里读的是 `output/ep05_sr_reassessment/` 的 displacement reassessment 产物：它们来自按 `acquisition_order` 排列的主 session 轨迹和 highpass NCC 位移诊断。
#
# 这页只回答“数据里是否有足够可见的微扫描运动可供 2x baseline 使用”。stage command 仍然只是位移 prior；可见 NCC shift 是 alignment 诊断，也不是全局 ground truth。

# %%
trajectory_table = trajectory_capacity_table(displacement_outputs["summary_json"])
display(trajectory_table)

# %% [markdown]
# 此表格汇总了 EP01 `is_sr_usable=True` clean main input 的累计位移轨迹特征，包括坐标覆盖度、图像端估算的可见位移跨度（Cumulative Span）以及累计路径长度（Cumulative Path Length）。这些量化指标提取自位移评估摘要文件，用以诊断红外热像序列中蕴含的亚像素微扫描运动丰富度；实际帧数与跨度以上表的当前重建结果为准。可见累计跨度越大，意味着在后续配准与超分辨率重建中能够利用的相位多样性越充分。由于热像序列中受光学点扩散函数（PSF）卷积及环境热演化的影响，此处的位移均从局部高通滤波后的图像响应中估计得到，不得将电动台驱动命令（stage command）直接作为全局配准真值。

# %%
visible_shift_table = visible_shift_key_table(
    displacement_outputs["measurements"],
    displacement_outputs["summary_by_class"],
)
display(
    visible_shift_table.round(
        {
            "command_norm_px": 4,
            "visible_norm_median_px": 4,
            "visible_norm_p90_px": 4,
            "projection_median_px": 4,
            "peak_ncc_median": 4,
        }
    )
)

# %% [markdown]
# 表格展示了在不同位移步长及运动类型下，基于高通滤波归一化互相关（Highpass NCC）所提取的亚像素级位移诊断结果。表中对比了由电动台物理参数估算得出的名义命令幅值（`command_norm_px`）、热像图像中观测到的可见位移幅值（`visible_norm` 的中位数及 P90 分位数）以及可见位移在命令先验方向上的投影量（`projection_median_px`）。
# 在图像尺度上，行内 $2\,\mu\text{m}$ 与 $4\,\mu\text{m}$ 的微小步进用于检测同一扫描线内短时间跨度的局部形变与物理响应，而 $40\,\mu\text{m}$ 端点（endpoint）则用于表征宏观尺度上的二维相位张角。由于低通光学系统（PSF 衰减）和热辐射噪声（Noise Floor $\approx 0.0724^\circ\text{C}$）的共同限制，实测可见位移通常小于名义命令值。特别是在跨越较大时间跨度（acquisition gap $\approx 16$ 帧）的行间切换（row reset）或 Y 轴对齐帧对中，热演化效应会显著干扰互相关峰值（Peak NCC），导致其不适合直接用于高精度的亚像素位移标定。
# 实验表明，行内 $2\,\mu\text{m}$ 与 $4\,\mu\text{m}$ 步进在图像域内分别呈现约 $0.097\text{ px}$ 与 $0.192\text{ px}$ 的平移响应，与几何投影模型基本吻合，证明 2x 超分辨率重建所依赖的亚像素运动确实存在。然而，考虑到局部温度梯度起伏和传感器不均匀性，名义命令和图像位移均只能作为配准的先验约束与诊断手段，不能被直接视为刚性配准真值。

# %%
show_fig(DISPLACEMENT_DIR / "main_session_cumulative_trajectory.png")

# %% [markdown]
# Figure 1: Main session cumulative trajectory. Visible highpass-NCC displacement is integrated over acquisition order.

# %%
show_fig(DISPLACEMENT_DIR / "visible_shift_by_pair_class.png")

# %% [markdown]
# Figure 2: Visible shift by pair class. Estimated displacement magnitudes are compared across motion classes.

# %%
show_fig(DISPLACEMENT_DIR / "endpoint_displacement_vectors.png")

# %% [markdown]
# Figure 3: Endpoint displacement vectors. Endpoint pair shifts are plotted as detector-space vectors.

# %% [markdown]
# 三幅诊断图分别从不同维度验证了微扫描序列的运动一致性与可行性：第一幅展示了主会话按采集时间轴（`acquisition_order`）积分得到的二维可见累计轨迹；第二幅为不同运动类型帧对（行内小步、row reset、端点等）的位移幅值分布密度图；第三幅绘制了端点处的二维位移矢量场（Endpoint Vectors）。这些图表共同用于对位移容量（Displacement Capacity）进行合理性校验（Sanity Check）。
# 有效的微扫描过程应在累计轨迹图上呈现出均匀展开的二维相位覆盖，且不同步长帧对在位移幅值分布上应表现出明显的分层现象。若各类型的幅值分布严重重叠，则说明当前位移诊断方法缺乏区分亚像素运动的灵敏度。端点矢量图应与旋转角 $\theta = 47.6^\circ$ 的扫描物理几何大体吻合。对于行间切换（row reset）及跨行 Y 轴帧对，其易受长采集间隔带来的热漂移干扰，使得其实测位移矢量偏离刚性平移模型。
# 可视化结果表明，主会话内部具备清晰且有规律的亚像素级二维微扫描运动。这为构建 2x 超分辨率重建的相位对齐基线提供了实验依据。然而，此阶段的位移容量验证仅属于前置条件，并不能作为高倍率超分辨率（如 4x 重建）成立的直接证明，亦不能用单帧叠加的视觉对齐作为最终的分辨率度量。
