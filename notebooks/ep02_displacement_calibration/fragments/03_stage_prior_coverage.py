# %% [markdown]
# ## 2. Stage Prior Coverage in Detector Coordinates
#
# `configs/stage_calibration.json` defines a coordinate prior: stage X/Y commands are mapped to detector-space shifts by theta and the 10 um/pixel sampling pitch. This prior is useful for coverage, initialization, and regularization; it is not the alignment truth.
#
# 这里的“coverage”不是说画面里所有结构都已经对齐正确，而是说 stage command 在 detector 坐标中提供了哪些候选采样相位。
# 对 2x SR 来说，理想情况是样本不要都落在同一个整数像素相位附近；半像素 phase bin 都有样本，后续重建才有机会利用多帧互补信息。

# %%
fig, phase2_bins = plot_stage_prior_coverage(
    frame_audit,
    theta_deg=REFERENCE_THETA_DEG,
    pixel_size_um=PIXEL_SIZE_UM,
    output_path=OUTPUT_DIR / "ep02_stage_prior_coverage.png",
)
fig

# %% [markdown]
# > **图表说明**: 左图把主 session 坐标映射到 detector dx/dy prior 空间；右图统计这些 prior 落入 2x SR 的四个半像素 phase bin 的数量。
# > **怎么读图**: dx/dy 是 detector-space displacement prior，单位是 pixel；点的位置来自 stage command 经 theta/pitch 换算，不来自 NCC。右图四个柱子对应 2x 网格的四类半像素相位覆盖。
# > **正常/异常理解**: 正常的 prior 覆盖应形成二维云图，且四个 2x phase bin 都非空。若所有点挤在同一 phase bin，2x SR 的采样互补性会很弱；若云图方向明显错误，通常应先检查 theta 或坐标解析。
# > **数据分布**: stage prior 覆盖形成旋转后的二维云图，四个 2x phase bin 都有样本。phase 计数来自命令坐标映射，不来自图像对齐估计。
# > **核心发现**: 文件名坐标足以提供全局覆盖和 phase 覆盖先验；真实配准仍必须由轮廓、NCC 或后续 localization anchor 等 data-driven 证据确认或修正。

# %%
prior_stats = stage_prior_summary(
    frame_audit,
    theta_deg=REFERENCE_THETA_DEG,
    pixel_size_um=PIXEL_SIZE_UM,
)
display(prior_stats)
display(phase2_bins)

# %% [markdown]
# > **数据说明**: 第一张表给出 theta、pixel pitch、detector-space 覆盖跨度和非空 2x phase bin 数；第二张表是四个 phase bin 的帧数。
# > **怎么读表**: theta/pitch 是换算参数；dx/dy span 描述 prior 在 detector 坐标里的范围；non-empty phase bin 数说明 2x 半像素相位是否被覆盖。
# > **正常/异常理解**: 40 um command span 对应约 4 detector pixels 的 prior 量级，这是坐标换算的数量级检查。phase bin 数为 4 表示覆盖完整，但不说明每帧实际位移误差为零。
# > **数据分布**: 二维旋转后 dx/dy 都有覆盖；2x phase bins 全部非空，且计数可用于判断采样相位是否严重偏置。
# > **核心发现**: EP02 可以证明坐标 prior 的覆盖价值，但不能把该 prior 当作每帧实际对齐位移。这个区分是进入 contour-level SR 的前置条件。
