# %% [markdown]
# ## 4. Data-Driven Alignment vs Filename/Stage Prior
#
# EP02 的坐标 prior 需要和 data-driven alignment 分工：prior 提供覆盖、初始化和约束；alignment evidence / anchor / quality gate 由图像数据中的 contour/NCC 一致性支撑。若 EP05 alignment score 已存在，本节直接读取；否则 core helper 会退回到轻量 EP02 NCC proxy。
#
# 这一节回答“后续 SR 到底信谁”。stage prior 告诉我们应该从哪里开始找；data-driven alignment 则用热像帧中的边缘、梯度、轮廓或 NCC 证据检查这个位移是否真的让结构对齐。
# 因此，prior 和 alignment 不是互相替代的两套结论，而是前后衔接的两层证据。

# %%
fig, alignment_summary, alignment_source = plot_alignment_comparison(
    PROJECT_ROOT,
    OUTPUT_DIR / "ep02_data_driven_alignment_comparison.png",
)
fig

# %% [markdown]
# > **图表说明**: 左图比较不同对齐策略的 holdout contour Chamfer 中位误差，右图比较 gradient correlation 中位数。`Stage prior only` 表示只用命令/配置 prior；`Data-driven` 表示由图像 NCC/轮廓证据估计位移。
# > **怎么读图**: Chamfer 误差越低，表示 holdout 轮廓在对齐后越接近；gradient correlation 越高，表示局部边缘/梯度方向越一致。一个可靠对齐方法应同时降低轮廓误差、提高梯度一致性。
# > **正常/异常理解**: 如果 data-driven 方法比 stage prior only 更好，说明图像证据确实修正了命令 prior。若某个 data-driven 方法只改善一个指标、恶化另一个指标，应谨慎看作局部过拟合或质量门控不足，而不是直接采纳。
# > **数据分布**: 当前输出读取自 EP05 alignment score 时，data-driven contour refined 的 Chamfer 误差最低，且相对 stage prior only 有更高的梯度一致性。
# > **核心发现**: filename/stage prior 更适合作初始化和覆盖模型；每帧 alignment evidence、anchor 和 quality gate 应由 data-driven contour/NCC 质量指标支撑。

# %%
alignment_gain = alignment_improvement_summary(alignment_summary)
display(
    alignment_summary[
        [
            "display_label",
            "holdout_chamfer_median_px",
            "holdout_chamfer_p90_px",
            "gradient_corr_median",
            "gradient_corr_p10",
            "shift_norm_median_px",
        ]
    ]
)
display(alignment_gain if not alignment_gain.empty else pd.DataFrame({"note": [f"Alignment source: {alignment_source}"]}))

# %% [markdown]
# > **数据说明**: 第一张表是对齐策略的量化 score；第二张表把 data-driven alignment 相对 stage prior 或 no-alignment 的改善量转换为百分比/相关系数增益。
# > **怎么读表**: `holdout_chamfer_median_px` 和 `p90` 越低越好；`gradient_corr_median` 越高越好；`gradient_corr_p10` 观察较差样本的下限；`shift_norm_median_px` 表示该策略估计出的典型位移幅值。
# > **正常/异常理解**: 如果策略需要很大的 shift 才得到轻微指标改善，可能是在追逐热场变化或噪声；如果 holdout 和 gradient 指标同时改善，才更像是有效 alignment。改善量表只能比较本次输入中的策略，不应外推成通用算法排名。
# > **数据分布**: data-driven 方法的 holdout contour 误差低于只用 stage prior 的方法；这说明直接从图像数据估计位移更适合作为对齐质量门控。
# > **核心发现**: EP02 应输出“stage prior + data-driven verification”的分工，而不是把命令位移写成真值。EP06 进入 2x contour-level SR 前应沿用这种分工。
