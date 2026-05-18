# %% [markdown]
# ## Step 4 — 内轮廓通过率与失败原因
#
# 内轮廓对应芯片内部结构/形状，是客户关心的 SR 目标区域。本节解释为什么它们在 localization-only gate 下通过率较低，并明确这些区域不是被放弃的区域。

# %%
inner_reason_table = failure_reason_table(inner_results, contour="inner")
display(
    inner_reason_table.assign(
        share_of_failed_rows=lambda df: (100.0 * df["share_of_failed_rows"]).round(1)
    )
)

# %% [markdown]
# > **数据说明**: 表格拆分 inner row-level `fail_reason`；同一行可能触发多个原因，因此各原因占比可超过 100%。
# > **读法**: 每一行是一个失败类别的计数和占失败行比例。因为一次 row-level 评估可能同时触发多个 gate，例如 NCC 不稳且 split-half 偏高，所以百分比相加可能超过 100%。
# > **正常/异常理解**: `ncc_unreliable` 表示局部红外纹理相关性不够可信；`low_phase_coverage` 表示多帧在该段法向上的采样相位跨度不足；`sigma_out_of_range` 表示拟合出的表观 ESF 宽度超出当前模型接受范围；`split_half_high` 表示奇偶帧子集给出的位置不一致；`fit_error` 表示模型拟合失败。这些都是 localization gate 的失败原因，不是光学轮廓真值判断。
# > **对本 Episode 的意义**: 内轮廓低通过率说明“只用边缘定位点交付内部结构”不够；EP06 应把这些失败区域作为形状 SR 改善目标，而不是从评估中删除。

# %%
fig = plot_inner_failure_reasons(inner_results, inner_segment_summary)
save_fig(fig, "inner_failure_reasons.png")

# %% [markdown]
# > **图表说明**: 左图显示 inner segment 按质量标签的 anchor pass rate，右图显示 row-level gate 失败原因占比。
# > **读法**: 左图回答“哪些质量标签更容易成为 anchor”，右图回答“失败主要卡在哪个 gate”。先读 pass rate 的高低，再看失败原因是否集中在少数机制上。
# > **正常/异常理解**: A/B 标签通过率高于 C/D 是合理的，但 A/B 失败也合理，因为内部结构的局部边缘可能不是一个干净的单阶跃 ESF。若失败原因主要是 `split_half_high`，说明定位可重复性不足；若主要是 `sigma_out_of_range`，说明当前表观边缘模型不匹配；若主要是 NCC 相关问题，说明局部帧间相位估计不可靠。
# > **对本 Episode 的意义**: EP04 对内轮廓的价值是标出“哪些内部段可作为局部 anchor，哪些只能作为 SR 目标/诊断区域”；这为 EP06 的 shape reconstruction 指标分层提供输入。
