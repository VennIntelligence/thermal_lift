# %% [markdown]
# ## Step 7 — Anchor Gate 深挖诊断
#
# 本节补回 EP04 经典诊断图，并新增只依赖已缓存 CSV 的轻量 quality-gate 审计。所有内容仍只服务 EP06 alignment anchor / quality gate：不输出 SR，不做 LR/bicubic/SR 对照，也不把 localization 当作客户交付或 SR 成败判定。

# %%
fig = plot_split_half_distribution(outer_segment_summary)
save_fig(fig, "split_half_distribution.png")

# %% [markdown]
# > **图表说明**: 这是经典 EP04 外轮廓 A-class segment split-half 分布图，横轴是每个 segment 在多条 scanline 上的 split-half 中位数，纵轴是 segment 数。
# > **怎么看**: split-half 越小，表示奇偶帧子集给出的局部边缘位置越一致；虚线标出中位数、P90 和 CRB 参考量级。这里的 “精度” 是局部 anchor repeatability，不是完整芯片形状重建精度。
# > **异常是否正常**: 长尾是正常的，因为有些外轮廓段虽然热对比高，但局部相位覆盖、曲率或 ESF 模型不稳定。超过阈值的段不能当强 anchor，但不等于局部结构不存在。
# > **核心发现**: 外轮廓中存在一批 repeatable anchor，可支撑 EP06 对齐；这张图只说明 anchor 可用性，不说明 SR 已经成功。

# %%
fig = plot_crb_ratio_scatter(outer_segment_summary)
save_fig(fig, "crb_ratio_scatter.png")

# %% [markdown]
# > **图表说明**: 每个点是一个外轮廓 segment，纵轴是 split-half / CRB ratio，颜色区分 segment-level gate pass/fail。
# > **怎么看**: ratio 接近 1 表示实测 repeatability 接近噪声理论下限；大于 3 或进入长尾表示定位稳定性明显弱于理想噪声模型。CRB 是下限参考，不是光学显微真值。
# > **异常是否正常**: fail 点仍可能有高 SNR 或高 NCC peak；CRB ratio 高通常提示 split-half、phase coverage 或 ESF 模型问题。不能用单个高 ratio 点否定全局 alignment 或内部 SR 目标。
# > **核心发现**: EP04 gate 能把接近 CRB 的稳定外轮廓段与长尾段分开，适合给 EP06 做 anchor quality gate。

# %%
fig = plot_phase_coverage_vs_precision(outer_segment_summary)
save_fig(fig, "phase_coverage_vs_precision.png")

# %% [markdown]
# > **图表说明**: 横轴是 data-driven highpass NCC 位移投影到 segment 法向后的相位覆盖，纵轴是 split-half difference，点大小随 SNR 变化，颜色区分 pass/fail。
# > **怎么看**: 横轴越大通常说明多帧在该局部边缘法向上提供了更多微扫描相位；纵轴越小表示定位更稳定。相位覆盖不足时，即使 SNR 不低，joint ESF 也可能缺少足够的几何约束。
# > **异常是否正常**: 大点但仍 fail 是正常现象：SNR 高只说明热对比足够，不保证 NCC 轨迹、ESF 宽度和 split-half 都稳定。stage command 这里只能作为 prior，不是相位覆盖真值。
# > **核心发现**: anchor gate 需要同时看相位覆盖和 repeatability；不能只用 SNR 或 stage command 选配准锚点。

# %%
fig = plot_failure_taxonomy(outer_segment_summary)
save_fig(fig, "failure_taxonomy.png")

# %% [markdown]
# > **图表说明**: 经典 failure taxonomy 图按 segment-level primary failure reason 统计外轮廓失败段数量。
# > **怎么看**: 这张图只显示每个失败 segment 的主要原因，适合快速看外轮廓 gate 卡在哪里；后面的 co-occurrence 表会展示 row-level 多标签原因。
# > **异常是否正常**: primary reason 是摘要字段，不代表其他 gate 没有同时失败。比如 `sigma_out_of_range` 和 `split_half_high` 可能同一行同时出现，所以不能把不同原因的比例强行相加为 100%。
# > **核心发现**: 外轮廓失败不是单一机制；EP06 应使用 gate 后的 anchor，并把失败段保留为弱约束或定性背景，而不是当作真值。

# %%
fig = plot_cross_scanline_consistency(outer_results, outer_segment_summary)
save_fig(fig, "cross_scanline_consistency.png")

# %% [markdown]
# > **图表说明**: 经典 cross-scanline consistency 图展示代表性外轮廓 segment 的 joint edge position 随 scanline Y 的变化；每条曲线减去了自己的中位位置。
# > **怎么看**: 曲线越平，说明同一个 segment 在不同 X scanline 上的局部定位越一致；局部跳变提示某条 scanline 或某段局部热场不稳定。
# > **异常是否正常**: 少量 scanline 偏离是正常的，因为 raster 采集时间、局部纹理和热场演化会影响 data-driven alignment。不能把单条线的偏离外推成 stage command 错或 SR 不可行。
# > **核心发现**: cross-scanline 诊断用于找稳定 anchor 和 held-out 检查线，不是客户最终形状交付。

# %%
fig = plot_segment_scanline_pass_heatmap(outer_results, inner_results)
save_fig(fig, "segment_scanline_pass_heatmap.png")

# %% [markdown]
# > **图表说明**: heatmap 的每个像素是一个 `segment x scanline` row-level gate，蓝色表示通过，灰色表示 reject；segment 按通过率从低到高排序。
# > **怎么看**: 横向连续灰带表示某个 segment 在多条 scanline 上都不稳定，属于“坏段”；纵向灰带表示某条 scanline 对很多 segment 都不稳定，属于“坏线”。内轮廓面板更高，表示内部候选段更多。
# > **异常是否正常**: 内轮廓灰色更多是正常的，因为内部热边缘可能更宽、更弯、更弱或更受局部热场影响。灰色表示不能当 alignment truth，不表示该区域应从 SR 目标中删除。
# > **核心发现**: EP04 的失败不是均匀随机噪声；它有局部坏段和局部坏线结构，EP06 应据此选择 alignment input 与 holdout scanline。

# %%
scanline_segment_layout = scanline_segment_failure_summary_table(outer_results, inner_results)
layout_display = scanline_segment_layout.copy()
for col in ["overall_row_pass_rate", "weakest_scanline_pass_rate", "weakest_segment_pass_rate"]:
    layout_display[col] = (100.0 * layout_display[col]).round(1)
display(layout_display)

# %% [markdown]
# > **数据说明**: 表格把 heatmap 压缩成每类 contour 的总体 row pass rate、最弱 scanline、最弱 segment，以及完全 0-pass 的 scanline/segment 数。
# > **怎么看**: `weakest_scanline_pass_rate` 很低说明有局部坏线；`zero_pass_segments` 多说明许多 segment 在所有 scanline 上都不能当 anchor。百分比是通过率，越高越适合作为 alignment 证据。
# > **异常是否正常**: inner 的 `zero_pass_segments` 多不等于内部结构不存在，而是这些内部段在当前 localization-only gate 下不能当真值。outer/inner 的弱线也不能被解释为 stage command 真值偏差。
# > **核心发现**: EP06 应把坏段排除出强 anchor，把弱线优先放入 holdout 或低权重诊断，而不是用它们监督 SR。

# %%
cooccurrence = failure_cooccurrence_table(outer_results, inner_results, top_n=8)
cooccurrence_display = cooccurrence.copy()
for col in ["share_of_failed_rows", "top_co_share_of_reason"]:
    cooccurrence_display[col] = (100.0 * cooccurrence_display[col]).round(1)
display(cooccurrence_display)

# %% [markdown]
# > **数据说明**: 表格按 row-level 多标签 `fail_reason` 统计失败原因，并列出每个原因最常一起出现的另一个原因。
# > **怎么看**: `share_of_failed_rows` 是“失败行中触发该原因的比例”，不是互斥分类；同一失败行可以同时触发 `sigma_out_of_range`、`split_half_high`、`low_phase_coverage` 等多个 gate。
# > **异常是否正常**: 各原因百分比相加超过 100% 是正常且预期的，因为这是多标签 gate。`top_co_reason` 高说明失败机制耦合，例如相位覆盖不足可能同时带来 ESF 拟合或 split-half 不稳。
# > **核心发现**: EP04 gate 失败需要按多原因解释，不能用单一 primary reason 得出过度简化结论。

# %%
ncc_esf_diag = ncc_esf_failure_diagnostic_table(outer_results, inner_results)
diag_display = ncc_esf_diag.copy()
share_cols = [
    "share_failed_ncc_peak_above_gate",
    "ncc_unreliable_share",
    "fit_error_share",
    "sigma_out_of_range_share",
    "split_half_high_share",
    "low_phase_coverage_share",
    "esf_or_stability_share",
]
for col in share_cols:
    diag_display[col] = (100.0 * diag_display[col]).round(1)
metric_cols = [
    "median_failed_ncc_peak",
    "p10_failed_ncc_peak",
    "median_failed_ncc_fit_ok_fraction",
    "median_failed_phase_coverage_px",
    "median_failed_sigma_px",
    "median_failed_split_half_px",
]
for col in metric_cols:
    diag_display[col] = diag_display[col].round(4)
display(diag_display)

# %% [markdown]
# > **数据说明**: 表格把失败行的 NCC 质量和 ESF/稳定性失败拆开看。`share_failed_ncc_peak_above_gate` 表示失败行里 NCC peak 仍高于 0.85 gate 的比例。
# > **怎么看**: 如果 NCC peak 中位数高、且大多数失败行仍高于 NCC gate，但 `esf_or_stability_share` 很高，就说明瓶颈主要不是“相关峰太低”，而是 sigma/fit/split-half/phase 等 localization 模型与稳定性条件。
# > **异常是否正常**: highpass NCC 只衡量局部红外纹理相关，不是位移真值。NCC 很高时仍可能因为 ESF 表观宽度贴边、split-half 长尾或相位覆盖不足而 reject。
# > **核心发现**: 当前 inner 的主要瓶颈应解释为 ESF/model/stability gate，而不是简单的 NCC 崩溃；这支持把 inner fail 段保留为 EP06 SR 目标但不当 alignment truth。

# %%
role_margin = ep06_role_margin_table(ep06_recommendations)
role_margin_display = role_margin.copy()
role_margin_display["median_pass_rate_margin"] = (100.0 * role_margin_display["median_pass_rate_margin"]).round(1)
for col in ["median_split_margin_px", "median_phase_margin_px"]:
    role_margin_display[col] = role_margin_display[col].round(4)
role_margin_display["median_crb_ratio_margin"] = role_margin_display["median_crb_ratio_margin"].round(2)
role_margin_display["p10_alignment_margin_min"] = role_margin_display["p10_alignment_margin_min"].round(3)
display(role_margin_display)

# %% [markdown]
# > **数据说明**: 表格审计 EP06 三类 role 距 alignment-input 数值阈值的 margin：pass rate 需高于 70%，split-half 需低于 0.06 px，CRB ratio 需低于 5x，phase coverage 需高于 0.15 px。
# > **怎么看**: 正 margin 表示离 alignment 阈值有余量；负 margin 表示达不到该项 alignment 输入条件。`p10_alignment_margin_min` 看每组中更靠近或低于阈值的尾部，`closest_alignment_gate` 指最常成为瓶颈的 gate。
# > **异常是否正常**: `sr_target_not_truth` 出现负 margin 是正常的，它的含义是“不能当真值或强 anchor”，不是“放弃区域”。holdout 可以接近阈值，因为它本来用于检查泛化，不直接训练/拟合 alignment。
# > **核心发现**: EP06 可以区分强 anchor、held-out QC 和 SR target-not-truth；这个分层保护 alignment，同时保留客户关心的内部结构目标。

# %%
fig = plot_normal_angle_coverage_comparison(outer_segment_summary, inner_segment_summary)
save_fig(fig, "normal_angle_coverage.png")

# %% [markdown]
# > **图表说明**: 极坐标图展示通过 gate 的 outer/inner segment 法向角覆盖；半径是相对计数，不是物理长度。
# > **怎么看**: 角度覆盖越分散，alignment anchor 对不同方向的位移误差越敏感；若角度集中在少数方向，对某些方向的对齐误差约束会弱。
# > **异常是否正常**: 内轮廓通过段少时，角度覆盖看起来稀疏是正常的。这个图说明 anchor 几何覆盖，不说明真实内部结构是否完整，也不替代 stage-to-pixel 标定。
# > **核心发现**: EP06 alignment 应优先组合不同 normal angle 的 anchor，并用 holdout 检查泛化；stage command 仍只作为 prior/初始化/正则。
