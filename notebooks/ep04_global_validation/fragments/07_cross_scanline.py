# %% [markdown]
# ## Step 6 — EP06 Alignment Gate 建议
#
# 将 EP04 段级证据转化为三类 EP06 输入：alignment 输入段、held-out 验证段、不可直接当真值但仍可作为 SR 目标/诊断的段。

# %%
ep06_recommendations = build_ep06_gate_recommendations(outer_segment_summary, inner_segment_summary)
save_ep06_gate_outputs(ep06_recommendations, OUTPUT_DIR)
ep06_summary = ep06_gate_recommendation_summary(ep06_recommendations)
display(
    ep06_summary.assign(
        median_split_half_px=lambda df: df["median_split_half_px"].round(4),
        median_crb_ratio=lambda df: df["median_crb_ratio"].round(2),
        median_pass_rate=lambda df: (100.0 * df["median_pass_rate"]).round(1),
    )
)

# %% [markdown]
# > **数据说明**: 表格按 outer/inner 和 EP06 role 聚合段数、split-half、CRB ratio 和 pass rate；CSV 同步写入 `output/ep04_global_validation/ep06_gate_recommendations.csv`。
# > **读法**: `alignment_input` 表示可参与 EP06 对齐估计或约束的稳定 anchor；`holdout_validation` 表示不参与拟合、用于检查对齐是否泛化的段；`sr_target_not_truth` 表示不能当定位真值，但仍保留为 SR 目标或诊断区域。读表时要同时看段数和中位质量指标。
# > **正常/异常理解**: 外轮廓更多进入 `alignment_input` 是正常的，因为它通常更稳定；内轮廓更多进入 `sr_target_not_truth` 也是正常的，因为内部结构正是需要 SR 改善的目标。异常做法是把 `sr_target_not_truth` 从后续视觉评估中删掉，或反过来把它当成精确真值监督。
# > **对本 Episode 的意义**: EP06 应使用 quality-gated anchors 估计/约束对齐，用 held-out 段验证对齐稳定性，并明确失败段不可作为真值但仍可用于 SR 目标区域诊断。

# %%
fig = plot_ep06_gate_recommendations(ep06_recommendations)
save_fig(fig, "ep06_gate_recommendations.png")

# %% [markdown]
# > **图表说明**: 堆叠柱图展示外轮廓和内轮廓被分配到 EP06 三类角色的数量。
# > **读法**: 每根柱子的颜色组成表示该 contour 类型被分配到不同 EP06 角色的结构。柱子高说明该类段数量多，某一颜色占比高说明该角色在该 contour 中占主导。
# > **正常/异常理解**: 外轮廓更适合作为 alignment input；内轮廓中较多段落被标为 `sr_target_not_truth`，表示这些内部结构不能直接做定位真值。若 holdout 太少，EP06 的泛化检查会变弱；若 alignment_input 太少，对齐估计会更依赖 prior，风险更高。
# > **对本 Episode 的意义**: 质量门控的作用是保护 EP06 对齐和验证，不是排除内部结构 SR；内部失败段应进入后续形状重建评估。

# %%
top_alignment_inputs = (
    ep06_recommendations[ep06_recommendations["ep06_role"].eq("alignment_input")]
    .sort_values(["contour", "split_half_median_px", "crb_ratio_median"])
    .head(12)
)
display(
    top_alignment_inputs[
        [
            "contour",
            "segment_id",
            "quality_label",
            "pass_rate",
            "split_half_median_px",
            "crb_ratio_median",
            "phase_coverage_median_px",
            "ep06_reason",
        ]
    ].assign(
        pass_rate=lambda df: (100.0 * df["pass_rate"]).round(1),
        split_half_median_px=lambda df: df["split_half_median_px"].round(4),
        crb_ratio_median=lambda df: df["crb_ratio_median"].round(2),
        phase_coverage_median_px=lambda df: df["phase_coverage_median_px"].round(3),
    )
)

# %% [markdown]
# > **数据说明**: 表格列出最优先的 EP06 alignment input 候选，按轮廓类型和定位稳定性排序。
# > **读法**: 这是给 EP06 的起步清单。优先选择 `pass_rate` 高、`split_half_median_px` 低、`crb_ratio_median` 合理、`phase_coverage_median_px` 足够的段；`ep06_reason` 解释为什么该段被分配为 alignment input。
# > **正常/异常理解**: 正常候选不一定全部来自内轮廓，也不应该为了覆盖内部结构而强行使用不稳定内轮廓段。若某段 stage prior 看起来合理但这些数据驱动质量指标不通过，它仍不应被当成强 anchor；stage command 只能作为初始化、先验或正则约束。
# > **对本 Episode 的意义**: EP06 初版对齐应从这些锚点开始，再用 holdout 段检查泛化；stage command 仍只可作为 prior，不可替代这些数据驱动质量证据。

# %%
create_ep04_anchor_gate_figures(
    reference_frame,
    outer_results,
    outer_segment_summary,
    inner_results,
    inner_segment_summary,
    ep06_recommendations,
    OUTPUT_DIR,
)
print("Saved EP04 anchor/gate figure set and EP06 recommendation CSVs.")

# %% [markdown]
# > **数据说明**: 最后一行统一刷新 EP04 anchor/gate 图集，确保脚本运行和 notebook 执行得到同一组输出文件。
# > **读法**: 这行输出表示图集和 EP06 推荐 CSV 已按当前片段逻辑写入 EP04 输出目录。它不是新增结论，而是把前面各节的结果整理成 EP06 可消费的文件集合。
# > **正常/异常理解**: 正常情况下，输出图集应与 Notebook 中展示的核心图一致；如果后续单独运行脚本得到不同结果，应先核对缓存、输入 CSV、noise floor、θ、代码版本和 `FORCE_RERUN` 设置。
# > **对本 Episode 的意义**: EP04 已形成 EP06 可直接消费的 anchor、gate、holdout 与不可当真值区域清单。这个清单限定了“哪些证据可用于对齐”，不扩大为“红外结果已经匹配光学显微真值”。
