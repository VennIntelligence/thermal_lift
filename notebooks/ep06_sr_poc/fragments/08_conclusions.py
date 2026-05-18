# %% [markdown]
# ## 7. Conclusions and EP07 Handoff
#
# EP06 的结论必须按证据强度书写：
#
# 1. **可以声明**: 在 255 帧主 session 上完成了 2x contour-level POC 的 classic SR 对比，包含 highpass 结构主轨和 raw-temperature 控制轨。
# 2. **可以讨论**: SAA-weighted 是否优于 SAA-uniform，IBP/MAP-TV 是否在直接视觉、split-half 和 artifact audit 上带来增益。
# 3. **不能声明**: 4x SR、5 um 实际空间分辨率、绝对温度计量 SR，或仅凭 sharper gradient 判定 SR 成功。
#
# EP07 建议把本 EP 的最佳候选方法固定下来，做更严格的 MTF/edge transfer、跨 ROI 稳定性和客户样例展示。

# %%
if not evaluation_summary.empty:
    highpass_summary = evaluation_summary[evaluation_summary["track"].eq("highpass")].copy()
    display(
        highpass_summary[
            ["label", "mean_gradient", "artifact_score", "contour_chamfer_lr_px", "corr_to_bicubic"]
        ].round(4)
    )
else:
    print("Run evaluation before filling the conclusion table.")

# %% [markdown]
# > **数据说明**: 这张结论表只保留 highpass 主轨的核心辅助指标，方便在 EP06 结尾快速比较候选方法。它不是新的实验输出，而是从 `evaluation_summary.csv` 中抽取与方法选择最相关的列。
# >
# > **怎么看**: `mean_gradient` 较高说明整体边缘响应更强，但仍要防止噪声和振铃；`artifact_score` 越低通常越稳；`contour_chamfer_lr_px` 越低通常表示与 EP04 segment proxy 更接近；`corr_to_bicubic` 用来判断结果是否过于接近插值 baseline 或偏离过大。
# >
# > **正常/异常**: 最终排序不能只按一个数值决定。若某方法梯度最高但 artifact score 或 split-half 明显更差，应降级为“锐但风险高”；若某方法指标温和但 fullview、ROI 和 center raw-temperature 都更可读，则可以作为更保守的候选。
# >
# > **核心发现**: 最终报告应以 `comparison_fullview.png`、三个 ROI、`comparison_center_raw_temperature.png`、`comparison_control_track.png`、split-half 和 artifact audit 为主证据；如果指标与视觉证据冲突，优先保守解释并写明风险。
