# %% [markdown]
# ## 7. Conclusions and EP07 Handoff
#
# EP06 的结论必须按证据强度书写：
#
# 1. **可以声明**: 在 248 clean SR-usable frames 上完成了 2x contour-level POC 的 classic SR 对比，包含 highpass 结构主轨和 raw-temperature 控制轨；原始 255 帧只作为历史主采集段说明。
# 2. **可以讨论**: SAA-weighted 是否优于 SAA-uniform，IBP/MAP-TV 是否在直接视觉、split-half、artifact audit、alignment ablation 和 data-driven alignment sweep 上带来稳健增益。
# 3. **必须保守**: alignment ablation/sweep 若缺失或显示策略敏感，只能写成“当前对齐策略下的候选结果”，不能写成对齐无关的稳健 SR 结论。
# 4. **不能声明**: 4x SR、5 um 实际空间分辨率、绝对温度计量 SR，或仅凭 sharper gradient 判定 SR 成功。
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

if ablation_figures or ablation_tables:
    print("Alignment ablation artifacts are present and should be cited in the final EP06 conclusion.")
else:
    print("Alignment ablation artifacts are not present in this checkout/output set.")
    print("Use the ablation section command before making alignment-stability claims.")

# %% [markdown]
# 此处的最终结论表提取自高通成像通道（Highpass Track）的量化评估结果，汇总了平均梯度、伪影得分、Chamfer 距离以及相对插值的相关性系数。该表用于协助在 EP06 实验结尾快速总结不同超分辨率算法分支的优缺点。
# 在参数判定中，各项指标应进行交叉验证：高梯度响应（平均梯度）必须与低伪影得分（`artifact_score`）及低 Chamfer 距离（`contour_chamfer_lr_px`）相匹配，以排除由高频噪声或数值振铃带来的假阳性锐化增益。若算法分支在某项锐度指标上突出，但在子集一致性或配准烧蚀敏感性（Alignment Ablation Sensitivity）测试中表现较差，则在物理结论中应界定为高伪影过拟合风险分支。
# 综上所述，超分辨率 POC 的论证需以全景与局部多算法对比图、中心区域温度分布图以及高通滤波/原始控制轨的多维视觉证据为主，定量物理指标作为辅助诊断。本阶段所确立的 2x 采样网格旨在改善工业检测下的芯片内部轮廓清晰度与结构边界稳定性，不能被直接等同于 5 µm 的光学计量级空间分辨率声明。后续的 EP07 仿真实验将在此 2x 基准上引入更严格的前向退化退卷积及局部边缘过渡函数（ESF）分析以进一步厘清分辨率增益的物理边界。
