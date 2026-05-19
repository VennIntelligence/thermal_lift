# %% [markdown]
# ## 5. Quantitative Evaluation
#
# 指标用于辅助解释直接视觉对比。Gradient magnitude 只能描述锐度，不能单独作为 SR 成功证据；split-half consistency 和 artifact audit 用于约束过拟合与伪影。

# %%
evaluation_summary = read_csv_if_exists("evaluation_summary.csv")
if not evaluation_summary.empty:
    display(
        evaluation_summary[
            [
                "track",
                "label",
                "finite",
                "std",
                "std_ratio_to_lr",
                "mean_gradient",
                "p95_gradient",
                "artifact_score",
                "nrmse_to_bicubic",
                "corr_to_bicubic",
                "contour_chamfer_lr_px",
            ]
        ].round(4)
    )

# %%
if not evaluation_summary.empty:
    method_eval = evaluation_summary[
        evaluation_summary["method"].isin(["saa_uniform", "saa_weighted", "ibp", "map_tv"])
    ].copy()
    method_eval["track_view"] = method_eval["track"].map(
        {
            "highpass": "highpass input",
            "raw_control_highpass_visual": "raw control visual",
        }
    )
    method_eval = method_eval[
        [
            "track_view",
            "method",
            "label",
            "std",
            "std_ratio_to_lr",
            "mean_gradient",
            "p95_gradient",
            "artifact_score",
            "contour_chamfer_lr_px",
            "corr_to_bicubic",
        ]
    ]
    display(method_eval.round(4))

lambda_selection = read_csv_if_exists("map_tv_lambda_selection.csv")
if not lambda_selection.empty:
    display(
        lambda_selection[
            [
                "track",
                "lambda_tv",
                "split_half_nrmse",
                "artifact_score",
                "mean_gradient",
                "std_excess_vs_saa",
                "selection_proxy",
                "selected",
            ]
        ].round(4)
    )

# %% [markdown]
# > **数据说明**: 这张表把每个 track 和方法的辅助指标汇总在一起。`std`/`std_ratio_to_lr` 描述输出动态范围是否膨胀，`mean_gradient` 和 `p95_gradient` 描述边缘/纹理响应强度，`artifact_score` 描述一阶伪影风险，`nrmse_to_bicubic` 和 `corr_to_bicubic` 描述相对 bicubic baseline 的接近程度，`contour_chamfer_lr_px` 是基于 EP04 segment points 的轮廓距离 proxy。
# >
# > **怎么看**: `p95_gradient` 越大通常表示较强边缘更多，但它不是“越大就绝对越好”；噪声、振铃和假边缘也会推高梯度。`artifact_score` 通常越小越好，因为它希望惩罚不自然的一阶结构；`contour_chamfer_lr_px` 通常越小越好，但它只是相对于 EP04 点集的 proxy，不是独立光学 ground truth。
# >
# > **正常/异常**: `finite=True` 是最基本的数值健康检查；如果某个方法梯度很高但 artifact score 或 std ratio 同时变高，或 Chamfer 没有改善，就要把“更锐”解释为可疑而不是自动解释为更清楚。`nrmse_to_bicubic` 和 `corr_to_bicubic` 也不能单独排名，因为太接近 bicubic 可能意味着保守，太远又可能意味着引入了伪影。第二张方法摘要中 raw-control 没有 `SAA uniform raw` 行时是脚本产物设计差异，不是缺数据错误；MAP-TV lambda 表中 `selected=True` 表示 split-half + artifact/std proxy 当前选中的正则强度。
# >
# > **核心发现**: 当前评估应读成方法画像，而不是单项排名：SAA 是多帧相位覆盖 baseline，IBP 检查 forward model 迭代是否带来额外结构，MAP-TV 用 split-half proxy 选择正则强度。EP06 需要同时满足轮廓更清楚、split-half 稳定、artifact 不恶化、Chamfer proxy 不矛盾，才适合把某个方法作为 EP07 候选。

# %%
display(show_png("gradient_magnitude_comparison.png"))

# %% [markdown]
# > **图表说明**: `gradient_magnitude_comparison.png` 把不同方法的梯度幅值可视化出来，显示哪里出现了强边缘或强局部变化。它回答的是“哪里更锐、边缘响应更强”，不是“真实分辨率提高了多少”。
# >
# > **怎么看**: 亮的区域代表梯度大，通常对应芯片边界、针脚边、内部结构边缘，也可能对应噪声和振铃。可以把它和 fullview/ROI highpass 图对照：如果亮边沿着稳定结构分布，可信度更高；如果亮点散乱或呈规则纹理，可信度更低。
# >
# > **正常/异常**: P95 gradient 这类高分位梯度指标容易被少量强边缘或伪影影响，因此不能简单认为数值越大越好。异常表现包括整幅图普遍变亮、细碎噪点增多、边缘两侧出现过宽的亮带。
# >
# > **核心发现**: 梯度图只能作为 contour sharpness 的辅助证据。它支持“哪里看起来更锐”的描述，但必须和 raw-temperature 中心图、split-half、artifact audit 一起使用。

# %%
display(show_png("split_half_consistency.png"))

# %% [markdown]
# > **图表说明**: `split_half_consistency.png` 把同一主 session 拆成两个子集后分别重建，再比较两半结果的一致性。它检查的是方法是否依赖偶然帧或噪声，而不是检查最终图是否最锐。
# >
# > **怎么看**: Split-half NRMSE 通常越小越好，表示两半数据得到的结构更一致；如果图中按 lambda 或方法展示曲线，最低点附近通常是稳定性较好的候选。MAP-TV 用这个 proxy 选择 lambda，是为了避免正则过弱导致噪声/伪影，也避免正则过强把结构抹平。
# >
# > **正常/异常**: NRMSE 很低不自动代表结构最真实，因为过度平滑也可能让两半看起来一致；NRMSE 很高则提示方法对帧选择敏感，可能在追逐噪声、热漂移或对齐误差。这个指标没有外部显微配准真值，因此只能当稳定性约束。
# >
# > **核心发现**: Split-half 的价值是给视觉结论加一个复现性门槛。一个方法即使 highpass 看起来更锐，只要 split-half 明显变差，就不应被直接升级为可靠 SR 增益。

# %%
display(show_png("artifact_audit.png"))

# %% [markdown]
# > **图表说明**: `artifact_audit.png` 用一阶统计或局部差分类指标检查重建图是否引入不自然的条纹、振铃、棋盘纹或局部尖峰。它关注的是“看起来变清楚”背后的代价。
# >
# > **怎么看**: Artifact score 通常越小越好；如果某个算法在视觉图中更锐，但 artifact audit 也显著升高，说明锐化可能夹带了伪影。要特别留意算法之间是否只是把边缘加强，还是同时把背景噪声也结构化了。
# >
# > **正常/异常**: 轻微升高不一定致命，因为真实边缘增强也会改变一阶统计；但大幅升高、局部异常集中或与 highpass 图中的条纹相互对应时，应把该方法标为高风险。Artifact audit 同样不是光学真值，它只是伪影 proxy。
# >
# > **核心发现**: EP06 的定量结论应采用保守合取逻辑：P95 gradient 可以高一些，但 split-half 不能明显变差，artifact score 不能明显恶化，Chamfer proxy 也不能和视觉证据强烈冲突。
