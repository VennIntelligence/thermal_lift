# %% [markdown]
# ## Intermediate Diagnostics
#
# 三种算法的“中间性结果”不完全一样：Drizzle 没有训练迭代，但有 pixfrac/coverage 过程诊断；MAP-TV 和 TGV 是迭代优化算法，后续重跑会保存 convergence records。对于当前已有长跑产物，本节先展示已经落盘的参数轨迹、split-half 分布、holdout 分布和 coverage 曲线。

# %%
show_optional_fig(
    "intermediate_parameter_diagnostics.png",
    "Pending: run `uv run python scripts/build_ep10_cache.py` after sweep CSVs exist.",
)

# %% [markdown]
# Figure 4: Intermediate parameter diagnostics. Drizzle, MAP-TV, and TGV process proxies explain candidate selection behavior.

# %% [markdown]
# > **图表说明**: 这张图把已有产物中的中间诊断压缩成四个面板。Drizzle 左上/右上展示 pixfrac 改变时稳定性、伪影和 coverage 的变化；MAP-TV 左下展示不同 PSF sigma 下 `lambda_TV` 的路径；TGV 右下展示各候选参数的稳定性和伪影 proxy。
# >
# > **怎么看**: Drizzle 的 pixfrac 越小，drop footprint 越小，理论上轮廓更尖但 coverage 更稀；MAP-TV 的 lambda 越大，正则越强，可能稳定但也可能过平滑；TGV 的 artifact score 升高通常提示背景纹理或振铃风险。
# >
# > **异常是否正常**: 这些是过程 proxy，不是最终清晰度证据。Drizzle 没有 objective curve 是算法性质决定的；MAP-TV/TGV 当前完整长跑旧产物没有保存每轮 records，后续 runner 已补 convergence CSV 输出。
# >
# > **核心发现**: 中间诊断能解释“为什么某个候选入选”，但最终仍要结合中心 ROI highpass 图、coverage 和 raw-control 控制图。

# %%
show_optional_fig(
    "split_holdout_distribution_diagnostics.png",
    "Pending: split_holdout_distribution_diagnostics.png (requires MAP-TV/TGV detail CSVs).",
)

# %% [markdown]
# Figure 5: Split and holdout distribution diagnostics. Per-split and per-frame distributions expose instability hidden by aggregate medians.

# %% [markdown]
# > **图表说明**: 这张图展示逐 split 和逐 holdout frame 的分布，而不是只看 CSV 中的 median。箱线图隐藏离群点，目的是看主体分布是否稳定。
# >
# > **怎么看**: 箱体越低越窄，说明不同 split 或 holdout frame 上结果越稳定。若 median 不差但箱体很宽，说明算法可能只在部分帧或部分拆分上表现好。
# >
# > **异常是否正常**: 如果 MAP-TV 细节 CSV 缺失，这是旧产物没有保存细节；新的 MAP-TV runner 已补。TGV 细节 CSV 已有，因此通常能看到 TGV 的分布。
# >
# > **核心发现**: 分布图能暴露单一 median 掩盖的不稳定性，是选择最终候选前必须看的质量门控。
