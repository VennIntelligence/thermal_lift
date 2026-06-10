# %% [markdown]
# ## Center ROI Highpass Visual Comparison
#
# 三算法最重要的同屏证据是中心 ROI highpass 横向图。Drizzle 使用指标 best pixfrac，MAP-TV 使用 `best_params.json` 的 top/best，TGV 使用 `run_summary.best_label` 对应的 best highpass 输出。

# %%
show_optional_fig(
    "center_roi_highpass_comparison.png",
    "Pending: run `uv run python scripts/build_ep10_cache.py` after algorithm HR NPY outputs exist.",
)

# %% [markdown]
# Figure 2: Center ROI highpass comparison. Best Drizzle, MAP-TV, and TGV candidates are shown on the same center crop and color scale.

# %% [markdown]
# > **图表说明**: 这张图把 Drizzle、MAP-TV 和 TGV 的 best 候选裁到同一中心 ROI，并使用统一的红蓝对称色标。白色表示接近局部背景零响应，红/蓝表示相对背景的正/负 highpass 结构响应。
# >
# > **怎么看**: 重点看芯片内部轮廓是否连续、直线/折线是否几何稳定、背景是否出现方向性条纹或棋盘纹。Highpass 图强调边缘和局部结构，不是普通温度图；边缘更亮不必然代表真实分辨率更高。
# >
# > **异常是否正常**: 如果 TGV 或 MAP-TV 的边缘更强但背景也出现大面积彩色纹理，应结合 artifact score 降级解释。若 Drizzle 与 raw-control 很接近，它可能更保守、更稳定，但未必提供最多 contour 增益。
# >
# > **核心发现**: 视觉结论必须和指标表合取：理想候选应同时具备清楚轮廓、低 split-half NRMSE、低 artifact score，并且没有明显违背 raw-control 的结构。

# %%
show_optional_fig(
    "auxiliary_control_views.png",
    "Pending: auxiliary_control_views.png (requires Drizzle coverage/raw-control NPY).",
)

# %% [markdown]
# Figure 3: Auxiliary control views. Drizzle coverage and raw-control highpass views provide controls for interpreting SR structure.

# %% [markdown]
# > **图表说明**: 辅助图优先展示 Drizzle best pixfrac 的 coverage 和 raw-control highpass。Coverage 表示同一 HR 网格位置被输入帧采样/加权覆盖的程度；raw-control highpass 是不使用迭代 SR 的普通控制轨结构参考。
# >
# > **怎么看**: Coverage 过低的区域更容易出现插值或权重伪影；raw-control highpass 中已经存在的结构可以作为“保守参照”。如果 SR 图中出现 raw-control 完全没有的规则纹理，需要谨慎解释。
# >
# > **异常是否正常**: Coverage 图的颜色不是温度，不能和 highpass 图的红蓝含义混用。Raw-control highpass 中白色仍表示接近零的局部变化，红/蓝仍表示相对背景的正/负结构。
# >
# > **核心发现**: 辅助图帮助区分真实 contour 增益和覆盖/高频增强带来的伪结构，是三算法视觉裁决的控制层。
