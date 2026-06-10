# %% [markdown]
# ## Three-Algorithm Metric Summary
#
# 这里把 Drizzle、MAP-TV 和 TGV 的真实数据 sweep 结果归一到同一组 proxy 指标。每个算法的“best”按 split-half NRMSE、artifact score、holdout MSE 的顺序排序；MAP-TV 若存在 `best_params.json`，优先使用其中记录的 Pareto top。

# %%
import pandas as pd
from IPython.display import Markdown, display

if not summary_table.empty:
    display(summary_table.round(6))
else:
    display(Markdown("**No comparable sweep rows found.**"))

# %%
top_tables = []
for method in METHOD_ORDER:
    table = rank_table(sweeps.get(method, pd.DataFrame()), n=3)
    if not table.empty:
        top_tables.append(table)
if top_tables:
    display(pd.concat(top_tables, ignore_index=True).round(6))

# %%
if not summary_table.empty:
    show_optional_fig(
        "core_metric_comparison.png",
        "Pending: run `uv run python scripts/build_ep10_cache.py` to build core_metric_comparison.png",
    )

# %% [markdown]
# Figure 1: Core metric comparison. Drizzle, MAP-TV, and TGV best candidates are compared on common stability, artifact, and holdout proxies.

# %% [markdown]
# > **数据说明**: 第一张表是三算法各自当前 best 候选；第二张表展开每个算法最多前三个候选。split-half NRMSE 衡量拆半稳定性，holdout MSE 衡量 forward prediction 对留出帧的贴合，artifact score 是伪影 proxy，raw-control correlation 是与 offset-corrected raw 控制轨的结构相关性。为了和 MAP-TV/TGV 统一口径，Drizzle 的 artifact score 若检测到对应 NPY，会在 Notebook 展示层按 `artifact_score(hr)` 重新计算，不使用旧 CSV 中带 LR overshoot 分量的值。
# >
# > **怎么看**: split-half NRMSE、holdout MSE、artifact score 通常越小越好；raw-control correlation 越大越好。但 split-half 很低可能来自过强平滑，artifact score 低也不保证轮廓真实，所以后面必须看 highpass ROI 图。
# >
# > **异常是否正常**: TGV CSV 使用 `split_half_nrmse_median`，这里归一成 `split_half_nrmse`；Drizzle 使用 `pixfrac` 而不是 `lambda/sigma`。`raw_control_corr` 在旧产物中由各算法自己的控制轨生成，只能辅助判断结构方向，不能作为严格公平排名列。
# >
# > **核心发现**: 这张表把 EP10 从单 MAP-TV 参数扫描提升为三算法同屏比较；候选筛选应以“稳定、低伪影、视觉轮廓合理”合取，而不是单列最优。
