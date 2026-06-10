# %% [markdown]
# ## Proxy Metrics
#
# 这些指标只辅助解释视觉结果：split-half NRMSE 看拆半稳定性，artifact score 看一阶伪影风险，raw-control correlation 看 highpass 结构与 raw-mean 控制轨道的一致性。它们不能替代中心 ROI 的视觉检查，也不能作为独立光学 ground truth。

# %%
summary = read_csv("comparison_summary.csv")
display(summary.round(6))

# %%
numeric = summary.copy()
for column in ("split_half_nrmse", "artifact_score", "raw_control_corr"):
    if column in numeric:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")

if {"method", "split_half_nrmse", "artifact_score", "raw_control_corr"}.issubset(numeric.columns):
    display(
        numeric[["method", "split_half_nrmse", "artifact_score", "raw_control_corr"]]
        .assign(
            split_half_direction="lower is better",
            artifact_direction="lower is usually better",
            raw_control_direction="higher is better",
        )
        .round(6)
    )

# %% [markdown]
# > **数据说明**: `comparison_summary.csv` 包含 EP12@2000 和裸 drizzle 4x 的共同 proxy。EP12 的 split-half NRMSE 是 benchmark 脚本用同一 checkpoint 对随机拆半子集重建后计算的；裸 drizzle 行未重算 split-half，artifact score 和 raw-control corr 仍可用于辅助解释。
# >
# > **怎么看**: split-half NRMSE 越小通常表示重建对帧子集更稳定；artifact score 越小通常表示 ringing/blockiness proxy 更低；raw-control correlation 越大通常表示结构方向更接近 raw-temperature 控制轨。裸 drizzle 的 artifact score 往往更高，因为稀疏采样网格会在 highpass 域留下强伪影。
# >
# > **异常是否正常**: split-half 很低常见于过度平滑，artifact 很低也可能只是高频被抹掉；raw-control correlation 很高常见于低频或强公共结构，不等于真实超分辨率。EP12 split-half 计算成本高，benchmark 默认 `n_splits=1` 是快速推演口径。
# >
# > **核心发现**: 指标只能帮助解释“看起来更清楚”是否伴随明显稳定性或伪影风险；最终仍要回到 Figure 1 的 EP12-vs-drizzle 同域中心 ROI。
