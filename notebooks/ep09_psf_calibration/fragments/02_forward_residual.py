# %% [markdown]
# ## Route A — Forward-Model 残差反推
#
# Route A 使用 EP06 MAP-TV 2x highpass 作为 pseudo-HR，扫描 Gaussian PSF sigma，然后把 forward model 投影回 LR，与 255 帧主 session highpass 观测比较。Train/val split 按采集顺序固定切分，避免按文件名制造伪时序。

# %%
display(show_png("forward_residual_curve.png"))

if not route_table.empty:
    display(route_table[route_table["route"].eq("A_forward")].round(4))

forward_curve = read_csv("forward_residual_fine_sweep.csv")
if not forward_curve.empty:
    best_rows = (
        forward_curve.sort_values(["split", "mean_mse"])
        .groupby("split", as_index=False)
        .head(5)
        [["split", "sigma_lr_px", "mean_mse", "median_mse", "p95_mse"]]
    )
    display(best_rows.round(6))

# %% [markdown]
# > **图表说明**: 残差曲线的横轴是 candidate PSF sigma，纵轴是 forward prediction 与真实 LR highpass frame 的 MSE；红色虚线是 Route A 的 train optimum。
# >
# > **怎么看**: 残差越低越好。如果曲线在最小值附近非常平，说明 forward residual 对 sigma 不敏感，即使优化器能给出数值，也不能把它当成强物理证据。
# >
# > **异常是否正常**: 本次 train 最小值约在 0.18 px，但相对边界的 residual depth 只有约 0.0001，val optimum 贴近 fine sweep 下界 0.13 px；这不是缺数据，而是曲线分辨率不足。
# >
# > **核心发现**: Route A 支持“小 effective sigma”方向，但 residual minimum 不够清晰，因此不能单独完成 ±0.05 px 的物理标定。
