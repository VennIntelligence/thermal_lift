# %% [markdown]
# ## Route A — Forward-Model 残差反推
#
# Route A 使用新 EP06 248 clean-frame MAP-TV 2x highpass 主 baseline (`output/ep06_sr_poc/map_tv_highpass.npy`) 作为 pseudo-HR，扫描 Gaussian PSF sigma，然后把 forward model 投影回 LR，与同一 clean-frame 筛选口径下的 248 帧 main-session highpass 观测比较。Train/val split 按采集顺序固定切分，避免按文件名制造伪时序。

# %%
show_fig("forward_residual_curve.png")

# %% [markdown]
# Figure 1: Forward residual PSF sweep. Candidate Gaussian sigma values are compared by LR highpass prediction error.

# %%
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
# > **异常是否正常**: 本次 Route A 最优值为 0.2257 px，95% CI 为 [0.2084, 0.2402] px，train/validation delta 约 0.0047 px；residual depth 约 0.0010，刚好清除单路线曲线门控。
# >
# > **核心发现**: Route A 在 EP06 248 clean-frame 主 baseline 上给出较稳定的小 effective sigma，但它仍必须接受 Route B/C 的交叉检查，不能单独完成物理可行性判决。
