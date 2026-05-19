# %% [markdown]
# ## Route C — 多帧联合 PSF 估计
#
# Route C 对每个候选 sigma 用 48 个 train frames 跑 20 步短预算 MAP-TV，再用 32 个 hold-out frames 计算 forward residual。它不依赖 EP06 固定 HR，但预算比完整 EP06 小，因此作为交叉验证而非主估计。

# %%
display(show_png("joint_sigma_curve.png"))

joint_sweep = read_csv("joint_sigma_sweep.csv")
if not joint_sweep.empty:
    display(
        joint_sweep[
            [
                "sigma_lr_px",
                "holdout_mse",
                "train_mse",
                "iterations",
                "final_relative_update",
                "stopped",
            ]
        ].round(6)
    )

# %% [markdown]
# > **图表说明**: 曲线展示短预算 MAP-TV 对不同 sigma 的 hold-out MSE。表格同时给出 train MSE 和最后一次迭代的 relative update。
# >
# > **怎么看**: Hold-out MSE 越低越好；如果最小值在 grid 边界，说明扫描区间只得到方向性结论，不能给出封闭的最优值。
# >
# > **异常是否正常**: 本次 grid minimum 在 0.10 px 下界，parabolic estimate 约 0.119 px。`stopped=False` 表示 20 步预算用完，不代表失败，而是说明 Route C 是低预算 cross-check。
# >
# > **核心发现**: Route C 与 Route A 同样偏向小 sigma，但它的最小值贴边，因此也不能独立满足“精确 ±0.05 px”的标定要求。
