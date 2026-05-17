# %% [markdown]
# ## 3. 亚像素互相关位移估计
#
# 使用中心 320×320 ROI、搜索半径 ±5 pixel、NCC 峰值 5×5 二次曲面拟合。  
# 注意：NCC 返回的是图像数组坐标，`dy_px` 正方向为行号向下。

# %%
measurements = measure_frame_pairs(
    frame_pairs,
    DATA_DIR,
    roi_size=ROI_SIZE,
    search_radius=SEARCH_RADIUS,
    method="ncc",
)
measurements.to_csv(OUTPUT_DIR / "displacement_measurements.csv", index=False)

quality = {
    "n_pairs": len(measurements),
    "fit_ok_rate": float(measurements["fit_ok"].mean()),
    "edge_peak_count": int(measurements["edge_peak"].sum()),
    "median_peak_ncc": float(measurements["peak_ncc"].median()),
}
print(json.dumps(quality, indent=2))
print("Saved: output/ep02_displacement_calibration/displacement_measurements.csv")

disp_summary = (
    measurements.groupby(["scan_axis", "delta_um"])[["dx_px", "dy_px", "peak_ncc"]]
    .agg(["count", "mean", "std", "median"])
)
disp_summary

# %% [markdown]
# > **数据说明**: `quality` 汇总 NCC 拟合质量，`disp_summary` 按扫描轴和命令步长汇总实测位移分量。
# > `dy_px` 使用图像行坐标，正方向为向下。
# >
# > **数据分布**: NCC 峰值中位数约 0.995，拟合成功率为 1.0，说明图像块之间高度相似。
# > 但高相关峰只证明局部图像能对齐，不证明位移模型满足刚性旋转。
# >
# > **核心发现**: 互相关本身能稳定找到峰值；真正的问题在于这些峰值对应的位移是否符合物理模型。
# > 后续必须用向量场、θ 拟合和线性度检查来验证。

# %%
fig, axes = make_figure("double_col", nrows=1, ncols=2, height=3.0)
axes[0].hist(measurements["dx_px"], bins=40, color=METHOD_COLOR_LIST[0], edgecolor="white")
axes[0].set_xlabel("dx [px]")
axes[0].set_ylabel("Pair Count")
axes[0].set_title("Measured dx Distribution")
axes[1].hist(measurements["dy_px"], bins=40, color=METHOD_COLOR_LIST[1], edgecolor="white")
axes[1].set_xlabel("dy [px]")
axes[1].set_ylabel("Pair Count")
axes[1].set_title("Measured dy Distribution")
save_fig(fig, "displacement_histograms.png")

# %% [markdown]
# > **图表说明**: 两个直方图分别展示全部相邻帧对的实测 `dx_px` 和 `dy_px` 分布。
# >
# > **数据分布**: 位移集中在小于 0.4 px 的亚像素范围内，但分布不是单峰；
# > X-scan、Y-scan 与 2/4 µm 步长混在一起形成多个簇。
# >
# > **核心发现**: 数据确实包含亚像素微扫描位移，但单看直方图无法判断 θ 是否正确。
# > 必须按扫描轴、步长和 session 分组解释。
