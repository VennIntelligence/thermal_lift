# %% [markdown]
# ## 8. 位移台线性度
#
# 线性度用旧 θ=47.6° 模型方向投影来检查。由于 Y-scan 已经显示非线性/非刚性行为，这里的回归主要用于量化偏差，不作为通过判据。

# %%
ref_dx, ref_dy = coordinate_to_shift(
    measurements_y_up["delta_X_um"],
    measurements_y_up["delta_Y_um"],
    REFERENCE_THETA_DEG,
    PIXEL_SIZE_UM,
)
ref_mag = np.hypot(ref_dx, ref_dy)
unit_dx = np.divide(ref_dx, ref_mag, out=np.zeros_like(ref_dx, dtype=float), where=ref_mag > 0)
unit_dy = np.divide(ref_dy, ref_mag, out=np.zeros_like(ref_dy, dtype=float), where=ref_mag > 0)
measured_parallel = measurements_y_up["dx_px"] * unit_dx + measurements_y_up["dy_px"] * unit_dy

linearity_projection = linearity_regression(ref_mag, measured_parallel)
linearity_dx = linearity_regression(ref_dx, measurements_y_up["dx_px"])
linearity_dy = linearity_regression(ref_dy, measurements_y_up["dy_px"])

linearity_df = pd.DataFrame([
    {"component": "projection", **linearity_projection},
    {"component": "dx", **linearity_dx},
    {"component": "dy", **linearity_dy},
])
linearity_df.to_csv(OUTPUT_DIR / "linearity.csv", index=False)
print(linearity_df.to_string(index=False))
print("Saved: output/ep02_displacement_calibration/linearity.csv")

# %% [markdown]
# > **数据说明**: 线性度表把旧 θ=47.6° 模型方向上的标称位移与实测投影做线性回归，
# > 同时分别检查 `dx` 和 `dy` 分量。
# >
# > **数据分布**: 投影方向的 R² 约 0.03，远低于接近 1 的理想线性关系。
# > 这说明误差不是简单的比例缩放可以修正。
# >
# > **核心发现**: 当前问题不能靠把 θ 或 pixel size 微调一下解决。
# > 需要先回查 Y-scan 帧对、采集顺序、方向符号和互相关估计偏差。

# %%
plot_linearity(
    ref_mag,
    measured_parallel,
    linearity_projection,
    xlabel="Nominal displacement magnitude [px]",
    ylabel="Measured projection [px]",
    title="Linearity Along Reference Model Direction",
    save_path="linearity_regression.png",
    save_fn=save_fig,
)

# %% [markdown]
# > **图表说明**: 左图显示标称位移幅值与实测投影的回归关系，右图显示回归残差。
# >
# > **数据分布**: 散点没有沿一条递增直线排列，残差呈结构化分布；
# > 这与 Y-scan 2 µm/4 µm 幅值反常相互印证。
# >
# > **核心发现**: 线性度失败是 EP02 最重要的否决证据之一。
# > 在修复这个问题前，不能把当前位移表用于 SR 重建。
