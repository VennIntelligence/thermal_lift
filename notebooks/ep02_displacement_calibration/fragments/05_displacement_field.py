# %% [markdown]
# ## 4. 位移向量场
#
# 实测向量场和 θ=47.6° 的旧模型不一致；主要异常来自 Y-scan：2 µm 和 4 µm 的实测位移不呈线性倍增。

# %%
plot_displacement_field(
    measurements,
    theta_deg=REFERENCE_THETA_DEG,
    save_path="displacement_vector_field.png",
    save_fn=save_fig,
)

# %% [markdown]
# > **图表说明**: 位移向量场图展示 X-scan 和 Y-scan 帧对的实测 `(dx, dy)`，
# > 并叠加 θ=47.6° 旧模型给出的参考方向。
# >
# > **数据分布**: 实测向量按扫描轴形成两个主要方向簇，但这些簇和旧模型参考向量并不重合。
# > Y-scan 的 2 µm/4 µm 聚类尤其异常。
# >
# > **核心发现**: 当前 NCC 位移不支持直接沿用 θ=47.6° 的单一刚性旋转模型。
# > 这不是“看起来有位移”能解决的问题，必须定量检查方向和幅值。

# %%
nominal_dx, nominal_dy = coordinate_to_shift(
    measurements["delta_X_um"],
    measurements["delta_Y_um"],
    REFERENCE_THETA_DEG,
    PIXEL_SIZE_UM,
)

fig, axes = make_figure("double_col", nrows=1, ncols=2, height=3.2)
axes[0].scatter(nominal_dx, measurements["dx_px"], s=18, alpha=0.7, color=METHOD_COLOR_LIST[0])
axes[0].plot([nominal_dx.min(), nominal_dx.max()], [nominal_dx.min(), nominal_dx.max()],
             "--", color="#555555", lw=0.8)
axes[0].set_xlabel("Reference-model dx [px]")
axes[0].set_ylabel("Measured dx [px]")
axes[0].set_title("dx: Measured vs Reference")

axes[1].scatter(nominal_dy, measurements["dy_px"], s=18, alpha=0.7, color=METHOD_COLOR_LIST[1])
axes[1].plot([nominal_dy.min(), nominal_dy.max()], [nominal_dy.min(), nominal_dy.max()],
             "--", color="#555555", lw=0.8)
axes[1].set_xlabel("Reference-model dy [px]")
axes[1].set_ylabel("Measured dy [px]")
axes[1].set_title("dy: Measured vs Reference")
save_fig(fig, "dx_dy_vs_coordinates.png")

# %% [markdown]
# > **图表说明**: 两个散点图把旧 θ=47.6° 模型预测的 `dx/dy` 与 NCC 实测 `dx/dy` 逐帧对比。
# > 虚线代表理想的一致关系。
# >
# > **数据分布**: 点云没有紧贴 1:1 虚线，尤其 Y 方向分量偏差明显。
# > 这说明偏差不是单个异常点，而是成组出现的模型不一致。
# >
# > **核心发现**: 旧 θ 不能解释当前实测位移场；不过这还不能证明“新 θ”可信，
# > 因为同一张图也显示了幅值和线性关系存在问题。

# %% [markdown]
# ### 4.1 Session 与步长方向诊断
#
# 旧版 EP02 因为继承了文件名排序 session，容易把早期低温/补采帧误解为多个 session。
# 当前版本只用主扫描 session；这里按扫描轴和步长拆开，检查 Y-scan 异常是否仍然存在。

# %%
measurements_y_up = measurements.copy()
measurements_y_up["dy_px"] = -measurements_y_up["dy_px"]

direction_rows = []
for (session, scan_axis, delta_um), subset in measurements_y_up.groupby(["session", "scan_axis", "delta_um"]):
    median_dx = float(subset["dx_px"].median())
    median_dy = float(subset["dy_px"].median())
    direction_rows.append({
        "session": int(session),
        "scan_axis": scan_axis,
        "delta_um": float(delta_um),
        "n_pairs": int(len(subset)),
        "median_dx_px": median_dx,
        "median_dy_px_y_up": median_dy,
        "median_magnitude_px": float(np.hypot(median_dx, median_dy)),
        "median_angle_deg_y_up": float(np.degrees(np.arctan2(median_dy, median_dx))),
    })
direction_df = pd.DataFrame(direction_rows)
direction_df.to_csv(OUTPUT_DIR / "motion_direction_diagnostic.csv", index=False)
direction_df

# %% [markdown]
# > **数据说明**: 这张表按 `session × scan_axis × delta_um` 汇总主扫描 session 内 NCC 位移的中位方向和中位幅值。
# > 角度使用 y-up 坐标，便于和旋转模型直观对比。
# >
# > **数据分布**: 早期低温/补采帧已被排除，因此这里不再比较多个温度 session。
# > X-scan 的 2/4 µm 幅值基本随步长增大；Y-scan 的 4 µm 幅值仍小于 2 µm。
# >
# > **核心发现**: session 排序修正后，Y-scan 非线性仍然存在。
# > 这说明问题不只是温度跳队污染，更可能在 Y-scan 帧对构造、真实运动方向、回程/间隙或 NCC 偏置上。

# %%
fig, axes = make_figure("double_col", nrows=1, ncols=2, height=3.2)
marker_by_delta = {2.0: "o", 4.0: "s"}
color_by_axis = {"x": METHOD_COLOR_LIST[0], "y": METHOD_COLOR_LIST[1]}

for (scan_axis, delta_um), subset in direction_df.groupby(["scan_axis", "delta_um"]):
    label = f"{scan_axis.upper()} {delta_um:.0f} um"
    axes[0].scatter(
        subset["session"],
        subset["median_angle_deg_y_up"],
        s=36,
        marker=marker_by_delta.get(float(delta_um), "o"),
        color=color_by_axis[scan_axis],
        alpha=0.8,
        label=label,
    )
    axes[1].scatter(
        subset["delta_um"],
        subset["median_magnitude_px"],
        s=36,
        marker=marker_by_delta.get(float(delta_um), "o"),
        color=color_by_axis[scan_axis],
        alpha=0.8,
        label=label,
    )

axes[0].set_xlabel("Session")
axes[0].set_ylabel("Median direction [deg, y-up]")
axes[0].set_title("Direction by Session")
axes[1].set_xlabel(r"Commanded step [$\mu$m]")
axes[1].set_ylabel("Median displacement [px]")
axes[1].set_title("Magnitude by Step")
axes[1].legend(loc="best", fontsize=7)
save_fig(fig, "motion_direction_diagnostic.png")

# %% [markdown]
# > **图表说明**: 左图显示主扫描 session 的中位位移方向，右图显示不同命令步长下的中位位移幅值。
# > 颜色区分 X/Y 扫描轴，符号区分 2 µm 和 4 µm。
# >
# > **数据分布**: 主扫描 session 内方向分成 X/Y 两组；更突出的异常出现在右图：
# > Y-scan 的 4 µm 位移幅值没有比 2 µm 更大，反而明显更小。
# >
# > **核心发现**: “跳队帧污染”已排除，但主扫描内部 Y-scan 仍不满足线性位移。
# > 目前最需要回查的是 Y-scan 的帧对构造、采集顺序、运动回程或 NCC 对温度场变化的偏置。
