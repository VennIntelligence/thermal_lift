# %% [markdown]
# ## 5. 旋转角 θ 拟合
#
# 这里同时报告两个坐标约定：
#
# - `image-row`: 直接使用数组行方向，`dy_px` 向下为正
# - `y-up diagnostic`: 将 `dy_px` 取反，用于和常见笛卡尔像面坐标对照
#
# 两者都不能把全部帧对解释为一个可信的刚性旋转模型。

# %%
fit_image_row = fit_rotation_angle(
    measurements["dx_px"],
    measurements["dy_px"],
    measurements["delta_X_um"],
    measurements["delta_Y_um"],
    PIXEL_SIZE_UM,
)

measurements_y_up = measurements.copy()
measurements_y_up["dy_px"] = -measurements_y_up["dy_px"]
fit_y_up = fit_rotation_angle(
    measurements_y_up["dx_px"],
    measurements_y_up["dy_px"],
    measurements_y_up["delta_X_um"],
    measurements_y_up["delta_Y_um"],
    PIXEL_SIZE_UM,
)

axis_fits = []
for scan_axis, subset in measurements_y_up.groupby("scan_axis"):
    fit = fit_rotation_angle(
        subset["dx_px"],
        subset["dy_px"],
        subset["delta_X_um"],
        subset["delta_Y_um"],
        PIXEL_SIZE_UM,
    )
    axis_fits.append({
        "scan_axis": scan_axis,
        "theta_deg": fit["theta_deg"],
        "rms_error_px": fit["rms_error_px"],
        "effective_pixel_size_um": fit["effective_pixel_size_um"],
        "n_pairs": fit["n_pairs"],
    })
axis_fit_df = pd.DataFrame(axis_fits)

print(f"Image-row theta: {fit_image_row['theta_deg']:.3f} deg, RMS={fit_image_row['rms_error_px']:.4f} px")
print(f"Y-up diagnostic theta: {fit_y_up['theta_deg']:.3f} deg, RMS={fit_y_up['rms_error_px']:.4f} px")
print(f"Reference theta: {REFERENCE_THETA_DEG:.3f} deg")
axis_fit_df

# %% [markdown]
# > **数据说明**: 这里用全部 NCC 位移反推旋转角，同时给出 image-row 和 y-up 两种坐标约定。
# > `axis_fit_df` 进一步把 X-scan 与 Y-scan 分开拟合。
# >
# > **数据分布**: y-up 诊断 θ 约为 34.11°，远离旧配置 47.6°；
# > 但全局旋转模型 RMS 残差约 0.175 px，已经超过 SR 位移建模的实用容差。
# >
# > **核心发现**: 当前数据不能用来“更新成 34.11°”。
# > 更准确的判断是：NCC 位移和旧模型不一致，同时也没有形成一个足够好的新单一旋转模型。

# %%
residual_df = measurements_y_up.copy()
pred_dx, pred_dy = coordinate_to_shift(
    residual_df["delta_X_um"],
    residual_df["delta_Y_um"],
    fit_y_up["theta_deg"],
    PIXEL_SIZE_UM,
)
residual_df["pred_dx_px"] = pred_dx
residual_df["pred_dy_px"] = pred_dy
residual_df["residual_dx_px"] = residual_df["dx_px"] - residual_df["pred_dx_px"]
residual_df["residual_dy_px"] = residual_df["dy_px"] - residual_df["pred_dy_px"]
residual_df["residual_norm_px"] = np.hypot(residual_df["residual_dx_px"], residual_df["residual_dy_px"])

fig, axes = make_figure("double_col", nrows=1, ncols=2, height=3.2)
axes[0].scatter(
    np.hypot(residual_df["delta_X_um"], residual_df["delta_Y_um"]),
    residual_df["residual_norm_px"],
    s=18,
    alpha=0.7,
    color=METHOD_COLOR_LIST[0],
)
axes[0].axhline(0.1, color=METHOD_COLOR_LIST[2], ls="--", lw=0.9, label="0.1 px")
axes[0].set_xlabel(r"Commanded step [$\mu$m]")
axes[0].set_ylabel("Residual norm [px]")
axes[0].set_title("Rotation Model Residual")
axes[0].legend(loc="best")

axes[1].scatter(residual_df["pred_dx_px"], residual_df["dx_px"], s=18, alpha=0.7,
                color=METHOD_COLOR_LIST[1], label="dx")
axes[1].scatter(residual_df["pred_dy_px"], residual_df["dy_px"], s=18, alpha=0.7,
                color=METHOD_COLOR_LIST[2], label="dy")
lims = [
    min(residual_df[["pred_dx_px", "pred_dy_px", "dx_px", "dy_px"]].min()),
    max(residual_df[["pred_dx_px", "pred_dy_px", "dx_px", "dy_px"]].max()),
]
axes[1].plot(lims, lims, "--", color="#555555", lw=0.8)
axes[1].set_xlabel("Predicted component [px]")
axes[1].set_ylabel("Measured component [px]")
axes[1].set_title("Fitted Model vs Measurement")
axes[1].legend(loc="best")
save_fig(fig, "theta_residuals.png")

# %% [markdown]
# > **图表说明**: 左图展示每个帧对在拟合旋转模型下的残差模长，右图展示模型预测分量与实测分量的 1:1 对比。
# >
# > **数据分布**: 很多帧对残差超过 0.1 px；预测-实测散点没有紧贴 1:1 虚线。
# > 残差不是随机小噪声，而是按扫描轴/步长形成结构性偏差。
# >
# > **核心发现**: 旋转角统计值本身不够用。只要残差仍在 0.1 px 量级以上，
# > 2×/4× SR 的亚像素位移模型就不能被认为可靠。
