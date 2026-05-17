# %% [markdown]
# ## 6. Bootstrap 置信区间
#
# Bootstrap 只量化“在当前 NCC 测量和 y-up 约定下”的 θ 统计不确定度；它不消除系统误差。由于模型 RMS 残差超过 0.1 pixel，本集不建议把该 θ 直接写入全局配置。

# %%
bootstrap = bootstrap_theta_ci(
    measurements_y_up,
    n_bootstrap=10000,
    ci=0.95,
    pixel_size_um=PIXEL_SIZE_UM,
    seed=20260517,
)
ci_bounds = (bootstrap["ci_lower"], bootstrap["ci_upper"])
reference_in_ci = ci_bounds[0] <= REFERENCE_THETA_DEG <= ci_bounds[1]

theta_estimate = {
    "reference_theta_deg": REFERENCE_THETA_DEG,
    "theta_deg_image_row": float(fit_image_row["theta_deg"]),
    "theta_deg_y_up_diagnostic": float(fit_y_up["theta_deg"]),
    "ci_lower_y_up": float(ci_bounds[0]),
    "ci_upper_y_up": float(ci_bounds[1]),
    "bootstrap_std_deg": float(bootstrap["theta_std"]),
    "reference_in_y_up_ci": bool(reference_in_ci),
    "rms_error_px_y_up": float(fit_y_up["rms_error_px"]),
    "n_pairs": int(fit_y_up["n_pairs"]),
    "valid_single_rotation_model": bool(fit_y_up["rms_error_px"] < 0.1),
}
(OUTPUT_DIR / "theta_estimate.json").write_text(
    json.dumps(theta_estimate, indent=2),
    encoding="utf-8",
)

print(json.dumps(theta_estimate, indent=2))
print("Saved: output/ep02_displacement_calibration/theta_estimate.json")

# %% [markdown]
# > **数据说明**: `theta_estimate.json` 保存当前 NCC 位移下的 θ 拟合结果、bootstrap 置信区间、
# > 旧 θ 是否落入区间，以及单一旋转模型是否通过 RMS 阈值。
# >
# > **数据分布**: bootstrap 区间很窄，但它围绕的是当前 NCC 测量得到的错误/不完整模型；
# > 47.6° 不在该区间内，模型 RMS 也没有通过 0.1 px 阈值。
# >
# > **核心发现**: bootstrap 只能说明当前测量内部的统计稳定性，不能排除系统误差。
# > 因此这个 JSON 适合记录“首轮验证失败”，不适合改写全局标定。

# %%
plot_theta_bootstrap(
    bootstrap["theta_samples"],
    ci_bounds,
    reference_deg=REFERENCE_THETA_DEG,
    save_path="theta_bootstrap.png",
    save_fn=save_fig,
)

# %% [markdown]
# > **图表说明**: 直方图展示 bootstrap 重采样得到的 θ 分布，虚线标出 95% CI，
# > 竖线标出旧配置 θ=47.6°。
# >
# > **数据分布**: 重采样分布集中在 34° 附近，旧 θ 明显落在分布外。
# >
# > **核心发现**: 旧 θ 没有通过当前 NCC 独立检查；
# > 但由于残差和线性度问题，新 θ 也不能直接进入 `stage_calibration.json`。
