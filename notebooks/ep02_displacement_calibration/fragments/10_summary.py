# %% [markdown]
# ## 9. 综合评估
#
# **结论**: 修正采集顺序并只保留主扫描 session 后，NCC 测量仍不能独立确认 θ=47.6°。
# 相邻帧互相关峰值很高，但实测位移不满足单一旋转 + 固定 20 µm/px 模型，
# 尤其 Y-scan 的 2 µm 与 4 µm 位移不呈线性关系。后续 SR 不应直接用本轮 NCC 结果替换全局 θ。

# %%
model_error_px = float(fit_y_up["rms_error_px"])
repeat_p95_px = float(repeatability_summary["p95_error_px_valid"])
sr_status = "not_supported" if model_error_px > 0.1 or repeat_p95_px > 0.1 else "supported"
repeat_median_text = (
    "n/a"
    if np.isnan(repeatability_summary["median_error_px_valid"])
    else f"{repeatability_summary['median_error_px_valid']:.4f} px = {repeatability_summary['median_error_um_valid']:.3f} um"
)
repeat_p95_text = (
    "n/a"
    if np.isnan(repeatability_summary["p95_error_px_valid"])
    else f"{repeatability_summary['p95_error_px_valid']:.4f} px = {repeatability_summary['p95_error_um_valid']:.3f} um"
)

summary_table = pd.DataFrame([
    {
        "parameter": "theta",
        "reference": f"{REFERENCE_THETA_DEG:.1f} deg",
        "ep02_value": f"{fit_y_up['theta_deg']:.2f} deg [{ci_bounds[0]:.2f}, {ci_bounds[1]:.2f}]",
        "status": "not accepted",
    },
    {
        "parameter": "rotation_model_rms",
        "reference": "< 0.1 px target",
        "ep02_value": f"{model_error_px:.4f} px",
        "status": "too high",
    },
    {
        "parameter": "repeatability_p95",
        "reference": "< 0.1 px target",
        "ep02_value": "n/a" if np.isnan(repeat_p95_px) else f"{repeat_p95_px:.4f} px",
        "status": "insufficient data" if np.isnan(repeat_p95_px) else ("risk" if repeat_p95_px > 0.1 else "ok"),
    },
    {
        "parameter": "linearity_projection_r2",
        "reference": "near 1.0",
        "ep02_value": f"{linearity_projection['r2']:.4f}",
        "status": "poor" if linearity_projection["r2"] < 0.9 else "ok",
    },
])
summary_table

# %% [markdown]
# > **数据说明**: 这张总表把主扫描 session 下的 θ、旋转模型残差、repeatability 和线性度压缩成 SR 决策指标。
# >
# > **数据分布**: θ 的数值偏离旧配置，但更关键的是模型 RMS、repeat p95 和线性 R² 都没有达到可接受水平。
# >
# > **核心发现**: 排除早期低温/补采帧后，EP02 的结论仍不是“找到了新 θ”，
# > 而是“当前 NCC 证据不足以支持可靠 SR 位移建模”。全局配置应保持不变。

# %%
fig, ax = make_figure("single_col", height=3.0)
labels = ["Model RMS"]
values = [model_error_px]
colors = [METHOD_COLOR_LIST[0]]
if not np.isnan(repeat_p95_px):
    labels.append("Repeat p95")
    values.append(repeat_p95_px)
    colors.append(METHOD_COLOR_LIST[1])
ax.bar(labels, values, color=colors)
if np.isnan(repeat_p95_px):
    ax.text(0.5, 0.78, "Repeat p95: insufficient valid same-session pairs",
            transform=ax.transAxes, ha="center", va="center", fontsize=8)
ax.axhline(0.05, color=METHOD_COLOR_LIST[2], ls="--", lw=0.9, label="0.05 px")
ax.axhline(0.10, color="#444444", ls=":", lw=1.0, label="0.10 px")
ax.set_ylabel("Position Error [px]")
ax.set_title("SR Impact Summary")
ax.legend(loc="best")
save_fig(fig, "sr_impact_summary.png")

# %% [markdown]
# > **图表说明**: 柱状图把旋转模型 RMS 和 repeat p95 与 0.05 px、0.10 px 两条 SR 位移误差阈值对比。
# >
# > **数据分布**: 两个误差指标都高于 0.10 px，明显超过 2× SR 的实用阈值；
# > 更不用说 4× SR 常见目标 0.05 px。
# >
# > **核心发现**: 当前位移证据不能支撑高倍率 SR。
# > 在位移标定链路重新验证前，继续做 SR 重建只会把模型误差带入结果。

# %%
y2_mag = float(direction_df[(direction_df["scan_axis"] == "y") & (direction_df["delta_um"] == 2.0)]["median_magnitude_px"].median())
y4_mag = float(direction_df[(direction_df["scan_axis"] == "y") & (direction_df["delta_um"] == 4.0)]["median_magnitude_px"].median())
x_angle_min = float(direction_df[direction_df["scan_axis"] == "x"]["median_angle_deg_y_up"].min())
x_angle_max = float(direction_df[direction_df["scan_axis"] == "x"]["median_angle_deg_y_up"].max())
y_angle_min = float(direction_df[direction_df["scan_axis"] == "y"]["median_angle_deg_y_up"].min())
y_angle_max = float(direction_df[direction_df["scan_axis"] == "y"]["median_angle_deg_y_up"].max())

report = f"""# EP02 Calibration Report

## Scope

This report uses same-session adjacent TXT thermal frames from EP01 and estimates image displacement with NCC plus quadratic peak fitting.
The calibration set is restricted to EP01's main acquisition session (`session={main_session}`), not filename-sorted pseudo-sessions.

## Main Result

The current NCC measurements do **not** validate the previous theta value of {REFERENCE_THETA_DEG:.1f} deg.

| Metric | Value |
|---|---:|
| Main-session adjacent frame pairs | {len(measurements)} |
| NCC fit-ok rate | {quality['fit_ok_rate']:.3f} |
| Median NCC peak | {quality['median_peak_ncc']:.5f} |
| Image-row theta fit | {fit_image_row['theta_deg']:.3f} deg |
| Y-up diagnostic theta fit | {fit_y_up['theta_deg']:.3f} deg |
| Y-up 95% bootstrap CI | [{ci_bounds[0]:.3f}, {ci_bounds[1]:.3f}] deg |
| Reference theta inside CI | {reference_in_ci} |
| Rotation-model RMS residual | {model_error_px:.4f} px |
| Valid repeatability pairs | {repeatability_summary['n_valid_repeat_pairs']} / {repeatability_summary['n_repeat_pairs_main_session']} main-session pairs |
| Repeatability median error (valid only) | {repeat_median_text} |
| Repeatability p95 error (valid only) | {repeat_p95_text} |
| Linearity projection R2 | {linearity_projection['r2']:.4f} |

## Interpretation

NCC produces high correlation peaks, but the measured displacement field is inconsistent with a single rigid rotation and a fixed {PIXEL_SIZE_UM:.1f} um/px scale. The clearest remaining failure mode is the Y-scan: 2 um and 4 um command steps do not scale linearly in the measured displacement.

Therefore EP02 should be treated as a failed independent validation of theta rather than a replacement calibration. The global `configs/stage_calibration.json` should remain unchanged until the displacement measurement method is improved or independently checked.

## Acquisition-Order and Motion Direction Check

EP01 found that the earlier 13-session interpretation was a filename-sorting artifact. Using acquisition time collapses the data into a short warm-up/repeat segment and one 255-frame main scan. EP02 now uses only that main scan.

Within the main scan, X-scan median directions span {x_angle_min:.2f} deg to {x_angle_max:.2f} deg in y-up coordinates, and Y-scan median directions span {y_angle_min:.2f} deg to {y_angle_max:.2f} deg. The larger anomaly is magnitude: Y-scan 2 um steps have median displacement {y2_mag:.4f} px, while Y-scan 4 um steps have median displacement {y4_mag:.4f} px. A 4 um command should not produce a smaller displacement than a 2 um command.

This means the low-temperature jump frames were not the main cause of the Y-scan failure. The next check should focus on Y-scan frame-pair construction, scan reversal/backlash, axis sign conventions, or NCC bias under thermal-field changes.

## SR Impact

The fitted model RMS residual is {model_error_px:.4f} px. This exceeds the 0.1 px practical threshold used for 2x SR feasibility and is above the 0.05 px target for 4x SR. Repeatability is also weakly constrained because only {repeatability_summary['n_valid_repeat_pairs']} main-session repeat pairs avoid boundary peaks. Current displacement evidence is insufficient for reliable SR reconstruction.

## Output Files

- `frame_pairs.csv`
- `displacement_measurements.csv`
- `motion_direction_diagnostic.csv`
- `theta_estimate.json`
- `repeatability.csv`
- `linearity.csv`
- `displacement_vector_field.png`
- `dx_dy_vs_coordinates.png`
- `motion_direction_diagnostic.png`
- `theta_bootstrap.png`
- `theta_residuals.png`
- `repeatability_boxplot.png`
- `linearity_regression.png`
- `sr_impact_summary.png`
"""
(REPORT_DIR / "calibration_report.md").write_text(report, encoding="utf-8")
print(f"SR status: {sr_status}")
print("Saved: reports/ep02_displacement_calibration/calibration_report.md")
