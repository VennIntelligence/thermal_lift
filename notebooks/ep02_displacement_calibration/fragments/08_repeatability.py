# %% [markdown]
# ## 7. 位移台重复定位精度
#
# 使用配置中列出的 repeat 坐标，同一坐标不同 R 的帧两两互相关。
# 当前主报告只保留同属主扫描 session 的 repeat pair；跨温度段 repeat 仅作为风险背景，不进入主统计。

# %%
repeat_coords = [tuple(coord) for coord in coord_config["repeat_coordinates"]]
repeat_pairs_all = build_repeat_pairs(df_audit, repeat_coords=repeat_coords)
repeat_pairs = repeat_pairs_all[
    repeat_pairs_all["session_a"].eq(main_session)
    & repeat_pairs_all["session_b"].eq(main_session)
].copy()
repeat_measurements = measure_frame_pairs(
    repeat_pairs,
    DATA_DIR,
    roi_size=ROI_SIZE,
    search_radius=SEARCH_RADIUS,
    method="ncc",
)
repeat_measurements["error_px"] = np.hypot(
    repeat_measurements["dx_px"],
    repeat_measurements["dy_px"],
)
repeat_measurements["error_um"] = repeat_measurements["error_px"] * PIXEL_SIZE_UM
repeat_measurements.to_csv(OUTPUT_DIR / "repeatability.csv", index=False)
repeat_valid = repeat_measurements[
    repeat_measurements["fit_ok"] & ~repeat_measurements["edge_peak"]
].copy()

valid_errors_px = repeat_valid["error_px"]
valid_errors_um = repeat_valid["error_um"]
repeatability_summary = {
    "n_repeat_pairs_all": int(len(repeat_pairs_all)),
    "n_repeat_pairs_main_session": int(len(repeat_measurements)),
    "n_valid_repeat_pairs": int(len(repeat_valid)),
    "n_invalid_repeat_pairs": int(len(repeat_measurements) - len(repeat_valid)),
    "median_error_px_valid": float(valid_errors_px.median()) if len(repeat_valid) else float("nan"),
    "std_error_px_valid": float(valid_errors_px.std(ddof=1)) if len(repeat_valid) > 1 else float("nan"),
    "p95_error_px_valid": float(valid_errors_px.quantile(0.95)) if len(repeat_valid) else float("nan"),
    "median_error_um_valid": float(valid_errors_um.median()) if len(repeat_valid) else float("nan"),
    "p95_error_um_valid": float(valid_errors_um.quantile(0.95)) if len(repeat_valid) else float("nan"),
}
print(json.dumps(repeatability_summary, indent=2))
print("Saved: output/ep02_displacement_calibration/repeatability.csv")

# %% [markdown]
# > **数据说明**: 这里用同一 (X,Y) 坐标、同属主扫描 session 的不同 repeat 帧做互相关。
# > 跨温度段 repeat pair 已排除，只把 `fit_ok=True` 且非边界峰的帧对纳入有效统计。
# >
# > **数据分布**: 主扫描 session 内 repeat pair 很少，因此该统计约束很弱。
# > 这比旧版跨 session repeatability 更干净，但样本数更少。
# >
# > **核心发现**: repeatability 仍不能被解释成强机械指标。
# > 如果需要定量重复定位，应专门采集同温度状态、同坐标、多 repeat 的控制数据。

# %%
fig, ax = make_figure("single_col", height=3.0)
labels = []
data = []
for (x, y), subset in repeat_valid.groupby(["X_a", "Y_a"]):
    labels.append(f"({x},{y})")
    data.append(subset["error_um"].to_numpy())
if data:
    ax.boxplot(data, tick_labels=labels, showfliers=True)
else:
    ax.text(0.5, 0.5, "No valid same-session repeat pairs", ha="center", va="center", transform=ax.transAxes)
ax.set_xlabel(r"Repeat coordinate [$\mu$m]")
ax.set_ylabel(r"Repeat displacement error [$\mu$m]")
ax.set_title("Repeatability from Valid Same-Coordinate Frame Pairs")
save_fig(fig, "repeatability_boxplot.png")

# %% [markdown]
# > **图表说明**: 箱线图按 repeat 坐标展示有效同坐标帧对的位移误差，单位换算为 µm。
# >
# > **数据分布**: 可画出的坐标很少，说明数据集的重复测量覆盖不足；
# > 个别有效帧对误差已经达到数 µm。
# >
# > **核心发现**: 这张图的重点不是证明位移台一定差，而是证明当前 repeat 数据不足以给出强保证。
# > 后续如果要定量重复定位，应专门采集同 session、同坐标、多 repeat 的控制数据。
