# %% [markdown]
# ## 5. 从磁盘消费数据包：总览图与指标摘要
#
# 这一节模拟后续算法或报告消费者的视角：不使用上游 cell 的内存变量，而是从 `manifest.csv` 指向的磁盘文件重新读取数据。这样可以验证 manifest/metadata 足够支撑独立复查。

# %%
hr_temperature_disk = np.load(DEMO_DIR / "hr_temperature_2x.npy")
hr_mask_disk = np.load(DEMO_DIR / "hr_mask_2x.npy")
hr_edge_disk = np.load(DEMO_DIR / "hr_edge_map_2x.npy")
lr_raw_disk = np.load(DEMO_DIR / "lr_burst_raw.npy")
lr_hp_disk = np.load(DEMO_DIR / "lr_burst_highpass.npy")
shifts_disk = np.load(DEMO_DIR / "shifts.npy")
lr_raw_disk_vis = _crop_lr_stack(lr_raw_disk, EDGE_VIS_MARGIN_LR_PX)
lr_hp_disk_vis = _crop_lr_stack(lr_hp_disk, EDGE_VIS_MARGIN_LR_PX)

fig, axes = plt.subplots(2, 3, figsize=(8.0, 5.4))
axes = axes.ravel()
show_image(
    axes[0],
    hr_temperature_disk,
    title="HR temperature truth",
    cmap=COLORMAPS["temperature"],
    colorbar_label="Temperature [$^\\circ$C]",
    robust=True,
)
show_image(
    axes[1],
    hr_mask_disk,
    title="HR structure mask",
    cmap="gray",
    colorbar_label="Mask",
    vmin=0,
    vmax=1,
)
show_image(
    axes[2],
    hr_edge_disk,
    title="HR contour proxy",
    cmap=COLORMAPS["coverage"],
    colorbar_label="Edge",
    vmin=0,
    vmax=max(1.0, float(hr_edge_disk.max())),
)
show_image(
    axes[3],
    lr_raw_disk_vis.mean(axis=0),
    title="LR raw mean (interior)",
    cmap=COLORMAPS["temperature"],
    colorbar_label="Temperature [$^\\circ$C]",
    robust=True,
)
show_image(
    axes[4],
    lr_hp_disk_vis.std(axis=0),
    title="LR highpass std (interior)",
    cmap="magma",
    colorbar_label="Std response [$^\\circ$C]",
    robust=True,
)
sc = axes[5].scatter(
    shifts_disk[:, 0],
    shifts_disk[:, 1],
    c=np.arange(len(shifts_disk)),
    cmap=COLORMAPS["coverage"],
    s=36,
    edgecolors="black",
    linewidths=0.25,
)
axes[5].set_title("Phase coverage")
axes[5].set_xlabel("dx [LR px]")
axes[5].set_ylabel("dy [LR px]")
axes[5].invert_yaxis()
axes[5].set_aspect("equal", adjustable="box")
axes[5].grid(True, alpha=0.25, linewidth=0.5)
format_colorbar(fig.colorbar(sc, ax=axes[5], fraction=0.046, pad=0.03), "Frame index")
for label, ax in zip(["a", "b", "c", "d", "e", "f"], axes, strict=True):
    add_panel_label(ax, label)
save_fig(fig, "demo_dataset_overview.png")

# %% [markdown]
# > **图表说明**: 这张总览图完全从磁盘重新读取 demo 数据，展示 HR temperature、HR mask、HR edge proxy、LR raw mean、LR highpass std 和 phase coverage。
# >
# > **怎么看**: HR 三个面板用于检查真值层；LR 两个面板用于检查算法输入层；phase plot 用 LR pixel 显示亚像素覆盖，不是微米坐标图。
# >
# > **异常是否正常**: LR 面板使用展示裁剪，以免 outside-FOV 边界伪影支配色标。Highpass std 越亮说明多帧结构响应变化越强，但噪声、边界和假结构也会使它变大，因此不能单独证明 SR 成功。
# >
# > **核心发现**: 数据包可以独立读回，并同时保留真值、观测和位移三类证据。

# %%
try:
    from tcforge import evaluate as tc_eval

    demo_eval_summary = tc_eval.summarize_scene(DEMO_DIR)
    EVAL_SOURCE = "tcforge.evaluate.summarize_scene"
except Exception as exc:
    shift_norms = np.linalg.norm(shifts_disk, axis=1)
    demo_eval_summary = {
        "scene_id": DEMO_CONFIG.scene_id,
        "n_frames": int(lr_raw_disk.shape[0]),
        "lr_rows": int(lr_raw_disk.shape[1]),
        "lr_cols": int(lr_raw_disk.shape[2]),
        "hr_rows": int(hr_temperature_disk.shape[0]),
        "hr_cols": int(hr_temperature_disk.shape[1]),
        "mask_coverage": float(np.mean(hr_mask_disk > 0)),
        "edge_density": float(np.mean(hr_edge_disk > 0)),
        "lr_raw_mean_c": float(np.mean(lr_raw_disk)),
        "lr_raw_std_c": float(np.std(lr_raw_disk)),
        "lr_highpass_abs_p95_c": float(np.percentile(np.abs(lr_hp_disk), 95)),
        "shift_norm_mean_px": float(np.mean(shift_norms)),
        "shift_norm_max_px": float(np.max(shift_norms)),
    }
    EVAL_SOURCE = f"notebook fallback summary ({exc.__class__.__name__})"

metric_explain = {
    "n_frames": "LR burst length",
    "mask_coverage": "fraction of HR pixels inside the synthetic chip structure",
    "edge_density": "fraction of HR pixels marked by the contour proxy",
    "lr_raw_mean_c": "ordinary LR temperature mean",
    "lr_raw_std_c": "ordinary LR temperature standard deviation",
    "lr_highpass_abs_p95_c": "95th percentile of absolute highpass response",
    "shift_norm_mean_px": "mean shift magnitude in LR pixels",
    "shift_norm_max_px": "max shift magnitude in LR pixels",
}
demo_metric_rows = []
for key, explanation in metric_explain.items():
    value = demo_eval_summary.get(key, np.nan)
    demo_metric_rows.append({"metric": key, "value": value, "interpretation": explanation})
demo_metrics = pd.DataFrame(demo_metric_rows)
display(compact_numeric_table(demo_metrics, ["metric", "value", "interpretation"]))
print(f"Evaluation summary source: {EVAL_SOURCE}")

# %% [markdown]
# > **数据说明**: 这张表展示 scene-level evaluate 摘要。它来自 `tcforge.evaluate.summarize_scene()`，若包不可导入才使用 notebook fallback。
# >
# > **怎么看**: `mask_coverage` 和 `edge_density` 是真值结构密度；`lr_highpass_abs_p95_c` 是结构响应尺度；`shift_norm_*` 描述 synthetic 位移覆盖。通常这些指标用于数据健康检查和 regression 分组，不是 SR 成功指标。
# >
# > **异常是否正常**: `lr_highpass_abs_p95_c` 可能受 outside-FOV 边界影响，尤其是很小的 demo frame；判断结构响应时应同时看上面的内部裁剪图。
# >
# > **核心发现**: Notebook 现在把 evaluate 层的数值摘要显式展示出来，不再只保存 CSV/JSON 后让读者自己找。

# %%
fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
row_idx = int(hr_temperature_disk.shape[0] * 0.5)
profile_slice = slice(int(hr_temperature_disk.shape[1] * 0.20), int(hr_temperature_disk.shape[1] * 0.82))
profile_x_hr = np.arange(profile_slice.start, profile_slice.stop)
mask_profile = hr_mask_disk[row_idx, profile_slice].astype(float)
temp_profile = hr_temperature_disk[row_idx, profile_slice]
temp_norm = (temp_profile - temp_profile.min()) / max(float(temp_profile.max() - temp_profile.min()), 1e-8)

axes[0].plot(profile_x_hr, mask_profile, color="#222222", linestyle="--", linewidth=1.4, label="HR mask")
axes[0].plot(profile_x_hr, temp_norm, color=METHOD_COLORS["accent_1"], linewidth=1.4, label="HR temperature (normalized)")
axes[0].set_title("HR profile across chip structure")
axes[0].set_xlabel("HR column [px]")
axes[0].set_ylabel("Normalized value")
axes[0].grid(axis="y", alpha=0.25, linewidth=0.5)
axes[0].legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2)

lr_row_idx = int(lr_raw_disk_vis.shape[1] * 0.5)
lr_profile_x = np.arange(lr_raw_disk_vis.shape[2])
raw_profile = lr_raw_disk_vis[frame_idx, lr_row_idx]
hp_profile = lr_hp_disk_vis[frame_idx, lr_row_idx]
raw_norm = (raw_profile - raw_profile.min()) / max(float(raw_profile.max() - raw_profile.min()), 1e-8)
hp_norm = hp_profile / max(float(np.max(np.abs(hp_profile))), 1e-8)
axes[1].plot(lr_profile_x, raw_norm, color=METHOD_COLORS["primary"], linewidth=1.4, label="LR raw (normalized)")
axes[1].plot(lr_profile_x, hp_norm, color=METHOD_COLORS["secondary"], linewidth=1.4, label="LR highpass (signed)")
axes[1].axhline(0.0, color="#666666", linestyle="--", linewidth=0.8)
axes[1].set_title("LR observation profile")
axes[1].set_xlabel("Interior LR column [px]")
axes[1].set_ylabel("Normalized / signed value")
axes[1].grid(axis="y", alpha=0.25, linewidth=0.5)
axes[1].legend(loc="lower center", bbox_to_anchor=(0.5, -0.34), ncol=2)
for label, ax in zip(["a", "b"], axes, strict=True):
    add_panel_label(ax, label)
save_fig(fig, "demo_profiles_generation_vs_observation.png")

# %% [markdown]
# > **图表说明**: 左图用一条 HR 剖面对比结构 mask 与温度 truth；右图用一条 LR 剖面对比 raw 温度观测和 signed highpass 响应。
# >
# > **怎么看**: HR mask 是结构阶跃，HR temperature 是温度目标；LR raw 是经过 forward 后的观测，LR highpass 围绕零上下波动，突出边缘/纹理而不是绝对温度。
# >
# > **异常是否正常**: Highpass 曲线穿过 0 且有正负值是正常的；它说明相对局部背景的偏差，不能读成真实温度正负。
# >
# > **核心发现**: 这张剖面图把“生成真值”和“算法观测”放在同一页解释，避免只看 heatmap 时误把 highpass 当普通温度图。

# %%
regression_rows = []
smoke_report_path = REGRESSION_DEMO_DIR / "smoke_test_report.json"
eval_csv_path = REGRESSION_EVAL_DIR / "evaluation_summary.csv"
if smoke_report_path.exists():
    smoke_report = json.loads(smoke_report_path.read_text(encoding="utf-8"))
    regression_rows.append(
        {
            "artifact": "smoke_test_report.json",
            "path": relative(smoke_report_path),
            "status": smoke_report.get("status", "unknown"),
            "detail": f"failures={len(smoke_report.get('failures', []))}, warnings={len(smoke_report.get('warnings', []))}",
        }
    )
else:
    regression_rows.append({"artifact": "smoke_test_report.json", "path": relative(smoke_report_path), "status": "missing", "detail": "not generated in this run"})
if eval_csv_path.exists():
    cli_eval_df = pd.read_csv(eval_csv_path)
    regression_rows.append(
        {
            "artifact": "evaluation_summary.csv",
            "path": relative(eval_csv_path),
            "status": "present",
            "detail": f"{len(cli_eval_df)} scene(s), columns={len(cli_eval_df.columns)}",
        }
    )
else:
    cli_eval_df = pd.DataFrame()
    regression_rows.append({"artifact": "evaluation_summary.csv", "path": relative(eval_csv_path), "status": "missing", "detail": "not generated in this run"})

display(pd.DataFrame(regression_rows))
if not cli_eval_df.empty:
    preferred_cols = [
        "scene_id",
        "difficulty",
        "n_frames",
        "lr_rows",
        "lr_cols",
        "mask_coverage",
        "edge_density",
        "lr_highpass_abs_p95_c",
        "shift_norm_max_px",
        "highpass_reference_max_abs_diff_c",
        "highpass_reference_allclose",
    ]
    display(compact_numeric_table(cli_eval_df, [col for col in preferred_cols if col in cli_eval_df.columns]))

# %% [markdown]
# > **数据说明**: 这张表读取已有 CLI regression demo 的 smoke/evaluate 产物（如果本机已生成）。这些产物位于 ignored `output/`，不会被 Git 跟踪。
# >
# > **怎么看**: `smoke_test_report.json` 的 `status=pass` 表示 P0 结构门控通过；`evaluation_summary.csv` 展示 CLI 级 scene 摘要，包括高通独立复算差异。
# >
# > **异常是否正常**: 如果 regression demo 文件缺失，不代表 notebook demo 失败，只说明当前机器还没有运行对应 CLI。正式验收应重新执行 smoke/evaluate CLI。
# >
# > **核心发现**: Notebook 现在会展示后台 CLI 指标与检测产物，避免“工作已经做了但报告没体现”的问题。
