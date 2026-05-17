# %% [markdown]
# ## 2. 帧对选择
#
# **策略**: 主扫描 session、`R=0`、只改变一个坐标轴、相邻步长不超过 4 µm。  
# 早期低温/补采帧和 30 µm 这类大间隔不进入主互相关标定。

# %%
pairs_x = build_frame_pairs(df_calib, axis="x", r_value=0, max_delta_um=MAX_PAIR_DELTA_UM)
pairs_y = build_frame_pairs(df_calib, axis="y", r_value=0, max_delta_um=MAX_PAIR_DELTA_UM)
frame_pairs = pd.concat([pairs_x, pairs_y], ignore_index=True)
frame_pairs.to_csv(OUTPUT_DIR / "frame_pairs.csv", index=False)

pair_summary = (
    frame_pairs.groupby(["scan_axis", "delta_um"])
    .size()
    .rename("n_pairs")
    .reset_index()
)
print(f"Frame pairs: {len(frame_pairs)}")
print(pair_summary.to_string(index=False))
print("Saved: output/ep02_displacement_calibration/frame_pairs.csv")

# %% [markdown]
# > **数据说明**: 这张帧对表只保留主扫描 session 内的 `R=0`、单轴移动、步长不超过 4 µm 的相邻帧对。
# > `pair_summary` 按扫描轴和命令步长统计可用于互相关的样本量。
# >
# > **数据分布**: 帧对来自同一个主温度带，覆盖 X-scan 与 Y-scan 的 2 µm/4 µm 步长。
# > 早期低温/补采帧已排除，不再影响主 θ 拟合。
# >
# > **核心发现**: 这个筛选修正了旧版 EP02 的 session 排序问题。
# > 后续剩余异常若仍存在，应优先解释为位移测量/扫描方向问题，而不是温度跳队污染。

# %%
pair_breakdown = frame_pairs.groupby(["scan_axis", "delta_um"]).size().reset_index(name="n_pairs")
fig, ax = make_figure("single_col", height=2.8)
labels = [f"{row.scan_axis.upper()} {row.delta_um:.0f}µm" for _, row in pair_breakdown.iterrows()]
colors = [METHOD_COLOR_LIST[0] if a == "x" else METHOD_COLOR_LIST[1]
          for a in pair_breakdown["scan_axis"]]
ax.bar(labels, pair_breakdown["n_pairs"], color=colors)
for i, v in enumerate(pair_breakdown["n_pairs"]):
    ax.text(i, v + 0.5, str(v), ha="center", va="bottom", fontsize=8)
ax.set_xlabel("Scan Axis × Step Size")
ax.set_ylabel("Frame Pair Count")
ax.set_title(f"Valid Calibration Pairs (session {frame_pairs['session'].iloc[0]}, n={len(frame_pairs)})")
save_fig(fig, "frame_pair_session_counts.png")

# %% [markdown]
# > **图表说明**: 柱状图按扫描轴 (X/Y) 和命令步长 (2/4 µm) 展示有效互相关帧对的数量分布。
# >
# > **数据分布**: 全部帧对均来自主扫描 session，按扫描轴和步长均匀分布。
# >
# > **核心发现**: 帧对筛选已完全排除早期低温/补采帧。
# > 后续位移诊断应聚焦于 X/Y 扫描轴和 2/4 µm 步长是否满足线性和旋转一致性。
