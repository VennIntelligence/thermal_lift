# %% [markdown]
# ## 3.3 Session 自动检测与采集顺序核查
#
# Session 必须按真实采集顺序检测，不能按重命名后的文件名字母序检测。
# 文件名排序会把 `R=1/2` repeat 插到主扫描中间，制造假的温度跳变。
#
# **结论**: 按真实采集顺序检测到 3 个温度段；其中最大的一段是 255 帧主扫描 session。

# %%
filename_session_ids, filename_break_indices, filename_threshold = detect_sessions(df_filename)
session_ids, break_indices, threshold = detect_sessions(df_acquisition)
n_sessions = int(session_ids.max()) + 1

df_acquisition["session"] = session_ids
session_lookup = df_acquisition[["file", "session"]]
df_filename = df_filename.drop(columns=["session"], errors="ignore").merge(session_lookup, on="file", how="left")
df_sorted = df_acquisition.copy()
df_sorted["session_source"] = "acquisition_order_mtime"

session_summary = (
    df_sorted.groupby("session")
    .agg(
        n_frames=("file", "count"),
        first_file=("file", "first"),
        last_file=("file", "last"),
        mean_temp=("T_mean", "mean"),
        min_temp=("T_mean", "min"),
        max_temp=("T_mean", "max"),
    )
    .reset_index()
)
main_session = int(session_summary.loc[session_summary["n_frames"].idxmax(), "session"])
df_sorted["is_main_session"] = df_sorted["session"].eq(main_session)

print(f"文件名排序会检测到: {int(filename_session_ids.max()) + 1} sessions")
print(f"采集顺序检测阈值: {threshold:.4f} °C")
print(f"采集顺序检测到: {len(break_indices)} 处断点 → {n_sessions} sessions")
print(f"主扫描 session: {main_session} ({int(session_summary.loc[session_summary['session'].eq(main_session), 'n_frames'].iat[0])} 帧)")
session_summary

# %% [markdown]
# > **数据说明**: 这里同时计算文件名排序和采集顺序排序下的 session 数，并列出采集顺序下每个温度段的帧数、首尾文件和温度范围。
# >
# > **数据分布**: 文件名排序会得到 13 个 session；采集顺序只得到 3 个温度段。
# > 最大温度段包含 255 帧，均温约 23.3°C，是后续位移标定和 SR 应使用的主扫描数据。
# >
# > **核心发现**: 13 sessions 是排序伪影，不应写入后续分析。
# > 早期低温/补采帧应被记录，但不应作为主扫描的一部分参与位移标定或 SR。

# %%
plot_sessions(
    df_sorted,
    session_ids,
    break_indices,
    save_path="session_detection.png",
    save_fn=save_fig,
)

# %% [markdown]
# > **图表说明**: 逐帧均温时间线按真实采集顺序排列，并叠加 session 分割结果。
# > 不同颜色区分温度段，竖线标注检测到的温度跳变。
# >
# > **数据分布**: 前 8 帧包含低温和补采/预热状态，之后进入长时间稳定的主扫描温度带。
# > 主扫描内部温度只缓慢漂移，没有文件名排序下反复出现的 3–4°C 跳变。
# >
# > **核心发现**: 后续 EP02 应继承 `session` 字段并只使用主扫描 session 做旋转角验证。
# > `acquisition_order` 是本项目后续所有时序分析的排序依据。

# %%
valid_coord_list = sorted(VALID_COORDS)
r0_acq = df_sorted[df_sorted["R"] == 0].sort_values("acquisition_order")
missing_coords = {tuple(coord) for coord in coord_config.get("known_missing_r0", [])}
expected_by_y = {
    y: [x for x in valid_coord_list if (x, y) not in missing_coords]
    for y in valid_coord_list
}
row_order_mismatches = []
for y, group in r0_acq.groupby("Y", sort=True):
    xs = group.sort_values("acquisition_order")["X"].astype(int).tolist()
    if xs != expected_by_y[int(y)]:
        row_order_mismatches.append({"Y": int(y), "observed_X": xs, "expected_X": expected_by_y[int(y)]})

audit_cols = ["file", "X", "Y", "R", "T_mean", "mtime", "acquisition_order", "session"]
if "source_file" in r0_acq.columns:
    audit_cols.insert(1, "source_file")
acquisition_order_audit = r0_acq[audit_cols].copy()
acquisition_order_audit.to_csv(OUTPUT_DIR / "acquisition_order_audit.csv", index=False)

print(f"R=0 采集 Y 顺序: {r0_acq.groupby('Y')['acquisition_order'].min().sort_values().index.astype(int).tolist()}")
print(f"R=0 行内 X 顺序不匹配行数: {len(row_order_mismatches)}")
print("Saved: output/ep01_data_processing/acquisition_order_audit.csv")

# %% [markdown]
# > **数据说明**: 这一段专门核查 R=0 主网格在真实采集顺序下的扫描结构。
# > 期望结构是 Y 从 0 到 40 递增，每条 Y 扫描线内 X 从 0 到 40 递增，跳过已知缺失坐标。
# >
# > **数据分布**: R=0 的 Y 顺序为 `[0, 2, 4, ..., 40]`，行内 X 顺序不匹配数为 0。
# > 这说明重命名后的 `(X,Y)` 与真实采集顺序互相一致。
# >
# > **核心发现**: 目前没有证据支持“重命名把坐标系统性搞错”。
# > 真正需要修正的是 session 检测和下游帧对选择必须使用采集顺序，而不是文件名字母序。
