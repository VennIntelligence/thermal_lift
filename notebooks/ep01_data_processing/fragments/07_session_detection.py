# %% [markdown]
# ## 3.2 采集顺序、Session 划分与主 session 覆盖
#
# Session 必须按真实采集顺序检测，不能按重命名后的文件名字母序检测。
# 文件名排序会把 `R=1/2` repeat 插到主扫描中间，制造假的温度跳变。
#
# **SR 关注点**: 后续微扫描 SR 必须继承 `acquisition_order`、`session` 和 `is_main_session`，
# 并默认只用主 session=255 帧进行对齐和重建。

# %%
df_sorted = model.df_acquisition.copy()
df_filename = model.df_filename.copy()
session_ids = model.session_ids
break_indices = model.break_indices
threshold = model.threshold
session_summary = model.session_summary.copy()
main_session = model.main_session
n_sessions = int(session_ids.max()) + 1
filename_session_count = int(model.filename_session_ids.max()) + 1

print(f"文件名排序会检测到: {filename_session_count} sessions")
print(f"采集顺序检测阈值: {threshold:.4f} °C")
print(f"采集顺序检测到: {len(break_indices)} 处断点 → {n_sessions} sessions")
print(f"主扫描 session: {main_session} ({int(session_summary.loc[session_summary['session'].eq(main_session), 'n_frames'].iat[0])} 帧)")
session_summary

# %% [markdown]
# > **数据说明**: 这里同时计算文件名排序和采集顺序排序下的 session 数，并列出采集顺序下每个温度段的帧数、首尾文件和温度范围。
# > “session” 在这里指连续采集、温度状态相近的一段数据，不是文件名里的 repeat。
# >
# > **怎么读**: `文件名排序会检测到` 是反例，用来说明按 `X_Y_R` 字符串排序会制造多少假温度段。
# > `采集顺序检测到` 才是本项目采用的 session 数；表格中的 `n_frames` 最大的一段就是主扫描候选。
# > 读表时重点看每个 session 的帧数、采集序号范围和均温范围是否互相一致。
# >
# > **数据分布**: 文件名排序会得到 13 个 session；采集顺序只得到 3 个温度段。
# > 最大温度段包含 255 帧，均温约 23.3°C，是后续 SR 应使用的主扫描数据。
# >
# > **正常/异常理解**: 如果文件名排序和采集顺序给出不同 session 数，应优先相信采集顺序。
# > 文件名在本项目里编码坐标，不编码拍摄时间；把它当时间轴会把 repeat 和补采帧插到错误位置。
# >
# > **核心发现**: 13 sessions 是排序伪影，不应写入后续分析。
# > 早期低温/补采帧应被记录，但不应作为主扫描的一部分参与 SR 对齐或重建。

# %%
plot_order_comparison(
    model,
    save_path="order_comparison.png",
    save_fn=save_fig,
)

# %% [markdown]
# > **图表说明**: 左图按重命名后的文件名字母序排列逐帧均温，右图按文件修改时间得到的真实采集顺序排列逐帧均温。
# > 颜色/断点来自同一套温度跳变检测逻辑，因此两图差异主要来自排序方式，而不是检测算法变化。
# >
# > **怎么读**: 如果左图跳变很多而右图只有少数连续温度段，说明文件名排序破坏了时间结构。
# > 右图中长而稳定的连续段，才适合作为后续帧间对齐和 SR 的输入池。
# >
# > **数据分布**: 文件名排序把少量 repeat/预热帧插入主扫描中间，导致温度曲线出现反复跳变；
# > 采集顺序则恢复为少量早期温度段 + 255 帧主扫描温度段。
# >
# > **正常/异常理解**: 正常的 step-and-shoot 扫描应在真实采集顺序下呈现连续 raster 路径和相对平滑的温度趋势。
# > 如果右图仍然频繁大跳，才需要怀疑采集中断、温控漂移或文件 mtime 不可信。
# >
# > **核心发现**: 后续 SR 的时间轴必须使用 `acquisition_order`。
# > 文件名只提供坐标和 stage command prior，不能替代采集时序。

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
# > 这张图把 session 表格里的断点放回时间线上，便于检查分割是否符合肉眼观察。
# >
# > **怎么读**: 横轴越往右表示采集越晚；同一颜色区域表示算法认为可以归为同一温度段。
# > 竖线附近如果同时出现均温台阶变化，说明 session 边界有数据支撑。
# >
# > **数据分布**: 前 8 帧包含低温和补采/预热状态，之后进入长时间稳定的主扫描温度带。
# > 主扫描内部温度只缓慢漂移，没有文件名排序下反复出现的 3–4°C 跳变。
# >
# > **正常/异常理解**: 少量早期非主扫描帧是可以接受的，只要它们被清楚标记并从主重建输入中排除。
# > 异常情况是主 session 内部被分割成多个大温度台阶，那会削弱同一 session 内重建假设。
# >
# > **核心发现**: 后续 Episode 应继承 `session` 字段，并默认只使用主扫描 session 做对齐和 SR。
# > `acquisition_order` 是本项目后续所有时序分析的排序依据。

# %%
plot_session_coverage_heatmaps(
    df_sorted,
    sorted(VALID_COORDS),
    main_session,
    save_path="session_coordinate_coverage.png",
    save_fn=save_fig,
)

# %% [markdown]
# > **图表说明**: 两张 16×16 坐标覆盖热力图分别统计主 session 和非主 session 在每个 `(X,Y)` 坐标上的帧数。
# > 它把“哪些帧属于主扫描”转化为空间覆盖问题：主 session 是否真的覆盖了微扫描网格。
# >
# > **怎么读**: 左/主 session 图应覆盖大部分坐标；右/非主 session 图如果只在少数坐标有值，
# > 说明非主帧只是额外采集状态，不是另一套完整扫描。
# >
# > **数据分布**: 主 session 包含 255 帧，覆盖全部 253 个实际存在的坐标；
# > 非主 session 只包含开头 8 帧，集中在 Y=0 的少量坐标上。
# >
# > **正常/异常理解**: 对后续 2x contour-level POC，正常情况是主 session 同时具备帧数优势和二维坐标覆盖。
# > 如果主 session 只覆盖一条线或局部区域，就不能作为二维 SR 的默认输入。
# >
# > **核心发现**: 主 session 不只是帧数最多，也覆盖了 SR POC 所需的二维扫描网格。
# > 非主 session 不能补足额外结构信息，混入反而会引入跨温度段偏差。

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
# > **怎么读**: `R=0 采集 Y 顺序` 应该是一串从 0 到 40 的递增 Y 坐标；
# > `行内 X 顺序不匹配行数` 越接近 0，说明真实采集顺序越符合 raster 扫描假设。
# > 生成的 `acquisition_order_audit.csv` 可用于逐帧追查具体文件。
# >
# > **数据分布**: R=0 的 Y 顺序为 `[0, 2, 4, ..., 40]`，行内 X 顺序不匹配数为 0。
# > 这说明重命名后的 `(X,Y)` 与真实采集顺序互相一致。
# >
# > **正常/异常理解**: 正常 raster 扫描应表现为行内 X 递增、行间 Y 递增。
# > 如果大量行内顺序不匹配，可能是命名解析、mtime 顺序或采集路径记录有问题，需要在进入 EP02/EP04 前解决。
# >
# > **核心发现**: 目前没有证据支持“重命名把坐标系统性搞错”。
# > 真正需要固化的是 session 检测和下游帧对选择必须使用采集顺序，而不是文件名字母序。
