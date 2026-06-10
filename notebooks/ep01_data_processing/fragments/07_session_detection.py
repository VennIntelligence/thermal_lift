# %% [markdown]
# ## 3.2 采集顺序、Session 划分与主 session 覆盖

# %%
model = cache.model
session_summary = cache.session_summary
main_session = int(model.main_session)
n_sessions = int(model.session_ids.max()) + 1
filename_session_count = int(model.filename_session_ids.max()) + 1

print(f"文件名排序会检测到: {filename_session_count} sessions")
print(f"采集顺序检测阈值: {model.threshold:.4f} °C")
print(f"采集顺序检测到: {len(model.break_indices)} 处断点 → {n_sessions} sessions")
print(
    f"主扫描 session: {main_session} ("
    f"{int(session_summary.loc[session_summary['session'].eq(main_session), 'n_frames'].iat[0])} 帧)"
)
display(session_summary)

# %% [markdown]
# > **Session 划分结论**：利用自适应温度变化率检测，真实采集顺序成功识别出 3 个物理 Session。其中 Session 2 包含 255 帧，为默认的黄金扫描段。若使用文件名排序，则会因为乱序在时序上制造出 13 个虚假的温度跳变段（伪 Session）。


# %%
show_fig("order_comparison.png")

# %% [markdown]
# Figure 4: Acquisition order comparison. Filename order is compared with the physical acquisition timeline.

# %% [markdown]
# > **时序重要性说明**：上图对比了文件名排序与物理采集时序。红点标记了真实的温度骤变位置。这证明了文件名排序无法作为时间线，超分辨率重建必须使用基于文件修改时间（`mtime`）恢复的 `acquisition_order`。


# %%
raster_row_summary = cache.raster_row_summary
raster_row_mismatches = cache.raster_row_mismatches
r0_y_order = (
    df[df["R"].eq(0)]
    .sort_values("acquisition_order")
    .groupby("Y")["acquisition_order"]
    .min()
    .sort_values()
    .index.astype(int)
    .tolist()
)

print(f"R=0 采集 Y 顺序: {r0_y_order}")
print(f"R=0 行内 X 顺序不匹配行数: {len(raster_row_mismatches)}")
print("缓存: output/ep01_data_processing/acquisition_order_audit.csv")
display(raster_row_summary.head(12))
if not raster_row_mismatches.empty:
    display(raster_row_mismatches)

# %% [markdown]
# > **扫描路径核对**：审计确认 $R=0$ 数据的真实采集路径符合标准 raster 扫描（行内 $X$ 递增，行间 $Y$ 递增），且文件坐标命名与采集顺序严格一致。


# %%
show_fig("session_detection_a.png")

# %% [markdown]
# Figure 5: Main session scan trajectory. The command grid path is colored by physical acquisition order.

# %% [markdown]
# > **图表解读**：上图展示了主 Session（Session 2）在二维空间坐标网格上的真实采集轨迹，轨迹按时序着色。


# %%
show_fig("session_detection_b.png")

# %% [markdown]
# Figure 6: Session coordinate timeline. Commanded coordinates are shown over acquisition order with session bands.

# %% [markdown]
# > **图表解读**：上图展示了扫描命令坐标随物理采集顺序的变化，背景色划分了主/非主 Session。
#
# ### 🕵️‍♂️ 主 Session 空间与时间覆盖诊断
#
# 采集轨迹图与时序变化图直观表明：
# - **空间覆盖**：主 Session（Session 2）完整覆盖了除 3 个缺失点外的全部扫描坐标网格。
# - **时序分布**：非主 Session 的开机温升帧（Session 0）全部集中在时间轴最前端，表明系统在第 9 帧起完全稳定，之后无异常断点。
#
# **💡 算法物理决策**：后续的位移估计（EP04）与超分辨率重建（EP10/11）必须有且仅有使用主 Session 的 255 帧作为默认输入，以保证图像平移在均一、稳定的物理热背景下进行。
