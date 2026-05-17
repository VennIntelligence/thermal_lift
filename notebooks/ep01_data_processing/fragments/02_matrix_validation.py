# %% [markdown]
# ## 2.1 矩阵读取验证
#
# 逐帧读取全部 TXT，验证尺寸一致性、NaN/Inf、温度范围。
#
# **结论**: 263 帧全部 480×640，无 NaN/Inf，温度范围 [18.21, 26.80]°C。

# %%
df = audit_all_frames(DATA_DIR)
rename_mapping_path = OUTPUT_DIR / "rename_mapping.csv"
rename_mapping = pd.read_csv(rename_mapping_path) if rename_mapping_path.exists() else pd.DataFrame()
if not rename_mapping.empty:
    txt_mapping = (
        rename_mapping[rename_mapping["ext"].str.lower().eq(".txt")]
        [["old_name", "new_name"]]
        .rename(columns={"old_name": "source_file", "new_name": "file"})
    )
    df = df.merge(txt_mapping, on="file", how="left")

print(f"✅ 成功读取: {len(df)} 帧")

# 尺寸一致性
shapes = df[["rows", "cols"]].drop_duplicates()
if len(shapes) == 1:
    print(f"✅ 所有帧尺寸一致: {int(shapes.iloc[0, 0])} 行 × {int(shapes.iloc[0, 1])} 列")

# 数据完好性
print(f"NaN 帧数: {int(df['has_nan'].sum())}  |  Inf 帧数: {int(df['has_inf'].sum())}")
print(f"全局温度范围: [{df['T_min'].min():.2f}, {df['T_max'].max():.2f}] °C")
print(f"全局均温范围: [{df['T_mean'].min():.2f}, {df['T_mean'].max():.2f}] °C")
print(f"采集顺序字段: acquisition_order={df['acquisition_order'].min()}–{df['acquisition_order'].max()}")

# %% [markdown]
# > **数据说明**: 这一组指标来自 263 个 TXT 温度矩阵的逐帧审计，检查每帧的尺寸、非有限值、
# > 全局温度范围和逐帧均温范围。
# >
# > **数据分布**: 所有帧尺寸一致，均为 480 行 × 640 列；NaN/Inf 均为 0。
# > 单像素温度覆盖 18.21–26.80°C，逐帧均温覆盖 19.69–23.85°C。
# > 每帧同时保留 `filename_order` 与 `acquisition_order`，后者来自文件修改时间，用于还原真实采集顺序。
# >
# > **核心发现**: TXT 原始矩阵本身可读且完整，EP01 后续统计可以建立在这 263 帧上。
# > 温度均值跨度达到 4°C 以上，提示后续必须按真实采集顺序检查 session 级温度状态差异。
