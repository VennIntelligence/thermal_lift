# %% [markdown]
# ## 2.1 TXT 温度矩阵审计
#
# 逐帧读取全部 TXT，验证尺寸一致性、NaN/Inf、温度范围，并补充逐帧中位温与稳健温度统计。
#
# **SR 关注点**: 后续 SR 的输入是 raw 温度矩阵；这一节确认每一帧都处在同一 detector grid 上。

# %%
df = audit_all_frames(DATA_DIR)
df = add_robust_temperature_stats(df, DATA_DIR)
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
print(f"逐帧中位温范围: [{df['T_median'].min():.2f}, {df['T_median'].max():.2f}] °C")
print(f"采集顺序字段: acquisition_order={df['acquisition_order'].min()}–{df['acquisition_order'].max()}")

# %% [markdown]
# > **数据说明**: 这一组指标来自 263 个 TXT 温度矩阵的逐帧审计。
# > 每个 TXT 可以理解为一张 480×640 的温度图，单元格不是颜色值，而是摄氏温度读数。
# > 这里同时检查矩阵尺寸、NaN/Inf、单像素温度范围、逐帧均温和逐帧中位温。
# >
# > **怎么读**: `rows × cols` 说明 detector grid 是否统一；NaN/Inf 帧数说明是否有无法参与数值计算的坏帧；
# > `T_min/T_max` 是所有像素层面的极值，`T_mean/T_median` 更接近一整帧的背景温度状态。
# > `acquisition_order` 是真实采集顺序字段，后续时间线必须按它排序。
# >
# > **数据分布**: 所有帧尺寸一致，均为 480 行 × 640 列；NaN/Inf 均为 0。
# > 单像素温度覆盖 18.21–26.80°C，逐帧均温覆盖 19.69–23.85°C，逐帧中位温也保留相同的温度段结构。
# > 每帧同时保留 `filename_order` 与 `acquisition_order`，后者来自文件修改时间，用于还原真实采集顺序。
# >
# > **正常/异常理解**: 正常数据应当尺寸一致、没有 NaN/Inf，并且温度范围处在合理物理区间。
# > 如果尺寸不一致，后续帧间配准和 SR 网格会立即失效；如果 NaN/Inf 不为 0，需要先隔离坏帧；
# > 如果逐帧均温跨度很大，通常不是“更丰富的信息”，而是不同采集温度状态混在了一起。
# >
# > **核心发现**: TXT 原始矩阵本身可读且完整，263 帧可以作为 EP01 的完整审计对象。
# > 温度均值跨度达到 4°C 以上，说明后续 SR 必须先按真实采集顺序建立 session 模型，再在 session 内重建。
