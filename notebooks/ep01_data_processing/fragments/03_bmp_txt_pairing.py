# %% [markdown]
# ## 2.2 BMP-TXT 配对检查
#
# **SR 关注点**: TXT 是数值重建输入，BMP 是同名视觉参考；配对完整可以降低后续异常帧排查成本。

# %%
pairing = check_bmp_txt_pairing(DATA_DIR)
print(f"TXT: {pairing['n_txt']}  |  BMP: {pairing['n_bmp']}  |  配对: {pairing['n_paired']}")
print(f"孤立 TXT: {len(pairing['only_txt'])}  |  孤立 BMP: {len(pairing['only_bmp'])}")

# %% [markdown]
# > **数据说明**: 这里检查同名 TXT 温度矩阵和 BMP 预览图是否一一对应。
# > TXT 是后续定量分析数据源，BMP 只作为人工核查和视觉参考；两者同名，说明它们描述同一次采集。
# >
# > **怎么读**: `TXT` 和 `BMP` 是两类文件总数，`配对` 是同时存在同名 TXT/BMP 的数量。
# > `孤立 TXT` 表示有温度矩阵但没有预览图；`孤立 BMP` 表示有图像预览但缺少可计算的温度矩阵。
# >
# > **数据分布**: 263 个 TXT 和 263 个 BMP 全部配对，没有孤立 TXT 或孤立 BMP。
# >
# > **正常/异常理解**: 对本项目而言，正常情况是 TXT/BMP 数量相等且配对数等于总帧数。
# > 少量孤立 BMP 不会直接影响 SR 输入，但会降低人工排查能力；孤立 TXT 可以计算，但需要确认不是命名错误。
# >
# > **核心发现**: 原始文件重命名后没有破坏 TXT/BMP 对应关系；后续 SR 可使用 TXT 做数值重建，
# > 同名 BMP 可作为定位和排查异常帧的辅助材料。

# %% [markdown]
# ## 2.2b 重命名消歧核查
#
# 核查原始连写文件名到 `X_Y_R` 的映射。重点检查带前导 0、中文逗号、以及 `2400` 这类容易误读的名字。

# %%
if rename_mapping.empty:
    rename_special = pd.DataFrame([{
        "status": "rename_mapping.csv missing or empty",
        "interpretation": "current files are already in X_Y_R form; no raw-name mapping table is available in output",
    }])
else:
    special_old_names = {
        "0200.txt", "0240.txt", "0280.txt", "0400.txt",
        "2000.txt", "2020.txt", "2040.txt", "2060.txt", "2080.txt",
        "2400.txt", "2，400.txt",
        "4000.txt", "4020.txt", "4040.txt", "4060.txt", "4080.txt",
    }
    rename_special = (
        rename_mapping[
            rename_mapping["old_name"].isin(special_old_names)
            | rename_mapping["old_name"].astype(str).str.contains("，", regex=False)
        ][["old_name", "new_name", "X", "Y", "R", "ext"]]
        .sort_values(["ext", "old_name"])
        .reset_index(drop=True)
    )
rename_special

# %% [markdown]
# > **数据说明**: 这张表列出最容易误读的原始文件名及其标准化结果；
# > 如果当前环境没有保留 `rename_mapping.csv`，表中会记录该核查产物缺失。
# > 例如 `0200` 因为前导 0 表示 X=0，所以应解释为 `(0,20,0)`；`2400` 按原始说明特判为 `(24,0,0)`。
# >
# > **怎么读**: `old_name` 是原始连写文件名，`new_name` 是标准化后的 `X_Y_R.ext`。
# > `X/Y` 是电动台命令坐标，单位是 µm；`R` 是重复编号，不是采集顺序。
# >
# > **数据分布**: 有映射表时，特殊项主要集中在 X 或 Y 为 0、20、24、40 的坐标上；
# > 中文逗号文件 `2，400` 应被保留为 `(2,40,0)`。
# >
# > **正常/异常理解**: 正常表格应把前导 0、中文逗号和 `2400` 这类歧义名映射到唯一合法坐标。
# > 如果同一个原始名能解释成多个坐标，或者标准化坐标不在允许集合内，就需要停止后续分析并修正命名规则。
# >
# > **核心发现**: 当前没有证据表明重命名把坐标大面积解错。
# > 后续 SR 可以把 `X_Y_R` 坐标作为 stage command prior 的来源，但真实帧时序仍必须使用 `acquisition_order`。
