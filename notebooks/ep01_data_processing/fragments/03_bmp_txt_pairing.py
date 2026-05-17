# %% [markdown]
# ## 2.2 BMP-TXT 配对检查
#
# **结论**: 263/263 完美配对，0 孤立文件。

# %%
pairing = check_bmp_txt_pairing(DATA_DIR)
print(f"TXT: {pairing['n_txt']}  |  BMP: {pairing['n_bmp']}  |  配对: {pairing['n_paired']}")
print(f"孤立 TXT: {len(pairing['only_txt'])}  |  孤立 BMP: {len(pairing['only_bmp'])}")

# %% [markdown]
# > **数据说明**: 这里检查同名 TXT 温度矩阵和 BMP 预览图是否一一对应。
# > TXT 是后续定量分析数据源，BMP 只作为人工核查和视觉参考。
# >
# > **数据分布**: 263 个 TXT 和 263 个 BMP 全部配对，没有孤立 TXT 或孤立 BMP。
# >
# > **核心发现**: 原始文件重命名后没有破坏 TXT/BMP 对应关系；后续可放心用 TXT 做数值分析，
# > 同名 BMP 可作为定位和排查异常帧的辅助材料。

# %% [markdown]
# ## 2.2b 重命名消歧核查
#
# 核查原始连写文件名到 `X_Y_R` 的映射。重点检查带前导 0、中文逗号、以及 `2400` 这类容易误读的名字。

# %%
if rename_mapping.empty:
    print("⏭️  rename_mapping.csv 不存在或为空，跳过重命名消歧核查")
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
# > **数据说明**: 这张表列出最容易误读的原始文件名及其标准化结果。
# > 例如 `0200` 因为前导 0 表示 X=0，所以应解释为 `(0,20,0)`；`2400` 按原始说明特判为 `(24,0,0)`。
# >
# > **数据分布**: 特殊项主要集中在 X 或 Y 为 0、20、24、40 的坐标上；
# > 中文逗号文件 `2，400` 被保留为 `(2,40,0)`。
# >
# > **核心发现**: 当前没有证据表明重命名把坐标大面积解错。
# > 后续发现的 session 异常更可能来自采集顺序判断，而不是 `X_Y_R` 解码本身。
