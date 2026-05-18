# %% [markdown]
# ## 2.3 坐标/R 分布与缺失数据校验
#
# 验证 `known_missing_r0` 声称的 3 个坐标，同时统计每个 repeat 编号和每个坐标的实际覆盖。
#
# **SR 关注点**: 坐标和 repeat 不是最终对齐真值，但它们定义了后续微扫描 SR 的采样 prior 和可用帧池。

# %%
coord_map = build_coord_repeat_map(df)

# 检查 known_missing_r0
known_missing = coord_config.get("known_missing_r0", [])
print("known_missing_r0 验证:")
for x, y in known_missing:
    repeats = sorted(coord_map.get((x, y), set()))
    status = "⚠️ 该坐标完全不存在" if not repeats else \
             ("❌ R=0 实际存在" if 0 in repeats else "✅ 确认缺失")
    print(f"  ({x:2d}, {y:2d}): R={repeats}  {status}")

# 重复次数统计
repeat_counts = {k: len(v) for k, v in coord_map.items()}
for n in [3, 2, 1]:
    coords_n = [k for k, c in repeat_counts.items() if c == n]
    print(f"  {n}-repeat: {len(coords_n)} 个" + (f"  → {sorted(coords_n)}" if n >= 2 else ""))
print(f"  总唯一坐标: {len(coord_map)}")

coord_repeat_summary = (
    pd.Series(repeat_counts, name="frame_count")
    .value_counts()
    .sort_index(ascending=False)
    .rename_axis("frames_per_coordinate")
    .reset_index(name="n_coordinates")
)
r_distribution = (
    df.groupby("R")
    .size()
    .rename("n_frames")
    .to_frame()
    .join(
        df.drop_duplicates(["R", "X", "Y"]).groupby("R").size().rename("n_unique_coordinates")
    )
    .reset_index()
)
display(coord_repeat_summary)
display(r_distribution)

# %% [markdown]
# > **数据说明**: 这一段把每个合法 (X,Y) 坐标下实际存在的 repeat 编号集合列出来，
# > 并汇总每个坐标拥有多少帧，用来核对缺失坐标、重复测量坐标和 repeat 分布。
# > 在这里，坐标描述电动台命令位置，repeat 描述同一坐标下重复拍摄的编号。
# >
# > **怎么读**: `known_missing_r0` 行逐个核查配置里声明的缺失坐标；
# > `coord_repeat_summary` 表告诉我们有多少坐标拥有 1/2/3 帧；
# > `r_distribution` 表则从 repeat 编号角度统计 R=0、R=1、R=2 分别覆盖多少帧和多少唯一坐标。
# >
# > **数据分布**: 共有 253 个唯一坐标；其中 247 个只有 1 次测量，4 个坐标有完整 3-repeat，
# > 另外 2 个坐标有 2-repeat。`known_missing_r0` 中的 3 个坐标实际上完全不存在。
# >
# > **正常/异常理解**: 对微扫描数据，理想情况是坐标网格大体完整；repeat 不完整本身不是致命问题，
# > 因为本项目的主输入是主 session 扫描网格，不是靠 repeat 平均来提高分辨率。
# > 异常情况包括大量坐标缺失、某个 repeat 编号覆盖异常集中，或配置声称缺失但数据中实际存在。
# >
# > **核心发现**: 缺失是坐标级缺失，不只是缺少某个 repeat 编号。
# > 对 SR 而言，这意味着主输入应来自真实存在的 253 个坐标；repeat 只能作为少量局部一致性诊断，不能替代主 session 内对齐。
