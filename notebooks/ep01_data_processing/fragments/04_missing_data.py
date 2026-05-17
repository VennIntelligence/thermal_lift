# %% [markdown]
# ## 2.3 缺失数据校验
#
# 验证 `known_missing_r0` 声称的 3 个坐标。
#
# **结论**: (14,6)(16,6)(16,16) **完全不存在**（不仅缺 R=0）。
# 实际仅 4 个坐标有 3-repeat: (0,0)(2,0)(6,0)(8,0)。

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

# %% [markdown]
# > **数据说明**: 这一段把每个合法 (X,Y) 坐标下实际存在的 repeat 编号集合列出来，
# > 用来核对旧配置中记录的缺失坐标和重复测量坐标。
# >
# > **数据分布**: 共有 253 个唯一坐标；其中 249 个只有 1 次测量，4 个坐标有完整 3-repeat，
# > 另外 2 个坐标有 2-repeat。`known_missing_r0` 中的 3 个坐标实际上完全不存在。
# >
# > **核心发现**: 旧项目“只缺 R=0”的说法不准确；缺失是坐标级缺失。
# > 可用于重复定位评估的坐标比旧记录更少，EP02 的 repeatability 统计必须谨慎解释。
