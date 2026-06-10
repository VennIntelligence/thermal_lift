# %% [markdown]
# ## 2.3 坐标/R 分布与缺失数据校验

# %%
known_missing = coord_config.get("known_missing_r0", [])
print("known_missing_r0 验证:")
for x, y in known_missing:
    sub = df[(df["X"].eq(x)) & (df["Y"].eq(y))]
    repeats = sorted(sub["R"].unique().tolist()) if not sub.empty else []
    status = "⚠️ 该坐标完全不存在" if not repeats else \
             ("❌ R=0 实际存在" if 0 in repeats else "✅ 确认缺失")
    print(f"  ({x:2d}, {y:2d}): R={repeats}  {status}")

for _, row in cache.coord_repeat_summary.iterrows():
    print(
        f"  {int(row['frames_per_coordinate'])}-repeat: "
        f"{int(row['n_coordinates'])} 个坐标"
    )

print(f"  总唯一坐标: {len(set(zip(df['X'], df['Y'])))}")
display(cache.coord_repeat_summary)
display(cache.r_distribution)
display(cache.missing_coordinate_table)

# %% [markdown]
# ### 🗺️ 空间覆盖完整度与缺失坐标
#
# 红外微扫描在设计上覆盖 $16 \times 16$ 共 256 个空间坐标。审计结果显示：
# - 实际覆盖 **253 个** 唯一坐标（覆盖率 98.8%）。
# - 坐标 $(14, 6)$、$(16, 6)$ 以及 $(16, 16)$ 完全缺失，原因为硬件同步触发漏帧。
#
# **💡 算法对策**：物理实测缺失点不可补造，以免引入错误先验。后续超分辨率重建算法（如 Drizzle）需对这 3 个数据盲区进行置信度平滑和掩模（Mask）标记，防止强行插值产生假阳性轮廓。
