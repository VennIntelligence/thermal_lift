# %% [markdown]
# ## 2. HR 几何与温度真值
#
# TCForge 在 2x HR 网格上生成二值结构 mask，再渲染连续温度场和形态学 edge proxy。4x sanity 阵列仅用于尺度一致性检查，不是交付目标。

# %%
if cache.demo_skipped:
    display(Markdown("HR 真值图不可用：请先重建 EP07 缓存。"))
else:
    display(compact_table(cache.scene_stats, ["array", "role", "shape", "dtype", "min", "max", "finite"]))
    show_fig("demo_hr_scene.png")

# %% [markdown]
# > **图表说明**: Figure 1 展示 2x HR mask、温度场、edge proxy，以及 4x sanity 对照；表格记录各 `.npy` 数组的 shape/dtype/数值范围。
# >
# > **怎么看**: mask 必须是 `uint8` 且值域 `{0,1}`；温度场为 `float32` 且全图 finite。温度边缘比 mask 更平滑，这是物理渲染预期，不是 bug。
# >
# > **核心发现**: `hr_temperature_2x` 是 forward 之前的清洁真值；后续 PSF 与噪声只作用在 LR observation 链路上。
