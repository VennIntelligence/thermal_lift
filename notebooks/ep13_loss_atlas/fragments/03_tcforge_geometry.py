# %% [markdown]
# ## A1. HR mask + 物理温度场
#
# **Step 1** `build_scene_mask_with_metadata()` → 二值 HR mask  
# **Step 2** `reconstruct_hr_temperature()` → 在 mask 上渲染前景/背景温差 + 低频背景
#
# 这两张图来自 **同一次 TCForge 生成**（`seed=13`，`medium` 难度），几何已是真实训练风格（通常包含旋转）。

# %%
save_fig("02_hr_mask_and_temperature.png")

# %% [markdown]
# > **图表说明**: 左 HR mask，右 HR 温度场（°C）。结构方向与真实 training_pool 一致。
# >
# > **怎么看**: 若这里看起来正正方方，说明没走 TCForge；本 EP13 已改为 TCForge 输出。
# >
# > **核心发现**: **target 温度 GT 在训练时不是从磁盘读取的 .npy**，而是由 mask + `metadata.json` 里的 `T_bg_c`、`delta_T_c` 等参数 **在线重建**。

