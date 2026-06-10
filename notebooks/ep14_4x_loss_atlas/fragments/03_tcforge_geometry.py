# %% [markdown]
# ## A2. HR mask 与物理温度场 target
#
# 在训练时，物理温度的真实标签（GT）不是从磁盘读取的巨大的 `.npy` 文件，而是使用 `hr_mask_4x.png` 结合 `metadata.json`（包含 `T_bg_c`, `delta_T_c` 及 seed 等物理参数）在线重构的。
#
# 这既节省了极大的磁盘 IO 和存储空间，也保证了前背景温度梯度的物理准确度。

# %%
save_fig("02_hr_mask_and_temperature_4x.png")

# %% [markdown]
# > **图表说明**: 左图是 4x HR mask 几何，右图是在线重建出来的 4x 温度目标 (GT)。
# >
# > **核心发现**: 这种在线重建方案使得数据集在磁盘上极其轻量（仅包含二进制 mask PNG，不含任何浮点温度矩阵），避免了极度高昂的磁盘读取瓶颈。
