# %% [markdown]
# ## A5. 磁盘上的 Compact 存储与切片加载
#
# 为了避免在磁盘上保存巨大的 LR burst 导致网络训练时的 IO 灾难，`training_pool_4x` 中每个 scene 在磁盘上仅以 `Compact` 形式保存。
#
# 磁盘保存项包括：
# - 4x 二值 mask PNG
# - 4x Drizzle (3ch) 特征压缩文件
# - 1x Fused (5ch) 特征压缩文件
# - `metadata.json`
#
# 这种机制减少了近 90% 的物理存储，并极大加快了训练中的读取速度。

# %%
save_fig("06_compact_storage_schematic_4x.png")

# %% [markdown]
# > **图表说明**: 磁盘上各文件的体积以及不需要存储的项。在 DataLoader 运行时，会随机截取 size 为 $256 \times 256$ 的 8ch 空间 patch 并执行 `Data Augmentation`（翻转、旋转），最后与同样 crop 出来的 4x HR temperature 算 loss。
