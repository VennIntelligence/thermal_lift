# %% [markdown]
# ## A1. 整体训练管线与数据流
#
# 下图展示了 EP12 4x 算法的训练数据流和离线特征构建。
#
# 与 2x 算法不同，4x 算法为了降低内存开销，同时保证 4x drizzle 特征（scatter-add）的保真度，在生成 `training_pool_4x` 时，离线保存了 `obs_features_1x.npz` (5ch 1x 融合特征) 以及 `obs_features_4x.npz` (3ch 4x Drizzle 特征)。
# 在 DataLoader 中在线加载时，将这二者拼接，并在 GPU 或 Dataloader 阶段将 1x 特征插值放大到 4x 尺度，最终形成 **8通道的输入** 送入网络。

# %%
save_fig("00_training_pipeline_schematic_4x.png")

# %% [markdown]
# > **图表说明**: 整个管线包含前向退化、4x Drizzle 特征直接映射、1x 融合特征、DataLoader 在线拼接和 crop，最终形成 8ch 4x 尺度张量送入 4x UNet，输出温度场和不确定度方差，通过 `ThermalSR4xLoss` 引导训练。
