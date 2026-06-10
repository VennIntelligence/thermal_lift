# %% [markdown]
# ## A1. 整体训练管线与数据流
#
# 下图展示的是 EP14 loss-atlas 使用的旧 EP12 4x same-grid 数据流，用作 loss 讲解历史上下文；当前 EP12 Hybrid 主流程已改为从 `lr_burst.npy + shifts.npy` 按需计算 2x drizzle。
#
# 旧 same-grid 路线曾离线保存 `obs_features_1x.npz` (5ch 1x 融合特征) 以及 `obs_features_4x.npz` (3ch 4x Drizzle 特征)。
# 新 Hybrid 路线不再要求这些 4x drizzle 预计算文件，训练池只需保留 `obs_features_1x.npz`、`lr_burst.npy`、`shifts.npy`、soft `hr_mask_4x.png` 和 metadata。

# %%
save_fig("00_training_pipeline_schematic_4x.png")

# %% [markdown]
# > **图表说明**: 该历史示意包含前向退化、4x Drizzle 特征直接映射、1x 融合特征、DataLoader 在线拼接和 crop，最终形成 8ch 4x 尺度张量送入旧 4x UNet。
