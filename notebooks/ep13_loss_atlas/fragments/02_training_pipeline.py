# %% [markdown]
# ## Part A — 训练数据从哪来？
#
# EP07 UNet 在 **合成 training_pool** 上训练。完整链路分两段：
#
# **离线（TCForge 生成 scene）**
#
# ```
# HR mask (旋转芯片几何)
#   -> render_temperature_field()  得到 HR 温度 GT
#   -> generate_lr_burst(N frames)  前向模型 + 探测器噪声 + 帧间 drift
#   -> fuse_burst_to_features()     对齐后融合成 5 通道 obs_features
#   -> 紧凑存盘：mask PNG + obs_features.npz + metadata（不存 248 帧 burst）
# ```
#
# **在线（PyTorch DataLoader）**
#
# ```
# 读 obs_features patch + 从 mask/metadata 重建 HR target patch
#   -> UNet(obs) -> pred
#   -> ContourSRLoss(pred, target)
# ```
#
# 下图是总览；后续各节展开每一步。

# %%
save_fig("00_training_pipeline_schematic.png")

# %% [markdown]
# > **图表说明**: 从 TCForge 离线生成到 UNet 训练再到 loss 的总流程示意。
# >
# > **怎么看**: 注意分叉——磁盘上只留 compact scene；248 帧 burst 只在离线阶段存在，训练时不再读取。
# >
# > **核心发现**: 你看到的「训练输入」本质是 **5 通道 1×LR 融合特征**，不是 248 张原始温度矩阵。
