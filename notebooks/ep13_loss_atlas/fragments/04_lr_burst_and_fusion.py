# %% [markdown]
# ## A3. 前向模型：LR burst（只示意见 6 帧）
#
# **Step 3** 对 HR 温度场施加微扫描位移 + PSF + block downsample，再叠加探测器噪声与帧间 drift（与 `training_pool_2x` 同序）：
#
# $$\hat{y}_k = \mathcal{F}_{forward}(x_{HR}, \delta_k) + \text{noise}_k + \text{drift}_k, \quad k=1..N$$
#
# - 本 demo：展示 $N_{demo}=16$ 帧样品；离线融合用完整 $N=248$ 帧（`n_frames_per_scene`，可配置）
# - 位移：默认 EP05 `real_default_contour_refined`；角度：47.6° ± jitter（可配置）
#
# 下面只画 **6 个采样帧**，避免把 248 张全铺开。

# %%
save_fig("03_lr_burst_samples.png")

# %% [markdown]
# ## A4. 对齐 + 融合：5 通道 obs_features
#
# **Step 4** 每帧带亚像素 shift，先对齐到 reference grid，再统计融合：
#
# | 通道 | 名称 | 含义 |
# |------|------|------|
# | ch0 | aligned mean | 对齐后均值（主强度） |
# | ch1 | aligned median | 对齐后中位数（抗 outlier） |
# | ch2 | coverage | 每像素被多少帧覆盖 |
# | ch3 | variance | 对齐后方差（稳定性） |
# | ch4 | highpass fused | 帧内 highpass 后再融合 |
#
# **这就是 UNet 的输入张量**（1×LR 分辨率，5 通道）。

# %%
save_fig("04_alignment_fusion_schematic.png")
save_fig("05_obs_feature_channels.png")

# %% [markdown]
# > **图表说明**: 上图展示两帧在不同 shift 下对齐到同一网格；下图展示融合后的 5 个通道。
# >
# > **怎么看**: ch2 coverage 边缘更低是正常现象（边缘帧覆盖少）；ch4 是评估域 highpass，与 loss 里的 highpass 同一族操作。
# >
# > **核心发现**: 网络从未直接看到 248 帧 tensor；看到的是 **已经对齐融合好的 5×H×W 特征图**。
