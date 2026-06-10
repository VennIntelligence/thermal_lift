# %% [markdown]
# ## A3. 前向退化：物理 LR Burst（只示意见 6 帧）
#
# 前向退化通过对 HR 施加位移 offset、PSF 卷积和 block averaging 降采样（倍率为 4），外加探测器噪声和热漂变，生成 LR 图像序列：
#
# $$\hat{y}_k = \mathcal{F}_{\text{forward}}(x_{\text{HR}}, \delta_k) + \text{noise}_k + \text{drift}_k$$
#
# LR 的空间大小为 HR 的 $1/4$。

# %%
save_fig("03_lr_burst_samples_4x.png")

# %% [markdown]
# ## A4. 对齐、Drizzle 与融合：8通道输入特征
#
# 将 $N=248$ 帧的 sub-pixel 观测在参考网格上对齐：
# 1. 对齐后进行 **Drizzle 4x 散点累加**，生成 3 通道 4x 特征：
#    - `ch0`: `drizzle_mean_4x` (对齐均值温度)
#    - `ch1`: `drizzle_coverage_4x` (覆盖度)
#    - `ch2`: `drizzle_variance_4x` (观测稳定性方差)
# 2. 对齐后进行 **1x 统计融合**，生成 5 通道 1x 特征，然后双线性插值到 4x 尺寸：
#    - `ch3-7`: 插值放大的 `aligned_mean`, `aligned_median`, `coverage`, `variance`, `highpass_fused`
#
# 拼接后得到最终输入网络的 8 通道特征：

# %%
save_fig("05_obs_feature_channels_4x.png")

# %% [markdown]
# > **图表说明**: 上图展示了 $6$ 个退化后的 $1/4$ LR 帧。下图展示了最终送入 4x UNet 的 8 个特征通道。
# >
# > **怎么看**: `ch1 drizzle_coverage_4x` 是 4x Drizzle 独有的覆盖率。通道 3 至 7 是经典的 1x 融合指标通过插值放大而来的平滑版本。这两者信息的互补为 UNet 在 4x 空间重建超分辨率轮廓提供了绝佳的位移与覆盖保障。
