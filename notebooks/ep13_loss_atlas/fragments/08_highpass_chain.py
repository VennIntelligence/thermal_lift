# %% [markdown]
# ## B3. Loss 2 — Highpass（训练主导项）
#
# $$hp = pred - \mathcal{G}_{\sigma=5}(pred)$$
#
# $$\mathcal{L}_{highpass} = \frac{\sum |hp_{pred}-hp_{target}| \cdot w_{ij}}{\sum w_{ij}}$$
#
# 权重 $w = 1 + 4.0 \cdot \text{norm\_gradient}_{\text{target}}$。

# %%
save_fig("10_highpass_1d_profile.png")

# %%
save_fig("11_highpass_maps.png")

# %%
save_fig("12_highpass_weighted_error.png")

# %% [markdown]
# > **图表说明**: 1D 剖面展示「减掉背景」；2D 图展示 pred/target 各走一遍；加权图展示连续的梯度结构加权。
# >
# > **怎么看**: highpass 忽略焊盘内部绝对温度，只盯局部结构响应；连续梯度权重赋予了轮廓边缘（如 Sobel 响应高的地方）更大的权重。
# >
# > **核心发现**: 现在的梯度连续加权代替了原来不稳定的几何二值骨架加权，不仅使训练梯度更平滑，而且在像素旋转网格上具备极佳的数值稳定性。

