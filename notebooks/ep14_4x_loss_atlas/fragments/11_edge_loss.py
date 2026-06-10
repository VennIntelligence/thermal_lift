# %% [markdown]
# ## B6. Edge Loss：多尺度 Sobel 与边界加权（Loss 3）
#
# **Edge Loss** 旨在拉紧结构轮廓，防止 4x 放大时边缘变肉、变模糊：
#
# $$\mathcal{L}_{\text{edge}} = \text{WeightedMean}(\left| \text{Sobel}(y_{\text{pred}}) - \text{Sobel}(y_{\text{target}}) \right|, w_{\text{edge}}) + 0.25 \cdot \text{mean} \left| \text{Sobel}(y_{\text{pred}}^{\downarrow 2}) - \text{Sobel}(y_{\text{target}}^{\downarrow 2}) \right|$$
#
# - **Fine Scale (1x)**：对提取的 Sobel 梯度计算 L1 损失，并使用 `edge_mask` 增强，对关键的物理结构边缘赋予更高的权重提升（例如 $1 + 2.0 \cdot \text{edge\_mask}$）。
# - **Coarse Scale (2x downsampled)**：在 $2\times$ 下采样尺度上算 Sobel L1 误差，用以增强边缘的整体连续性。

# %%
save_fig("14_edge_loss_4x.png")

# %% [markdown]
# > **图表说明**: 依次展示了 Sobel 细尺度误差、边界加权权重图以及最终加权后的边缘损失。
# >
# > **核心发现**: 多尺度 Sobel 损失在约束温度断面上非常关键，它强迫网络重建清晰、陡峭的 4x 温度跃变边界。
