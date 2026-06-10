# %% [markdown]
# ## B2. LF Loss：低频基准与温度校准（Loss 1）
#
# **LF (Low Frequency) Loss** 对预测图与目标图施加一个非常大的高斯模糊（$\sigma_{\text{lf}} = 8.0$），再计算 L1 Loss：
#
# $$\mathcal{L}_{\text{lf}} = \frac{1}{\Omega} \sum \left| \text{Blur}_{\sigma}(y_{\text{pred}}) - \text{Blur}_{\sigma}(y_{\text{target}}) \right|$$
#
# **为什么用 LF？** 
# 在高倍率下，如果只用高通或边缘损失，网络可能会因为失去低频绝对值锚定而产生温度的整体漂移。LF 过滤掉了高频边缘细节，使网络强行对齐大尺度温区。

# %%
save_fig("09_lf_loss_4x.png")

# %% [markdown]
# > **图表说明**: 展示了经过 Gaussian blur $\sigma=8.0$ 之后的 Target 和 Pred 低频图像，以及它们的残差。
# >
# > **核心发现**: 尽管 Pred 中有明显的伪影，但在 LF 模糊后，局部的高频震荡被抹平，只保留了大趋势的温差，这使得网络能够专注低频校准而不受高频噪声干扰。
