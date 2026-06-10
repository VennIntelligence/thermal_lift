# %% [markdown]
# ## B4. Heteroscedastic NLL Loss：置信度感知负对数似然（Loss 5）
#
# 在红外超分辨率重建中，由于局部噪声的非均匀分布以及结构边缘的亚像素对齐波动，各像素的误差分布是不均匀的（异方差性）。
# 
# 4x UNet 模型除了输出预测温度 $y_{\text{pred}}$ 以外，还输出一个不确定性对数方差通道 $\log(\sigma^2)$：
#
# $$\mathcal{L}_{\text{nll}} = \frac{1}{\Omega} \sum w_i \left( \frac{1}{2} e^{-\log(\sigma_i^2)} (y_{\text{pred}, i} - y_{\text{target}, i})^2 + \frac{1}{2} \log(\sigma_i^2) \right)$$
#
# 这一机制允许网络“承认”某些区域（例如突变的结构边缘、极低覆盖的边缘带）很难完全精准预测。通过提高这些像素的 $\log(\sigma^2)$ 值，可以自适应平抑这些区域给梯度带来的负面惩罚，使整体重建更加稳定和清晰。

# %%
save_fig("12_heteroscedastic_nll_flow.png")

# %% [markdown]
# > **图表说明**: 依次展示了绝对预测误差、网络自学习输出的不确定性标准差（$\sigma$）、加权 NLL 损失分布。
# >
# > **怎么看**: 在预测误差较大的地方，网络自发地学到了较高的 uncertainty sigma（例如边缘处变白），从而在 NLL 中对该处的二次平方项误差进行衰减，防止网络因个别高难度边缘像素而过度改变主体温度预测。
