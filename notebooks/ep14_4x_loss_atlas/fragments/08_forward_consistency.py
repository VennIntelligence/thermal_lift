# %% [markdown]
# ## B3. Forward Consistency Loss：物理前向投影约束（Loss 4）
#
# **前向物理保真约束** 模拟了红外相机的成像光学过程：
#
# 一个物理合理的 4x HR 预测在经过光学 PSF 模糊和探测器像元积分（4x4 均值下采样）后，应该与真实的 1x 观测（observed drizzle mean）保持一致。
#
# $$\mathcal{L}_{\text{forward}} = \text{WeightedMean} \left( \left| \text{Pool}_{4\times 4}(\text{PSF}(y_{\text{pred}})) - y_{\text{obs\_lr}} \right|, \text{coverage}_{\text{lr}} \right)$$
#
# 这里的 $y_{\text{obs\_lr}}$ 是在 1x LR 尺度上由 drizzle_mean 与 drizzle_coverage 合成的真实探测器观测。

# %%
save_fig("11_forward_consistency_flow.png")

# %% [markdown]
# > **图表说明**: 从 4x pred 通过 PSF 卷积和 $4 \times 4$ 下采样，映射到 LR 尺度，与 Observed 观测进行对比。
# >
# > **核心发现**: 这是一个极其关键的无监督/自监督物理约束。即使是在实测芯片没有 4x GT 标签时，我们仍能通过此项约束，迫使网络生成的 4x HR 结果能够“物理上解释”探测器的 1x 原始观测输入。
