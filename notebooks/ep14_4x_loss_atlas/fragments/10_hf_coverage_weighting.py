# %% [markdown]
# ## B5. HF 与 HF Detail 损失：基于覆盖率（Coverage）的局部细节引导（Loss 2 & Loss 6）
#
# 对于高频超分辨率细节，4x 算法基于 `drizzle_coverage_4x` 对高频（High Frequency）项施加了两种完全互补的权重方案：
#
# 1. **Loss 2 — HF Loss (Coverage 加权)**：
#    对 coverage 较高的像素给予更高的关注，确保有充足 sub-pixel 对齐证据的中心主要轮廓被还原：
#    $$\mathcal{L}_{\text{hf}} = \text{WeightedMean} \left( \left| \text{HP}(y_{\text{pred}}) - \text{HP}(y_{\text{target}}) \right|, w_{\text{cov}} \right)$$
#    $$w_{\text{cov}} = 1.0 + 4.0 \cdot \sqrt{\text{norm\_coverage}}$$
#
# 2. **Loss 6 — HF Detail Loss (反 Coverage 加权)**：
#    对 coverage 较低的像素（比如边缘或边界采样较少的盲区）给予更高的关注，避免边缘区域细节被主干轮廓支配而被过度忽略：
#    $$\mathcal{L}_{\text{hf\_detail}} = \text{WeightedMean} \left( \left| \text{HP}(y_{\text{pred}}) - \text{HP}(y_{\text{target}}) \right|, w_{\text{inv\_cov}} \right)$$
#    $$w_{\text{inv\_cov}} = 1.0 + 4.0 \cdot (1.0 - \sqrt{\text{norm\_coverage}})$$

# %%
save_fig("13_hf_coverage_weighting.png")

# %% [markdown]
# > **图表说明**: 上图展示高频 L1 误差通过 coverage 加权图放大核心区域高覆盖边缘的损失；下图展示通过反 coverage 加权放大低覆盖盲区细节的损失。
# >
# > **核心发现**: 这种双加权设计使得模型在主要高频轮廓被还原的同时，能够稳健地找回位于亚像素运动边界处的微弱细节。
