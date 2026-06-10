# %% [markdown]
# ## B4. Loss 3 — Edge (Sobel)
#
# $$\mathcal{L}_{edge} = \mathrm{mean}|S(pred)-S(target)| + 0.25\cdot\mathrm{mean}|S(pred^{\downarrow2})-S(target^{\downarrow2})|$$

# %%
save_fig("13_edge_loss.png")

# %% [markdown]
# ## B5. Loss 4 — SSIM
#
# $$\mathcal{L}_{ssim} = 1 - \mathrm{SSIM}(pred, target)$$

# %%
save_fig("14_ssim_loss.png")

# %% [markdown]
# > **图表说明**: Edge 比梯度强度；SSIM 看局部结构统计相似。
# >
# > **核心发现**: Edge 与 highpass 同类，都会推锐边；SSIM 相对温和，通常可保留。
