# %% [markdown]
# ## B1. 温度场对与绝对误差
#
# 验收应优先看这一层（°C 域），不是 highpass 红蓝图。

# %%
save_fig("08_temperature_pair.png")

# %% [markdown]
# ## B2. Loss 1 — MSE
#
# $$\mathcal{L}_{mse}=\mathrm{mean}\big((pred-target)^2\big), \quad w_{mse}=0.02$$

# %%
save_fig("09_mse_loss.png")

# %% [markdown]
# > **图表说明**: 右图 MSE map 亮区=温度偏差最大处；当前权重极小，对 total 几乎无话语权。
# >
# > **核心发现**: 要平滑温度场，应增强 mse，而不是删掉它。
