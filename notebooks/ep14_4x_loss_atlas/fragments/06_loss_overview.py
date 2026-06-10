# %% [markdown]
# ## B1. Loss 评估：ThermalSR4xLoss 6项配方
#
# 下面我们通过截取中心区域一个 $128 \times 128$ 大小的 HR patch，并对其施加人工干扰（模拟 UNet 产生的振铃及随机噪声），同时模拟由不确定度预测分支输出的 `log_var`（不确定度）。
# 
# 我们使用这组 Pred - Target 对来逐步拆解说明 4x 超分辨率重建的六项损失子项的物理作用。

# %%
save_fig("08_temperature_pair_4x.png")

# %% [markdown]
# > **图表说明**: 左图为 4x HR 真值，中图为带人工伪影/噪声的预测图，右图为绝对温差 $|\text{pred} - \text{target}|$。
