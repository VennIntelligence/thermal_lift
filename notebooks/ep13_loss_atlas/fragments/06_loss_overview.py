# %% [markdown]
# ## Part B — 进入 Loss：ContourSRLoss 总公式
#
# UNet 输出 `pred` 与在线重建的 `target` 比较：
#
# $$\mathcal{L}_{total} = 0.02\mathcal{L}_{mse} + 1.0\mathcal{L}_{highpass} + 0.1\mathcal{L}_{edge} + 0.1\mathcal{L}_{ssim} + 0.5\mathcal{L}_{anti}$$
#
# 下面用 **同一块 TCForge 中心 patch** 逐步可视化每一项。
# demo 里的 `pred` 在 GT 上 **人工加了振铃**（教学用），便于对照 highpass/edge 与温度域差异。

# %% [markdown]
# > **数据说明**: Part B 的 pred 不是 checkpoint 推理结果，而是「GT + 可控振铃」的教学替身。
# >
# > **核心发现**: 即使输入来自 TCForge，loss 仍可能鼓励边缘而伤害温度平滑——问题在 loss 配方，不在数据管线画错。
