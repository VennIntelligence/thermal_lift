# %% [markdown]
# ## 总结：两条线，别混在一起
#
# | 线索 | 看什么 | 本 Notebook 对应章节 |
# |------|--------|---------------------|
# | **训练输入线** | TCForge 几何、burst、融合、compact 存盘、patch | Part A (图 00–07) |
# | **Loss 监督线** | mse / highpass / edge / ssim | Part B (图 08–16) |
#
# ### 调参决策卡
#
# | 组件 | 温度场优先建议 |
# |------|----------------|
# | mse | **增强 0.5~1.0** |
# | highpass | **删或 ≤0.1** |
# | edge | **删或 ≤0.05** |
# | ssim | 保留 0.1~0.2 |
#
# ### 和 EP11 的关系
#
# - EP11 在 **真实 248 帧** 上跑 UNet checkpoint；本 EP13 在 **TCForge 合成场景** 上解释管线。
# - EP11 的 highpass 对比图 = 评估域；EP13 说明该操作也出现在 **训练输入 ch4** 与 **loss highpass** 两处。
# - 验收顺序：**temperature ROI → highpass → proxy 指标**。


# %% [markdown]
# > **核心发现**: Loss Atlas 现在同时回答两个问题——「训练吃什么数据」和「loss 怎么算」。改配方时，回到 Part A 确认输入没理解错，再回到 Part B 决定删哪项 loss。
