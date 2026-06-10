# %% [markdown]
# ## Notes and Decision Boundary

# %%
notes = read_text("comparison_notes.md")
display(Markdown(notes))

# %% [markdown]
# > **数据说明**: 上面的 notes 是 EP11 脚本在生成图和指标后写出的短结论，记录输入帧数、设备、ROI 设置、proxy 数值和下一步投入建议。
# >
# > **怎么看**: 文字结论应和 Figure 1 一起读：谁的内结构/直角边缘更清楚，谁有更明显 ringing、假边缘或过平滑，以及 proxy 是否支持继续投入。
# >
# > **异常是否正常**: 如果 notes 里没有给出绝对“赢家”，这符合 EP11 边界。UNet@20000 是 synthetic 预训练 checkpoint，真实数据 domain gap 仍然存在；TGV 是 classical highpass-domain baseline，也不是计量级真值。
# >
# > **核心发现**: EP11 的合理产出是“哪个方向更值得继续投入”的工程判断，而不是 5 um 物理分辨率证明或温度计量精度声明。
