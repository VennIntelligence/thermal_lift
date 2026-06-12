# %% [markdown]
# ## S-F10 — V9A fine-window 演化条带（supp D.3.2 配图）
#
# 中心细线「梯子」窗口（2x 网格 rows 384:518, cols 478:674）上的 checkpoint
# 演化：TGV 参照、v8.1a 60K（1x 输入对照）与 V9A 5K→60K 序列。

# %%
figS10 = show_figure(
    PAPER_FIGS / "figS10_v9a_strip.png",
    "uv run python scripts/paper_figures/figS10_v9a_strip.py",
)
figS10

# %% [markdown]
# > **图表说明**: 9 个面板共用 inferno colormap，每面板按自身 1–99 百分位归一化
# > ——经典链与训练链的全局温度偏置不同（median offset correction），绝对温标
# > 跨面板不可比，本图只看相对结构；左下角标注 fine-window 保真（fid =
# > hp_corr_input）与锐度 proxy。
# > **怎么看**: 关注中心两条细线「梯子」：TGV 与 V9A 20K 能部分分辨梯子内部
# > 条纹（fid 0.96/0.974）；30K 起梯子糊成团块而粗 zigzag 反而更锐、对比饱和
# > （fid 跌到 0.906、sharp 冲到 1.15+）——锐度增益与去相关同时发生，即幻觉过冲。
# > **异常是否正常**: v8.1a 60K 与 V9A 60K 视觉相近是定量结论的体现（hybrid
# > 早期增益在 60K 被先验侵蚀抹平）；面板间背景色阶差异来自 per-panel 归一化。
# > **核心发现**: 「保真悬崖」可肉眼定位在 25K→30K 之间；V9A 的交付价值锁定
# > 在 20K 一带的 checkpoint，60K 不可交付。
# > **状态**: ✅ 新落地（学术重绘，替代 tmp 时代的诊断条带）。
