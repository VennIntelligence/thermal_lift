# %% [markdown]
# ## 3. Main Highpass Results
#
# 主轨使用 per-frame highpass 后的结构图作为输入，输出仍是结构图。这里不再展示整幅缩略图，而是固定芯片中心做约 3x 视觉放大，对比 LR、bicubic、SAA、IBP 和 MAP-TV。

# %%
display(show_png("comparison_fullview.png"))

# %% [markdown]
# > **图表说明**: `comparison_fullview.png` 是 EP06 的主视觉证据，横向比较 LR reference、bicubic、SAA、IBP、MAP-TV 等 2x 输出。图像已经裁到芯片中心并做约 3x 视觉放大，便于直接检查中心针脚/折线区域；这里显示的是 per-frame highpass 后的结构图，不是普通温度图。
# >
# > **怎么看**: 白色附近通常表示局部变化接近 0，也就是没有明显边缘响应；红色和蓝色是相对局部背景的正/负响应，常以成对边缘的形式出现在结构边界两侧。评价时要看轮廓是否更连续、边界位置是否稳定、内部形状是否更容易辨认，而不是只看颜色是否更强烈。
# >
# > **正常/异常**: Highpass 图会天然压掉大面积慢变温度背景，因此“大片发白”是正常现象，不代表缺数据。相反，如果某个方法出现成片棋盘纹、重复条纹、边缘周围过强红蓝振铃，或只把噪声也一起放大，就要把它视为伪影风险。
# >
# > **核心发现**: 这张中心放大图只能支持 contour-level 的可见性判断：2x grid 表示输出采样网格更密，不等价于已经证明 5 um 计量级空间分辨率。后续更高倍率 ROI、raw-temperature 中心图和指标表用于约束这个视觉判断是否稳定。
