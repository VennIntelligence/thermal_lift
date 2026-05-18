# %% [markdown]
# ## 4. Raw-Temperature Control Track
#
# Raw track 对原始温度帧只做 per-frame offset correction，再重建；可视化时对输出做 highpass 以便和主轨对照。这里同样使用芯片中心约 3x 放大，检查中心结构是否不是 highpass 输入人为制造出来的。

# %%
display(show_png("comparison_control_track.png"))

# %% [markdown]
# > **图表说明**: `comparison_control_track.png` 把 highpass-input 主轨和 raw-temperature 控制轨放在一起，并且同样裁到芯片中心约 3x 放大。上排来自先 highpass 再重建的主流程；下排先在原始温度帧上做 offset correction 和重建，再在输出端 highpass，目的是检查中心结构是否依赖某个预处理步骤。
# >
# > **怎么看**: 两个轨道使用相同算法和同一套 EP05 对齐约定，主要差别是输入是否先做 highpass。如果真实结构在两个轨道中位置一致，说明它更可能来自数据中的稳定轮廓；如果只在 highpass 主轨出现，而 raw 控制轨没有对应响应，就要谨慎。
# >
# > **正常/异常**: 控制轨的对比度可能弱一些，这是正常的，因为 raw-temperature 里仍保留了慢变热背景。异常情况是主轨出现很漂亮的红蓝边，但控制轨同一区域完全没有对应结构，或边缘位置明显漂移；这提示 highpass 预处理可能放大了局部噪声或热场漂移。
# >
# > **核心发现**: 这张图不是为了证明 raw track 本身就是最佳 SR，而是为了防止把预处理制造出来的边缘误判为芯片内部结构。EP06 的结论只有在主轨视觉增益和控制轨不矛盾时才应写得更强。
