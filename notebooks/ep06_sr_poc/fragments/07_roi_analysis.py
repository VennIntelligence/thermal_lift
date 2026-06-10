# %% [markdown]
# ## 6. ROI Analysis
#
# ROI 图用于检查中心针脚/内部折线区域的局部轮廓、结构分离和算法伪影。三个 ROI 现在都固定在芯片中心，只改变 crop 尺寸，形成“中心全貌 -> 中等放大 -> 更强放大”的读图顺序。

# %%
show_fig("comparison_roi_1.png")

# %% [markdown]
# Figure 14: Center ROI comparison at the widest crop. Highpass structure responses are compared across LR, bicubic, and SR methods.

# %% [markdown]
# > **图表说明**: `comparison_roi_1.png` 是中心区域的较大 crop，对同一中心位置横向展示 LR、bicubic 和多种 SR 输出。ROI 图放大的是 highpass 结构响应，因此主要看中心边缘和局部轮廓。
# >
# > **怎么看**: 关注同一条轮廓在不同方法里是否更连续、是否断裂减少、是否保持在相同物理位置。Bicubic 只是把单帧放大，边缘会更平滑但不会提供新的多帧信息；SAA/IBP/MAP-TV 如果有效，应表现为结构更可分且位置不乱跳。
# >
# > **正常/异常**: Highpass ROI 里白色背景表示局部变化接近零，红蓝边表示相对背景的正负响应，不是温度本身变成红蓝。异常情况包括边缘旁边出现过强双边振铃、背景出现网格状纹理，或某方法把本来连续的结构打碎。
# >
# > **核心发现**: ROI 1 用来建立中心区域的上下文，确认主图里看到的 contour 增益是否在芯片中心仍然成立；单个 ROI 不能独立证明 SR 成功，但可以暴露缩略图看不出的伪影。

# %%
show_fig("comparison_roi_2.png")

# %% [markdown]
# Figure 15: Center ROI comparison at medium magnification. The same chip-center region is inspected with a tighter highpass crop.

# %% [markdown]
# > **图表说明**: `comparison_roi_2.png` 是同一芯片中心的中等放大窗口，保持与 ROI 1 相同的列顺序和 highpass 解释方式。这里不是换到别的边角，而是在中心继续放大。
# >
# > **怎么看**: 重点看中心针脚/折线之间的间隔是否比 ROI 1 更容易分辨。若某个方法真实提升了 contour-level 可见性，放大后应仍然保持边缘连续、位置稳定；如果只是红蓝边变粗或重复出现，则更像过锐化。
# >
# > **正常/异常**: 有些 ROI 天然结构弱、热对比低，因此 SR 增益可能不如强边缘区域明显，这是正常的。需要警惕的是算法在弱结构区域“凭空造边”，例如出现与 LR/bicubic/其他方法都不一致的孤立红蓝条。
# >
# > **核心发现**: ROI 2 用于检查中心结构在更高放大下是否仍然可信。它帮助区分“中心内部结构更容易看”与“某些边被锐化得更显眼”。

# %%
show_fig("comparison_roi_3.png")

# %% [markdown]
# Figure 16: Center ROI comparison at the tightest crop. The most magnified highpass view checks fine contour continuity and artifacts.

# %% [markdown]
# > **图表说明**: `comparison_roi_3.png` 是同一芯片中心的最强放大 crop，继续用同一 highpass 色标比较多个方法。它专门服务于“中心针脚/内部细分隔能不能分开”的问题。
# >
# > **怎么看**: 这里同样看结构位置、边缘连续性和内部轮廓是否更清楚。若 MAP-TV 或 IBP 的正则/迭代让边缘更干净，但同时保留了与 LR reference 一致的几何位置，这是较好的信号；若只是边缘变粗或出现重复边，则不能当作可靠分辨率增益。
# >
# > **正常/异常**: ROI 图是 highpass 结构图，所以不能用颜色直接读温度高低，也不能把红蓝对当作两个真实物体。真正异常的是不同方法给出的结构位置互相矛盾，或某方法产生明显条纹、块状拼接、局部过冲。
# >
# > **核心发现**: 三个 ROI 的结论应按放大顺序合并阅读：可接受的 EP06 结论需要在中心区域越放大越能看清真实结构，同时不和 raw 控制轨、split-half 和 artifact audit 冲突。

# %% [markdown]
# ## 6.1 Center Raw-Temperature Visual Check
#
# 前面的 ROI 是 highpass 结构图，适合看边缘是否变锐，但不适合直接判断中心针脚/内部块状区域是不是在普通温度图里真的分开。这里改用 raw-temperature 控制轨的中心 crop，使用普通温度色标检查中心结构。

# %%
show_fig("comparison_center_raw_temperature.png")

# %% [markdown]
# Figure 17: Center raw-temperature comparison at 2x. Offset-corrected temperature reconstructions are shown without highpass coloring.

# %% [markdown]
# > **图表说明**: `comparison_center_raw_temperature.png` 显示中心区域的 offset-corrected raw-temperature 重建，不做 highpass，也不使用红蓝差分色标。左侧加入 `LR raw reference` 和 `Bicubic raw reference`，后面才是多帧 SR 方法，因此可以直接比较原始 raw、普通插值和 SR 输出在中心针脚区域的差别。
# >
# > **怎么看**: 重点看中心针脚/内部轮廓之间的分隔是否变清楚：暗缝是否更连续，亮块边界是否更稳定，相邻结构是否更容易分开。这里不是看颜色谁更鲜艳，而是看同一物理结构的边界是否在不同方法中更明确、更一致。
# >
# > **正常/异常**: 普通温度图保留慢变热背景，所以不会像 highpass 图那样大面积发白，边缘也不会天然出现红蓝对响应。若中心区域在 highpass 图里边缘很强，但在 raw-temperature 中仍然糊成一片，说明当前方法可能主要增强了外轮廓或强边缘，内部细分隔还没有被可靠恢复。
# >
# > **核心发现**: 这张新图是 EP06 判断中心结构的关键补充：它把“边缘响应更强”转化为更贴近使用场景的问题，即中心针脚和内部轮廓在普通视觉上是否更可分。最终结论应同时引用 highpass ROI 和这张 raw-temperature 中心检查。

# %% [markdown]
# ## 6.2 Center Raw-Temperature 4X SR Reconstruct & Contrast
#
# 4X 尺度下的普通温度图中心裁剪对比由 `scripts/build_ep06_cache.py` 预生成（SAA/IBP/MAP-TV 快速 4x 迭代）。Notebook 只展示缓存 PNG，不在 cell 内 subprocess 重建。

# %%
show_fig("comparison_center_raw_temperature.png", subdir="4x")

# %% [markdown]
# Figure 18: Center raw-temperature comparison at 4x. The same physical center crop is displayed on a denser output grid for visualization only.

# %% [markdown]
# > **图表说明**: 4X 版本的 `comparison_center_raw_temperature.png` 展示了上采样网格扩大为 4 倍（HR 尺寸 1920x2560）时的芯片中心局部区域。
# >
# > **对比分析**: 可以与上方的 2X 版本进行直观对比。4X 的重构在像素颗粒感上比 2X 更加细腻，但因为 4X 下亚像素对齐的病态程度剧增，使得它在插值和去卷积过程中对位移先验偏差和噪声的敏感度更高，部分物理结构的边缘容易产生振铃或柔和的模糊。
