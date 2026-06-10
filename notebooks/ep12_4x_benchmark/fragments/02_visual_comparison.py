# %% [markdown]
# ## Center 3x Visual Comparison
#
# 这一节展示 EP12 4x benchmark 的核心交付图：中心 1/3 ROI 经过 3x 显示放大后，EP12@2000 和裸 drizzle 4x 在同一 highpass 域、同一 colormap 范围下并排比较。第二张图在普通温度域做同样 ROI 的并排 sanity view。

# %%
show_fig("ep12_vs_drizzle_4x_center_zoom3x_highpass.png")

# %% [markdown]
# Figure 1: Same-domain highpass comparison for the center-third ROI, displayed at 3x zoom.

# %% [markdown]
# > **图表说明**: 这张图比较 highpass response，也就是从温度图中减去 `sigma=5.0` Gaussian 背景后保留下来的局部结构响应。左 panel 是 EP12 4x UNet@2000 输出；右 panel 是裸 tcforge scatter-add drizzle mean，两者都在同一 highpass 域。
# >
# > **怎么看**: 红/蓝表示相对局部背景的正/负结构响应，白色通常接近零变化。裸 drizzle 常见的棋盘格/空洞纹理在 highpass 域会表现为重复红蓝颗粒；若 EP12 把这些采样伪影压下去，同时保留更连续的芯片走线和焊盘边缘，则说明 4x contour-level 可视化有增益。
# >
# > **异常是否正常**: highpass 图会强化边缘，也会放大噪声、ringing 和假边缘；红蓝边不等同于普通温度热点/冷点。EP12 背景更平滑、对比度更低，不一定代表“更差”，也可能只是 drizzle 伪影被抑制后的正常表现。
# >
# > **核心发现**: EP12 4x 的首要判断应来自这张图：EP12 是否比裸 drizzle 更清楚地呈现中心区域内轮廓，同时有没有 synthetic-domain 假纹理风险。

# %%
show_fig("ep12_vs_drizzle_4x_center_zoom3x_temperature.png")

# %% [markdown]
# Figure 2: Temperature-domain side-by-side sanity view for the same center-third ROI, displayed at 3x zoom.

# %% [markdown]
# > **图表说明**: 这张图在普通温度域并排显示 EP12@2000 和裸 drizzle 4x 的 HR 温度场，用同样中心 ROI 和 3x 显示放大检查绝对热场结构是否自然。
# >
# > **怎么看**: 裸 drizzle 右侧常见的黄紫棋盘格是稀疏采样网格，不是真实温度细节；EP12 左侧若更平滑、焊盘/走线边界更连续，说明模型在填补 drizzle 空洞。温度域适合检查 EP12 是否只是 highpass 边缘增强，还是内部结构也一起变清楚。
# >
# > **异常是否正常**: 温度图不会像 highpass 图那样强化边缘，因此轮廓看起来更柔和是正常的；重点看裸 drizzle 的网格伪影是否在 EP12 中被抑制。
# >
# > **核心发现**: highpass 图负责公平 contour benchmark，temperature 图负责检查 EP12 4x 是否仍像可信热像，而不是只产生边缘增强纹理。
