# %% [markdown]
# ## Center 3x Visual Comparison
#
# 这一节展示 EP11 的核心交付图：中心 1/3 ROI 经过 3x 显示放大后，UNet@EP07 step40000 和 EP10 TGV best 在同一 highpass 域、同一 colormap 范围下并排比较。第二张图只显示 UNet 自身的普通温度域 sanity view，不把 raw-mean 上采样当作算法对比。

# %%
show_fig("unet_vs_tgv_2x_center_zoom3x_highpass.png")

# %% [markdown]
# Figure 1: Same-domain highpass comparison for the center-third ROI, displayed at 3x zoom.

# %% [markdown]
# > **图表说明**: 这张图只比较 highpass response，也就是从温度图中减去 `sigma=5.0` Gaussian 背景后保留下来的局部结构响应。UNet@40000 的温度输出先转换到同样 highpass 域；TGV best 直接读取 EP10 已保存的 highpass 产物。
# >
# > **怎么看**: 红/蓝表示相对局部背景的正/负结构响应，白色通常接近零变化；直角边缘、内部轮廓和重复出现的结构线如果更连续、更稳定，说明 contour-level 可视化更有价值。两 panel 共用同一个 99th-percentile symmetric 色阶，所以颜色强弱可以横向比较。
# >
# > **异常是否正常**: highpass 图会强化边缘，也会放大噪声、ringing 和假边缘；红蓝边不等同于普通温度热点/冷点，白色也不表示缺数据。局部更“锐”不能单独证明 SR 成功.
# >
# > **核心发现**: EP11 的首要判断应来自这张图：UNet 是否比 TGV 更清楚地呈现中心区域内轮廓，同时有没有更明显的振铃、棋盘纹或 synthetic-domain 纹理风险。

# %%
manifest = read_json("run_manifest.json")
temp_fig = next(
    Path(p).name for p in manifest["figures"] if "temperature" in p
)
show_fig(temp_fig)

# %% [markdown]
# Figure 2: UNet temperature-domain sanity view for the same center-third ROI, displayed at 3x zoom.

# %% [markdown]
# > **图表说明**: 这张图只显示 UNet 2x 输出的 HR 温度场，用同样中心 ROI 和 3x 显示放大检查普通温度域是否自然。它不是算法对比图。
# >
# > **怎么看**: 普通温度域适合检查中心区域的整体热场、低频背景和内部轮廓是否自然。若 UNet 温度图出现明显块状、过冲或与主热场脱节的结构，highpass 图中的锐边就需要谨慎解释。
# >
# > **异常是否正常**: 这张图不会像 highpass 图那样强化边缘，因此轮廓看起来更柔和是正常的；这里重点是检查是否有不自然纹理、过冲或块状拼接痕迹。
# >
# > **核心发现**: highpass 图负责公平视觉 benchmark，raw-temperature 图负责检查 UNet 输出是否仍像可信热像，而不是只产生边缘增强纹理。
