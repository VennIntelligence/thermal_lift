# %% [markdown]
# ## 3. Small-Step Diagnostics as Smoke Tests
#
# 小步 NCC 只回答局部问题：行内 X 时间相邻帧是否有稳定方向和短时线性；Y 坐标相邻帧为什么不能用于定量标定。它不回答“多帧 contour-level SR 是否可行”。
#
# 本节的关键是“诊断”二字。NCC 在局部高通图像上寻找最相似的位移投影，因此它能提示某些边缘/纹理是否随 stage command 方向移动。
# 但红外热像会随时间漂移，局部热结构也可能变化；当 acquisition gap 变大时，NCC 峰值混入的就不只是几何位移。
# 所以这里不会把小步 NCC 或 ESF 局部响应外推为全局 SR 成功/失败结论。

# %%
from thermal_core.ep02 import small_step_metrics

show_fig("ep02_small_step_smoke_tests.png")

# %% [markdown]
# Figure 3: Small-step displacement diagnostics. Local NCC responses compare time-adjacent X steps with delayed Y-adjacent pairs.

# %% [markdown]
# ### 🔍 局部位移响应与时序稳定性诊断
#
# 局部位移互相关（NCC）投影分析主要用于验证步进指令在探测器空间内的方向响应性及线性特征：
# 1. **时序相邻帧对的有效性**：$X$ 轴相邻帧对的物理采集间隔中位数仅为 $1$（即时序连续帧），这为排除热场演化干扰、评估短时位移线性响应提供了理想的基准。
# 2. **方向与线性相应性**：在 $X$ 轴方向上，随着名义位移从 $2\,\mu\text{m}$ 增至 $4\,\mu\text{m}$，基于高通滤波后图像提取的可见 NCC 投影量呈现出合理的单调递增规律（比值接近 2.0），验证了运动阶段方向的局部稳定性。
# 3. **光栅扫描时序污染**：相反，物理空间中 $Y$ 轴相邻帧对（位移 nominal prior 分别为 $2\,\mu\text{m}$ 和 $4\,\mu\text{m}$）的采集时间间隔中位数高达 16 帧，受温漂与背景演化干扰严重，导致其 $4\,\mu\text{m}$ 与 $2\,\mu\text{m}$ 投影比值严重偏离线性（甚至低于 1.0），违反了物理位移的单调性规律。
#
# **💡 算法决策**：鉴于 $Y$ 轴相邻帧对存在严重的时序滞后污染，不能将其用于定量的位移标定。后续的多帧超分辨率几何配准必须依赖基于主 Session 的全局或区域数据驱动对齐（如 EP04 localization 门控），而非仅从小步长相邻帧对的局部 NCC 结果进行外推。

# %%
small_metrics = small_step_metrics(OUTPUT_DIR)
display(small_metrics)

# %% [markdown]
# ### 📊 小步长位移定量响应特征
#
# 定量指标表明，$X$ 轴方向上互相关估计的局部分量虽然受到物理孔径、点扩散函数（PSF）平滑及热背景起伏的压制（导致绝对投影值略小于名义先验），但仍保留了良好的比例单调性。而 $Y$ 轴方向因长采集间隙受到噪声和物理热场畸变的严重干扰，其尺度比例呈现非物理的衰减。
#
# **💡 算法决策**：此定量结果支撑了将 $X$ 轴时序连续帧用作局部对齐“哨兵帧”（Diagnostic Pair）的决策，同时指明了 $Y$ 轴空间相邻帧在无时序对齐约束下的失效机制。在超分辨率重构中，名义位移先验应作为非凸优化的初始迭代种子或正则项，并通过数据本身的结构特征自适应校正。
