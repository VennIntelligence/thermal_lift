# %% [markdown]
# ## 1. Raster Acquisition Path
#
# 主 session 必须按 `acquisition_order` 解释。文件名里的 `X_Y_R` 是坐标标识，不是时间线。
# 判断“相邻帧”必须使用 `acquisition_order`：只有采集顺序相邻，才有资格被当作短时间位移诊断；只是在坐标上相邻，仍可能隔了很多帧。

# %%
from thermal_core.ep02 import raster_summary

show_fig("ep02_raster_acquisition_path.png")

# %% [markdown]
# Figure 1: Raster acquisition path. Command coordinates are plotted against physical acquisition order.

# %% [markdown]
# ### 📈 光栅扫描时序轨迹诊断
#
# 光栅扫描（Raster Scan）轨迹的时序可视化展示了主 Session 内 $X$ 坐标和 $Y$ 坐标随物理采集顺序（时间轴）的演化规律：
# 1. **物理运动规律**：在物理采集过程中，$X$ 坐标呈现锯齿状的连续递增与复位，而 $Y$ 坐标呈阶梯式跃迁，这符合标准的逐行步进停拍（Step-and-Shoot）物理运动模式。
# 2. **时序相邻性差异**：行内相邻的 $X$ 坐标点（采集时序差 $\Delta t_{\text{index}} = 1$）代表真正意义上的时序连续帧；而在物理空间中沿 $Y$ 轴相邻但 $X$ 固定的帧，其在采集序列中通常相隔整行扫描时间（$\Delta t_{\text{index}} \approx 16$ 帧）。
#
# **💡 算法决策**：由于物理热场演化和环境漂移会随时间累积，沿 $Y$ 轴空间相邻但采集时间差较大的帧对无法用于高精度的定量位移标定。

# %%
raster_stats = raster_summary(frame_audit)
display(raster_stats)

# %% [markdown]
# ### 📊 扫描轨迹节点统计
#
# 统计结果表明，主 Session（共 255 帧）主要由 $R=0$ 的基准光栅网格构成。行内 $X$ 轴转移的数量显著多于行间换行转移。
#
# **💡 算法决策**：在后续的位移评估与对齐诊断中，必须严格区分“物理空间相邻”与“时间序列相邻”两种关系。任何基于相邻帧对的位移估计，均应优先基于时序连续的行内 $X$ 轴转移进行，以最小化时间热场漂移对互相关计算的污染。
