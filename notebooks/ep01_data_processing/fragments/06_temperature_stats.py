# %% [markdown]
# ## 3.1 温度统计与主 session 初步建模

# %%
model = cache.model
main_session = int(model.main_session)

show_fig("frame_temperature_statistics.png")

# %% [markdown]
# Figure 2: Frame temperature statistics. The empirical distribution separates warm-up frames from the stable main session.

# %% [markdown]
# > **图表解读**：累积分布函数（ECDF）显示，绝大部分帧的均温高度聚集在 24°C 左右（主扫描段），而开机前期的少数帧则分布在 20°C 的低温段。


# %%
show_fig("robust_temperature_timeline.png")

# %% [markdown]
# Figure 3: Robust temperature timeline. Median frame temperature is plotted against physical acquisition order.

# %% [markdown]
# ### 📈 探测器热平衡演化与温漂诊断
#
# 随采集顺序推移，探测器在开机后经历了一个快速升温并趋于热平衡的过程：
# 1. **开机过渡段**（前 8 帧）：温升达 4°C（近 50 倍噪声底），处于非平衡状态。
# 2. **主扫描段**（第 9 帧开始）：温度波动标准差低于 $0.05^\circ\text{C}$，已与探测器噪声底（$0.0724^\circ\text{C}$）处于同一量级，处于理想的热力学稳定状态。
#
# **💡 算法决策**：超分辨率重建仅能使用处于热平衡的主扫描帧。若混入开机过渡帧，大范围的温漂将被算法误判为空间细节，导致亚像素对齐失效。
