# %% [markdown]
# ## Step 1 — Pixel Pitch、空间分辨率与 2x Grid
#
# `10 um/pixel` 是 TXT 温度矩阵的 detector sampling pitch；`20 um` 是当前系统的空间分辨率；`5 um` 目标意味着输出网格至少要到 2x sampling grid。这里先把三个量画在同一个物理坐标轴上，避免把采样 pitch、空间分辨率和 SR 显示倍率混为一谈。
#
# 对不熟悉红外成像的读者，关键是不要把“采样更密”和“光学真的分辨得更细”混在一起。2x SR 输出网格可以把结果显示在 5 um 间隔上，但这只是重建网格；它必须通过多帧相位、对齐质量和结构一致性证明有新增 contour 信息，不能直接改写系统空间分辨率。

# %%
sampling_resolution = cache.sampling_resolution
grid_nyquist = cache.grid_nyquist

display(sampling_resolution)
display(grid_nyquist)

show_fig("sampling_resolution_distinction.png")

# %% [markdown]
# Figure 2: Sampling and resolution distinction. Detector pitch, optical resolution, and SR output grids are placed on one physical axis.

# %% [markdown]
# ### 📏 探测器像元间距、光学分辨率与超分辨率网格的物理定义
#
# 定量分析了探测器像元间距（Sampling Pitch）、光学系统实际空间分辨率（Spatial Resolution）以及不同超分辨率重构网格（2x, 4x Grid）对应的奈奎斯特频率及其物理极限周期：
# 1. **像元间距与光学分辨率的物理分工**：探测器的像元物理采样间距为 $10.0\,\mu\text{m/pixel}$，而当前光学系统的空间分辨率限制在 $20.0\,\mu\text{m}$。这表明单帧图像在物理上已经由于点扩散函数（PSF）的低通作用发生了空间平滑。
# 2. **2x 重建网格定义**：2x 超分辨率重建输出网格的插值采样间距为 $5.0\,\mu\text{m}$，其奈奎斯特限制极限周期（$\text{Nyquist Period} = 2 \times \text{sampling\_interval}$）为 $10.0\,\mu\text{m}$。
#
# **💡 算法决策**：必须明确区分“重构网格采样率”与“物理空间分辨率”。2x 重构网格提供高密度插值空间以支持轮廓边缘（Contour-level）的亚像素对齐，但并不等同于将物理空间分辨率直接提升至 $5.0\,\mu\text{m}$。所有超分辨率增益必须由多帧采样的相位互补与结构一致性来证明。

# %%
pixel_pitch_summary = cache.pixel_pitch_summary
display(pixel_pitch_summary)
show_fig("pixel_size_measurement.png")

# %% [markdown]
# Figure 3: Pixel size measurement. Independent scale checks support the 10 um per pixel detector pitch.

# %% [markdown]
# ### 📊 物理尺度标定与像元尺寸交叉验证
#
# 汇总了通过坐标轴 mm 标尺与 TXT 温度矩阵图像特征进行像元大小（Detector Pitch）交叉标定的定量结果。分析表明，通过外部微动平台标定和图像轮廓多物理场重合匹配得到的物理像元间距均收敛在 $10.0\,\mu\text{m/pixel}$ 左右。
#
# **💡 算法决策**：像元间距 $10.0\,\mu\text{m/pixel}$ 作为系统几何标定常数，被锁定在后续所有的超分辨率前向成像和逆向求解模型中。光学空间分辨率 $20.0\,\mu\text{m}$ 作为光学传递函数的截止频率约束，在滤波与正则化设计中起到阻尼作用。
