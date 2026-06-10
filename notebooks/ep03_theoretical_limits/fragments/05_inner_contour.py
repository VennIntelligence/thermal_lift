# %% [markdown]
# ## Step 5 — ESF/CRB 定位精度只定义 Anchor 置信度
#
# 1D ESF 的 Cramér-Rao Bound 可以说明“某个局部边缘在给定噪声、温差、PSF 和相位覆盖下，边缘位置能否作为稳定 anchor”。它不能证明整体形状已经被重建，也不能替代 EP06 SR POC 的 contour/shape evidence。
#
# 可以把 CRB 理解成“在模型完全正确、噪声假设成立时，任何无偏估计器也很难优于这个位置误差”。因此它适合做乐观的理论下界，而不是实测性能保证。真实数据里还会有热漂移、配准误差、非理想 PSF、局部纹理混叠等因素。

# %%
crb_table = cache.crb_table

crb_display = crb_table.pivot_table(
    index=["delta_t_c", "sigma_psf_px"],
    columns="model",
    values="crb_px",
).reset_index()
display(crb_display)

nominal_single = float(
    crb_table[
        crb_table["delta_t_c"].eq(0.7)
        & crb_table["sigma_psf_px"].eq(0.5)
        & crb_table["model"].eq("single_frame")
    ]["crb_px"].iloc[0]
)
nominal_multi = float(
    crb_table[
        crb_table["delta_t_c"].eq(0.7)
        & crb_table["sigma_psf_px"].eq(0.5)
        & crb_table["model"].eq("16_frame_known_shift")
    ]["crb_px"].iloc[0]
)
wide_edge_multi = float(
    crb_table[
        crb_table["delta_t_c"].eq(2.0)
        & crb_table["sigma_psf_px"].eq(1.0)
        & crb_table["model"].eq("16_frame_known_shift")
    ]["crb_px"].iloc[0]
)
print(f"Nominal single-frame CRB (DeltaT=0.7 C, sigma=0.5 px): {nominal_single:.4f} px")
print(f"Nominal 16-frame CRB (known 1 px phase coverage): {nominal_multi:.4f} px")
print(f"Wide but strong edge 16-frame CRB (DeltaT=2.0 C, sigma=1.0 px): {wide_edge_multi:.4f} px")

# %% [markdown]
# ### 📐 边缘扩散函数 (ESF) 与克拉美-罗下界 (CRB) 局部定位精度分析
#
# 定量分析了在不同局部温差对比度（$\Delta T$）、边缘物理宽度（$\sigma_{\text{PSF}}$）以及帧数（单帧 vs 16 帧已知相位覆盖）条件下一维边缘扩散函数（ESF）的 Cramér-Rao 理论下界（CRB）。
# 1. **物理定位下界**：CRB 定义了在给定探测器信噪比与几何成像模型下，任何无偏估计器所能达到的最小参数估计方差：
#    $$ \text{CRB}(\theta) = \left[ N \cdot \sum_{i} \frac{1}{\sigma_n^2} \left( \frac{\partial f(x_i; \theta)}{\partial \theta} \right)^2 \right]^{-1} $$
#    对于典型芯片内部轮廓（$\Delta T \approx 0.7^\circ\text{C}$，$\sigma_{\text{PSF}} = 0.5$ 像素），单帧定位精度下界约为 0.1 像素，而引入 16 帧已知空间相位分布的多帧观测后，理论定位精度上限可推进至 0.02 像素。
# 2. **约束边界**：即使物理边缘较宽且模糊（如 $\sigma_{\text{PSF}} = 1.0$ 像素），只要局部热温差对比度足够高（$\Delta T \approx 2.0^\circ\text{C}$），多帧累积依然能够提供低于 0.05 像素的超高理论定位精度。
#
# **💡 算法决策**：ESF/CRB 定量计算为配准算法的选择提供了数学基准。它证实了高对比度内部轮廓在多帧数据融合下具备亚像素级（$<0.05$ 像素）定位潜力。后续配准算法应锁定这部分高置信度区域作为对齐锚点（Alignment Anchor），用于修正全局温漂。

# %%
show_fig("crb_esf_localization_anchor.png")

# %% [markdown]
# Figure 8: ESF localization anchor CRB. Simulated edge profiles connect local contrast with theoretical position precision.

# %% [markdown]
# ### 📈 边缘过渡退化与定位下界仿真验证
#
# 模拟了受高斯白噪声污染的 ESF 边缘剖面及其在不同积分时间（帧数）与物理温差下的理论 CRB 表现。
# 1. **边缘剖面物理解析**：边缘剖面随噪声的扰动呈现波动，表明低信噪比边缘的定位估计具有不确定性。
# 2. **多帧相位效应**：CRB 随物理帧数的增加呈 $\sqrt{N}$ 级别的收敛速度，但其前提是各物理帧之间具备合理的亚像素物理相位覆盖。
#
# **💡 算法决策**：仿真验证为配准流程确立了置信度门限。对于低对比度（$<0.3^\circ\text{C}$）的过渡边缘，由于单帧及少帧下的 CRB 接近甚至超过 0.5 像素，必须在对齐阶段予以排除，防止不稳定的局部估计污染全局位移场。

# %%
crb_sensitivity = cache.crb_sensitivity
crb_gate_summary = cache.crb_gate_summary

crb_nominal_slice = crb_sensitivity[
    crb_sensitivity["delta_t_c"].eq(0.7)
    & crb_sensitivity["sigma_psf_px"].isin([0.35, 0.5])
    & crb_sensitivity["phase_coverage_px"].isin([0.0, 0.5, 1.0])
].pivot_table(
    index=["sigma_psf_px", "phase_coverage_px"],
    columns="n_frames",
    values="crb_px",
    aggfunc="first",
).reset_index()

display(crb_nominal_slice)
display(crb_gate_summary)

# %% [markdown]
# ### 📊 物理帧数与空间相位覆盖的 CRB 灵敏度分析
#
# 定量分析了在固定对比度 $\Delta T = 0.7^\circ\text{C}$ 条件下，边缘宽度、物理帧数以及空间相位覆盖范围（Phase Coverage）对定位下界（CRB）的联合灵敏度响应。
# 1. **相位完整性影响**：当相位覆盖宽度从无（0.0 像素）拓展到完整（1.0 像素）时，多帧估计的几何位置歧义被彻底消除，CRB 显著下降并表现出高鲁棒性。
# 2. **定位精化标准**：统计了实现 0.05 像素与 0.10 像素定位误差门限所需的最低物理温差，表明多帧已知位移约束能大幅降低超分辨率对单帧信噪比的硬性要求。
#
# **💡 算法决策**：灵敏度扫描确立了对齐质量门控（Quality Gate）的数学标准。在 EP04 的全局配准与 EP06 的 SR 重建中，应根据此灵敏度模型，动态设定 0.05 像素（严格）或 0.10 像素（宽松）的几何位置门槛，确保只有通过置信度检验的物理帧参与超分辨率网格的插值运算。

# %%
show_fig("crb_sensitivity_surface.png")

# %% [markdown]
# Figure 9: CRB sensitivity surface. Localization precision is shown as a function of frame count and local temperature contrast.

# %% [markdown]
# ### 🗺️ 多维定位误差极限（CRB）敏感度曲面
#
# 敏感度曲面直观展示了在特定边缘宽度和相位覆盖切片下，定位精度下界随帧数（纵轴）与局部温差（横轴）的变化趋势，并以等值线标出了 0.05 像素与 0.10 像素的物理对齐置信区。
#
# **💡 算法决策**：敏感度曲面被定义为系统级 ROI 选择与算法风险管理的量化图谱。后续重建应优先选取曲面中处于 0.05 像素等值线以内的“安全区”（即高信噪比、多帧采样区）结构作为评估超分辨率效果的黄金区域，对于落入 0.10 像素等值线以外的“高风险区”结构则进行降权处理或引入更强空域正则化约束。
