# %% [markdown]
# ## Step 2 — PSF/MTF 高频边界
#
# Gaussian PSF 会按空间频率衰减信号。这里扫描 `sigma=0.2/0.35/0.5 px`，并在 1x、2x、4x grid 的 Nyquist 频率处取样。这个图回答“2x 为什么是合理 POC，4x 为什么风险更高”。
#
# 直观理解：芯片内部的细线条和边界可以看成不同空间频率的组合。频率越高，代表结构越细。MTF 越低，说明这些细结构的对比度越容易被光学模糊和噪声淹没。这里不追求精确反演真实 PSF，只用一组合理假设给 SR 目标划风险边界。

# %%
mtf_table = cache.mtf_table

mtf_display = (
    mtf_table.pivot_table(
        index=["grid_label", "grid_pitch_um", "nyquist_cyc_per_detector_px"],
        columns="sigma_psf_px",
        values="mtf_amplitude",
    )
    .reset_index()
    .rename_axis(None, axis=1)
)
display(mtf_display)

show_fig("mtf_psf_frequency_response.png")

# %% [markdown]
# Figure 1: PSF and MTF frequency response. Gaussian blur assumptions are evaluated at 1x, 2x, and 4x Nyquist limits.

# %% [markdown]
# ### 📈 点扩散函数与调制传递函数频响特征评估
#
# 定量分析了不同点扩散函数宽度（$\sigma_{\text{PSF}} = 0.2, 0.35, 0.5$ 像素）在 1x、2x、4x 超分辨率采样网格对应的奈奎斯特极限频率下的调制传递函数（MTF）响应幅度。
# 1. **高频信号衰减规律**：调制传递函数曲线随空间频率的提高呈指数级低通衰减。当空间频率达到 2x 重建网格的奈奎斯特极限频率（$0.1\,\mu\text{m}^{-1}$）时，MTF 在 $\sigma_{\text{PSF}} = 0.35$ 像素的典型光学模糊下仍保留了约 $10\%\text{--}30\%$ 的响应度，具备携带高频可恢复信息的物理基础。
# 2. **更高倍率局限性**：在 4x 重建网格的奈奎斯特极限频率处（$0.2\,\mu\text{m}^{-1}$），MTF 已趋近于零（衰减至 $1\%$ 以下），此时空间细节完全被系统截止频率抹除，淹没于探测器热噪声中。
#
# **💡 算法决策**：频响特征评估为超分辨率重构的目标设定了清晰的物理限制边界。本算法框架聚焦于 2x contour-level 重建（物理分辨率约 $10\,\mu\text{m}$），以提高芯片内部结构/形状的轮廓可见性为核心。这在 MTF 能量传递上是可行且稳健的；而直接重构 4x 倍率在物理上面临极高的信噪比惩罚，不作为本阶段的交付目标。
