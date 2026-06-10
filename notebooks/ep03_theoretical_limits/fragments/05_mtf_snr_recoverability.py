# %% [markdown]
# ## Step 4 — MTF x SNR 可恢复性风险图
#
# 前两节分别看了 PSF/MTF 和局部温差/SNR。这里把它们合成一个必要条件：
#
# `effective_snr = DeltaT * MTF(f, sigma) / noise`
#
# 这个数值回答“某个局部温差，在某个输出网格 Nyquist 频率和某个 PSF 假设下，还剩多少噪声倍数”。它是风险图，不是 SR 成功证明；即使 effective SNR 很高，真实重建仍可能被对齐误差、热漂移、非理想 PSF 或结构不一致性破坏。

# %%
recoverability_table = cache.recoverability_table
recoverability_gate_summary = cache.recoverability_gate_summary

recoverability_display = recoverability_table.pivot_table(
    index=["contrast_label", "delta_t_c", "input_snr"],
    columns=["grid_label", "sigma_psf_px"],
    values="effective_snr",
    aggfunc="first",
)
recoverability_display.columns = [
    f"{grid_label} sigma={sigma_px:.2f}" for grid_label, sigma_px in recoverability_display.columns
]
recoverability_display = recoverability_display.reset_index()

display(recoverability_display)
display(recoverability_gate_summary)

# %% [markdown]
# ### 📈 MTF 与信噪比的综合可恢复性评估
#
# 定量分析了在不同局部温差对比度（Delta T）及不同光学低通退化程度（$\sigma_{\text{PSF}}$）下，超分辨率输出网格截止频率处对应的有效信噪比（Effective SNR）。
# 1. **有效信噪比计算机制**：
#    $$ \text{SNR}_{\text{effective}}(f) = \frac{\Delta T \cdot \text{MTF}(f, \sigma_{\text{PSF}})}{\sigma_n} $$
#    此指标衡量了物理结构在穿过光学系统和探测器采样退化后，残存的高频信号强度是否能高于噪声底。
# 2. **可恢复性极限**：在 2x 超分辨率网格奈奎斯特频率处，内轮廓的中位温差对比度仍能维持高于 $3\sigma_n$ 或 $5\sigma_n$ 的有效信噪比，证明其高频分量未被噪声完全淹没。但在 4x 奈奎斯特频率处，受 MTF 指数级衰减压制，几乎所有对比度级别的有效信噪比均跌破 1.0。
#
# **💡 算法决策**：超分辨率算法将优先依赖高对比度段提供的信号以稳定重建。由于 4x 高频信号在物理上落入噪声主导区，本系统将重建倍率锁定在 2x，以此在噪声敏感度与轮廓恢复能力之间取得物理平衡。

# %%
show_fig("mtf_snr_recoverability_heatmap.png")

# %% [markdown]
# Figure 7: MTF and SNR recoverability heatmap. Effective SNR is mapped across contrast levels, PSF widths, and reconstruction grids.

# %% [markdown]
# ### 🗺️ 截止频率可恢复性有效信噪比热力图分析
#
# 热力图展示了不同温差对比度与重建频率组合下的 $\log_{10}(\text{SNR}_{\text{effective}})$ 能量分布情况。
# 1. **频域能量衰减可视**：热力图清晰呈现了沿低频向高频（1x 到 2x 到 4x）推移时，高频信号的剧烈衰减。
# 2. **参数边界标定**：当 $\sigma_{\text{PSF}} = 0.5$ 像素且温差对比度低于 $0.3^\circ\text{C}$ 时，有效信噪比已接近噪声底。
#
# **💡 算法决策**：该热力图构成了评估超分辨率重建可行性的理论边界判定矩阵。在后续的亚像素配准（EP04）及 2x 超分辨率验证（EP06）中，应优先选择在热力图中落在有效信噪比大于 $3.0$（即亮色区域）的图像特征或感兴趣区域（ROI），规避在物理上已被截止的无信号高频成分。
