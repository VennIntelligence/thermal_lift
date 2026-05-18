# %% [markdown]
# ## Step 2 — PSF/MTF 高频边界
#
# Gaussian PSF 会按空间频率衰减信号。这里扫描 `sigma=0.2/0.35/0.5 px`，并在 1x、2x、4x grid 的 Nyquist 频率处取样。这个图回答“2x 为什么是合理 POC，4x 为什么风险更高”。
#
# 直观理解：芯片内部的细线条和边界可以看成不同空间频率的组合。频率越高，代表结构越细。MTF 越低，说明这些细结构的对比度越容易被光学模糊和噪声淹没。这里不追求精确反演真实 PSF，只用一组合理假设给 SR 目标划风险边界。

# %%
mtf_table = build_mtf_attenuation_table(
    detector_pitch_um=DETECTOR_PITCH_UM,
    sigmas_px=PSF_SIGMAS,
    grid_factors=(1, 2, 4),
)
mtf_table.to_csv(OUTPUT_DIR / "mtf_psf_attenuation.csv", index=False)

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

fig = plot_mtf_psf_curves(mtf_table, sigmas_px=PSF_SIGMAS)
save_fig(fig, "mtf_psf_frequency_response.png")

# %% [markdown]
# > **图表说明**: 表格给出 1x、2x、4x Nyquist 频率处的 MTF amplitude；曲线是 Gaussian PSF 的完整 MTF，竖线分别标出这些 grid 的 Nyquist 频率。
# > **怎么读**: 横轴越往右，代表越细的结构；纵轴越低，代表该尺度结构传到探测器时剩余对比度越少。比较不同 `sigma_psf_px` 时，sigma 越大表示光学扩散越宽，曲线下降越快。
# > **正常/异常理解**: 正常判断不是“MTF 低就完全不能做 SR”，而是“越接近高频端，越需要强 SNR、可靠位移相位和更强正则化”。如果 4x Nyquist 处的 MTF 已很低，却仍宣称 4x 或 5 um 计量级分辨率，就属于过度声明。
# > **核心发现**: 2x grid 是“用多帧相位和正则化改善轮廓表达”的先验合理 POC；4x 需要额外 MTF/SNR 和结构一致性证据，不能作为默认交付倍率。MTF 分析只是理论风险图，不能替代主 session 的真实 SR 验证。
