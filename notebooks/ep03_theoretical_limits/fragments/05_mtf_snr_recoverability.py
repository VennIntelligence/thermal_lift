# %% [markdown]
# ## Step 4 — MTF x SNR 可恢复性风险图
#
# 前两节分别看了 PSF/MTF 和局部温差/SNR。这里把它们合成一个必要条件：
#
# `effective_snr = DeltaT * MTF(f, sigma) / noise`
#
# 这个数值回答“某个局部温差，在某个输出网格 Nyquist 频率和某个 PSF 假设下，还剩多少噪声倍数”。它是风险图，不是 SR 成功证明；即使 effective SNR 很高，真实重建仍可能被对齐误差、热漂移、非理想 PSF 或结构不一致性破坏。

# %%
recoverability_contrasts = snr_reference[snr_reference["label"].ne("Noise floor")].copy()
recoverability_table = build_mtf_snr_recoverability_table(
    recoverability_contrasts,
    noise_sigma_c=NOISE_SIGMA,
    sigmas_px=PSF_SIGMAS,
    grid_factors=(1, 2, 4),
    detector_pitch_um=DETECTOR_PITCH_UM,
)
recoverability_table.to_csv(OUTPUT_DIR / "mtf_snr_recoverability.csv", index=False)

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

recoverability_gate_summary = (
    recoverability_table.groupby(["grid_label", "sigma_psf_px"], as_index=False)
    .agg(
        min_effective_snr=("effective_snr", "min"),
        median_effective_snr=("effective_snr", "median"),
        max_effective_snr=("effective_snr", "max"),
        pass_3x_fraction=("passes_3x_noise", "mean"),
        pass_5x_fraction=("passes_5x_noise", "mean"),
    )
)
recoverability_gate_summary.to_csv(OUTPUT_DIR / "mtf_snr_recoverability_gate_summary.csv", index=False)

display(recoverability_display)
display(recoverability_gate_summary)

# %% [markdown]
# > **数据说明**: 第一张表按 contrast level、grid Nyquist 和 PSF sigma 列出 effective SNR；第二张表按 grid/sigma 汇总有多少 contrast level 仍高于 3x 或 5x 噪声门槛。这里的 `inner/outer median edge` 来自参考帧的局部 contour 统计，其余行是固定参考温差。
# > **怎么读**: 数值越大，表示该频率处的剩余温差越不容易被噪声淹没；3x 噪声是基本可见门槛，5x 噪声更适合作为稳健候选。沿着 1x -> 2x -> 4x 或 sigma 变大，MTF 衰减会让 effective SNR 快速下降。
# > **正常/异常理解**: 如果某格 effective SNR 低于 1，说明该频率成分大概率落进噪声主导区；如果高于 5，也只说明“信号幅度有机会被看见”，不代表 alignment 已正确，也不代表 SR 输出形状一定更真实。
# > **核心发现**: 2x contour-level POC 应优先依赖高对比、可门控的局部结构；4x Nyquist 在保守 PSF 下很容易进入噪声主导区，因此不能从这张表推出 4x 可行性声明。MTF x SNR 是必要条件/风险图，不是 SR 成功证明。

# %%
fig = plot_mtf_snr_recoverability_heatmap(recoverability_table)
save_fig(fig, "mtf_snr_recoverability_heatmap.png")

# %% [markdown]
# > **图表说明**: 热力图把上表的 effective SNR 画成 `log10 effective SNR`，每个格子里的数字仍是线性 SNR 值。行表示局部温差尺度，列表示输出网格 Nyquist 频率与 Gaussian PSF sigma 的组合。
# > **怎么读**: 颜色越亮、数字越大，表示该 contrast 在该频率/PSF 假设下越容易高于噪声；颜色变暗并不等于 SR 必然失败，而是提示该声明需要更强的局部证据、正则化和质量门控。
# > **正常/异常理解**: 2x 在乐观 sigma 下仍可保留部分高对比 contour 信息，但在 sigma=0.5 px 下只适合保守解读；4x 在大多数保守组合中数值贴近或低于噪声，不应作为默认目标。
# > **核心发现**: EP03 的正确结论是“把 2x 作为 contour-level POC，并对高频声明做风险标注”。它不证明 SR 已成功，也不把 5 um output sample 解释成 5 um spatial resolution。
