# %% [markdown]
# ## S-F15 — MTF 频响与有效 SNR 可恢复性边界（supp A.1 配图）
#
# 「2x 条件可行 / 4x 出界」claim boundary 的两张理论图：Gaussian PSF MTF
# 频响曲线与（对比度 × 网格 × σ）的有效 SNR 热图。

# %%
REBUILD_S = (
    "uv run python scripts/build_ep03_cache.py → EP03 notebook → "
    "uv run python scripts/paper_figures/collect_promoted_supp.py"
)
display(Markdown("**figS15a — 有效 SNR 可恢复性热图（log10 SNR_eff）**"))
display(show_figure(PAPER_FIGS / "figS15a_mtf_snr_heatmap.png", REBUILD_S))
display(Markdown("**figS15b — Gaussian PSF MTF 频响曲线**"))
display(show_figure(PAPER_FIGS / "figS15b_mtf_frequency_response.png", REBUILD_S))

# %% [markdown]
# > **图表说明**: (a) 行=六档局部对比度（0.217–2.49 °C，含实测内/外轮廓中位），
# > 列=输出网格 × 假设 PSF σ，格内数字为 SNR_eff = ΔT·MTF/σ_n；
# > (b) σ∈{0.2, 0.35, 0.5} 的 MTF(f) 曲线与 1x/2x/4x Nyquist 位置。
# > **怎么看**: (a) 中 2x 列在实测对比度行（1.94/2.49 °C）保持 SNR_eff
# > 0.19–15.6（σ 依赖，条件可行），4x 列除 σ=0.2 外全部 ≪1（noise-dominated）；
# > (b) 给出该判决的频域机制（2x Nyquist 处 MTF 0.454→0.007）。
# > **异常是否正常**: 热图采用 log10 色标，1e-8 量级格为理论值非测量值；
# > 判据是**必要条件**，过门不等于 SR 成功（supp A.1.5 边界声明）。
# > **核心发现**: 交付网格定在 2x、4x 仅作 contour oversampling 的决策由
# > 第一性原理先行设定，并被 EP12 4x 负结果实证互证（supp D.4.2）。
# > **状态**: ✅ 选编收编（06-12）。
