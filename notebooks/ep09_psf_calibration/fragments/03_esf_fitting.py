# %% [markdown]
# ## Route B — 1D ESF 拟合
#
# Route B 在 EP04 外轮廓 anchor 上沿法线提取 1D 温度剖面，拟合 Gaussian-convolved step function。筛选条件是外轮廓、温差大于 2.0°C、法线投影大于 0.5、拟合 R² 大于 0.95。

# %%
show_fig("esf_sigma_histogram.png")

# %% [markdown]
# Figure 2: ESF sigma distribution. Quality-gated edge-spread fits summarize apparent boundary widths across contour segments.

# %%
esf_fits = read_csv("esf_sigma_distribution.csv")
if not esf_fits.empty:
    display(
        esf_fits[
            [
                "segment_id",
                "normal_projection",
                "abs_delta_t_c",
                "sigma_lr_px",
                "sigma_std_err_lr_px",
                "r2",
                "rmse_c",
                "valid",
            ]
        ]
        .sort_values("sigma_lr_px")
        .round(4)
        .head(12)
    )
    display(esf_fits[esf_fits["valid"]]["sigma_lr_px"].describe().to_frame("valid_sigma_lr_px").round(4))

# %% [markdown]
# > **图表说明**: 直方图展示所有通过质量门控的 ESF sigma。表格列出较小 sigma 的 segment，便于检查低端是否接近 Route A/C。
# >
# > **怎么看**: ESF sigma 越大表示边缘过渡越宽。R² 越高只表示 error-function 模型拟合这条边缘很好，不自动证明该宽度完全来自光学 PSF。
# >
# > **异常是否正常**: 本次 32/32 个候选 segment 通过 R² 门控，中位 sigma 约 1.145 px，远大于 Route A/C。这种冲突更像是 ESF 包含真实热边缘宽度、边缘倾斜、材料热扩散或单帧采样影响。
# >
# > **核心发现**: Route B 是强烈的反证：外轮廓 edge-spread 宽度不能被忽略，但它不能直接替代 forward-model effective sigma。
