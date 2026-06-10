# %% [markdown]
# ## Exploratory 4x Drizzle
#
# 这里把 Drizzle 4x 只作为 **contour oversampling / visualization candidate**。4x 输出网格是 1920×2560，但图中只显示中心 1/3 物理区域，等价于约 3× 放大后的中心 crop，避免全图过大和视觉重点分散。

# %%
from IPython.display import Markdown, display

drizzle_4x = artifacts["drizzle_4x"]["sweep"].copy()
if not drizzle_4x.empty:
    display(
        drizzle_4x[
            [
                "pixfrac",
                "scale",
                "split_half_nrmse",
                "holdout_mse",
                "artifact_score",
                "raw_control_corr",
                "coverage_lt1_fraction",
                "coverage_p05",
                "coverage_median",
                "coverage_p95",
            ]
        ].round(6)
    )
else:
    display(Markdown("**4x Drizzle output missing**: run `algos/ep10_drizzle/scripts/run_drizzle.py --scale 4` first."))

# %%
show_optional_fig(
    "drizzle_4x_diagnostics.png",
    "Pending: run `uv run python scripts/build_ep10_cache.py` after 4x sweep_results.csv exists.",
)

# %% [markdown]
# Figure 6: Exploratory 4x Drizzle diagnostics. Pixfrac, stability, artifact, and coverage proxies are evaluated on the denser output grid.

# %% [markdown]
# > **图表说明**: 这张图展示 4x Drizzle 的 pixfrac sweep。左图看 split-half 与 artifact，中图看 coverage 分位数，右图看 coverage 小于 1 的 HR 像素比例。
# >
# > **怎么看**: 4x 网格更密，coverage 比 2x 更容易变稀。低 pixfrac 如果让 coverage 大量低于 1，就算视觉边缘更锐，也要按高风险处理。
# >
# > **异常是否正常**: 4x 的 split-half 或 artifact 指标比 2x 更差并不意外，因为 4x 每格只有 2.5 um，已经远小于 20 um 空间分辨率。这里检验的是中心轮廓显示是否更顺，而不是物理 2.5 um 分辨率。
# >
# > **核心发现**: 4x 是否值得保留，关键看中心 crop 视觉收益是否超过 coverage 稀疏和伪影风险。

# %%
show_optional_fig(
    "drizzle_2x_vs_4x_center_third_crop.png",
    "Pending: drizzle_2x_vs_4x_center_third_crop.png (requires 2x and 4x Drizzle HR NPY).",
)

# %% [markdown]
# Figure 7: Drizzle 2x versus 4x center crop. The same physical center region is compared to assess visualization benefit and artifact risk.

# %% [markdown]
# > **图表说明**: 这张图只显示中心 1/3 物理区域（即 3× 视觉放大）。4x 原图是 1920×2560，这里裁成约 640×853；2x 取相同物理区域后上采样到相近显示尺度，便于直接比较两者轮廓。
# >
# > **怎么看**: 比较 2x 与 4x 的边缘，看 4x 是否让芯片内部轮廓更连续、边界走向更平滑。若 4x 只是把噪声或条纹放大，甚至出现断裂或虚假斑点，应按显示伪影处理。
# >
# > **异常是否正常**: 4x 图更“细”不等于 2.5 µm 物理分辨率；它只是更密的输出网格。高通图中的白色接近零局部变化，红/蓝是正/负结构响应，不是绝对温度。
# >
# > **核心发现**: 4x Drizzle 只能作为 contour-level visualization supplement；若中心 crop 没有稳定视觉收益，应继续以 2x Drizzle 作为主交付图。
