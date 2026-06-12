# %% [markdown]
# ## F1 — 系统、标定链与采样/分辨率区分（§3，双栏全宽）
#
# 主文 §3 的开篇图：(a) raster 微扫描几何与 θ=47.6° 旋转下的 command→px 向量；
# (b) 观测算子链 y_k = D·B·H·S x + n（每个算子标注测量来源）；
# (c) detector pitch 10 µm ≠ 空间分辨率 20 µm ≠ SR 输出网格 5 µm 的同轴示意。

# %%
fig01 = show_figure(
    PAPER_FIGS / "fig01_system_calibration.png",
    "uv run python scripts/paper_figures/fig01_system_calibration.py",
)
fig01

# %% [markdown]
# > **图表说明**: 三联系统图——微扫描几何、观测算子链、三个空间尺度的同轴对比；
# > 全部数字来自 `configs/stage_calibration.json`、`configs/noise_floor.json` 与
# > `output/ep15_info_limit/m3_sigma/sigma_summary.json`（source map 存于同名 .json）。
# > **怎么看**: (a) 中 stage 命令向量经 θ 旋转映射到像素系；(b) 中每个算子都有
# > 实测参数（θ、σ∈[0.2,0.5]、10 µm box、0.0724 °C 噪声底）；(c) 中三把"尺子"
# > 画在同一物理距离轴上——5 µm 是输出采样间距，不是分辨率。
# > **核心发现**: forward model 是测量出来的而非假设的（贡献 C1 的视觉背书）；
# > pitch/分辨率/输出网格三个量被显式区分，封死"5 µm 分辨率"的误读。
# > **状态**: ✅ 终稿可用（Task A 落地，CVPR 风格）。
