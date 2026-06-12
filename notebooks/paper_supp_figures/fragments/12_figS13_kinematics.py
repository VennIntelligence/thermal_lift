# %% [markdown]
# ## S-F13 — 主 session 累计位移轨迹（supp B.3 配图）
#
# 248 帧 clean session 的实测累计位移（contour-refined 链路），按采集序着色
# ——raster 运动学与慢漂移的直接可视化。

# %%
figS13 = show_figure(
    PAPER_FIGS / "figS13_cumulative_trajectory.png",
    "EP05 notebook 管线重建源图 → uv run python scripts/paper_figures/collect_promoted_supp.py",
)
figS13

# %% [markdown]
# > **图表说明**: 横纵轴为累计实测位移 (dx, dy)（px），颜色为采集序；16 条
# > 斜线段对应 16 条 raster 扫描行（行内 X 连续步进），行间跳变对应 Y 步进
# > 与行重置。
# > **怎么看**: 轨迹包络 2.49 × 7.11 px、路径长 61.18 px（supp B.3.3 表）；
# > 行与行之间的整体缓慢平移即热漂移/机械漂移的累计效应——它解释了为什么
# > 行间 Y 邻帧（隔 ~16 帧）不能做定量标定（supp B.3.2 的 0.64× 失效）。
# > **异常是否正常**: 各行斜率一致且与 θ=47.6° 投影方向吻合，是 raster 几何
# > 的预期形态；行末端的轻微弯曲为细步进测量噪声。
# > **核心发现**: 实测轨迹同时支撑三件事——raster 各向异性（TGV 设计动机）、
# > 慢漂移的存在（split 设计动机）、stage 设计与实测的总体一致性。
# > **状态**: ✅ 选编收编（06-12）。
