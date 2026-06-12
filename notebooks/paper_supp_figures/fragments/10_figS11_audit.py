# %% [markdown]
# ## S-F11 — 数据审计链：假 session 教训与采集序覆盖（supp B.1 配图）
#
# EP01 审计的两张核心图：排序方式对 session 检测的影响、raster 网格的
# 采集序与缺失/重复坐标标注。

# %%
REBUILD_S = (
    "uv run python scripts/build_ep01_cache.py（重建审计）→ EP01 notebook → "
    "uv run python scripts/paper_figures/collect_promoted_supp.py"
)
display(Markdown("**figS11a — 文件名序 vs 采集序的 session 检测对比**"))
display(show_figure(PAPER_FIGS / "figS11a_order_comparison.png", REBUILD_S))
display(Markdown("**figS11b — raster 网格按采集序着色（含缺失坐标与 session 边界）**"))
display(show_figure(PAPER_FIGS / "figS11b_raster_acquisition.png", REBUILD_S))

# %% [markdown]
# > **图表说明**: (a) 同一批 263 帧按文件名字母序排列得到 13 个表观温度段
# > （12 次 >0.5 °C 跳变），按 mtime 采集序排列只有 3 个物理段、2 个边界跳变；
# > (b) 16×16 command 网格按采集序着色，标出主 session 起止、其他 session
# > 帧（红圈）与 3 个缺失坐标（红叉）。
# > **怎么看**: (a) 是「文件名是坐标标识不是时序」教训的直接可视化——错误
# > 排序会制造大量假 session 边界；(b) 验证主 session 严格逐行推进（行内
# > gap=1）且覆盖 248/256 坐标。
# > **核心发现**: session 检测必须基于 acquisition_order；263→255→248 的
# > 剔除链条有清晰的空间与时间结构支撑（supp B.1 全部数字的图形背书）。
# > **状态**: ✅ 选编收编（06-12，源图本就是 CVPR 风格）。
