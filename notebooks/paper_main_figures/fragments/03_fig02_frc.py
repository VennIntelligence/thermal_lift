# %% [markdown]
# ## F2 — 信息存在性：phase-stratified split-half FRC + 控制组（§5，单栏）
#
# 回答 Q1「数据里到底有没有超出单帧网格的相干信息」。本图为 2026-06-12 新落地的
# CVPR 风格重绘（替代 `output/ep15_info_limit/m2_frc/` 下的工作版 PNG）。

# %%
fig02 = show_figure(
    PAPER_FIGS / "fig02_frc.png",
    "uv run python scripts/paper_figures/fig02_frc.py",
)
fig02

# %% [markdown]
# > **图表说明**: (a) 248 帧 clean session 的相位分层 split-half FRC 均值曲线
# > （3 seeds），叠加 1/7 与 half-bit 两种判据、逐 seed cutoff 刻度与 10–12 µm
# > 反弹风险带；(b) 四个控制组曲线（正/负/漂移控制）。横轴空间频率，顶轴换算周期。
# > **怎么看**: FRC 越高表示两半独立重建在该频带越一致（越可信）；曲线首次跌破
# > 判据线的位置即 cutoff（17.0 µm，std 0.50 µm）。灰色带内的高 FRC **不是**分辨率
# > 证据——负控制（shift-shuffle）在同一频带也保持 FRC≈0.9，说明该反弹由
# > coverage/格纹与热漂移驱动。
# > **异常是否正常**: 正控制（bicubic）cutoff 13.6 µm 未如预期差于 main——控制组
# > 部分失效是本文如实披露的限制（Limitations 第 3 条）；负控制低频为负值属于
# > shuffle 后的反相关，正常。
# > **核心发现**: 主张钉死在 **17.0 µm cutoff**——超出 20 µm 分辨率的相干信息真实
# > 存在但有限；10–12 µm 反弹只作风险标注。
# > **状态**: ✅ 新落地（本次图优化）；supp 档案版见 S-F1。
