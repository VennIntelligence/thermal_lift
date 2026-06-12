# %% [markdown]
# ## S-F1 — FRC 全档案（supp D.2 / A.4 配图）
#
# 主文 F2 的完整版：逐 seed cutoff、控制组各自的 1/7 cutoff 位置、
# band 表（周期插值 FRC）与 fine-grid 零覆盖统计。

# %%
figS01 = show_figure(
    PAPER_FIGS / "figS01_frc_archive.png",
    "uv run python scripts/paper_figures/fig02_frc.py",
)
figS01

# %% [markdown]
# > **图表说明**: (a) 主曲线 + 判据 + 逐 seed cutoff 刻度；(b) 控制组全曲线，
# > 竖虚线为各自 1/7 cutoff（正控制 13.6 µm、负/漂移控制 26.2 µm）；
# > (c) band 表分组柱（颜色与 (b) 图例一致）；(d) 每个 split half 的零覆盖比例
# > （均值 ~27%）。
# > **怎么看**: (c) 中 12–10 µm 频带 main 与负控制、漂移控制同步高企——反弹的
# > 非分辨率来源一目了然；(d) 解释了 5x 诊断网格上 coverage 伪相关的物理基础。
# > **核心发现**: 17.0 µm cutoff 跨 seed 稳定（16.2/16.2/17.0）；控制组的失效
# > 模式（正控制不更差、负控制高频不塌缩）如实入档，支撑「只主张 17 µm」的边界。
# > **状态**: ✅ 新落地（本次图优化）。
