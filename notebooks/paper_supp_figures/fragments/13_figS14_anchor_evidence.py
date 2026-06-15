# %% [markdown]
# ## S-F14 — MAP-TV 去卷积基准的结构证据（supp D.2.3 配图）
#
# 「FWHM 114→100 µm 有限轮廓增强」主张的视觉来源：三条 zigzag 剖面的
# bare drizzle vs MAP-TV 对照，以及四方法 highpass 全景。

# %%
REBUILD_S = (
    "cd algos/ep15_info_limit && uv run python scripts/run_m4_deconv_anchor.py --device cuda → "
    "uv run python scripts/paper_figures/collect_promoted_supp.py"
)
display(Markdown("**figS14a — zigzag 剖面（bare drizzle vs MAP-TV，固定横截线）**"))
display(show_figure(PAPER_FIGS / "figS14a_zigzag_profiles.png", REBUILD_S))
display(Markdown("**figS14b — 四方法 highpass 对照（bicubic / drizzle / MAP-TV / EP07-v6 ×2.5up）**"))
display(show_figure(PAPER_FIGS / "figS14b_four_arm_highpass.png", REBUILD_S))

# %% [markdown]
# > **图表说明**: (a) 三条固定横截线上的反相线对比剖面（°C），红=MAP-TV、
# > 蓝=bare drizzle——FWHM/dip 指标即从这些剖面提取（median FWHM 114→100 µm、
# > dip 0.929→0.934）；(b) 同一中心区域的 highpass 结构图四方法并排（白≈无变化、
# > 红/蓝=相对局部背景的正/负响应）。
# > **怎么看**: (a) 中 MAP-TV 的谷更深、肩更陡，但三条剖面**逐条混合**
# > （2 条变宽、1 条显著收窄）——这就是论文措辞钉死「limited contour
# > enhancement」的原因；(b) 中 bare drizzle 的格纹（棋盘纹理）清晰可见，
# > MAP-TV 显著压制，UNet 方法边缘最锐但需对照保真证据（S-F9/S-F10）。
# > **异常是否正常**: (a) 剖面尾部的小振荡为 drizzle 格纹泄漏；(b) 中 highpass
# > 图突出边缘、不代表温度读数（AGENTS 教程式解读标准第 3 条）。
# > **核心发现**: MAP-TV 基准把 17 µm 频带的相干信息转化为有限但真实的轮廓
# > 增益——它因此成为学习方法必须同时在 FRC 一致性与轮廓指标上击败的验收 gate。
# > **状态**: ✅ 选编收编（06-12）。
