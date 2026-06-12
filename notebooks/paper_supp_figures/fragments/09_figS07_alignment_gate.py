# %% [markdown]
# ## S-F7 — 对齐链、相位 bin 保持与 EP04 gate（supp B.2 配图）
#
# 从 EP04/EP05 产物中选编收编（已是学术风格）：对齐五步链的 held-out Chamfer、
# 2x 相位 bin 占用、gate 通过/失败的空间分布。

# %%
REBUILD_S = (
    "EP04/EP05 notebook 管线重建源图 → "
    "uv run python scripts/paper_figures/collect_promoted_supp.py"
)
display(Markdown("**figS07a — 对齐方法五步链（held-out Chamfer + 梯度一致性）**"))
display(show_figure(PAPER_FIGS / "figS07a_alignment_chain.png", REBUILD_S))
display(Markdown("**figS07b — 2x 相位 bin 占用（按对齐源）**"))
display(show_figure(PAPER_FIGS / "figS07b_phase_bins.png", REBUILD_S))
display(Markdown("**figS07c — EP04 gate 外轮廓通过/失败空间分布**"))
display(show_figure(PAPER_FIGS / "figS07c_gate_map.png", REBUILD_S))

# %% [markdown]
# > **图表说明**: (a) 五种对齐源的 held-out Chamfer（median + P90，越小越好）与
# > 梯度一致性（越大越好）——supp B.2.2 表格的图形版（0.398→0.246→0.170→0.156→0.133 px）；
# > (b) 各对齐源下 248 帧落入 2x 四个相位格的帧数与熵；(c) 外轮廓 84 段的
# > gate 通过（绿）/失败（红）空间分布叠加在真实热像上。
# > **怎么看**: (a) 中 stage prior 到数据驱动的两级下降是「command 是 prior 不是
# > truth」的定量体现；(b) 中除 no-alignment 外四格全占、熵≈1.0——对齐修正
# > **没有破坏**相位多样性（2x 重建前提成立）；(c) 中失败段集中在弱对比区域。
# > **异常是否正常**: (a) 里 contour refined 的梯度一致性略低于 NCC init 属
# > 指标间正常权衡（Chamfer 才是 held-out 主指标）；(b) 的 no-alignment 行
# > 248 帧全堆一格是未对齐的定义性结果。
# > **核心发现**: 数据驱动对齐链每一步都有 held-out 改善且保持相位覆盖；
# > gate 把不可靠锚点显式排除在「真值」角色之外。
# > **状态**: ✅ 选编收编（06-12）；⚠️ (c) 含整片芯片热像，终稿形态受客户
# > 许可约束（同 F5），必要时换中心 ROI 版本。
