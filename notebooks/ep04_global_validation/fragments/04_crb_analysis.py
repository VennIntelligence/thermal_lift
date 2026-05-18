# %% [markdown]
# ## Step 3 — Anchor Coverage Map
#
# 本节把 segment gate 映射回温度参考帧，并把通过 segment 在各条 X scanline 上的支持情况拆成单独图。目标是告诉 EP06 哪些 contour/scanline 可以作为 alignment anchor，哪些空间区域缺锚点。

# %%
fig = plot_anchor_coverage_map(
    reference_frame,
    outer_segment_summary,
    inner_segment_summary,
    outer_results,
    inner_results,
)
save_fig(fig, "anchor_coverage_map.png")

# %% [markdown]
# > **图表说明**: 图中在参考温度帧上叠加 outer/inner segment 的 anchor pass/reject，并把视野裁到芯片中间结构附近。
# > **读法**: 把这张图当成“哪里有可用锚点”的地图，而不是当成芯片真实轮廓标注。蓝色点/方块表示这个局部 segment 在多条 X scanline 上有足够稳定的红外定位证据；浅灰色表示该局部不能直接作为 alignment anchor。圆点和方块只区分 outer/inner。
# > **正常/异常理解**: 外轮廓形成较连续 anchor 是正常的；内轮廓更稀疏也正常，因为内部热结构可能弱、宽、弯曲或受局部热场演化影响。空间上大片缺锚是 EP06 alignment 的风险提示，但不是“这里没有内部结构”的证据。
# > **对本 Episode 的意义**: EP06 alignment 应优先使用空间上稳定且跨 scanline 支持充分的段；内部缺锚区域仍是 SR 目标区域，只是不能把 EP04 localization 当成光学真值或最终 SR 真值。

# %%
fig = plot_anchor_scanline_support(
    outer_results,
    inner_results,
    outer_segment_summary,
    inner_segment_summary,
)
save_fig(fig, "anchor_scanline_support.png")

# %% [markdown]
# > **图表说明**: 图中按 scanline Y 坐标统计已通过 segment-level gate 的 anchor，在该 scanline 上有多少 row-level 检查被评估、多少实际通过。
# > **读法**: 浅灰柱是该 scanline 上被检查的 anchor 数量，蓝色/橙色内柱分别是 outer/inner 中实际通过 row-level gate 的数量。彩色柱越接近浅灰柱，说明该 scanline 对已接受 anchor 的支持越连续。
# > **正常/异常理解**: 某些 scanline 支持少是正常的，因为局部热纹理、NCC 相位覆盖和采集时间位置会改变 row-level 稳定性。这里的柱状图只表达配准锚点可用性，不表达真实结构是否存在。
# > **对本 Episode 的意义**: EP06 可以用这些 scanline 统计选择 alignment 输入和 held-out 检查线，避免把某一条扫描线的局部好坏外推成全局结论。

# %%
coverage_table = (
    pd.concat(
        [
            outer_results.assign(contour="outer"),
            inner_results.assign(contour="inner"),
        ],
        ignore_index=True,
    )
    .assign(pass_bool=lambda df: df["pass_fail"].astype(str).str.lower().isin(["true", "1", "yes"]))
    .groupby(["contour", "scanline_y_um"], dropna=False)
    .agg(
        evaluated_rows=("segment_id", "count"),
        passed_rows=("pass_bool", "sum"),
        row_pass_rate=("pass_bool", "mean"),
    )
    .reset_index()
)
display(coverage_table.assign(row_pass_rate=lambda df: (100.0 * df["row_pass_rate"]).round(1)))

# %% [markdown]
# > **数据说明**: 表格按轮廓类型和 scanline Y 坐标统计 row-level 评估数量、通过数量和通过率。
# > **读法**: `evaluated_rows` 是该 scanline 上被检查的 segment-row 数，`passed_rows` 是通过 gate 的数量，`row_pass_rate` 是局部通过比例。读表时应同时看数量和比例：样本很少时，一个 0% 或 100% 的比例都不应过度解释。
# > **正常/异常理解**: scanline 之间通过率不同是正常的，因为每条线的局部热纹理、NCC 相位覆盖、ESF 条件和采集时间位置都不同。异常情况是某些 scanline 完全没有可用 anchor，这会降低对齐约束；但这仍不能被简化为 stage command 是否准确。
# > **对本 Episode 的意义**: EP06 可以用这些 scanline 统计选择 alignment 输入和 held-out 检查线，避免把某一条扫描线的局部好坏外推成全局结论。
