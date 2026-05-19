# %% [markdown]
# ## 4. Overlay Evidence and Visual Sanity Appendix
#
# overlay 证据先作为 group-level sanity check 展示，再看 edge-density 版本。它不能替代 SR 指标，也不能把“看起来更集中”直接解释为真实分辨率提升。
#
# 叠图的作用是提供人工可读的 sanity check：如果 alignment 真正把同一物理轮廓对到一起，那么多帧 edge 会在参考轮廓附近形成更窄、更集中的密度带；如果只是显示方式改变，或把不同结构错误叠加，密度会变宽、分叉或在背景区域增加。这里诚实保留 filename affine 与 data-driven contour 的相互胜负关系，不把 overlay 当作 EP06 的 SR metric。
#
# 注意：overlay evidence 不是 SR 输出，也不是最终验收。它只回答“对齐后的轮廓堆叠是否更集中、是否与数值指标方向一致”。EP06 仍需要用独立重建指标验证 2x contour-level gain。

# %%
overlay_winner_table = overlay_group_winner_table(overlay_outputs["summary"])
display(
    overlay_winner_table.round(
        {
            "best_median_chamfer_px": 4,
            "filename_median_chamfer_px": 4,
            "contour_median_chamfer_px": 4,
            "contour_minus_filename_px": 4,
        }
    )
)

# %% [markdown]
# > **数据说明**: 表格从 `overlay_alignment_summary.csv` 汇总每个 overlay group 的 median Chamfer 最优方法，并单独比较 filename affine 与 data-driven contour 的 median Chamfer。`contour_minus_filename_px` 小于 0 表示 contour 更低，大于 0 表示 filename affine 更低。
# > **怎么看**: 这一表优先看 group 内相对关系，而不是把某一方法全局宣布为赢家。`All R=0` 和多个 scanline/column group 都是人工 sanity check 口径，不是 SR 重建评分。
# > **正常/异常理解**: filename affine 在多数组里 median Chamfer 更低是需要保留的事实；`scanline_y20` 中 data-driven contour 更低也同样保留。overlay 口径和 held-out contour 口径不完全等价，所以出现 group-level 胜负差异并不矛盾。
# > **核心发现**: overlay 证据不支持把 contour refined 一概说成所有叠图最优；它支持的是“filename affine 是强 prior/control，scanline_y20 等局部区域 data-driven contour 可更好”。EP06 应保留两类对照，而不是把 overlay 当 SR 指标。

# %%
overlay_group_table = overlay_group_summary_table(overlay_outputs["summary"])
display(
    overlay_group_table.round(
        {
            "median_chamfer_px": 4,
            "p90_chamfer_px": 4,
            "mean_chamfer_px": 4,
        }
    )
)

# %% [markdown]
# > **数据说明**: 这张明细表展示 `overlay_alignment_summary.csv` 的所有 group/method 组合，`best_by_median=yes` 标记该 group 内 median Chamfer 最低的方法。
# > **怎么看**: Chamfer 越低表示 overlay edge 到参考 edge 的几何距离越小。median 看典型帧，P90 看尾部；`mean_edge_count` 不在本表展示，因为不同方法的 edge 数基本一致，主要判据是 Chamfer。
# > **正常/异常理解**: no alignment 应该偏差最大，stage prior 应有部分改善，filename affine 和 data-driven contour 应明显更低。若某个 group 中 filename affine 更优，这并不推翻 data-driven alignment 的 held-out contour 结果，而是说明 overlay group 仍应作为 sanity appendix 读。
# > **核心发现**: overlay 明细确认 filename affine 在多数组别中是很强的 baseline；data-driven contour 的优势是局部和质量门控性质，尤其 `scanline_y20`，不能被扩大成“overlay 已证明 SR”。

# %%
display(Image(filename=str(OVERLAY_DIR / "all_main_4x4_txt_bmp_overlay.png"), width=1200))
display(Image(filename=str(OVERLAY_DIR / "all_main_4x4_edge_line_overlay.png"), width=1200))

# %% [markdown]
# > **图表说明**: 两张图是 all-main 4x4 的 TXT/BMP overlay 和 edge-line overlay visual sanity appendix，用于快速检查多组叠图是否存在明显错位、方向反常或边缘发散。
# > **怎么看**: 这些图只适合看叠图是否大体合理、局部是否有可疑错位。它们不是 SR 输出，也不是 SR metric；不能用“图像更密/更清楚”直接宣称 4x 或真实 5 µm 分辨率。
# > **正常/异常理解**: overlay 中局部线条重合更好是正常的 visual sanity signal；局部发散、交叉或双线结构则提示该组不适合作为稳定 SR 证据。TXT thermal edge 与 BMP edge 也没有光学-红外集成配准，因此只能做辅助检查。
# > **核心发现**: all-main overlay appendix 没有改变 EP05 的证据边界：它帮助发现明显异常，但 EP06 的验收仍必须依赖 2x forward/split-half/held-out contour 体系。

# %%
overlay_table = overlay_density_table(outputs["overlay_density"])
display(
    overlay_table.round(
        {
            "density_peak": 4,
            "density_p99": 4,
            "near_reference_edge_mean": 4,
            "off_reference_edge_mean": 4,
            "near_off_ratio": 2,
        }
    )
)

# %% [markdown]
# > **数据说明**: 表格比较 sampled contour stack 在参考边缘附近和远离参考边缘区域的 edge-density 分布。`near_reference_edge_mean` 衡量参考轮廓附近的平均密度，`off_reference_edge_mean` 衡量远离参考轮廓的背景密度，`near_off_ratio` 越大表示轮廓附近相对背景越集中。
# > **怎么读**: 先看 `sampled_frames` 是否一致，再比较 near 和 off 两类区域。理想 alignment 会让 near 区域保持或升高，同时让 off 区域下降；如果 near/off ratio 升高但 near 本身下降很多，需要小心是否只是把大部分边缘过滤掉了。
# > **正常/异常理解**: 正常的 refinement 应该减少背景散射密度，并让高密度区域更贴近参考轮廓。异常情况包括：off-reference 密度升高，说明对齐后产生更多远离参考轮廓的假边缘；density peak 很高但 p99/near 没有改善，可能只是少数位置过度堆叠；near/off ratio 极高时也要回到图像检查，确认不是分母接近 0 导致的数值假象。
# > **核心发现**: data-driven refined stack 的参考边缘附近密度更集中，远离参考边缘的背景密度更低。叠图证据与 holdout Chamfer 一致，说明 refinement 提升的是轮廓对齐集中度，而不是单纯显示倍率。

# %%
display(Image(filename=str(CAPACITY_DIR / "alignment_overlay_evidence.png")))

# %% [markdown]
# > **图表说明**: 左图是不对齐 edge density，中图是 data-driven contour refined 后的 edge density，右图显示 refined 减去 no alignment 的密度差。前两图看“边缘是否集中”，差分图看“alignment 让哪些位置变强或变弱”。
# > **怎么读**: 如果 refinement 有效，中图的高密度带应比左图更贴近参考轮廓且更窄；右图中正值应主要出现在参考边缘附近，负值应主要出现在原本散开的边缘区域。若正负变化杂乱分布，说明对齐收益不稳定。
# > **正常/异常理解**: 正常模式是轮廓附近增强、背景散射减弱。异常模式包括：高密度带变宽，表示多个结构被糊在一起；差分图在背景大面积增强，表示 refinement 可能引入假轮廓；局部增强只出现在极少数点，说明不能把它外推为全局 SR 证据。
# > **核心发现**: refined 后的高密度区域更贴近参考轮廓，差分图给出与 Chamfer 指标一致的视觉 sanity check。真正的 SR 成功仍需 EP06 的 split-half、held-out contour 和结构一致性指标共同验证。
