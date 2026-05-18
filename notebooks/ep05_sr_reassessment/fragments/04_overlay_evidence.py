# %% [markdown]
# ## 3. Contour Stack Evidence
#
# 叠图证据使用 edge density，而不是把 255 帧低透明度线条直接糊在一起。这样可以同时看见整体集中程度和 data-driven alignment 相对 no alignment 的变化。
#
# 叠图的作用是提供人工可读的 sanity check：如果 alignment 真正把同一物理轮廓对到一起，那么多帧 edge 会在参考轮廓附近形成更窄、更集中的密度带；如果只是显示方式改变，或把不同结构错误叠加，密度会变宽、分叉或在背景区域增加。
#
# 注意：overlay evidence 不是 SR 输出，也不是最终验收。它只回答“对齐后的轮廓堆叠是否更集中、是否与数值指标方向一致”。EP06 仍需要用独立重建指标验证 2x contour-level gain。

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
display(Image(filename=str(OUTPUT_DIR / "alignment_overlay_evidence.png")))

# %% [markdown]
# > **图表说明**: 左图是不对齐 edge density，中图是 data-driven contour refined 后的 edge density，右图显示 refined 减去 no alignment 的密度差。前两图看“边缘是否集中”，差分图看“alignment 让哪些位置变强或变弱”。
# > **怎么读**: 如果 refinement 有效，中图的高密度带应比左图更贴近参考轮廓且更窄；右图中正值应主要出现在参考边缘附近，负值应主要出现在原本散开的边缘区域。若正负变化杂乱分布，说明对齐收益不稳定。
# > **正常/异常理解**: 正常模式是轮廓附近增强、背景散射减弱。异常模式包括：高密度带变宽，表示多个结构被糊在一起；差分图在背景大面积增强，表示 refinement 可能引入假轮廓；局部增强只出现在极少数点，说明不能把它外推为全局 SR 证据。
# > **核心发现**: refined 后的高密度区域更贴近参考轮廓，差分图给出与 Chamfer 指标一致的视觉 sanity check。真正的 SR 成功仍需 EP06 的 split-half、held-out contour 和结构一致性指标共同验证。
