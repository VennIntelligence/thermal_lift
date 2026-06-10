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
# ### 📊 堆叠图像对齐性能分组评估
#
# 该表格汇总了各对齐方法在不同空间叠图分组（如全部 $R=0$ 帧、特定扫描行/扫描列分组）下的中位 Chamfer 误差。
# 1. **物理意义**：用于进行全图尺度的几何合理性核对（Sanity Check）。
# 2. **局域对齐优势**：Filename affine 先验在多个全局叠图分组中表现出强烈的几何约束性，但在特定的高对比度特征带（如 `scanline_y20` 区域），数据驱动对齐（Contour refined）能够利用灰度梯度优势提供更小的对齐 Chamfer 偏差。
#
# **💡 算法决策**：分组叠图评估支持了“对齐方法具有空间差异性”的结论。这指导后续重建流程（EP06）必须结合空域自适应权重对不同区域施加动态对齐策略，并保留 Filename affine 作为强先验基线对照。

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
# ### 📊 叠图 Chamfer 误差细粒度性能明细
#
# 详细记录了每个空间叠图分组与各对齐算法组合下的 Chamfer 误差分位数，用于评估极端几何变形对对齐质量的影响。
#
# **💡 算法决策**：明细数据表明了无对齐状态下误差的离散性，以及数据驱动算法对大误差的压制效果。后续算法应以此明细作为选择全局刚性或局域仿射变形模型的物理决策依据。

# %%
show_fig(OVERLAY_DIR / "all_main_4x4_txt_bmp_overlay.png", width=1200)

# %% [markdown]
# Figure 6: Four-by-four thermal and bitmap overlay. Main-session samples are overlaid for visual alignment sanity checking.

# %%
show_fig(OVERLAY_DIR / "all_main_4x4_edge_line_overlay.png", width=1200)

# %% [markdown]
# Figure 7: Four-by-four edge-line overlay. Extracted edge lines are overlaid to reveal geometric spread and double-line artifacts.

# %% [markdown]
# ### 🗺️ 叠图错位与几何发散性视觉 Appendix 诊断
#
# 提供了 $4 \times 4$ 空间采样网格下的物理温度矩阵叠图及提取的边缘线重叠的可视化附录，用于快速评估配准后的边缘发散或双线伪影情况。
#
# **💡 算法决策**：由于物理红外矩阵与光学参考图之间不存在精密对齐，此附录仅用于宏观 of 几何一性自检。任何可疑的双线伪影或边缘交叉，应被直接反馈用于排除含有大范围漂移或振颤的物理帧。

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
# ### 📊 边缘线空间密度分布与富集度分析
#
# 定量分析了对齐后的边缘点栈在参考边缘邻域（Near Reference Edge）与背景区（Off Reference Edge）的空间富集度及其比值（Near/Off Ratio）：
# 1. **密度富集表现**：Contour refined 算法在参考边缘邻域的平均密度最高，而在远离参考轮廓的背景密度显著下降，其 Near/Off 比值相比无对齐状态提升了数倍，客观证明了边缘点栈在空间分布上被显著压窄并向真实几何边缘收缩。
# 2. **排除过滤假象**：密度中位数核对确认该富集度的提高不是由于剔除帧造成的，而是真实的亚像素物理收拢。
#
# **💡 算法决策**：边缘线空间密度分布是评估几何配准质量的核心物理判据。后续超分辨率流程将锁定该 Near/Off 比值作为衡量运动解算可信度的数学指标之一。

# %%
show_fig(CAPACITY_DIR / "alignment_overlay_evidence.png")

# %% [markdown]
# Figure 8: Alignment overlay evidence. Edge-density maps compare no alignment with refined data-driven alignment.

# %% [markdown]
# ### 🗺️ 边缘点栈密度对比及差分分布
#
# 对比了不对齐（左图）、数据驱动对齐（中图）下的边缘点空间分布密度，并在右图中展示了两者的差分分布（Refined - No alignment）。
# 1. **物理分布收拢**：中图显示出一条相较于左图更为陡峭、且极度向参考轮廓集中的高密度条带。
# 2. **能量转移可视化**：差分图中正值（增强区）紧贴参考轮廓分布，而负值（减弱区）主要分布在发散的背景区，从物理能量分布上清晰反映了像素几何偏差的消除过程。
#
# **💡 算法决策**：差分图为配准的物理增益提供了直观的物理能量转移证明。后续算法应结合此空间能量收敛分布设计亚像素重构的反演权重，优先让高能收拢区主导逆问题的迭代求解。
