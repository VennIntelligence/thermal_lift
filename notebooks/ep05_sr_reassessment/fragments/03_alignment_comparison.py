# %% [markdown]
# ## 2. Alignment Method Comparison
#
# 对齐优劣用 held-out contour Chamfer 和 gradient correlation 交叉检查。这里的 held-out points 不参与 refinement，用于避免把局部吸附结果误当作泛化指标。
#
# 对齐评估需要同时看“轮廓位置是否贴近”和“局部强度结构是否一致”。Chamfer 距离衡量预测轮廓到参考轮廓的几何距离，越小越好；gradient correlation 衡量边缘附近梯度方向/强度的一致性，越高越好。两者互补：只优化几何距离可能把轮廓吸到错误边缘，只看相关性又可能忽略局部偏移。
#
# stage command 在这一节只作为 baseline。它给出电动台命令坐标经旋转角映射后的期望位移，适合做初始化、先验或正则约束；但热场漂移、机械误差、局部热结构变化和 ROI 内真实响应都会让“命令位移”与“数据实际支持的最佳对齐”不同。因此不能把 stage command 当成 ground truth 去判定 data-driven alignment 是对还是错。

# %%
method_table = ordered_method_table(outputs["method_summary"])
display(
    method_table.round(
        {
            "holdout_chamfer_median_px": 4,
            "holdout_chamfer_p90_px": 4,
            "gradient_corr_median": 4,
            "gradient_corr_p10": 4,
            "shift_norm_median_px": 4,
            "shift_norm_p90_px": 4,
        }
    )
)

# %% [markdown]
# > **数据说明**: 表格比较 no alignment、stage prior、filename affine、NCC init 和 contour refined 的 held-out Chamfer、gradient correlation 与 shift norm。`median` 反映典型帧，`P90` 或 `P10` 反映尾部风险，`shift_norm` 反映该方法施加的位移幅值。
# > **怎么读**: 好的 alignment 应同时满足 Chamfer median/P90 较低、gradient correlation median 较高、gradient correlation P10 不崩坏。`shift_norm` 不是越大或越小越好，而是用来检查方法是否出现不合理的大幅修正。
# > **正常/异常理解**: no alignment 应该最差，这是 sanity check；stage prior 如果优于 no alignment，说明物理命令方向有信息；data-driven 方法如果进一步降低 held-out Chamfer，说明数据约束补偿了 command prior 的误差。异常情况包括：Chamfer 下降但 gradient correlation 明显恶化，可能是吸附到错误轮廓；P90 很高，说明少数帧会严重失配；data-driven shift 大幅偏离 stage/filename，说明需要人工复核。
# > **核心发现**: `data_driven_contour_refined` 的 Chamfer median/P90 最低，`data_driven_ncc_init` 的 gradient correlation 最强。EP06 的主对齐应以 data-driven NCC init 为连续初值，并用 contour refinement 做局部锚定和质量门控；stage/filename 保留为 prior 和对照，而不是最终真值。

# %%
display(Image(filename=str(OUTPUT_DIR / "alignment_method_comparison.png")))

# %% [markdown]
# > **图表说明**: 左图显示 held-out Chamfer median 和 P90，右图显示 gradient correlation median 和 P10。左图越低越好，右图越高越好；两个分位数一起看，可以区分“平均表现好”和“尾部帧稳定”。
# > **怎么读**: 先看 no alignment 到 stage prior 是否有改善，再看 filename affine 和 data-driven 方法是否继续改善。若某方法只在 median 上变好、但 P90 或 P10 变差，说明它可能只改善典型帧而牺牲难帧。
# > **正常/异常理解**: 本图中轮廓 holdout 误差从 no alignment 到 data-driven refinement 整体下降，是符合微扫描 alignment 预期的。stage prior 不应被要求达到最优，因为它没有使用图像局部证据；它的价值是提供物理方向和相位先验。如果 stage prior 最优，反而要检查 data-driven 过程是否过拟合、ROI 是否缺少可用边缘，或 held-out 点是否定义错误。
# > **核心发现**: data-driven contour refined 比 filename/stage 更适合作为 EP06 对齐起点；filename/stage 应保留为 prior 和对照组，用于区分“命令位移带来的显示变化”和“数据驱动对齐带来的轮廓集中度提升”。
