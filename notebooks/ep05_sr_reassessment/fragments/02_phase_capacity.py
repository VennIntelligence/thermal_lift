# %% [markdown]
# ## 1. 2x Phase-Bin Capacity
#
# 这里检查每种对齐方法落入 2x SR 四个 sub-pixel phase bin 的覆盖情况。对 EP06 而言，关键不是宣称更高倍率已经成立，而是确认 2x phase diversity 是否足够支撑 contour-level POC。
#
# 微扫描 SR 依赖一个基本前提：多帧之间的相对位移不能全都落在同一个像素相位上。以 2x SR 为例，一个 LR 像素会被拆成 `2 x 2 = 4` 个 sub-pixel phase bin；如果 255 帧只重复采到同一个 bin，那么它们主要提供降噪和平均，不提供 2x 网格所需的相位多样性。
#
# 这里的 phase capacity 只回答“采样几何是否足够”，不回答“SR 是否已经成功”。真正的 SR 成功还需要 EP06 检查 forward consistency、split-half 稳定性和轮廓结构一致性。

# %%
phase_table = phase_capacity_table(outputs["phase_summary_2x"])
display(phase_table.round({"entropy_fraction": 3, "expected_count": 2}))

# %% [markdown]
# > **数据说明**: 表格统计每种对齐方法在 2x phase bins 上的覆盖情况。`occupied_bins` 是被至少一帧命中的相位格数量，`bad_bins` 是没有有效样本的相位格数量，`entropy_fraction` 越接近 1 表示四个 bin 越均匀，`min_count/max_count` 表示最少/最多 bin 中的帧数。
# > **怎么读**: 对 2x SR，理想情况是 `occupied_bins = 4`、`bad_bins = 0`，并且 `min_count` 不要接近 0。`expected_count` 约为 `255 / 4 = 63.75`，所以每格大约六十多帧代表相位分布很均衡。
# > **正常/异常理解**: `no_alignment` 只占一个 bin 是预期现象，因为不施加相对位移时所有帧都被视为同一像素相位；如果某个可用 alignment 方法仍有空 bin，EP06 的 2x 重建会在对应相位上缺少观测约束，容易产生插值式假细节。stage prior 覆盖 4/4 bins 只说明命令位移的相位几何合理，不说明命令位移就是真实热像对齐。
# > **核心发现**: 255 帧主 session 对 2x SR 的 phase coverage 是充分的；EP06 可以启动 2x contour-level POC，但应继续用 data-driven 对齐作为主线，用 stage/filename 保留为 prior 和对照。

# %%
display(Image(filename=str(OUTPUT_DIR / "phase_bin_coverage_2x.png")))

# %% [markdown]
# > **图表说明**: 每一行是一种对齐方法，横向堆叠条形图把 255 帧按四个 2x phase bin 分解，并在右侧标出 occupied/empty bin 与 entropy。颜色块越接近等宽，说明四个 sub-pixel phase 的样本越均匀。
# > **怎么读**: 先看是否有空 bin，再看颜色块是否极端失衡。2x POC 的最低门槛是四个 bin 都有样本；更好的情况是每个 bin 都有足够多的帧，避免某一相位主要靠插值补齐。
# > **正常/异常理解**: 不对齐时只有一个 bin 是 sanity check；stage prior、filename affine、NCC init 和 contour refined 都覆盖四个 bin 是正常且有利的。若 data-driven refinement 后出现空 bin，说明局部修正可能破坏了 phase diversity，需要在 EP06 中限制 refinement 或回退到连续位移 prior。
# > **核心发现**: phase-bin 证据支持启动 2x contour-level SR POC；但它只是“容量证据”，不是重建质量证据。4x 仍只作为后续风险项，不在本 Episode 宣称可行。
