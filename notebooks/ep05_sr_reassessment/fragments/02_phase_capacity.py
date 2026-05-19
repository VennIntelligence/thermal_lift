# %% [markdown]
# ## 2. Multi-Scale Phase Coverage Risk
#
# 这里检查每种对齐方法落入 2x、3x、4x sub-pixel phase bin 的覆盖情况。对 EP06 而言，关键不是宣称更高倍率已经成立，而是确认 2x phase diversity 是否足够支撑 contour-level POC，并把 3x/4x 的 phase collapse 风险提前暴露出来。
#
# 微扫描 SR 依赖一个基本前提：多帧之间的相对位移不能全都落在同一个像素相位上。以 2x SR 为例，一个 LR 像素会被拆成 `2 x 2 = 4` 个 sub-pixel phase bin；如果 255 帧只重复采到同一个 bin，那么它们主要提供降噪和平均，不提供 2x 网格所需的相位多样性。
#
# 这里的 phase capacity 只回答“采样几何是否足够”，不回答“SR 是否已经成功”。3x/4x 即使占满 bin，也只是 occupancy 诊断；尤其 contour refined 是局部吸附后的结果，高倍率 phase collapse 不能被解读成 4x 可行性证据。

# %%
multi_scale_phase_table = multi_scale_phase_coverage_table(summary_json, outputs["holdout_scores"])
display(multi_scale_phase_table.round({"entropy_fraction": 3}))

# %% [markdown]
# > **数据说明**: 表格把同一批 alignment shifts 投到 2x、3x、4x phase bins。`occupied_bins/bad_bins/total_bins` 描述有多少相位格被命中，`min_count/max_count` 描述最稀和最密的 bin，`entropy_fraction` 越接近 1 表示分布越均匀。
# > **怎么看**: 2x 的最低门槛是 4 个 bin 都有样本且没有空 bin；3x/4x 只作为风险诊断，不能因为 occupancy 看起来完整就宣称高倍率 SR 可行。真正的可行性还要看 PSF/SNR、forward model、split-half 稳定性和 held-out contour。
# > **正常/异常理解**: `no_alignment` 只落在一个 bin 是 sanity check。stage prior、filename affine、NCC init 在 3x/4x 上能占满 bin，只说明这些连续位移 prior 在几何上有分布；`Contour refined` 在 3x/4x 只占 4 个 bin，是局部 contour refinement 把位移吸附到少数 offset 后的 phase collapse，不能拿来证明 4x。
# > **核心发现**: EP05 支持的是 2x phase/alignment baseline。3x/4x 结果暴露了风险边界：高倍率 occupancy 或局部 refinement 后的 bin 数都不是 SR 成功证据，EP06 应聚焦 2x contour-level POC。

# %% [markdown]
# ### 2.1 2x Phase-Bin Capacity

# %%
phase_table = phase_capacity_table(outputs["phase_summary_2x"])
display(phase_table.round({"entropy_fraction": 3, "expected_count": 2}))

# %% [markdown]
# > **数据说明**: 表格统计每种对齐方法在 2x phase bins 上的覆盖情况。`occupied_bins` 是被至少一帧命中的相位格数量，`bad_bins` 是没有有效样本的相位格数量，`entropy_fraction` 越接近 1 表示四个 bin 越均匀，`min_count/max_count` 表示最少/最多 bin 中的帧数。
# > **怎么读**: 对 2x SR，理想情况是 `occupied_bins = 4`、`bad_bins = 0`，并且 `min_count` 不要接近 0。`expected_count` 约为 `255 / 4 = 63.75`，所以每格大约六十多帧代表相位分布很均衡。
# > **正常/异常理解**: `no_alignment` 只占一个 bin 是预期现象，因为不施加相对位移时所有帧都被视为同一像素相位；如果某个可用 alignment 方法仍有空 bin，EP06 的 2x 重建会在对应相位上缺少观测约束，容易产生插值式假细节。stage prior 覆盖 4/4 bins 只说明命令位移的相位几何合理，不说明命令位移就是真实热像对齐。
# > **核心发现**: 255 帧主 session 对 2x SR 的 phase coverage 是充分的；EP06 可以启动 2x contour-level POC，但应继续用 data-driven 对齐作为主线，用 stage/filename 保留为 prior 和对照。

# %%
display(Image(filename=str(CAPACITY_DIR / "phase_bin_coverage_2x.png")))

# %% [markdown]
# > **图表说明**: 每一行是一种对齐方法，横向堆叠条形图把 255 帧按四个 2x phase bin 分解，并在右侧标出 occupied/empty bin 与 entropy。颜色块越接近等宽，说明四个 sub-pixel phase 的样本越均匀。
# > **怎么读**: 先看是否有空 bin，再看颜色块是否极端失衡。2x POC 的最低门槛是四个 bin 都有样本；更好的情况是每个 bin 都有足够多的帧，避免某一相位主要靠插值补齐。
# > **正常/异常理解**: 不对齐时只有一个 bin 是 sanity check；stage prior、filename affine、NCC init 和 contour refined 都覆盖四个 bin 是正常且有利的。若 data-driven refinement 后出现空 bin，说明局部修正可能破坏了 phase diversity，需要在 EP06 中限制 refinement 或回退到连续位移 prior。
# > **核心发现**: phase-bin 证据支持启动 2x contour-level SR POC；但它只是“容量证据”，不是重建质量证据。4x 仍只作为后续风险项，不在本 Episode 宣称可行。

# %%
phase_distribution_table = fractional_phase_distribution_table(outputs["holdout_scores"], scale=4)
display(
    phase_distribution_table.round(
        {
            "frac_x_p10": 3,
            "frac_x_median": 3,
            "frac_x_p90": 3,
            "frac_y_p10": 3,
            "frac_y_median": 3,
            "frac_y_p90": 3,
        }
    )
)

# %% [markdown]
# > **数据说明**: 这张表把每种方法的 fractional phase 映射到 4x 的 `y x` bin count matrix。每个矩阵包含 4 行，每行 4 个数字；数字表示落入对应 sub-pixel phase bin 的帧数。`frac_x/frac_y` 分位数给出 shift 小数部分在 0–1 像素内的分布。
# > **怎么看**: 如果 4x count matrix 里很多格为 0，说明该方法在 4x 网格上有 phase 空洞；如果所有格都有样本，也只能说明相位覆盖，不代表光学或噪声条件支持 4x 重建。
# > **正常/异常理解**: `Contour refined` 的 4x 矩阵只集中在少数格，这是局部轮廓细化的吸附效应，适合做 quality gate，但不适合直接拿来当高倍率 SR phase prior。NCC init 和 filename affine 的分布更连续，更适合作为 2x SR 的 phase prior。
# > **核心发现**: fractional phase 表进一步确认 EP05 的边界：连续 prior 可支撑 2x baseline；contour refined 可做锚定和门控；4x 只保留为风险诊断，不进入本阶段可行性声明。
