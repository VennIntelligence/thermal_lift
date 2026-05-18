# %% [markdown]
# ## Step 3 — Noise Floor 与局部温差尺度
#
# 噪声底是 `0.0724 C`。SR 不应该只看全图均值或单一 sharpness，而要检查局部结构/轮廓的温差是否高于噪声、方向是否能提供微扫描法线相位、局部曲率是否适合做 alignment anchor。
#
# 对热像数据来说，边缘是否“看得见”不只取决于几何尺寸，也取决于边缘两侧温差是否大到超过探测器噪声。这里的目标是找到适合作为 alignment anchor / quality gate 的局部结构，而不是证明所有内部结构都能被 2x 重建。

# %%
segments, observability_summary, outer_mask, outer_contour, inner_contours = measure_contour_observability(
    reference_frame,
    theta_deg=THETA_DEG,
    noise_sigma_c=NOISE_SIGMA,
)
segments.to_csv(OUTPUT_DIR / "local_contour_observability_segments.csv", index=False)
observability_summary.to_csv(OUTPUT_DIR / "local_contour_observability_summary.csv", index=False)

snr_reference = build_snr_reference_table(NOISE_SIGMA, observability_summary)
snr_reference.to_csv(OUTPUT_DIR / "snr_noise_reference.csv", index=False)

display(observability_summary)
display(snr_reference)

print(f"Detected contour segments: {len(segments)}")
print(f"Detected inner contour count: {len(inner_contours)}")
print(f"Noise floor: {NOISE_SIGMA:.4f} C")

# %% [markdown]
# > **数据说明**: 第一张表按 outer/inner contour 汇总局部温差、SNR、法线投影和 anchor-candidate 比例；第二张表把噪声底、3x noise gate、0.3/0.7/1.0 C 参考温差和实测中位温差统一换算成 SNR。
# > **怎么读**: `|Delta T|` 或类似温差列表示局部轮廓两侧的热对比；SNR 是温差除以 0.0724 C 噪声底；法线投影描述微扫描位移是否主要跨过边缘，而不是沿着边缘滑动；anchor-candidate 比例表示有多少局部段同时通过这些条件。
# > **正常/异常理解**: 正常情况是有一部分 segment SNR 明显高于 3x noise gate，但不会每个 segment 都合格。如果所有 segment 都接近噪声底，后续 SR 应更保守；如果 SNR 高但法线投影弱，它仍可能不适合作为位移/边缘定位 anchor。
# > **核心发现**: 噪声底不直接否定 2x contour-level POC；它要求 EP05 在局部结构上做质量门控，只让可观测、高置信度区域参与或主导 SR 评估。这些表不能替代真实数据 SR 输出的轮廓一致性检查。

# %%
fig = plot_noise_floor_snr(
    snr_reference,
    segments,
    noise_sigma_c=NOISE_SIGMA,
)
save_fig(fig, "noise_floor_snr_contrast.png")

# %% [markdown]
# > **图表说明**: 左图把局部温差映射到 SNR，右图展示参考帧外/内轮廓 segment 的实测 `|Delta T|` 分布，并标出 1x/3x/5x noise gate。
# > **怎么读**: 左图可用于把“摄氏度温差”翻译成“噪声倍数”；右图看的是实际轮廓段落有多少落在这些 gate 之上。超过 3x noise gate 通常表示该段不太可能只是随机噪声，但还要继续看方向和曲率。
# > **正常/异常理解**: 0.3 C 已经约为 4.1x 噪声，0.7 C 约为 9.7x 噪声；实测轮廓段中存在大量高于这些参考尺度的局部结构。异常情况包括分布整体贴近 1x 噪声线，或只有极少数离群段支撑全部结论。
# > **核心发现**: EP03 支持“用局部 contour/edge 证据做可观测性筛选”，不支持用全局均值噪声或单一低响应局部段落做全局否定。高 SNR 是进入 POC 的条件，不是 POC 成功的证据。

# %%
fig = plot_local_contour_candidate_map(
    reference_frame,
    outer_contour,
    inner_contours,
    segments,
)
save_fig(fig, "local_contour_candidate_map.png")

# %% [markdown]
# > **图表说明**: 这张图只展示空间位置证据：外轮廓、内轮廓和 anchor candidates 叠加到参考温度矩阵。它回答“哪些地方可能有可用结构”，不回答“SR 后是否更清楚”。
# > **怎么读**: 先看背景温度矩阵中的芯片形状，再看轮廓线落在哪些边界上，最后看 anchor candidates 是否覆盖了内部结构和关键边缘。候选点越贴近真实轮廓、空间分布越均衡，越利于后续对齐门控。
# > **正常/异常理解**: 正常情况是候选点只出现在部分局部结构上，而不是铺满全图。若候选点集中在孤立角落，说明后续 SR 评估不能代表整个芯片；若候选点落在明显非结构区域，需要检查轮廓检测或温差门控。
# > **核心发现**: 这张图用于确认可用局部结构在哪里，不把 segment 统计散点混在同一画布里，避免暗示两类信息有一一对应关系。空间候选只能指导 EP05 选 anchor/ROI，不能代替最终 SR 可视化结论。

# %%
fig = plot_local_anchor_confidence(segments)
save_fig(fig, "local_anchor_confidence_scatter.png")

# %% [markdown]
# > **图表说明**: 这张图只展示 segment-level 门控统计：每个局部 segment 的 SNR 与 X 微扫描法线投影。一个点代表一个局部轮廓段，而不是一个像素或一整帧。
# > **怎么读**: 横向或纵向位置反映该 segment 的温差可信度和位移几何是否合适。理想 anchor 位于高 SNR、较高法线投影区域；低 SNR 点不稳定，低法线投影点对跨边缘定位帮助有限。
# > **正常/异常理解**: 可用 anchor 是局部性的：有些结构温差强但法线投影弱，有些内部轮廓提供了外轮廓没有的方向覆盖。若高 SNR 点与高投影点几乎不重叠，后续 alignment 需要更严格筛选。
# > **核心发现**: 局部 ESF/CRB 应作为 alignment anchor 和 quality gate；它不是最终交付目标，也不能替代对芯片内部结构/形状的 SR 评估。EP05 仍需要用真实多帧重建结果证明 contour-level 增益。
