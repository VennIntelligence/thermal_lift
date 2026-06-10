# %% [markdown]
# ## Step 3 — Noise Floor 与局部温差尺度
#
# 噪声底是 `0.0724 C`。SR 不应该只看全图均值或单一 sharpness，而要检查局部结构/轮廓的温差是否高于噪声、方向是否能提供微扫描法线相位、局部曲率是否适合做 alignment anchor。
#
# 对热像数据来说，边缘是否“看得见”不只取决于几何尺寸，也取决于边缘两侧温差是否大到超过探测器噪声。这里的目标是找到适合作为 alignment anchor / quality gate 的局部结构，而不是证明所有内部结构都能被 2x 重建。

# %%
observability_summary = cache.observability_summary
snr_reference = cache.snr_reference
segments = cache.segments

display(observability_summary)
display(snr_reference)

print(f"Detected contour segments: {len(segments)}")
print(f"Noise floor: {NOISE_SIGMA:.4f} C")

# %% [markdown]
# ### 🔭 局部轮廓信噪比与可观测性评估
#
# 评估了芯片外部轮廓（Outer Contour）与内部结构轮廓（Inner Contour）的局部温差绝对值、信噪比（SNR）、名义位移的边缘法线投影分量（Normal Projection）以及对齐候选点（Anchor Candidates）的比率。
# 1. **对比度与信噪比**：轮廓两侧的温差绝对值 $|\Delta T|$ 除以探测器噪声底（$\sigma_n = 0.0724^\circ\text{C}$）即为局部信噪比。内轮廓的中位温差接近 $0.3\text{--}0.7^\circ\text{C}$，对应信噪比约为 $4.1\text{--}9.7$ 倍噪声底，表明边缘特征在探测器响应范围内具有显著的可观测性。
# 2. **几何法线投影**：微扫描位移在边缘法线方向的投影分量，表征了采样点跨越边缘进行空间采样的有效度，是保证几何定位精度的关键几何因子。
#
# **💡 算法决策**：局部轮廓的高信噪比与法线方向覆盖率是筛选几何对齐锚点（Alignment Anchor）的核心指标。在后续的对齐与超分辨率阶段，算法应将这些高可观测性段落作为几何配准的门控锚点（Quality Gate），从而保护重构不被低对比度无结构区域的噪声污染。

# %%
show_fig("noise_floor_snr_contrast.png")

# %% [markdown]
# Figure 4: Noise floor and local contour contrast. Local temperature differences are compared with detector noise gates.

# %% [markdown]
# ### 📈 局部温差与噪声门控阈值分布
#
# 局部温差分布与信噪比门控阈值（1x、3x、5x 噪声底）的对比清晰展示了物理边缘的可靠性。实验结果表明，绝大部分实测轮廓段的局部温差都显著超越了 $3\sigma_n = 0.217^\circ\text{C}$ 的噪声门控上限。
#
# **💡 算法决策**：实测轮廓两侧强烈的热对比度证明了将局部结构作为对齐参考的物理合理性。后续算法应采用自适应的 $3\sigma_n$ 作为图像噪声滤波器 and 边缘提取的最低阈值线，以防把噪声斑点误判为轮廓高频特征。

# %%
show_fig("local_contour_candidate_map.png")

# %% [markdown]
# Figure 5: Local contour candidate map. Outer and inner contour candidates are overlaid on the reference thermal frame.

# %% [markdown]
# ### 🗺️ 物理轮廓与配准锚点空间分布图
#
# 该分布图将提取的外轮廓、内轮廓以及候选对齐锚点（Anchor Candidates）叠加在低分辨率参考温度矩阵上，直观指明了可用结构的空间分布状况。候选锚点主要聚集在芯片的电极边缘、金属线以及明显的内部几何边界上，空间拓扑分布合理。
#
# **💡 算法决策**：空间分布的合理性为后续进行多区域局部对齐提供了拓扑支持。在超分辨率重建中，应基于该候选图样在空间上均衡选择配准计算域，确保局部运动向量不仅描述外边框，还能对内部芯片结构形状起到形变纠正作用。

# %%
show_fig("local_anchor_confidence_scatter.png")

# %% [markdown]
# Figure 6: Local anchor confidence scatter. Candidate segments are compared by SNR and normal-direction phase support.

# %% [markdown]
# ### 📊 配准锚点综合置信度分布分析
#
# 分析了各个轮廓段在局部信噪比（SNR）与 $X$ 轴微扫描法线投影分量下的二维散点分布情况。
# 1. **有效锚点筛选**：理想的对齐锚点应处于右上象限（即同时具备高信噪比以抑制温漂，及高法线投影以保障跨边缘采样的亚像素定位精度）。
# 2. **无效锚点剥离**：处于左下象限的低信噪比段在对齐过程中会引入较大的定位噪声，应在几何配准中予以屏蔽。
#
# **💡 算法决策**：散点分布直接定义了局部锚点置信度的过滤规则。在 EP04 及其后续的亚像素对齐流程中，算法将基于该散点统计结果，通过设定信噪比与法线夹角的双重门控，动态剔除低置信度的物理段，保障超分辨率前向投影模型的精度。
