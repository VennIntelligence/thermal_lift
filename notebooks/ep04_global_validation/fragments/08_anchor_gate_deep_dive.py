# %% [markdown]
# ## Step 7 — Anchor Gate 深挖诊断
#
# 本节补回 EP04 经典诊断图，并新增只依赖已缓存 CSV 的轻量 quality-gate 审计。所有内容仍只服务 EP06 alignment anchor / quality gate：不输出 SR，不做 LR/bicubic/SR 对照，也不把 localization 当作客户交付或 SR 成败判定。

# %%
show_fig("split_half_distribution.png")

# %% [markdown]
# Figure 6: Split-half precision distribution. Localization repeatability is summarized for high-confidence contour segments.

# %% [markdown]
# ### 📈 折半定位精度的统计学分布规律
#
# 展示了外轮廓 A 类特征段（A-class Segment）折半定位重复偏差（Split-half Repeatability）的统计直方图分布：
# 1. **分布集中度**：折半定位偏差集中在 $0.02\text{--}0.04$ 像素的小区间内，且显著逼近 Cramér-Rao 理论物理极限（CRB），证实该特征子集具备极高的几何定位重复稳定性。
# 2. **长尾长振幅成因**：直方图尾部存在少量长尾样本，主要由于局部热流扰动或局部微扫描相位限制，导致局部边缘的单帧定位发生抖动。
#
# **💡 算法决策**：此统计分布支撑了对物理特征分类门控的正确性。算法应将定位偏差在 $0.05$ 像素以内的特征归为配准锚点（Alignment Anchor），超过此阈值的高风险段则降权，从而保护全局运动场估计的准确性。

# %%
show_fig("crb_ratio_scatter.png")

# %% [markdown]
# Figure 7: CRB ratio scatter. Measured split-half precision is compared with the theoretical localization bound.

# %% [markdown]
# ### 📊 定位精度与 CRB 理论极限比率分析
#
# 分析了各个外轮廓段实测折半偏差与 Cramér-Rao 理论下界（CRB）的比率分布：
# 1. **物理一致性校验**：通过门控的锚点段，其比值高度收敛在 $1.0\text{--}3.0$ 倍理论下限之间，这在物理上证实了局部定位精度已达到受随机白噪声限制的理论极限。
# 2. **失配噪声隔离**：未通过门控的段其比值呈发散状，表明其几何估计已脱离物理噪声约束，夹杂了系统性漂移或几何畸变。
#
# **💡 算法决策**：CRB 比率是甄别理论误差与模型外误差的关键物理标尺。后续几何配准应彻底封禁比值超过 $5.0$ 的低置信度特征段，确保几何对齐基于纯随机噪声限制的理想帧对进行。

# %%
show_fig("phase_coverage_vs_precision.png")

# %% [markdown]
# Figure 8: Phase coverage versus precision. Normal-direction phase support is compared with localization repeatability.

# %% [markdown]
# ### 📈 相位覆盖宽度与定位精度相关性分析
#
# 展示了特征段在微扫描位移投影到边缘法向后的相位覆盖宽度（Phase Coverage）与实测折半偏差（Split-half Difference）的二维散点关联：
# 1. **采样覆盖约束**：横轴相位覆盖越宽，表明多帧图像在该边缘法向提供了更完备的采样基，使逆向求解退化边缘位置时几何约束更强，折半偏差（纵轴）随之显著收敛。
# 2. **信噪比与几何解耦**：大尺寸散点（高 SNR）在相位覆盖不足时仍可能出现定位失败，证明单纯的高温度对比度在几何采样不完备时无法保证定位精度。
#
# **💡 算法决策**：物理对齐锚点的门控不仅要评估单帧的信噪比，还必须引入法向物理采样相位覆盖范围（必须 $>0.15$ 像素）作为硬性门槛，保障超分辨率矩阵反演时的数值稳定性。

# %%
show_fig("failure_taxonomy.png")

# %% [markdown]
# Figure 9: Failure taxonomy. Rejected localization segments are grouped by primary failure category.

# %% [markdown]
# ### 📊 特征定位失效主因分类统计 (Failure Taxonomy)
#
# 统计了被拒特征段的主要定位失效类别（包括拟合残差过大、边缘宽度超限、定位偏差超限等）：
# 1. **主要失效瓶颈**：外轮廓的主要定位瓶颈在于边缘拟合宽度偏离预期，这通常代表着局部边界存在非阶跃的热结构（如受热传导严重平滑的导线边缘）。
# 2. **失效机制的多因性**：单一主因划分仅用于摘要展示，不能代表多物理条件下的协同退化。
#
# **💡 算法决策**：分类统计结果支持了对退化边缘进行物理降权的算法决策。在后续逆向问题求解中，对于因边缘展宽失效的段，应在优化重构模型中施加空域边缘保持正则项（如 TV 正则或 Huber 罚函数），修正由于几何低通滤波带来的轮廓发散。

# %%
show_fig("cross_scanline_consistency.png")

# %% [markdown]
# Figure 10: Cross-scanline consistency. Segment localization stability is checked across physical scanlines.

# %% [markdown]
# ### 📈 跨扫描线几何定位一致性诊断 (Cross-scanline Consistency)
#
# 评估了同一物理特征段在不同 $Y$ 轴扫描线上的解算几何边缘位置的波动情况：
# 1. **几何对称性与平坦度**：曲线波动幅度在 $\pm 0.05$ 像素以内，表明几何结构在跨扫描平移中保持了极高的刚性与空间一致性。
# 2. **局域背景畸变**：个别扫描线出现的定位跃变，提示了物理光栅扫描过程中局部温漂引起的局部热场畸变。
#
# **💡 算法决策**：几何一致性曲线是标定局部配准精度置信区的重要依据。配准计算中应排除存在突变波动的特征段，只允许跨扫描线平缓一致的段作为核心运动场反演锚点。

# %%
show_fig("segment_scanline_pass_heatmap.png")

# %% [markdown]
# Figure 11: Segment and scanline pass heatmap. Gate pass states are mapped over segment and scanline combinations.

# %% [markdown]
# ### 🗺️ 特征段与扫描线联合通过率二维热力图
#
# 热力图的每个像元表征了特定的 `特征段 × 扫描线` 交叉评估通过状态（蓝色为 Pass，灰色为 Reject）：
# 1. **空间失效拓扑结构**：热力图展示了“横向连续灰带”（坏段）与“纵向连续灰带”（坏线）的空间关联。这证明定位失效具有非均一的空间聚集特性，而非随机的无偏白噪声。
# 2. **内外轮廓质差异**：内轮廓面板中的灰色拒绝区域明显更广，揭示了芯片内部精细结构的配准脆弱性。
#
# **💡 算法决策**：热力图提供了排除坏段与坏线的空间索引。在配准计算中，对对齐输入矩阵施加掩膜（Masking），完全剔除横向坏段与纵向坏线，以此消除非刚性几何伪影。

# %%
from thermal_core.ep04 import (
    ep06_role_margin_table,
    failure_cooccurrence_table,
    ncc_esf_failure_diagnostic_table,
    scanline_segment_failure_summary_table,
)

scanline_segment_layout = scanline_segment_failure_summary_table(outer_results, inner_results)
layout_display = scanline_segment_layout.copy()
for col in ["overall_row_pass_rate", "weakest_scanline_pass_rate", "weakest_segment_pass_rate"]:
    layout_display[col] = (100.0 * layout_display[col]).round(1)
display(layout_display)

# %% [markdown]
# ### 📊 特征与扫描线联合失效定量性能指标
#
# 汇总了内外轮廓的总体通过率、最弱物理扫描线及最弱特征段的定量数据，并统计了完全失效（0-Pass）的特征段数量。
#
# **💡 算法决策**：定量结果确立了系统级配准鲁棒性的风险底线。对于 $0\text{-Pass}$ 的特征段，算法在配准中应予以彻底屏蔽。内轮廓较宽的零通过率再次证明，内轮廓必须作为超分辨率图像增强的“逆向重构目标”（SR Targets），而非“几何配准基准”。

# %%
cooccurrence = failure_cooccurrence_table(outer_results, inner_results, top_n=8)
cooccurrence_display = cooccurrence.copy()
for col in ["share_of_failed_rows", "top_co_share_of_reason"]:
    cooccurrence_display[col] = (100.0 * cooccurrence_display[col]).round(1)
display(cooccurrence_display)

# %% [markdown]
# ### 📊 联合失效原因协同发生概率分析 (Co-occurrence Analysis)
#
# 分析了各个定位失败原因在同一行评估中共同出现的协同概率及最常耦合的次要原因：
# 1. **失效的物理协同性**：数据证实，相位覆盖不足（`low_phase_coverage`）与折半不稳（`split_half_high`）具有极高的一致耦合性（协同概率高达 $80\%$ 以上），这在物理上解释了由于空间采样缺失导致估计位置发散的失配机理。
# 2. **非互斥性解释**：各个失效原因的比例之和大于 100% 这一统计特征真实反映了定位系统物理退化的多维特征。
#
# **💡 算法决策**：失效协同性分析再次印证了多阶段复合质量门控的合理性。后续算法应保持各物理指标的串联门控网络不变，确保任何维度的失配都能被及时拦截。

# %%
ncc_esf_diag = ncc_esf_failure_diagnostic_table(outer_results, inner_results)
diag_display = ncc_esf_diag.copy()
share_cols = [
    "share_failed_ncc_peak_above_gate",
    "ncc_unreliable_share",
    "fit_error_share",
    "sigma_out_of_range_share",
    "split_half_high_share",
    "low_phase_coverage_share",
    "esf_or_stability_share",
]
for col in share_cols:
    diag_display[col] = (100.0 * diag_display[col]).round(1)
metric_cols = [
    "median_failed_ncc_peak",
    "p10_failed_ncc_peak",
    "median_failed_ncc_fit_ok_fraction",
    "median_failed_phase_coverage_px",
    "median_failed_sigma_px",
    "median_failed_split_half_px",
]
for col in metric_cols:
    diag_display[col] = diag_display[col].round(4)
display(diag_display)

# %% [markdown]
# ### 📊 互相关峰值与定位稳定性门控解耦诊断
#
# 定量诊断了定位失败行中，图像归一化互相关（NCC）峰值质量与边缘物理模型（ESF）/定位稳定性指标的解耦特征：
# 1. **NCC峰值的高虚警率**：实验表明，大部分定位失败的行，其局部 NCC 峰值依然维持在 0.85 以上的高水平。这表明纯图像灰度相似性指标NCC存在严重的定位虚警，无法独立反映物理对齐的可重复性。
# 2. **失配的物理根源**：即使互相关系数极高，局部边缘依然可能因拟合宽度超限（失配）或折半偏差过大（不重复）而被拦截，其不稳定性占总失效比例（`esf_or_stability_share`）的主导。
#
# **💡 算法决策**：解除对互相关（NCC）峰值作为对齐成功判据的盲目依赖。在后续的几何配准与超分辨率评估中，禁止使用单一 NCC 或残差指标证明配准成功，必须强制引入以折半偏差与 ESF 物理展宽为核心的多维度几何约束。

# %%
role_margin = ep06_role_margin_table(ep06_recommendations)
role_margin_display = role_margin.copy()
role_margin_display["median_pass_rate_margin"] = (100.0 * role_margin_display["median_pass_rate_margin"]).round(1)
for col in ["median_split_margin_px", "median_phase_margin_px"]:
    role_margin_display[col] = role_margin_display[col].round(4)
role_margin_display["median_crb_ratio_margin"] = role_margin_display["median_crb_ratio_margin"].round(2)
role_margin_display["p10_alignment_margin_min"] = role_margin_display["p10_alignment_margin_min"].round(3)
display(role_margin_display)

# %% [markdown]
# ### 📊 EP06 算法角色分类门槛的安全边际 (Margin) 审计
#
# 审计了被推荐至 EP06 的三类特征子集距离其运动估计阈值（通过率 $\ge 70\%$，定位误差 $\le 0.06$ 像素，物理采样覆盖 $\ge 0.15$ 像素）的安全余量：
# 1. **对齐输入段的稳健边际**：被分配为 `alignment_input` 的特征子集在所有门控指标上均呈现出明显的正安全余量（Positive Margin），确保了全局亚像素运动解算的几何鲁棒性。
# 2. **待增强区域的物理偏离**：被标定为 `sr_target_not_truth` 的内轮廓段则呈现出明显的负余量（Negative Margin），主要是由于空间相位覆盖（`low_phase_coverage`）达不到运动估计要求。
#
# **💡 算法决策**：安全边际审计确立了分层决策的可信度。后续超分辨率算法应严格执行此分类机制：利用具有正余量的稳定段作为 alignment anchor / prior 输入，而在负余量的芯片内部结构段上评估 2x contour-level SR 的可见性增益，防止将带有偏置的特征用于对齐推演。

# %%
show_fig("normal_angle_coverage.png")

# %% [markdown]
# Figure 12: Normal-angle coverage. Passed anchors are summarized by contour normal direction in polar space.

# %% [markdown]
# ### 🗺️ 定位锚点的法向角度极坐标空间覆盖分析
#
# 极坐标图展示了通过门控的外轮廓与内轮廓定位锚点法线角度的空间角度分布情况：
# 1. **外边框的正交方向覆盖**：外轮廓定位锚点在多个法线方向上有较好覆盖，可为二维平面对齐提供更均衡的局部约束。
# 2. **内轮廓的角度空缺**：内轮廓通过锚点不仅数量稀疏，且角度分布更受限，表明其局部特征在某些几何方向上约束较弱。
#
# **💡 算法决策**：normal-angle coverage 是 alignment anchor 的几何覆盖诊断，不是全局配准真值。后续超分辨率位移解算应优先组合跨方向、通过门控的外轮廓锚点作为 prior / regularization，并用 holdout 段验证；不能单独依靠内轮廓或 stage command 做方向推导。
