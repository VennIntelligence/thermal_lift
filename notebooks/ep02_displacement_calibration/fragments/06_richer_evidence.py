# %% [markdown]
# ## 5. Richer Evidence for the Current EP02 Mainline
#
# 前面几节已经把 raster path、stage prior 和 data-driven alignment 的分工讲清楚。本节把 EP02 历史补强产物重新收拢成当前主线可用的证据：时间相邻方法对照、Y-only 失败诊断、AVI theta 方向旁证、AVI-TXT 命名映射，以及旧 coordinate-adjacent NCC 失败审计。
#
# 这些表都不是新的配置写入。它们的共同目的，是告诉后续 EP04/EP05/EP06：哪些证据可以作为 prior、哪些只能作为局部 smoke test、哪些失败只能说明 pair 构造不合适，不能外推成 SR no-go。

# %%
from thermal_core.ep02 import (
    avi_theta_compact_table,
    avi_txt_line_match_table,
    historical_ncc_failure_audit,
    stage_prior_contract_table,
    time_adjacent_method_comparison,
    y_coordinate_failure_table,
)

prior_contract = stage_prior_contract_table(
    frame_audit,
    theta_deg=REFERENCE_THETA_DEG,
    pixel_size_um=PIXEL_SIZE_UM,
    max_rows=10,
)
display(prior_contract)

# %% [markdown]
# ### 🗺️ 探测器空间位移先验映射协议 (Stage Prior Mapping)
#
# 该表格截取了主 Session 前 10 帧的名义映射结果，具体展示了如何将电动台指令坐标 $(x_{\text{um}}, y_{\text{um}})$，结合系统标定的旋转角 $\theta = 47.6^\circ$ 和探测器像元大小 $10.0\,\mu\text{m/pixel}$，投影转换至探测器图像空间的名义位移 $(\Delta x_{\text{nominal}}, \Delta y_{\text{nominal}})$ 及其对应的 2x 半像素相位区间。
#
# **💡 算法决策**：此名义映射协议构成后续重构的基础先验。名义位移值可直接用于全局对齐的初始条件与非凸重构的正则项，但不能当作真实的物理配准位移使用，必须在后续阶段通过基于图像内容的数据驱动对齐算法予以精确修正。

# %%
time_adjacent_methods = time_adjacent_method_comparison(OUTPUT_DIR)
display(time_adjacent_methods)

# %% [markdown]
# ### 🔄 时间相邻位移估计方法对比
#
# 对比了 Raw NCC、High-pass NCC、Gradient NCC 以及相位相关（Phase Correlation）方法在时序连续 $X$ 轴步进帧（Acquisition Gap = 1）与换行转移帧（Row-transition）上的性能指标：
# 1. **物理响应表征**：在时序连续的 $X$ 轴步进过程中，各 NCC 变体估计的可见投影与先验的比例（Visible/Prior Ratio）均表现出良好的方向响应度。
# 2. **相位相关方法限制**：由于步进位移极小，传统的相位相关（Phase Correlation）容易出现亚像素分辨率退化甚至归零的现象，表明该方法在细粒度扫描场景下适用性受限。
#
# **💡 算法决策**：互相关算法在时序连续帧对上的局部响应可用于对齐方向的快速自检，但不足以支撑全局超分辨率所需的全部几何配准。后续超分辨率配准算法应规避直接采用相位相关法进行细微位移估计的缺陷。

# %%
y_failure = y_coordinate_failure_table(OUTPUT_DIR)
display(y_failure)

# %% [markdown]
# ### 📉 Y轴空间相邻物理帧配准失效机理诊断
#
# 汇总了在固定 $X$ 轴、仅改变 $Y$ 轴坐标的相邻帧对上，不同滤波器和图像特征处理下的位移单调性表现：
# 1. **单调性违背现象**：三种不同的预处理方式在 $2\,\mu\text{m}$ 与 $4\,\mu\text{m}$ 步长下得到的 NCC 可见投影比值（Visible 4um/2um）均稳定在 0.64 左右，严重背离了物理位移本应具有的线性递增趋势（比值应接近 2.0）。
# 2. **物理本质剖析**：该失效模式具有全局一致性，其根源并非滤波器缺陷，而是由光栅扫描轨迹造成的。空间相邻但固定 $X$ 的 $Y$ 轴帧对，在物理采集时序上相隔整行扫描周期，这导致探测器的热平衡状态发生演化，温度场的动态漂移彻底污染了互相关函数的峰值搜索。
#
# **💡 算法决策**：从物理机制上彻底排除利用 Y-only 空间相邻帧对进行定量标定的可行性。在后续的几何配准中，任何针对 $Y$ 轴方向位移的估计，都必须放在全局时序对齐的约束框架（如全局优化或 localization anchor 质量门控）下进行，严禁使用孤立的空间相邻帧对作局部标定。

# %%
avi_theta_table = avi_theta_compact_table(OUTPUT_DIR, reference_theta_deg=REFERENCE_THETA_DEG)
display(avi_theta_table)

# %% [markdown]
# ### 🔭 基于连续扫描 AVI 的旋转角 $\theta$ 旁证分析
#
# 汇总了利用 AVI 连续扫描视频流估计旋转角 $\theta$ 的统计结果。多组 $X$ 轴与 $Y$ 轴扫描估计合并得到的综合旋转角中位数为 $47.14^\circ$，其 95% 置信区间 [46.36°, 47.92°] 完整覆盖了全局标定配置中的 $47.6^\circ$ 基准值。
#
# **💡 算法决策**：虽然连续扫描 AVI 的估计结果从物理上验证了 $47.6^\circ$ 配置的方向合理性，但由于 AVI 是渲染后的 8-bit 低动态图像且存在大量重复帧（去重前约 67%），其绝对精度不足以用于更新全局标定参数。故后续的超分辨率重构依然保持 $47.6^\circ$ 物理旋转先验不变。

# %%
bracket_plot_path = cache.figure_path("avi_theta_bracket_plot.png")
if bracket_plot_path.exists():
    show_fig("avi_theta_bracket_plot.png")

# %% [markdown]
# Figure 5a: AVI theta pooled bracket summary. Y-scan, combined, and X-scan pooled estimates bracket the configured reference near 47°.

# %% [markdown]
# ### 📐 旋转角汇总哑铃图（汇报主图）
#
# 上图将 16 路 AVI 独立估计压缩为三个 pooled 汇总点，横轴收窄到 $44^\circ$–$50^\circ$，便于汇报时一眼读出「旋转角大约 $47^\circ$」：
# 1. **Y-scan pooled**（约 $45.6^\circ$）与 **X-scan pooled**（约 $48.7^\circ$）从两侧夹住配置参考线 $47.6^\circ$；底部 bracket 标出这一「包络」关系。
# 2. **Combined pooled** 综合估计为 $47.14^\circ$，与参考值仅差 $0.46^\circ$；浅绿色带为 combined 95% CI $[46.36^\circ, 47.92^\circ]$，完整覆盖 $47.6^\circ$。
#
# **💡 算法决策**：哑铃图适合 PPT 主汇报；它支持「独立验证 $\theta \approx 47^\circ$」的结论，但不改变 AVI 仅作旁证、不替换 `stage_calibration.json` 的决策。

# %%
forest_plot_path = cache.figure_path("avi_theta_forest_plot.png")
if forest_plot_path.exists():
    show_fig("avi_theta_forest_plot.png")

# %% [markdown]
# Figure 5b: AVI theta forest plot. Independent continuous-scan estimates are shown with uncertainty intervals.

# %%
if not bracket_plot_path.exists() and not forest_plot_path.exists():
    display(
        pd.DataFrame(
            {
                "note": [
                    "Missing avi_theta_bracket_plot.png / avi_theta_forest_plot.png "
                    "— run build_ep02_cache.py with AVI data present"
                ]
            }
        )
    )

# %% [markdown]
# ### 🌲 旋转角估计的不确定度森林图分析（技术附录）
#
# 森林图展示了各路 AVI 独立估计的旋转角及其 95% 置信区间的分布情况。
# 1. **数据一致性**：大部分独立估计的置信区间均包络了 $47.6^\circ$ 基准参考线，在统计学上证实了该旋转先验在全局物理系统中的可信度。
# 2. **轴间系统偏差**：$X$ 轴扫描与 $Y$ 轴扫描的估计中心点存在约 $3^\circ$ 的系统性偏差，表明在连续扫描模式下系统存在非对称的动力学延迟或图像重建伪影。
#
# **💡 算法决策**：鉴于 AVI 视频源存在非对称物理延迟，其定位精度受限，森林图的分析结果再次印证了不能使用 AVI 数据作为最终物理旋转参数更新源的决策。

# %%
avi_txt_match = avi_txt_line_match_table(OUTPUT_DIR)
display(avi_txt_match)

# %% [markdown]
# ### 🏷️ 视频流与温度矩阵坐标命名映射一致性审计
#
# 审计了连续扫描 AVI 文件名（xN.avi, yN.avi）与 TXT 温度矩阵中对应固定行/列（Fixed X/Y = N）的坐标命名对应关系。轴差指标接近于零，表明图像命名规则在物理硬件和软件层面具有高度一致性，没有发生坐标轴混淆。
#
# **💡 算法决策**：排除文件命名混淆或坐标轴定义颠倒导致 $Y$ 轴位移标定失效的假说。这进一步确证了 $Y$ 轴相邻物理帧互相关失效的唯一根源是光栅扫描引起的时序温漂污染，而非命名或软件映射错误。

# %%
historical_failure = historical_ncc_failure_audit(OUTPUT_DIR)
display(historical_failure)

# %% [markdown]
# ### 🕵️ 历史互相关标定失效审计
#
# 汇总并审计了早期尝试利用空间相邻帧对直接进行 theta 角与 Y 轴位移标定遭遇失败的技术原因。
# 1. **失效表现**：历史方法由于未剔除时序温漂，导致标定置信区间不覆盖物理真值、判定系数 $R^2$ 趋近于零、重复测量一致性极差等。
# 2. **物理解释**：失败的本质是未能解耦空间位移与时间维度的热平衡演化。
#
# **💡 算法决策**：历史标定失效的诊断为本算法体系确立了“禁止直接在空间相邻但时序非连续的帧对上进行局部定量标定”的底线。这也指明了后续 2x contour-level SR 重建绝不能使用简单的局部配准，而必须应用 EP04 localization 全局对齐锚点和质量门控。
