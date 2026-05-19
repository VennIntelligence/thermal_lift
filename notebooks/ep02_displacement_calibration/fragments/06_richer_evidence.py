# %% [markdown]
# ## 5. Richer Evidence for the Current EP02 Mainline
#
# 前面几节已经把 raster path、stage prior 和 data-driven alignment 的分工讲清楚。本节把 EP02 历史补强产物重新收拢成当前主线可用的证据：时间相邻方法对照、Y-only 失败诊断、AVI theta 方向旁证、AVI-TXT 命名映射，以及旧 coordinate-adjacent NCC 失败审计。
#
# 这些表都不是新的配置写入。它们的共同目的，是告诉后续 EP04/EP05/EP06：哪些证据可以作为 prior、哪些只能作为局部 smoke test、哪些失败只能说明 pair 构造不合适，不能外推成 SR no-go。

# %%
prior_contract = stage_prior_contract_table(
    frame_audit,
    theta_deg=REFERENCE_THETA_DEG,
    pixel_size_um=PIXEL_SIZE_UM,
    max_rows=10,
)
display(prior_contract)

# %% [markdown]
# > **数据说明**: 这张表取主 session 前 10 帧，展示每帧由 `X/Y` command、theta=47.6 deg 和 10 um/pixel pitch 换算出的 detector-space prior dx/dy 与 2x phase bin。
# > **怎么看**: `stage_prior_dx_px` 和 `stage_prior_dy_px` 是命令坐标投到 detector 坐标后的预期位移；phase 列只说明它在 2x 半像素网格中的位置。`contract` 列明确写出这些数值只能用于 prior、初始化或正则。
# > **正常/异常理解**: 这些数值来自元数据和配置，不来自 NCC 或轮廓配准；即使 phase 覆盖合理，也不表示每一帧已经真实对齐。
# > **核心发现**: EP02 可以把 stage prior 明确传给后续重建，但后续 alignment evidence、anchor 和质量门控仍必须由 data-driven contour/NCC 支撑。

# %%
time_adjacent_methods = time_adjacent_method_comparison(OUTPUT_DIR)
display(time_adjacent_methods)

# %% [markdown]
# > **数据说明**: 这张表读取 `time_adjacent_method_summary.csv`，比较 raw NCC、high-pass NCC、gradient NCC 和 phase correlation 在真实时间相邻 X-step 与 row-transition 上的 projection ratio、RMS 残差和峰值分数。
# > **怎么看**: `visible/prior projection` 越接近 1，表示当前预处理下 NCC 可见投影越接近 stage prior；`RMS vs prior` 越小，表示相对固定 10 um/pixel prior 的残差越小。peak score 只是匹配峰强度，高分不等于物理位移正确。
# > **正常/异常理解**: X-step 的 acquisition gap=1，因此可以作为局部方向和短时线性 smoke test。row-transition 虽然也时间相邻，但它同时包含 Y advance 和 X reset，不是干净的 Y-only 小步；phase correlation 在极小位移上退化到 0 px 是方法限制，不是 stage 不动。
# > **核心发现**: X 小步只能证明当前 ROI/预处理下存在可见的短时响应；它不能把 stage command 升级为 alignment truth，也不能裁判多帧 contour-level SR 成败。

# %%
y_failure = y_coordinate_failure_table(OUTPUT_DIR)
display(y_failure)

# %% [markdown]
# > **数据说明**: 这张表读取 `y_coordinate_method_summary.csv`，把 Y-only 坐标相邻 pair 的 2 um 与 4 um 结果按预处理方法汇总。
# > **怎么看**: 如果这些 pair 可用于定量 Y 标定，4 um 的可见投影应约为 2 um 的两倍，因此 `visible 4um/2um` 应接近 2。`RMS 2um / 4um` 越小只说明局部拟合残差较小，不能修复非单调性。
# > **正常/异常理解**: 三种预处理都给出约 0.64 的 4/2 比例，说明失败跨 raw/high-pass/gradient 稳定存在；这不是某个滤波器偶然失败，而是 pair 构造受 raster acquisition gap 和热场演化污染。
# > **核心发现**: 固定 X 的 Y 坐标相邻 TXT pair 不能作为 Y 位移定量标定。它们可以保留为命令坐标元数据和失败诊断证据。

# %%
avi_theta_table = avi_theta_compact_table(OUTPUT_DIR, reference_theta_deg=REFERENCE_THETA_DEG)
display(avi_theta_table)

# %% [markdown]
# > **数据说明**: 这张表读取 `avi_theta_summary.csv`，把 AVI 连续扫描方向换算成 theta 的 X-only、Y-only 和 combined 估计。
# > **怎么看**: 重点看 `gradient / combined` 行：mean theta 约 47.14 deg，95% CI 覆盖配置中的 47.6 deg。X-only 与 Y-only 分别偏在两侧，说明 AVI 派生几何里还有系统差异。
# > **正常/异常理解**: AVI 是渲染后的 8-bit 视频，且约 67% 为重复帧；它能做连续运动方向旁证，但不能替换 raw 温度矩阵，也不能提供更高精度的配置 theta。
# > **核心发现**: AVI gradient combined 支持 47.6 deg 配置的方向合理性，但 `configs/stage_calibration.json` 不应由 AVI 结果覆盖。

# %%
forest_plot_path = OUTPUT_DIR / "avi_theta_forest_plot.png"
if forest_plot_path.exists():
    from IPython.display import Image

    display(Image(filename=str(forest_plot_path)))
else:
    display(pd.DataFrame({"note": [f"Missing {forest_plot_path.name}"]}))

# %% [markdown]
# > **图表说明**: 这张森林图展示逐个 AVI 文件的 theta 估计，以及 combined summary 与 47.6 deg 参考线的位置关系。
# > **怎么看**: 每条横线是一个 AVI 方向估计的不确定区间；点和区间越集中，说明同一类扫描内部方向越稳定。参考线落入 gradient combined CI，表示独立方向旁证与配置一致。
# > **正常/异常理解**: X-scan 和 Y-scan 的分组中心存在约 3 deg 差异，这是 AVI 证据不能直接替换配置的主要原因。图中 tight subgroup 不等于高精度全局标定。
# > **核心发现**: AVI 方向证据的正确使用方式是辅助验证 theta 方向，而不是生成新的 stage calibration。

# %%
avi_txt_match = avi_txt_line_match_table(OUTPUT_DIR)
display(avi_txt_match)

# %% [markdown]
# > **数据说明**: 这张表读取 `avi_txt_xline_match_summary.csv` 和 `avi_txt_yline_match_summary.csv`，比较 xN/yN AVI 文件名与 TXT 固定 X/Y 线的对应关系。
# > **怎么看**: 轴差越小，说明该 TXT 线的轮廓/NCC 方向越接近对应 AVI 连续扫描方向。`decision=expected` 的行应比 `rejected` 行小得多。
# > **正常/异常理解**: xN.avi 对应 TXT fixed Y=N，yN.avi 对应 TXT fixed X=N；这说明 AVI 前缀表示运动轴，数字表示固定的正交坐标。Y expected 行的 acquisition gap 中位数仍为 16，正是 Y-only TXT NCC 失败的关键背景。
# > **核心发现**: 没有证据表明 x/y 命名映射反了。Y-only TXT 失败主要来自 raster 路径下的非时间相邻 pair 和热场演化，而不是文件命名或坐标轴解释错误。

# %%
historical_failure = historical_ncc_failure_audit(OUTPUT_DIR)
display(historical_failure)

# %% [markdown]
# > **数据说明**: 这张表把旧 coordinate-adjacent NCC 的 theta、线性度、repeatability 和 Y-only 单调性问题收拢成失败审计。
# > **怎么看**: 这些值回答的是“旧 pair 构造为什么不能独立标定 theta 或 Y 位移”。例如 old theta CI 不覆盖 47.6 deg、projection R2 接近 0、repeat pair 都不可用，都是失败诊断。
# > **正常/异常理解**: 旧结果不应恢复成“更新 theta”或“SR no-go”叙事。它们只说明 coordinate-adjacent NCC 把真实采集时间差和热场演化混进了位移估计。
# > **核心发现**: 历史 NCC 失败现在只作为证据边界存在：它提醒我们不要用坐标相邻替代时间相邻，也不要用局部 NCC 失败裁判 contour-level SR。
