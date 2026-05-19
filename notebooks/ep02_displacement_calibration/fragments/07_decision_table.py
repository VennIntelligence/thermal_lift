# %% [markdown]
# ## 6. Evidence-Use Table / 证据使用边界表
#
# EP02 的最终产物不是一个替代 theta 或替代 detector pitch 的新配置，而是一张证据分工表：哪些信息用于 prior，哪些信息支撑 alignment evidence / anchor / quality gate，哪些 pair 只能用于诊断。
#
# 这张双语 Markdown 表是前面所有图表的“使用说明”。它刻意把证据分成可用于 prior、可用于诊断、可用于质量门控、不可用于定量标定等不同层级。
# 对后续 SR 来说，这比单独记住某个数值更重要：同一个指标放错使用位置，就会把局部现象误解释成全局结论。

# %% [markdown]
# | 证据 | Evidence | 可用于 | Use for | 禁止用于 | Do not use for | 原因 / Reason |
# |---|---|---|---|---|---|---|
# | stage/文件名坐标 prior | Stage/filename coordinate prior | 覆盖规划、初始化、正则约束 | coverage planning, initialization, regularization | alignment truth 或成功指标 | alignment truth or success metric | commanded X/Y 是元数据；实际帧间 alignment evidence 需要由图像数据支撑 |
# | 数据驱动 contour/NCC alignment | Data-driven contour/NCC alignment | 2x contour-level SR 前的 alignment anchor 与质量门控 | alignment anchor and quality gate before 2x contour-level SR | 替代物理坐标系统 | replacing the physical coordinate system | held-out contour 与 gradient scores 直接检验图像一致性 |
# | X 方向时间相邻小步 | X time-adjacent small steps | 局部方向与短时间线性 smoke test | local direction and short-time linearity smoke test | 全局 SR 可行性声明或绝对 stage-amplitude truth | global SR feasibility claim or absolute stage-amplitude truth | 这些 pair 采集相邻，但只在一个 ROI/preprocess 选择下探测局部响应 |
# | Y 坐标相邻 pair | Y coordinate-adjacent pairs | raster-path 失败诊断与坐标元数据解释 | raster-path failure diagnosis and coordinate metadata | Y 位移定量标定 | Y displacement calibration | fixed-X Y neighbors 通常隔一整条 raster row，热场演化会污染 NCC |
# | AVI 连续扫描 | AVI continuous scans | 辅助方向验证与命名 sanity check | auxiliary direction and naming sanity check | SR 输入或高精度 theta 替代 | SR input or high-precision theta replacement | AVI 是渲染 8-bit 视频且有大量重复帧，不是 raw 温度矩阵 |

# %% [markdown]
# > **数据说明**: 这张表列出 EP02/后续重建中每类证据的用途、禁止用途和原因。
# > **怎么读表**: 先看“allowed/use”列，确认某类证据能支持什么；再看“forbidden/use”或原因列，确认它不能支持什么。尤其要区分 command prior、局部 NCC/ESF 响应和最终 alignment evidence / anchor / quality gate。
# > **正常/异常理解**: 正常的证据链应是 stage/filename 坐标提供 prior，X 小步提供短时方向/线性诊断，Y 坐标相邻 pair 标记为不适合定量标定，data-driven alignment 承担质量门控。若某一行把 prior 写成 truth，或把 Y-only NCC 写成标定来源，就是需要修正的过度外推。
# > **数据分布**: stage/filename 坐标、X 小步、Y 坐标相邻 pair、AVI 连续扫描和 data-driven alignment 分别落在不同证据层级。
# > **核心发现**: EP02 提供 raster 路径、坐标 prior 和相邻小步诊断；后续 EP06 需要在主 session 上做 data-driven 对齐与质量门控后，再进入 2x contour-level SR。

# %% [markdown]
# ## EP02 Conclusion
#
# - 主 session 是 raster 采集：X 行内连续，Y 行间跳转。
# - `stage_calibration.json` 的 theta/pitch 提供 detector-space prior 和 2x phase 覆盖，不提供 alignment truth。
# - X 行内小步 NCC 是局部方向/线性 smoke test，不能外推成多帧 SR 结论。
# - Y-only 坐标相邻 pair 受 acquisition gap 和热场演化污染，不能做定量 Y 位移标定。
# - 当前可用的 alignment score 显示 data-driven contour/NCC 对齐比 filename/stage prior 更适合支撑 alignment anchor 和质量门控。
#
# 换句话说，EP02 的结论是“如何使用位移证据”，不是“已经完成了 SR 重建”。stage prior 给出合理起点，NCC/ESF 等局部指标帮助诊断可见响应，最终是否进入 2x contour-level SR 还需要依赖主 session 内的 data-driven alignment 质量门控和重建后一致性检查。
