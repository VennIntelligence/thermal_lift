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
# ### 📝 物理证据边界与算法决策准则
#
# 上述证据使用边界表（Evidence-Use Table）明确了各物理数据在超分辨率重建算法体系中的角色与限制。它将标定先验、图像相似性、时序关系以及辅助视频流划分为不同的应用能级，确立了“因果匹配、严防越界”的数据处理准则。
#
# **💡 算法决策**：
# 1. 严格限制电机指令的定位能级，仅将其作为优化计算的全局几何先验（Prior），避免将其误判为对齐真值导致累积偏差。
# 2. 封禁 $Y$ 轴空间相邻帧对的直接标定功能，转由全局对齐锚点（EP04 Localization Gate）接管。
# 3. 将连续视频 AVI 作为辅助方向验证的第二信源，保持系统物理参数的稳定性。

# %% [markdown]
# ## 🏁 EP02 研究结论与后续算法指导
#
# 1. **光栅扫描时序拓扑**：确认主 Session 采用 Step-and-Shoot 逐行光栅扫描模式。行内 $X$ 轴平移属于时序连续帧，而行间 $Y$ 轴平移受限于约 16 帧的时序延迟。
# 2. **物理先验有效性**：验证了由标定旋转角 $\theta = 47.6^\circ$ 和探测器间距 $10.0\,\mu\text{m}$ 构成的几何转换矩阵在 $2 \times 2$ 超分辨率子网格上具有完整的半像素空间相位覆盖。
# 3. **失效机理与规避**：阐明了 Y-only 空间相邻帧对因热场演化造成的互相关单调性失效，为后续几何配准算法排除了“局部标定 $Y$ 位移”的错误路线。
# 4. **对齐策略定位**：确立了“位移先验提供初始化分布，数据驱动提供最终配准锚点”的分工模式。后续的 2x contour-level 超分辨率重建流程应基于此对齐架构设计质量控制闸门。
