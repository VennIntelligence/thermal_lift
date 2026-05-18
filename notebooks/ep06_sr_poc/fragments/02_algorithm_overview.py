# %% [markdown]
# ## 1. Algorithm Overview
#
# EP06 比较 5 个 2x contour-level 输出：
#
# | 方法 | 角色 | 输出解释 |
# |---|---|---|
# | LR reference | 原始采样参考 | 只作为当前 detector grid 的直接对照 |
# | Bicubic reference | 显示倍率 baseline | 只说明插值外观，不提供新信息 |
# | SAA-uniform | 多帧微扫描 baseline | 直接用 EP05 正向 alignment shift 回填到 2x 网格 |
# | SAA-weighted | 质量门控 baseline | 用 EP05 NCC/held-out Chamfer proxy 加权 |
# | IBP | forward-model baseline | 预测观测时由 forward model 对 reference HR 图施加反向位移 |
# | MAP-TV | 正则化物理 baseline | 用 split-half consistency 选择 lambda |
#
# 统一位移约定：
#
# - `align_dx_px/align_dy_px` 和 `refined_align_dx_px/refined_align_dy_px` 表示把 LR 帧移动到 reference 坐标系的位移，单位 LR pixel。
# - SAA 是回填式对齐，直接使用正向 shift。
# - IBP/MAP-TV 是观测预测式算法，脚本把同一 shifts 传给算法，forward model 内部负责反向位移。
#
# 目标函数：
#
# `IBP`: 迭代最小化观测残差，将 `y_i - H_i(x)` back-project 到 HR grid。
#
# `MAP-TV`: 最小化 `0.5/N * sum_i ||y_i - H_i(x)||^2 + lambda * TV(x)`，lambda 由 split-half consistency proxy 选择。
#
# **边界声明**: 这里的 highpass SR 是结构图；raw-temperature track 是控制轨。输出 2x grid 只是重建网格，不声明 4x，也不声明 5 um 计量级实际分辨率。
