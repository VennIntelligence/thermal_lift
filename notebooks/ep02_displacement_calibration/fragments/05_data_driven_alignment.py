# %% [markdown]
# ## 4. Data-Driven Alignment vs Filename/Stage Prior
#
# EP02 的坐标 prior 需要和 data-driven alignment 分工：prior 提供覆盖、初始化和约束；alignment evidence / anchor / quality gate 由图像数据中的 contour/NCC 一致性支撑。若 EP05 alignment score 已存在，本节直接读取；否则 core helper 会退回到轻量 EP02 NCC proxy。
#
# 这一节回答“后续 SR 到底信谁”。stage prior 告诉我们应该从哪里开始找；data-driven alignment 则用热像帧中的边缘、梯度、轮廓或 NCC 证据检查这个位移是否真的让结构对齐。
# 因此，prior 和 alignment 不是互相替代的两套结论，而是前后衔接的两层证据。

# %% [markdown]
# ### 读图前：五种对齐策略与 (a)(b) 指标说明
#
# 下图（Figure 4）比较 **五种几何配准策略** 在主 session 248 帧 clean set 上的表现。每一行横条代表一种方法；若 EP05 alignment score 已生成则读取完整五法对比，否则退回到 EP02 轻量 NCC proxy（仅两行）。
#
# | 方法 | 英文标签 | 含义 |
# |------|----------|------|
# | 无对齐 | No alignment | 不做位移修正，直接把各帧当作已对齐（baseline，通常最差） |
# | 电机标定先验 | Stage prior only | 把电动台命令坐标 (X, Y) 经旋转角 θ=47.6° 映射为像素位移；**仅作 prior/初始化，不是对齐真值** |
# | 文件名仿射先验 | Filename affine prior | 用文件名坐标在全 session 上拟合仿射映射，比 stage prior 多一步数据拟合，但仍非图像精对齐 |
# | 数据驱动 NCC 初值 | Data-driven NCC init | 在 ROI 内用高通 + NCC 从图像估出每帧亚像素位移，作为 data-driven 初值 |
# | 数据驱动轮廓精化 | Data-driven contour refined | 在 NCC 初值基础上，用轮廓 Chamfer 做局部 refine，是当前推荐的对齐 anchor |
#
# **Stage prior 与 data-driven 的分工**：prior 告诉我们「应该从哪开始找」；data-driven 用图像中的边缘/梯度/轮廓证据检验「这个位移是否真的让结构对齐」。两者是前后衔接的两层证据，不是互相替代的两套结论。
#
# #### 子图 (a) Contour Holdout Error — 轮廓 holdout 误差
#
# - **横轴**：`median holdout Chamfer [px]`，单位像素；**柱子越短越好**。
# - **Chamfer 距离**：对齐后，当前帧边缘点平移到参考帧上，离参考轮廓还有多远；越小表示轮廓贴得越准。
# - **Holdout（留出）**：对齐优化时只用一半边缘点；评估时用 **另一半未参与优化的点** 算 Chamfer，避免「对着答案调参」。
# - **怎么看**：例如 No alignment ≈ 0.40 px、Stage prior only ≈ 0.25 px、Data-driven contour refined ≈ 0.13 px，说明 data-driven 显著减小轮廓错位。
#
# #### 子图 (b) Contour/Gradient Agreement — 轮廓/梯度一致性
#
# - **横轴**：`median gradient correlation`，范围约 0–1；**柱子越长越好**。
# - **梯度相关系数**：对齐后，比较当前帧与参考帧在边缘附近 **梯度方向/强度** 是否一致；越大表示局部边缘结构越吻合。
# - **与 (a) 互补**：Chamfer 主要看轮廓 **位置** 是否对齐（可能贴到错误边缘）；gradient correlation 主要看边缘 **纹理/方向** 是否一致（可能忽略整体偏移）。两者需交叉阅读。
#
# #### 能得出什么、不能得出什么
#
# | 能说明 | 不能说明 |
# |--------|----------|
# | 在主 session 上 data-driven 对齐比单纯信 stage command 更可靠 | 真实物理分辨率已达 5 µm |
# | 可作为后续 2× contour-level SR 的 alignment quality gate | 温度计量精度或 SR 最终成功 |
# | 「先验引导、数据精化」的双层位移逻辑成立 | 单独用任一指标宣告 SR 成功 |
#
# > **图表说明**：Figure 4 为五种（或 proxy 两行）对齐策略的 holdout Chamfer 与 gradient correlation 对比。
# > **数据分布**：通常 No alignment 误差最大；Stage prior 居中；Data-driven refined 在 (a) 最短、(b) 较高。
# > **核心发现**：图像证据支持的数据驱动对齐优于仅依赖电机指令先验，后续 SR 应以 data-driven contour refined 作为对齐 anchor。

# %%
import pandas as pd
from thermal_core.ep02 import alignment_improvement_summary, load_alignment_comparison

alignment_summary, alignment_source = load_alignment_comparison(PROJECT_ROOT)
show_fig("ep02_data_driven_alignment_comparison.png")

# %% [markdown]
# Figure 4: Data-driven alignment comparison. Alignment strategies are compared by held-out contour and gradient metrics.

# %% [markdown]
# ### 📈 数据驱动对齐与名义位移先验的对比评估
#
# 针对不同几何配准策略（仅名义先验对齐 vs. 数据驱动优化对齐）在独立测试集（Holdout Set）上的 Chamfer 轮廓距离中位数及梯度互相关系数的对比分析：
# 1. **几何特征评估维度**：Chamfer 轮廓误差表征对齐后图像边缘与参考轮廓的几何吻合程度，值越小代表几何一致性越高；梯度相关系数衡量重构图像局部边缘梯度的方向吻合度，值越大表明边缘高频方向越趋于真实分布。
# 2. **配准表现增益**：实验结果表明，引入图像灰度及轮廓梯度作为约束的数据驱动对齐算法（Data-driven Alignment），在 Chamfer 误差下降与梯度一致性提升两个维度上均显著优于仅使用电机指令先验（Stage prior only）的方案。
#
# **💡 算法决策**：实验结果证实了“先验引导、数据精化”的双层位移估计逻辑的正确性。在后续的超分辨率重构中，名义位移先验提供全局收敛的采样覆盖，而数据驱动的亚像素配准则修正制造装配及运动机械误差，防止由于几何错位在重构图像中产生运动模糊或伪影。

# %%
alignment_gain = alignment_improvement_summary(alignment_summary)
display(
    alignment_summary[
        [
            "display_label",
            "holdout_chamfer_median_px",
            "holdout_chamfer_p90_px",
            "gradient_corr_median",
            "gradient_corr_p10",
            "shift_norm_median_px",
        ]
    ]
)
display(alignment_gain if not alignment_gain.empty else pd.DataFrame({"note": [f"Alignment source: {alignment_source}"]}))

# %% [markdown]
# ### 📊 配准优化定量性能增益评估
#
# 汇总了各对齐算法在 Holdout 轮廓 Chamfer 距离分位数（中位数及 90% 分位数）与梯度相关系数（中位数及 10% 分位数）上的定量表现。相较于名义先验，数据驱动算法在削减几何偏差上限（P90 Chamfer）和改善极端恶劣样本对齐下限（P10 Gradient Correlation）方面展现了明确的物理增益。
#
# **💡 算法决策**：数据驱动对齐不仅降低了平均几何误差，还显著收窄了对齐误差的尾部分布（P90 误差收敛），这对于确保多帧超分辨率算法在极端帧条件下的鲁棒性至关重要。因此，超分辨率重构流程应锁定该对齐优化分工，并以此作为进入亚像素重构的质量门控标准。
