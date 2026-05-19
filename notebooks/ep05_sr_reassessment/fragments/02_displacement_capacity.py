# %% [markdown]
# ## 1. Alignment and Displacement Capacity Preface
#
# EP05 先检查主 session 是否真的提供了可见位移和二维相位覆盖，再讨论具体的对齐方法。这里读的是 `output/ep05_sr_reassessment/` 的 displacement reassessment 产物：它们来自按 `acquisition_order` 排列的主 session 轨迹和 highpass NCC 位移诊断。
#
# 这页只回答“数据里是否有足够可见的微扫描运动可供 2x baseline 使用”。stage command 仍然只是位移 prior；可见 NCC shift 是 alignment 诊断，也不是全局 ground truth。

# %%
trajectory_table = trajectory_capacity_table(displacement_outputs["summary_json"])
display(trajectory_table)

# %% [markdown]
# > **数据说明**: 这张表概括主 session 的累计轨迹：帧数、坐标覆盖、可见位移 span 和累计路径长度都来自 `displacement_reassessment_summary.json`。
# > **怎么看**: `Cumulative span` 和 `Cumulative path length` 越大，说明主 session 中可用于 phase/alignment 诊断的可见运动越充分；但这些量是从图像响应估计出来的路径诊断，不是把 stage command 当作真值。
# > **正常/异常理解**: 如果主 session 帧数不是 255，或累计 span 接近 0，EP06 的 2x SR POC 就会缺少基础输入；如果 span 有二维覆盖，说明它不是只沿单条边缘重复采样。路径长度大于 endpoint span 是正常的，因为 raster acquisition 会逐步走完整个扫描路径。
# > **核心发现**: 主 session 提供 255 帧和约 `2.88 x 9.04 px` 的可见累计 span，足以作为进入 EP06 前的 2x phase/alignment baseline 输入；它不支持把任一局部位移估计外推为全局真实位移。

# %%
visible_shift_table = visible_shift_key_table(
    displacement_outputs["measurements"],
    displacement_outputs["summary_by_class"],
)
display(
    visible_shift_table.round(
        {
            "command_norm_px": 4,
            "visible_norm_median_px": 4,
            "visible_norm_p90_px": 4,
            "projection_median_px": 4,
            "peak_ncc_median": 4,
        }
    )
)

# %% [markdown]
# > **数据说明**: 表格列出 highpass NCC 下最关键的位移诊断：主 session 行内 2/4 µm 小步、40 µm endpoint、以及 raster row reset。`command_norm_px` 是 stage/filename 模型给出的命令幅值，`visible_norm_*` 是热像数据中估计到的可见位移幅值，`projection_median_px` 是可见位移投影到命令方向上的中位数。
# > **怎么看**: 行内 2/4 µm 小步用于看短时间、同一扫描线上的位移响应；endpoint 行用于看更长跨度的二维覆盖。`peak_ncc_median` 越高说明局部相关峰越稳定，但它仍只是 alignment 诊断。
# > **正常/异常理解**: 可见位移小于 stage command 是允许的，因为热像结构、PSF、噪声和局部温度场都会衰减图像响应。row reset 和 Y column 的数值不能直接拿来替代细步进标定，因为它们跨越更长 acquisition gap，热场演化会进入误差项。
# > **核心发现**: 2 µm 与 4 µm 行内步进在 highpass NCC 中呈现约 0.097 px 与 0.192 px 的可见响应，40 µm endpoint 提供 px 级相位跨度。这支持“2x baseline 可做”，但仍不支持把 stage command 或局部 NCC shift 当作对齐真值。

# %%
display(Image(filename=str(DISPLACEMENT_DIR / "main_session_cumulative_trajectory.png")))
display(Image(filename=str(DISPLACEMENT_DIR / "visible_shift_by_pair_class.png")))
display(Image(filename=str(DISPLACEMENT_DIR / "endpoint_displacement_vectors.png")))

# %% [markdown]
# > **图表说明**: 三张图分别展示主 session 按采集顺序累计出来的可见轨迹、不同 pair class 的 visible shift 分布、以及 endpoint displacement vector。它们都是 displacement capacity 的 sanity check，不是 SR 输出图。
# > **怎么看**: 累计轨迹图看二维路径是否展开；pair-class 图看小步、row reset、endpoint 的响应是否分层；endpoint vector 图看长跨度方向是否与扫描几何一致。图中较大的 endpoint 或 row-reset 响应只说明覆盖范围，不说明高倍率 SR 已经成立。
# > **正常/异常理解**: 可见轨迹不必等于 stage-command 轨迹，尤其 Y-only 或 row-reset 类 pair 更容易受 acquisition gap 和热漂移影响。若 pair-class 分布完全混在一起，说明 displacement 诊断不具备区分能力；若 endpoint 向量方向大面积反常，则需要回到 stage/像素旋转模型和 acquisition order 检查。
# > **核心发现**: 三张图共同给出进入 EP06 的前置证据：主 session 有可见的二维微扫描运动，但 EP05 仍只建立 2x phase/alignment baseline，不证明 4x，也不把任何 overlay 或 command shift 当作 SR 指标。
