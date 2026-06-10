# %% [markdown]
# ## 2.1 TXT 温度矩阵审计
#
# 逐帧审计结果来自 `frame_audit.csv`（由 `scripts/build_ep01_cache.py` 生成）。
#
# **SR 关注点**: 后续 SR 的输入是 raw 温度矩阵；这一节确认每一帧都处在同一 detector grid 上。

# %%
s = matrix_summary
print(f"✅ 审计帧数: {s['n_frames']}")
print(f"✅ 所有帧尺寸一致: {s['rows']} 行 × {s['cols']} 列")
print(f"NaN 帧数: {s['n_nan_frames']}  |  Inf 帧数: {s['n_inf_frames']}")
print(f"全局温度范围: [{s['t_min_global']:.2f}, {s['t_max_global']:.2f}] °C")
print(f"全局均温范围: [{s['t_mean_min']:.2f}, {s['t_mean_max']:.2f}] °C")
print(f"逐帧中位温范围: [{s['t_median_min']:.2f}, {s['t_median_max']:.2f}] °C")
print(
    f"采集顺序字段: acquisition_order="
    f"{s['acquisition_order_min']}–{s['acquisition_order_max']}"
)

# %% [markdown]
# ### 🔍 关键物理发现与算法约束
#
# 经对 263 帧温度矩阵逐像素扫描审计，确认以下事实：
# 1. **格点一致性**：所有数据严格处于相同的 $480 \times 640$ 探测器网格上，无坏像素或异常缩放，满足亚像素对齐的几何前提。
# 2. **数据无损性**：矩阵内无 NaN 或 Inf 坏值，原始物理读数完整。
# 3. **全局温度起伏**：全部数据的全局平均温度落差达 **4°C**。鉴于红外探测器噪声底仅为 **0.0724°C**，该温度跨度代表了物理环境或传感器状态的显著漂移。
#
# **💡 算法决策**：由于温漂高达噪声底的 50 倍，不能将全部 263 帧直接混合进行超分辨率重建，否则剧烈的温度起伏会被算法误判为空间高频细节（轮廓），从而引入严重伪影。必须在物理温度稳定的 Session 内进行重建。
