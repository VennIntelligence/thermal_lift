# %% [markdown]
# ## 3.1 温度统计
#
# 四子图: 逐帧均温/标准差/最小/最大 直方图。
# 时间线: 按真实采集顺序的温度变化趋势，可直观看出预热/补采帧和主扫描温度带。

# %%
plot_temperature_histograms(df,
                            save_path="frame_temperature_statistics.png", save_fn=save_fig)

# %% [markdown]
# > **图表说明**: 四子图直方图，分别展示 263 帧的均温 (T_mean)、温度标准差 (T_std)、
# > 最低温 (T_min)、最高温 (T_max) 的分布。
# >
# > **数据分布**: 均温集中在 19.5–22°C 主峰，但在 ~23–24°C 出现明显的第二峰，
# > 提示数据采集跨越了不同温度状态的 session。标准差分布窄而集中，
# > 说明单帧内温度梯度在帧间高度一致。
# >
# > **核心发现**: 温度的**双峰分布**是 session 跳变的直接证据 — 不同采集批次之间
# > 存在 3–4°C 的温度状态差异。这验证了「跨 session 帧绝不能混合使用」的硬约束。

# %%
df_filename = df.sort_values("file").reset_index(drop=True)
df_acquisition = df.sort_values(["acquisition_order", "file"]).reset_index(drop=True)

plot_temperature_timeline(
    df_acquisition,
    order_label="acquisition order",
    save_path="temperature_timeline.png",
    save_fn=save_fig,
)

# %% [markdown]
# > **图表说明**: 按真实采集顺序排列的逐帧均温时间线，横轴为采集序号，纵轴为均温 (°C)。
# >
# > **数据分布**: 开头存在少量低温/补采帧，随后进入约 23.3°C 的主扫描温度带。
# > 主扫描内部只有缓慢漂移，不再呈现文件名字母序下那种反复跳变。
# >
# > **核心发现**: 之前“13 sessions”的主要来源是错误使用文件名字母序。
# > 真实采集顺序下，主数据应被视为一个稳定的主扫描 session，早期低温/补采帧不应混入主标定或 SR。
