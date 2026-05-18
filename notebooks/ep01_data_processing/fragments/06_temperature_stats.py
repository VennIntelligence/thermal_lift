# %% [markdown]
# ## 3.1 温度统计与主 session 初步建模
#
# 四子图: 逐帧均温/标准差/最小/最大 ECDF + rug 分布诊断图。
# 时间线: 按真实采集顺序展示均温、中位温和稳健温度曲线，可直观看出预热/补采帧和主扫描温度带。

# %%
model = build_session_model(df)
df_filename = model.df_filename.copy()
df_acquisition = model.df_acquisition.copy()
df_temperature_stats = df_acquisition.copy()
main_session = model.main_session

plot_temperature_histograms(df_temperature_stats,
                            save_path="frame_temperature_statistics.png", save_fn=save_fig)

# %% [markdown]
# > **图表说明**: 四子图使用 ECDF（累计分布）和 rug tick（每个 tick 对应一帧）展示
# > 263 帧的均温 (T_mean)、温度标准差 (T_std)、最低温 (T_min)、最高温 (T_max)。
# > 阴影带表示 IQR，虚线表示中位数，蓝色为主扫描 session，红色为早期非主扫描帧。
# > ECDF 可以把“温度值分成几个簇”看得比普通直方图更稳定。
# >
# > **怎么读**: 横轴是温度或统计值，纵轴是小于等于该值的帧比例。
# > 曲线陡升表示很多帧集中在同一范围；水平平台表示中间范围几乎没有帧。
# > rug tick 用来确认每个小簇实际有多少帧，避免只看曲线形状误判样本量。
# >
# > **数据分布**: 均温、最低温和最高温都呈现清晰的分段结构：少量早期帧落在低温段，
# > 主扫描帧集中在约 23.3°C 的窄范围内。ECDF 的水平平台直接显示了中间温度区间没有帧，
# > rug tick 则保留了每个小簇的实际帧数信息。
# >
# > **正常/异常理解**: 如果所有帧属于同一稳定采集状态，均温 ECDF 通常会形成一个主要陡升段。
# > 这里出现多个温度簇，说明数据包含不同采集状态；这不是坏帧的直接证据，但说明不能把全部帧视为同一批 SR 输入。
# >
# > **核心发现**: 这些“离群点”不是普通统计噪声，而是采集温度段差异造成的非主扫描帧。
# > 后续 SR 默认应使用主扫描 session，早期帧只能作为单独的温度漂移/预热信息分析。

# %%
plot_robust_temperature_curve(
    model,
    save_path="robust_temperature_timeline.png",
    save_fn=save_fig,
)

# %% [markdown]
# > **图表说明**: 按真实采集顺序排列的逐帧温度时间线，横轴为采集序号，
# > 纵轴同时展示帧均温、帧中位温和 5–95% trimmed mean，并用背景色标出 session。
# > trimmed mean 会去掉最冷和最热的一小部分像素，帮助判断结论是否被极端像素主导。
# >
# > **怎么读**: 先看背景色是否把时间线分成几个连续温度段，再比较均温、中位温和 trimmed mean 是否同步。
# > 三条曲线如果一起跳变，通常代表整帧温度状态改变；如果只有最大/最小值异常，才更像局部坏点或局部热源变化。
# >
# > **数据分布**: 开头存在少量低温/补采帧，随后进入约 23.3°C 的主扫描温度带。
# > 主扫描内部只有缓慢漂移，不再呈现文件名字母序下那种反复跳变；中位温和稳健温度曲线与均温同步变化。
# >
# > **正常/异常理解**: 对 SR 对齐来说，理想输入应来自同一连续温度段，因为算法希望帧间主要差异来自微位移和噪声。
# > 跨段跳变如果混入同一次重建，会把温度状态变化误当成空间结构差异。
# >
# > **核心发现**: 主 session 是后续 SR 的可用输入池；跨 session 的温度跳变远大于主 session 内部漂移。
# > 后续对齐和重建应在主 session 内完成，而不是把所有 263 帧直接混合。
