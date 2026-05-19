# %% [markdown]
# ## 4. Deep Decoder Results
#
# Deep Decoder 是非 INR 深度 prior 对照，用于检查连续坐标 INR 是否真的优于低参数 CNN decoder。

# %%
deep_decoder_status = method_status("deep_decoder")
display(deep_decoder_status)

deep_decoder_metrics = read_csv_if_exists(OUTPUT_DIR / "deep_decoder" / "metrics.csv")
if not deep_decoder_metrics.empty:
    display(deep_decoder_metrics.round(4))

# %% [markdown]
# > **数据说明**: 状态表检查 Deep Decoder 的 result/metric 产物；指标表只读取已存在的 `metrics.csv`。
# >
# > **怎么看**: Deep Decoder 可以作为保守 prior。若它更平滑，split-half 可能较稳定，但过度平滑会损失轮廓细节，因此不能只看 NRMSE。
# >
# > **正常/异常**: 若 Deep Decoder artifact 很低但 highpass/raw-control 都缺少内部结构，应解释为保守或欠表达，而不是自动优于 INR。
# >
# > **核心发现**: Deep Decoder 的价值是提供 CNN decoder 对照，帮助区分“深度 prior 有用”和“连续 INR 表示有用”这两个问题。

# %%
display(show_png_if_exists(OUTPUT_DIR / "deep_decoder" / "highpass_result.png"))
display(show_png_if_exists(OUTPUT_DIR / "deep_decoder" / "raw_control_result.png"))

# %% [markdown]
# > **图表说明**: 若存在，这两张图分别展示 Deep Decoder highpass 结构响应和 raw-control 普通视觉对照。
# >
# > **怎么看**: 重点看内部轮廓是否被保留、边缘是否被过度平滑，以及背景区域是否出现 decoder 纹理。
# >
# > **正常/异常**: 输出过平、边界被抹掉或出现 CNN 上采样纹理，都不能解释为 contour-level SR 改进。
# >
# > **核心发现**: Deep Decoder 需要在稳定性和轮廓可见性之间取得平衡，才有资格进入四方对比。
